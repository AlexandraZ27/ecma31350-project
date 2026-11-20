"""NPIV (Theorem 1 of NPIV.tex) applied to Ajzenman, Dominguez & Undurraga (AEJ:Applied 2023).

Tuning follows ECMA31350-Notes.tex.

Stage 1  p(a | z, x): Gaussian conditional kernel density with a product kernel (one bandwidth each for z, x, a).
         (h_z, h_x) minimise the leave-one-out CV squared error of the kernel conditional mean E[A | z, x]
         (i.e. they maximise first-stage goodness of fit); h_a then maximises the leave-one-out density likelihood.
Stage 2  f(a, x): honest random forest. Each of B trees:
           * draws a subsample of size n/2 without replacement and splits it into I1, I2;
           * grows a deep tree on I1 (random subset of p = 2 of the 3 covariates at each split, every leaf holds at
             least a share alpha of I1) on the augmented data (M draws a~ ~ p(.|z,x) per observation);
           * estimates the leaf values on I2 by minimising the EXACT loss
             sum_{i in I2} (Y_i - (1/M) sum_j f(a~_ij, x_i))^2 + ridge.
         The forest averages the trees. alpha and the ridge penalty are chosen by grouped (by commune) 5-fold CV of
         the exact loss.
Effect   average derivative of f in a, by centred finite differences over observations in the 5%-95% range of A.

Two samples:  "cross" = 101-commune long difference (Tables 6-9 specification),
              "panel" = 101 communes x 10 years cell means, commune and year FE partialled out
                        (Table A.4 Panel B specification, collapsed to cells).

Usage (from project/): python -I npiv/npiv.py rep_data/Replication/data/analysis npiv/results [B_boot] [B_perm]
"""
import sys, json, time, warnings
import numpy as np, pandas as pd
from scipy import sparse
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from sklearn.tree import DecisionTreeRegressor
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
D, OUT = sys.argv[1], sys.argv[2]
B_BOOT = int(sys.argv[3]) if len(sys.argv) > 3 else 200
B_PERM = int(sys.argv[4]) if len(sys.argv) > 4 else 200

M = 50                                   # draws of A~ per observation
P_FEAT = 2                               # covariates tried at each split (of 3: a, x1, x2)
ALPHAS = [0.02, 0.05, 0.10, 0.20]        # min share of I1 observations per leaf
LAMS = 10.0 ** np.arange(-3, 2.01)       # ridge penalty relative to mean diag of the leaf kernel matrix
TREES_MAIN, TREES_CV, TREES_BOOT = 500, 50, 100
N_FOLDS, TRIM = 5, 0.05
DELTAS = [0.25, 0.5, 1.0]                # finite-difference steps in sd(A); 0.5 is the headline
OUTCOMES = ["pc_19", "pc_20", "del_tend_barrio", "del_tend_com", "del_tend_pais", "vict_agreg"]
rng = np.random.default_rng(20241120)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- data
def load_cross():
    d = pd.read_stata(f"{D}/enusc_IV.dta", convert_categoricals=False)
    return dict(kind="cross", G=len(d), T=1,
                Y={y: 100 * d[y].values.astype(float) for y in OUTCOMES},
                A=d.deltaimm.values.astype(float), Z=d.deltaimm_instr.values.astype(float),
                X=np.column_stack([d.age, d.hombre]).astype(float))


def load_panel():
    cols = ["cod_com", "year", "edad", "ismale", "lnrate_stock_imm", "imm_instr"] + OUTCOMES
    d = pd.read_stata(f"{D}/IV_panel.dta", columns=cols, convert_categoricals=False)
    d["cod_com"] = d.cod_com.astype(float)
    m = d.groupby(["cod_com", "year"]).mean().sort_index()
    G, T = m.index.get_level_values(0).nunique(), m.index.get_level_values(1).nunique()
    assert len(m) == G * T and not m.isna().any().any()
    f = lambda c: m[c].values.astype(float).reshape(G, T)
    return dict(kind="panel", G=G, T=T, Y={y: 100 * f(y) for y in OUTCOMES},
                A=f("lnrate_stock_imm"), Z=f("imm_instr"), X=np.stack([f("edad"), f("ismale")], -1))


def twoway(x):                           # balanced-panel commune + year demeaning
    return x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + x.mean()


def prep(S, idx):
    """Resample communes `idx`, remove fixed effects (panel), flatten."""
    if S["kind"] == "cross":
        return dict(Y={y: v[idx] for y, v in S["Y"].items()}, A=S["A"][idx], Z=S["Z"][idx],
                    X=S["X"][idx], cl=np.arange(len(idx)))
    f = lambda v: twoway(v[idx]).ravel()
    return dict(Y={y: f(v) for y, v in S["Y"].items()}, A=f(S["A"]), Z=f(S["Z"]),
                X=np.column_stack([f(S["X"][..., k]) for k in range(S["X"].shape[-1])]),
                cl=np.repeat(np.arange(len(idx)), S["T"]))


# ----------------------------------------------------------------------------- stage 1: conditional KDE
class Std:
    """Standardised (a, z, x) with pairwise squared distances."""
    def __init__(s, P):
        s.sa = P["A"].std()
        s.a = P["A"] / s.sa
        s.z = P["Z"] / P["Z"].std()
        s.x = P["X"] / P["X"].std(0)
        s.n = len(s.a)
        s.da2 = (s.a[:, None] - s.a[None, :]) ** 2
        s.dz2 = (s.z[:, None] - s.z[None, :]) ** 2
        s.dx2 = sum((s.x[:, k][:, None] - s.x[:, k][None, :]) ** 2 for k in range(s.x.shape[1]))


def logw(s, hz, hx, loo):
    lw = -0.5 * s.dx2 / hx ** 2
    if hz is not None:
        lw = lw - 0.5 * s.dz2 / hz ** 2
    if loo:
        lw = lw.copy(); np.fill_diagonal(lw, -np.inf)
    return lw


def loo_logdens(s, ha, hz, hx):
    lw = logw(s, hz, hx, True)
    return logsumexp(lw - 0.5 * s.da2 / ha ** 2, axis=1) - logsumexp(lw, axis=1) - np.log(ha * np.sqrt(2 * np.pi))


def loo_cm_mse(s, hz, hx):
    """Leave-one-out CV squared error of the kernel conditional mean of A (the CV statistic in the notes)."""
    lw = logw(s, hz, hx, True)
    w = np.exp(lw - lw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    return ((s.a - w @ s.a) ** 2).mean()


def select_bw(s, use_z):
    lo, hi = np.log(0.03), np.log(20.0)
    clip = lambda t: np.exp(np.clip(t, lo, hi))
    nll = (lambda t: loo_cm_mse(s, *clip(t))) if use_z else (lambda t: loo_cm_mse(s, None, clip(t)[0]))
    starts = [[.3, .5], [.6, 1.], [1., 2.]] if use_z else [[.3], [.6], [1.5]]
    best = None
    for st in starts:
        r = minimize(nll, np.log(st), method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-6, maxiter=400))
        if best is None or r.fun < best.fun:
            best = r
    hh = clip(best.x); hz, hx = (hh[0], hh[1]) if use_z else (None, hh[0])
    ha = np.exp(minimize_scalar(lambda t: -loo_logdens(s, np.exp(t), hz, hx).sum(), bounds=(lo, hi), method="bounded").x)
    return ha, hz, hx


def draw_treatment(s, h, rng):
    """M draws of A~ ~ p_hat(. | z_i, x_i) for every i (original units of A)."""
    ha, hz, hx = h
    lw = logw(s, hz, hx, False)
    w = np.exp(lw - lw.max(1, keepdims=True)); cdf = np.cumsum(w, 1); cdf /= cdf[:, -1:]
    u = rng.random((s.n, M))
    j = np.stack([np.searchsorted(cdf[i], u[i]) for i in range(s.n)]).clip(0, s.n - 1)
    return (s.a[j] + ha * rng.standard_normal((s.n, M))) * s.sa


# ----------------------------------------------------------------------------- stage 2: honest forest, exact loss
def design(At, X):
    return np.column_stack([At.ravel(), np.repeat(X, At.shape[1], axis=0)])


def grow_tree(Fall, yc, alpha, seed, obs):
    """One honest tree. Rows of Fall are stacked by observation (M consecutive rows each)."""
    rs = np.random.default_rng(seed)
    sub = rs.permutation(obs)[: len(obs) // 2]; h = len(sub) // 2; i1, i2 = sub[:h], sub[h:]
    rows = lambda ii: (ii[:, None] * M + np.arange(M)).ravel()
    tree = DecisionTreeRegressor(min_samples_leaf=max(1, int(np.ceil(alpha * h * M))), max_features=P_FEAT,
                                 random_state=seed).fit(Fall[rows(i1)], np.repeat(yc[i1], M))
    leaf = tree.apply(Fall[rows(i2)])
    Phi = sparse.csr_matrix((np.full(leaf.size, 1.0 / M), (np.repeat(np.arange(len(i2)), M), leaf)),
                            shape=(len(i2), tree.tree_.node_count))
    return dict(tree=tree, Phi=Phi, K=(Phi @ Phi.T).toarray(), y2=yc[i2])


def grow_forest(Fall, yc, alpha, B, seed, obs):
    return Parallel(n_jobs=-1, prefer="threads")(delayed(grow_tree)(Fall, yc, alpha, seed + b, obs) for b in range(B))


def leaf_values(t, lam):
    K = t["K"]
    al = np.linalg.solve(K + lam * max(np.mean(np.diag(K)), 1e-12) * np.eye(len(K)), t["y2"])
    return t["Phi"].T @ al


def predict(trees, vals, F):
    return np.mean([v[t["tree"].apply(F)] for t, v in zip(trees, vals)], axis=0)


def cv_select(At, X, y, cl):
    """Grouped 5-fold CV of the exact loss over (alpha, ridge)."""
    yc = y - y.mean(); Fall = design(At, X); n = len(y)
    comm = np.unique(cl); fold_of = dict(zip(comm, rng.permutation(len(comm)) % N_FOLDS))
    fold = np.array([fold_of[c] for c in cl]); loss = {(a, l): 0.0 for a in ALPHAS for l in LAMS}
    for f in range(N_FOLDS):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]; Fte = design(At[te], X[te])
        for a in ALPHAS:
            trees = grow_forest(Fall, yc, a, TREES_CV, 1000 * f, tr)
            leaves = [t["tree"].apply(Fte) for t in trees]
            for l in LAMS:
                vals = [leaf_values(t, l) for t in trees]
                pred = np.mean([v[lf] for v, lf in zip(vals, leaves)], axis=0).reshape(len(te), M).mean(1)
                loss[(a, l)] += ((yc[te] - pred) ** 2).sum()
    best = min(loss, key=loss.get)
    return best, {f"{a}|{l:g}": v / n for (a, l), v in loss.items()}


def fit_forest(At, X, y, alpha, lam, B, seed):
    yc = y - y.mean(); trees = grow_forest(design(At, X), yc, alpha, B, seed, np.arange(len(y)))
    vals = [leaf_values(t, lam) for t in trees]
    return lambda F: predict(trees, vals, F)


def avg_deriv(f, A, X, deltas):
    lo, hi = np.quantile(A, [TRIM, 1 - TRIM]); k = (A >= lo) & (A <= hi); out = []
    for d in deltas:
        dl = d * A.std()
        up, dn = np.column_stack([A[k] + dl, X[k]]), np.column_stack([A[k] - dl, X[k]])
        out.append(float(((f(up) - f(dn)) / (2 * dl)).mean()))
    return out


# ----------------------------------------------------------------------------- linear benchmark
def lin_iv(P, y, ssc_iv, ssc_fs):
    n = len(y); C = np.column_stack([np.ones(n), P["X"]]); cl = P["cl"]
    res = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    yt, at, zt = res(y), res(P["A"]), res(P["Z"])
    def cluster_se(score, denom, ssc):
        g = pd.Series(score).groupby(cl).sum().values
        return np.sqrt(ssc * (g ** 2).sum()) / abs(denom)
    b = (zt @ yt) / (zt @ at); e = yt - b * at
    se = cluster_se(zt * e, zt @ at, ssc_iv)
    pi = (zt @ at) / (zt @ zt); v = at - pi * zt
    se_fs = cluster_se(zt * v, zt @ zt, ssc_fs)
    return dict(beta=b, se=se, pi=pi, se_pi=se_fs, F=(pi / se_fs) ** 2, partial_r2=(zt @ at) ** 2 / ((zt @ zt) * (at @ at)))


# ----------------------------------------------------------------------------- one sample
def run_sample(S, name):
    log(f"=== {name}: G={S['G']} T={S['T']}")
    P = prep(S, np.arange(S["G"])); s = Std(P); n = s.n; G = S["G"]
    ssc_iv, ssc_fs = (1.0, n / (n - 4)) if S["kind"] == "cross" else (G / (G - 1),) * 2

    # --- first stage: kernel density, bandwidths chosen for fit
    h = select_bw(s, True); h0 = select_bw(s, False); va = s.a.var()
    r2 = lambda hz, hx: 1 - loo_cm_mse(s, hz, hx) / va
    ll1, ll0 = loo_logdens(s, h[0], h[1], h[2]), loo_logdens(s, h0[0], None, h0[2])
    gain_r2, gain_ll = r2(h[1], h[2]) - r2(None, h0[2]), ll1.sum() - ll0.sum()
    dz2 = s.dz2.copy(); zs = s.z.copy(); null_r2, null_ll = [], []
    for _ in range(B_PERM):
        zp = rng.permutation(zs); s.dz2 = (zp[:, None] - zp[None, :]) ** 2
        null_r2.append(r2(h[1], h[2]) - r2(None, h0[2])); null_ll.append(loo_logdens(s, *h).sum() - ll0.sum())
    s.dz2 = dz2
    pv = lambda g, nl: (1 + np.sum(np.array(nl) >= g)) / (1 + B_PERM)
    fs = dict(n=n, h_a=h[0], h_z=h[1], h_x=h[2], h_a_noz=h0[0], h_x_noz=h0[2],
              loo_r2=r2(h[1], h[2]), loo_r2_noz=r2(None, h0[2]), gain_r2=gain_r2, perm_p_r2=pv(gain_r2, null_r2),
              loo_ll_mean=ll1.mean(), loo_ll_mean_noz=ll0.mean(), gain_ll_mean=gain_ll / n, perm_p_ll=pv(gain_ll, null_ll),
              linear=lin_iv(P, P["Y"][OUTCOMES[0]], ssc_iv, ssc_fs))
    log("first stage:", {k: round(float(v), 4) for k, v in fs.items() if k != "linear"})
    log("linear first stage:", {k: round(float(v), 3) for k, v in fs["linear"].items()})

    # --- stage 2
    At = draw_treatment(s, h, rng); res = {}
    for y in OUTCOMES:
        (alpha, lam), cv = cv_select(At, P["X"], P["Y"][y], P["cl"])
        f = fit_forest(At, P["X"], P["Y"][y], alpha, lam, TREES_MAIN, 1)
        res[y] = dict(npiv=avg_deriv(f, P["A"], P["X"], DELTAS), alpha=alpha, ridge=lam, cv=cv,
                      lin=lin_iv(P, P["Y"][y], ssc_iv, ssc_fs))
        log(f"{y}: NPIV {np.round(res[y]['npiv'], 3)} (steps {DELTAS} sd)  linear 2SLS {res[y]['lin']['beta']:.3f} "
            f"({res[y]['lin']['se']:.3f})  alpha={alpha} ridge={lam:g}")

    # --- bootstrap over communes (bandwidths, alpha and ridge held fixed)
    boot = {y: [] for y in OUTCOMES}
    for b in range(B_BOOT):
        Pb = prep(S, rng.integers(0, G, G)); sb = Std(Pb); Atb = draw_treatment(sb, h, rng)
        for y in OUTCOMES:
            f = fit_forest(Atb, Pb["X"], Pb["Y"][y], res[y]["alpha"], res[y]["ridge"], TREES_BOOT, 7 * b)
            boot[y].append(avg_deriv(f, Pb["A"], Pb["X"], DELTAS))
        if (b + 1) % 10 == 0:
            log(f"bootstrap {b + 1}/{B_BOOT}")
    for y in OUTCOMES:
        v = np.array(boot[y])
        res[y].update(se=list(v.std(0, ddof=1)), ci=[list(np.quantile(v[:, k], [.025, .975])) for k in range(len(DELTAS))])
    return dict(first_stage=fs, outcomes=res, deltas=DELTAS, n=n)


if __name__ == "__main__":
    out = {}
    for S, name in ((load_cross(), "cross"), (load_panel(), "panel")):
        out[name] = run_sample(S, name)
        json.dump(out, open(f"{OUT}/npiv_results.json", "w"), indent=1, default=float)
    log("done")
