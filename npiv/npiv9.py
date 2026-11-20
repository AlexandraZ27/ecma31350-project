"""NPIV with 3 x 3 estimator classes for the first stage g and the structural function f, applied to the outcome
with the largest 2SLS estimate in Ajzenman, Dominguez & Undurraga (2023), Tables 6-9: del_problema3
("crime as first or second concern", Table 6 col. 1, 2SLS = 18.61).

g  (first stage)  A = m(z, x) + e,  e independent of (z, x).  m is estimated by one of
     kernel : Gaussian product-kernel regression, bandwidths by leave-one-out least-squares CV (statsmodels KernelReg 'cv_ls')
     forest : honest random forest (half-samples without replacement, random p of the covariates per split, deep trees,
              leaf share alpha, leaf means on the other half), alpha and p by grouped 5-fold CV
     net    : feed-forward ReLU net, normalised-gradient descent with momentum, early stopping,
              depth / width / step size / stopping step by grouped 5-fold CV
   m is cross-fitted (grouped by commune), so the draws and the fit measures never use a model trained on the
   observation they predict.  Draws:  A~_ij = m_i + resampled out-of-fold residual + Gaussian jitter,  M = 50 per observation.
f  (structural)  minimises the exact loss (1/n) sum_i (Y_i - (1/M) sum_j f(A~_ij, x_i))^2 over
     kernel : Gaussian-bump sieve (centres at the observed (a, x)), ridge;  bandwidth and ridge by grouped CV
     forest : honest forest with exact-loss leaf values (npiv.py);  alpha and ridge by grouped CV
     net    : ReLU net, normalised-gradient descent with momentum, early stopping; architecture and stopping by grouped CV
Effect: average derivative of f in a (centred finite differences, trimmed tails), bootstrap over communes with
        all tuning parameters held fixed.  Samples: "cross" (101 communes, Tables 6-9 spec), "panel" (1,010 cells, FE removed).

Usage (from project/): python -I npiv/npiv9.py rep_data/Replication/data/analysis npiv/results9 [B_boot] [B_perm] [cross|panel|both]
"""
import sys, os, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.tree import DecisionTreeRegressor
from joblib import Parallel, delayed

SAMPLES = sys.argv[5] if len(sys.argv) > 5 else "both"
import npiv                                     # data loading, FE removal, exact-loss honest forest, linear 2SLS, derivative

warnings.filterwarnings("ignore")
YV = "del_problema3"
npiv.OUTCOMES = [YV]
D, OUT, B_BOOT, B_PERM, M, DELTAS = npiv.D, npiv.OUT, npiv.B_BOOT, npiv.B_PERM, npiv.M, npiv.DELTAS
npiv.rng = np.random.default_rng(int(os.environ.get("NPIV_SEED", 20241120)))   # seed override for stability checks
rng = npiv.rng
log = npiv.log
GS, FS = ["kernel", "forest", "net"], ["kernel", "forest", "net"]
NET_GRID = [(L, K, eta) for L in (1, 2) for K in (8, 32) for eta in (0.02, 0.1)]
T_MAX, REC, GAMMA = 400, 10, 0.5


def make_folds(cl, K=5):
    comm = np.unique(cl); fo = dict(zip(comm, rng.permutation(len(comm)) % K))
    return np.array([fo[c] for c in cl])


# ----------------------------------------------------------------------------- neural-net helpers (notes: normalised GD + momentum)
def make_net(din, L, K):
    layers, d = [], din
    for _ in range(L):
        layers += [torch.nn.Linear(d, K), torch.nn.ReLU()]; d = K
    return torch.nn.Sequential(*layers, torch.nn.Linear(d, 1))


def train_net(net, lossfn, eta, T, evalfn=None):
    """theta_k = theta_{k-1} - eta_k g/||g|| + gamma (theta_{k-1} - theta_{k-2}); eta_k = eta/(1 + k/100)."""
    params = list(net.parameters()); prev = [torch.zeros_like(p) for p in params]; hist = []
    for k in range(1, T + 1):
        net.zero_grad(); lossfn().backward()
        with torch.no_grad():
            gn = torch.sqrt(sum((p.grad ** 2).sum() for p in params)) + 1e-12; step = eta / (1 + k / 100)
            for p, pv in zip(params, prev):
                d = -step * p.grad / gn + GAMMA * pv; p.add_(d); pv.copy_(d)
        if evalfn is not None and k % REC == 0:
            with torch.no_grad():
                hist.append(float(evalfn()))
    return hist


def net_cv(fit_eval, fold):
    """Pick (architecture, stopping step) minimising the fold-averaged held-out loss. fit_eval(tr, te, cfg, seed) -> list."""
    best = (np.inf, None)
    for cfg in NET_GRID:
        h = np.mean([fit_eval(fold != f, fold == f, cfg, f) for f in np.unique(fold)], axis=0)
        j = int(np.argmin(h))
        if h[j] < best[0]:
            best = (h[j], cfg + ((j + 1) * REC,))
    return best[1]


# ----------------------------------------------------------------------------- first-stage classes: regress A on W = (z, x)
class KernelG:
    @staticmethod
    def tune(W, a, fold):
        return np.minimum(KernelReg(a, W, "c" * W.shape[1], reg_type="lc", bw="cv_ls").bw, 50.0)

    @staticmethod
    def fitpred(Wtr, atr, Wte, bw, B=None):
        lw = -0.5 * (((Wte[:, None, :] - Wtr[None, :, :]) / bw) ** 2).sum(2)
        w = np.exp(lw - lw.max(1, keepdims=True)); return (w @ atr) / w.sum(1)


def grow_reg_tree(W, a, alpha, p, seed, obs):
    rs = np.random.default_rng(seed); sub = rs.permutation(obs)[: len(obs) // 2]; h = len(sub) // 2; i1, i2 = sub[:h], sub[h:]
    tree = DecisionTreeRegressor(min_samples_leaf=max(1, int(np.ceil(alpha * h))), max_features=p, random_state=seed).fit(W[i1], a[i1])
    leaf = tree.apply(W[i2]); nn = tree.tree_.node_count
    c = np.bincount(leaf, minlength=nn); s = np.bincount(leaf, weights=a[i2], minlength=nn)
    return tree, np.where(c > 0, s / np.maximum(c, 1), tree.tree_.value[:, 0, 0])


class ForestG:
    @staticmethod
    def fitpred(Wtr, atr, Wte, hyper, B=200):
        alpha, p = hyper
        tv = Parallel(n_jobs=-1, prefer="threads")(delayed(grow_reg_tree)(Wtr, atr, alpha, p, 31 * b, np.arange(len(atr))) for b in range(B))
        return np.mean([v[t.apply(Wte)] for t, v in tv], axis=0)

    @staticmethod
    def tune(W, a, fold):
        best = (np.inf, None)
        for alpha in npiv.ALPHAS:
            for p in range(1, W.shape[1] + 1):
                mse = np.mean([((a[fold == f] - ForestG.fitpred(W[fold != f], a[fold != f], W[fold == f], (alpha, p), B=50)) ** 2).mean()
                               for f in np.unique(fold)])
                if mse < best[0]:
                    best = (mse, (alpha, p))
        return best[1]


class NetG:
    @staticmethod
    def _fit(Wtr, atr, cfg, T, evalset=None, seed=0):
        L, K, eta = cfg[:3]; torch.manual_seed(seed); mu, sd = atr.mean(), atr.std()
        Xt, yt = torch.tensor(Wtr, dtype=torch.float32), torch.tensor((atr - mu) / sd, dtype=torch.float32)
        net = make_net(Wtr.shape[1], L, K)
        ev = None
        if evalset is not None:
            Xe, ye = torch.tensor(evalset[0], dtype=torch.float32), torch.tensor((evalset[1] - mu) / sd, dtype=torch.float32)
            ev = lambda: ((net(Xe).squeeze(1) - ye) ** 2).mean()
        hist = train_net(net, lambda: ((net(Xt).squeeze(1) - yt) ** 2).mean(), eta, T, ev)
        return net, mu, sd, hist

    @staticmethod
    def fitpred(Wtr, atr, Wte, hyper, B=None):
        net, mu, sd, _ = NetG._fit(Wtr, atr, hyper, hyper[3])
        with torch.no_grad():
            return net(torch.tensor(Wte, dtype=torch.float32)).squeeze(1).numpy() * sd + mu

    @staticmethod
    def tune(W, a, fold):
        return net_cv(lambda tr, te, cfg, sd: NetG._fit(W[tr], a[tr], cfg, T_MAX, (W[te], a[te]), sd)[3], fold)


GCLS = dict(kernel=KernelG, forest=ForestG, net=NetG)


def crossfit(g, W, a, fold, hyper, B=200):
    out = np.zeros(len(a))
    for f in np.unique(fold):
        out[fold == f] = g.fitpred(W[fold != f], a[fold != f], W[fold == f], hyper, B=B)
    return out


def resid_bw(e):
    return 0.9 * min(e.std(), (np.percentile(e, 75) - np.percentile(e, 25)) / 1.34) * len(e) ** (-0.2)


def gof(a, m):
    """Held-out R^2 of m and mean leave-one-out log density of the out-of-fold residual (Gaussian KDE)."""
    e = a - m; h = resid_bw(e); k = np.exp(-0.5 * ((e[:, None] - e[None, :]) / h) ** 2) / np.sqrt(2 * np.pi); np.fill_diagonal(k, 0)
    return 1 - (e ** 2).sum() / ((a - a.mean()) ** 2).sum(), float(np.log(k.sum(1) / ((len(e) - 1) * h)).mean())


def draw_loc(m, a, rg):
    e = a - m; n = len(a); k = rg.integers(0, n, (n, M))
    return m[:, None] + e[k] + resid_bw(e) * rg.standard_normal((n, M))


# ----------------------------------------------------------------------------- structural classes: minimise the exact loss
class Scale:
    def __init__(s, F): s.mu, s.sd = F.mean(0), F.std(0)
    def __call__(s, F): return (F - s.mu) / s.sd


H_GRID = [0.3, 0.6, 1.0, 2.0]


def phi_kernel(Fs, Cs, h, n):
    out = np.empty((n, len(Cs)))
    for s in range(0, n, 50):
        blk = Fs[s * M:min(n, s + 50) * M]
        out[s:s + 50] = rbf_kernel(blk, Cs, gamma=1 / (2 * h * h)).reshape(-1, M, len(Cs)).mean(1)
    return out


def ridge_solve(K, y, lam):
    return np.linalg.solve(K + lam * np.mean(np.diag(K)) * np.eye(len(K)), y)


class KernelF:
    @staticmethod
    def tune(At, X, y, cl, A_obs):
        n = len(y); sc = Scale(npiv.design(At, X)); Fs = sc(npiv.design(At, X)); Cs = sc(np.column_stack([A_obs, X]))
        yc = y - y.mean(); fold = make_folds(cl); best = (np.inf, None)
        for h in H_GRID:
            Phi = phi_kernel(Fs, Cs, h, n); K = Phi @ Phi.T
            for lam in npiv.LAMS:
                loss = 0.0
                for f in np.unique(fold):
                    tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
                    loss += ((yc[te] - K[np.ix_(te, tr)] @ ridge_solve(K[np.ix_(tr, tr)], yc[tr], lam)) ** 2).sum()
                if loss < best[0]:
                    best = (loss, (h, lam))
        return best[1]

    @staticmethod
    def fit(At, X, y, hyper, A_obs, B=None):
        h, lam = hyper; n = len(y); sc = Scale(npiv.design(At, X)); Cs = sc(np.column_stack([A_obs, X]))
        Phi = phi_kernel(sc(npiv.design(At, X)), Cs, h, n); theta = Phi.T @ ridge_solve(Phi @ Phi.T, y - y.mean(), lam)
        return lambda F: rbf_kernel(sc(F), Cs, gamma=1 / (2 * h * h)) @ theta


class ForestF:
    @staticmethod
    def tune(At, X, y, cl, A_obs):
        return npiv.cv_select(At, X, y, cl)[0]

    @staticmethod
    def fit(At, X, y, hyper, A_obs, B=500):
        return npiv.fit_forest(At, X, y, hyper[0], hyper[1], B, 1)


class NetF:
    @staticmethod
    def _fit(Ft, yt_np, n, cfg, T, evalset=None, seed=0):
        L, K, eta = cfg[:3]; torch.manual_seed(seed); ym, ys = yt_np.mean(), yt_np.std()
        Xt, yt = torch.tensor(Ft, dtype=torch.float32), torch.tensor((yt_np - ym) / ys, dtype=torch.float32)
        net = make_net(Ft.shape[1], L, K); ev = None
        if evalset is not None:
            Xe, ye, ne = torch.tensor(evalset[0], dtype=torch.float32), torch.tensor((evalset[1] - ym) / ys, dtype=torch.float32), len(evalset[1])
            ev = lambda: ((net(Xe).view(ne, M).mean(1) - ye) ** 2).mean()
        hist = train_net(net, lambda: ((net(Xt).view(n, M).mean(1) - yt) ** 2).mean(), eta, T, ev)
        return net, ym, ys, hist

    @staticmethod
    def tune(At, X, y, cl, A_obs):
        n = len(y); F = npiv.design(At, X); sc = Scale(F); Fs = sc(F); fold = make_folds(cl)
        rows = lambda m: (np.where(m)[0][:, None] * M + np.arange(M)).ravel()
        return net_cv(lambda tr, te, cfg, sd: NetF._fit(Fs[rows(tr)], y[tr], tr.sum(), cfg, T_MAX, (Fs[rows(te)], y[te]), sd)[3], fold)

    @staticmethod
    def fit(At, X, y, hyper, A_obs, B=1):
        n = len(y); F = npiv.design(At, X); sc = Scale(F); Fs = sc(F); nets = []
        for s in range(B):
            net, ym, ys, _ = NetF._fit(Fs, y, n, hyper, hyper[3], seed=100 + s); nets.append((net, ys))
        def f(Fn):
            with torch.no_grad():
                return np.mean([net(torch.tensor(sc(Fn), dtype=torch.float32)).squeeze(1).numpy() * ys for net, ys in nets], axis=0)
        return f


FCLS = dict(kernel=KernelF, forest=ForestF, net=NetF)


def jsonable(h):
    return [float(v) for v in h] if isinstance(h, (tuple, list, np.ndarray)) else h


# ----------------------------------------------------------------------------- one sample
def build(P):
    W = np.column_stack([P["Z"], P["X"]]); return (W - W.mean(0)) / W.std(0)


def run_sample(S, name):
    log(f"=== {name}: G={S['G']} T={S['T']}  outcome {YV}")
    P = npiv.prep(S, np.arange(S["G"])); n = len(P["A"]); G = S["G"]; W = build(P); a = P["A"]; y = P["Y"][YV]
    ssc_iv, ssc_fs = (1.0, n / (n - 4)) if S["kind"] == "cross" else (G / (G - 1),) * 2
    fold = make_folds(P["cl"]); res = dict(n=n, outcome=YV, lin=npiv.lin_iv(P, y, ssc_iv, ssc_fs), gof={}, combos={})
    log("linear 2SLS:", {k: round(float(v), 3) for k, v in res["lin"].items()})

    hyp_g, centers = {}, {}
    for gn in GS:
        g = GCLS[gn]; t0 = time.time()
        hz, hx = g.tune(W, a, fold), g.tune(W[:, 1:], a, fold)               # with / without the instrument
        mz, mx = crossfit(g, W, a, fold, hz), crossfit(g, W[:, 1:], a, fold, hx)
        r2z, llz = gof(a, mz); r2x, llx = gof(a, mx); null = []
        for _ in range(B_PERM):
            Wp = W.copy(); Wp[:, 0] = rng.permutation(Wp[:, 0]); r2p, llp = gof(a, crossfit(g, Wp, a, fold, hz)); null.append((r2p - r2x, llp - llx))
        null = np.array(null); pv = lambda v, k: float((1 + np.sum(null[:, k] >= v)) / (1 + B_PERM))
        res["gof"][gn] = dict(r2=r2z, r2_noz=r2x, gain_r2=r2z - r2x, p_r2=pv(r2z - r2x, 0), ll=llz, ll_noz=llx, gain_ll=llz - llx,
                              p_ll=pv(llz - llx, 1), hyper=jsonable(hz), hyper_noz=jsonable(hx))
        hyp_g[gn], centers[gn] = hz, mz
        log(f"g={gn}: R2 {r2z:.3f} (no z {r2x:.3f}, p={res['gof'][gn]['p_r2']:.3f})  loglik {llz:.3f} (no z {llx:.3f}, p={res['gof'][gn]['p_ll']:.3f})  "
            f"hyper {hz}  [{time.time() - t0:.0f}s]")

    hyp_f = {}
    for gn in GS:
        At = draw_loc(centers[gn], a, rng)
        for fn in FS:
            t0 = time.time(); h = FCLS[fn].tune(At, P["X"], y, P["cl"], a)
            f = FCLS[fn].fit(At, P["X"], y, h, a, B=(500 if fn == "forest" else 5 if fn == "net" else None))
            est = npiv.avg_deriv(f, a, P["X"], DELTAS); hyp_f[(gn, fn)] = h
            res["combos"][f"{gn}|{fn}"] = dict(npiv=est, hyper=jsonable(h))
            log(f"g={gn} f={fn}: NPIV {np.round(est, 3)}  hyper {h}  [{time.time() - t0:.0f}s]")
        json.dump(res, open(f"{OUT}/{name}_partial.json", "w"), indent=1, default=float)

    boot = {k: [] for k in res["combos"]}
    for b in range(B_BOOT):
        t0 = time.time(); Pb = npiv.prep(S, rng.integers(0, G, G)); Wb = build(Pb); ab = Pb["A"]; yb = Pb["Y"][YV]; fb = make_folds(Pb["cl"])
        for gn in GS:
            mb = crossfit(GCLS[gn], Wb, ab, fb, hyp_g[gn], B=100); Atb = draw_loc(mb, ab, rng)
            for fn in FS:
                f = FCLS[fn].fit(Atb, Pb["X"], yb, hyp_f[(gn, fn)], ab, B=(100 if fn == "forest" else 1 if fn == "net" else None))
                boot[f"{gn}|{fn}"].append(npiv.avg_deriv(f, ab, Pb["X"], DELTAS))
        log(f"bootstrap {b + 1}/{B_BOOT}  [{time.time() - t0:.0f}s]")
    for k, v in boot.items():
        v = np.array(v)
        if len(v) > 1:
            res["combos"][k].update(se=list(v.std(0, ddof=1)), ci=[list(np.quantile(v[:, j], [.025, .975])) for j in range(len(DELTAS))])
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); out = {"deltas": DELTAS}
    for S, name in ((npiv.load_cross(), "cross"), (npiv.load_panel(), "panel")):
        if SAMPLES in ("both", name):
            out[name] = run_sample(S, name)
            json.dump(out, open(f"{OUT}/npiv9_results.json", "w"), indent=1, default=float)
    log("done")
