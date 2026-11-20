"""Dantzig-selector IV versions of Tables VI-IX in Ajzenman, Dominguez & Undurraga
(2023, AEJ: Applied), "Immigration, Crime, and Crime (Mis)Perceptions".

Instead of the single Bartik instrument (deltaimm_instr), the first stage uses the
11 country-specific 2008 immigrant shares as separate instruments and selects among
them with the Dantzig selector (Candes and Tao 2007). The selected shares are then
used in a standard 2SLS second stage ("post-Dantzig 2SLS"), in the spirit of
Belloni, Chen, Chernozhukov and Hansen's (2012) post-Lasso IV, with Dantzig in
place of Lasso in the first stage (see the lecture notes, Section 1.1).

Usage (from the project root):
    python -I rep_scripts/dantzig_iv.py rep_data/Replication/data/analysis rep_results
    # work-permit or work-visa immigration measures (Appendix Table A.4):
    python -I rep_scripts/dantzig_iv.py rep_data/Replication/data/analysis rep_results --variant permiso
    # list the variables in the data:
    python -I rep_scripts/dantzig_iv.py rep_data/Replication/data/analysis rep_results --inspect

Outputs: rep_results/dantzig_iv_output[_variant].txt and dantzig_iv_results[_variant].csv

Built-in checks against the paper (baseline variant): the shares times the shifts should
rebuild deltaimm_instr exactly, the Bartik first-stage F should be 17.35, and the
"2SLS Bartik" column should match rep_scripts/replicate.py (e.g. 18.61 (6.79) for
del_problema3).

Table X (homicides) is not produced: it needs confidential data that is not in the
replication package (same reason replicate.py skips Tables 5 and 10).

Requires: numpy, pandas, scipy (the Dantzig LP is solved with scipy's HiGHS solver;
cvxpy, suggested in the lecture notes, gives the same solution).
"""
import argparse, glob, os, re, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Configuration (variable names as they appear in enusc_IV.dta)
# --------------------------------------------------------------------------------------
IV_FILE = "enusc_IV.dta"       # municipality-level 2008-2017 differences (N = 101)
OLS_FILE = "enusc_OLS.dta"     # individual-level panel; used only for the mean of each DV

# The paper's 11 countries, with the codes used in the data. Note that "chi" is China,
# not Chile, and "eeuu" (Estados Unidos) is the USA.
COUNTRIES = {
    "arg": "Argentina", "bol": "Bolivia", "bra": "Brazil", "chi": "China",
    "col": "Colombia", "ecu": "Ecuador", "hai": "Haiti", "per": "Peru",
    "esp": "Spain", "eeuu": "USA", "ven": "Venezuela",
}

# Immigration measure. "all" is the paper's baseline (all residence permits); "permiso"
# and "visa" are the work-permit and work-visa measures of Appendix Table A.4. Each has
# its own endogenous variable, Bartik instrument and set of 2008 shares.
VARIANTS = {
    "all":     dict(endog="deltaimm",         bartik="deltaimm_instr",         suffix=""),
    "permiso": dict(endog="deltaimm_permiso", bartik="deltaimm_instr_permiso", suffix="_permiso"),
    "visa":    dict(endog="deltaimm_visa",    bartik="deltaimm_instr_visa",    suffix="_visa"),
}


def share_cols(suffix):
    """theta^n_{m,2008} (equation 3): nationality n's share of municipality m's
    immigrants in 2008, e.g. share2008_per or share2008_per_permiso."""
    return [f"share2008_{c}{suffix}" for c in COUNTRIES]


# Delta lnMIGR^n (equation 4): growth of nationality n's migrant stock in destination
# countries other than Chile. It is constant across municipalities, so it only rescales
# each share; Dantzig standardizes the shares anyway. It is used here only to check that
# the shares rebuild the paper's Bartik instrument.
SHIFT_COLS = [f"delta_ln_orig_{c}" for c in COUNTRIES]

CONTROLS = ["age", "hombre"]   # municipality average age and share male in 2017

# Outcomes: the same variables as rep_scripts/replicate.py, which reproduce the paper's
# Tables VI-IX (e.g. OLS 4.08 and 2SLS 18.61 for del_problema3 in Table VI).
TABLES = {
    6: ["del_problema3", "del_afecta3", "del_calvida3", "del_inseg5", "del_vict", "pc_19"],
    7: ["del_tend_barrio", "del_tend_com", "del_tend_pais"],
    8: ["Index_Vivienda", "Index_Vecinos", "del_arma", "pc_20"],
    9: ["vict_hurto", "vict_robosorpr", "vict_robovehi", "vict_roboviv",
        "vict_lesiones", "vict_roboviolent", "vict_agreg"],
}
ROMAN = {6: "VI", 7: "VII", 8: "VIII", 9: "IX"}

COUNTRY_PATTERN = re.compile(
    r"argent|bolivia|bras|brazil|china|colomb|ecuad|haiti|peru|espa|spain|venez|eeuu|"
    r"estados|_(arg|bol|bra|chi|col|ecu|hai|per|esp|ven)(_|$)", re.I)


# --------------------------------------------------------------------------------------
# Linear algebra helpers
# --------------------------------------------------------------------------------------
def residualize(v, W):
    """Residuals from regressing v (vector or matrix) on W (Frisch-Waugh-Lovell)."""
    return v - W @ np.linalg.lstsq(W, v, rcond=None)[0]


def independent_cols(Zs, W):
    """Indices of a maximal set of columns of Zs that are linearly independent of each
    other and of W. If the shares summed to a constant across municipalities they would
    be collinear with the intercept, and 2SLS and the first-stage F would need one share
    dropped. (The Dantzig LP itself is unaffected, so all shares stay in selection.)"""
    keep, base = [], W
    r = np.linalg.matrix_rank(base)
    for j in range(Zs.shape[1]):
        cand = np.column_stack([base, Zs[:, j]])
        if np.linalg.matrix_rank(cand) > r:
            keep.append(j)
            base, r = cand, r + 1
    return keep


def tsls(y, X, Z, hc):
    """2SLS of y on X using instruments Z (Z includes the exogenous controls).
    With Z = X this is OLS. Heteroskedasticity-robust sandwich variance:
    hc=0 matches Stata ivreg2, robust; hc=1 matches Stata reg, robust (n/(n-k))."""
    Xh = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]          # first-stage fitted values
    b = np.linalg.solve(Xh.T @ X, Xh.T @ y)
    e = y - X @ b
    A = np.linalg.inv(Xh.T @ X)
    V = A @ ((Xh * (e ** 2)[:, None]).T @ Xh) @ A.T
    n, k = X.shape
    return b, V * (n / (n - k) if hc == 1 else 1.0)


def first_stage_stats(d, Zs, W):
    """Robust (HC1) first-stage Wald F for the instruments in Zs, and their partial R^2."""
    X = np.column_stack([Zs, W])
    b, V = tsls(d, X, X, hc=1)
    q = Zs.shape[1]
    F = b[:q] @ np.linalg.solve(V[:q, :q], b[:q]) / q
    dt, Zt = residualize(d, W), residualize(Zs, W)
    fit = Zt @ np.linalg.lstsq(Zt, dt, rcond=None)[0]
    return F, 1 - ((dt - fit) @ (dt - fit)) / (dt @ dt)


# --------------------------------------------------------------------------------------
# Dantzig selector
# --------------------------------------------------------------------------------------
def dantzig(Zt, d, lam):
    """Solve  min ||pi||_1  s.t.  max_j |(1/n) sum_i Zt_ij (d_i - Zt_i' pi)| <= lam
    as a linear program: write pi = u - v with u, v >= 0, minimize sum(u + v), and
    split the absolute-value constraint into two linear inequalities."""
    n, p = Zt.shape
    G, c = Zt.T @ Zt / n, Zt.T @ d / n
    A_ub = np.block([[G, -G], [-G, G]])
    b_ub = np.concatenate([lam + c, lam - c])
    res = linprog(np.ones(2 * p), A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, None)] * (2 * p), method="highs")
    if not res.success:
        raise RuntimeError(f"Dantzig LP failed: {res.message}")
    pi = res.x[:p] - res.x[p:]
    pi[np.abs(pi) < 1e-9] = 0.0                       # clean LP round-off
    return pi


def choose_lambda(Zt, d):
    """Penalty level: Candes and Tao's lambda, rescaled to the 1/n form of the constraint.

    Candes and Tao (2007) consider y = X b + z with z ~ N(0, sigma^2 I), normalize every
    column of X to unit Euclidean norm (||X_j||_2 = 1), and impose
        max_j |X_j'(y - X b)| <= sigma * sqrt(2 log p).
    Our lecture notes (and this script) write the constraint as an average,
        max_j |(1/n) sum_i Z_ij (y_i - Z_i' b)| <= lambda,
    with columns standardized so that (1/n) sum_i Z_ij^2 = 1, i.e. ||Z_j||_2 = sqrt(n).
    Substituting X_j = Z_j / sqrt(n) into Candes and Tao's constraint gives
        max_j |(1/n) Z_j'(y - Z b)| <= sigma * sqrt(2 log p) / sqrt(n),
    so the equivalent penalty is
        lambda = sigma * sqrt(2 log p / n).
    The lecture notes print sqrt(2 log p) / n, which omits sigma and has n rather than
    sqrt(n) in the denominator; with n = 101 that would be about 10 times smaller
    (times sigma), leaving the Dantzig solution almost identical to the unpenalized OLS
    first stage. We therefore use sigma * sqrt(2 log p / n).

    Ingredients:
      p     = 11, the number of penalized instruments (the country shares). The intercept
              and the two controls are not penalized: they are partialled out of the
              endogenous variable and the shares first (FWL), which is exactly
              equivalent to leaving them unpenalized in the Dantzig program.
      sigma = unknown first-stage error s.d. Because p is small relative to n here, we
              plug in the OLS residual s.d. from regressing the (residualized) endogenous
              variable on all 11 (residualized) shares, with a degrees-of-freedom
              correction n - rank(shares) - (number of partialled-out columns). In truly
              high-dimensional problems (p >= n) this OLS step is unavailable and one
              would estimate sigma iteratively instead.
    Caveats: the rule assumes homoskedastic Gaussian errors. Data-driven choices
    (cross-validation) or heteroskedasticity-robust penalty loadings (Belloni et al.
    2012) are deliberately left for the next stage of the project."""
    n, p = Zt.shape
    k_partialled = 1 + len(CONTROLS)
    e = d - Zt @ np.linalg.lstsq(Zt, d, rcond=None)[0]
    dof = n - np.linalg.matrix_rank(Zt) - k_partialled
    sigma = np.sqrt(e @ e / dof)
    return sigma * np.sqrt(2 * np.log(p) / n), sigma


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def inspect(data_dir):
    """Print every variable in the IV file, and every variable in any .dta file whose
    name or label mentions one of the 11 countries."""
    with pd.io.stata.StataReader(os.path.join(data_dir, IV_FILE)) as r:
        labels = r.variable_labels()
    print(f"All variables in {IV_FILE}:")
    for name, lab in labels.items():
        print(f"  {name:28s} {lab}")
    print("\nVariables mentioning one of the 11 countries (name or label), all .dta files:")
    for path in sorted(glob.glob(os.path.join(data_dir, "**", "*.dta"), recursive=True)):
        try:
            with pd.io.stata.StataReader(path) as r:
                labels = r.variable_labels()
        except Exception as exc:
            print(f"  [could not read {path}: {exc}]")
            continue
        hits = [(nm, lb) for nm, lb in labels.items()
                if COUNTRY_PATTERN.search(nm) or COUNTRY_PATTERN.search(lb or "")]
        if hits:
            print(f"  {os.path.relpath(path, data_dir)}:")
            for nm, lb in hits:
                print(f"    {nm:26s} {lb}")


def load(data_dir, cols, needed, out):
    iv = pd.read_stata(os.path.join(data_dir, IV_FILE), convert_categoricals=False)
    missing = [c for c in cols + needed if c not in iv.columns]
    if missing:
        sys.exit(f"Variables not found in {IV_FILE}: {missing}. Run with --inspect.")
    bad = iv[cols + needed].isna().any(axis=1)
    if bad.any():
        out(f"WARNING: dropping {bad.sum()} municipalities with missing shares, instrument, "
            f"endogenous variable or controls (N falls from {len(iv)} to "
            f"{len(iv) - bad.sum()}; the paper has 101).")
        iv = iv.loc[~bad].reset_index(drop=True)
    return iv


def check_bartik(iv, cols, bartik):
    """Rebuild the paper's instrument as sum_n share_n x shift_n (equation 4) and compare
    it with the stored instrument. Returns (max |difference|, correlation, N compared),
    or None if the shifts are unavailable."""
    if any(c not in iv.columns for c in SHIFT_COLS):
        return None
    rebuilt = (iv[cols].to_numpy(float) * iv[SHIFT_COLS].to_numpy(float)).sum(axis=1)
    stored = iv[bartik].to_numpy(float)
    ok = np.isfinite(rebuilt) & np.isfinite(stored)
    if ok.sum() < 2:
        return None
    diff = np.abs(rebuilt[ok] - stored[ok]).max()
    corr = np.corrcoef(rebuilt[ok], stored[ok])[0, 1]
    return diff, corr, int(ok.sum()), np.abs(stored[ok]).max()


def dv_means(data_dir, outcomes):
    """Mean of each outcome (x100) in the individual-level panel, as reported in the
    paper's tables and in replicate.py. Returns NaNs if the file is unavailable."""
    try:
        ols = pd.read_stata(os.path.join(data_dir, OLS_FILE), convert_categoricals=False,
                            columns=outcomes)
        return {y: 100 * ols[y].mean() for y in outcomes}
    except Exception:
        return {y: np.nan for y in outcomes}


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--variant", choices=list(VARIANTS), default="all",
                    help="immigration measure: all permits (baseline), permiso, or visa")
    ap.add_argument("--inspect", action="store_true", help="list the variables in the data")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.data_dir)
        return

    v = VARIANTS[args.variant]
    endog, bartik, cols = v["endog"], v["bartik"], share_cols(v["suffix"])
    outcomes = sum(TABLES.values(), [])

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    iv = load(args.data_dir, cols, [endog, bartik] + CONTROLS + outcomes, out)
    n = len(iv)
    W = np.column_stack([np.ones(n)] + [iv[c].to_numpy(float) for c in CONTROLS])
    d = iv[endog].to_numpy(float)
    S = iv[cols].to_numpy(float)

    out("Dantzig-selector IV: Ajzenman, Dominguez & Undurraga (2023), Tables VI-IX")
    out(f"Immigration measure: {args.variant}  (endogenous: {endog}; Bartik: {bartik})")

    # ---- Check that the shares are the ones behind the paper's instrument
    chk = check_bartik(iv, cols, bartik)
    if chk is None:
        out("Check skipped: shift variables unavailable or missing.")
    else:
        diff, corr, k, scale_b = chk
        out(f"Check: sum_n share_n x shift_n vs. {bartik}: max |diff| = {diff:.2e}, "
            f"corr = {corr:.6f} (N = {k})")
        if diff > 1e-4 * max(1.0, scale_b):
            out("WARNING: the shares and shifts do not rebuild the stored instrument "
                "exactly. Check how the instrument was built before interpreting results.")

    # ---- First stage: Dantzig selection among the 11 shares. It is the same for every
    #      outcome, because the sample, the endogenous variable and the instruments are
    #      fixed. Partial out the unpenalized controls, then standardize each share.
    St, dt = residualize(S, W), residualize(d, W)
    scale = np.sqrt((St ** 2).mean(axis=0))
    keep = scale > 1e-12
    if not keep.all():
        out(f"WARNING: dropping shares with no variation: "
            f"{[c for c, k_ in zip(cols, keep) if not k_]}")
    names = [c for c, k_ in zip(cols, keep) if k_]
    S, St, scale = S[:, keep], St[:, keep], scale[keep]
    Zt = St / scale                                    # (1/n) sum Zt_ij^2 = 1
    lam, sigma = choose_lambda(Zt, dt)
    pi = dantzig(Zt, dt, lam)
    sel = np.flatnonzero(pi != 0)

    full = independent_cols(S, W)                      # for the unselected all-shares 2SLS
    if len(full) < S.shape[1]:
        out(f"WARNING: shares are collinear with the controls/intercept; dropping "
            f"{[names[j] for j in range(len(names)) if j not in full]} from the "
            f"all-shares 2SLS and F-statistic.")
    if len(sel):
        sel = sel[independent_cols(S[:, sel], W)]

    out(f"N = {n} municipalities; p = {len(names)} share instruments; "
        f"controls (unpenalized): intercept, {', '.join(CONTROLS)}")
    out(f"sigma_hat = {sigma:.4f};  lambda = sigma_hat * sqrt(2 log p / n) = {lam:.4f}")
    out("\nFirst stage: Dantzig coefficients on standardized shares")
    for nm, coef in zip(names, pi):
        country = COUNTRIES[nm.replace("share2008_", "").replace(v["suffix"], "")]
        flag = "   <- selected" if coef != 0 else ""
        out(f"  {nm:26s} {country:10s} {coef:9.4f}{flag}")

    F_b, r2_b = first_stage_stats(d, iv[[bartik]].to_numpy(float), W)
    F_a, r2_a = first_stage_stats(d, S[:, full], W)
    out(f"\n  Bartik instrument (paper):  robust F = {F_b:6.2f}, partial R2 = {r2_b:.3f}")
    out(f"  All shares ({len(full)} used):      robust F = {F_a:6.2f}, partial R2 = {r2_a:.3f}")
    if len(sel):
        F_s, r2_s = first_stage_stats(d, S[:, sel], W)
        out(f"  Dantzig-selected ({len(sel)}):      robust F = {F_s:6.2f}, partial R2 = {r2_s:.3f}")
    else:
        out("  Dantzig selected no instruments: post-Dantzig estimates are not reported "
            "(the first stage is too weak at this lambda).")
    zhat = Zt @ pi                                     # Dantzig first-stage fitted value

    # ---- Second stage for each outcome
    estimators = {
        "OLS": None,
        "2SLS Bartik": iv[[bartik]].to_numpy(float),
        "2SLS all shares": S[:, full],
        "Post-DS 2SLS": S[:, sel] if len(sel) else None,
        "DS-fit IV": zhat[:, None] if len(sel) else None,
    }
    means = dv_means(args.data_dir, outcomes)
    rows = []
    X = np.column_stack([d, W])
    for t, ys in TABLES.items():
        out(f"\n=== Table {ROMAN[t]} ===")
        out(f"{'':18s}" + "".join(f"{k:>24s}" for k in estimators))
        for y in ys:
            yy = 100 * iv[y].to_numpy(float)
            cells = []
            for est, Zinst in estimators.items():
                if est == "OLS":
                    b, V = tsls(yy, X, X, hc=1)        # Stata reg, robust: HC1, t(n-k)
                    se = np.sqrt(V[0, 0])
                    pv = 2 * stats.t.sf(abs(b[0] / se), n - X.shape[1])
                elif Zinst is None:
                    b, se, pv = [np.nan], np.nan, np.nan
                else:
                    b, V = tsls(yy, X, np.column_stack([Zinst, W]), hc=0)  # ivreg2: HC0, z
                    se = np.sqrt(V[0, 0])
                    pv = 2 * stats.norm.sf(abs(b[0] / se))
                cells.append(f"{b[0]:7.2f} ({se:5.2f}) [{pv:.3f}]")
                rows.append(dict(variant=args.variant, table=ROMAN[t], outcome=y,
                                 estimator=est, coef=b[0], se=se, pval=pv,
                                 mean_dv=means[y], n=n))
            out(f"{y:18s}" + "".join(f"{c:>24s}" for c in cells))
        out(f"{'mean DV':18s}" + "  ".join(f"{y}={means[y]:.2f}" for y in ys))

    out("\nNotes: coefficients are the effect of doubling the immigrant share, in percentage "
        "points (outcomes x100).\nOLS: HC1 robust SEs, t p-values. IV columns: HC0 robust "
        "SEs, normal p-values (matching Stata ivreg2, robust).\nPost-DS 2SLS: 2SLS using "
        "the Dantzig-selected shares as instruments; SEs treat the selected set as fixed.\n"
        "DS-fit IV: just-identified IV using the Dantzig first-stage fitted value as the "
        "single instrument.\nTable X (homicides) is omitted: its data are confidential and "
        "not in the replication package.")

    tag = "" if args.variant == "all" else f"_{args.variant}"
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"dantzig_iv_output{tag}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, f"dantzig_iv_results{tag}.csv"),
                              index=False)


if __name__ == "__main__":
    main()
