"""Panel extension of Singh, Sahani & Gretton (2019) KIV.

Y_mt = alpha_m + f(A_mt, X_mt, year_t) + u_mt, with f = b_t + g.
Commune intercepts alpha_m and year main effects b_t are unpenalized. The
nonlinear g uses continuous Gaussian features times categorical year features,
allowing immigration/control/year interactions but no commune interactions.
Additive effects are absorbed AFTER constructing the kernel features.
Default first-stage inputs: 11 raw origin shares plus age, male share and year.
The paper's Algorithm 2 selects lambda/xi; independent outer commune folds
evaluate first-stage loss. Split/bandwidth variation is NOT confidence coverage.
Run: python3 project/npiv/kernel/run_kernel.py
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ecma31350-kernel-matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, stats
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist, pdist
from threadpoolctl import threadpool_limits

# NumPy 2.0 with macOS Accelerate may report stale floating-point flags on
# otherwise finite matmul results. Keep other warnings and explicitly check
# every fitted coefficient, loss and evaluation for finiteness below.
warnings.filterwarnings("ignore", message=r"(divide by zero|overflow|invalid value) encountered in matmul", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "rep_data/Replication/data/analysis/IV_panel.dta"
OUTCOMES = {"pc_19": "Crime concerns index", "pc_20": "Preventive behavior index",
            "vict_agreg": "Total victimization", "del_problema3": "Crime among top concerns"}
LAM_GRID = np.logspace(-7, 1, 9)
XI_GRID = np.logspace(-8, 2, 11)
COUNTRIES = ["hai", "per", "col", "ven", "bol", "arg", "ecu", "eeuu", "esp", "chi", "bra"]
SHARES = [f"share2008_{c}" for c in COUNTRIES]
SHIFTS = [f"ln_orig_{c}" for c in COUNTRIES]
COMPONENTS = [f"z_component_{c}" for c in COUNTRIES]
SPECIFICATION = {
    "equation": "Y_mt = alpha_m + f(A_mt, X_mt, year_t) + u_mt",
    "f_decomposition": "f(A, X, t) = b_t + g(A, X, t)",
    "commune_effect": "unpenalized additive intercept; no commune input in g",
    "year_main_effect": "unpenalized b_t inside f; first observed year normalized to zero",
    "time_kernel": "categorical Gaussian on full year indicators; baseline bandwidth 1",
    "interactions": "full immigration x demographics x categorical-year product kernel",
    "feature_absorption": "commune and year main effects projected AFTER kernel construction in both stages",
    "causal_interpretation": "requires valid panel IV moments and identification of the nonlinear response; not established by fitting",
}


def clean_json(x):
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())
    if isinstance(x, np.generic):
        return clean_json(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def save_json(path, x):
    path.write_text(json.dumps(clean_json(x), indent=2, allow_nan=False) + "\n")


def load_cells(outcome, instrument_mode="raw_shares"):
    cols = ["cod_com", "year", "edad", "ismale", "lnrate_stock_imm", "imm_instr", outcome] + SHARES + SHIFTS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.DtypeWarning)
        warnings.filterwarnings("ignore", category=UnicodeWarning)
        d = pd.read_stata(DATA, columns=cols, convert_categoricals=False)
    original_n = len(d)
    d = d.dropna(subset=cols).copy()
    d["cod_com"] = d.cod_com.astype(int)
    d["year"] = d.year.astype(int)
    groups = d.groupby(["cod_com", "year"], observed=True, sort=True)
    assert groups.lnrate_stock_imm.nunique().max() == 1
    assert groups.imm_instr.nunique().max() == 1
    c = groups.agg(y=(outcome, "mean"), age=("edad", "mean"),
                   male_share=("ismale", "mean"), A=("lnrate_stock_imm", "first"),
                   Z=("imm_instr", "first"), respondents=(outcome, "size")).reset_index()
    extra = groups[SHARES + SHIFTS].first().reset_index()
    c = c.merge(extra, on=["cod_com", "year"], validate="one_to_one")
    assert (c.groupby("cod_com")[SHARES].max() - c.groupby("cod_com")[SHARES].min()).to_numpy().max() == 0
    assert (c.groupby("year")[SHIFTS].max() - c.groupby("year")[SHIFTS].min()).to_numpy().max() == 0
    assert np.all(np.abs(c[SHIFTS].to_numpy(float)) > 1e-12)
    for share, shift, component in zip(SHARES, SHIFTS, COMPONENTS):
        c[component] = c[share].to_numpy(float) * c[shift].to_numpy(float)
    sum_error = float(np.max(np.abs(c[COMPONENTS].sum(axis=1) - c.Z)))
    assert sum_error < 1e-5, "Country-specific terms do not reproduce the supplied Bartik sum"
    c["y"] *= 100
    # Stata float columns otherwise serialize with float32's shorter decimal
    # representation. Preserve the exact values used by the float64 solver so
    # CSVs and saved models reproduce one another to numerical precision.
    c = c.astype({name: np.float64 for name in c.select_dtypes(include=["floating"]).columns})
    c["trim_keep"] = c.groupby("year").A.transform(
        lambda s: s.between(s.quantile(.05), s.quantile(.95)))
    assert len(c) == 1010 and c.cod_com.nunique() == 101
    c.attrs["instrument_mode"] = instrument_mode
    return c, {"file_rows": original_n, "eligible_respondents": len(d), "cells": len(c),
               "communes": c.cod_com.nunique(), "years": sorted(c.year.unique().tolist()),
               "min_cell_count": int(c.respondents.min()), "max_cell_count": int(c.respondents.max()),
               "instrument_mode": instrument_mode, "countries": COUNTRIES,
               "components_sum_max_error": sum_error,
               "shares_constant_within_commune": True, "shocks_nonzero_and_common_within_year": True}


def inputs(d):
    mode = d.attrs.get("instrument_mode", "bartik")
    z = SHARES if mode == "raw_shares" else COMPONENTS if mode == "components" else ["Z"]
    return (d[["A", "age", "male_share", "year"]].to_numpy(float),
            d[z + ["age", "male_share", "year"]].to_numpy(float))


class Absorb:
    """M_D sqrt(w), with unpenalized commune and categorical year effects."""
    def __init__(self, d, weighting="equal"):
        w = np.ones(len(d)) if weighting == "equal" else d.respondents.to_numpy(float)
        self.w = w / w.mean()
        self.sw = np.sqrt(self.w)
        C = pd.get_dummies(d.cod_com, dtype=float).to_numpy()
        T = pd.get_dummies(d.year, drop_first=True, dtype=float).to_numpy()
        self.n_communes = C.shape[1]
        self.D = np.column_stack([C, T])
        self.Q = linalg.qr(self.sw[:, None] * self.D, mode="economic", check_finite=False)[0]
        self.rank = self.Q.shape[1]

    def project(self, a):
        return a - self.Q @ (self.Q.T @ a)

    def transform(self, a):
        return self.project(self.sw * a if a.ndim == 1 else self.sw[:, None] * a)

    def cross_kernel(self, k, other):
        return other.transform(self.transform(k).T).T

    def raw_coefficients(self, a):
        return self.sw * self.project(a)


def widths(x):
    """Training median continuous widths; last column is a categorical year label."""
    distances = [pdist(x[:, j:j+1]) for j in range(x.shape[1]-1)]
    h = np.array([np.median(a) if np.median(a) > 0 else np.median(a[a > 0]) if np.any(a > 0) else 1. for a in distances])
    return np.r_[np.maximum(h, np.maximum(np.std(x[:,:-1], axis=0) * 1e-6, 1e-10)), 1.]


def kernel(x, z, h):
    continuous = np.exp(-.5 * cdist(x[:,:-1] / h[:-1], z[:,:-1] / h[:-1], metric="sqeuclidean"))
    # Equivalent to a Gaussian kernel on a full one-hot year block. Equality
    # alone matters: numeric year distances never enter this categorical factor.
    return continuous * np.exp(-(x[:,-1,None] != z[None,:,-1]).astype(float) / h[-1]**2)


def spectrum(k):
    val, vec = linalg.eigh((k + k.T) / 2, check_finite=False)
    if val.min() < -1e-7 * max(1., val.max()):
        raise ValueError("Kernel lost positive semidefiniteness")
    return np.maximum(val, 0), vec


def first_stage(kv, kv12, ks, ks12, ks22, fixed_lambda=None):
    n, m = kv12.shape
    ev, U = spectrum(kv)
    R = U.T @ kv12
    # Evaluate every ridge candidate in the same target RKHS.
    gram = U.T @ ks @ U
    cross = U.T @ ks12
    records, best = [], None
    candidates = LAM_GRID if fixed_lambda is None else [fixed_lambda]
    for lam in candidates:
        G = R / (ev[:, None] + n * lam)
        loss = (np.trace(ks22) - 2 * np.sum(cross * G) + np.sum(G * (gram @ G))) / m
        if not np.isfinite(loss) or loss < -1e-8:
            raise FloatingPointError("Invalid first-stage loss")
        rec = {"lambda": float(lam), "embedding_loss": float(loss)}
        records.append(rec)
        if best is None or loss < best[0]:
            best = (loss, lam, U @ G)
    loss, lam, gamma = best
    return gamma, {"lambda": float(lam), "embedding_loss": float(loss), "grid": records,
                   "lambda_at_grid_boundary": bool(lam == LAM_GRID[0] or lam == LAM_GRID[-1]),
                   "first_stage_effective_df": float(np.sum(ev / (ev + n * lam)))}


def blocks(d1, d2, weighting="equal", bandwidth=1., excluded=True, shared_widths=None, instrument_bandwidth=1., time_bandwidth=1.):
    assert d1.attrs.get("instrument_mode", "bartik") == d2.attrs.get("instrument_mode", "bartik")
    s1, v1 = inputs(d1)
    s2, v2 = inputs(d2)
    if shared_widths is None:
        hs, hv = widths(s1), widths(v1)
        hs[:-1] *= bandwidth
        hs[-1] *= time_bandwidth
        hv[-1] *= time_bandwidth
    else:
        hs, hv = shared_widths
    hv = hv.copy()
    hv[:v1.shape[1]-3] *= instrument_bandwidth
    p1, p2 = Absorb(d1, weighting), Absorb(d2, weighting)
    if not excluded:
        n_excluded = v1.shape[1] - 3
        v1, v2, hv = v1[:, n_excluded:], v2[:, n_excluded:], hv[n_excluded:]
    ks = p1.cross_kernel(kernel(s1, s1, hs), p1)
    ks12 = p1.cross_kernel(kernel(s1, s2, hs), p2)
    ks22 = p2.cross_kernel(kernel(s2, s2, hs), p2)
    kv = p1.cross_kernel(kernel(v1, v1, hv), p1)
    kv12 = p1.cross_kernel(kernel(v1, v2, hv), p2)
    return p1, p2, s1, hs, (kv, kv12, ks, ks12, ks22)


def fit(d1, d2, weighting="equal", bandwidth=1., instrument_bandwidth=1., time_bandwidth=1.):
    p1, p2, s1, hs, matrices = blocks(d1, d2, weighting, bandwidth, instrument_bandwidth=instrument_bandwidth, time_bandwidth=time_bandwidth)
    gamma, meta = first_stage(*matrices)
    ks = matrices[2]
    y1 = p1.transform(d1.y.to_numpy(float))
    y2 = p2.transform(d2.y.to_numpy(float))
    q = gamma.T @ ks @ gamma
    ev, U = spectrum(q)
    R = gamma @ U
    pred = ks @ R
    uy = U.T @ y2
    records, best = [], None
    for xi in XI_GRID:
        b = uy / (ev + len(d2) * xi)
        # EXACT Algorithm 2 tuning criterion, after absorbing panel nuisance terms.
        # This is structural-prediction MSE, NOT an IV moment test.
        loss = np.mean((y1 - pred @ b) ** 2)
        if not np.isfinite(loss):
            raise FloatingPointError("Invalid second-stage loss")
        records.append({"xi": float(xi), "structural_validation_mse": float(loss)})
        if best is None or loss < best[0]:
            best = (loss, xi, R @ b)
    loss, xi, alpha = best
    if not np.isfinite(alpha).all():
        raise FloatingPointError("Nonfinite KIV coefficients")
    positive = ev[ev > max(ev.max() * 1e-10, 1e-12)]
    meta.update({"xi": float(xi), "structural_validation_mse": float(loss), "xi_grid": records,
                 "xi_at_grid_boundary": bool(xi == XI_GRID[0] or xi == XI_GRID[-1]),
                 "second_stage_effective_df": float(np.sum(ev / (ev + len(d2) * xi))),
                 "second_stage_eigenvalues": ev[::-1].tolist(),
                 "numerical_embedding_rank": len(positive), "n_stage1": len(d1), "n_stage2": len(d2),
                 "structural_bandwidths": hs.tolist(),
                 "time_kernel": "categorical_year_indicators", "time_bandwidth": time_bandwidth,
                 "commune_in_structural_kernel": False, "unpenalized_commune_intercepts": True,
                 "unpenalized_year_main_effects_inside_f": True,
                 "instrument_mode": d1.attrs.get("instrument_mode", "bartik"),
                 "instrument_bandwidth_multiplier": instrument_bandwidth,
                 "instrument_input_bandwidths": (widths(inputs(d1)[1]) * np.r_[np.repeat(instrument_bandwidth, inputs(d1)[1].shape[1]-3), np.ones(2), time_bandwidth]).tolist()})
    return {"centers": s1, "widths": hs, "coef": p1.raw_coefficients(alpha)}, meta


def evaluate(model, s):
    k = kernel(s, model["centers"], model["widths"])
    values = k @ model["coef"]
    derivative = (k * (model["centers"][:, 0][None, :] - s[:, 0, None]) /
                  model["widths"][0] ** 2) @ model["coef"]
    if not np.isfinite(values).all() or not np.isfinite(derivative).all():
        raise FloatingPointError("Nonfinite KIV evaluation")
    return values, derivative


def split_cells(d, seed):
    ids = np.sort(d.cod_com.unique())
    shuffled = np.random.default_rng(seed).permutation(ids)
    a = set(shuffled[:len(ids)//2])
    return d[d.cod_com.isin(a)].copy(), d[~d.cod_com.isin(a)].copy()


def linear_benchmark(d, weighting="equal", instrument_mode=None):
    p = Absorb(d, weighting)
    y = p.transform(d.y.to_numpy(float))
    a = p.transform(d.A.to_numpy(float))
    mode = instrument_mode or d.attrs.get("instrument_mode", "bartik")
    zcols = ["Z"] if mode == "bartik" else COMPONENTS
    z = p.transform(d[zcols].to_numpy(float))
    C = p.transform(d[["age", "male_share"]].to_numpy(float))
    Q = linalg.qr(C, mode="economic")[0]
    residual = lambda v: v - Q @ (Q.T @ v)
    yr, ar, zr = map(residual, (y, a, z))
    U, singular, Vt = linalg.svd(zr, full_matrices=False, check_finite=False)
    rank = int(np.sum(singular > singular[0] * 1e-10))
    if rank == 0:
        raise ValueError("Excluded instruments have no residual variation")
    U = U[:, :rank]
    pi_orthogonal = U.T @ ar
    fitted_a = U @ pi_orthogonal
    beta = (fitted_a @ yr) / (fitted_a @ ar)
    e = yr - beta * ar
    codes = pd.factorize(d.cod_com)[0]
    G, n = len(np.unique(codes)), len(d)
    correction = G / (G - 1) * (n - 1) / (n - p.rank - 3)
    meat = lambda v: np.sum(np.bincount(codes, weights=v) ** 2)
    se = np.sqrt(correction * meat(fitted_a * e) / (fitted_a @ ar) ** 2)
    fs_e = ar - fitted_a
    score = np.column_stack([np.bincount(codes, weights=U[:, k] * fs_e, minlength=G) for k in range(rank)])
    fs_correction = G / (G-1) * (n-1) / (n-p.rank-2-rank)
    covariance = fs_correction * score.T @ score
    covrank = int(np.linalg.matrix_rank(covariance))
    joint_F = float(pi_orthogonal @ np.linalg.pinv(covariance, rcond=1e-12) @ pi_orthogonal / rank)
    transform = Vt[:rank].T / singular[:rank]
    pi = transform @ pi_orthogonal
    fs_se = np.sqrt(np.maximum(np.diag(transform @ covariance @ transform.T), 0))
    ols = (ar @ yr) / (ar @ ar)
    if not np.isfinite([beta, se, ols, joint_F]).all() or not np.isfinite(pi).all() or not np.isfinite(fs_se).all():
        raise FloatingPointError("Nonfinite linear benchmark")
    return {"estimate": float(beta), "cluster_se": float(se), "clusters": G,
            "ci95_t": [float(beta - stats.t.ppf(.975, G-1)*se),
                       float(beta + stats.t.ppf(.975, G-1)*se)],
            "ols": float(ols), "excluded_instrument_count": len(zcols), "excluded_instrument_rank": rank,
            "linear_instrument_specification": "aggregate Bartik" if mode == "bartik" else "11 separate country share times annual log migration-stock components",
            "first_stage_coefficients": dict(zip(zcols, pi.tolist())),
            "first_stage_coefficient_cluster_se": dict(zip(zcols, fs_se.tolist())),
            "first_stage_F": joint_F, "first_stage_F_covariance_rank": covrank,
            "first_stage_partial_R2": float(1 - (fs_e @ fs_e) / (ar @ ar)),
            "se_convention": "CR1, full commune/year/control parameter count, t(100); conditional on supplied shift-share shocks"}


def outer_diagnostics(d, seed=731, instrument_bandwidth=1.):
    folds = np.array_split(np.random.default_rng(seed).permutation(np.sort(d.cod_com.unique())), 5)
    rows = []
    for f, held in enumerate(folds):
        train, test = d[~d.cod_com.isin(held)], d[d.cod_com.isin(held)]
        a, b = split_cells(train, seed + 100 + f)
        shared_inner = (widths(inputs(a)[0]), widths(inputs(a)[1]))
        shared_outer = (widths(inputs(train)[0]), widths(inputs(train)[1]))
        row = {"fold": f, "train_communes": train.cod_com.nunique(), "test_communes": test.cod_com.nunique(),
               "test_n": len(test), "test_commune_ids": sorted(test.cod_com.unique().tolist())}
        for included, label in [(True, "with_Z"), (False, "without_Z")]:
            *_, inner = blocks(a, b, shared_widths=shared_inner, excluded=included, instrument_bandwidth=instrument_bandwidth)
            _, chosen = first_stage(*inner)
            p1, p2, _, _, outer = blocks(train, test, shared_widths=shared_outer, excluded=included, instrument_bandwidth=instrument_bandwidth)
            gamma, score = first_stage(*outer, fixed_lambda=chosen["lambda"])
            target = p2.transform(test.A.to_numpy(float))
            pred = gamma.T @ p1.transform(train.A.to_numpy(float))
            row[label] = {"lambda": chosen["lambda"], "embedding_loss": score["embedding_loss"],
                          "mean_prediction_sse": float(np.sum((target-pred)**2)),
                          "mean_prediction_sst": float(target @ target)}
        row["embedding_loss_gain"] = row["without_Z"]["embedding_loss"] - row["with_Z"]["embedding_loss"]
        rows.append(row)
    result = {"folds": rows, "instrument_mode": d.attrs.get("instrument_mode", "bartik"),
              "instrument_bandwidth_multiplier": instrument_bandwidth,
              "note": "Nested tuning; target RKHS identical with/without Z in each outer fold; held-out commune/year effects projected at feature level."}
    for label in ["with_Z", "without_Z"]:
        result[label] = {"embedding_loss": float(np.average([r[label]["embedding_loss"] for r in rows], weights=[r["test_n"] for r in rows])),
                         "oof_mean_R2": float(1 - sum(r[label]["mean_prediction_sse"] for r in rows) / sum(r[label]["mean_prediction_sst"] for r in rows))}
    result["embedding_loss_gain"] = result["without_Z"]["embedding_loss"] - result["with_Z"]["embedding_loss"]
    return result


def ensemble(d, repeats, weighting, bandwidth, seed, outdir, label, instrument_bandwidth=1., time_bandwidth=1.):
    s, _ = inputs(d)
    keep = d.trim_keep.to_numpy(bool)
    weights = np.ones(len(d)) if weighting == "equal" else d.respondents.to_numpy(float)
    models, all_meta, split_effects, derivs, vals = [], [], [], [], []
    for r in range(repeats):
        a, b = split_cells(d, seed + r)
        split_derivatives = []
        for direction, (d1, d2) in enumerate([(a,b), (b,a)]):
            model, meta = fit(d1, d2, weighting, bandwidth, instrument_bandwidth, time_bandwidth)
            v, derivative = evaluate(model, s)
            vals.append(v); derivs.append(derivative); split_derivatives.append(derivative)
            meta.update({"split": r, "direction": direction,
                         "stage1_communes": sorted(d1.cod_com.unique().tolist()),
                         "stage2_communes": sorted(d2.cod_com.unique().tolist())})
            all_meta.append(meta); models.append(model)
        split_effects.append(float(np.average(np.mean(split_derivatives, axis=0)[keep], weights=weights[keep])))
        print(f"  {label} split {r+1}/{repeats}: derivative {split_effects[-1]:.4f}", flush=True)
    derivative, value = np.mean(derivs, axis=0), np.mean(vals, axis=0)
    payload = {}
    for j, model in enumerate(models):
        for key, val in model.items():
            payload[f"model_{j}_{key}"] = val
    np.savez_compressed(outdir / f"models_{label}.npz", **payload)
    result = {"weighting": weighting, "bandwidth_multiplier": bandwidth,
              "time_bandwidth": time_bandwidth, "specification": SPECIFICATION,
              "instrument_mode": d.attrs.get("instrument_mode", "bartik"),
              "instrument_bandwidth_multiplier": instrument_bandwidth,
              "average_derivative_trimmed": float(np.average(derivative[keep], weights=weights[keep])),
              "average_derivative_all_cells": float(np.average(derivative, weights=weights)),
              "split_average_derivatives": split_effects,
              "split_range_not_CI": [float(min(split_effects)),float(max(split_effects))],
              "split_sd_not_SE": float(np.std(split_effects, ddof=1)) if repeats > 1 else None,
              "fits": all_meta}
    cells = d.copy()
    cells["g_component"] = value
    cells["marginal_effect"] = derivative
    # Recover nuisance fixed effects using structural residuals, not fitted-stage regressors.
    p = Absorb(d, weighting)
    nuisance = linalg.lstsq(p.sw[:,None]*p.D, p.sw*(d.y.to_numpy()-value))[0]
    cells["commune_fixed_effect_component"] = p.D[:,:p.n_communes] @ nuisance[:p.n_communes]
    cells["year_main_effect_inside_f"] = p.D[:,p.n_communes:] @ nuisance[p.n_communes:]
    cells["f_component"] = value + cells.year_main_effect_inside_f
    cells["structural_fitted"] = cells.f_component + cells.commune_fixed_effect_component
    residual = p.sw * (d.y.to_numpy() - cells.structural_fitted.to_numpy())
    assert np.max(np.abs(p.Q.T @ residual)) < 1e-8
    cells.to_csv(outdir / f"cells_{label}.csv", index=False)
    result["time_heterogeneity"] = time_heterogeneity(d, models, outdir, label)
    return result, cells


def time_heterogeneity(d, models, outdir, label):
    """Hold A and demographics fixed across years, with an empirical support check."""
    names = ["A", "age", "male_share"]
    profiles = d[names].to_numpy(float)
    years = sorted(d.year.unique())
    lower = d.groupby("year")[names].quantile(.05).max().to_numpy(float)
    upper = d.groupby("year")[names].quantile(.95).min().to_numpy(float)
    keep = ((profiles >= lower) & (profiles <= upper)).all(axis=1)
    scale = np.maximum(profiles.std(axis=0), 1e-10)
    for year in years:
        # Intersection of empirical convex hulls is a check against conspicuous
        # extrapolation, not a proof of conditional IV support or identification.
        hull = Delaunay(d.loc[d.year == year, names].to_numpy(float) / scale)
        keep &= hull.find_simplex(profiles / scale, tol=1e-10) >= 0
    common = profiles[keep]
    info = {"profile_count": len(common), "profile_weighting": "equal pooled observed profiles",
            "support_rule": "within each year's 5th-95th marginal intervals and empirical 3D convex hull for A/age/male share",
            "support_caveat": "Empirical overlap screen; not a conditional instrument-support or identification test.",
            "common_marginal_lower": dict(zip(names, lower)), "common_marginal_upper": dict(zip(names, upper))}
    columns = ["year", "observed_profile_average_derivative", "common_profile_average_derivative",
               "common_profile_split_min_not_CI", "common_profile_split_max_not_CI", "common_profile_count", "observed_profile_count"]
    if not len(common):
        info["status"] = "no_common_profiles; no common-profile effect reported"
        pd.DataFrame(columns=columns).to_csv(outdir / f"time_heterogeneity_{label}.csv", index=False)
        return info
    rows = []
    assert len(models) % 2 == 0
    for year in years:
        reference = np.column_stack([common, np.full(len(common), year)])
        means = np.array([evaluate(model, reference)[1].mean() for model in models])
        split_means = means.reshape(-1, 2).mean(axis=1)
        observed = d.loc[(d.year == year) & d.trim_keep]
        observed_means = [evaluate(model, inputs(observed)[0])[1].mean() for model in models]
        rows.append({"year": int(year), "observed_profile_average_derivative": float(np.mean(observed_means)),
                     "common_profile_average_derivative": float(means.mean()),
                     "common_profile_split_min_not_CI": float(split_means.min()),
                     "common_profile_split_max_not_CI": float(split_means.max()),
                     "common_profile_count": len(common), "observed_profile_count": len(observed)})
    table = pd.DataFrame(rows, columns=columns)
    table.to_csv(outdir / f"time_heterogeneity_{label}.csv", index=False)
    info.update({"status": "reported", "estimates": rows})
    if label == "primary":
        d.loc[keep, ["cod_com", "year"] + names].rename(columns={"cod_com": "source_commune", "year": "source_year"}).to_csv(outdir / "common_time_profiles.csv", index=False)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        ax.plot(table.year, table.common_profile_average_derivative, marker="o", color="#185b75", label=f"Same {len(common)} profiles in every year")
        ax.plot(table.year, table.observed_profile_average_derivative, marker="s", ls="--", color="#bb6b31", label="Each year's observed profiles")
        ax.axhline(0, color="grey", lw=.8)
        ax.set_xticks(years, [str(year) for year in years], rotation=45)
        ax.set_ylabel("Estimated marginal response\n(outcome points / log immigration)")
        ax.set_title(OUTCOMES[outdir.name] + " — time heterogeneity\nAdditive commune effects; no confidence bands", fontsize=11)
        ax.legend(fontsize=9)
        fig.savefig(outdir / "time_heterogeneity.png", dpi=150)
        plt.close(fig)
    return info


def figures(cells, outcome, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = cells[cells.trim_keep].copy()
    d["age_group"] = pd.qcut(d.age, 3, labels=["Lower mean age", "Middle mean age", "Higher mean age"])
    d["male_group"] = pd.qcut(d.male_share, 3, labels=["Lower male share", "Middle male share", "Higher male share"])
    rows=[]
    fig, axs = plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax, group, title in zip(axs,["year","age_group","male_group"],["By year","By commune mean age","By commune male share"]):
        tab=d.groupby(group,observed=True).marginal_effect.agg(["mean","size"])
        ax.plot(np.arange(len(tab)),tab["mean"],marker="o",color="#185b75")
        ax.set_xticks(np.arange(len(tab)), [str(i) for i in tab.index],rotation=45,ha="right")
        ax.axhline(0,color="grey",lw=.8); ax.set_title(title)
        for k,r in tab.iterrows(): rows.append({"grouping":group,"group":str(k),"average_derivative":r["mean"],"cells":int(r["size"])})
    axs[0].set_ylabel("Estimated marginal response\n(outcome points / log immigration)")
    fig.suptitle(OUTCOMES[outcome]+" — average effects at observed cell profiles\nDescriptive heterogeneity; no confidence bands",fontsize=12)
    fig.savefig(outdir / "heterogeneity.png",dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(outdir / "heterogeneity.csv",index=False)


def self_checks():
    rng=np.random.default_rng(59)
    d=pd.DataFrame({"cod_com":np.repeat(np.arange(8),5),"year":np.tile(np.arange(5),8),"respondents":rng.integers(20,100,40)})
    p=Absorb(d,"respondents")
    X=rng.normal(size=(40,4))
    Dw=p.sw[:,None]*p.D
    explicit=p.sw[:,None]*X-Dw@linalg.lstsq(Dw,p.sw[:,None]*X)[0]
    assert np.allclose(p.transform(X),explicit,atol=1e-10)
    assert np.allclose(p.Q.T@p.transform(X),0,atol=1e-10)
    assert np.allclose(p.cross_kernel(X@X.T,p), p.transform(X)@p.transform(X).T,atol=1e-10)
    # Kernel dual equals explicit two-stage ridge in finite-dimensional features.
    F=rng.normal(size=(30,4)); Z=rng.normal(size=(30,6)); Z2=rng.normal(size=(23,6)); y=rng.normal(size=23)
    lam,xi=.03,.04
    gamma=linalg.solve(Z@Z.T+30*lam*np.eye(30),Z@Z2.T,assume_a="pos")
    Fhat=gamma.T@F
    primal=linalg.solve(Fhat.T@Fhat+23*xi*np.eye(4),Fhat.T@y,assume_a="pos")
    dual=gamma@linalg.solve(Fhat@Fhat.T+23*xi*np.eye(23),y,assume_a="pos")
    assert np.allclose(primal,F.T@dual,atol=1e-10)
    # Analytic treatment derivatives must agree with numerical derivatives.
    model={"centers":rng.normal(size=(20,4)),"widths":np.array([.7,1.2,.6,2.]),"coef":rng.normal(size=20)}
    s=rng.normal(size=(12,4)); eps=1e-5
    sp=s.copy();sm=s.copy();sp[:,0]+=eps;sm[:,0]-=eps
    analytic=evaluate(model,s)[1]
    numeric=(evaluate(model,sp)[0]-evaluate(model,sm)[0])/(2*eps)
    assert np.allclose(analytic,numeric,atol=1e-7)
    # Unpenalized FE shifts cannot alter projected features or stage outcomes.
    shift=p.D@rng.normal(size=p.D.shape[1])
    assert np.allclose(p.transform(shift),0,atol=1e-10)
    # Verify the overidentified benchmark against explicit dummy-variable 2SLS.
    test=pd.DataFrame({"cod_com":np.repeat(np.arange(25),8),"year":np.tile(np.arange(8),25),
                       "respondents":100,"age":rng.normal(size=200),"male_share":rng.normal(size=200)})
    for name in COMPONENTS:test[name]=rng.normal(size=200)
    error=rng.normal(size=200)
    test["A"]=test[COMPONENTS[0]]-.6*test[COMPONENTS[1]]+.5*error+rng.normal(size=200)
    test["y"]=3*test.A+.7*test.age+error
    test.attrs["instrument_mode"]="components"
    ap=Absorb(test)
    controls=np.column_stack([ap.D,test[["age","male_share"]]])
    regressors=np.column_stack([test.A,controls])
    instruments=np.column_stack([test[COMPONENTS],controls])
    iq=linalg.qr(instruments,mode="economic")[0]
    fitted=iq@(iq.T@regressors)
    explicit_beta=linalg.solve(fitted.T@regressors,fitted.T@test.y.to_numpy())[0]
    actual=linear_benchmark(test)
    assert np.isclose(actual['estimate'],explicit_beta,atol=1e-10)
    assert actual['excluded_instrument_rank']==11
    # The year factor must be full one-hot Gaussian, invariant to numeric labels.
    s=np.column_stack([rng.normal(size=(40,3)),np.tile(np.arange(2008,2018),4)])
    h=widths(s)
    onehot=pd.get_dummies(s[:,-1],dtype=float).to_numpy()/h[-1]
    expanded=np.column_stack([s[:,:-1]/h[:-1],onehot])
    assert np.allclose(kernel(s,s,h),np.exp(-.5*cdist(expanded,expanded,"sqeuclidean")),atol=1e-12)
    renamed=s.copy();renamed[:,-1]=9000-17*renamed[:,-1]
    assert np.allclose(kernel(s,s,h),kernel(renamed,renamed,h),atol=1e-12)
    # Finite-feature analogue: both-stage absorption equals joint regressions
    # with separate, unpenalized additive effects in the two commune samples.
    panel=pd.DataFrame({"cod_com":np.repeat(np.arange(20),6),"year":np.tile(np.arange(6),20),"respondents":100})
    d1,d2=split_cells(panel,55);p1,p2=Absorb(d1),Absorb(d2)
    F=rng.normal(size=(len(d1),5));Z1=rng.normal(size=(len(d1),7));Z2=rng.normal(size=(len(d2),7))
    yy=rng.normal(size=len(d2));lam,xi=.03,.04
    B=linalg.solve(Z1.T@p1.project(Z1)+len(d1)*lam*np.eye(7),Z1.T@p1.project(F),assume_a="pos")
    H=p2.project(Z2@B)
    theta=linalg.solve(H.T@H+len(d2)*xi*np.eye(5),H.T@p2.project(yy),assume_a="pos")
    A1=np.column_stack([p1.D,Z1]);pen1=np.diag(np.r_[np.zeros(p1.rank),np.full(7,len(d1)*lam)])
    coef1=linalg.solve(A1.T@A1+pen1,A1.T@F,assume_a="pos")
    assert np.allclose(coef1[p1.rank:],B,atol=1e-10)
    A2=np.column_stack([p2.D,Z2@B]);pen2=np.diag(np.r_[np.zeros(p2.rank),np.full(5,len(d2)*xi)])
    coef2=linalg.solve(A2.T@A2+pen2,A2.T@yy,assume_a="pos")
    assert np.allclose(coef2[p2.rank:],theta,atol=1e-10)
    return {"weighted_FE_matches_explicit_dummies":True,"kernel_dual_matches_explicit_two_stage_ridge":True,
            "analytic_derivatives_match_finite_differences":True,"fixed_effect_invariance":True,
            "projected_kernel_matches_projected_features":True,"vector_IV_matches_explicit_dummy_2SLS":True,
            "categorical_year_kernel_matches_full_one_hot_Gaussian":True,"year_labels_have_no_numeric_order":True,
            "absorbed_two_stage_ridge_matches_joint_unpenalized_effects":True}


def simulation_check(out):
    """End-to-end diagnostic with known contextual heterogeneity; no coverage claim."""
    rng=np.random.default_rng(5102)
    G,T=80,10;n=G*T
    ids=np.repeat(np.arange(G),T); years=np.tile(np.arange(2008,2018),G)
    time_s=(years-2012.5)/3
    age=rng.normal(size=n)+np.repeat(rng.normal(size=G),T)*.3
    gender=rng.normal(size=n)
    z=rng.normal(size=n)+.2*time_s
    u=rng.normal(size=n)
    a=1.5*z+.3*u+.2*rng.normal(size=n)+.1*age
    f=4*a+.8*a*age+.4*a*time_s+.3*gender**2
    y=f+2*u+np.repeat(rng.normal(size=G),T)+.5*time_s
    d=pd.DataFrame({"cod_com":ids,"year":years,"age":age,"male_share":gender,"A":a,"Z":z,"y":y,"respondents":100})
    d["trim_keep"]=d.groupby("year").A.transform(lambda s:s.between(s.quantile(.05),s.quantile(.95)))
    keep=d.trim_keep.to_numpy(bool);truth=4+.8*age+.4*time_s
    a1,a2=split_cells(d,79)
    estimates=[];detail=[];time_estimates=[]
    time_reference=np.column_stack([np.zeros((T,3)),np.arange(2008,2018)])
    for d1,d2 in [(a1,a2),(a2,a1)]:
        model,meta=fit(d1,d2)
        estimates.append(evaluate(model,inputs(d)[0])[1]);detail.append(meta)
        time_estimates.append(evaluate(model,time_reference)[1])
    estimate=np.mean(estimates,axis=0)
    result={"purpose":"Known-truth implementation diagnostic, not inference calibration", "communes":G,"years":T,
            "true_average_derivative":float(truth[keep].mean()), "estimated_average_derivative":float(estimate[keep].mean()),
            "derivative_rmse":float(np.sqrt(np.mean((estimate[keep]-truth[keep])**2))),
            "derivative_correlation":float(np.corrcoef(estimate[keep],truth[keep])[0,1]),
            "time_heterogeneity_at_A_age_male_zero": {
                "years": list(range(2008,2018)),
                "true_derivatives": (4+.4*(np.arange(2008,2018)-2012.5)/3).tolist(),
                "estimated_derivatives": np.mean(time_estimates,axis=0).tolist()},
            "application_caveat": "Simulation instruments vary within communes; this diagnostic does not establish identification with the application's baseline shares.",
            "linear_benchmark":linear_benchmark(d),"fits":detail}
    save_json(out/"simulation_check.json",result)
    print(f"Known-truth simulation: average derivative {result['estimated_average_derivative']:.3f} vs truth {result['true_average_derivative']:.3f}; heterogeneity correlation {result['derivative_correlation']:.3f}",flush=True)
    return result


def write_report(results, out):
    mode=next(iter(results.values()))['data']['instrument_mode']
    lines=["PANEL KERNEL NPIV: ADDITIVE COMMUNE EFFECTS AND TIME-VARYING RESPONSE", "",
           "Specification: Ybar_mt = alpha_m + f(A_mt, agebar_mt, male_share_mt, year_t) + u_mt.",
           "Implementation: f(A,X,t) = b_t + g(A,X,t). Commune intercepts alpha_m and year main effects b_t are unpenalized.",
           "Year main effects are inside f; the first year's b_t is normalized to zero. There is no commune identity input in g.",
           "Gaussian product kernel for immigration/age/male share times a categorical year-indicator kernel; full interactions.",
           "The year factor is exp(-1[years differ]/h_time^2), with baseline h_time=1. Numeric distances between years are not used.",
           "Commune and year main effects are absorbed from constructed features in BOTH stages, not from raw A before constructing features.",
           "Primary target: equal-cell average derivative, excluding the bottom/top 5% of A within each year.",
           "Primary results average reciprocal first/second-stage fits over disjoint commune partitions; dependence from common shocks is not removed by this split.",
           "", "INFERENCE: The 2019 KIV paper supplies convergence results, not operational confidence intervals.",
           "KIV split ranges, bandwidth comparisons and heterogeneity plots are sensitivity results, NOT confidence intervals or hypothesis tests.",
           "Linear benchmark standard errors cluster by commune; they do not address dependence from shared shift-share shocks.",
           "", "PANEL EXTENSION: Projecting both structural and instrument features on commune/year effects adapts KIV to additive nuisance effects.",
           "The original independent-observation KIV guarantees do not automatically cover this panel extension.",
           "The fitted function is contextual (commune demographics), not individual treatment heterogeneity.",
           "Time inside the product kernel permits different treatment-response curves by year; unpenalized b_t absorbs common annual shifts.",
           "Causal interpretation requires valid within-panel IV moments and sufficient relevance for the nonlinear features. FE and regularization do not prove identification.",
           "", "TUNING: Continuous coordinate bandwidths equal training-sample median interpoint distances (paper Appendix A.11).",
           "The categorical time bandwidth is our panel modeling choice. Half/double-time-bandwidth sensitivities alter pooling across years in both stages.",
           "Lambda minimizes held-out projected feature error. Xi uses the paper's Algorithm 2 structural prediction MSE on the other sample.",
           "That xi score is not an IV moment test, and under endogeneity prediction MSE need not select the best causal estimator.",
           "Reported first-stage scores use a separate outer 5-fold split by commune, with lambda chosen inside each training fold.",
           "Target feature kernels and folds are identical in instrument/no-instrument comparisons.",
           "Absolute L1 levels are not directly comparable to earlier runs with different structural kernels.",
           "", "TIME HETEROGENEITY: Report both observed-profile year averages and averages over the SAME reference profiles in every year.",
           "Reference profiles are pooled observed A/age/male-share triples within every year's 5th-95th marginal intervals AND empirical 3D convex hull.",
           "The empirical overlap screen does not establish conditional instrument support or causal identification. No confidence bands are provided.",
           "", "DATA: Individual records are collapsed separately for each outcome after complete-case filtering.",
           "Cell demographics use the same respondents as the outcome; no additional survey weights are applied.",
           "The matched 2SLS benchmark uses these same cells and weights, so it is not the published respondent-level estimate.",
           f"PRIMARY INSTRUMENT REPRESENTATION: {mode}.",
           "Raw-share KIV inputs: 11 share2008_* variables, mean age, male share and year.",
           "Component KIV inputs: 11 share2008_country * ln_orig_country variables, mean age, male share and year.",
           "Raw shares are constant within commune: their standalone features are absorbed, but interactions with year/time-varying controls remain.",
           "The full share and share-times-shock vectors are invertible given year (all shocks are verified nonzero).",
           "They contain the same information conditional on year, but yield different Gaussian-kernel distances and regularization.",
           "The scalar Bartik sum is not used as the excluded instrument in the primary raw-share run.",
           "Linear benchmark: 11 country-specific share-times-shock instruments; raw shares alone are collinear with commune effects.",
           "Its joint first-stage Wald F uses all 11 terms; partial R2 is their joint incremental explanatory power.",
           "The linear instrument span is a restricted subset of the flexible share/year kernel instrument space.",
           "Using the separate vector requires validity of that richer instrument information; validity of one aggregate moment alone is insufficient.",
           "Original scalar-Bartik results are preserved in bartik/; run_kernel_bartik_original.py preserves the old implementation.",
           "Previous unrestricted commune/year-indicator results are preserved in indicators_in_f/ with run_kernel_indicators.py.",
           "Earlier additive-FE results with a numeric-year Gaussian kernel remain in absorbed_effects/.",
           "Zero median coordinate distances (tied shares) use the median positive distance; constant coordinates use bandwidth one.", ""]
    for y, r in results.items():
        p=r["primary"];fs=r["first_stage_diagnostics"];b=r["linear_equal"]
        lines += [OUTCOMES[y]+f" ({y})", f"  Respondents: {r['data']['eligible_respondents']:,}; cells: {r['data']['cells']}; communes: 101",
                  f"  Matched 2SLS: {b['estimate']:.4f}, cluster SE {b['cluster_se']:.4f}; first-stage F {b['first_stage_F']:.3f}, partial R2 {b['first_stage_partial_R2']:.4f}",
                  f"  Linear instrument rank: {b['excluded_instrument_rank']}; joint Wald covariance rank: {b['first_stage_F_covariance_rank']}",
                  f"  Aggregate-Bartik 2SLS reference: {r['linear_bartik_equal']['estimate']:.4f}, SE {r['linear_bartik_equal']['cluster_se']:.4f}, partial R2 {r['linear_bartik_equal']['first_stage_partial_R2']:.4f}",
                  f"  KIV average derivative: {p['average_derivative_trimmed']:.4f}",
                  f"  KIV split range (not CI): {p['split_range_not_CI']}",
                  f"  Outer embedding loss: with Z {fs['with_Z']['embedding_loss']:.6f}, without Z {fs['without_Z']['embedding_loss']:.6f}, gain {fs['embedding_loss_gain']:.6f}",
                  f"  Outer mean-prediction R2: with Z {fs['with_Z']['oof_mean_R2']:.4f}, without Z {fs['without_Z']['oof_mean_R2']:.4f}"]
        tm=p['time_heterogeneity']
        lines.append(f"  Common profiles for year comparison: {tm['profile_count']}; status: {tm['status']}")
        for row in tm.get('estimates',[]):
            lines.append(f"    {row['year']}: common-profile derivative {row['common_profile_average_derivative']:.6f}; observed-profile derivative {row['observed_profile_average_derivative']:.6f}")
        for name,s in r["sensitivity"].items(): lines.append(f"  Sensitivity {name} ({s['instrument_mode']}): average derivative {s['average_derivative_trimmed']:.4f}; split range {s['split_range_not_CI']}")
        for name,score in r.get("first_stage_sensitivity",{}).items():
            lines.append(f"  Outer first-stage sensitivity {name}: embedding loss gain {score['embedding_loss_gain']:.6f}, mean-prediction R2 with Z {score['with_Z']['oof_mean_R2']:.4f}")
        at_lam=sum(m['lambda_at_grid_boundary'] for m in p['fits']);at_xi=sum(m['xi_at_grid_boundary'] for m in p['fits'])
        lines += [f"  Tuning-grid boundary selections: lambda {at_lam}/{len(p['fits'])}, xi {at_xi}/{len(p['fits'])}",""]
        if fs['embedding_loss_gain'] <= 0:
            lines += ["  Interpretation: adding the instrument did not improve held-out embedding prediction overall.",
                      "  Small estimated effects may reflect regularization and limited identifying information; they are not evidence of a zero causal effect.", ""]
    lines += ["FILES", "  results.json: complete specification, tuning grids, split memberships, diagnostics and results.",
              "  summary.csv: compact comparison table.", "  OUTCOME/cells_primary.csv: observed profiles, marginal effects, g, year main effects inside f, and additive commune effects.",
              "  OUTCOME/heterogeneity.csv and .png: averages at observed profiles (composition differs across groups).",
              "  OUTCOME/time_heterogeneity_primary.csv and time_heterogeneity.png: year effects compared at common profiles and observed profiles.",
              "  OUTCOME/time_heterogeneity_VARIANT.csv: the same time comparison for every sensitivity fit.",
              "  OUTCOME/common_time_profiles.csv: the exact reference profiles held fixed across years.",
              "  OUTCOME/models_*.npz: model centers, bandwidths and coefficients for reproducing predictions.",
              "  self_checks.json: numerical correctness checks.", "", "Run: python3 project/npiv/kernel/run_kernel.py", ""]
    sim=out/"simulation_check.json"
    if sim.exists():
        sc=json.loads(sim.read_text())
        lines += ["KNOWN-TRUTH SIMULATION",f"  True average derivative: {sc['true_average_derivative']:.4f}; estimated: {sc['estimated_average_derivative']:.4f}.",
                  f"  Cell derivative RMSE: {sc['derivative_rmse']:.4f}; correlation with true heterogeneity: {sc['derivative_correlation']:.4f}.",
                  "  One diagnostic design; not a confidence-interval coverage study.", ""]
    (out/"report.txt").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--outcomes",nargs="+",default=["pc_19","pc_20","vict_agreg"],choices=list(OUTCOMES))
    parser.add_argument("--instruments",choices=["raw_shares","components","bartik"],default="raw_shares")
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--sensitivity-repeats",type=int,default=3)
    parser.add_argument("--seed",type=int,default=20261125)
    parser.add_argument("--output",type=Path,default=HERE)
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--simulation-only",action="store_true")
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    start=time.time()
    with threadpool_limits(limits=2):
        checks=self_checks();save_json(out/"self_checks.json",checks)
        print("Numerical checks passed",flush=True)
        if args.check_only:return
        simulation_check(out)
        if args.simulation_only:return
        results={};summary=[]
        for outcome in args.outcomes:
            print(f"\n{outcome}",flush=True)
            od=out/outcome;od.mkdir(exist_ok=True)
            d,metadata=load_cells(outcome,args.instruments)
            metadata.update({"specification":SPECIFICATION,"commune_identity_in_kernel":False,"year_indicators_in_kernel":True})
            primary,cells=ensemble(d,args.repeats,"equal",1.,args.seed,od,"primary")
            print("  independent outer first-stage diagnostics",flush=True)
            diagnostics=outer_diagnostics(d,args.seed+10000)
            sensitivity={}
            diagnostic_sensitivity={}
            for name,weight,bw in [("bandwidth_half","equal",.5),("bandwidth_double","equal",2.),("respondent_weights","respondents",1.)]:
                sensitivity[name],_=ensemble(d,args.sensitivity_repeats,weight,bw,args.seed,od,name)
            for name,time_bw in [("time_bandwidth_half",.5),("time_bandwidth_double",2.)]:
                sensitivity[name],_=ensemble(d,args.sensitivity_repeats,"equal",1.,args.seed,od,name,time_bandwidth=time_bw)
            if args.instruments != "bartik":
                alternate=d.copy()
                alternate.attrs["instrument_mode"]="components" if args.instruments=="raw_shares" else "raw_shares"
                name="alternate_instrument_representation"
                sensitivity[name],_=ensemble(alternate,args.sensitivity_repeats,"equal",1.,args.seed,od,name)
                diagnostic_sensitivity[name]=outer_diagnostics(alternate,args.seed+10000)
                name="instrument_bandwidth_double"
                sensitivity[name],_=ensemble(d,args.sensitivity_repeats,"equal",1.,args.seed,od,name,instrument_bandwidth=2.)
                diagnostic_sensitivity[name]=outer_diagnostics(d,args.seed+10000,instrument_bandwidth=2.)
            r={"data":metadata,"linear_equal":linear_benchmark(d),"linear_respondent_weights":linear_benchmark(d,"respondents"),
               "linear_bartik_equal":linear_benchmark(d,instrument_mode="bartik"),
               "primary":primary,"first_stage_diagnostics":diagnostics,"sensitivity":sensitivity,
               "first_stage_sensitivity":diagnostic_sensitivity}
            results[outcome]=r
            figures(cells,outcome,od)
            summary.append({"outcome":outcome,"kernel_instruments":args.instruments,"linear_instruments":"11 country share-times-shock terms" if args.instruments!="bartik" else "aggregate Bartik",
                            "2sls":r['linear_equal']['estimate'],"2sls_cluster_se":r['linear_equal']['cluster_se'],
                            "bartik_2sls_reference":r['linear_bartik_equal']['estimate'],
                            "kiv_average_derivative":primary['average_derivative_trimmed'],"split_min_not_CI":primary['split_range_not_CI'][0],
                            "split_max_not_CI":primary['split_range_not_CI'][1],"embedding_loss_gain":diagnostics['embedding_loss_gain'],
                            "first_stage_F":r['linear_equal']['first_stage_F'],"first_stage_partial_R2":r['linear_equal']['first_stage_partial_R2']})
            save_json(out/"results.json",{"specification":SPECIFICATION,"instrument_mode":args.instruments,"seed":args.seed,"repeats":args.repeats,"sensitivity_repeats":args.sensitivity_repeats,
                      "lambda_grid":LAM_GRID,"xi_grid":XI_GRID,
                      "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "source_sha256":hashlib.sha256(DATA.read_bytes()).hexdigest(),"elapsed_seconds":time.time()-start,"outcomes":results})
            pd.DataFrame(summary).to_csv(out/"summary.csv",index=False)
            write_report(results,out)
        print(f"\nSaved results to {out}; {time.time()-start:.1f} seconds",flush=True)


if __name__=="__main__":
    main()
