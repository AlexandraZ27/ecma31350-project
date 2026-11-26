"""Panel data, fixed effects, folds and serialization for the full neural NPIV fit."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ecma31350-neural-matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import linalg

warnings.filterwarnings("ignore", message=r"(divide by zero|overflow|invalid value) encountered in matmul", category=RuntimeWarning)
HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "rep_data/Replication/data/analysis/IV_panel.dta"
OUTCOMES = {"pc_19": "Crime concerns index", "pc_20": "Preventive behavior index",
            "vict_agreg": "Total victimization"}
COUNTRIES = ["hai", "per", "col", "ven", "bol", "arg", "ecu", "eeuu", "esp", "chi", "bra"]
SHARES = [f"share2008_{c}" for c in COUNTRIES]
# Retain the original sample checks against the replication data. These fields
# do not enter either neural network; the eleven raw shares are the instruments.
SHIFTS = [f"ln_orig_{c}" for c in COUNTRIES]
COMPONENTS = [f"z_component_{c}" for c in COUNTRIES]
SPECIFICATION = {'equation': 'Y_mt = alpha_m + f(A_mt, X_mt, year_t) + u_mt',
 'f_decomposition': 'f(A, X, t) = b_t + g(A, X, t)',
 'commune_effect': 'unpenalized additive intercept; no commune input in g',
 'year_main_effect': 'unpenalized b_t inside f; first observed year normalized '
                     'to zero',
 'interactions': 'immigration, demographics and categorical year enter jointly '
                 'through smooth tanh hidden layers',
 'feature_absorption': 'commune and year main effects projected from '
                       'integrated network predictions and outcomes; raw A '
                       'never demeaned',
 'causal_interpretation': 'requires valid panel IV moments and identification '
                          'of the nonlinear response; not established by '
                          'fitting',
 'first_stage': 'Gaussian mixture-density neural network on 11 raw shares, '
                'demographics and year indicators',
 'structural_class': 'compact tanh neural network in immigration, demographics '
                     'and year indicators',
 'coefficient_penalty': 0.0,
 'dropout': 0.0,
 'complexity_selection': 'network architecture and number of Adam steps chosen '
                         'by commune-grouped held-out integrated loss',
 'first_stage_selection': 'nested commune-held-out negative log density; '
                          'architecture, mixture count and Adam steps',
 'first_stage_scale_floor_in_training_A_sd': 0.05,
 'integration': 'adaptive rowwise Gauss-Legendre quadrature checked against '
                'componentwise Gauss-Hermite; density network frozen in stage '
                'two',
 'year_inputs': 'year indicators inside the outcome network, with unrestricted '
                'network interactions'}

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
    if instrument_mode != "raw_shares":
        raise ValueError("The retained neural specification uses eleven raw shares")
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



def inputs(d):
    if d.attrs.get("instrument_mode", "raw_shares") != "raw_shares":
        raise ValueError("The retained neural specification uses eleven raw shares")
    return (d[["A", "age", "male_share", "year"]].to_numpy(float),
            d[SHARES + ["age", "male_share", "year"]].to_numpy(float))


def folds(d, seed, n_folds=5):
    groups = np.random.default_rng(seed).permutation(np.sort(d.cod_com.unique()))
    return [d.cod_com.isin(ids).to_numpy() for ids in np.array_split(groups, n_folds)]
