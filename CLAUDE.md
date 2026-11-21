# ECMA31350 project: replication of Ajzenman, Dominguez & Undurraga (2023) and NPIV application

Paper: "Immigration, Crime, and Crime (Mis)Perceptions", AEJ: Applied 15(4), 2023. Main text is `paper.pdf`; the online appendix is `../appendix.pdf` (course folder, one level up). Replication package: openICPSR (see `requirements.txt`), unzipped into `rep_data/Replication/` (gitignored).

## Layout
- `rep_scripts/replicate.py`: Python re-implementation of Tables 1-4 and 6-9 (no Stata available). Output: `rep_results/replication_output.txt`.
- `npiv/npiv.py` + `npiv/make_tex.py`: NPIV on six summary outcomes (kernel-density first stage, honest forest f). Output: `npiv/results/`.
- `npiv/npiv9.py` + `npiv/make_tex9.py` + `npiv/reading9.tex`: 3x3 grid of first-stage class (kernel / forest / net) x f class (kernel / forest / net) for `del_problema3`. Outputs: `npiv/results9/` (cross-section), `npiv/results9p/` (panel), seed reruns in `npiv/seeds/` (5, cross-section) and `npiv/seedsp/` (3, panel). `npiv/test*/` are smoke tests and known-truth simulations (`npiv/test/sim_check.py`).
- `NPIV.tex`: the original NPIV notes plus two generated sections between `% BEGIN/END NPIV RESULTS` and `% BEGIN/END NPIV9 RESULTS`. Regenerate with `make_tex.py` / `make_tex9.py` (do not hand-edit those blocks). `specifications.tex` lists the paper's regression specifications. Other files in `rep_results/` and `rep_scripts/` (`dantzig_iv*`) are not from this work.
- Compile with `latexmk -lualatex NPIV.tex` (`adenc.sty` is in `coursework/`, loaded as `../../../adenc`).

## Environment
Stata is not installed; Python is 3.9. `pyfixest` and `pyreadstat` do not work on 3.9. A venv with `pandas numpy scipy statsmodels scikit-learn torch joblib pypdf` is enough (it lived in the session scratchpad and may need rebuilding). Read `.dta` with `pd.read_stata(..., convert_categoricals=False)`; `cod_com` can load as a category, so cast to float.

## Replication status
- Tables 1-4 (TWFE, `areg` with cluster) and 6-9 (OLS HC1, 2SLS HC0 with normal p-values, as Stata `ivreg2, robust`) match the paper to the printed digit, including the first stage (6.81 (1.64), F = 17.35, partial R^2 = 0.10). One cosmetic difference: Table 1 col. 1 s.e. is 1.48 vs 1.49 published.
- Table 5 (and Table 10, homicides) cannot be reproduced: `homicides_OLS.dta` is confidential and not in the package.
- Data: only `data/analysis` and `code/` matter. One raw `.xls` with a non-ASCII filename failed to unzip; later raw files may be truncated. Not needed.
- Run (from `project/`): `python -I rep_scripts/replicate.py rep_data/Replication/data/analysis rep_results`.

## Instruments (shipped in the data, not rebuilt)
Shift-share (Bartik): sum over 11 origin countries of (2008 municipal share of nationality n) x (log change / log level of nationality-n migrants in destinations other than Chile, UN data). Cross-section: `deltaimm_instr` for treatment `deltaimm` (`enusc_IV.dta`). Panel: `imm_instr` for `lnrate_stock_imm` (`IV_panel.dta`). Not re-derived from raw UN data.

## NPIV setup and findings
- Conventions: `p-hat(a|z,x)` is the first stage (the user's notation; avoid calling it g); f is the structural function. Tune per `../ECMA31350-Notes.tex` (honest forest: half-samples without replacement, random covariate subset, deep trees, leaf share alpha by CV; kernel bandwidths by leave-one-out CV; nets by normalised-gradient descent with momentum and early stopping).
- Samples: cross-section (101 communes, Tables 6-9 spec) and panel (1,010 commune-year cell means, two-way FE removed; analogue of Table A.4 Panel B, not respondent-level).
- Stage 2 minimises the exact loss in NPIV.tex. Do not fit Y on draws directly: it attenuates the slope (about 1.8 vs a true 10 in simulation).
- Finite-difference step 0.5 sd(A), trimmed 5%-95%; small leaves bias the derivative downward.
- `npiv9.py` first stages regress A on (z, x), cross-fitted, then resample out-of-fold residuals (location-model assumption, ours). Run: `python -I npiv/npiv9.py rep_data/Replication/data/analysis <outdir> <B_boot> <B_perm> [cross|panel|both]`; env var `NPIV_SEED` changes the seed. Full run takes about 7 minutes.
- Findings: cross-section NPIV estimates are unstable across seeds (e.g. 0.02 to 26.3 for one cell) and should not be interpreted. Panel: random-forest f gives 0.7-2.1 everywhere (shrinkage); kernel f with forest or net first stage gives 7.3-10.1 vs linear 2SLS 12.34 (4.35). First-stage out-of-fold R^2 is 0.01-0.10 (cross-section) and 0.14-0.16 (panel), gains all permutation p = 0.02 (min attainable with 50 draws).
- Paper's `del_problema3` estimates: TWFE 2.00 (1.49), OLS 4.08 (1.68), 2SLS 18.61 (6.79); the largest coefficient in Tables 6-9.

## Caveats to keep
- Bootstrap intervals hold tuning parameters and folds fixed; they do not cover seed variation or selection (some percentile intervals exclude their own point estimate).
- Tuning and fit reporting use the same folds (mildly optimistic first-stage fit).
- The paper's design is a continuous-treatment difference-in-differences (TWFE, and the 2008-2017 long difference); it has no spillover analysis and no modern heterogeneity-robust DiD estimators. Heterogeneity appears only as subgroup splits (Tables 11-13, Appendix Table A.16, Figure A.6).
- Known repo issue: `.gitignore` contains unresolved merge-conflict markers.
