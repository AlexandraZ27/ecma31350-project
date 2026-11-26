# ECMA31350 project: replication of Ajzenman, Dominguez & Undurraga (2023) and NPIV application

Paper: "Immigration, Crime, and Crime (Mis)Perceptions", AEJ: Applied 15(4), 2023. Main text is `ref/immigration.pdf` (the same file previously called `paper.pdf`); the appendix is `ref/immigration-appendix.pdf`. Replication package: openICPSR (see `requirements.txt`), unzipped into `rep_data/Replication/` (gitignored).

## Start here — handoff for project partner's Claude

This file is the persistent project memory; the prior chat and the original machine's temporary environment are not required. All paths and commands below are relative to the Git repository root (called `project/` on the original machine). The repository is `AlexandraZ27/ecma31350-project`, branch `main`. Commit `495d0c3` contains the original completed full neural analysis and memo. The user's subsequent cleanup request removed all other NPIV implementations and results; the standalone retained implementation now lives in `npiv/neural/`.

- **Completed:** both neural stages, held-out diagnostics, analytic treatment derivatives, common-profile year curves, and 399 commune-bootstrap draws per outcome. The selected specification and numerical results are recorded below. `NPIV.tex` / `aux/NPIV.pdf` contain the memo in the requested order: specification, fit settings, first-stage diagnostics, results and inference. The user's original model exposition and LaTeX template are preserved.
- **Next requested research extension:** heterogeneous average marginal treatment effects by commune average age, commune education, and year. The section **Heterogeneous average marginal treatment effects: age, education, and year** is an implementation roadmap, not completed education results. Begin with the education-code audit in subsection 1; do not launch a new neural fit before validating the moderator. The current response allows the marginal effect of immigration to vary with commune mean age and year (immigration–age and immigration–year interactions). Education is absent, so an immigration–education interaction has not been fitted. Dedicated age/education subgroup estimates and inference have not been produced.
- **Decisions to retain:** full-panel average marginal derivatives as the primary target, with within-year 5th--95th percentile averages as sensitivity only; commune-year observations; eleven raw origin shares as instruments; flexible neural density and response; separate additive commune/year nuisance effects plus year inputs in the response; exact integrated second-stage loss; no explicit L2 penalty. The latest implemented specification supersedes earlier requests to put commune indicators inside the neural function. Use the current specification below, not the legacy notes, when extending it.
- **Inputs outside Git:** the raw replication package and the external `adenc.sty` LaTeX style. Obtain the package using the instructions in root `requirements.txt`; confirm `rep_data/Replication/data/analysis/IV_panel.dta` exists. That root file is a data-download note, **not** a pip requirements file. Python dependencies are in `npiv/neural/requirements-neural.txt`. The committed PDF can be read without installing TeX or obtaining the style.
- **Git hygiene and current scope:** run `git status --short --branch` before editing. The user explicitly requested removal of all other NPIV methods, including `results_series*`. Only `npiv/neural/` should remain under `npiv/`; do not restore kernel, forest, series, mixed-method comparison, or seed-experiment directories. Paper replication outside `npiv/` and reference papers are separate project materials. Keep `full_neural/` as the selected reference result and write future neural extensions separately.

Read `aux/NPIV.pdf`, this file's current-specification section, and the following saved artifacts before changing code:

| Artifact | Purpose |
| --- | --- |
| `npiv/neural/full_neural/results.json` | Sample/source hashes, folds, selected settings, point estimates and environment provenance |
| `npiv/neural/full_neural/<outcome>/first_stage_diagnostics.json` | Held-out density loss with and without instruments; outcomes are `pc_19`, `pc_20`, `vict_agreg` |
| `npiv/neural/full_neural/inference/summary.csv` | Primary full-panel estimates over all 1,010 cells, bootstrap SEs, p-values and intervals |
| `npiv/neural/full_neural/inference/trimmed_sensitivity.csv` | Separate 5th--95th percentile sensitivity averages over 910 original cells, with their own inference |
| `npiv/neural/full_neural/inference/time_inference.csv` and `time_endpoint_contrasts.csv` | Year curves and paired 2017-minus-2008 tests |
| `npiv/neural/full_neural/validation.json` and `inference/validation.json` | Saved verification of the estimator and inference outputs |

Outstanding substantive choices are annual versus 2008 baseline moderators and the harmonized education definition. Recommendations below are not already fitted choices. Audit the available data/dictionaries first; if the education mapping or intended target remains unresolved, explain the specific missing information before proceeding. Do not interpret a failure to reject zero as proof that immigration has no effect.

## Layout

- `npiv/neural/run_full_neural.py`: the only retained NPIV estimator; mixture-density first stage and tanh response network.
- `npiv/neural/panel.py` and `outcome_network.py`: standalone data/fold/fixed-effect utilities and neural-response helpers. No imports from deleted estimators.
- `npiv/neural/reference_design.json` and `reference_profiles/`: the selected sample hash, fold assignments and fixed time-comparison profiles; these contain no other-method estimates.
- `npiv/neural/full_neural/`: saved networks, diagnostics, derivatives and 399-draw inference. `bootstrap_full_neural.py` refits both neural stages under commune resampling with selected settings fixed.
- `npiv/neural/verify_full_neural.py`: independent reconstruction of saved predictions, CV losses, derivatives and fixed effects; `full_neural/refactor_validation.json` records the historical cleanup audit (then-primary trimmed target). `verify_inference.py` verifies current full-panel and trimmed sensitivity inference.
- `NPIV.tex` / `aux/NPIV.pdf`: preserved original exposition/template followed by the full neural memo. Edit memo prose in `npiv/neural/full_neural/memo_work/memo_body.tex` and regenerate with `npiv/neural/write_full_neural_memo.py`. The exact original exposition is stored separately in `memo_work/model_exposition.tex`, without archived superseded result sections.
- `rep_scripts/replicate.py` and `rep_results/`: the original paper replication, outside the NPIV cleanup. `specifications.tex` records the paper's regression specifications.
- Compile from the repository root with `latexmk -lualatex NPIV.tex`. The project `.latexmkrc` sends the PDF and all build intermediates to `aux/`; the final memo is `aux/NPIV.pdf`. Keep `NPIV.tex` in the root and do not restore a root PDF/build files. The template loads the external course style `../../../adenc.sty`.

## Environment

The recorded neural environment used Python 3.9.6 and the versions in `npiv/neural/requirements-neural.txt`. Create a local environment rather than relying on the absolute temporary interpreter path saved in the manifest:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r npiv/neural/requirements-neural.txt
```

Record the actual Python/package versions if the partner's platform requires changes, and run the numerical preflight below before fitting. These commands do not reproduce the original Python version automatically. The original machine had no Stata; the separate legacy paper replication also used `statsmodels`, `scikit-learn`, and `joblib`. Read `.dta` with `pd.read_stata(..., convert_categoricals=False)`; `cod_com` can load as a category, so cast to float. Do not use pooled education value labels without the year-specific validation described below.

## Replication status
- Tables 1-4 (TWFE, `areg` with cluster) and 6-9 (OLS HC1, 2SLS HC0 with normal p-values, as Stata `ivreg2, robust`) match the paper to the printed digit, including the first stage (6.81 (1.64), F = 17.35, partial R^2 = 0.10). One cosmetic difference: Table 1 col. 1 s.e. is 1.48 vs 1.49 published.
- Table 5 (and Table 10, homicides) cannot be reproduced: `homicides_OLS.dta` is confidential and not in the package.
- The completed paper replication used `data/analysis` and `code/`. One raw `.xls` with a non-ASCII filename failed to unzip; later raw files may be truncated. Those files were not needed for that replication, but the proposed education harmonization may require original annual files/dictionaries; do not assume the local raw-data extraction is complete.
- Run (from `project/`): `python -I rep_scripts/replicate.py rep_data/Replication/data/analysis rep_results`.

## Paper replication instruments (separate from neural NPIV)
Shift-share (Bartik): sum over 11 origin countries of (2008 municipal share of nationality n) x (log change / log level of nationality-n migrants in destinations other than Chile, UN data). Cross-section: `deltaimm_instr` for treatment `deltaimm` (`enusc_IV.dta`). Panel: `imm_instr` for `lnrate_stock_imm` (`IV_panel.dta`). Not re-derived from raw UN data.

## Current full neural specification — session record (2026-11-26)

This is the user's selected and only retained NPIV implementation. All other NPIV implementations and output directories were deleted at the user's request, including series methods. The original model exposition and LaTeX preamble were preserved exactly. Neural weights, cell derivatives, bootstrap draws and inferential estimates are unchanged by the cleanup. Original fit hashes are retained as provenance in `results.json`; `refactor_compatibility` separately records the extracted code and reference-design hashes. Original source files are available in Git history, not runtime dependencies.

- **Sample and outcomes:** 101 communes × 10 years (2008–2017), 1,010 equally weighted commune-year cells. Outcomes are complete-case cell means of `pc_19` (crime concerns), `pc_20` (preventive behavior), and `vict_agreg` (victimization), each multiplied by 100. Eligible respondent counts are 180,039 / 243,096 / 244,115. Controls are mean `edad` and mean `ismale` within the same outcome-specific cells; year is categorical.
- **Treatment units:** `A = lnrate_stock_imm = log(1 + rate_stock_imm)`, where the rate is immigrant stock per 10,000 residents. Verified numerically in `IV_panel.dta`; it is not simply log(rate). Derivatives are per unit of A. For a raw-rate derivative, multiply each cell's derivative by `1/(1+r)`; for a local 1% raw-rate increase, average `0.01*r/(1+r)*derivative`, not merely `0.01*average_derivative`.
- **Instruments:** the 11 separate raw `share2008_*` values (Haiti, Peru, Colombia, Venezuela, Bolivia, Argentina, Ecuador, US, Spain, China, Brazil), with demographics and year inputs. Do not substitute the scalar Bartik or multiply by annual shocks without explicitly changing the specification. Shares are time invariant, so usable within-panel instrument variation comes through share/time or other changing-feature interactions; their additive main effects are absorbed by commune effects.
- **Structural equation:** `Y_mt = alpha_m + b_t + g(A_mt, age_mt, male_share_mt, year indicators) + u_mt`. Both density and response are neural. Commune intercepts and year main effects are separate, unpenalized nuisance terms; `b_2008=0`. Commune ID is not a neural input. Project Y and the integrated neural prediction off commune/year indicators during every outcome step, then recover nuisance coefficients from `Y - integrated_prediction`. Never demean raw A before passing it through g. Year inputs allow nonlinear treatment/time interactions.
- **Final fitting settings:** density MLP hidden widths `[32,16]`, three Gaussian components per initialization, two seeds `[127,229]`; outcome widths `[8]` for crime/victimization and `[16,8]` for prevention, averaged over seeds `[101,211,307,401,503]`. Both stages: 50 full-batch Adam updates, learning rate 0.005, tanh hidden activations, zero L2 and dropout, train-only standardization, CPU float64. Density SD is softplus plus 0.05 in standardized A units. Finite architecture and early stopping still regularize. PyTorch 2.8.0 / NumPy 2.0.2 were used.
- **Selection:** five commune outer folds, seed 20261125; three commune inner folds for outer-training density selection/cross-fitting. MDN grid: `[16]`/`[32,16]`, 1/3/10 mixtures, 0/50/100/200/400 steps. Outcome grid: `[8]`/`[16,8]`, 0/50/100/200/400/600 steps; validation seeds `[17,43]`. Use held-out density NLL for stage one and integrated, nuisance-projected MSE for stage two. No-shares diagnostics are independently tuned. Final first-stage densities remain cross-fitted. The selected minimum outcome score is not an untouched test score; held-out nuisance effects are profiled using that fold's outcomes, so it is not a raw-level forecast for a new commune.
- **First stage:** with-shares NLL 1.214427 / 1.242907 / 1.243048, versus no-shares 1.529845 / 1.501870 / 1.501984; gains 0.315419 / 0.258963 / 0.258936. All improve in 4/5 folds. Mean-prediction R² is about .644 / .633 / .633 with shares, .151 / .166 / .165 without. These are level prediction scores, not within-FE partial R², F tests, or completeness tests. Outer with-shares MDNs select `[16]`, K=10, 50 steps; diagnostics evaluate the nested selection procedure, not an independent test of the final `[32,16]`, K=3 fit.
- **Primary point results:** full-panel average derivatives 0.884343 / 1.132278 / 0.980407 over all 1,010 equally weighted commune-year cells. Derivatives are computed analytically and averaged across five outcome networks. The within-year 5th--95th percentile averages 0.894663 / 1.181371 / 1.011892 (910 cells) are sensitivity checks only. This reporting change reuses the original networks, cell derivatives and bootstrap draws; no new model fitting or tuning was needed.
- **Prediction and year comparisons:** selected integrated CV MSE remains 14.394631 / 6.192943 / 15.159819. Time curves retain fixed common-support reference profiles (300 / 266 / 261), a separate standardized comparison from the primary full-panel average. The 2017-minus-2008 common-profile contrasts remain 0.252666 / 1.302860 / -0.022614, with bootstrap SEs 0.079000 / 0.353453 / 0.098679 and centered p-values 0.0275 / 0.0050 / 0.8100 (unadjusted, conditional approximation).
- **Inference:** each of the 399 saved draws refitted all ten cross-fitted MDNs and five response networks. Whole commune histories were resampled within original fold strata, with duplicate sources kept in one fold and assigned distinct intercept IDs. Seed 20261126; selected architectures/stopping/initialization seeds fixed. Scaling and FE projection were recomputed. The primary statistic averages all resampled cells; only the sensitivity statistic recomputes within-year 5th--95th cutoffs. Time reference profiles remain fixed. Primary bootstrap SEs are 0.338137 / 0.612911 / 0.477368; centered p-values are 0.0550 / 0.0875 / 0.0775. Symmetric 95% intervals are [-0.019044, 1.787730] / [-0.480142, 2.744698] / [-0.121880, 2.082693]. Primary bootstrap mean shifts are -0.256123 / -0.292061 / -0.187074. Trimmed sensitivity SEs remain 0.343948 / 0.638274 / 0.487603 with p-values 0.0550 / 0.0875 / 0.0775. Use `inference/summary.csv` for primary results and `inference/trimmed_sensitivity.csv` for sensitivity; do not transfer SEs between targets. Initialization ranges are not sampling intervals.
- **Saved-field compatibility:** in the original raw draw JSONs, `average_derivative_all_cells` is the full-panel target and `average_derivative` is the legacy trimmed alias. New draws additionally name the latter `average_derivative_trimmed`. Aggregation explicitly selects these fields; `inference/bootstrap_draws.csv` now uses `estimate` for the full-panel statistic and `trimmed_sensitivity_estimate` for the trimmed statistic. The point manifest retains both explicit averages, and `initialization_range_all_cells_not_CI` is the full-panel initialization range. Original fit hashes remain unchanged; `reporting_update` separately records compatible reporting-code hashes.
- **Numerical verification:** original point fits use adaptive Legendre quadrature and Hermite64 checks. Bootstrap uses a validated 32-node mixture Gaussian rule with an adaptive-rule fallback and a Hermite64 check on every fit; original-data refits recover point estimates within 1e-6, and nine resampled-fit comparisons agree within 1e-5. Both Torch and BLAS threading are limited for the parallel bootstrap, including `VECLIB_MAXIMUM_THREADS=1` for NumPy's macOS Accelerate backend.
- **Interpretation:** inference is a conditional bootstrap approximation, not weak-IV-robust inference or a proved result for early-stopped neural NPIV. It assumes independent communes conditional on common time shocks, permits within-commune dependence, and omits tuning uncertainty and approximation/early-stopping bias. Bootstrap mean shifts are reported. Good density/moment prediction does not establish exclusion or identification of heterogeneous causal derivatives.
- **References:** Hartford et al. (2017), `ref/hartford17a.pdf`, motivates the density/response objectives and held-out losses. `ref/og_npiv.pdf` is the January 2002 revision of Newey–Powell, published in Econometrica (2003); it discusses completeness, ill-posedness, and constrained-estimator consistency. Neither supplied paper proves the particular panel bootstrap used here.

### Reproduction and reporting

Run from the repository root with the environment above. Choose the relevant operation; these are **not** a single sequence that must be rerun for a handoff.

**Numerical preflight without fitting the panel or changing the reference results:**

```sh
.venv/bin/python npiv/neural/run_full_neural.py --check-only --output /tmp/ecma31350-neural-preflight
```

**Verify the saved full fit:** `.venv/bin/python npiv/neural/verify_full_neural.py`. This requires the original raw data with the recorded source hash; it reconstructs saved predictions/derivatives and refreshes verification outputs, without retuning the model. Review the resulting Git diff.

**Rebuild the primary and sensitivity inference from existing draws without fitting:**

```sh
.venv/bin/python npiv/neural/bootstrap_full_neural.py --summarize-only --replicates 399
.venv/bin/python npiv/neural/verify_inference.py
```

**Rebuild only the memo from saved estimates and inference:**

```sh
.venv/bin/python npiv/neural/write_full_neural_memo.py
latexmk -lualatex NPIV.tex
```

Edit memo prose in `npiv/neural/full_neural/memo_work/memo_body.tex`; the generator inserts tables/figures and replaces only the full-neural block. `memo_work/model_exposition.tex` is the exact preserved exposition/template prefix, not the current report. The template loads `../../../adenc.sty`, outside this repository. Supply the original style from the course workspace when compiling; retain the user's template. If restricted caches prevent compilation, create a writable directory under `/tmp` and set `TEXMFVAR` and `TEXMFCACHE` to it; likewise use a writable `MPLCONFIGDIR` for figures.

**Refit only when intended:** `.venv/bin/python npiv/neural/run_full_neural.py --output npiv/neural/full_neural_rerun` fits the existing specification in a fresh directory. It reads `reference_design.json` for the original source hash/fold design and `reference_profiles/<outcome>.csv` for fixed time profiles. These are neural-only design inputs. `--reuse-density` can reuse compatible saved density checkpoints in the chosen output directory, but still retunes/refits the outcome stage and is not needed for the memo. A fresh directory alone does not add education to the feature construction.

**Bootstrap only when intended:** `.venv/bin/python npiv/neural/bootstrap_full_neural.py --replicates 399 --workers 4` uses the existing baseline model and resumes its saved draws. For a new model, first change the source-directory handling as described in heterogeneity subsection 5. Its `--output` option changes only the inference destination, not the model being read. Never mix checkpoint draws after changing features, sample, tuning, or evaluation targets; use fresh directories and verify hashes. `memo_quadrature_check.py` performs additional numerical integration checks, not a new model selection. Existing completed results do not require rerunning fits or bootstrap for review.

## Heterogeneous average marginal treatment effects: age, education, and year

**Meaning of heterogeneity:** immigration is the treatment throughout. Age and education are commune characteristics that may moderate its effect. The target is the immigration derivative `partial g(A, age, educ, male_share, t) / partial A`, averaged at different age or education values or within prespecified groups. These comparisons concern immigration–age and immigration–education interactions, with immigration–year interactions for time heterogeneity; they do not estimate the causal effects of changing age or education. A joint nonlinear network can represent these interactions without explicit linear product terms.

This section is the requested implementation roadmap. The education extension and age/education subgroup inference have **not** been fitted in this session. Preserve `full_neural/` as the reference result; write the extension to a new `npiv/neural/full_neural_heterogeneity/` directory.

### Exposition: how heterogeneous immigration effects are computed

Use this explanation when briefing the project partner. Distinguish the fitted response, the derivative of that response, and the population over which derivatives are averaged. The year results already exist; the age and education extensions below must not be described as completed analyses.

#### What varies, and what remains the treatment

Immigration is always the treatment: `A = log(1 + r)`, with r the immigrant stock per 10,000 residents. Mean age, resident education, male share and year describe a commune-year profile. We ask whether a small change in immigration has a different fitted effect at different profiles. The current structural response is

`Y_mt = alpha_m + b_t + g(A_mt, age_mt, male_share_mt, t) + u_mt`.

The planned education extension adds education as another joint input of g. For intuition only, a simple interaction model could contain `g(A,h,e,t) = q(h,e,t) + A*(beta_0 + beta_age*h + beta_educ*e + gamma_t)`. Its immigration derivative is `beta_0 + beta_age*h + beta_educ*e + gamma_t`: the slope with respect to immigration varies with age, education and year. This illustrative formula is not our fitted specification. The tanh network permits more general dependence, including slopes that also vary with the immigration level itself, without explicitly adding linear product variables.

Thus an age comparison concerns the **immigration slope at different mean ages**, not the slope of the outcome with respect to age. An education comparison has the same interpretation. Neither comparison estimates a causal effect of changing a commune's age or schooling composition.

#### How the two stages produce the response whose slope we evaluate

The first-stage mixture-density network estimates `p(A | eleven raw origin shares, controls, year)`. The second stage learns g by matching observed outcomes to the integral of g over that fitted treatment distribution, after accounting for additive commune and year effects. It uses the whole fitted distribution, not only a first-stage predicted treatment mean. Cross-fitting keeps a commune out of the density-training sample used for its fitted density.

Once fitted, we evaluate **the derivative of the structural network g with respect to A**. We do not differentiate the first-stage density, the integrated conditional-moment prediction, or the year fixed-effect coefficient. The additive terms `alpha_m` and `b_t` have zero derivative with respect to A. Including year inside g nevertheless lets the immigration slope differ across years.

This distinction is why a year curve can vary despite year main effects being absorbed. Likewise, age entering jointly with A allows immigration–age interactions; simply placing age outside g as an additive linear control would not represent those interactions.

#### From a fitted network to an average marginal effect

For each commune-year and each of the five fitted outcome-network initializations, compute `d_mt,s = partial_A g_s(A_mt, X_mt, t)`, holding the other inputs fixed. `outcome_network.py::evaluate` propagates the derivative analytically through the tanh layers using the chain rule. It accounts for treatment standardization and output scaling, so the result is in outcome points per unit of the original `A = log(1+r)`. Average the five derivatives to obtain `d_mt`. The stored `cells_primary.csv::marginal_effect` is this ensemble derivative.

For a small treatment change `Delta A`, the fitted local outcome change is approximately `d_mt * Delta A`. This is a local slope, not the effect of an arbitrarily large immigration change. For a raw-rate derivative use `d_mt/(1+r_mt)`; for a local 1% increase in the raw rate use `0.01*r_mt/(1+r_mt)*d_mt`. Apply these transformations cell by cell before averaging.

The primary result is `mean(d_mt)` over **all 1,010 commune-year cells**, giving every cell equal weight. The 5th--95th within-year treatment trim produces a separate 910-cell sensitivity average. Both use exactly the same fitted networks. Since derivatives also depend on the profile, every heterogeneous average must specify both what is being changed and which profiles are being averaged.

#### Two distinct ways to describe heterogeneity

**Observed-group averages** answer, for example, "What is the average fitted immigration slope among communes in the older-age group?" Define the group before examining its estimated effects, select its observed cells, and average their d values. Compute the younger-group average the same way. Their difference includes differences in the groups' immigration, male-share and year distributions as well as age. For an untrimmed observed-year average, average the 101 observed cell derivatives in that year.

**Standardized curves** instead keep a common collection of reference profiles and its weights fixed. For an age grid value h, replace only the age coordinate in each reference profile with h, evaluate the immigration derivative, and average: `B_age(h) = mean_i partial_A g(A_i, h, other_controls_i, t_i)`. After education is validated and added to both stages, construct the education curve analogously. Compare only supported profile combinations. If the data do not support a common reference population across the requested grid, narrow the grid or explicitly change the target rather than relying on extrapolation.

For year heterogeneity, replace only the year category: `B_year(t) = mean_(a,x in C) partial_A g(a,x,t)`. Year is categorical, so we set its indicators and then differentiate with respect to **immigration**, not with respect to time. We use one shared fitted response for all years, not a separately fitted model for each year. Differences across this curve reflect variation in the fitted immigration response at the same reference profiles.

The year figure in `NPIV.tex` implements this standardized comparison. C contains the same 300 / 266 / 261 treatment-age-male-share profiles for every year for crime concerns / prevention / victimization. These profiles pass the intersection of the years' marginal 5th--95th percentile ranges and empirical convex hulls. That support screen is separate from the primary full-panel average and its trimmed sensitivity check. The figure is therefore **not** the average over each year's actual 101 communes.

**Saved-column caveat:** `full_neural/<outcome>/time_heterogeneity.csv::common_profile_average_derivative` is the standardized curve used in the memo. Its existing `observed_profile_average_derivative` column is an older, within-year **trimmed** observed-profile average. Do not relabel it as the full-panel-by-year result. Full-panel observed-year point estimates can be computed from `cells_primary.csv::marginal_effect`; their bootstrap uncertainty is not recoverable from the saved overall and common-profile averages alone.

#### What uncertainty and a heterogeneity test mean

Each of the 399 existing bootstrap draws resamples complete commune histories and refits both neural stages under the selected fitting settings. It then computes the full-panel average, trimmed average and standardized year curve. The distribution across draws supplies sampling uncertainty under the conditional bootstrap approximation. The spread across five initialization seeds measures optimization sensitivity and is not a sampling standard error.

To test heterogeneity, compute a **difference within each draw**. For example, form `Delta_b = B_year,b(2017) - B_year,b(2008)`, then use the SD of those paired differences for the contrast SE and the centered-bootstrap calculation below for its p-value and interval. Apply the same principle to older-minus-younger or higher-minus-lower education comparisons once those targets are computed in every draw. A significant effect in one group and an insignificant effect in another do not establish a significant difference between them.

The memo currently provides pointwise year bands and the paired 2017-minus-2008 test. It does not provide an omnibus test of all years or simultaneous bands. The bootstrap conditions on selected architecture/stopping choices; the time analysis also conditions on fixed reference profiles. Predictive first-stage gains and narrow bands do not by themselves establish instrument validity, completeness, or accurate heterogeneous causal derivatives.

#### What the partner can say has actually been done

| Quantity | Current status |
| --- | --- |
| Immigration derivative at each observed commune-year | Estimated and saved for all three outcomes |
| Full-panel average and trimmed sensitivity average | Reported with separate bootstrap inference |
| Immigration–year heterogeneity at fixed common profiles | Reported with pointwise bootstrap bands and an endpoint contrast |
| Immigration–age heterogeneity | Allowed by the current response; dedicated age curves/groups and their inference have not been produced |
| Immigration–education heterogeneity | Not fitted; requires validated education data, expansion of both stages and new target-specific bootstrap calculations |

The short explanation to give the partner is: "We fit an IV response function, calculate its immigration slope at a specified profile, and average those slopes over a clearly defined set of profiles. We then compare the averages across age, education or year, using paired commune-bootstrap calculations to assess uncertainty in the differences."

### 1. Define and validate the commune characteristics

- **Age:** the current `age` column is outcome-specific commune-year mean respondent age. Use it for contemporaneous age heterogeneity. If the target is heterogeneity by a predetermined commune characteristic, construct 2008 mean age once and carry it across all years; include that baseline moderator in both stages. State which target is being estimated. Do not silently replace annual mean age with a mean over the whole 2008–2017 period.
- **Education:** the panel contains `nivel` (highest level), `curso`, and `termino` (completion), rather than a verified comparable years-of-schooling measure. Harmonize these using the original year-specific ENUSC dictionaries/questionnaires before aggregation. **Do not average numeric `nivel` codes or trust a pooled Stata value-label mapping across years.** The current pooled labels assign about 19–24% of pre-2015 records to “Prekinder/Kinder” but almost none from 2015 onward: this is a coding/label comparability warning, not evidence of a sudden schooling change. The original `code/1_create/ENUSC.do` appends annual files. Inspect its inputs and their dictionaries; if absent from the extracted package, obtain those originals before constructing education.
- A defensible default is the **share of resident respondents age 25+ who completed secondary education or more**, after the year-specific mapping and completion rules are verified. Alternatively use mean completed schooling years only if a validated crosswalk from level, grade, and completion exists. Document denominator, age restriction, treatment of old schooling systems, and missing/unknown codes; never turn missing schooling into zero. Record eligible education counts and missingness by commune/year. Do not substitute immigrant education (`*_years_ed_inmigr`) for resident commune education.
- Aggregate the chosen education measure consistently with the outcome sample and cell weighting, merge by `(cod_com, year)` with a validated many-to-one/one-to-one key, and report any changes from the 1,010-cell reference sample. For baseline education moderation, construct the validated 2008 measure and carry it across years. Annual age and education may be affected by immigration-induced composition changes; baseline moderation and contemporaneous conditional moderation answer different questions.

### 2. Refit both stages with the expanded features

Use the shared response `Y_mt = alpha_m + b_t + g(A_mt, age_mt, educ_mt, male_share_mt, year indicators) + u_mt`, adding baseline moderators explicitly if that is the chosen target. Put the same exogenous features in the MDN with the eleven raw shares; retain commune-clustered folds, additive nuisance effects, and nonlinear year interactions. Education must enter the network if education-dependent response slopes are the target. Grouping the old fitted derivatives by a new education variable only summarizes composition; it does not estimate an education interaction absent from the fitted function.

Refactor the hard-coded feature slices in `panel.py::inputs`, `run_full_neural.py::continuous_inputs` / density scaling, and `outcome_network.py::scaling`, `context`, and `evaluate`. Current code assumes continuous response inputs `[A, age, male_share]`; appending a CSV column alone will silently fail to add education. Use a named feature-order manifest, training-only scales, analytic differentiation with respect to **A**, and an expanded common-profile schema. Retune both stages with the existing nested commune-fold loss criteria; use the current grids/seeds as the starting design, without assuming the same architectures or 50-step stopping point remain optimal. Check held-out MDN loss with/without shares for the expanded model and report support and sample changes.

### 3. Define the averages before inspecting their results

Let `d_mt = partial g(A_mt, X_mt, t) / partial A`. Use equal cell weights and all available panel cells for the primary averages. Report within-year 5th--95th treatment trimming as sensitivity only; do not silently impose it on the primary heterogeneity target.

- **Observed-group AMTE:** for a prespecified age or education group G, report `mean(d_mt | G)` on the full panel, overall and by year, plus separately labeled trimmed sensitivity averages. Fix bin definitions in advance; for baseline commune groups, use quantiles over the 101 distinct communes, not over 1,010 repeated rows. Report commune counts and cell counts in each bin/year. This comparison includes differences in the distributions of other characteristics.
- **Standardized age curve:** evaluate `mean_i partial_A g(A_i, age=a, educ_i, male_share_i, t_i)` on an age grid, using the same supported reference profiles for each grid point. Define the analogous education curve by changing only education. These isolate variation in the fitted conditional slope from changes in the averaging distribution; they are not effects of intervening on age or education themselves.
- **Year heterogeneity:** evaluate the same fixed `(A, age, educ, male_share)` profiles at each of the ten year categories, then average derivatives. For age-by-year or education-by-year displays, keep the other coordinates and reference weights fixed. Recompute common support after adding education: the current three-dimensional hull/quantile screen is insufficient for a four-dimensional reference vector. Restrict grids to supported regions; if no useful overlap remains, report that result instead of extrapolating or silently changing reference populations across years.
- Keep derivative units explicit: the current treatment is `log(1+r)` for immigration per 10,000 residents. Transform cell derivatives by the chain rule before averaging if a raw-rate or percentage-change estimand is requested. These are commune-level average derivatives, not individual respondent treatment effects.

### 4. Conduct inference on the heterogeneity itself

Extend the commune bootstrap to refit **both expanded stages** and recompute every requested group average/curve using the same draw for all groups and years. Retain source-commune fold separation, including duplicate histories. State whether moderators, cutoffs, and reference profiles are fixed evaluation targets or re-estimated population summaries; the existing time procedure conditions on fixed profiles. Re-estimate full-sample tuning inside each draw only if claiming inference that includes model-selection uncertainty; otherwise retain and label the current conditional-on-selection approximation.

Save each draw's vector of AMTEs. Report point estimates, cluster-bootstrap SEs, centered-bootstrap p-values, and symmetric test-matched intervals with the same documented definitions as the memo. For heterogeneity tests, bootstrap **paired differences** such as older-minus-younger, higher-minus-lower education, or 2017-minus-2008; do not infer differences from separate significance tests or overlap of marginal intervals. If testing an entire curve, use a centered joint max-statistic from the same bootstrap vectors and label resulting simultaneous bands; pointwise bands alone are not a joint heterogeneity test. Declare any multiplicity correction across outcomes, groups, and years. Include sensitivity to sample support, network/stopping choices, and bootstrap mean shifts; neither a predictive first-stage gain nor a small conditional p-value proves nonparametric causal identification.

Deliver an education codebook/crosswalk, cell-level moderator file, named model-feature manifest, held-out diagnostics, AMTE and contrast tables (estimate/SE/p/interval/counts), bootstrap vectors, plots with labeled uncertainty, and an updated memo. Validate that the mean of the reported cell derivatives reproduces each aggregate and that constant year/commune main effects have zero treatment derivative. Do not claim that the present saved networks already estimate education heterogeneity.

### 5. Implementation map and checkpoint safety

The current runner imports only `panel.py` and `outcome_network.py`. The first stage is an MDN. Inspect both shared feature constructors before modifying one caller; the removed methods are not required.

| File under `npiv/neural/` | Required extension work |
| --- | --- |
| `panel.py` (`load_cells`, `inputs`) | Add validated outcome-specific education aggregates and a named feature order; preserve keys, cell weights and explicit raw-share mode. `inputs` always uses the eleven raw shares and rejects a different instrument mode. Keep this invariant after merges/resampling; the loader retains original replication-data checks only to preserve the exact sample. |
| `run_full_neural.py` (`continuous_inputs`, density scaling, outcome packing/integration) | Extend both stages, remove fixed positional slices, and save feature/data hashes with every fit. For one added annual education scalar, input counts become 24 with shares, 13 without shares, and 14 in the response (including ten year indicators). Additional baseline moderators add further inputs. |
| `outcome_network.py` (`scaling`, `context`, `evaluate`) | Extend response scaling/context and analytic treatment differentiation; ensure derivative is with respect to original A units, not standardized A. |
| `bootstrap_full_neural.py` | Parameterize model source separately from output; propagate source to worker initialization. Current `SOURCE=HERE/'full_neural'` is hard-coded. Extend `fit_statistic` to return the full AMTE/contrast vector; update fixed three-coordinate profiles and 101-commune/10-year assertions if the validated sample changes. |
| `verify_full_neural.py` | Parameterize `OUT`, validate the feature manifest and changed input dimensions, and independently reconstruct new predictions and derivatives. It currently reads only `full_neural/` and checks 23/12 density inputs. |
| `write_full_neural_memo.py` and memo numerical checks | Parameterize their baseline paths or create extension-specific reporting/checking entry points. The current memo generator hard-codes `SOURCE=HERE/'full_neural'`; a new output folder does not redirect it. Preserve the original exposition/template and reference results. |

Use a dedicated extension runner or parameterize these paths before producing new-model inference; do not claim that a baseline bootstrap written into `full_neural_heterogeneity/` estimates the extension. Recompute four-coordinate common support `(A, age, educ, male_share)` rather than reusing the original three-coordinate profile CSVs.

Saved baseline draw JSONs contain resampling information, overall derivatives, and common-profile year averages; they do not retain fitted bootstrap networks or cell-level derivatives for new age/education groupings. They cannot provide subgroup SEs by relabeling the existing overall draws. Existing point-fit networks can produce age curves, but new age-group inference requires computing those targets in refits; education interactions require both expanded stages to be fitted.

For each new draw, save the model/data/feature/target hashes, resampled source communes, group counts, the ordered vector of estimates and integration checks. Use identical resamples for all contrasts and compatible outcome comparisons. Validate metadata before resuming; the current baseline runner skips existing draw filenames without a complete compatibility audit.

### 6. First implementation milestone and completion criteria

1. **Audit before fitting:** write the year-specific education crosswalk, validate its implied distributions, and save a moderator audit with missingness/denominators by commune/year. State annual versus baseline targets and fixed bin/reference definitions. Compare outcome-specific cell counts and keys with the reference data.
2. **Smoke check one outcome:** exercise the expanded pipeline and one bootstrap resample in a fresh directory before launching the full grid or 399-draw run. Check input dimensions/names, no source-commune leakage across folds, training-only scaling, density normalization/integration, and analytic derivatives against a small finite-difference check. Confirm additive commune/year effects have zero treatment derivative and that group averages match the saved cell derivatives. This is a validation milestone, not sufficient final inference.
3. **Fit and report:** tune both stages, fit all three outcomes, compute prespecified AMTEs/paired contrasts on supported profiles, and bootstrap the whole expanded estimator. Preserve full within-commune histories. Explain sample changes and whether tuning, cutoffs and reference profiles are held fixed. Do not describe pointwise uncertainty as simultaneous coverage.
4. **Finish the handoff:** save the artifacts listed in subsection 4, update this memory with exact commands/paths and completed versus outstanding work, and update the memo in the established order. Review `git diff --check` and `git status`; include only intended code/results, keep raw replication data excluded, and preserve unrelated local work.

For consistency with the current memo, for scalar target `theta_hat` and B bootstrap estimates define `e_b = theta_b - theta_hat`, SE as the sample SD of `theta_b` (ddof=1), and `p = (1 + count(abs(e_b) >= abs(theta_hat))) / (B+1)`. The reported symmetric 95% interval is `theta_hat ± c`, where c is the `ceil(.95*(B+1))`-th ordered absolute error (380th when B=399). Apply this to each **paired contrast** as its own scalar target. A joint curve test needs a separately documented centered max-statistic and simultaneous-band construction; these scalar formulas do not provide it automatically. Retain the conditional-inference and identification limitations stated above.
