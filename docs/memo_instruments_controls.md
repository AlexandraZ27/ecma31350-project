# Instruments, Controls, and Interactions: Recommendations for the First-Stage Extensions

*Internal memo for the implementation team. Covers the triple-selection Lasso (cross-section) and NPIV (panel) first stages, and the heterogeneity analysis.*

## 0. Rebuilding the datasets

The `data/` folder is gitignored, so each of us builds the datasets locally before running anything. The build takes a few minutes and is fully deterministic: everyone ends up with identical files.

### Prerequisites

You need Python 3.10 or newer (check with `python3 --version`; on a Mac, Homebrew's `python` or the python.org installer both work), and a few GB of free memory, since the build reads the 680 MB permit file and the 220 MB individual-level ENUSC panel. You also need the authors' replication package, unzipped so that its top-level `Replication` folder sits at `rep_data/Replication` in the repository root. That folder should contain `data/analysis`, `data/intermediate` and `data/raw`. The package is too large for the repo, so get it from [here](https://www.openicpsr.org/openicpsr/project/161061/version/V1/view?path=/openicpsr/161061/fcr:versions/V1&type=project). If you store it somewhere else, pass its location with `--rep-root` (see below).

### One-time setup

From the repository root, create a virtual environment and install the three dependencies:

```bash
cd /path/to/our-repo                 # the repository root
python3 -m venv .venv                 # add .venv/ to .gitignore if it isn't there yet
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas pyarrow
```

### Build

With the environment activated, run from the repository root:

```bash
source .venv/bin/activate             # if not already active
python -I dataset_construction/build_datasets.py
# or, if the replication package lives elsewhere:
# python -I dataset_construction/build_datasets.py --rep-root /path/to/Replication
```

The script must be run from the repository root, because all output paths are relative to it. It writes:

| Path | Contents | Tracked by git? |
|---|---|---|
| `data/cross_section.parquet`, `.dta` | 101 municipalities × 317 columns | no |
| `data/panel.parquet`, `.dta` | 1,010 municipality-years × 1,062 columns | no |
| `docs/data_dictionary.csv` | definition, timing and source of every variable | yes |
| `docs/variable_groups.json` | named lists of outcomes, instrument sets, control pools | yes |
| `docs/nationality_tiers.csv` | 2008 counts and tier membership per nationality | yes |
| `miscellaneous/build_log.txt` | construction log and validation checks | yes |

Use the `.parquet` files in Python and the `.dta` files in Stata or R. They contain the same data.

### Verify the build

Three quick checks confirm that your build matches everyone else's:

```bash
grep -E "Check (1|4)" miscellaneous/build_log.txt
git status --short docs/ miscellaneous/
python -c "import pandas as pd; print(pd.read_parquet('data/cross_section.parquet').shape, pd.read_parquet('data/panel.parquet').shape)"
```

The first command should show Check 1 as `exact` and Check 4 as `18.61 (6.79); paper 18.61 (6.79)`. The second should print nothing: the regenerated files in `docs/` and `miscellaneous/` should be identical to the committed versions, so any difference means your replication package or package versions differ from the reference build. If that happens, do not commit the changed files; post the `git diff` in the group chat instead. One difference is harmless: if you used `--rep-root`, the second line of `build_log.txt` records your path. The third should print `(101, 317) (1010, 1062)`.

If the script stops with "Required file not found" or "not found; pass --rep-root", the replication package isn't where the script expects it. Check the path and rerun with `--rep-root`.

## Summary

| | Cross-section, triple-selection Lasso | Panel, NPIV |
|---|---|---|
| Data | `data/cross_section.parquet` (101 × 317) | `data/panel.parquet` (1,010 × 1,062) |
| Outcome | `d_<outcome>`, 2017 − 2008 change, ×100 | `y_<outcome>`, municipality-year mean, ×100 |
| Endogenous variable | `deltaimm` (paper) | `lnrate_stock_imm` (paper's panel measure) |
| Main instrument set | `expanded_2008_primary` (35 shares) | `expanded_2008_primary_x_year` (35 shares × 9 years), dimension-reduced as in §4 |
| Comparison sets | `bartik`, `paper11_2008`, `expanded_2008_broad` (68), `expanded_2002_primary` | `bartik`, `paper11_2008_x_year` |
| Controls | `pretreatment_pool`, selected by Lasso | Two-way fixed effects plus a short list of baseline controls × year |
| Fixed-effect removal | Long difference (built into the data) | **Assumed: two-way within transformation** (§4) |

All variable sets referenced in this memo are named lists in `docs/variable_groups.json`, so nothing should need to be typed by hand:

```python
import json, pandas as pd
G  = json.load(open("docs/variable_groups.json"))
cs = pd.read_parquet("data/cross_section.parquet")
Z  = cs[G["cross_section"]["instruments"]["expanded_2008_primary"]]
X  = cs[G["cross_section"]["controls"]["pretreatment_pool"]]
```

Definitions, timing (pre- or post-treatment), and sources for every variable are in `docs/data_dictionary.csv`. Tier membership and 2008 counts per nationality are in `docs/nationality_tiers.csv`.

The build reproduces the paper where it should. The paper's 2008 shares are rebuilt from the raw permit and census data to within 3 × 10⁻⁸, and the cross-section reproduces Table VI's 2SLS estimate for crime concerns exactly: 18.61 (6.79). Any differences you find from the paper therefore come from your specification choices, not from the data.

---

## 1. Datasets in brief

The **cross-section** has one row per ENUSC municipality (101). Outcomes are long differences, 2017 minus 2008, scaled by 100 so coefficients are in percentage points and comparable to the paper's tables.

The **panel** is a balanced 101 × 10 panel (2008–2017). Its outcomes are municipality-year means of the individual ENUSC responses, again scaled by 100. `n_resp` (minimum 50, median 244 respondents per cell) is included so the regressions can weight cells the way the paper's individual-level regressions implicitly do.

Both files carry three immigration measures (all permits, permanent residence `_permiso`, visas `_visa`), every candidate instrument, and every candidate control. Interaction terms are not pre-built, with one exception: shares × year dummies in the panel, which are instruments, not regressors.

## 2. Identification framework (common to both estimators)

Both extensions identify the effect of immigration from the 2008 nationality shares, in the sense of Goldsmith-Pinkham, Sorkin and Swift (2020, "GPSS"). The paper's Bartik instrument is one particular weighting of the share instruments, with weights fixed by the UN shifts. Our designs let the data choose the weighting. The identifying assumption is that **each share used as an instrument** is uncorrelated with differential trends in the outcomes, conditional on the controls, except through later immigration. That makes this a shares-only design, and the shifts play no role in identification.

The paper's case for this assumption rests on Peru, which carries most of its Rotemberg weight, and on pre-trend tests for Peru and the top five countries. Moving to the primary tier changes what has to be argued in three ways.

**The primary tier adds 25 nationalities the paper never discusses**, and drops one it does. The 35 primary nationalities include 10 of the paper's 11. Haiti, with 406 immigrants in 2008, falls to the broad tier. The 25 additions fall roughly into three groups, each with a different threat:

*Latin American and Caribbean migrants* (Uruguay, Cuba, Mexico, Paraguay, the Dominican Republic, Panama, Costa Rica) follow the same network logic as the paper's Andean countries, and the paper's argument extends to them most naturally.

*High-income origins* (Germany, France, Italy, Canada, the UK, Sweden, Australia, Japan, Belgium, the Netherlands, Austria, South Korea) are concentrated in affluent municipalities, so their 2008 shares may proxy for municipal income or amenity trends rather than for immigrant networks. This is the most serious threat the expansion introduces.

*Smaller historically rooted communities* (Palestine, Israel, Taiwan, Russia, Poland, India) may have settlement patterns shaped by long-standing community institutions, which are plausibly exogenous, but they are thin in many municipalities.

The pre-registered tier rules (primary: ≥500 immigrants in 2008 and present in ≥25 sample municipalities; broad: ≥50 and ≥10) are the commitment. The resulting lists are whatever the data produce.

**Diagnostics to report with every share set.** First, a *balance table*: regress each share on the pre-treatment pool and report which characteristics predict it. GPSS recommend this for the high-weight shares, and the paper omits it. Second, *pre-trend tests*: regress pre-period outcomes on the shares. The ENUSC methodology changed in 2008, so pre-period comparisons are only partially informative, as the paper itself notes (§6). Third, for any estimator that weights shares, report *which shares carry the weight* (Rotemberg weights for Bartik-type estimators, the selected set and coefficients for Lasso). Fourth, an *overidentification test* (Hansen J). Its power is low, and a rejection can mean either heterogeneous effects or invalid instruments, but it is a diagnostic the just-identified paper design lacks.

**Near-collinearity.** The 35 primary shares sum to about 0.97 in each municipality, because the remaining nationalities are excluded. They are therefore nearly collinear with the intercept. Lasso handles this, but a post-Lasso 2SLS that keeps nearly all shares can become numerically unstable. In that case, drop the smallest selected share and note it.

## 3. Cross-section: triple-selection Lasso

### Specification

Take the structural equation Δy_m = α Δmigr_m + x_m′β + ε_m and the first stage Δmigr_m = z_m′π + x_m′γ + v_m. Here z is the share set and x the control pool. We recommend the Chernozhukov, Hansen and Spindler (2015) procedure for many instruments and many controls, in its union-of-selections form:

**Step 1, first stage.** Run a Lasso of `deltaimm` on (x, z), with both penalized. Keep the selected instruments ẑ and selected controls x₁, and form the post-Lasso fitted value d̂.

**Step 2, outcome equation.** Run a Lasso of the outcome on x and keep the selected controls x₂.

**Step 3, fitted-value equation.** Run a Lasso of d̂ on x and keep the selected controls x₃.

**Step 4, estimation.** Run 2SLS of the outcome on `deltaimm` and x₁ ∪ x₂ ∪ x₃, instrumenting with ẑ.

Report all three selected control sets and the selected instruments. As a check, also run the partialling-out (orthogonal-score) version, which R's `hdm::rlassoIV` implements. The two estimates should be close.

### Penalty

Use the plug-in penalty of Belloni, Chen, Chernozhukov and Hansen (2012), with heteroskedasticity-robust penalty loadings (the `hdm::rlasso` defaults: c = 1.1, γ = 0.1/log n), not cross-validation. Cross-validation optimizes prediction and comes without the guarantees that make post-selection inference valid.

In `hdm`'s normalization the objective is (1/n)·RSS + (λ/n)·Σⱼ ψⱼ|bⱼ|. scikit-learn's `Lasso` minimizes (1/2n)·RSS + α·‖b‖₁. So pass α = λ/(2n), and divide each standardized column by its loading ψⱼ before fitting. Standardize all regressors first. Partial out the unpenalized intercept, and the paper's `age` and `hombre` if they are forced in (see below).

With n = 101, the theory requires very sparse selections, a handful of variables per equation. If a step selects more than about 10, treat the result as fragile. Report how the selected sets and α̂ move across λ ∈ {0.5, 1, 2} × plug-in.

### Instruments

The main specification uses `expanded_2008_primary` (35 shares). Report four specifications alongside it.

**Paper replication.** `bartik`, just-identified, with `age` and `hombre`. This reproduces Table VI exactly and anchors everything else.

**Same instruments as the paper, data-weighted.** `paper11_2008` under the same Lasso procedure. This separates the effect of re-weighting from the effect of expanding the set.

**Sensitivity.** `expanded_2008_broad` (68 shares). Expect more selection instability and some shares that work like municipality dummies; Pakistan, for instance, has 75% of its sample immigrants in one municipality.

**Robustness.** `expanded_2002_primary`. These census-only shares predate the 2005–08 permit flows, which addresses the concern that the 2008 shares partly reflect early stages of the same shock. The paper's own 2002 shares are `paper11_2002` (`sh02p_*`); ours are `sh02_*`.

For the permanent-residence and visa endogenous variables (`deltaimm_permiso`, `deltaimm_visa`), the paper pairs each with shares built from the same permit type (`paper11_2008_permiso`, `paper11_2008_visa`). Our expanded shares exist only for all permits combined. Instrument validity does not require the shares to match the permit type, so the expanded shares remain valid instruments for either variable, but say so when reporting.

**Endogenous variable.** Use `deltaimm` as the main measure, for comparability with the paper. `d_lnrate_stock_imm` builds the 2002 base from *all* foreign-born residents, consistent with the panel and with our expanded shares, whereas `deltaimm` uses only 13 countries. Report it as a robustness check. Both are approximately log changes in the immigrant stock per capita, so the coefficients should be of similar magnitude, but verify that before comparing them directly.

### Controls

Use `pretreatment_pool` and let Lasso choose. Do not use `x_radio_pc` (measured in 2019) or any contemporaneous variable. The paper's own controls, `age` and `hombre`, are CASEN 2006 household-head means, so they are pre-treatment as well. The paper's text describes them as 2017 values, but the code (`IV_1.do`) builds them from CASEN 2006. Replicating the paper and restricting to pre-treatment controls therefore don't conflict. Two specifications are worth reporting: the main one, where nothing is forced in, and a comparison where `age` and `hombre` are forced in, for continuity with the paper. The pool also contains `x_c06_age` and `x_c06_hombre`, which are essentially the same CASEN 2006 measures. Lasso will choose one or the other, and post-Lasso estimates can become unstable if both enter, so drop the `x_c06_` versions whenever `age` and `hombre` are forced in.

A few groups in the pool deserve particular attention when interpreting results.

**Baseline immigration** (`x_lnrate_imm08`, `x_imm_flow_rate2008`). Municipalities with large 2008 shares of established groups also tend to have high 2008 immigration levels. If high-immigration places were on different outcome trends, this is the confounder, so check whether it is selected.

**Baseline outcome levels** (`x_y08_<outcome>`). These capture mean reversion, which the paper addresses only through a robustness check interacting the baseline outcome with time. Be careful: each municipality's 2008 level is estimated from a few hundred respondents, and its sampling error also enters the 2017 − 2008 change with the opposite sign. If `x_y08_<outcome>` is selected, report results with and without it. The pre-2008 levels (`x_pre06_*`) avoid that mechanical correlation, but have partial coverage and use the older methodology.

**Income and skill composition** (`x_c06_ln_ypc`, `x_c06_esc`, `x_imm02_univ_sh`). These are the natural controls for the high-income-origin threat described in §2.

**Missing values.** The pool contains some missing values (see `contains_missing` in the groups file): CASEN 2000 and 2003 coverage, and the ENUSC 2003/2005/2006 pre-period levels. Lasso cannot use rows with missing values. The rule we recommend is to drop any variable missing for more than 15% of the 101 municipalities. For the rest, impute the sample mean and keep the corresponding missing-indicator column (`x_c00_missing`, `x_c03_missing`, `x_pre0X_missing`) in the pool.

### Inference

Use heteroskedasticity-robust standard errors from the final 2SLS. Because municipalities with similar shares may have correlated errors (Adão, Kolesár and Morales 2019), also report standard errors clustered by region. There are only 15 regions, so use the wild cluster bootstrap for those. Finally, report a weak-identification-robust test: either the Anderson–Rubin test on the selected instruments, or the BCCH sup-score test, which uses all candidate shares and does not depend on the selection step.

## 4. Panel: NPIV first stage

### Fixed-effect removal (assumption)

**This memo assumes the two-way within transformation.** The model is y_mt = g(x_mt) + α_m + λ_t + ε_mt, where x_mt = `lnrate_stock_imm`. Both α_m and λ_t are removed by within transformation. We recommend this approach because it matches the paper's own panel specification (Table A4, Panel B, which uses municipality and year fixed effects in levels) and because the shares × year instruments are designed for it. Everything below assumes this choice. If the NPIV team prefers another approach, the data support it as well, and the instruments change accordingly. Under first differences, Δy_mt = g(x_mt) − g(x_m,t−1) + Δε_mt, so the network or series takes `lnrate_stock_imm` and `lnrate_stock_imm_lag` as inputs, and the instruments become the differenced share × year terms, sh × (D_t − D_t−1). Under a long difference (2017 versus 2008), use the cross-section with both endpoint levels of x as inputs and the time-invariant shares as instruments. Under correlated random effects, keep the shares × year instruments and add the municipality means of x and of the instruments as regressors.

For a balanced, unweighted panel the transformation is ẅ = w − w̄_m − w̄_t + w̄, applied to the outcome, every instrument, every control, and every basis function:

```python
def within(df, cols, unit="cod_com", time="year"):
    out = df[cols] - df.groupby(unit)[cols].transform("mean") \
                   - df.groupby(time)[cols].transform("mean") + df[cols].mean()
    return out
```

How this works depends on the estimator. For **series NPIV**, apply the transformation to each basis term pⱼ(x_mt), *not* to x before the basis is built: g(x − x̄) ≠ g(x) − mean of g(x), but because g is linear in its coefficients, demeaning the basis terms is valid. For **neural networks**, g is not linear in its parameters, so the transformation must be applied inside the loss, to the network's outputs g(x_mt). With only 1,010 rows, train on the full batch so the municipality and year means are always computed on the complete panel.

When weighting the regressions by `n_resp`, the panel matches the paper's individual-level regressions. Use the weights in the main specification and report an unweighted version as a robustness check. With weights, the closed-form demeaning above is no longer exact. Instead, either partial out the municipality and year dummies by weighted least squares, or alternate weighted demeaning by municipality and by year until the changes are negligible (a few iterations suffice in a balanced panel). For the neural networks, apply the same weighted transformation inside the loss. Cluster standard errors by municipality (101 clusters).

### Instruments

The main set is `expanded_2008_primary_x_year`: 35 shares × 9 years (2009–2017, with 2008 omitted because the base year is absorbed by the fixed effects) gives 315 instruments. That is a large number relative to 1,010 observations and 101 clusters, and an overfit first stage biases 2SLS toward fixed-effects OLS. Reduce the dimension in one of two ways. Report the first; use the second as a robustness check.

**(a) Lasso-selected shares × year.** Run the BCCH plug-in Lasso of each within-transformed basis term on the within-transformed instruments. Use penalty loadings that are cluster-robust by municipality, and take the union of the selected instruments across basis terms.

**(b) Shares × linear trend.** sh08_n × (year − 2008), giving 35 instruments. This mirrors the structure of the Bartik instrument, whose shift component trends smoothly over the period. The implementation team builds these from the time-invariant shares in `time_invariant_shares`.

Nonlinear basis terms need instruments that predict them, and instruments linear in the shares may not predict p₂(x), p₃(x), … well. Add powers of the first-stage fitted value: regress x on the selected instruments, form x̂, and include x̂² and x̂³ in the instrument set. Since x̂ is a function of the exogenous instruments, it is a valid instrument, and this is the standard way to supply the K ≥ J instrument functions Newey (2013) requires.

The comparison sets are `paper11_2008_x_year` (99 instruments, under the same dimension reduction) and `bartik` (`imm_instr`, just-identified, the paper's panel instrument).

### Basis and number of terms

For series NPIV, use J = 2 (linear) as the replication anchor, then J = 3 and J = 4 (quadratic and cubic, or a B-spline with one or two interior knots). Use the paper's panel first-stage partial R² as a rough guide to what is feasible. Newey's Gaussian heuristic (2013, §IV) implies each additional term inflates the coefficient variance by a factor of about 1/ρ². Report the first-stage partial R² for each basis term. Unless the linear term's within partial R² exceeds about 0.25, do not interpret anything beyond J = 3 substantively. Report linear functionals of g, such as the average derivative and g at the 25th, 50th and 75th percentiles of x, rather than the full curve, using 2SLS standard errors clustered by municipality as Newey suggests.

For **neural networks** (DeepIV, deep GMM, or adversarial GMM), 1,010 rows and 101 independent clusters are small for these methods. Use small architectures with strong regularization. Do cross-fitting and hyperparameter tuning with folds grouped by municipality, never splitting a municipality across folds. Compare the fitted g to the series estimate at J = 3: if the network shows curvature the series estimate does not, assume it is overfitting until shown otherwise.

### Controls

The within transformation absorbs every time-invariant baseline control, so baseline characteristics can enter only interacted with year. Their role here is to allow differential trends. Use a short, pre-registered list rather than the whole pool:

| Control | Variable | Reason |
|---|---|---|
| Baseline immigration level | `x_lnrate_imm08` | main confounder of share-based designs |
| Income | `x_c06_ln_ypc` | proxy for the high-income-origin threat (§2) |
| Skill mix of 2002 immigrants | `x_imm02_univ_sh` | same |
| Size | `x_ln_pop` | urban trends |
| North | 1[`region` ∈ {1, 2, 3, 15}] | Andean migration corridor; regional economic trends (mining) |

Interact each with year dummies (2009–2017), giving 45 columns, or with a linear trend for a parsimonious version. Selecting from the full pool × year with Lasso is possible but not recommended with 101 clusters.

Do not use `xt_casen_*` (CASEN waves after 2008 are post-treatment). Use the contemporaneous respondent means `edad` and `ismale` only when replicating the paper, since the paper's panel includes them as individual-level controls.

## 5. Interactions and heterogeneous effects

### Interactions the specifications above require

The panel needs two kinds: the share × year instruments, which are already built, and the baseline control × year terms of §4, which the implementation team constructs. The cross-section needs none for the main estimates.

### Heterogeneity

The moderator variables are still to be decided. Two rules should govern the choice. **Moderators must be pre-treatment**, otherwise the heterogeneity estimates conflate effect modification with effects of treatment on the moderator. **Pre-register a small number**, no more than three, to limit multiple testing. Candidates, all already in the datasets:

| Moderator | Variable | Motivation |
|---|---|---|
| Baseline immigration exposure | `x_lnrate_imm08` | do effects differ where immigrants were already common? (contact hypothesis) |
| Income | `x_c06_ln_ypc` | economic-threat channel |
| Baseline victimization | `x_y08_vict_agreg` | do concerns rise more where crime was already salient? |
| Ethnic distance of the immigrant population | `x_dist08_mean_distance` | the paper's own channel (Tables XI–XII), here at its 2008 value |
| Skill mix of earlier immigrants | `x_imm02_univ_sh` | the paper's education channel, measured before 2008 |
| North | region indicator, as above | regional migration corridor |

The paper's media-exposure channel, `x_radio_pc`, is measured in 2019. Use it only as an explicitly exploratory moderator, under the assumption that radio coverage changed little over the decade.

**Implementation, cross-section.** Estimate Δy = α Δmigr + δ (Δmigr × w̃) + w̃θ + x′β + ε, where w̃ is the standardized moderator, which also enters on its own. Instrument the interaction with d̂ × w̃, where d̂ is the first-stage fitted value. This is more parsimonious than interacting every selected share with w̃, and it keeps the instrument count at two. Report the Sanderson–Windmeijer conditional F statistic for the interaction's first stage. The paper's attempt to instrument interactions failed for weak instruments (footnote 32), so expect this to be the binding constraint. Prefer continuous moderators to median splits: with 101 municipalities, a split leaves about 50 per group, and a first stage within each group is likely to be very weak.

**Implementation, panel NPIV.** Heterogeneity in g along x itself (the shape of the dose-response) is what NPIV already estimates. Letting g also depend on a moderator, g(x, w), adds a full dimension, which 1,010 observations and 101 clusters cannot support beyond a single binary moderator. If needed, estimate g separately for north and non-north municipalities, and treat the results as exploratory.

**Not recommended at this sample size:** forest-based heterogeneity estimators (e.g., instrumental forests). Their data requirements far exceed 101 municipalities.

## 6. Data caveats

**Two endogenous variable definitions.** The paper's cross-section `deltaimm` uses a 2002 census base of 13 countries; its panel `lnrate_stock_imm` uses all foreign-born residents. Both are in the data (`deltaimm` and `d_lnrate_stock_imm` in the cross-section). Cross-section and panel estimates are not fully comparable for this reason alone.

**The PCA indices differ across datasets.** The paper normalizes `pc_19` and `pc_20` over 2008 and 2017 respondents in the cross-section, but over all 2008–2017 respondents in the panel. The panel indices are the paper's panel versions. Their 2017 − 2008 changes do not equal the cross-section's values (by up to 1.5 points on the ×100 scale), and the levels are not comparable across datasets. All other outcomes match to within rounding.

**Permits are cumulated.** Following the paper, stocks are cumulative counts of all permits granted, so someone who receives a visa and later permanent residence is counted twice. This affects stocks and shares, in the paper and in our data alike.

**Marga Marga municipalities in the paper's shares.** The census files code Quilpué, Limache and Villa Alemana with their pre-2010 codes (5106, 5505, 5108), while the permits and ENUSC use the post-2010 codes (5801, 5802, 5804). Our rebuild of the paper's shares matches the published values exactly *without* recoding. This suggests the paper's 2008 stocks for these three municipalities omit the 2002 census base. The paper shares (`sh08p_*`) reproduce the paper as published; the expanded shares (`sh08_*`) use harmonized codes. Before mentioning this externally, confirm with:

```python
c = pd.read_stata("rep_data/Replication/data/intermediate/CensoInmigr2002.dta")["cod_com"]
print("old codes:", c.isin([5106, 5505, 5108]).sum(), " new codes:", c.isin([5801, 5802, 5804]).sum())
```

Old codes present and new codes absent confirms the omission.

**Overlapping nationalities.** "yugoslavia" (from the census) and "serbia y montenegro" (from the permits) partly cover the same population. Both are in the broad tier only, so this matters only for broad-tier sensitivity checks. Also, 4,153 census residents (2.3%) have unnamed country codes; they enter denominators but not as instruments.

**Later inflows are not captured.** Haiti and Venezuela had small 2008 presences (Haiti 406 immigrants, broad tier only; Venezuela 6,880, primary tier). No share-based design can capture their post-2014 inflows, ours or the paper's. Our estimates, like the paper's, are identified mainly from the growth of established communities.

**ENUSC methodology changed in 2008.** The pre-2008 controls (`x_pre03_*`, `x_pre05_*`, `x_pre06_*`) are levels measured under the old methodology. They are fine as controls. Use them for pre-trend tests only with that caveat, and do not compute changes between them and 2008 levels as if they were on the same scale.

**Population is fixed at 2010.** Immigration rates use the INE 2010 population projection for every year, following the paper, so changes in rates reflect changes in immigrant stocks only.

**Media coverage.** `x_radio_pc` is missing for 2 municipalities whose names did not match, and it is measured in 2019.
