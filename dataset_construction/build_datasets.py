"""Build the cross-sectional and panel datasets for the group project extending
Ajzenman, Dominguez & Undurraga (2023), "Immigration, Crime, and Crime (Mis)Perceptions".

Outputs (relative to the project root):
  data/cross_section.parquet / .dta   101 municipalities, 2008-2017 long differences
                                      (triple-selection Lasso first stage)
  data/panel.parquet / .dta           101 municipalities x 2008-2017 (NPIV first stage)
  docs/data_dictionary.csv            every variable: group, timing, source, definition
  docs/variable_groups.json           ready-made lists of outcomes, endogenous variables,
                                      instrument sets and control pools
  miscellaneous/build_log.txt         construction log and validation checks

Each dataset contains the 20 outcomes of Tables VI-IX, the three immigration measures,
all candidate instruments and all candidate controls. Interaction terms are left to the
estimation scripts, except shares x year dummies in the panel, which are instruments.

Share definitions
  sh08p_<c>   the paper's 2008 shares for its 11 countries, taken from enusc_IV.dta
              (census 2002 base for 13 countries + cumulative 2005-2008 permits).
              The script also rebuilds them from the raw data as a check.
  sh08_<n>    expanded 2008 shares for every nationality (2002 census of all foreign-born
              residents + cumulative 2005-2008 permits, numerator and denominator alike).
  sh02p_<c>   the paper's 2002-baseline shares (enusc_IV_share2002.dta).
  sh02_<n>    expanded 2002 shares (2002 census only, all foreign-born residents).
  Expanded shares are kept for the "broad" tier (>= 10 municipalities and >= 50
  immigrants in 2008 within the sample); "primary" (>= 25 municipalities and >= 500
  immigrants) is a subset. Tier lists are in docs/variable_groups.json.

Usage (from the project root):
    python -I dataset_construction/build_datasets.py
    python -I dataset_construction/build_datasets.py --rep-root rep_data/Replication

Requires: numpy, pandas (pyarrow for .parquet output; without it only .dta is written).
"""
import argparse, json, os, re, sys, unicodedata, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
PAPER11 = ["hai", "per", "col", "ven", "bol", "arg", "ecu", "eeuu", "esp", "chi", "bra"]
# cat_pais values in immigration_Correciones.dta, as recoded in IV_1.do
PAPER_CATPAIS = {"hai": "haiti", "per": "peru", "col": "colombia", "ven": "venezuela",
                 "bol": "bolivia", "arg": "argentina", "ecu": "ecuador",
                 "eeuu": "estados unidos", "esp": "espana", "chi": "china", "bra": "brasil"}
DOCS = {"": None, "_permiso": "pd", "_visa": "visa"}      # all permits, permanent, visas

TABLES = {
    "VI": ["del_problema3", "del_afecta3", "del_calvida3", "del_inseg5", "del_vict", "pc_19"],
    "VII": ["del_tend_barrio", "del_tend_com", "del_tend_pais"],
    "VIII": ["Index_Vivienda", "Index_Vecinos", "del_arma", "pc_20"],
    "IX": ["vict_hurto", "vict_robosorpr", "vict_robovehi", "vict_roboviv",
           "vict_lesiones", "vict_roboviolent", "vict_agreg"],
}
OUTCOMES = sum(TABLES.values(), [])
OUTCOME_TABLE = {y: t for t, ys in TABLES.items() for y in ys}

YEARS = list(range(2008, 2018))
IV_YEARS = list(range(2009, 2018))           # shares x year dummies; 2008 is the base year
TIERS = {"primary": dict(min_count=500, min_munis=25),
         "broad": dict(min_count=50, min_munis=10)}

# Municipalities of the Marga Marga province were recoded in 2010 (old -> new code).
# Sources differ in which code they use; codes are harmonized to the sample's code.
RECODES = {5106: 5801, 5505: 5802, 5507: 5803, 5108: 5804}

NONCOUNTRY = re.compile(r"^$|apatrid|no informa|no indica|sin info|desconoc|otr[oa]s?$|"
                        r"^nan$|indetermin", re.I)
# census 2002 name -> candidate names in the permit data (first one present is used)
CENSUS_ALIASES = {
    "uk": ["reino unido", "gran bretana", "inglaterra"],
    "paises bajos": ["holanda paises bajos", "holanda", "paises bajos"],
    "taiwan china": ["taiwan"],
    "egipto republica arabe de": ["egipto"],
    "republica de corea": ["republica de corea", "corea del sur", "corea"],
    "federacion de rusia": ["federacion de rusia", "rusia"],
}

CASEN06_VARS = ["hombre", "age", "ypc", "esc", "years_educ", "post_18years_ed_nativos"]
CASEN_OLD_VARS = ["hombre", "age", "ypc"]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
class Log:
    def __init__(self):
        self.lines = []

    def __call__(self, s=""):
        print(s)
        self.lines.append(str(s))


def norm_name(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = unicodedata.normalize("NFKD", str(v).strip()).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()


def map_unique(s, f):
    u = pd.unique(s)
    return s.map(dict(zip(u, [f(x) for x in u])))


def read(path, cols=None, required=True, log=None):
    if not os.path.exists(path):
        if required:
            sys.exit(f"Required file not found: {path}")
        if log:
            log(f"  note: optional file not found, skipped: {path}")
        return None
    if cols is not None:
        with pd.io.stata.StataReader(path) as r:
            have = list(r.variable_labels())
        missing = [c for c in cols if c not in have]
        if missing:
            if required:
                sys.exit(f"{path} lacks {missing}")
            cols = [c for c in cols if c in have]
    return pd.read_stata(path, columns=cols, convert_categoricals=False)


def harmonize(codes, sample):
    """Map Marga Marga municipality codes to whichever version the sample uses."""
    c = pd.to_numeric(codes, errors="coerce").astype("Int64")
    for old, new in RECODES.items():
        if new in sample:
            c = c.mask(c == old, new)
        elif old in sample:
            c = c.mask(c == new, old)
    return c


def slugify(name, used, width=14):
    s = re.sub(r"_+", "_", re.sub(r"[^a-z0-9]", "_", name))[:width].strip("_")
    base, k = s, 2
    while s in used:
        s = f"{base[:width - 2]}{k}"
        k += 1
    used.add(s)
    return s


def tsls(y, X, Z):
    """2SLS with HC0 standard errors (Stata ivreg2, robust)."""
    Xh = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    b = np.linalg.solve(Xh.T @ X, Xh.T @ y)
    e = y - X @ b
    A = np.linalg.inv(Xh.T @ X)
    V = A @ ((Xh * (e ** 2)[:, None]).T @ Xh) @ A.T
    return b, np.sqrt(np.diag(V))


# --------------------------------------------------------------------------------------
# Shares
# --------------------------------------------------------------------------------------
def paper_shares(perm, census13):
    """Rebuild share2008_<c>[_permiso|_visa] exactly as IV_1.do does: stocks are the
    13-country census base plus cumulative permits, read off the municipality's 2008 row
    (municipalities without permits in 2008 get zero shares)."""
    p = perm[perm["year"] <= 2008].copy()
    inv = {v: k for k, v in PAPER_CATPAIS.items()}
    p["code"] = p["cat_pais"].map(inv)
    has2008 = set(p.loc[p["year"] == 2008, "cod_com"])
    flows = {}
    for j, doc in DOCS.items():
        q = p if doc is None else p[p["doc"] == doc]
        flows[f"imm{j}"] = q.groupby("cod_com").size()
        for c in PAPER11:
            flows[f"{c}{j}"] = q[q["code"] == c].groupby("cod_com").size()
    F = pd.DataFrame(flows).fillna(0)
    C = census13.set_index("cod_com")
    idx = sorted(set(F.index) | set(C.index))
    F, C = F.reindex(idx, fill_value=0), C.reindex(idx).fillna(0)
    out = pd.DataFrame(index=idx)
    for j in DOCS:
        stock = C["imm"] + F[f"imm{j}"]
        for c in PAPER11:
            sh = (C[f"imm_{c}"] + F[f"{c}{j}"]) / stock
            sh = sh.replace([np.inf, -np.inf], np.nan).fillna(0)
            out[f"share2008_{c}{j}"] = sh.where(out.index.isin(has2008), 0.0)
    return out


def full_counts(perm, census, sample, log):
    """Counts by municipality x nationality for the expanded shares: 2002 census of all
    foreign-born residents, and permits granted 2005-2008. Census names are matched to
    permit names; Chile is excluded."""
    p = perm[perm["year"] <= 2008].copy()
    p["cod_com"] = harmonize(p["cod_com"], sample)
    p["name"] = map_unique(p["cat_pais"], norm_name)
    p = p[p["name"] != "chile"]
    perm_names = set(p["name"].unique())

    c = census.copy()
    c["cod_com"] = harmonize(c["cod_com"], sample)
    raw = map_unique(c["cat_pais"], norm_name)
    mapping, report = {}, []
    for nm in sorted(raw.unique()):
        if nm.isdigit():
            mapping[nm] = f"census code {nm}"
        elif nm in perm_names:
            mapping[nm] = nm
        elif nm in CENSUS_ALIASES and any(a in perm_names for a in CENSUS_ALIASES[nm]):
            mapping[nm] = next(a for a in CENSUS_ALIASES[nm] if a in perm_names)
            report.append((nm, mapping[nm], "alias"))
        else:
            hits = [q for q in perm_names if nm in q or q in nm]
            if len(hits) == 1:
                mapping[nm] = hits[0]
                report.append((nm, hits[0], "containment"))
            else:
                mapping[nm] = nm
                report.append((nm, "(none)", "unmatched: census-only nationality"))
    c["name"] = raw.map(mapping)
    n_by = c.groupby("name").size()
    log("  census names matched to permit names other than exactly:")
    for a, b, how in report:
        log(f"    {a:32s} -> {b:28s} [{how}; {n_by.get(mapping[a], 0):,} census residents]")
    unnamed = c["name"].str.startswith("census code")
    log(f"  census residents with an unnamed country code: {unnamed.sum():,} of {len(c):,} "
        "(kept in denominators only)")

    flows = p.groupby(["cod_com", "name"]).size().unstack(fill_value=0)
    cens = c.groupby(["cod_com", "name"]).size().unstack(fill_value=0)
    return flows, cens


def tier_table(num, sample):
    S = num.reindex(sorted(sample), fill_value=0)
    t = pd.DataFrame({"count_sample": S.sum(), "n_munis_present": (S > 0).sum()})
    keep = [n for n in t.index if not n.startswith("census code") and not NONCOUNTRY.search(n)]
    t = t.loc[keep].sort_values("count_sample", ascending=False)
    for k, r in TIERS.items():
        t[k] = (t["count_sample"] >= r["min_count"]) & (t["n_munis_present"] >= r["min_munis"])
    return t


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep-root", default="rep_data/Replication")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--docs-dir", default="docs")
    ap.add_argument("--misc-dir", default="miscellaneous")
    args = ap.parse_args()
    log = Log()
    D = os.path.join(args.rep_root, "data")
    ana, inter, raw = (os.path.join(D, x) for x in ("analysis", "intermediate", "raw"))
    if not os.path.isdir(ana):
        sys.exit(f"{ana} not found; pass --rep-root")

    log("Dataset build: Ajzenman, Dominguez & Undurraga (2023) extension")
    log(f"Replication data: {D}")

    # ---- Load ------------------------------------------------------------------------
    log("\n=== Loading inputs")
    ivx = read(os.path.join(ana, "enusc_IV.dta"))
    sample = set(ivx["cod_com"].astype(int))
    log(f"  enusc_IV.dta: {len(ivx)} municipalities (the sample)")
    iv02 = read(os.path.join(ana, "enusc_IV_share2002.dta"),
                ["cod_com"] + [f"share2002_{c}" for c in PAPER11])
    ivpm = read(os.path.join(inter, "IV_panel.dta"))                 # municipality x year
    indiv_cols = ["cod_com", "year", "edad", "ismale", "trabaja", "lnrate_stock_imm",
                  "imm_instr"] + OUTCOMES
    log("  reading analysis/IV_panel.dta (individual level, ~220 MB)...")
    indiv = read(os.path.join(ana, "IV_panel.dta"), indiv_cols)
    log("  reading raw/immigration_Correciones.dta (~680 MB)...")
    perm = read(os.path.join(raw, "immigration_Correciones.dta"),
                ["cat_pais", "cod_com", "comuna", "year", "doc"])
    census13 = read(os.path.join(inter, "CensoInmigr2002.dta"))
    census = read(os.path.join(inter, "censo_etnias.dta"), ["cat_pais", "cod_com"])
    c06 = read(os.path.join(inter, "controlesCASEN06.dta"), ["cod_com"] + CASEN06_VARS)
    c03 = read(os.path.join(inter, "controlesCASEN03.dta"), ["cod_com"] + CASEN_OLD_VARS,
               required=False, log=log)
    c00 = read(os.path.join(inter, "controlesCASEN00.dta"), ["cod_com"] + CASEN_OLD_VARS,
               required=False, log=log)
    ct = read(os.path.join(inter, "controlesCASEN06_17.dta"),
              ["cod_com", "year", "hombre", "age", "ypc"], required=False, log=log)
    coll02 = read(os.path.join(inter, "collapse2002.dta"), required=False, log=log)
    coll02b = read(os.path.join(inter, "collapse2002_2.dta"), required=False, log=log)
    tab1 = read(os.path.join(ana, "TableI.dta"), ["cod_com", "imm_flow_rate2008",
                                                  "stock_imm08_rate"], required=False, log=log)
    pre = {yr: read(os.path.join(inter, f"pre{str(yr)[2:]}.dta"), required=False, log=log)
           for yr in (2003, 2005, 2006)}
    dist = read(os.path.join(inter, "distance_index.dta"), ["cod_com", "year", "mean_distance"],
                required=False, log=log)
    media = read(os.path.join(raw, "countTS.dta"), required=False, log=log)
    codes = read(os.path.join(raw, "cod_com.dta"))

    # ---- Municipality codes ----------------------------------------------------------
    log("\n=== Municipality codes")
    for old, new in RECODES.items():
        where = {nm: (old in set(df["cod_com"].astype(int)), new in set(df["cod_com"].astype(int)))
                 for nm, df in (("sample", ivx), ("permits", perm), ("census", census),
                                ("CASEN06", c06), ("IV_panel(muni)", ivpm))}
        flags = ", ".join(f"{nm}: {'old' if o else ''}{'+' if o and n else ''}{'new' if n else ''}"
                          for nm, (o, n) in where.items() if o or n)
        if flags:
            log(f"  {old}/{new}: {flags}")
    ind_set = set(indiv["cod_com"].astype(int))
    log(f"  individual panel municipalities: {len(ind_set)}; "
        f"not in sample: {sorted(ind_set - sample)}; sample not in panel: {sorted(sample - ind_set)}")

    # ---- Paper shares: rebuild and compare ------------------------------------------
    log("\n=== Check 1: rebuild the paper's 2008 shares from the raw data")
    perm = perm[~((perm["comuna"] == "florida") & (perm["cod_com"] == 8404))]
    perm = perm[perm["cod_com"] != 96]
    perm["cat_pais"] = perm["cat_pais"].astype(str)
    rebuilt = paper_shares(perm, census13)
    iv_idx = ivx.set_index(ivx["cod_com"].astype(int))
    worst = 0.0
    for c in PAPER11:
        for j in DOCS:
            v = f"share2008_{c}{j}"
            a = rebuilt[v].reindex(iv_idx.index).fillna(0).to_numpy()
            d = np.abs(a - iv_idx[v].to_numpy(float))
            worst = max(worst, d.max())
            if d.max() > 1e-5:
                bad = iv_idx.index[d > 1e-5].tolist()
                log(f"  {v}: max |diff| {d.max():.2e} in {len(bad)} municipalities {bad[:6]}")
    log(f"  overall max |rebuilt - paper| = {worst:.2e}  "
        f"({'exact' if worst < 1e-5 else 'NOT exact; the datasets use the paper values'})")

    # ---- Expanded shares ---------------------------------------------------------------
    log("\n=== Expanded shares (all nationalities)")
    flows, cens = full_counts(perm, census, sample, log)
    names = sorted(set(flows.columns) | set(cens.columns))
    flows, cens = flows.reindex(columns=names, fill_value=0), cens.reindex(columns=names, fill_value=0)
    munis = sorted(set(flows.index) | set(cens.index) | sample)
    flows, cens = flows.reindex(munis, fill_value=0), cens.reindex(munis, fill_value=0)
    num08 = cens + flows
    sh08 = num08.div(num08.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    sh02 = cens.div(cens.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    # census check: the 13 countries in censo_etnias should equal CensoInmigr2002
    c13 = census13.copy()
    c13["cod_com"] = harmonize(c13["cod_com"], sample)
    c13 = c13.groupby("cod_com").sum(numeric_only=True)
    diffs = [np.abs(cens.reindex(c13.index, fill_value=0).get(norm_name(PAPER_CATPAIS[c]),
                                                                 0) - c13[f"imm_{c}"]).max()
             for c in PAPER11]
    log(f"  Check 2: census counts of the 11 countries vs CensoInmigr2002: max |diff| = "
        f"{max(diffs):.0f} residents")

    tiers = tier_table(num08, sample)
    used = set()
    slug = {n: slugify(n, used) for n in tiers.index}
    broad = tiers.index[tiers["broad"]].tolist()
    primary = tiers.index[tiers["primary"]].tolist()
    log(f"  nationalities with 2008 immigrants in the sample: {(tiers['count_sample'] > 0).sum()}")
    log(f"  tier 'broad' (>= {TIERS['broad']['min_munis']} munis, >= {TIERS['broad']['min_count']}): "
        f"{len(broad)};  tier 'primary' (>= {TIERS['primary']['min_munis']} munis, "
        f">= {TIERS['primary']['min_count']}): {len(primary)}")
    log(f"  {'nationality':28s} {'slug':15s} {'count':>9s} {'munis':>6s} tier")
    for n, r in tiers[tiers["count_sample"] > 0].head(70).iterrows():
        t = "primary" if r["primary"] else ("broad" if r["broad"] else "")
        log(f"  {n[:28]:28s} {slug[n]:15s} {r['count_sample']:9,.0f} {r['n_munis_present']:6d} {t}")
    corr = []
    for c in PAPER11:
        nm = norm_name(PAPER_CATPAIS[c])
        if nm in sh08:
            a = sh08[nm].reindex(iv_idx.index).to_numpy()
            corr.append(np.corrcoef(a, iv_idx[f"share2008_{c}"].to_numpy(float))[0, 1])
    log(f"  Check 3: expanded vs paper 2008 shares for the 11 countries: correlation "
        f"min {min(corr):.3f}, mean {np.mean(corr):.3f} (differ by design: all-nationality "
        "census base)")

    # ---- Panel outcomes (collapse individual panel) -----------------------------------
    log("\n=== Collapsing the individual panel to municipality x year")
    indiv["cod_com"] = indiv["cod_com"].astype(int)
    g = indiv.groupby(["cod_com", "year"])
    pan = pd.DataFrame({f"y_{y}": 100 * g[y].mean() for y in OUTCOMES})
    pan["n_resp"] = g.size()
    for v in ("edad", "ismale", "trabaja"):
        pan[v] = g[v].mean()
    pan["chk_lnrate"] = g["lnrate_stock_imm"].agg(lambda s: s.max() - s.min())
    pan["chk_lnrate_val"] = g["lnrate_stock_imm"].first()
    pan = pan.reset_index()
    pan["year"] = pan["year"].astype(int)
    full = pd.MultiIndex.from_product([sorted(sample), YEARS], names=["cod_com", "year"])
    pan = pan.set_index(["cod_com", "year"]).reindex(full).reset_index()
    log(f"  {len(pan)} municipality-years; cells without respondents: {pan['n_resp'].isna().sum()}; "
        f"respondents per cell: min {pan['n_resp'].min():.0f}, median {pan['n_resp'].median():.0f}")

    # ---- Cross-section ---------------------------------------------------------------
    log("\n=== Cross-section")
    cs = pd.DataFrame({"cod_com": ivx["cod_com"].astype(int), "comuna": ivx["comuna"],
                       "region": ivx["region"].astype(int), "population": ivx["population"]})
    for y in OUTCOMES:
        cs[f"d_{y}"] = 100 * ivx[y].to_numpy(float)
    for j in DOCS:
        cs[f"deltaimm{j}"] = ivx[f"deltaimm{j}"].to_numpy(float)
        cs[f"deltalevel{j}"] = ivx[f"deltalevel{j}"].to_numpy(float)
    ivpm["cod_com"] = harmonize(ivpm["cod_com"], sample).astype(int)
    ivpm["year"] = ivpm["year"].astype(int)
    pm = ivpm.set_index(["cod_com", "year"])
    for j in DOCS:
        v = f"lnrate_stock_imm{j}"
        cs[f"d_{v}"] = (pm[v].xs(2017, level="year") - pm[v].xs(2008, level="year")
                        ).reindex(cs["cod_com"]).to_numpy()
        cs[f"deltaimm_instr{j}"] = ivx[f"deltaimm_instr{j}"].to_numpy(float)
    for c in PAPER11:
        cs[f"delta_ln_orig_{c}"] = ivx[f"delta_ln_orig_{c}"].to_numpy(float)
    for c in PAPER11:
        for j in DOCS:
            cs[f"sh08p_{c}{j}"] = ivx[f"share2008_{c}{j}"].to_numpy(float)
    i02 = iv02.set_index(iv02["cod_com"].astype(int))
    for c in PAPER11:
        cs[f"sh02p_{c}"] = i02[f"share2002_{c}"].reindex(cs["cod_com"]).to_numpy(float)
    for n in broad:
        cs[f"sh08_{slug[n]}"] = sh08[n].reindex(cs["cod_com"]).fillna(0).to_numpy()
    for n in broad:
        if n in sh02 and sh02[n].reindex(cs["cod_com"]).fillna(0).sum() > 0:
            cs[f"sh02_{slug[n]}"] = sh02[n].reindex(cs["cod_com"]).fillna(0).to_numpy()

    # controls: paper (CASEN 2006 household heads, as in IV_1.do)
    cs["age"] = ivx["age"].to_numpy(float)
    cs["hombre"] = ivx["hombre"].to_numpy(float)
    # pre-treatment pool
    def merge_muni(df, cols, prefix, src_year=None):
        if df is None:
            return
        d = df.copy()
        if src_year is not None and "year" in d:
            d = d[d["year"] == src_year]
        d["cod_com"] = harmonize(d["cod_com"], sample)
        d = d.dropna(subset=["cod_com"]).groupby("cod_com").first()
        for v in cols:
            if v in d:
                cs[f"{prefix}{v}"] = d[v].reindex(cs["cod_com"]).to_numpy(float)
    merge_muni(c06, CASEN06_VARS, "x_c06_")
    cs["x_c06_ln_ypc"] = np.log(cs["x_c06_ypc"])
    merge_muni(c03, CASEN_OLD_VARS, "x_c03_")
    merge_muni(c00, CASEN_OLD_VARS, "x_c00_")
    for yr in ("03", "00"):
        if f"x_c{yr}_age" in cs:
            cs[f"x_c{yr}_missing"] = cs[f"x_c{yr}_age"].isna().astype(int)
    if coll02 is not None:
        d = coll02.copy()
        d["cod_com"] = harmonize(d["cod_com"], sample)
        d = d.groupby("cod_com").sum(numeric_only=True).reindex(cs["cod_com"])
        imm = d["imm"].to_numpy(float)
        cs["x_imm02_rate"] = imm * 10000 / cs["population"].to_numpy(float)
        cs["x_imm02_lnrate"] = np.log(cs["x_imm02_rate"] + 1)
        cs["x_imm02_univ_sh"] = d["est_univers"].to_numpy(float) / imm
        cs["x_imm02_lowed_sh"] = (d["est_prebasico"] + d["est_basico"] + d["est_nada"]).to_numpy(float) / imm
    if coll02b is not None:
        d = coll02b.copy()
        d["cod_com"] = harmonize(d["cod_com"], sample)
        d = d.groupby("cod_com").sum(numeric_only=True).reindex(cs["cod_com"])
        imm = d["imm"].to_numpy(float)
        cs["x_imm02_young_sh"] = (d["mujer_joven"] + d["hombre_joven"]).to_numpy(float) / imm
        cs["x_imm02_female_sh"] = (d["mujer_joven"] + d["mujer_nojoven"]).to_numpy(float) / imm
    for j in DOCS:
        cs[f"x_lnrate_imm08{j}"] = pm[f"lnrate_stock_imm{j}"].xs(2008, level="year"
                                                                 ).reindex(cs["cod_com"]).to_numpy()
    merge_muni(tab1, ["imm_flow_rate2008"], "x_")
    p08 = pan[pan["year"] == 2008].set_index("cod_com").reindex(cs["cod_com"])
    for y in OUTCOMES:
        cs[f"x_y08_{y}"] = p08[f"y_{y}"].to_numpy()
    for v in ("ismale", "edad", "trabaja"):
        cs[f"x_resp08_{v}"] = p08[v].to_numpy()
    for yr, df in pre.items():
        if df is None:
            continue
        d = df.copy()
        d["cod_com"] = harmonize(d["cod_com"], sample)
        d = d.groupby("cod_com").first().reindex(cs["cod_com"])
        tag = str(yr)[2:]
        avail = [y for y in OUTCOMES if y in d]
        for y in avail:
            cs[f"x_pre{tag}_{y}"] = 100 * d[y].to_numpy(float)
        cs[f"x_pre{tag}_missing"] = d[avail[0]].isna().astype(int).to_numpy() if avail else 1
    merge_muni(dist, ["mean_distance"], "x_dist08_", src_year=2008)
    if media is not None:
        name_col = next((c for c in media.columns if not c.startswith("TS_")), None)
        if name_col:
            cmap = dict(zip(codes["comuna"].map(norm_name), codes["cod_com"].astype(int)))
            m = media.assign(cod_com=media[name_col].map(norm_name).map(cmap)).dropna(subset=["cod_com"])
            m["cod_com"] = harmonize(m["cod_com"], sample)
            m = m.groupby("cod_com")["TS_all"].sum().reindex(cs["cod_com"])
            cs["x_radio_pc"] = m.to_numpy(float) * 10000 / cs["population"].to_numpy(float)
            log(f"  media: matched {m.notna().sum()} of {len(cs)} municipalities by name")
    cs["x_ln_pop"] = np.log(cs["population"].astype(float))
    log(f"  {cs.shape[0]} rows x {cs.shape[1]} columns")

    # Check 4: replicate Table VI col. 2 (2SLS, del_problema3): 18.61 (6.79)
    y = cs["d_del_problema3"].to_numpy()
    W = np.column_stack([np.ones(len(cs)), cs["age"], cs["hombre"]])
    b, se = tsls(y, np.column_stack([cs["deltaimm"], W]), np.column_stack([cs["deltaimm_instr"], W]))
    log(f"  Check 4: 2SLS del_problema3 on deltaimm = {b[0]:.2f} ({se[0]:.2f}); paper 18.61 (6.79)")

    # ---- Panel -------------------------------------------------------------------------
    log("\n=== Panel")
    pv = ["lnrate_stock_imm", "rate_stock_imm", "stock_imm", "imm", "imm_instr"]
    keep = ["cod_com", "year"] + [f"{v}{j}" for v in pv for j in DOCS] + \
           [f"ln_orig_{c}" for c in PAPER11]
    keep = [k for k in keep if k in ivpm.columns]
    pan = pan.merge(ivpm[keep], on=["cod_com", "year"], how="left")
    pan = pan.merge(cs[["cod_com", "comuna", "region", "population"]], on="cod_com", how="left")
    d = np.abs(pan["chk_lnrate_val"] - pan["lnrate_stock_imm"])
    log(f"  Check 5: lnrate_stock_imm, individual panel vs municipality panel: max |diff| "
        f"{d.max():.2e}; within-cell variation {pan['chk_lnrate'].max():.2e}")
    pan = pan.drop(columns=["chk_lnrate", "chk_lnrate_val"])
    pan = pan.sort_values(["cod_com", "year"]).reset_index(drop=True)
    for j in DOCS:
        lag_stock = pan[f"stock_imm{j}"] - pan[f"imm{j}"]
        pan[f"lnrate_stock_imm{j}_lag"] = np.log(lag_stock * 10000 / pan["population"] + 1)
    prev = pan.groupby("cod_com")["lnrate_stock_imm"].shift(1)
    d = np.abs(pan["lnrate_stock_imm_lag"] - prev).dropna()
    log(f"  Check 6: lag built from stock - flow vs previous year's value: max |diff| {d.max():.2e}")

    # Check 7: panel 2017-2008 differences vs the cross-section outcomes
    w = pan.pivot(index="cod_com", columns="year")
    log("  Check 7: panel (2017 - 2008) vs cross-section outcome, max |diff| (x100 scale):")
    msgs = []
    for y in OUTCOMES:
        dd = (w[f"y_{y}"][2017] - w[f"y_{y}"][2008]).reindex(cs["cod_com"]).to_numpy()
        msgs.append(f"{y}={np.nanmax(np.abs(dd - cs[f'd_{y}'].to_numpy())):.2e}")
    for i in range(0, len(msgs), 4):
        log("    " + "  ".join(msgs[i:i + 4]))
    log("    (pc_19 and pc_20 are expected to differ: the paper normalizes the PCA scores over "
        "2008 and 2017 in the cross-section and over 2008-2017 in the panel)")

    # time-invariant shares and shares x year dummies
    inv = cs.set_index("cod_com")
    share_cols = [f"sh08p_{c}" for c in PAPER11] + [f"sh02p_{c}" for c in PAPER11] + \
                 [f"sh08_{slug[n]}" for n in broad]
    share_cols = [s for s in share_cols if s in inv]
    pan = pan.merge(inv[share_cols].reset_index(), on="cod_com", how="left")
    yy = pan["year"].to_numpy()
    block = {f"{s}_y{yr}": pan[s].to_numpy() * (yy == yr) for s in share_cols for yr in IV_YEARS}
    pan = pd.concat([pan, pd.DataFrame(block)], axis=1)
    # controls: time-varying CASEN (carried forward between waves, as in OLS_1.do)
    if ct is not None:
        t = ct.copy()
        t["cod_com"] = harmonize(t["cod_com"], sample)
        t["year"] = t["year"].astype(int)
        t = t.groupby(["cod_com", "year"]).first().reset_index()
        t = t.rename(columns={v: f"xt_casen_{v}" for v in ("hombre", "age", "ypc")})
        pan = pan.merge(t[["cod_com", "year", "xt_casen_hombre", "xt_casen_age", "xt_casen_ypc"]],
                        on=["cod_com", "year"], how="left")
    xcols = [c for c in cs.columns if c.startswith("x_")] + ["age", "hombre"]
    pan = pan.merge(cs[["cod_com"] + xcols].rename(columns={"age": "age_c06", "hombre": "hombre_c06"}),
                    on="cod_com", how="left")
    log(f"  {pan.shape[0]} rows x {pan.shape[1]} columns")

    # ---- Dictionary, groups, output ----------------------------------------------------
    dic = pd.DataFrame([describe(c, "cross_section", slug, tiers) for c in cs.columns] +
                       [describe(c, "panel", slug, tiers) for c in pan.columns])
    groups = make_groups(cs, pan, slug, broad, primary)
    for p in (args.data_dir, args.docs_dir, args.misc_dir):
        os.makedirs(p, exist_ok=True)
    for name, df in (("cross_section", cs), ("panel", pan)):
        write(df, os.path.join(args.data_dir, name), dic[dic["dataset"] == name], log)
    dic.to_csv(os.path.join(args.docs_dir, "data_dictionary.csv"), index=False)
    tiers.assign(slug=[slug[n] for n in tiers.index]).to_csv(
        os.path.join(args.docs_dir, "nationality_tiers.csv"), index_label="nationality")
    with open(os.path.join(args.docs_dir, "variable_groups.json"), "w") as f:
        json.dump(groups, f, indent=1)
    log(f"\nWrote {args.data_dir}/cross_section.*, {args.data_dir}/panel.*, "
        f"{args.docs_dir}/data_dictionary.csv, {args.docs_dir}/variable_groups.json, "
        f"{args.docs_dir}/nationality_tiers.csv")
    with open(os.path.join(args.misc_dir, "build_log.txt"), "w") as f:
        f.write("\n".join(log.lines) + "\n")


def write(df, stem, dic, log):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    try:
        df.to_parquet(stem + ".parquet", index=False)
    except Exception as exc:
        log(f"  note: parquet not written ({exc.__class__.__name__}); install pyarrow")
    labels = {r["variable"]: str(r["definition"])[:80] for _, r in dic.iterrows()}
    df.to_stata(stem + ".dta", write_index=False, version=118, variable_labels=labels)


def describe(c, ds, slug, tiers):
    inv_slug = {v: k for k, v in slug.items()}
    row = dict(variable=c, dataset=ds, group="", subgroup="", timing="", source="", definition="")
    def s(**k):
        row.update(k)
        return row
    if c in ("cod_com", "year", "comuna", "region"):
        return s(group="id", source="enusc_IV / IV_panel", definition=f"{c} (2010 municipality codes)")
    if c == "population":
        return s(group="id", timing="2010", source="raw/population.dta",
                 definition="municipality population (INE 2010 projection), used for rates")
    m = re.match(r"^(d|y)_(.+)$", c)
    if m and m.group(2) in OUTCOME_TABLE:
        y = m.group(2)
        what = "2017 minus 2008 change" if m.group(1) == "d" else "municipality-year mean"
        return s(group="outcome", subgroup=f"Table {OUTCOME_TABLE[y]}", timing="outcome",
                 source="enusc_IV.dta" if ds == "cross_section" else "analysis/IV_panel.dta (collapsed)",
                 definition=f"{y}: {what}, x100 (percentage points)")
    if c in ("n_resp",):
        return s(group="weight", source="analysis/IV_panel.dta",
                 definition="number of ENUSC respondents in the municipality-year")
    if re.match(r"^deltaimm(_permiso|_visa)?$", c):
        return s(group="endogenous", subgroup="paper", source="enusc_IV.dta",
                 definition="paper's Delta migr: 2017-2008 change in ln(1000*imm per 100k + 1); "
                            "13-country census base")
    if c.startswith("deltalevel"):
        return s(group="endogenous", subgroup="paper (levels)", source="enusc_IV.dta",
                 definition="2017-2008 change in immigrants per 100k (levels)")
    if c.startswith("d_lnrate_stock_imm"):
        return s(group="endogenous", subgroup="alternative", source="intermediate/IV_panel.dta",
                 definition="2017-2008 change in ln(stock per 10k + 1); all-foreign-born census base")
    if re.match(r"^(lnrate_stock_imm|rate_stock_imm|stock_imm|imm)(_permiso|_visa)?(_lag)?$", c):
        return s(group="endogenous" if c.startswith("lnrate") else "immigration (aux)",
                 subgroup="paper panel" + (" lag" if c.endswith("_lag") else ""),
                 source="intermediate/IV_panel.dta",
                 definition=f"{c}: immigrant stock/flow measures (census 2002 base + permits)")
    if re.match(r"^(deltaimm_instr|imm_instr)", c):
        return s(group="instrument", subgroup="bartik", source="enusc_IV / IV_panel",
                 definition="paper's shift-share instrument")
    if re.match(r"^(delta_ln_orig|ln_orig)_", c):
        return s(group="instrument (aux)", subgroup="shift", source="enusc_IV / IV_panel",
                 definition="UN log stock (change) of the nationality in other destinations")
    m = re.match(r"^(sh08p|sh02p|sh08|sh02)_([a-z0-9_]+?)(_permiso|_visa)?(_y(\d{4}))?$", c)
    if m:
        kind, key, doc, _, yr = m.groups()
        nm = PAPER_CATPAIS.get(key, inv_slug.get(key, key))
        tier = ""
        if kind in ("sh08", "sh02") and nm in tiers.index:
            tier = "primary" if tiers.loc[nm, "primary"] else "broad"
        base = {"sh08p": "paper 2008 share", "sh02p": "paper 2002 share",
                "sh08": "expanded 2008 share (census 2002 + permits 2005-08)",
                "sh02": "expanded 2002 share (census 2002)"}[kind]
        return s(group="instrument", subgroup=kind + (f" x year {yr}" if yr else "") +
                 (f" [{tier}]" if tier else ""), timing="baseline (2008)" if "08" in kind else "2002",
                 source="enusc_IV / enusc_IV_share2002" if kind.endswith("p") else "censo_etnias + permits",
                 definition=f"{base}: {nm}{doc or ''}" + (f" x 1[year={yr}]" if yr else ""))
    if c in ("age", "hombre"):
        return s(group="control", subgroup="paper", timing="pre (CASEN 2006)", source="enusc_IV.dta",
                 definition=f"paper control: {c} of household heads, CASEN 2006")
    if c in ("edad", "ismale", "trabaja"):
        return s(group="control", subgroup="paper panel" if c != "trabaja" else "contemporaneous",
                 timing="contemporaneous", source="analysis/IV_panel.dta",
                 definition=f"mean {c} of ENUSC respondents in the municipality-year")
    if c.startswith("xt_casen_"):
        return s(group="control", subgroup="time-varying", timing="post-treatment after 2008",
                 source="controlesCASEN06_17.dta",
                 definition=f"{c[9:]} of household heads, CASEN waves carried forward (OLS_1.do)")
    if c.startswith("x_") or c in ("age_c06", "hombre_c06"):
        timing = "pre-treatment"
        if c == "x_radio_pc":
            timing = "measured 2019 (post)"
        src = {"x_c06": "controlesCASEN06", "x_c03": "controlesCASEN03", "x_c00": "controlesCASEN00",
               "x_imm02": "collapse2002", "x_lnrat": "intermediate/IV_panel (2008)",
               "x_y08_": "analysis/IV_panel (2008)", "x_resp0": "analysis/IV_panel (2008)",
               "x_pre03": "pre03", "x_pre05": "pre05", "x_pre06": "pre06",
               "x_dist0": "distance_index (2008)", "x_radio": "countTS", "x_ln_po": "population",
               "x_imm_f": "TableI"}.get(c[:7], "")
        return s(group="control", subgroup="pre-treatment pool", timing=timing, source=src,
                 definition=control_definition(c))
    return s(group="other", definition=c)


def control_definition(c):
    m = re.match(r"^x_c(06|03|00)_(.+)$", c)
    if m:
        yr, v = m.groups()
        if v == "missing":
            return f"1 if the municipality is not in CASEN 20{yr}"
        if v == "ln_ypc":
            return f"log per-capita household income, CASEN 20{yr} household heads"
        return f"CASEN 20{yr} household heads: {v} (expansion-weighted municipal mean)"
    m = re.match(r"^x_pre(03|05|06)_(.+)$", c)
    if m:
        yr, v = m.groups()
        if v == "missing":
            return f"1 if the municipality is not in ENUSC 20{yr}"
        return f"ENUSC 20{yr} level of {v}, x100 (pre-period; partial coverage)"
    m = re.match(r"^x_y08_(.+)$", c)
    if m:
        extra = " (panel PCA normalization)" if m.group(1).startswith("pc_") else ""
        return f"2008 level of {m.group(1)}, x100{extra}"
    m = re.match(r"^x_resp08_(.+)$", c)
    if m:
        return f"2008 mean of {m.group(1)} among ENUSC respondents"
    m = re.match(r"^x_lnrate_imm08(.*)$", c)
    if m:
        return f"baseline immigration: ln(stock per 10k + 1) in 2008{m.group(1)}"
    return {
        "x_imm02_rate": "foreign-born residents per 10k population, census 2002",
        "x_imm02_lnrate": "ln(foreign-born residents per 10k + 1), census 2002",
        "x_imm02_univ_sh": "share of 2002 foreign-born residents with university education",
        "x_imm02_lowed_sh": "share of 2002 foreign-born residents with basic education or less",
        "x_imm02_young_sh": "share of 2002 foreign-born residents classified young (collapse2002_2)",
        "x_imm02_female_sh": "share of 2002 foreign-born residents who are women",
        "x_imm_flow_rate2008": "permits granted in 2008 per 10k population (TableI.dta)",
        "x_dist08_mean_distance": "mean genetic distance of immigrants to Chileans, 2008",
        "x_radio_pc": "local radio stations per 10k population (SUBTEL, 2019)",
        "x_ln_pop": "log population (INE 2010 projection)",
    }.get(c, c)


def make_groups(cs, pan, slug, broad, primary):
    def cols(df, pat):
        return [c for c in df.columns if re.match(pat, c)]
    p_slugs = [slug[n] for n in primary]
    b_slugs = [slug[n] for n in broad]
    g = {"cross_section": {
        "id": ["cod_com", "comuna", "region", "population"],
        "outcomes": [f"d_{y}" for y in OUTCOMES],
        "outcomes_by_table": {t: [f"d_{y}" for y in ys] for t, ys in TABLES.items()},
        "endogenous": {"paper": ["deltaimm", "deltaimm_permiso", "deltaimm_visa"],
                       "alternative_all_foreign_born_base": cols(cs, r"^d_lnrate_stock_imm")},
        "instruments": {
            "bartik": ["deltaimm_instr", "deltaimm_instr_permiso", "deltaimm_instr_visa"],
            "paper11_2008": [f"sh08p_{c}" for c in PAPER11],
            "paper11_2008_permiso": [f"sh08p_{c}_permiso" for c in PAPER11],
            "paper11_2008_visa": [f"sh08p_{c}_visa" for c in PAPER11],
            "paper11_2002": [f"sh02p_{c}" for c in PAPER11],
            "expanded_2008_primary": [f"sh08_{s}" for s in p_slugs],
            "expanded_2008_broad": [f"sh08_{s}" for s in b_slugs],
            "expanded_2002_primary": [f"sh02_{s}" for s in p_slugs if f"sh02_{s}" in cs],
            "expanded_2002_broad": [f"sh02_{s}" for s in b_slugs if f"sh02_{s}" in cs]},
        "controls": {"paper": ["age", "hombre"],
                     "pretreatment_pool": [c for c in cols(cs, r"^x_") if c != "x_radio_pc"],
                     "post_or_uncertain": ["x_radio_pc"] if "x_radio_pc" in cs else []}},
        "panel": {
        "id": ["cod_com", "year", "comuna", "region", "population"],
        "weight": ["n_resp"],
        "outcomes": [f"y_{y}" for y in OUTCOMES],
        "endogenous": {"paper": ["lnrate_stock_imm", "lnrate_stock_imm_permiso",
                                 "lnrate_stock_imm_visa"],
                       "lags": cols(pan, r"^lnrate_stock_imm.*_lag$")},
        "instruments": {
            "bartik": cols(pan, r"^imm_instr"),
            "paper11_2008_x_year": [c for c in cols(pan, r"^sh08p_.*_y\d{4}$")],
            "paper11_2002_x_year": [c for c in cols(pan, r"^sh02p_.*_y\d{4}$")],
            "expanded_2008_primary_x_year": [f"sh08_{s}_y{y}" for s in p_slugs for y in IV_YEARS],
            "expanded_2008_broad_x_year": [f"sh08_{s}_y{y}" for s in b_slugs for y in IV_YEARS],
            "time_invariant_shares": [c for c in cols(pan, r"^sh0[28]p?_") if not re.search(r"_y\d{4}$", c)]},
        "controls": {"paper": ["edad", "ismale"],
                     "time_varying_post": cols(pan, r"^xt_casen_"),
                     "time_invariant_pretreatment": [c for c in cols(pan, r"^x_") if c != "x_radio_pc"]
                     + ["age_c06", "hombre_c06"]}}}
    g["cross_section"]["contains_missing"] = [c for c in cs.columns if cs[c].isna().any()]
    g["panel"]["contains_missing"] = [c for c in pan.columns if pan[c].isna().any()]
    return g


if __name__ == "__main__":
    main()
