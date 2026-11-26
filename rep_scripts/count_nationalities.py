"""How many nationalities are in the residence-permit data, and how many could serve
as share instruments? Companion to rep_scripts/dantzig_iv.py (Ajzenman, Dominguez &
Undurraga 2023, "Immigration, Crime, and Crime (Mis)Perceptions").

A shares-only design does not need the UN shift data, so it is no longer limited to the
paper's 11 countries. This script

  1. finds the permit microdata: files with nationality, municipality and year (or grant
     date) columns. Survey files (ENUSC, CASEN) are skipped, nationality columns must
     hold names or integer codes (not continuous values), and column detection uses
     variable names only (not Stata labels);
  2. counts distinct nationalities, before and after cleaning the names (accents, case,
     spacing, non-country codes such as "SIN INFORMACION", Chile itself);
  3. rebuilds the paper's 2008 shares for its 11 countries and compares them with the
     share2008_* variables in enusc_IV.dta, under two definitions of the 2008 "stock":
        flow  = permits granted in 2008
        stock = permits granted from the first year in the data through 2008
     and two denominators (all nationalities, or the 11 countries only). The paper
     reports that its 11 countries are 87% of 2008 permits and 94% of 2017 permits,
     which is a second check;
  4. using the best-matching definition, restricted to the ENUSC municipalities, reports
     how many nationalities pass thresholds on the number of municipalities they appear
     in and on their total count. Those are the usable candidate instruments.

Usage (from the project root; data_dir is searched recursively):
    python -I rep_scripts/count_nationalities.py rep_data/Replication/data rep_results \\
        --files raw/immigration_Correciones.dta --nat-col pais_origen --year-col year
    # only list candidate files and detected columns:
    python -I rep_scripts/count_nationalities.py rep_data/Replication/data rep_results --scan-only
    # also write the municipality x nationality share matrix for the candidates:
    python -I rep_scripts/count_nationalities.py ... --write-shares

Outputs: rep_results/nationality_inventory.txt, nationality_counts.csv and, with
--write-shares, shares2008_all_nationalities.csv.

Requires: numpy, pandas (plus openpyxl for .xlsx and xlrd for .xls inputs).
"""
import argparse, csv, glob, os, re, sys, unicodedata, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

IV_FILE = "enusc_IV.dta"
DATA_EXTS = (".dta", ".csv", ".tsv", ".txt", ".xlsx", ".xls")
SKIP_PREFIXES = ("enusc", "casen")      # survey files, never permit records
YEAR_RANGE = (1990, 2030)               # plausible permit years

# The paper's 11 countries (codes as in enusc_IV.dta) and the spellings that may appear
# in the permit data, after normalization (upper case, no accents or punctuation).
PAPER = {
    "arg": ["ARGENTINA", "ARGENTINO"],
    "bol": ["BOLIVIA", "BOLIVIANA", "BOLIVIANO", "BOLIVIA ESTADO PLURINACIONAL DE"],
    "bra": ["BRASIL", "BRAZIL", "BRASILENA", "BRASILENO"],
    "chi": ["CHINA", "CHINO", "CHINA REPUBLICA POPULAR", "REPUBLICA POPULAR CHINA"],
    "col": ["COLOMBIA", "COLOMBIANA", "COLOMBIANO"],
    "ecu": ["ECUADOR", "ECUATORIANA", "ECUATORIANO"],
    "hai": ["HAITI", "HAITIANA", "HAITIANO"],
    "per": ["PERU", "PERUANA", "PERUANO"],
    "esp": ["ESPANA", "SPAIN", "ESPANOLA", "ESPANOL"],
    "eeuu": ["ESTADOS UNIDOS", "EEUU", "EE UU", "USA", "ESTADOS UNIDOS DE AMERICA",
             "NORTEAMERICANA", "NORTEAMERICANO", "ESTADOUNIDENSE", "UNITED STATES"],
    "ven": ["VENEZUELA", "VENEZOLANA", "VENEZOLANO", "VENEZUELA REPUBLICA BOLIVARIANA DE"],
}
ALIAS_TO_CODE = {a: c for c, al in PAPER.items() for a in al}
NONCOUNTRY = re.compile(
    r"^$|^(SIN|NO) (INFO|INFORMA|ESPECIF|DECLARA|REGISTRA)|DESCONOC|APATRID|INDETERM|"
    r"^OTR(O|A|OS|AS)( |$)|^S ?I$|^NAN$|^NONE$|^N ?D$|^NS ?NR$|^IGNORAD", re.I)


def is_chile(key):
    """CHILE, CHILENA, CHILENO, "CHILENA EXCLUSIVAMENTE", etc."""
    return key == "CHILE" or key.startswith("CHILEN") or key == "CHILEAN"


# Column detection, on variable names only. Dates of birth and expiry dates are excluded.
PAT = {
    "nat":   re.compile(r"nacionalid|nationalit|^pais|country|^nac$|^nacion", re.I),
    "muni":  re.compile(r"comuna|municip|^cut|cod_?com", re.I),
    "year":  re.compile(r"^(a(n|ñ|gn)o|anio|year|yr)(_.*)?$|_(a(n|ñ|gn)o|anio|year)$", re.I),
    "date":  re.compile(r"fecha|date|^fec_", re.I),
    "count": re.compile(r"^n$|^count|cantidad|^total|frecuen|^num_?(permisos|visas)?$", re.I),
}
EXCL = {
    "nat":  re.compile(r"nacim|birth|fecha|date|share|delta|tend|dist|_id$|^id_", re.I),
    "muni": re.compile(r"region|prov|share|delta|tend|glosa|nombre", re.I),
    "year": re.compile(r"nacim|birth|llega|arriv", re.I),
    "date": re.compile(r"nacim|birth|venc|expir|caduc|nac$", re.I),
}
PREFERRED_DATE = re.compile(r"resol|otorg|conces|emis|solicit|grant|issue", re.I)


# --------------------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------------------
def norm(v):
    """Upper case, no accents, no punctuation, single spaces. Integer-valued numbers are
    written without decimals, so 604.0 and "604" give the same key."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (int, np.integer)) or (isinstance(v, (float, np.floating))
                                            and float(v).is_integer()):
        return str(int(v))
    s = unicodedata.normalize("NFKD", str(v).strip()).encode("ascii", "ignore").decode()
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9 ]+", " ", s)).strip().upper()


def map_unique(s, f):
    """Apply f to the unique values of s only (fast on millions of rows)."""
    u = pd.unique(s)
    return s.map(dict(zip(u, [f(x) for x in u])))


def muni_key(v):
    """Municipality codes become integer strings ("01101", 1101.0 -> "1101");
    names are normalized like nationalities."""
    k = norm(v)
    return str(int(k)) if k.isdigit() else (k or None)


# --------------------------------------------------------------------------------------
# Reading files
# --------------------------------------------------------------------------------------
def sniff(path):
    with open(path, "rb") as f:
        raw = f.read(50000)
    enc = "utf-8-sig"
    try:
        sample = raw.decode(enc)
    except UnicodeDecodeError:
        enc, sample = "latin-1", raw.decode("latin-1")
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = "\t" if path.lower().endswith(".tsv") else ","
    return enc, sep


def header(path, header_row):
    """Column names, without loading the data."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dta":
        with pd.io.stata.StataReader(path) as r:
            return list(r.variable_labels())
    if ext in (".csv", ".tsv", ".txt"):
        enc, sep = sniff(path)
        return [str(c) for c in pd.read_csv(path, nrows=5, sep=sep, encoding=enc,
                                            header=header_row).columns]
    return [str(c) for c in pd.read_excel(path, nrows=5, header=header_row).columns]


def detect(cols):
    def find(key):
        return [c for c in cols if PAT[key].search(str(c))
                and not (key in EXCL and EXCL[key].search(str(c)))]
    found = {k: find(k) for k in PAT}
    found["date"].sort(key=lambda c: not PREFERRED_DATE.search(c))   # grant dates first
    return found


def read_columns(path, use, header_row, nrows=None):
    """Read only the needed columns of one file (all sheets of an Excel workbook)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dta":
        def rd(cat):
            if nrows is None:
                return pd.read_stata(path, columns=use, convert_categoricals=cat)
            with pd.read_stata(path, iterator=True, convert_categoricals=cat) as r:
                return r.read(nrows, columns=use)
        try:
            return rd(True)
        except ValueError as exc:   # non-unique value labels: keep the numeric codes
            if "not found" in str(exc):
                raise
            if nrows is None:
                print(f"  note: value labels in {path} could not be applied; numeric codes kept")
            return rd(False)
    if ext in (".csv", ".tsv", ".txt"):
        enc, sep = sniff(path)
        for e in (enc, "latin-1"):
            try:
                return pd.read_csv(path, usecols=use, sep=sep, encoding=e, dtype=str,
                                   header=header_row, low_memory=False, nrows=nrows)
            except UnicodeDecodeError:
                continue
    sheets = pd.read_excel(path, sheet_name=None, header=header_row, nrows=nrows)
    frames = []
    for name, df in sheets.items():
        df.columns = [str(c) for c in df.columns]
        if all(c in df.columns for c in use):
            frames.append(df[use])
        elif nrows is None:
            print(f"  note: sheet '{name}' of {path} lacks {set(use) - set(df.columns)}; skipped")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=use)


def nat_column_ok(values):
    """A nationality column must hold names or integer codes. Continuous values (e.g. an
    outcome like del_tend_pais) are rejected. Returns (ok, reason)."""
    v = values.dropna()
    if len(v) == 0:
        return False, "no non-missing values"
    num = pd.to_numeric(v.astype(str) if v.dtype == object else v, errors="coerce")
    frac_num = num.notna().mean()
    if frac_num > 0.9:
        if (num.dropna() % 1 != 0).any():
            return False, "continuous numeric values, not nationalities"
        return True, "integer codes (names may be in another column)"
    return True, ""


def to_year(s, is_date):
    if not is_date:
        return pd.to_numeric(s, errors="coerce")
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.year
    def parse(x):
        t = pd.to_datetime(x, errors="coerce", dayfirst=True)
        return np.nan if pd.isna(t) else t.year
    return map_unique(s.astype(str), parse)


# --------------------------------------------------------------------------------------
# Shares
# --------------------------------------------------------------------------------------
def share_matrix(df, years, munis=None, denom_codes=None):
    """Counts and shares by municipality (rows) and nationality (columns) over `years`.
    Chile is excluded. Non-country entries stay in the denominator (they are immigrants)
    unless denom_codes restricts the denominator to those nationalities."""
    sub = df[df["year"].isin(years) & ~df["is_chile"]]
    if munis is not None:
        sub = sub[sub["muni"].isin(munis)]
    counts = sub.groupby(["muni", "nat"])["w"].sum().unstack(fill_value=0.0)
    if munis is not None:
        counts = counts.reindex(sorted(munis), fill_value=0.0)
    den = counts[[c for c in counts.columns if c in denom_codes]].sum(axis=1) \
        if denom_codes is not None else counts.sum(axis=1)
    return counts, counts.div(den.replace(0, np.nan), axis=0)


def find_iv(data_dir, iv_path):
    if iv_path:
        return iv_path
    hits = glob.glob(os.path.join(data_dir, "**", IV_FILE), recursive=True)
    return hits[0] if hits else None


def iv_id_keys(iv, forced=None):
    """Candidate municipality-id columns of the IV file, as sets of normalized keys."""
    cands = [forced] if forced else [c for c in iv.columns if PAT["muni"].search(c)
                                     or re.search(r"(^|_)id($|_)|codigo", c, re.I)]
    return {c: iv[c].map(muni_key) for c in cands if c in iv.columns}


def best_overlap(keys, iv_keys):
    """Largest number of IV municipalities found among `keys`, and which IV column."""
    best = (None, 0)
    for c, k in iv_keys.items():
        m = int(k.isin(keys).sum())
        if m > best[1]:
            best = (c, m)
    return best


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--files", nargs="+", help="permit files (relative to data_dir or absolute)")
    ap.add_argument("--nat-col"); ap.add_argument("--muni-col")
    ap.add_argument("--year-col"); ap.add_argument("--date-col")
    ap.add_argument("--count-col", help="only if rows are aggregated counts, not permits")
    ap.add_argument("--header-row", type=int, default=0, help="header row for csv/xlsx")
    ap.add_argument("--iv-file", help=f"path to {IV_FILE} (default: search data_dir)")
    ap.add_argument("--iv-muni-col", help="municipality id column in the IV file")
    ap.add_argument("--base-year", type=int, default=2008)
    ap.add_argument("--stock-start", type=int,
                    help="first year of the cumulative 'stock' definition (default: first "
                         "plausible year in the data)")
    ap.add_argument("--definition", choices=["auto", "flow", "stock"], default="auto")
    ap.add_argument("--aliases", help="csv with columns raw,canonical to merge spellings")
    ap.add_argument("--min-munis", type=int, default=10,
                    help="candidate needs a positive share in at least this many municipalities")
    ap.add_argument("--min-count", type=float, default=50,
                    help="candidate needs at least this many base-year immigrants in sample")
    ap.add_argument("--write-shares", action="store_true")
    ap.add_argument("--scan-only", action="store_true")
    args = ap.parse_args()

    lines = []
    def out(s=""):
        print(s)
        lines.append(str(s))

    # The IV file is read first: its municipality ids are used to choose among several
    # municipality columns in a permit file.
    iv_path = find_iv(args.data_dir, args.iv_file)
    iv = pd.read_stata(iv_path, convert_categoricals=False) if iv_path else None
    iv_keys = iv_id_keys(iv, args.iv_muni_col) if iv is not None else {}

    # ---- 1. Find candidate permit files ------------------------------------------------
    if args.files:
        paths = [p if os.path.isabs(p) else os.path.join(args.data_dir, p) for p in args.files]
    else:
        paths = sorted(p for p in glob.glob(os.path.join(args.data_dir, "**", "*"),
                                            recursive=True)
                       if p.lower().endswith(DATA_EXTS))
    out("Nationality inventory of the residence-permit data")
    out(f"Searching: {args.data_dir}\n")
    out("File scan (nat = nationality, muni = municipality, year/date = time column):")
    plan, n_skipped = [], 0
    for p in paths:
        rel = os.path.relpath(p, args.data_dir)
        base = os.path.basename(p).lower()
        if not args.files and base.startswith(SKIP_PREFIXES):
            out(f"  skip  {rel}  (survey file)")
            continue
        try:
            cols = header(p, args.header_row)
        except Exception as exc:
            out(f"  skip  {rel}  (could not read: {exc})")
            continue
        f = detect(cols)
        notes = []

        # nationality: first candidate whose values are names or integer codes
        nat_cands = [args.nat_col] if args.nat_col else f["nat"]
        nat = None
        for c in nat_cands:
            if c not in cols:
                notes.append(f"column {c} not found")
                continue
            ok, why = nat_column_ok(read_columns(p, [c], args.header_row, nrows=5000)[c])
            if ok:
                nat = c
                if why:
                    notes.append(f"nat {c}: {why}")
                break
            notes.append(f"nat candidate {c} rejected: {why}")

        year = args.year_col or (None if args.date_col else (f["year"] or [None])[0])
        date = None if year else (args.date_col or (f["date"] or [None])[0])
        muni_cands = [args.muni_col] if args.muni_col else f["muni"]
        muni_cands = [c for c in muni_cands if c in cols]
        for k, c in (("year", year), ("date", date), ("count", args.count_col),
                     ("muni", args.muni_col)):
            if c and c not in cols:
                notes.append(f"{k} column {c} not found in this file")
                if k == "year":
                    year = None
                elif k == "date":
                    date = None
        ok = bool(nat and muni_cands and (year or date)) and not (
            args.count_col and args.count_col not in cols)
        if not ok:
            # not a permit file: print it only if some permit-like column was found
            if f["nat"] or args.files:
                out(f"        {rel}")
                for k in ("nat", "muni", "year", "date"):
                    if f[k]:
                        out(f"          {k:5s}: {f[k][:6]}")
                for nt in notes:
                    out(f"          {nt}")
            else:
                n_skipped += 1
            continue

        # municipality: with several candidates, the one that best matches the IV file
        muni, why = muni_cands[0], ""
        if len(muni_cands) > 1 and iv_keys:
            scores = {}
            for c in muni_cands:
                keys = set(map_unique(read_columns(p, [c], args.header_row)[c], muni_key))
                scores[c] = best_overlap(keys, iv_keys)[1]
            muni = max(scores, key=scores.get)
            why = "   (IV matches: " + ", ".join(f"{c}={v}" for c, v in scores.items()) + ")"
        out(f"  USE   {rel}")
        out(f"          nat  : {nat}")
        out(f"          muni : {muni}{why}")
        out(f"          {'year' if year else 'date'} : {year or date}")
        for nt in notes:
            out(f"          {nt}")
        if f["count"] and not args.count_col:
            out(f"          note: {f['count'][:6]} may be a count column. If rows are aggregated "
                "counts rather than permits, rerun with --count-col.")
        plan.append((p, rel, dict(nat=nat, muni=muni, year=year, date=date,
                                  count=args.count_col)))
    if n_skipped:
        out(f"  ({n_skipped} other files have no nationality column and are not listed)")

    if not plan:
        out("\nNo file with nationality, municipality and year/date columns was found.")
        out("Rerun with --files and the column options (--nat-col, --muni-col, --year-col).")
        return finish(args, lines)
    if len(plan) > 1:
        out(f"\nUsing {len(plan)} files pooled. If any of them are not permit records, or if "
            "the same person appears in both a visa file and a permanent-residence file, "
            "restrict the set with --files.")
    if args.scan_only:
        return finish(args, lines)

    # ---- 2. Load and clean -------------------------------------------------------------
    frames = []
    for p, rel, pick in plan:
        use = [c for c in (pick["nat"], pick["muni"], pick["year"] or pick["date"], pick["count"]) if c]
        df = read_columns(p, list(dict.fromkeys(use)), args.header_row)
        std = pd.DataFrame({
            "nat_raw": df[pick["nat"]],
            "muni": map_unique(df[pick["muni"]], muni_key),
            "year": to_year(df[pick["year"] or pick["date"]], is_date=not pick["year"]),
            "w": pd.to_numeric(df[pick["count"]], errors="coerce").fillna(0) if pick["count"] else 1.0,
            "file": rel,
        })
        bad_year = std["year"].notna() & ~std["year"].between(*YEAR_RANGE)
        out(f"\nLoaded {rel}: {len(std):,} rows; years {std['year'].min():.0f}-{std['year'].max():.0f}; "
            f"{std['nat_raw'].nunique():,} distinct nationality values; "
            f"{std['year'].isna().sum():,} rows without a year; "
            f"{std['muni'].isna().sum():,} without a municipality")
        if bad_year.any():
            out(f"  WARNING: {bad_year.sum():,} rows have years outside {YEAR_RANGE[0]}-"
                f"{YEAR_RANGE[1]} and are dropped. Check that '{pick['year'] or pick['date']}' "
                "is the permit year (not, e.g., year of birth or arrival).")
            std = std[~bad_year]
        frames.append(std)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["year"].notna() & df["muni"].notna()]
    df["year"] = df["year"].astype(int)

    df["nat"] = map_unique(df["nat_raw"], norm)
    if args.aliases:
        al = pd.read_csv(args.aliases)
        amap = dict(zip(al["raw"].map(norm), al["canonical"].map(norm)))
        df["nat"] = df["nat"].replace(amap)
    df["code"] = df["nat"].map(ALIAS_TO_CODE)
    # merge spelling variants of the paper's 11 countries into one key each
    canon = {c: PAPER[c][0] for c in PAPER}
    df.loc[df["code"].notna(), "nat"] = df.loc[df["code"].notna(), "code"].map(canon)
    df["is_chile"] = map_unique(df["nat"], is_chile).astype(bool)
    df["noncountry"] = map_unique(df["nat"], lambda k: bool(NONCOUNTRY.search(k))).astype(bool)

    n_raw = df["nat_raw"].astype(str).str.strip().nunique()
    keys = df.loc[~df["is_chile"] & ~df["noncountry"], "nat"].unique()
    out("\n=== Distinct nationalities (all years, all municipalities) ===")
    out(f"  distinct raw values:                       {n_raw}")
    out(f"  after normalization and merging aliases:   {df['nat'].nunique()}")
    out(f"  of which non-country codes:                {df.loc[df['noncountry'], 'nat'].nunique()}"
        f"  ({df.loc[df['noncountry'], 'w'].sum():,.0f} rows/permits)")
    out(f"  Chile listed as nationality:               {df.loc[df['is_chile'], 'w'].sum():,.0f} rows/permits (excluded)")
    out(f"  countries:                                 {len(keys)}")
    if df.loc[df["is_chile"], "w"].sum() / df["w"].sum() > 0.5:
        out("  WARNING: most rows are Chilean. This looks like survey data, not permit records.")
    noncountry_vals = df.loc[df["noncountry"], "nat"].value_counts()
    if len(noncountry_vals):
        out(f"  non-country codes found: {', '.join(noncountry_vals.index[:10])}")
    named = pd.Series([k for k in keys if re.search(r"[A-Z]", k)])
    if len(named) < 0.5 * len(keys):
        out("  NOTE: most nationality values are numeric codes, so the paper's 11 countries "
            "cannot be recognized by name. Use the column with country names, or map the "
            "codes with --aliases.")
    if len(named):
        dup = [v for v in named.groupby(named.str[:5]).apply(list) if len(v) > 1]
        if dup:
            out("  Possible duplicate spellings (same first 5 letters; review, then merge "
                "with --aliases):")
            for g in dup[:25]:
                out(f"    {g}")
    found_codes = sorted(df["code"].dropna().unique())
    miss = [c for c in PAPER if c not in found_codes]
    if miss:
        out(f"  WARNING: none of the spellings for {miss} were found; check the names in "
            "nationality_counts.csv and add them with --aliases.")

    # coverage of the paper's 11 countries (paper: 87% in 2008, 94% in 2017)
    out("\n  Share of permits from the paper's 11 countries (all municipalities):")
    for yr, paper_val in ((args.base_year, 87), (2017, 94)):
        sub = df[(df["year"] == yr) & ~df["is_chile"]]
        if len(sub):
            out(f"    {yr}: {100 * sub.loc[sub['code'].notna(), 'w'].sum() / sub['w'].sum():5.1f}%"
                + (f"   (paper: {paper_val}%)" if yr in (2008, 2017) else ""))

    # ---- 3. Validate against the paper's shares ----------------------------------------
    first = args.stock_start or int(df["year"].min())
    defs = {"flow": [args.base_year], "stock": list(range(first, args.base_year + 1))}
    sample, best_def = None, None
    if iv is None:
        out(f"\n{IV_FILE} not found: no validation; statistics use all municipalities.")
    else:
        col, nmatch = best_overlap(set(df["muni"].unique()), iv_keys)
        if col is None or nmatch < 0.5 * len(iv):
            out(f"\nCould not match the municipalities of {IV_FILE} to the permit data "
                f"(best: {col}, {nmatch} of {len(iv)}). Use --muni-col / --iv-muni-col, or a "
                "crosswalk if one file has codes and the other names. Statistics use all "
                "municipalities.")
        else:
            ivkeys = iv_keys[col]
            out(f"\n=== Validation against {os.path.basename(iv_path)} ===")
            out(f"  municipality id: {col}; matched {nmatch} of {len(iv)} municipalities")
            if nmatch < len(iv):
                out(f"  unmatched ids: {sorted(iv.loc[~ivkeys.isin(df['muni']), col].tolist())[:20]}"
                    "  (codes changed for some municipalities, e.g. the 2018 creation of the "
                    "Nuble region; a crosswalk may be needed)")
            sample = set(ivkeys[ivkeys.isin(df["muni"])])
            paper_cols = {c: f"share2008_{c}" for c in PAPER if f"share2008_{c}" in iv.columns}
            if args.base_year != 2008 or not paper_cols:
                out("  share2008_* variables unavailable (or base year is not 2008): no validation.")
            else:
                ivs = iv.assign(_k=ivkeys).set_index("_k").loc[sorted(sample)]
                tot = ivs[list(paper_cols.values())].sum(axis=1)
                out(f"  sum of the 11 paper shares per municipality: min {tot.min():.3f}, "
                    f"median {tot.median():.3f}, max {tot.max():.3f}")
                out(f"  {'definition':34s} {'mean corr':>10s} {'min corr':>9s} {'max |diff|':>11s}")
                results = {}
                for dname, yrs in defs.items():
                    for den in ("all", "11 only"):
                        _, S = share_matrix(df, yrs, sample,
                                            [PAPER[c][0] for c in PAPER] if den == "11 only" else None)
                        corrs, diffs = [], []
                        for c, v in paper_cols.items():
                            mine = S[PAPER[c][0]] if PAPER[c][0] in S else pd.Series(0.0, S.index)
                            a, b = mine.fillna(0).to_numpy(), ivs[v].fillna(0).to_numpy()
                            scale = 100.0 if b.max() > 1.5 else 1.0
                            corrs.append(np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan)
                            diffs.append(np.abs(a * scale - b).max())
                        results[(dname, den)] = (np.nanmean(corrs), np.nanmin(corrs), max(diffs))
                        r = results[(dname, den)]
                        label = f"{dname} {yrs[0]}-{yrs[-1]}, denom {den}"
                        out(f"  {label:34s} {r[0]:10.4f} {r[1]:9.4f} {r[2]:11.2e}")
                best_def = max(results, key=lambda k: (results[k][0], -results[k][2]))
                out(f"  best match: {best_def[0]}, denominator {best_def[1]}")
                if results[best_def][0] < 0.99:
                    out("  WARNING: no definition reproduces the paper's shares closely. They may "
                        "be built from a different file (e.g. visas only), different years, or "
                        "different municipality assignments. Interpret the counts below with care.")

    # ---- 4. Candidate instruments ------------------------------------------------------
    dname = args.definition if args.definition != "auto" else (best_def[0] if best_def else "flow")
    yrs = defs[dname]
    counts_s, S = share_matrix(df, yrs, sample)
    counts_all, _ = share_matrix(df, yrs, None)
    nmun = len(counts_s)
    where = f"the {nmun} ENUSC municipalities" if sample is not None else f"all {nmun} municipalities"
    empty = int((counts_s.sum(axis=1) == 0).sum())
    out(f"\n=== Base-year composition: definition '{dname}' (years {yrs[0]}-{yrs[-1]}), {where} ===")
    if empty:
        out(f"  WARNING: {empty} municipalities have no immigrants in the base year (shares undefined).")

    cand_cols = [c for c in counts_s.columns if not is_chile(c) and not NONCOUNTRY.search(c)]
    stats = pd.DataFrame({
        "nationality": cand_cols,
        "paper_code": [next((k for k in PAPER if PAPER[k][0] == c), "") for c in cand_cols],
        "count_all_munis": [counts_all[c].sum() if c in counts_all else 0 for c in cand_cols],
        "count_sample": [counts_s[c].sum() for c in cand_cols],
        "n_munis_present": [(counts_s[c] > 0).sum() for c in cand_cols],
        "mean_share": [S[c].mean() for c in cand_cols],
        "sd_share": [S[c].std() for c in cand_cols],
        "top_muni_frac": [counts_s[c].max() / counts_s[c].sum() if counts_s[c].sum() else np.nan
                          for c in cand_cols],
    })
    # nationalities present in other years but absent from the base year
    absent = sorted(set(keys) - set(cand_cols))
    stats = pd.concat([stats, pd.DataFrame({
        "nationality": absent, "paper_code": "", "count_all_munis": 0, "count_sample": 0,
        "n_munis_present": 0, "mean_share": 0.0, "sd_share": 0.0, "top_muni_frac": np.nan})],
        ignore_index=True).sort_values("count_sample", ascending=False)
    stats["candidate"] = (stats["n_munis_present"] >= args.min_munis) & \
                         (stats["count_sample"] >= args.min_count)

    pos = stats[stats["count_sample"] > 0]
    out(f"  countries with any base-year immigrants: {(stats['count_all_munis'] > 0).sum()} nationally, "
        f"{len(pos)} in {where}")
    out("\n  Nationalities with a positive share in at least k municipalities:")
    ks = sorted({1, 2, 5, 10, 25, 50, nmun})
    out("    " + "".join(f"{'k=' + str(k):>8s}" for k in ks))
    out("    " + "".join(f"{(stats['n_munis_present'] >= k).sum():8d}" for k in ks))
    out("\n  Nationalities with at least c base-year immigrants in the sample:")
    cs = [1, 10, 50, 100, 500, 1000]
    out("    " + "".join(f"{'c=' + str(c):>8s}" for c in cs))
    out("    " + "".join(f"{(stats['count_sample'] >= c).sum():8d}" for c in cs))
    total = pos["count_sample"].sum()
    cum = pos["count_sample"].cumsum() / total
    out("\n  Nationalities needed to cover the base-year immigrant population of the sample:")
    for q in (0.90, 0.95, 0.99):
        out(f"    {int(q * 100)}%: {int((cum < q).sum()) + 1}")
    ncand = int(stats["candidate"].sum())
    out(f"\n  Candidate instruments (>= {args.min_munis} municipalities and >= {args.min_count:g} "
        f"immigrants): {ncand}   (paper: 11; N = {nmun})")
    out(f"    of which among the paper's 11: {int(stats.loc[stats['candidate'], 'paper_code'].ne('').sum())}")

    out("\n  Top 30 nationalities by base-year count in the sample:")
    out(f"    {'nationality':28s} {'paper':>5s} {'count':>9s} {'munis':>6s} {'mean sh':>8s} "
        f"{'sd sh':>7s} {'top muni':>9s} {'cand':>5s}")
    for _, r in stats.head(30).iterrows():
        out(f"    {r['nationality'][:28]:28s} {r['paper_code']:>5s} {r['count_sample']:9,.0f} "
            f"{r['n_munis_present']:6d} {r['mean_share']:8.4f} {r['sd_share']:7.4f} "
            f"{r['top_muni_frac']:9.2f} {'yes' if r['candidate'] else '':>5s}")

    out("\nNotes:")
    out("  - 'top muni' is the fraction of a nationality's sample immigrants living in its "
        "largest municipality. Values near 1 mean the share is close to a dummy for one "
        "municipality: a high-leverage instrument identified by a single observation.")
    out("  - The thresholds (--min-munis, --min-count) are placeholders. Pre-register them.")
    out("  - Shares over all nationalities sum to one in each municipality. If every "
        "nationality enters, one is collinear with the intercept (dantzig_iv.py drops it "
        "automatically in the all-shares 2SLS).")
    out("  - Every candidate needs the paper's exogeneity argument to hold for its 2008 "
        "share, not only Peru's.")

    os.makedirs(args.out_dir, exist_ok=True)
    stats.to_csv(os.path.join(args.out_dir, "nationality_counts.csv"), index=False)
    if args.write_shares:
        keep = stats.loc[stats["candidate"], "nationality"].tolist()
        slug = {c: "share%d_%s" % (args.base_year, re.sub(r"\W+", "_", c.lower())) for c in keep}
        W = S[keep].rename(columns=slug)
        W.index.name = "muni_key"
        W.to_csv(os.path.join(args.out_dir, f"shares{args.base_year}_all_nationalities.csv"))
        out(f"\nWrote {len(keep)} share columns for {len(W)} municipalities "
            f"(definition '{dname}'). muni_key matches the IV file's municipality id.")
    return finish(args, lines)


def finish(args, lines):
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "nationality_inventory.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
