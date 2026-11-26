"""Inventory of the Ajzenman, Dominguez & Undurraga (2023) replication package, used to
design the cross-sectional and panel datasets.

Writes two text files:
  miscellaneous/replication_inventory.txt   every variable in the analysis and
      intermediate .dta files (and selected raw files): type, label, non-missing count,
      unique count, range or top values; plus whether municipality x year uniquely
      identifies the rows of each file.
  miscellaneous/replication_dofiles.txt     the .do files that build the analysis data
      (code/1_create, non-homicide versions) and the ones that run Tables VI-IX and the
      robustness tables.

Usage (from the project root):
    python -I dataset_construction/inspect_replication.py
    python -I dataset_construction/inspect_replication.py --rep-root rep_data/Replication --out-dir miscellaneous

Requires: pandas, numpy.
"""
import argparse, glob, os, re, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Raw files worth inventorying. The CASEN and raw ENUSC survey waves are very large and
# their municipality-level versions are in data/intermediate, so they are only listed.
RAW_FILES = ["immigration_Correciones.dta", "cod_com.dta", "COMUNAS.dta", "cod_reg.dta",
             "population.dta", "population_2010.dta", "base_final_2008_2017.dta",
             "countTS.dta", "PERSONA_comuna_distrito.dta", "bilateral_replic.dta"]

# Analysis do-files to dump in addition to all of code/1_create (homicide versions are
# skipped: they rely on confidential data and mirror the non-homicide files).
ANALYSIS_DOS = ["Tables_VI_VII_VIII_IX.do", "Table_A4.do", "Table_A5.do",
                "Tables_A6_A7_A8PanelA.do", "Table_A8PanelB.do", "Table_A9_Col2.do",
                "Table_A9_Cols1and3.do", "Tables_XI_XII_XIII_A16.do", "Tables_I_II_III_IV.do",
                "Table_A1.do"]

ID_PAT = re.compile(r"^(cod_com|comuna|id_com|id_comuna\d*|cut|cod_comuna)$", re.I)
YEAR_PAT = re.compile(r"^(year|ano|anio|agno)$", re.I)
MAX_BYTES_FULL = 400 * 1024 ** 2      # read files up to 400 MB in full
SAMPLE_ROWS = 200_000                 # otherwise summarize the first 200k rows
MAX_VARS = 400                        # per file


def read_dta(path):
    """Return (data, labels, complete). Value labels are kept as codes so that numeric
    variables stay numeric; labels of the variables come from the file header."""
    with pd.io.stata.StataReader(path) as r:
        labels = r.variable_labels()
    if os.path.getsize(path) <= MAX_BYTES_FULL:
        return pd.read_stata(path, convert_categoricals=False), labels, True
    with pd.read_stata(path, iterator=True, convert_categoricals=False) as r:
        return r.read(SAMPLE_ROWS), labels, False


def describe_values(s):
    v = s.dropna()
    if len(v) == 0:
        return "all missing"
    if pd.api.types.is_numeric_dtype(v) and not pd.api.types.is_bool_dtype(v):
        lo, hi, mu = v.min(), v.max(), v.mean()
        return f"min {lo:.6g}  mean {mu:.6g}  max {hi:.6g}"
    if pd.api.types.is_datetime64_any_dtype(v):
        return f"{v.min()} .. {v.max()}"
    top = v.astype(str).str.strip().value_counts().head(4)
    return "top: " + "; ".join(f"{k[:30]} ({n})" for k, n in top.items())


def key_summary(df):
    ids = [c for c in df.columns if ID_PAT.match(c)]
    yrs = [c for c in df.columns if YEAR_PAT.match(c)]
    lines = []
    for c in ids:
        lines.append(f"{c}: {df[c].nunique()} unique")
    for c in yrs:
        y = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(y):
            lines.append(f"{c}: {y.min():.0f}-{y.max():.0f} ({y.nunique()} unique)")
    if ids:
        keys = [ids[0]] + yrs[:1]
        dup = df.duplicated(subset=keys).sum()
        lines.append(f"({', '.join(keys)}) uniquely identifies rows: "
                     f"{'yes' if dup == 0 else f'no ({dup:,} duplicates)'}")
    return "; ".join(lines) if lines else "no municipality/year columns"


def inventory(paths, rep_root, out):
    for p in paths:
        rel = os.path.relpath(p, rep_root)
        size = os.path.getsize(p) / 1024 ** 2
        try:
            df, labels, complete = read_dta(p)
        except Exception as exc:
            out(f"\n=== {rel}  ({size:.1f} MB): could not read ({exc})")
            continue
        nrows = f"{len(df):,}" if complete else f">= {len(df):,} (first {SAMPLE_ROWS:,} rows summarized)"
        out(f"\n=== {rel}  ({size:.1f} MB; {nrows} rows x {df.shape[1]} vars)")
        out(f"  keys: {key_summary(df)}")
        out(f"  {'variable':32s} {'type':9s} {'nonmiss':>9s} {'unique':>8s}  label | values")
        for c in list(df.columns)[:MAX_VARS]:
            s = df[c]
            lab = (labels.get(c) or "")[:60]
            out(f"  {c[:32]:32s} {str(s.dtype)[:9]:9s} {s.notna().sum():9,d} "
                f"{s.nunique():8,d}  {lab} | {describe_values(s)}")
        if df.shape[1] > MAX_VARS:
            out(f"  ... {df.shape[1] - MAX_VARS} more variables: "
                f"{', '.join(list(df.columns)[MAX_VARS:])}")


def dump_dofiles(rep_root, out_path):
    code = os.path.join(rep_root, "code")
    files = [os.path.join(code, "Master_do.do")]
    files += sorted(p for p in glob.glob(os.path.join(code, "1_create", "*.do"))
                    if "homicide" not in os.path.basename(p).lower())
    files += [os.path.join(code, "2_analysis", f) for f in ANALYSIS_DOS]
    with open(out_path, "w", encoding="utf-8") as f:
        for p in files:
            rel = os.path.relpath(p, rep_root)
            f.write(f"\n\n{'=' * 100}\n=== {rel}\n{'=' * 100}\n")
            if not os.path.exists(p):
                f.write("[file not found]\n")
                continue
            raw = open(p, "rb").read()
            for enc in ("utf-8", "latin-1"):
                try:
                    f.write(raw.decode(enc))
                    break
                except UnicodeDecodeError:
                    continue
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep-root", default="rep_data/Replication")
    ap.add_argument("--out-dir", default="miscellaneous")
    args = ap.parse_args()
    if not os.path.isdir(os.path.join(args.rep_root, "data")):
        sys.exit(f"{args.rep_root}/data not found; pass --rep-root")
    os.makedirs(args.out_dir, exist_ok=True)

    lines = []
    def out(s=""):
        lines.append(str(s))

    data = os.path.join(args.rep_root, "data")
    analysis = sorted(glob.glob(os.path.join(data, "analysis", "*.dta")))
    intermediate = sorted(glob.glob(os.path.join(data, "intermediate", "*.dta")))
    raw = [os.path.join(data, "raw", f) for f in RAW_FILES
           if os.path.exists(os.path.join(data, "raw", f))]

    out("Replication package inventory: Ajzenman, Dominguez & Undurraga (2023)")
    out(f"Root: {args.rep_root}")
    out(f"pandas {pd.__version__}, numpy {np.__version__}")
    out(f"\nFiles listed but not inventoried (large survey microdata): "
        + ", ".join(os.path.relpath(p, args.rep_root) for p in sorted(
            glob.glob(os.path.join(data, "raw", "casen*.dta"))
            + glob.glob(os.path.join(data, "raw", "ENUSC*.dta"))
            + glob.glob(os.path.join(data, "raw", "ENUSC", "*.dta")))))

    for title, group in (("ANALYSIS FILES", analysis), ("INTERMEDIATE FILES", intermediate),
                         ("SELECTED RAW FILES", raw)):
        out(f"\n\n{'#' * 100}\n# {title} ({len(group)})\n{'#' * 100}")
        print(f"Inventorying {title.lower()} ({len(group)})...")
        inventory(group, args.rep_root, out)

    inv_path = os.path.join(args.out_dir, "replication_inventory.txt")
    with open(inv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    do_path = os.path.join(args.out_dir, "replication_dofiles.txt")
    dofiles = dump_dofiles(args.rep_root, do_path)
    print(f"Wrote {inv_path} ({len(lines):,} lines)")
    print(f"Wrote {do_path} ({len(dofiles)} do-files)")


if __name__ == "__main__":
    main()
