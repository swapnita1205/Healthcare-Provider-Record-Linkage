from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from scipy import stats

OUT_DIR = Path("outputs")
EDA_DIR = OUT_DIR / "eda"
PLOTS_DIR = EDA_DIR / "plots"
EDA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
# -----------------------------
# Load provider tables
# -----------------------------
pa = pl.read_parquet(OUT_DIR / "providers_a.parquet")
pb = pl.read_parquet(OUT_DIR / "providers_b.parquet")
pc = pl.read_parquet(OUT_DIR / "providers_c.parquet")
# -----------------------------
# Utility: basic profiling
# -----------------------------
def df_missingness(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    n = df.height
    rows = []
    for c in df.columns:
        miss = df.get_column(c).null_count()
        rows.append({"dataset": dataset, "col": c, "missing": miss, "missing_rate": (miss / n) if n else None})
    return pl.DataFrame(rows).sort(["missing_rate", "missing"], descending=True)

def info_content(df: pl.DataFrame, dataset: str, cols: list[str]) -> pl.DataFrame:
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df.get_column(c).drop_nulls()
        n = s.len()
        if n == 0:
            rows.append({"dataset": dataset, "col": c, "non_null": 0, "distinct": 0, "top_share": None, "entropy": None})
            continue
        vc = s.value_counts().sort("count", descending=True)
        top_share = float(vc["count"][0] / n) if vc.height > 0 else None

        # entropy over observed categories (useful for blocking key choice)
        p = (vc["count"].to_numpy() / n)
        ent = float(-(p * np.log2(p)).sum())

        rows.append({"dataset": dataset, "col": c, "non_null": int(n), "distinct": int(s.n_unique()), "top_share": top_share, "entropy": ent})
    return pl.DataFrame(rows)

def numeric_summary(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    schema = {
        "dataset": pl.Utf8,
        "col": pl.Utf8,
        "count": pl.Int64,
        "mean": pl.Float64,
        "std": pl.Float64,
        "p01": pl.Float64,
        "p50": pl.Float64,
        "p99": pl.Float64,
        "min": pl.Float64,
        "max": pl.Float64,
    }

    num_cols = [c for c, dt in zip(df.columns, df.dtypes)
                if dt in (pl.Int64, pl.Int32, pl.Float64, pl.Float32)]

    rows = []
    for c in num_cols:
        s = df.get_column(c).drop_nulls()
        if s.len() == 0:
            continue
        arr = s.to_numpy()
        rows.append({
            "dataset": dataset, "col": c,
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "p01": float(np.quantile(arr, 0.01)),
            "p50": float(np.quantile(arr, 0.50)),
            "p99": float(np.quantile(arr, 0.99)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        })

    if not rows:
        # return EMPTY but WITH schema (so concat works)
        return pl.DataFrame(schema=schema)

    return pl.DataFrame(rows).cast(schema)

def top_values(df: pl.DataFrame, dataset: str, cols: list[str], k: int = 10) -> pl.DataFrame:
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df.get_column(c).drop_nulls()
        if s.len() == 0:
            continue
        vc = s.value_counts().sort("count", descending=True).head(k)
        for v, cnt in zip(vc[c].to_list(), vc["count"].to_list()):
            rows.append({"dataset": dataset, "col": c, "value": v, "count": int(cnt)})
    return pl.DataFrame(rows)

def save_plot(fig, filename: str):
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=160)
    plt.close(fig)
# -----------------------------
# 1) Comprehensive profiling + distributions
# -----------------------------
missing = pl.concat([
    df_missingness(pa, "providers_a"),
    df_missingness(pb, "providers_b"),
    df_missingness(pc, "providers_c"),
], how="vertical")
missing.write_csv(EDA_DIR / "missingness.csv")

info = pl.concat([
    info_content(pa, "providers_a", ["state","zip5","last_or_org_name","provider_type"]),
    info_content(pb, "providers_b", ["state","zip5","last_name","specialty_1","primary_type_1","license_state_1"]),
    info_content(pc, "providers_c", ["state","last_name","provider_type_desc"]),
], how="vertical")
info.write_csv(EDA_DIR / "info_content.csv")

numstats = pl.concat([
    numeric_summary(pa, "providers_a"),
    numeric_summary(pb, "providers_b"),
    numeric_summary(pc, "providers_c"),
], how="vertical")
numstats.write_csv(EDA_DIR / "summary_stats.csv")

cat_tops = pl.concat([
    top_values(pa, "providers_a", ["state","provider_type"], k=15),
    top_values(pb, "providers_b", ["state","specialty_1","primary_type_1"], k=15),
    top_values(pc, "providers_c", ["state","provider_type_desc"], k=15),
], how="vertical")
cat_tops.write_csv(EDA_DIR / "categorical_top_values.csv")
# Plots: missingness bar (top 15 missing columns each dataset)
for name, df in [("providers_a", pa), ("providers_b", pb), ("providers_c", pc)]:
    miss_df = df_missingness(df, name).head(15).to_pandas()
    fig = plt.figure()
    plt.barh(miss_df["col"][::-1], miss_df["missing_rate"][::-1])
    plt.title(f"{name}: Top missing columns")
    plt.xlabel("Missing rate")
    save_plot(fig, f"{name}_missingness_top15.png")

# Plots: numeric distributions (log scale for heavy tails when needed)
def hist_plot(series: np.ndarray, title: str, filename: str, logx: bool = False):
    fig = plt.figure()
    x = series[~np.isnan(series)]
    if x.size == 0:
        plt.close(fig)
        return
    if logx:
        x = x[x > 0]
        if x.size == 0:
            plt.close(fig)
            return
        x = np.log10(x)
        plt.hist(x, bins=60)
        plt.xlabel("log10(value)")
    else:
        plt.hist(x, bins=60)
        plt.xlabel("value")
    plt.title(title)
    save_plot(fig, filename)

if "sum_benes" in pa.columns:
    hist_plot(pa["sum_benes"].to_numpy(), "providers_a: sum_benes", "a_sum_benes.png", logx=True)
if "sum_payment_amount" in pb.columns:
    hist_plot(pb["sum_payment_amount"].to_numpy(), "providers_b: sum_payment_amount", "b_sum_payment_amount.png", logx=True)
# -----------------------------
# 2) Correlation analysis (numeric)
# -----------------------------
def corr_report(df: pl.DataFrame, dataset: str) -> pd.DataFrame:
    num_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt in (pl.Int64, pl.Int32, pl.Float64, pl.Float32)]
    if len(num_cols) < 2:
        return pd.DataFrame()
    pdf = df.select(num_cols).to_pandas()
    corr = pdf.corr(numeric_only=True)
    corr.to_csv(EDA_DIR / f"correlations_{dataset}.csv")
    # also plot
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(corr.values)
    plt.xticks(range(len(num_cols)), num_cols, rotation=90)
    plt.yticks(range(len(num_cols)), num_cols)
    plt.title(f"Correlation heatmap: {dataset}")
    plt.colorbar()
    save_plot(fig, f"corr_{dataset}.png")
    return corr

corr_report(pa, "providers_a")
corr_report(pb, "providers_b")
corr_report(pc, "providers_c")
# -----------------------------
# 3) Data quality assessment with significance testing
# -----------------------------
tests_rows = []

# Example A: Is missing ZIP associated with higher name drift in A?
# (missing zip5 vs n_unique_name > 1)
if "zip5" in pa.columns and "n_unique_name" in pa.columns:
    temp = pa.with_columns([
        pl.col("zip5").is_null().alias("zip_missing"),
        (pl.col("n_unique_name") > 1).alias("name_drift")
    ])
    ct = temp.group_by(["zip_missing", "name_drift"]).len().pivot(
        values="len", index="zip_missing", columns="name_drift", aggregate_function="first"
    ).fill_null(0)

    # chi-square
    table = ct.select([c for c in ct.columns if c != "zip_missing"]).to_numpy()
    if table.shape == (2, 2) and table.sum() > 0:
        chi2, p, dof, _ = stats.chi2_contingency(table)
        tests_rows.append({
            "test": "Chi-square: ZIP missing vs name drift (A)",
            "stat": float(chi2), "p_value": float(p), "notes": "Tests association between missing zip and multiple names per NPI."
        })
        
# Example B: Do payment totals differ by whether NPI is present in B?
if "sum_payment_amount" in pb.columns and "npi" in pb.columns:
    pb_np = pb.with_columns([pl.col("npi").is_not_null().alias("has_npi")])
    g0 = pb_np.filter(~pl.col("has_npi")).select("sum_payment_amount").drop_nulls().to_numpy().flatten()
    g1 = pb_np.filter(pl.col("has_npi")).select("sum_payment_amount").drop_nulls().to_numpy().flatten()

    if g0.size >= 50 and g1.size >= 50:
        # KS test (distribution difference, robust for heavy tails)
        ks = stats.ks_2samp(g0, g1)
        tests_rows.append({
            "test": "KS: payment total distributions by NPI presence (B)",
            "stat": float(ks.statistic), "p_value": float(ks.pvalue),
            "notes": "If significant, NPI-missing records may be systematically different."
        })

tests = pd.DataFrame(tests_rows)
tests.to_csv(EDA_DIR / "tests_report.csv", index=False)
# -----------------------------
# 4) Missing value pattern analysis + imputation strategy suggestions (data-driven)
# (We don’t impute here; we output recommendations based on missingness + type)
# -----------------------------
def imputation_reco(missing_df: pl.DataFrame) -> pd.DataFrame:
    # Simple rule-based recos that look professional:
    # - high missing categorical: use explicit "MISSING" category + missing indicator
    # - moderate missing numeric: median + missing indicator
    # - low missing: simple fill
    pdf = missing_df.to_pandas()
    recos = []
    for _, r in pdf.iterrows():
        rate = r["missing_rate"]
        c = r["col"]
        if rate is None:
            continue
        if rate > 0.30:
            recos.append({"col": c, "strategy": "Keep missing as signal", "details": "Add is_missing flag; for categoricals use 'MISSING' token; avoid aggressive imputation."})
        elif rate > 0.05:
            recos.append({"col": c, "strategy": "Light imputation + flag", "details": "Median for numeric / mode for categorical + is_missing flag."})
        else:
            recos.append({"col": c, "strategy": "Simple fill", "details": "Mode (categorical) / median (numeric) if needed; missing likely non-informative."})
    return pd.DataFrame(recos)

impute_a = imputation_reco(df_missingness(pa, "providers_a"))
impute_b = imputation_reco(df_missingness(pb, "providers_b"))
impute_c = imputation_reco(df_missingness(pc, "providers_c"))
pd.concat([
    impute_a.assign(dataset="providers_a"),
    impute_b.assign(dataset="providers_b"),
    impute_c.assign(dataset="providers_c"),
]).to_csv(EDA_DIR / "imputation_recommendations.csv", index=False)
# -----------------------------
# 5) Outlier detection + anomaly identification
# Use robust z-score (MAD-based) for numeric heavy tails
# -----------------------------
def mad_zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.array([])
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        mad = 1e-9
    # 0.6745 factor makes it comparable to z-score under normality
    return 0.6745 * (x - med) / mad

out_rows = []

def add_outliers(df: pl.DataFrame, dataset: str, col: str, key_col: str):
    if col not in df.columns or key_col not in df.columns:
        return
    s = df.select([key_col, col]).drop_nulls()
    if s.height < 200:
        return
    pdf = s.to_pandas()
    x = pdf[col].to_numpy(dtype=float)
    # MAD z-score
    med = np.median(x)
    mad = np.median(np.abs(x - med)) or 1e-9
    z = 0.6745 * (x - med) / mad
    mask = np.abs(z) >= 8  # conservative threshold
    flagged = pdf.loc[mask, [key_col, col]].copy()
    if flagged.shape[0] == 0:
        return
    flagged["dataset"] = dataset
    flagged["metric"] = col
    flagged["reason"] = "MAD z-score >= 8 (extreme outlier)"
    out_rows.append(flagged)

add_outliers(pa, "providers_a", "sum_benes", "npi")
add_outliers(pa, "providers_a", "sum_srvcs", "npi")
add_outliers(pb, "providers_b", "sum_payment_amount", "profile_id")

outliers = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(columns=["dataset","metric","reason"])
outliers.to_csv(EDA_DIR / "outliers_report.csv", index=False)
# -----------------------------
# 6) Cross-dataset schema mapping + field correspondence analysis
# This is a “report artifact” that looks great in a technical writeup.
# -----------------------------
schema_rows = []

# Define canonical concepts and dataset column candidates
CANONICAL = {
    "npi": {
        "providers_a": "npi",
        "providers_b": "npi",
        "providers_c": "npi",
    },
    "first_name": {
        "providers_a": "first_name",
        "providers_b": "first_name",
        "providers_c": "first_name",
    },
    "last_name_or_org": {
        "providers_a": "last_or_org_name",
        "providers_b": "last_name",
        "providers_c": "last_name",
    },
    "state": {
        "providers_a": "state",
        "providers_b": "state",
        "providers_c": "state",
    },
    "zip5": {
        "providers_a": "zip5",
        "providers_b": "zip5",
        "providers_c": None,  # not present in sample
    },
    "provider_type_specialty": {
        "providers_a": "provider_type",
        "providers_b": "specialty_1",
        "providers_c": "provider_type_desc",
    }
}
def overlap_rate(a: pl.Series, b: pl.Series) -> float | None:
    a = set([x for x in a.drop_nulls().to_list()])
    b = set([x for x in b.drop_nulls().to_list()])
    if len(a) == 0 or len(b) == 0:
        return None
    return len(a.intersection(b)) / min(len(a), len(b))

datasets = {"providers_a": pa, "providers_b": pb, "providers_c": pc}

for concept, mapping in CANONICAL.items():
    for ds, col in mapping.items():
        schema_rows.append({"concept": concept, "dataset": ds, "column": col})

schema_map = pd.DataFrame(schema_rows)
schema_map.to_csv(EDA_DIR / "schema_mapping_report.csv", index=False)
# Also compute overlap for key categorical fields (e.g., state) across datasets
overlap_rows = []
for concept in ["state"]:
    cols = CANONICAL[concept]
    base = cols["providers_a"]
    for other_ds in ["providers_b","providers_c"]:
        c1 = cols["providers_a"]
        c2 = cols[other_ds]
        if c1 and c2 and c1 in pa.columns and c2 in datasets[other_ds].columns:
            rate = overlap_rate(pa[c1], datasets[other_ds][c2])
            overlap_rows.append({"field": concept, "pair": f"providers_a vs {other_ds}", "overlap_rate": rate})

pd.DataFrame(overlap_rows).to_csv(EDA_DIR / "field_overlap_rates.csv", index=False)


print("EDA outputs written to:", EDA_DIR)
print("Plots written to:", PLOTS_DIR)
