from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import polars as pl

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    average_precision_score, roc_auc_score, precision_recall_curve
)
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

# -----------------------------
# Paths / config
# -----------------------------
OUT_DIR = Path("outputs")
FEAT_DIR = OUT_DIR / "features"
MODEL_DIR = OUT_DIR / "models"
REPORT_DIR = OUT_DIR / "stat_validation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PATH_BA = FEAT_DIR / "pair_features_ba.parquet"
PATH_BC = FEAT_DIR / "pair_features_bc.parquet"

PA_PATH = OUT_DIR / "providers_a.parquet"
PB_PATH = OUT_DIR / "providers_b.parquet"
PC_PATH = OUT_DIR / "providers_c.parquet"

RANDOM_STATE = 42
TEST_SIZE = 0.20

GROUP_MODE = "profile" 

# Cluster bootstrap
BOOT_N = 400
BOOT_SEED = 123

# Threshold:
THRESH_JSON = MODEL_DIR / "thresholds.json"
THRESH_KEY = "precision_target_threshold"
DEFAULT_THR = 0.5

# Sensitivity: threshold grid
THR_GRID = np.linspace(0.05, 0.99, 40)

def load_pairs() -> pl.DataFrame:
    ba = pl.read_parquet(PATH_BA).with_columns(pl.lit("BA").alias("pair_type"))
    bc = pl.read_parquet(PATH_BC).with_columns(pl.lit("BC").alias("pair_type"))
    ba = ba.with_columns(pl.col("npi_a").alias("candidate_npi")).drop("npi_a")
    bc = bc.with_columns(pl.col("npi_c").alias("candidate_npi")).drop("npi_c")
    return pl.concat([ba, bc], how="vertical")

df = load_pairs()

NON_FEATURES = {"profile_id", "pass", "block_key", "label_weak_npi", "candidate_npi", "pair_type"}
feature_cols = [c for c in df.columns if c not in NON_FEATURES]

labeled = (
    df.filter(pl.col("label_weak_npi").is_not_null())
      .with_columns(pl.col("label_weak_npi").cast(pl.Int8).alias("y"))
      .drop("label_weak_npi")
)

print("Labeled rows:", labeled.height, "Features:", len(feature_cols))

pdf = labeled.to_pandas()
X = pdf[feature_cols].astype(float).values
y = pdf["y"].astype(int).values

# -----------------------------
# Choose grouping
# -----------------------------
if GROUP_MODE == "profile":
    groups = pdf["profile_id"].values
elif GROUP_MODE == "provider":
    mask = pd.notna(pdf["candidate_npi"])
    pdf = pdf.loc[mask].reset_index(drop=True)
    X = pdf[feature_cols].astype(float).values
    y = pdf["y"].astype(int).values
    groups = pdf["candidate_npi"].values
else:
    raise ValueError("GROUP_MODE must be 'profile' or 'provider'")

print("Group mode:", GROUP_MODE, "Unique groups:", len(pd.unique(groups)))

# -----------------------------
# Threshold
# -----------------------------
thr = DEFAULT_THR
if THRESH_JSON.exists():
    d = json.loads(THRESH_JSON.read_text())
    thr = float(d.get("thresholds", {}).get(THRESH_KEY, DEFAULT_THR))
print("Using threshold:", thr)

# -----------------------------
# Helpers
# -----------------------------
def safe_scores(model, Xb: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xb)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(Xb)
        return 1.0 / (1.0 + np.exp(-z))
    return model.predict(Xb).astype(float)
def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray, thr_val: float) -> dict:
    y_pred = (scores >= thr_val).astype(int)
    out = {
        "threshold": float(thr_val),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }
    out["roc_auc"] = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None
    return out

def by_pair_type_metrics(test_df: pd.DataFrame, scores: np.ndarray, thr_val: float) -> dict:
    out = {}
    for pt in ["BA", "BC"]:
        m = (test_df["pair_type"].values == pt)
        if m.sum() == 0:
            continue
        out[pt] = metrics_from_scores(test_df.loc[m, "y"].values.astype(int), scores[m], thr_val)
    return out

def precision_target_threshold(y_true: np.ndarray, scores: np.ndarray, target_precision: float) -> float:
    prec, rec, thr_arr = precision_recall_curve(y_true, scores)
    thr_full = np.concatenate([thr_arr, [1.0]])
    idxs = np.where(prec >= target_precision)[0]
    if idxs.size == 0:
        f1 = (2 * prec * rec) / np.clip((prec + rec), 1e-12, None)
        return float(thr_full[int(np.nanargmax(f1))])
    bestp = idxs[np.argmax(rec[idxs])]
    return float(thr_full[bestp])

# ---- Cluster bootstrap by groups ----
def cluster_bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    groups_arr: np.ndarray,
    thr_val: float,
    n_boot: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    uniq = pd.unique(groups_arr)
    # mapping group -> indices
    group_to_idx = {}
    for i, g in enumerate(groups_arr):
        group_to_idx.setdefault(g, []).append(i)

    pr_aucs, roc_aucs = [], []
    precs, recs, f1s = [], [], []

    for _ in range(n_boot):
        sampled_groups = rng.choice(uniq, size=len(uniq), replace=True)
        idx = []
        for g in sampled_groups:
            idx.extend(group_to_idx[g])
        idx = np.asarray(idx, dtype=int)

        yt = y_true[idx]
        sc = scores[idx]

        # metrics
        pr_aucs.append(average_precision_score(yt, sc))
        if len(np.unique(yt)) > 1:
            roc_aucs.append(roc_auc_score(yt, sc))
        else:
            roc_aucs.append(np.nan)

        yp = (sc >= thr_val).astype(int)
        precs.append(precision_score(yt, yp, zero_division=0))
        recs.append(recall_score(yt, yp, zero_division=0))
        f1s.append(f1_score(yt, yp, zero_division=0))

    def ci(a):
        a = np.asarray(a, dtype=float)
        a = a[~np.isnan(a)]
        return {
            "mean": float(np.mean(a)) if a.size else None,
            "p025": float(np.quantile(a, 0.025)) if a.size else None,
            "p975": float(np.quantile(a, 0.975)) if a.size else None,
        }

    return {
        "pr_auc": ci(pr_aucs),
        "roc_auc": ci(roc_aucs),
        "precision": ci(precs),
        "recall": ci(recs),
        "f1": ci(f1s),
        "n_boot": int(n_boot),
    }

# Paired cluster bootstrap test for difference in PR-AUC / ROC-AUC (A vs B)
def paired_cluster_bootstrap_diff(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    groups_arr: np.ndarray,
    metric: str,
    n_boot: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    uniq = pd.unique(groups_arr)
    group_to_idx = {}
    for i, g in enumerate(groups_arr):
        group_to_idx.setdefault(g, []).append(i)

    diffs = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(uniq, size=len(uniq), replace=True)
        idx = []
        for g in sampled_groups:
            idx.extend(group_to_idx[g])
        idx = np.asarray(idx, dtype=int)

        yt = y_true[idx]
        sa = scores_a[idx]
        sb = scores_b[idx]

        if metric == "pr_auc":
            ma = average_precision_score(yt, sa)
            mb = average_precision_score(yt, sb)
        elif metric == "roc_auc":
            if len(np.unique(yt)) <= 1:
                continue
            ma = roc_auc_score(yt, sa)
            mb = roc_auc_score(yt, sb)
        else:
            raise ValueError("metric must be pr_auc or roc_auc")

        diffs.append(ma - mb)

    diffs = np.asarray(diffs, dtype=float)
    if diffs.size == 0:
        return {"metric": metric, "diff_mean": None, "p_value_two_sided": None, "note": "Insufficient bootstrap samples."}

    p = 2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    p = float(min(p, 1.0))

    return {
        "metric": metric,
        "diff_mean": float(diffs.mean()),
        "diff_p025": float(np.quantile(diffs, 0.025)),
        "diff_p975": float(np.quantile(diffs, 0.975)),
        "p_value_two_sided": p,
        "n_boot_used": int(diffs.size),
    }

models = {}

best_path = MODEL_DIR / "best_model.joblib"
if best_path.exists():
    models["best_saved"] = joblib.load(best_path)

models["logreg"] = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
])

models["grad_boost"] = HistGradientBoostingClassifier(random_state=RANDOM_STATE)

# -----------------------------
# (1) Grouped CV
# -----------------------------
cv = GroupKFold(n_splits=3)
cv_rows = []
oof_scores = {name: np.zeros(len(pdf), dtype=float) for name in models.keys()}

for fold, (tr, te) in enumerate(cv.split(X, y, groups=groups), start=1):
    for name, m in models.items():
        model = clone(m)
        model.fit(X[tr], y[tr])
        sc = safe_scores(model, X[te])
        oof_scores[name][te] = sc

        met = metrics_from_scores(y[te], sc, thr)
        cv_rows.append({"model": name, "fold": fold, **met})

cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(REPORT_DIR / f"cv_grouped_{GROUP_MODE}.csv", index=False)

# Storing fold-level PR-AUC distributions
cv_pr_summary = (
    cv_df.groupby(["model"])["pr_auc"]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
)
cv_pr_summary.to_csv(REPORT_DIR / f"cv_pr_auc_summary_{GROUP_MODE}.csv", index=False)

# -----------------------------
# (2) Grouped holdout split + cluster bootstrap CI
# -----------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
tr_idx, te_idx = next(gss.split(X, y, groups=groups))

X_tr, y_tr = X[tr_idx], y[tr_idx]
X_te, y_te = X[te_idx], y[te_idx]
groups_te = groups[te_idx]
test_pdf = pdf.iloc[te_idx].copy()

holdout_summary = {}
scores_store = {}

for name, m in models.items():
    model = clone(m)
    model.fit(X_tr, y_tr)
    sc = safe_scores(model, X_te)
    scores_store[name] = sc

    base = metrics_from_scores(y_te, sc, thr)
    base["by_pair_type"] = by_pair_type_metrics(test_pdf, sc, thr)

    ci = cluster_bootstrap_ci(
        y_true=y_te, scores=sc, groups_arr=groups_te,
        thr_val=thr, n_boot=BOOT_N, seed=BOOT_SEED
    )

    holdout_summary[name] = {"metrics": base, "cluster_bootstrap_ci": ci}

with open(REPORT_DIR / f"holdout_cluster_bootstrap_{GROUP_MODE}.json", "w") as f:
    json.dump(holdout_summary, f, indent=2)

# -----------------------------
# (3) Significance testing of model differences (paired cluster bootstrap diffs)
# -----------------------------
if "best_saved" in scores_store:
    A, B = "best_saved", "grad_boost"
else:
    A, B = "logreg", "grad_boost"

diff_tests = {
    "pr_auc": paired_cluster_bootstrap_diff(y_te, scores_store[A], scores_store[B], groups_te, "pr_auc", BOOT_N, BOOT_SEED),
    "roc_auc": paired_cluster_bootstrap_diff(y_te, scores_store[A], scores_store[B], groups_te, "roc_auc", BOOT_N, BOOT_SEED),
}
with open(REPORT_DIR / f"paired_cluster_bootstrap_{A}_vs_{B}_{GROUP_MODE}.json", "w") as f:
    json.dump({"A": A, "B": B, **diff_tests}, f, indent=2)

# -----------------------------
# (4) Threshold sensitivity analysis
# -----------------------------
sens_rows = []
for name, sc in scores_store.items():
    for t in THR_GRID:
        met = metrics_from_scores(y_te, sc, float(t))
        sens_rows.append({"model": name, "threshold": float(t), "precision": met["precision"], "recall": met["recall"], "f1": met["f1"]})

sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(REPORT_DIR / f"threshold_sensitivity_{GROUP_MODE}.csv", index=False)

# Recomputing precision-target thresholds at multiple targets on the same holdout
targets = [0.90, 0.95, 0.98]
pt_rows = []
for name, sc in scores_store.items():
    for tp in targets:
        tpthr = precision_target_threshold(y_te, sc, tp)
        met = metrics_from_scores(y_te, sc, tpthr)
        pt_rows.append({
            "model": name,
            "target_precision": float(tp),
            "threshold": float(tpthr),
            "achieved_precision": met["precision"],
            "recall": met["recall"],
            "f1": met["f1"],
        })
pd.DataFrame(pt_rows).to_csv(REPORT_DIR / f"precision_target_thresholds_{GROUP_MODE}.csv", index=False)

# -----------------------------
# (5) Error buckets
# -----------------------------
pa = pl.read_parquet(PA_PATH).select([
    pl.col("npi").alias("npi_a"),
    pl.col("first_name").alias("a_first"),
    pl.col("last_or_org_name").alias("a_last_or_org"),
    pl.col("street1").alias("a_street1"),
    pl.col("city").alias("a_city"),
    pl.col("state").alias("a_state"),
    pl.col("zip5").alias("a_zip5"),
])

pb = pl.read_parquet(PB_PATH).select([
    "profile_id",
    pl.col("first_name").alias("b_first"),
    pl.col("last_name").alias("b_last"),
    pl.col("street1").alias("b_street1"),
    pl.col("city").alias("b_city"),
    pl.col("state").alias("b_state"),
    pl.col("zip5").alias("b_zip5"),
])

pc = pl.read_parquet(PC_PATH).select([
    pl.col("npi").alias("npi_c"),
    pl.col("first_name").alias("c_first"),
    pl.col("last_name").alias("c_last"),
    pl.col("org_name").alias("c_org"),
    pl.col("state").alias("c_state"),
])

bucket_model = "best_saved" if "best_saved" in scores_store else "grad_boost"
sc_bucket = scores_store[bucket_model]
yhat_bucket = (sc_bucket >= thr).astype(int)

tmp = test_pdf.copy()
tmp["yhat"] = yhat_bucket

test_pl = pl.from_pandas(tmp[["profile_id", "pair_type", "candidate_npi", "pass", "block_key", "y", "yhat"]])

ba_enriched = (
    test_pl.filter(pl.col("pair_type") == "BA")
    .join(pb, on="profile_id", how="left")
    .with_columns(pl.col("candidate_npi").alias("npi_a"))
    .join(pa, on="npi_a", how="left")
    .with_columns([
        pl.col("a_first").alias("x_first"),
        pl.col("a_last_or_org").alias("x_last"),
        pl.lit("").alias("x_org"),
        pl.col("a_street1").alias("x_street1"),
        pl.col("a_city").alias("x_city"),
        pl.col("a_state").alias("x_state"),
        pl.col("a_zip5").alias("x_zip5"),
    ])
)

bc_enriched = (
    test_pl.filter(pl.col("pair_type") == "BC")
    .join(pb, on="profile_id", how="left")
    .with_columns(pl.col("candidate_npi").alias("npi_c"))
    .join(pc, on="npi_c", how="left")
    .with_columns([
        pl.lit("").alias("x_street1"),
        pl.lit("").alias("x_city"),
        pl.lit("").alias("x_zip5"),
        pl.col("c_first").alias("x_first"),
        pl.col("c_last").alias("x_last"),
        pl.col("c_org").alias("x_org"),
        pl.col("c_state").alias("x_state"),
    ])
)

def align_pair(df_left: pl.DataFrame, df_right: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    all_cols = sorted(set(df_left.columns) | set(df_right.columns))
    dtypes_left = dict(zip(df_left.columns, df_left.dtypes))
    dtypes_right = dict(zip(df_right.columns, df_right.dtypes))

    def add_missing(df: pl.DataFrame, missing_cols: list[str], dtype_source: dict) -> pl.DataFrame:
        exprs = [pl.lit(None).cast(dtype_source.get(c, pl.Utf8)).alias(c) for c in missing_cols]
        return df.with_columns(exprs) if exprs else df

    left_missing = [c for c in all_cols if c not in df_left.columns]
    right_missing = [c for c in all_cols if c not in df_right.columns]

    df_left2 = add_missing(df_left, left_missing, dtypes_right)
    df_right2 = add_missing(df_right, right_missing, dtypes_left)

    return df_left2.select(all_cols), df_right2.select(all_cols)

ba_al, bc_al = align_pair(ba_enriched, bc_enriched)
enriched = pl.concat([ba_al, bc_al], how="vertical", rechunk=True)
def bucket_expr() -> pl.Expr:
    ORG_TOKENS = [" LLC", " INC", " HOSP", " HOSPITAL", " CLINIC", " CENTER", " MEDICAL", " GROUP", " ASSOCIATES", " PC", " PLLC"]

    x_last = pl.col("x_last").fill_null("")
    x_org  = pl.col("x_org").fill_null("")
    b_last = pl.col("b_last").fill_null("")
    b_first = pl.col("b_first").fill_null("")
    x_first = pl.col("x_first").fill_null("")
    b_st = pl.col("b_state").fill_null("")
    x_st = pl.col("x_state").fill_null("")
    b_zip = pl.col("b_zip5").fill_null("")
    x_zip = pl.col("x_zip5").fill_null("")
    b_st1 = pl.col("b_street1").fill_null("")
    x_st1 = pl.col("x_street1").fill_null("")

    org_like = pl.any_horizontal([(x_last + " " + x_org).str.contains(tok) for tok in ORG_TOKENS])

    name_first_match = (b_first.str.len_chars() > 0) & (x_first.str.len_chars() > 0) & (b_first == x_first)
    name_last_match  = (b_last.str.len_chars() > 0) & (x_last.str.len_chars() > 0) & (b_last == x_last)

    geo_match = (b_st.str.len_chars() > 0) & (x_st.str.len_chars() > 0) & (b_st == x_st)
    zip_match = (b_zip.str.len_chars() > 0) & (x_zip.str.len_chars() > 0) & (b_zip == x_zip)
    addr_match = (b_st1.str.len_chars() > 0) & (x_st1.str.len_chars() > 0) & (b_st1 == x_st1)

    return (
        pl.when(org_like & (b_first.str.len_chars() > 0))
          .then(pl.lit("organization_vs_individual"))
        .when(name_first_match & ~name_last_match)
          .then(pl.lit("married_or_name_change"))
        .when(name_first_match & name_last_match & geo_match & ~(zip_match & addr_match))
          .then(pl.lit("multiple_practice_locations"))
        .when(name_first_match & name_last_match & geo_match)
          .then(pl.lit("same_name_different_person"))
        .otherwise(pl.lit("other"))
    )
mistakes = (
    enriched
    .with_columns([
        (((pl.col("y") == 0) & (pl.col("yhat") == 1)).alias("is_fp")),
        (((pl.col("y") == 1) & (pl.col("yhat") == 0)).alias("is_fn")),
        bucket_expr().alias("error_bucket"),
    ])
    .filter(pl.col("is_fp") | pl.col("is_fn"))
)

mistakes.write_parquet(REPORT_DIR / f"mistakes_{bucket_model}_{GROUP_MODE}.parquet")

bucket_counts = (
    mistakes.group_by(["error_bucket", "pair_type"])
    .agg(pl.len().alias("n"))
    .sort("n", descending=True)
)
bucket_counts.write_csv(REPORT_DIR / f"error_buckets_{bucket_model}_{GROUP_MODE}.csv")

print("-- Wrote reports to:", REPORT_DIR)
print(" - CV:", REPORT_DIR / f"cv_grouped_{GROUP_MODE}.csv")
print(" - CV PR summary:", REPORT_DIR / f"cv_pr_auc_summary_{GROUP_MODE}.csv")
print(" - Holdout+cluster bootstrap:", REPORT_DIR / f"holdout_cluster_bootstrap_{GROUP_MODE}.json")
print(" - Paired cluster bootstrap:", REPORT_DIR / f"paired_cluster_bootstrap_{A}_vs_{B}_{GROUP_MODE}.json")
print(" - Threshold sensitivity:", REPORT_DIR / f"threshold_sensitivity_{GROUP_MODE}.csv")
print(" - Precision-target thresholds:", REPORT_DIR / f"precision_target_thresholds_{GROUP_MODE}.csv")
print(" - Error buckets:", REPORT_DIR / f"error_buckets_{bucket_model}_{GROUP_MODE}.csv")
print(" - Mistakes parquet:", REPORT_DIR / f"mistakes_{bucket_model}_{GROUP_MODE}.parquet")
