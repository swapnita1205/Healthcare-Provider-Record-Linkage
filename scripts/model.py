from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import polars as pl

from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance
import joblib
# -----------------------------
# Config
# -----------------------------
OUT_DIR = Path("outputs")
FEAT_DIR = OUT_DIR / "features"
MODEL_DIR = OUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PATH_BA = FEAT_DIR / "pair_features_ba.parquet"
PATH_BC = FEAT_DIR / "pair_features_bc.parquet"

RANDOM_STATE = 42

# Training-set construction
NEG_PER_POS = 10
HARD_NEG_FRAC = 0.5
MAX_TRAIN_ROWS = 1_200_000

N_SPLITS = 3
N_ITER_LR = 12
N_ITER_GB = 16
N_ITER_SGD = 14

# Active learning
UNLABELED_SAMPLE = 400_000
ACTIVE_QUERY_N = 200

# Calibration
CALIBRATE_LINEAR_MODELS = True

# -----------------------------
# Helpers
# -----------------------------
def tune_threshold(y_true: np.ndarray, scores: np.ndarray, target_precision: float = 0.95) -> dict:
    prec, rec, thr = precision_recall_curve(y_true, scores)
    thr_full = np.concatenate([thr, [1.0]])

    f1 = (2 * prec * rec) / np.clip((prec + rec), 1e-12, None)
    best_idx = int(np.nanargmax(f1))
    best_f1_thr = float(thr_full[best_idx])

    idxs = np.where(prec >= target_precision)[0]
    if idxs.size > 0:
        bestp = idxs[np.argmax(rec[idxs])]
        prec_thr = float(thr_full[bestp])
    else:
        prec_thr = best_f1_thr

    return {
        "target_precision": float(target_precision),
        "best_f1_threshold": best_f1_thr,
        "precision_target_threshold": prec_thr,
    }

def metrics_at(y_true: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    pred = (scores >= thr).astype(int)
    return {
        "threshold": float(thr),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }

def eval_by_pair_type(df_test: pd.DataFrame, scores: np.ndarray, thr: float) -> dict:
    out = {}
    for pt in ["BA", "BC"]:
        mask = (df_test["pair_type"].values == pt)
        if mask.sum() == 0:
            continue
        out[pt] = metrics_at(df_test.loc[mask, "y"].values.astype(int), scores[mask], thr)
    return out

def safe_predict_proba(est, X: np.ndarray) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    if hasattr(est, "decision_function"):
        z = est.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    return est.predict(X).astype(float)

# -----------------------------
# Loading schema
# -----------------------------
ba_head = pl.read_parquet(PATH_BA, n_rows=5)
bc_head = pl.read_parquet(PATH_BC, n_rows=5)

ba_cols = set(ba_head.columns)
bc_cols = set(bc_head.columns)

NON_FEATURES = {"profile_id", "pass", "block_key", "label_weak_npi", "npi_a", "npi_c"}
feature_cols = sorted(list((ba_cols & bc_cols) - NON_FEATURES))

print("Using #features:", len(feature_cols))

ba_scan = pl.scan_parquet(PATH_BA).with_columns([
    pl.lit("BA").alias("pair_type"),
    pl.col("npi_a").alias("candidate_npi"),
])
bc_scan = pl.scan_parquet(PATH_BC).with_columns([
    pl.lit("BC").alias("pair_type"),
    pl.col("npi_c").alias("candidate_npi"),
])

cols_needed = ["profile_id", "block_key", "pair_type", "pass", "candidate_npi", "label_weak_npi"] + feature_cols
df_scan = pl.concat([
    ba_scan.select(cols_needed),
    bc_scan.select(cols_needed),
], how="vertical")

df_scan = df_scan.with_columns([
    pl.col(c).cast(pl.Float64).fill_null(0.0).alias(c) for c in feature_cols
])

labeled = df_scan.filter(pl.col("label_weak_npi").is_not_null()).with_columns([
    pl.col("label_weak_npi").cast(pl.Int8).alias("y")
]).drop("label_weak_npi")

# -----------------------------
# Building TRAINING SET: all positives + sampled negatives
# -----------------------------
pos = labeled.filter(pl.col("y") == 1)
neg = labeled.filter(pl.col("y") == 0)

pos_counts = pos.group_by("pair_type").agg(pl.len().alias("n_pos")).collect()
print("Pos counts by pair_type:\n", pos_counts)

hard_neg_filter = None
if "sim_jw_lastname" in feature_cols and "sim_jw_fullname" in feature_cols:
    hard_neg_filter = (
        (pl.col("sim_jw_lastname") >= 0.92) |
        (pl.col("sim_jw_fullname") >= 0.92)
    )

pos_total = pos.select(pl.len()).collect().item()
neg_target_total = min(int(pos_total * NEG_PER_POS), MAX_TRAIN_ROWS - int(pos_total))
hard_target_total = int(neg_target_total * HARD_NEG_FRAC)
rand_target_total = neg_target_total - hard_target_total

print(f"Pos total: {pos_total}")
print(f"Neg target total: {neg_target_total} (hard={hard_target_total}, random={rand_target_total})")

def lazy_sample_n(lf: pl.LazyFrame, n: int, seed: int) -> pl.LazyFrame:
    if n <= 0:
        return lf.limit(0)

    df_small = lf.select(pl.arange(0, pl.len()).alias("_row_id")).collect()
    total = df_small.height
    if total == 0:
        return lf.limit(0)

    n = min(n, total)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(total, size=n, replace=False)
    chosen_df = pl.DataFrame({"_row_id": chosen})

    return (
        lf.with_row_count("_row_id")
          .join(chosen_df.lazy(), on="_row_id", how="inner")
          .drop("_row_id")
    )

# Hard negatives
if hard_neg_filter is not None:
    hard_pool = neg.filter(hard_neg_filter)
    hard_cnt = hard_pool.select(pl.len()).collect().item()
    hard_take = min(hard_target_total, hard_cnt)
    hard_sample = lazy_sample_n(hard_pool, hard_take, RANDOM_STATE)
else:
    hard_sample = neg.limit(0)

# Random negatives
neg_cnt = neg.select(pl.len()).collect().item()
rand_take = min(rand_target_total, neg_cnt)
rand_sample = lazy_sample_n(neg, rand_take, RANDOM_STATE + 1)

train_lf = pl.concat([pos, hard_sample, rand_sample], how="vertical")
train_df = train_lf.collect(streaming=True)
print("Train rows:", train_df.height, "Pos rate:", float(train_df["y"].mean()))

train_pdf = train_df.to_pandas()
X = train_pdf[feature_cols].values
y = train_pdf["y"].values.astype(int)
groups = train_pdf["profile_id"].values

# -----------------------------
# Holdout split (grouped)
# -----------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
test_pdf = train_pdf.iloc[test_idx].copy()

# -----------------------------
# Models
# -----------------------------
cv = GroupKFold(n_splits=N_SPLITS)

# Logistic Regression (strong baseline)
lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
])

# SGD logistic regression
sgd = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("clf", SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=2000,
        tol=1e-4,
    )),
])

# Gradient Boosting
gb = HistGradientBoostingClassifier(random_state=RANDOM_STATE)

param_space_lr = {"clf__C": np.logspace(-3, 2, 15)}
param_space_sgd = {
    "clf__alpha": np.logspace(-6, -3, 12),
    "clf__l1_ratio": [0.0, 0.15, 0.5, 0.85, 1.0],
}
param_space_gb = {
    "learning_rate": [0.03, 0.06, 0.1],
    "max_depth": [3, 5, 7],
    "max_iter": [200, 400, 700],
    "min_samples_leaf": [20, 40, 80],
    "l2_regularization": [0.0, 1e-3, 1e-2],
}

def fit_search(estimator, params, n_iter, name: str, n_jobs: int):
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(X_train, y_train, groups=groups[train_idx])
    return search
print("=== LR HPO ===")
lr_search = fit_search(lr, param_space_lr, N_ITER_LR, "lr", n_jobs=2)

print("=== SGD(log-loss) HPO ===")
sgd_search = fit_search(sgd, param_space_sgd, N_ITER_SGD, "sgd", n_jobs=2)

print("=== GB HPO ===")
gb_search = fit_search(gb, param_space_gb, N_ITER_GB, "gb", n_jobs=2)

searches = {
    "logreg": lr_search,
    "sgd_logloss": sgd_search,
    "grad_boost": gb_search,
}


def maybe_calibrate(name: str, estimator):
    if not CALIBRATE_LINEAR_MODELS:
        return estimator
    if name in ("logreg", "sgd_logloss"):
        return CalibratedClassifierCV(estimator, method="sigmoid", cv=3)
    return estimator

# -----------------------------
# Compare models on holdout
# -----------------------------
results_rows = []
details = {}

for name, search in searches.items():
    best = search.best_estimator_
    best.fit(X_train, y_train)

    calibrated = maybe_calibrate(name, best)
    calibrated.fit(X_train, y_train)

    scores = safe_predict_proba(calibrated, X_test)
    thr = tune_threshold(y_test, scores, target_precision=0.95)

    m_best = metrics_at(y_test, scores, thr["best_f1_threshold"])
    m_prec = metrics_at(y_test, scores, thr["precision_target_threshold"])

    results_rows.append({
        "model": name,
        "best_params": json.dumps(search.best_params_),
        "holdout_pr_auc": m_best["pr_auc"],
        "holdout_roc_auc": m_best["roc_auc"],
        "holdout_f1_at_best": m_best["f1"],
        "holdout_precision_at_best": m_best["precision"],
        "holdout_recall_at_best": m_best["recall"],
        "holdout_precision_at_prec_target": m_prec["precision"],
        "holdout_recall_at_prec_target": m_prec["recall"],
        "best_f1_threshold": thr["best_f1_threshold"],
        "precision_target_threshold": thr["precision_target_threshold"],
        "calibrated": bool(CALIBRATE_LINEAR_MODELS and name in ("logreg", "sgd_logloss")),
    })

    details[name] = {
        "best_cv_pr_auc": float(search.best_score_),
        "holdout_best_f1": m_best,
        "holdout_precision_target": m_prec,
        "holdout_by_pair_type_at_prec_target": eval_by_pair_type(test_pdf, scores, thr["precision_target_threshold"]),
        "cv_top3": pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(3)[
            ["rank_test_score", "mean_test_score", "std_test_score", "params"]
        ].to_dict(orient="records"),
    }
results_df = pd.DataFrame(results_rows).sort_values("holdout_pr_auc", ascending=False)
results_df.to_csv(MODEL_DIR / "all_models_results.csv", index=False)
with open(MODEL_DIR / "cv_metrics.json", "w") as f:
    json.dump(details, f, indent=2)

best_model_name = results_df.iloc[0]["model"]
best_est = searches[best_model_name].best_estimator_

best_est.fit(X, y)
best_model = maybe_calibrate(best_model_name, best_est)
best_model.fit(X, y)

joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

# Thresholds computed on holdout for reporting
best_scores = safe_predict_proba(best_model, X_test)
best_thr = tune_threshold(y_test, best_scores, target_precision=0.95)

with open(MODEL_DIR / "thresholds.json", "w") as f:
    json.dump({"best_model": best_model_name, "thresholds": best_thr}, f, indent=2)

metrics_out = {
    "train_rows": int(len(train_pdf)),
    "pos_rate_train": float(y.mean()),
    "best_model": best_model_name,
    "calibrated": bool(CALIBRATE_LINEAR_MODELS and best_model_name in ("logreg", "sgd_logloss")),
    "holdout": {
        "best_f1": metrics_at(y_test, best_scores, best_thr["best_f1_threshold"]),
        "precision_target": metrics_at(y_test, best_scores, best_thr["precision_target_threshold"]),
        "by_pair_type_at_prec_target": eval_by_pair_type(test_pdf, best_scores, best_thr["precision_target_threshold"]),
    },
}
with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)

# -----------------------------
# Interpretability
# -----------------------------
# LR coefficients
lr_best = lr_search.best_estimator_
lr_best.fit(X, y)
lr_cal = maybe_calibrate("logreg", lr_best)
lr_cal.fit(X, y)

coef_model = lr_best
coef = coef_model.named_steps["clf"].coef_.flatten()
pd.DataFrame({"feature": feature_cols, "coef": coef}).sort_values("coef", ascending=False)\
  .to_csv(MODEL_DIR / "feature_importance_lr.csv", index=False)

# Permutation importance for the best model on holdout
perm = permutation_importance(
    best_model, X_test, y_test,
    n_repeats=5, random_state=RANDOM_STATE, scoring="average_precision"
)
pd.DataFrame({
    "feature": feature_cols,
    "perm_importance_mean": perm.importances_mean,
    "perm_importance_std": perm.importances_std,
}).sort_values("perm_importance_mean", ascending=False)\
  .to_csv(MODEL_DIR / "feature_importance_best_model.csv", index=False)
  
# -----------------------------
# Active learning queue
# -----------------------------
unlabeled = df_scan.filter(pl.col("label_weak_npi").is_null()).drop("label_weak_npi")
unl_sample = lazy_sample_n(unlabeled, UNLABELED_SAMPLE, RANDOM_STATE).collect(streaming=True)
unl_pdf = unl_sample.to_pandas()
X_unl = unl_pdf[feature_cols].values

unl_scores = safe_predict_proba(best_model, X_unl)
uncertainty = np.abs(unl_scores - 0.5)

idx_unc = np.argsort(uncertainty)[: int(ACTIVE_QUERY_N * 0.6)]
idx_hi = np.argsort(-unl_scores)[: int(ACTIVE_QUERY_N * 0.2)]
idx_lo = np.argsort(unl_scores)[: int(ACTIVE_QUERY_N * 0.2)]
idx_q = np.unique(np.concatenate([idx_unc, idx_hi, idx_lo]))[:ACTIVE_QUERY_N]

queue = unl_pdf.iloc[idx_q].copy()
queue["score"] = unl_scores[idx_q]
queue["uncertainty"] = uncertainty[idx_q]
queue[["profile_id", "candidate_npi", "pair_type", "pass", "block_key", "score", "uncertainty"]].to_csv(
    MODEL_DIR / "active_learning_queue.csv", index=False
)

print("Saved outputs to:", MODEL_DIR)
print("Best model:", best_model_name)
print("Train rows used:", len(train_pdf), "Pos rate:", y.mean())
print("Model comparison:", MODEL_DIR / "all_models_results.csv")
print("Active learning queue:", MODEL_DIR / "active_learning_queue.csv")
