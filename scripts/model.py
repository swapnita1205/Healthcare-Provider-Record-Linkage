from pathlib import Path
import json
import numpy as np
import pandas as pd
import polars as pl

from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
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

from matching_utils import FEATURE_COLUMNS, safe_predict_proba

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
N_ITER_RF = 12

# Active learning
UNLABELED_SAMPLE = 400_000
ACTIVE_QUERY_N = 200

# Calibration — always calibrate every model; tree models use isotonic, linear use sigmoid
CALIBRATE_MODELS = True

# Org-vs-person negative oversampling multiplier on top of hard negatives
ORG_PERSON_NEG_MULTIPLIER = 2

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

# -----------------------------
# Loading schema
# -----------------------------
ba_head = pl.read_parquet(PATH_BA, n_rows=5)
bc_head = pl.read_parquet(PATH_BC, n_rows=5)

ba_cols = set(ba_head.columns)
bc_cols = set(bc_head.columns)

NON_FEATURES = {"profile_id", "pass", "block_key", "label_weak_npi", "npi_a", "npi_c"}
# Use the canonical order from matching_utils.FEATURE_COLUMNS so that the
# feature indices the model learns match exactly what api.py passes at inference.
available = (ba_cols & bc_cols) - NON_FEATURES
feature_cols = [c for c in FEATURE_COLUMNS if c in available]

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
        lf.with_row_index("_row_id")
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

# Org-vs-person conflict negatives: the model ignores org_vs_person_conflict (zero perm importance)
# because these pairs are underrepresented.  Oversample them explicitly so the model learns
# that a person-vs-org mismatch is a strong negative signal.
if "org_vs_person_conflict" in feature_cols:
    org_person_pool = neg.filter(pl.col("org_vs_person_conflict") == 1)
    org_person_cnt = org_person_pool.select(pl.len()).collect().item()
    org_person_take = min(int(hard_target_total * ORG_PERSON_NEG_MULTIPLIER), org_person_cnt)
    org_person_sample = lazy_sample_n(org_person_pool, org_person_take, RANDOM_STATE + 3)
    print(f"Org-vs-person negatives added: {org_person_take} (pool size: {org_person_cnt})")
else:
    org_person_sample = neg.limit(0)
    print("org_vs_person_conflict not in features; skipping oversampling")

train_lf = pl.concat([pos, hard_sample, rand_sample, org_person_sample], how="vertical")
train_df = train_lf.collect(engine="streaming")
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

# Random Forest — no scaling needed (trees are scale-invariant)
rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=2,
)

# Gradient Boosting
gb = HistGradientBoostingClassifier(random_state=RANDOM_STATE)

param_space_lr = {"clf__C": np.logspace(-3, 2, 15)}
param_space_rf = {
    "n_estimators": [100, 200, 400],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 5, 20],
    "max_features": ["sqrt", "log2", 0.5],
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

print("=== Random Forest HPO ===")
rf_search = fit_search(rf, param_space_rf, N_ITER_RF, "rf", n_jobs=2)

print("=== GB HPO ===")
gb_search = fit_search(gb, param_space_gb, N_ITER_GB, "gb", n_jobs=2)

searches = {
    "logreg": lr_search,
    "random_forest": rf_search,
    "grad_boost": gb_search,
}


def maybe_calibrate(name: str, estimator):
    if not CALIBRATE_MODELS:
        return estimator
    # LR: Platt sigmoid; tree models: isotonic regression (more flexible, better for GBM/RF)
    method = "sigmoid" if name == "logreg" else "isotonic"
    return CalibratedClassifierCV(estimator, method=method, cv=3)

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
        "calibrated": bool(CALIBRATE_MODELS),
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

# Final model: CalibratedClassifierCV(cv=3) trained on ALL labeled data.
# The calibration layer is fitted via 3-fold CV internally, avoiding leakage.
best_model = maybe_calibrate(best_model_name, best_est)
best_model.fit(X, y)

joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

# Thresholds: evaluate on the held-out test split using a model trained only on X_train.
# This avoids leakage since the final model was fitted on X (includes X_test).
best_est_for_thr = searches[best_model_name].best_estimator_
thr_model = maybe_calibrate(best_model_name, best_est_for_thr)
thr_model.fit(X_train, y_train)
best_scores = safe_predict_proba(thr_model, X_test)
best_thr = tune_threshold(y_test, best_scores, target_precision=0.95)

with open(MODEL_DIR / "thresholds.json", "w") as f:
    json.dump({"best_model": best_model_name, "thresholds": best_thr}, f, indent=2)

metrics_out = {
    "train_rows": int(len(train_pdf)),
    "pos_rate_train": float(y.mean()),
    "best_model": best_model_name,
    "calibrated": bool(CALIBRATE_MODELS),
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
# LR coefficients — extract from a fresh LR trained on all data (not calibrated wrapper)
lr_best = lr_search.best_estimator_
lr_best.fit(X, y)

coef_model = lr_best
coef = coef_model.named_steps["clf"].coef_.flatten()
pd.DataFrame({"feature": feature_cols, "coef": coef}).sort_values("coef", ascending=False)\
  .to_csv(MODEL_DIR / "feature_importance_lr.csv", index=False)

# Permutation importance: use thr_model (trained on X_train only) evaluated on X_test
perm = permutation_importance(
    thr_model, X_test, y_test,
    n_repeats=5, random_state=RANDOM_STATE, scoring="average_precision"
)
pd.DataFrame({
    "feature": feature_cols,
    "perm_importance_mean": perm.importances_mean,
    "perm_importance_std": perm.importances_std,
}).sort_values("perm_importance_mean", ascending=False)\
  .to_csv(MODEL_DIR / "feature_importance_best_model.csv", index=False)
  
# Active learning: start from a small seed, then at each round query the pairs
# the model is least confident about. NPI matches proxy for human review labels.

AL_ITERS = 3
AL_QUERY_PER_ITER = 100

rng_al = np.random.default_rng(RANDOM_STATE + 10)
all_idx = np.arange(len(X))
pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]

# Seed: 15% of positives + 3× that many negatives
seed_n = max(30, int(0.15 * len(pos_idx)))
seed_pos = rng_al.choice(pos_idx, size=min(seed_n, len(pos_idx)), replace=False)
seed_neg = rng_al.choice(neg_idx, size=min(seed_n * 3, len(neg_idx)), replace=False)
al_labeled = np.concatenate([seed_pos, seed_neg])
al_pool = np.setdiff1d(all_idx, al_labeled)

# Use a fast LogisticRegression for the AL loop (not the final best model)
al_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
])

al_history = []
print("\n=== Active Learning Loop ===")
for al_round in range(AL_ITERS):
    al_clf.fit(X[al_labeled], y[al_labeled])
    al_scores = al_clf.predict_proba(X_test)[:, 1]
    al_pr_auc = float(average_precision_score(y_test, al_scores))
    al_f1 = float(f1_score(y_test, (al_scores >= 0.5).astype(int), zero_division=0))

    al_history.append({
        "round": al_round,
        "labeled_size": int(len(al_labeled)),
        "n_positives": int((y[al_labeled] == 1).sum()),
        "val_pr_auc": al_pr_auc,
        "val_f1": al_f1,
    })
    print(f"  round {al_round}: labeled={len(al_labeled)}, PR-AUC={al_pr_auc:.4f}, F1={al_f1:.4f}")

    if len(al_pool) == 0:
        break

    # Query the most uncertain pairs from the unlabeled pool
    pool_scores = al_clf.predict_proba(X[al_pool])[:, 1]
    uncertainty = np.abs(pool_scores - 0.5)
    n_query = min(AL_QUERY_PER_ITER, len(al_pool))
    queried = al_pool[np.argsort(uncertainty)[:n_query]]

    al_labeled = np.concatenate([al_labeled, queried])
    al_pool = np.setdiff1d(al_pool, queried)

pd.DataFrame(al_history).to_csv(MODEL_DIR / "active_learning_curve.csv", index=False)
print("Active learning curve saved:", MODEL_DIR / "active_learning_curve.csv")

# Also export the current uncertainty queue from the full unlabeled pool
# (pairs without any NPI-based label — candidates for human review)
unlabeled = df_scan.filter(pl.col("label_weak_npi").is_null()).drop("label_weak_npi")
unl_sample = lazy_sample_n(unlabeled, UNLABELED_SAMPLE, RANDOM_STATE).collect(engine="streaming")
unl_pdf = unl_sample.to_pandas()
X_unl = unl_pdf[feature_cols].values

unl_scores = safe_predict_proba(best_model, X_unl)
uncertainty_unl = np.abs(unl_scores - 0.5)

idx_unc = np.argsort(uncertainty_unl)[: int(ACTIVE_QUERY_N * 0.6)]
idx_hi  = np.argsort(-unl_scores)[: int(ACTIVE_QUERY_N * 0.2)]
idx_lo  = np.argsort(unl_scores)[: int(ACTIVE_QUERY_N * 0.2)]
idx_q   = np.unique(np.concatenate([idx_unc, idx_hi, idx_lo]))[:ACTIVE_QUERY_N]

queue = unl_pdf.iloc[idx_q].copy()
queue["score"] = unl_scores[idx_q]
queue["uncertainty"] = uncertainty_unl[idx_q]
queue[["profile_id", "candidate_npi", "pair_type", "pass", "block_key", "score", "uncertainty"]].to_csv(
    MODEL_DIR / "active_learning_queue.csv", index=False
)

print("\nSaved outputs to:", MODEL_DIR)
print("Best model:", best_model_name)
print("Train rows used:", len(train_pdf), "Pos rate:", y.mean())
print("Model comparison:", MODEL_DIR / "all_models_results.csv")
print("Active learning queue:", MODEL_DIR / "active_learning_queue.csv")
