"""Tests for Scenario 5: Dynamic Model Adaptation.

Covers the Page-Hinkley drift detector and the core adaptation
loop using synthetic data — no file I/O required.

The helper classes and functions are defined here directly since
scenario 5 is a test-level concern, not a production pipeline step.
"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------------------
# Drift detector
# -------------------------------------------------------------------

class PageHinkleyDetector:
    """One-sided Page-Hinkley test for upward shifts in error rate.

    After each new observation x_t:
      - Update the running mean
      - Accumulate x_t - mean - delta into the PH statistic
      - Flag drift when PH > lambda_
    """

    def __init__(self, delta: float = 0.005, lambda_: float = 0.05):
        self.delta = delta
        self.lambda_ = lambda_
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.sum = 0.0
        self.min_sum = 0.0
        self.mean = 0.0

    def update(self, x: float) -> bool:
        """Feed one error value. Returns True if drift is detected."""
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.sum += x - self.mean - self.delta
        self.min_sum = min(self.min_sum, self.sum)
        return (self.sum - self.min_sum) > self.lambda_


# -------------------------------------------------------------------
# Helpers shared across tests
# -------------------------------------------------------------------

def _make_model() -> Pipeline:
    """Fast logistic regression — cheap to retrain in tests."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   solver="lbfgs", random_state=42)),
    ])


def _score_batch(model: Pipeline, X: np.ndarray, y: np.ndarray,
                 thr: float = 0.5) -> dict:
    """Return a dict of five metrics for one batch."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= thr).astype(int)
    pr_auc = (float(average_precision_score(y, proba))
              if len(np.unique(y)) > 1 else float("nan"))
    return {
        "pr_auc": pr_auc,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "error_rate": float((pred != y).mean()),
    }


def _fit_simple_model():
    """Train a tiny model on linearly separable synthetic data."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 4))
    y = (X[:, 0] + rng.normal(0, 0.3, 200) > 0).astype(int)
    model = _make_model()
    model.fit(X, y)
    return model, X, y


def _make_synthetic_batches(n_batches: int = 4, rows_per_batch: int = 80):
    """Batches where the second half has a shifted positive rate (simulates drift)."""
    rng = np.random.default_rng(99)
    batches = []
    for i in range(n_batches):
        X = rng.standard_normal((rows_per_batch, 4))
        pos_rate = 0.2 if i < n_batches // 2 else 0.7
        y = (rng.random(rows_per_batch) < pos_rate).astype(int)
        batches.append((X, y))
    return batches


# -------------------------------------------------------------------
# Page-Hinkley detector tests
# -------------------------------------------------------------------

def test_page_hinkley_no_drift_on_stable_stream() -> None:
    """A steady low-error stream should not trigger a drift alert."""
    detector = PageHinkleyDetector(delta=0.005, lambda_=0.05)
    for _ in range(50):
        triggered = detector.update(0.05)
    assert not triggered


def test_page_hinkley_detects_sudden_increase() -> None:
    """A sudden jump from low to high error should be detected."""
    detector = PageHinkleyDetector(delta=0.005, lambda_=0.05)
    for _ in range(20):
        detector.update(0.05)
    detected = False
    for _ in range(30):
        if detector.update(0.90):
            detected = True
            break
    assert detected, "Drift should be flagged after error jumps from 0.05 to 0.90"


def test_page_hinkley_reset_clears_state() -> None:
    """After reset, the detector should behave as if freshly created."""
    detector = PageHinkleyDetector(delta=0.005, lambda_=0.05)
    for _ in range(20):
        detector.update(0.05)
    for _ in range(10):
        detector.update(0.95)

    detector.reset()

    triggered = detector.update(0.05)
    assert not triggered
    assert detector.n == 1


def test_page_hinkley_identical_to_fresh_after_reset() -> None:
    """A reset detector and a freshly created one should behave identically."""
    det_reset = PageHinkleyDetector()
    det_fresh = PageHinkleyDetector()

    for _ in range(15):
        det_reset.update(0.8)
    det_reset.reset()

    rng = np.random.default_rng(0)
    values = rng.uniform(0.0, 0.2, 20).tolist()
    assert [det_reset.update(v) for v in values] == [det_fresh.update(v) for v in values]


# -------------------------------------------------------------------
# _score_batch tests
# -------------------------------------------------------------------

def test_score_batch_returns_expected_keys() -> None:
    """_score_batch should always return the five expected metric keys."""
    model, X, y = _fit_simple_model()
    assert set(_score_batch(model, X, y).keys()) == {
        "pr_auc", "f1", "precision", "recall", "error_rate"
    }


def test_score_batch_values_in_range() -> None:
    """All metric values should be floats in [0, 1]."""
    model, X, y = _fit_simple_model()
    for key, val in _score_batch(model, X, y).items():
        assert isinstance(val, float), f"{key} should be float"
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"


def test_score_batch_near_perfect_model_has_low_error() -> None:
    """A near-perfectly separable dataset should yield error_rate close to 0."""
    rng = np.random.default_rng(7)
    X = np.hstack([rng.standard_normal((100, 3)), np.zeros((100, 1))])
    y = np.array([0] * 50 + [1] * 50)
    X[:50, 3] = -10.0
    X[50:, 3] = 10.0

    model = _make_model()
    model.fit(X, y)
    assert _score_batch(model, X, y)["error_rate"] < 0.05


# -------------------------------------------------------------------
# Adaptation loop tests
# -------------------------------------------------------------------

def test_adaptation_loop_runs_all_batches() -> None:
    """The detect-retrain loop should complete all batches without raising."""
    detector = PageHinkleyDetector(delta=0.005, lambda_=0.05)
    batches = _make_synthetic_batches()

    model = None
    history = []
    for batch_idx, (X_batch, y_batch) in enumerate(batches):
        if model is None:
            model = _make_model()
            model.fit(X_batch, y_batch)
            continue

        pre = _score_batch(model, X_batch, y_batch)
        errors = (np.round(model.predict_proba(X_batch)[:, 1]) != y_batch).astype(float)
        drift = any(detector.update(e) for e in errors)

        if drift:
            detector.reset()
            model = _make_model()
            model.fit(X_batch, y_batch)

        post = _score_batch(model, X_batch, y_batch)
        history.append({"batch": batch_idx, "drift": drift,
                        "pre_f1": pre["f1"], "post_f1": post["f1"]})

    assert len(history) == len(batches) - 1  # first batch is seed-only


def test_drift_triggers_on_flipped_labels() -> None:
    """Completely reversed labels should trigger drift within a few batches."""
    detector = PageHinkleyDetector(delta=0.005, lambda_=0.05)
    rng = np.random.default_rng(0)

    X_train = rng.standard_normal((200, 4))
    y_train = (X_train[:, 0] > 0).astype(int)
    model = _make_model()
    model.fit(X_train, y_train)

    any_drift = False
    for _ in range(5):
        X_new = rng.standard_normal((50, 4))
        y_new = (X_new[:, 0] <= 0).astype(int)  # flipped decision rule
        errors = (np.round(model.predict_proba(X_new)[:, 1]) != y_new).astype(float)
        for e in errors:
            if detector.update(e):
                any_drift = True
                break
        if any_drift:
            break

    assert any_drift, "Detector should flag drift when the label rule is completely reversed"
