"""API tests for linkage service endpoints."""

from __future__ import annotations

import numpy as np
import polars as pl
from fastapi.testclient import TestClient

import api


class DummyModel:
    """Simple deterministic model for test fallback."""

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return low-variance pseudo probabilities."""
        n = x.shape[0]
        probs = np.linspace(0.6, 0.9, n)
        return np.column_stack([1 - probs, probs])


def _ensure_runtime_state() -> None:
    """Inject deterministic in-memory state so tests never rely on disk artifacts.

    Always installs the DummyModel — the saved joblib may have been trained on
    a different feature count (e.g. before new features were added), so we must
    not let tests depend on whatever happens to be in outputs/.
    """
    api.app.state.providers_a = pl.DataFrame(
        {
            "npi": ["1234567890", "9999999999"],
            "first_name": ["John", "Alice"],
            "last_or_org_name": ["Smith", "Medical Group"],
            "state": ["TX", "CA"],
            "zip5": ["78701", "90210"],
            "street1": ["123 Main St", "1 Sunset Blvd"],
            "city": ["Austin", "Beverly Hills"],
            "credentials": ["MD", ""],
        }
    )
    api.app.state.idf_name = {"JOHN": 1.0, "SMITH": 1.0, "ALICE": 1.0}
    api.app.state.model = DummyModel()
    api.app.state.model_loaded = True
    api.app.state.model_name = "dummy_model_for_tests"


def test_health_returns_200() -> None:
    """Health endpoint should return status ok."""
    with TestClient(api.app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert "model_loaded" in payload


def test_match_pair_valid_payload_returns_matches() -> None:
    """Pair endpoint should return 200 and include matches key."""
    with TestClient(api.app) as client:
        _ensure_runtime_state()
        payload = {
            "profile_id": "profile-1",
            "first_name": "John",
            "last_name": "Smith",
            "state": "TX",
            "zip5": "78701",
            "street1": "123 Main St",
            "city": "Austin",
            "npi": None,
        }
        response = client.post("/match/pair", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert "matches" in body
        assert body["profile_id"] == "profile-1"


def test_match_pair_missing_required_fields_returns_422() -> None:
    """Pair endpoint should validate required fields."""
    with TestClient(api.app) as client:
        response = client.post(
            "/match/pair",
            json={
                "first_name": "John",
                "last_name": "Smith",
            },
        )
        assert response.status_code == 422


def test_stats_returns_expected_keys() -> None:
    """Stats endpoint should return required summary fields."""
    with TestClient(api.app) as client:
        response = client.get("/stats")
        assert response.status_code == 200
        body = response.json()
        assert "providers" in body
        assert "model_name" in body
        assert "pr_auc" in body
