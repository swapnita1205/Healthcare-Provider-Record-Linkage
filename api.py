"""FastAPI service for provider record linkage inference."""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import polars as pl
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from matching_utils import FEATURE_COLUMNS, build_idf_from_texts, features_row, safe_predict_proba, zip3

OUT_DIR = Path("outputs")
MODEL_PATH = OUT_DIR / "models" / "best_model.joblib"
METRICS_PATH = OUT_DIR / "models" / "metrics.json"
PROVIDERS_A_PATH = OUT_DIR / "providers_a.parquet"
PROVIDERS_B_PATH = OUT_DIR / "providers_b.parquet"
PROVIDERS_C_PATH = OUT_DIR / "providers_c.parquet"


class ProviderRecord(BaseModel):
    """Incoming provider profile for matching."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(..., min_length=1)
    first_name: str = ""
    last_name: str = ""
    state: str = ""
    zip5: str = ""
    street1: str = ""
    city: str = ""
    npi: Optional[str] = None


class MatchCandidate(BaseModel):
    """One scored candidate match."""

    profile_id: str
    candidate_npi: str
    candidate_first_name: Optional[str]
    candidate_last_or_org_name: Optional[str]
    candidate_state: Optional[str]
    candidate_zip5: Optional[str]
    similarity_score: float
    match_probability: float


class PairMatchResponse(BaseModel):
    """Pair-match response for one request profile."""

    profile_id: str
    matches: list[MatchCandidate]


class BatchMatchResponse(BaseModel):
    """Batch-match response."""

    results: list[PairMatchResponse]


def _safe_read_json(path: Path) -> dict[str, Any]:
    """Read JSON if present, else return empty dictionary."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_feature_row(
    incoming: ProviderRecord, cand: dict[str, Any], idf_name: dict[str, float]
) -> dict[str, Union[float, int]]:
    """Build one feature row matching training schema."""
    row = {
        "b_first": incoming.first_name or "",
        "b_last": incoming.last_name or "",
        "b_suffix": "",
        "b_cred": "",
        "b_street1": incoming.street1 or "",
        "b_city": incoming.city or "",
        "b_state": incoming.state or "",
        "b_zip5": incoming.zip5 or "",
        "x_first": cand.get("first_name") or "",
        "x_last": cand.get("last_or_org_name") or "",
        "x_suffix": "",
        "x_cred": cand.get("credentials") or "",
        "x_street1": cand.get("street1") or "",
        "x_city": cand.get("city") or "",
        "x_state": cand.get("state") or "",
        "x_zip5": cand.get("zip5") or "",
    }
    return features_row(row, idf_name=idf_name)


def _block_candidates(providers_a: pl.DataFrame, record: ProviderRecord, default_limit: int = 1000) -> pl.DataFrame:
    """Apply state + zip3 blocking to reduce candidate set."""
    if providers_a.height == 0:
        return providers_a

    state = (record.state or "").strip().upper()
    z3 = zip3(record.zip5)

    lf = providers_a.lazy()
    if state and z3:
        filtered = lf.filter(
            (pl.col("state").cast(pl.Utf8).str.to_uppercase() == state)
            & pl.col("zip5").cast(pl.Utf8).fill_null("").str.starts_with(z3)
        ).collect()
        if filtered.height > 0:
            return filtered

    if state:
        filtered = lf.filter(pl.col("state").cast(pl.Utf8).str.to_uppercase() == state).collect()
        if filtered.height > 0:
            return filtered.head(default_limit)

    return providers_a.head(default_limit)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model/data artifacts during app startup."""
    app.state.model = None
    app.state.model_loaded = False
    app.state.model_name = None
    app.state.metrics = {}
    app.state.providers_a = pl.DataFrame()
    app.state.providers_b_count = 0
    app.state.providers_c_count = 0
    app.state.idf_name = {}

    if PROVIDERS_A_PATH.exists():
        app.state.providers_a = pl.read_parquet(PROVIDERS_A_PATH)
        names = (
            app.state.providers_a.select(
                pl.concat_str(
                    [
                        pl.col("first_name").cast(pl.Utf8).fill_null(""),
                        pl.col("last_or_org_name").cast(pl.Utf8).fill_null(""),
                    ],
                    separator=" ",
                ).alias("name")
            )["name"]
            .fill_null("")
            .to_list()
        )
        app.state.idf_name = build_idf_from_texts(names)

    if PROVIDERS_B_PATH.exists():
        app.state.providers_b_count = pl.scan_parquet(PROVIDERS_B_PATH).select(pl.len()).collect().item()
    if PROVIDERS_C_PATH.exists():
        app.state.providers_c_count = pl.scan_parquet(PROVIDERS_C_PATH).select(pl.len()).collect().item()

    app.state.metrics = _safe_read_json(METRICS_PATH)
    app.state.model_name = app.state.metrics.get("best_model")

    if MODEL_PATH.exists():
        app.state.model = joblib.load(MODEL_PATH)
        app.state.model_loaded = True

    yield


app = FastAPI(title="Healthcare Provider Linkage API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": bool(app.state.model_loaded)}


@app.post("/match/pair", response_model=PairMatchResponse)
def match_pair(record: ProviderRecord) -> PairMatchResponse:
    """Return top-5 candidate matches for one incoming provider record."""
    if app.state.providers_a.height == 0:
        raise HTTPException(
            status_code=503,
            detail="providers_a.parquet is missing. Run scripts/ingest.py first.",
        )
    if not app.state.model_loaded or app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact outputs/models/best_model.joblib not found. Run scripts/model.py first.",
        )

    candidates_df = _block_candidates(app.state.providers_a, record)
    if candidates_df.height == 0:
        return PairMatchResponse(profile_id=record.profile_id, matches=[])

    cand_rows = candidates_df.to_dicts()
    feature_rows: list[dict[str, Union[float, int]]] = [
        _build_feature_row(record, cand, app.state.idf_name) for cand in cand_rows
    ]

    x_mat = np.array([[float(feat.get(c, 0.0)) for c in FEATURE_COLUMNS] for feat in feature_rows], dtype=float)
    probs = safe_predict_proba(app.state.model, x_mat)

    matches: list[MatchCandidate] = []
    for cand, feat, prob in zip(cand_rows, feature_rows, probs):
        similarity_score = float(feat.get("sim_jw_fullname", 0.0))
        matches.append(
            MatchCandidate(
                profile_id=record.profile_id,
                candidate_npi=str(cand.get("npi") or ""),
                candidate_first_name=cand.get("first_name"),
                candidate_last_or_org_name=cand.get("last_or_org_name"),
                candidate_state=cand.get("state"),
                candidate_zip5=cand.get("zip5"),
                similarity_score=similarity_score,
                match_probability=float(prob),
            )
        )

    top5 = sorted(matches, key=lambda m: m.match_probability, reverse=True)[:5]
    return PairMatchResponse(profile_id=record.profile_id, matches=top5)


@app.post("/match/batch", response_model=BatchMatchResponse)
def match_batch(records: list[ProviderRecord]) -> BatchMatchResponse:
    """Return top matches for each incoming provider record."""
    results = [match_pair(record) for record in records]
    return BatchMatchResponse(results=results)


@app.get("/stats")
def stats() -> dict[str, Any]:
    """Return data and model summary metrics."""
    metrics = app.state.metrics or {}
    holdout = metrics.get("holdout", {})
    best_f1 = holdout.get("best_f1", {})
    pr_auc = best_f1.get("pr_auc")

    providers_a_count = app.state.providers_a.height if app.state.providers_a is not None else 0
    return {
        "providers": {
            "a": int(providers_a_count),
            "b": int(app.state.providers_b_count),
            "c": int(app.state.providers_c_count),
        },
        "model_name": app.state.model_name,
        "pr_auc": pr_auc,
        "metrics_available": bool(metrics),
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
