# Tests

This folder contains all unit, integration, and scenario-specific tests for the Healthcare Provider Record Linkage project.

---

## Running Tests

```bash
# Run all tests
pytest -q

# Run only fast tests (exclude slow integration tests)
pytest -q -m "not slow"

# Run only slow integration tests
pytest -q -m slow
```

---

## File Overview

| File | Type | Scenarios Covered |
|---|---|---|
| `conftest.py` | Infrastructure | All |
| `test_features.py` | Unit | 1, 2 |
| `test_blocking.py` | Unit | 1, 2, 4 |
| `test_api.py` | Unit | 3 |
| `test_pipeline.py` | Integration | 3, 4 |
| `test_dynamic_model_adaptation.py` | Unit | 5 |

---

## Scenario Mapping

### Scenario 1 — High-Quality Data Linkage
> Medicare data with complete NPI information. Focus: sophisticated similarity measures.

- **`test_features.py`** — validates all similarity functions (`jaro_winkler`, `lev_sim`, `tfidf_cosine`, `char3_cosine`, `token_jaccard`, `soundex`) and the full feature vector schema (`features_row`, `FEATURE_COLUMNS`)
- **`test_blocking.py`** — `test_exact_npi_blocking_zero_false_negatives` confirms that exact NPI blocking recovers all true matches with zero false negatives

---

### Scenario 2 — Dirty Data Challenge
> Open Payments records without NPI identifiers. Focus: robust fuzzy matching and data cleaning.

- **`test_features.py`** — phonetic (`soundex`) and edit-distance (`lev_sim`) tests directly target name variation and OCR-error tolerance
- **`test_blocking.py`** — `test_canopy_blocking_threshold` validates threshold-based blocking for records where exact NPI matching is unavailable; `test_sorted_neighborhood_window_2` confirms deterministic windowed candidate generation

---

### Scenario 3 — Multi-source Integration
> All three datasets combined with temporal matching. Focus: unified provider profiles.

- **`test_api.py`** — tests the `/health`, `/match/pair`, and `/stats` endpoints of the linkage service, including valid payloads, missing-field validation (422), and response schema
- **`test_pipeline.py`** — integration tests that run `main.py --steps ingest` end-to-end and assert the expected parquet artifacts are produced

---

### Scenario 4 — Large-Scale Statistical Analysis
> Full annual Medicare dataset (~24M records). Focus: algorithmic efficiency and scalable blocking.

- **`test_blocking.py`** — `test_sorted_neighborhood_window_2` and `test_canopy_blocking_threshold` cover the two scalable blocking strategies used to reduce the candidate space before pairwise comparison
- **`test_pipeline.py`** — validates that the pipeline runner completes successfully and produces output artifacts, covering the ingest step's correctness at scale

---

### Scenario 5 — Dynamic Model Adaptation
> Streaming provider updates with concept drift. Focus: online learning and statistical change detection.

- **`test_dynamic_model_adaptation.py`** — dedicated file covering:
  - `PageHinkleyDetector` (no-drift on stable streams, detection of sudden error jumps, reset behaviour)
  - `_score_batch` metric helper (key presence, value ranges, near-perfect model sanity check)
  - Adaptation loop (all batches complete without error, drift triggers retraining on flipped labels)

---

## Shared Infrastructure

### `conftest.py`
Prepends the project root and `scripts/` directory to `sys.path` so all test files can import production modules without installing the package. No test logic lives here.

---

## Markers

| Marker | Meaning |
|---|---|
| *(none)* | Fast unit test, always runs |
| `@pytest.mark.slow` | Integration test that invokes subprocesses or requires sample data files under `data/` |
