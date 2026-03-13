# Healthcare Provider Record Linkage

End-to-end pipeline for linking healthcare provider records across:

- **Dataset A** (Medicare-style provider table)
- **Dataset B** (Open Payments-style provider profiles)
- **Dataset C** (PECOS-style provider records)

Pipeline order:

`ingest -> eda -> blocking -> features -> model -> stat_validation`

All generated artifacts are written to `outputs/`.

---

## 1) Prerequisites

- Python **3.9+**
- macOS/Linux shell (commands below use bash/zsh)

---

## 2) Setup

From project root:

```bash
python3 -m venv myenv
source myenv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi uvicorn pytest httpx
```

---

## 3) Data Placement

Download the data from: https://drive.google.com/file/d/13KBEpDKeCme_IpvBYtFvsYBkIApulwrb/view?usp=drive_link

Place raw source files under `data/` (as expected by `scripts/ingest.py`) using: 

```bash
unzip data.zip
```

---

## 4) Run Full Pipeline

Use the pipeline runner:

```bash
python main.py
```

What `main.py` does:

- runs all 6 scripts in order as subprocesses
- logs start/end/duration for each step
- stops on first failure with a clear error
- prints final pass/fail summary table

---

## 5) Run API

Start server:

```bash
python api.py
```

API starts on at `http://0.0.0.0:8000/docs`.

- `http://127.0.0.1:8000/docs`

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check; confirms model is loaded |
| `POST` | `/match/pair` | Return top-5 NPI matches for a single provider record |
| `POST` | `/match/batch` | Return top-5 matches for a list of provider records |
| `GET` | `/stats` | Provider counts, model PR-AUC, and active thresholds |

---

### Example: /health

```bash
curl http://0.0.0.0:8000/health
```

**Response:**

```json
{ "status": "ok", "model_loaded": true }
```

---

### POST /match/pair

Match a single provider record.  All fields except `profile_id` are optional — omit or pass `null`/`""` for unknown values.

```bash
curl -X POST http://0.0.0.0:8000/match/pair \
  -H "Content-Type: application/json" \
  -d '{
    "profile_id": "test-1",
    "first_name": "LOTIKA",
    "last_name": "SINGH",
    "state": "TX",
    "zip5": "77030",
    "street1": "1 BAYLOR PLZ",
    "city": "HOUSTON",
    "npi": null
  }'
```
---

### POST /match/batch

Match multiple records in one call.  Pass a JSON array; each element has the same fields as `/match/pair`.

```bash
curl -X POST http://0.0.0.0:8000/match/batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "profile_id": "batch-1",
      "first_name": "LOTIKA",
      "last_name": "SINGH",
      "state": "TX",
      "zip5": "77030",
      "street1": "1 BAYLOR PLZ",
      "city": "HOUSTON",
      "npi": null
    },
    {
      "profile_id": "batch-2",
      "first_name": "ALARICE",
      "last_name": "LOWE",
      "state": "CA",
      "zip5": "94305",
      "street1": "300 PASTEUR DR",
      "city": "STANFORD",
      "npi": null
    }
  ]'
```

---

### GET /stats

```bash
curl http://0.0.0.0:8000/stats
```

**Response** (example):

```json
{
  "providers": { "a": 60751, "b": 105203, "c": 2391071 },
  "model_name": "grad_boost",
  "pr_auc": 0.985,
  "metrics_available": true,
  "thresholds": {
    "best_f1": 0.38,
    "precision_target": 0.04
  }
}
```

---

> **Startup requirements:** the API loads `outputs/models/best_model.joblib`, `outputs/models/thresholds.json`, and `outputs/providers_a.parquet` at startup. If any model artifact is missing, matching endpoints return `503` with a descriptive message. Run the full pipeline (`python main.py`) or at minimum `scripts/model.py` to generate them.

---

## 6) Run Tests

```bash
pytest -q
```
---

## 7) Key Output Locations

- `outputs/providers_*.parquet` - normalized provider tables
- `outputs/eda/` - profiling and schema mapping outputs
- `outputs/candidates/` - blocking candidate sets and blocking diagnostics
- `outputs/features/` - pairwise feature matrices
- `outputs/models/` - trained model, thresholds, metrics, feature importance
- `outputs/stat_validation/` - CV, bootstrap CIs, significance and error buckets