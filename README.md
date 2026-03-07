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
pip install fastapi uvicorn pytest
```

> `fastapi`, `uvicorn`, and `pytest` are used by the API and tests.

---

## 3) Data Placement

Download the data from: https://drive.google.com/file/d/13KBEpDKeCme_IpvBYtFvsYBkIApulwrb/view?usp=drive_link

Place raw source files under `data/` (as expected by `scripts/ingest.py`) using: 

```bash
unzip data.zip
```

Then run:

```bash
python scripts/ingest.py
```

This should create:

- `outputs/providers_a.parquet`
- `outputs/providers_b.parquet`
- `outputs/providers_c.parquet`

---

## 4) Run Full Pipeline (Recommended)

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

## 5) Run Partial Pipeline

Examples:

```bash
# Skip EDA
python main.py --skip-eda

# Skip model + stat validation
python main.py --skip-model

# Run specific subset (in canonical order)
python main.py --steps ingest,blocking,features
```

---

## 6) Run API

Start server:

```bash
python api.py
```

API starts on:

- `http://0.0.0.0:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

### API Endpoints

- `GET /health`
- `POST /match/pair`
- `POST /match/batch`
- `GET /stats`

### Example: /health

```bash
curl http://127.0.0.1:8000/health
```

### Example: /match/pair

```bash
curl -X POST http://127.0.0.1:8000/match/pair \
  -H "Content-Type: application/json" \
  -d '{
    "profile_id": "demo-1",
    "first_name": "John",
    "last_name": "Smith",
    "state": "TX",
    "zip5": "78701",
    "street1": "123 Main St",
    "city": "Austin",
    "npi": null
  }'
```

> The API loads `outputs/models/best_model.joblib` and `outputs/providers_a.parquet` at startup.  
> If model artifacts are missing, matching endpoints return a clear `503` message.

---

## 7) Run Tests

```bash
pytest -q
```

Run only fast tests:

```bash
pytest -q -m "not slow"
```

Run only integration/slow tests:

```bash
pytest -q -m slow
```

---

## 8) Key Output Locations

- `outputs/providers_*.parquet` - normalized provider tables
- `outputs/eda/` - profiling and schema mapping outputs
- `outputs/candidates/` - blocking candidate sets and blocking diagnostics
- `outputs/features/` - pairwise feature matrices
- `outputs/models/` - trained model, thresholds, metrics, feature importance
- `outputs/stat_validation/` - CV, bootstrap CIs, significance and error buckets

---

## 9) Typical First Run

```bash
python main.py
python api.py
```

Then open:

- `http://127.0.0.1:8000/docs`

---

## 10) Troubleshooting

- **`ModuleNotFoundError`**  
  Activate your venv and reinstall dependencies.

- **`best_model.joblib` missing**  
  Run `python main.py` (or at least through the `model` step in `scripts/model.py`).

- **`providers_a.parquet` missing**  
  Run `python scripts/ingest.py` (or `python main.py --steps ingest`).

- **No matches returned from API**  
  Check that incoming `state`/`zip5` are present and well-formatted; blocking uses state + zip5 first.

