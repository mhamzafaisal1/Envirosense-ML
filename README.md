# EnviroSense ML API

Crop recommendation and field assessment from soil and climate readings. This powers the live demo on the EnviroSense site and the mobile app.

**Live API:** `https://<your-render-url>` · interactive docs at `/docs`

## Results (reproducible: `python scripts/train.py`)

22 crops, 2,200 samples, 7 features (N, P, K, temperature, humidity, pH, rainfall). 80/20 stratified hold-out, with 5-fold CV on the training split.

| Model | 5-fold CV accuracy | Hold-out accuracy |
|---|---|---|
| Random Forest (200 trees) | 99.4% ± 0.5% | 99.5% |
| Logistic Regression (baseline) | 96.8% ± 0.7% | 97.3% |

Full per-class F1 and permutation importances live in `models/metrics.json`, served at `GET /metrics`.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| POST | `/predict` | Top-3 crops (RF) plus the logistic regression baseline, and a field assessment against the crop's observed ranges |
| GET | `/metrics` | Evaluation results from the last training run |
| GET | `/crops` | Supported crops and their 10th–90th percentile condition ranges |
| GET | `/health` | Liveness |

```bash
curl -X POST $API/predict -H 'content-type: application/json' \
  -d '{"N":90,"P":42,"K":43,"temperature":20.9,"humidity":82,"ph":6.5,"rainfall":203}'
```

Pass an optional `"target_crop": "maize"` to assess a field for a crop you've already planted.

## Run locally

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
python scripts/train.py        # optional; the models are committed
uvicorn app.main:app --reload  # http://localhost:8000/docs
pytest -q
```

## What changed since the paper (v1 → v2)

The paper's models (`legacy/`) were trained on a rice-only dataset in which most rows, and the yield, planting-time and next-crop labels, were synthetic. The "field condition" label was produced by the same threshold rules the model was meant to learn. v2 replaces all of this with the full public dataset and a real crop-recommendation task, evaluated on held-out data. Field assessment is now an explicit, explainable comparison against each crop's observed ranges instead of a classifier trained to reproduce its own rules.

## Data

Crop Recommendation Dataset by Atharva Ingle (Kaggle), built from Indian agricultural data sources.
