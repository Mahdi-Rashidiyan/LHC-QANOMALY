# Quick Start Guide

Get up and running with lhc-qanomaly in 5 minutes.

## 1. Install

```bash
cd lhc-qanomaly
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

## 2. Download Dataset

Visit https://zenodo.org/records/4536377 and download `events_anomalydetection_v2.features.h5` into the `data/` folder.

```bash
# Or use wget
wget https://zenodo.org/records/4536377/files/events_anomalydetection_v2.features.h5 \
  -O data/events_anomalydetection_v2.features.h5
```

## 3. Train

```bash
lhc_qanomaly train
```

Expected: Creates `models/autoencoder.pt` (checkpoint with model + scaler).

## 4. Score

```bash
lhc_qanomaly score \
  --features data/events_anomalydetection_v2.features.h5 \
  --output scores.csv
```

Expected: Creates `scores.csv` with anomaly_score column.

## 5. Run API

```bash
python -m uvicorn lhc_qanomaly.api:app --reload
```

Visit http://localhost:8000/docs for interactive docs.

Test with:
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"features": [10.5, -5.2, 8.1, 2.3, 0.1, 0.2, 0.3, 11.0, -4.8, 8.5, 2.1, 0.12, 0.22, 0.32]}'
```

## 6. Run Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

## Troubleshooting

**Model not found during scoring**: Run `lhc_qanomaly train` first.

**CUDA out of memory**: Edit `src/lhc_qanomaly/config.py` and reduce `TRAIN_BATCH_SIZE`.

**API health check fails**: Make sure model is trained before starting the API.
