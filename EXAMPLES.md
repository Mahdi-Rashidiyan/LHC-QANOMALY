# Example Usage Guide

This guide shows how to use lhc-qanomaly in different scenarios.

## Scenario 1: Command-Line Training and Scoring

The simplest approach using the CLI:

```bash
# Step 1: Download dataset
wget https://zenodo.org/records/4536377/files/events_anomalydetection_v2.features.h5 \
  -O data/events_anomalydetection_v2.features.h5

# Step 2: Train model (takes ~5-10 minutes depending on hardware)
lhc_qanomaly train --features data/events_anomalydetection_v2.features.h5

# Step 3: Score events
lhc_qanomaly score \
  --features data/events_anomalydetection_v2.features.h5 \
  --output scores.csv

# Step 4: Analyze scores
python -c "
import pandas as pd
df = pd.read_csv('scores.csv')
print(f'Total events: {len(df)}')
print(f'Background events: {(df.label==0).sum()}')
print(f'Signal events: {(df.label==1).sum()}')
print(f'Mean anomaly score: {df.anomaly_score.mean():.6f}')
print(f'Std anomaly score: {df.anomaly_score.std():.6f}')
print(f'Top 10 anomalies:')
print(df.nlargest(10, 'anomaly_score')[['label', 'anomaly_score']])
"
```

## Scenario 2: Programmatic API

Use lhc-qanomaly as a Python library:

```python
from lhc_qanomaly.data_loader import LHCOFeatureDataset
from lhc_qanomaly.train_classical import train_autoencoder
from lhc_qanomaly.infer_classical import load_model_and_scaler
import torch

# Train the model
print("Training...")
train_autoencoder(features_path="data/events_anomalydetection_v2.features.h5")

# Load trained model and scaler
print("Loading model...")
model, scaler, config = load_model_and_scaler(device="cuda")

# Load data
print("Loading data...")
dataset = LHCOFeatureDataset("data/events_anomalydetection_v2.features.h5")
dataset.fit_scaler()

# Score some events
features = dataset.get_features_only()[:100]
scaled = torch.from_numpy(scaler.transform(features)).float().cuda()
with torch.no_grad():
    scores = model.reconstruction_error(scaled)

print(f"Computed {len(scores)} anomaly scores")
print(f"Min: {scores.min():.6f}, Max: {scores.max():.6f}")
```

## Scenario 3: REST API

Start the API and score events via HTTP:

```bash
# Terminal 1: Start the API server
python -m uvicorn lhc_qanomaly.api:app --reload

# Terminal 2: Score events
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "features": [10.5, -5.2, 8.1, 2.3, 0.1, 0.2, 0.3, 11.0, -4.8, 8.5, 2.1, 0.12, 0.22, 0.32]
  }'

# Response: {"anomaly_score": 0.0234}
```

### Python Requests Example

```python
import requests
import json

url = "http://localhost:8000/score"
headers = {"Content-Type": "application/json"}

# Score a single event
event = {
    "features": [10.5, -5.2, 8.1, 2.3, 0.1, 0.2, 0.3, 11.0, -4.8, 8.5, 2.1, 0.12, 0.22, 0.32]
}

response = requests.post(url, json=event, headers=headers)
result = response.json()
print(f"Anomaly Score: {result['anomaly_score']}")

# Score multiple events
features_list = [
    [10.5, -5.2, 8.1, 2.3, 0.1, 0.2, 0.3, 11.0, -4.8, 8.5, 2.1, 0.12, 0.22, 0.32],
    [11.0, -4.8, 8.5, 2.1, 0.12, 0.22, 0.32, 10.2, -5.5, 8.3, 2.4, 0.11, 0.21, 0.31],
    # ... more events
]

scores = []
for features in features_list:
    response = requests.post(url, json={"features": features}, headers=headers)
    scores.append(response.json()["anomaly_score"])

print(f"Scored {len(scores)} events")
```

## Scenario 4: Batch Processing with Pandas

Process scores efficiently with pandas:

```python
import pandas as pd
from lhc_qanomaly.infer_classical import score_features_h5

# Score the entire dataset
score_features_h5(
    features_path="data/events_anomalydetection_v2.features.h5",
    output_csv="scores.csv"
)

# Load and analyze results
df = pd.read_csv("scores.csv")

# Statistics
print(df.describe())

# Separate background and signal
bg = df[df.label == 0]
sig = df[df.label == 1]

print(f"\nBackground anomaly score stats:")
print(f"  Mean: {bg.anomaly_score.mean():.6f}")
print(f"  Std:  {bg.anomaly_score.std():.6f}")
print(f"  Max:  {bg.anomaly_score.max():.6f}")

print(f"\nSignal anomaly score stats:")
print(f"  Mean: {sig.anomaly_score.mean():.6f}")
print(f"  Std:  {sig.anomaly_score.std():.6f}")
print(f"  Max:  {sig.anomaly_score.max():.6f}")

# Find outliers (anomalies)
threshold = bg.anomaly_score.mean() + 3 * bg.anomaly_score.std()
anomalies = df[df.anomaly_score > threshold]

print(f"\nFound {len(anomalies)} anomalies (>{threshold:.6f})")
print(f"  Background: {(anomalies.label == 0).sum()}")
print(f"  Signal: {(anomalies.label == 1).sum()}")
print(f"  Signal detection rate: {(anomalies.label == 1).sum() / len(sig) * 100:.2f}%")

# Visualization (requires matplotlib)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram
axes[0].hist(bg.anomaly_score, bins=100, alpha=0.7, label="Background")
axes[0].hist(sig.anomaly_score, bins=100, alpha=0.7, label="Signal")
axes[0].axvline(threshold, color='r', linestyle='--', label="Threshold")
axes[0].set_xlabel("Anomaly Score")
axes[0].set_ylabel("Frequency")
axes[0].legend()
axes[0].set_yscale('log')

# ROC-like plot
axes[1].scatter(bg.anomaly_score, [0]*len(bg), alpha=0.3, s=1, label="Background")
axes[1].scatter(sig.anomaly_score, [1]*len(sig), alpha=0.3, s=1, label="Signal")
axes[1].axvline(threshold, color='r', linestyle='--', label="Threshold")
axes[1].set_xlabel("Anomaly Score")
axes[1].set_ylabel("True Label")
axes[1].legend()

plt.tight_layout()
plt.savefig("anomaly_analysis.png", dpi=150)
print("\nVisualization saved to anomaly_analysis.png")
```

## Scenario 5: Custom Model Configuration

Adjust model hyperparameters:

```python
# Edit src/lhc_qanomaly/config.py

# Increase model capacity
MODEL_HIDDEN_DIM = 64      # Was 32
MODEL_LATENT_DIM = 8       # Was 4

# Faster training (fewer epochs)
TRAIN_EPOCHS = 20          # Was 50

# Smaller batches for limited memory
TRAIN_BATCH_SIZE = 64      # Was 128

# Then retrain:
# lhc_qanomaly train
```

## Scenario 6: Docker Deployment

Deploy the API in a container:

```bash
# Build
docker build -t lhc-qanomaly:latest .

# Run with volume for models
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data:ro \
  lhc-qanomaly:latest

# Or use Docker Compose
docker-compose up

# Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"features": [10.5, -5.2, 8.1, 2.3, 0.1, 0.2, 0.3, 11.0, -4.8, 8.5, 2.1, 0.12, 0.22, 0.32]}'
```

## Scenario 7: Monitoring and Metrics

Extract performance metrics:

```python
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd

df = pd.read_csv("scores.csv")

# Convert label to binary (0=background, 1=signal)
y_true = df.label.values
y_score = df.anomaly_score.values

# AUC Score
auc = roc_auc_score(y_true, y_score)
print(f"AUC-ROC: {auc:.4f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_score)

# Find optimal threshold
idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[idx]
print(f"Optimal threshold: {optimal_threshold:.6f}")

# Performance at optimal threshold
predictions = (y_score > optimal_threshold).astype(int)
tn = ((y_true == 0) & (predictions == 0)).sum()
fp = ((y_true == 0) & (predictions == 1)).sum()
fn = ((y_true == 1) & (predictions == 0)).sum()
tp = ((y_true == 1) & (predictions == 1)).sum()

tpr_opt = tp / (tp + fn)
fpr_opt = fp / (fp + tn)
print(f"\nAt optimal threshold:")
print(f"  True Positive Rate: {tpr_opt:.4f}")
print(f"  False Positive Rate: {fpr_opt:.4f}")
print(f"  Accuracy: {(tp + tn) / len(y_true):.4f}")
```

## Tips and Tricks

1. **GPU Training**: Set `USE_CUDA=1` before training for ~10x speedup.
2. **Memory**: If OOM, reduce `TRAIN_BATCH_SIZE` in config.py.
3. **Hyperparameter Tuning**: Try `MODEL_LATENT_DIM` values: 2, 4, 8, 16.
4. **Data Exploration**: Use `pandas.read_csv("scores.csv")` and `df.describe()`.
5. **API Health**: Always check `/health` before scoring.
6. **Logging**: Add `import logging; logging.basicConfig(level=DEBUG)` for debug output.

## Advanced: Custom Dataset

If you have a different HDF5 format:

```python
from lhc_qanomaly.data_loader import LHCOFeatureDataset

# Subclass to handle custom format
class CustomDataset(LHCOFeatureDataset):
    def _load_features(self):
        # Your custom loading logic
        pass

# Then use normally
dataset = CustomDataset("path/to/custom.h5")
```

## Questions?

- Check [README.md](README.md) for full documentation
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup
- Open an issue on GitHub for bugs or feature requests
