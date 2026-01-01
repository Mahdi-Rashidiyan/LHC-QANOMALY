"""FastAPI service for real-time anomaly scoring."""

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import DEVICE, FEATURES
from .infer_classical import load_model_and_scaler

# Global state
_model = None
_scaler = None
_device = None

# Load model on startup
def _load_model():
    """Load model and scaler on app start."""
    global _model, _scaler, _device

    print("Loading model and scaler...")
    _device = torch.device(DEVICE)
    try:
        _model, _scaler, _ = load_model_and_scaler(device=DEVICE)
        print("Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"Warning: Could not load model: {e}")
        print("API will not be functional without a trained model.")


app = FastAPI(
    title="LHC Anomaly Detection API",
    description="Real-time anomaly scoring for LHC events.",
    version="0.1.0",
)

# Load model on startup
_load_model()


class EventFeatures(BaseModel):
    """Pydantic model for event features."""

    features: list[float] = Field(
        ...,
        description=f"List of {len(FEATURES)} feature values.",
        example=[10.0, -5.0, 8.0, 2.5] + [0.1] * (len(FEATURES) - 4),
    )


class AnomalyScoreResponse(BaseModel):
    """Response model for anomaly score."""

    anomaly_score: float = Field(..., description="Anomaly score (MSE).")


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    if _model is None:
        return {"status": "degraded", "message": "Model not loaded"}
    return {"status": "ok"}


@app.post("/score", response_model=AnomalyScoreResponse)
async def score_event(event: EventFeatures) -> AnomalyScoreResponse:
    """
    Score a single event.

    Args:
        event: Event features.

    Returns:
        Anomaly score.
    """
    if _model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Start training first."
        )

    # Validate feature count
    if len(event.features) != len(FEATURES):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURES)} features, got {len(event.features)}",
        )

    try:
        # Convert to numpy and reshape
        features_array = np.array(event.features, dtype=np.float32).reshape(1, -1)

        # Scale
        scaled_features = _scaler.transform(features_array)

        # Score
        features_tensor = torch.from_numpy(scaled_features).float().to(_device)
        with torch.no_grad():
            anomaly_score = _model.reconstruction_error(features_tensor)
        score = float(anomaly_score.cpu().numpy()[0])

        return AnomalyScoreResponse(anomaly_score=score)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")
