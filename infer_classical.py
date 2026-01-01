"""Inference pipeline for classical autoencoder."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import CHECKPOINT_PATH, DEVICE, FEATURES
from .data_loader import LHCOFeatureDataset
from .model_classical import TabularAutoencoder


def load_model_and_scaler(device: str = DEVICE):
    """
    Load trained model and scaler from checkpoint.

    Args:
        device: Device to load model on ('cpu' or 'cuda').

    Returns:
        Tuple of (model, scaler, model_config).
    """
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    # Reconstruct model
    model_config = checkpoint["model_config"]
    model = TabularAutoencoder(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        latent_dim=model_config["latent_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Reconstruct scaler
    dataset = LHCOFeatureDataset.__new__(LHCOFeatureDataset)
    dataset.features_df = None
    dataset.scaler = None
    dataset.set_scaler_state(checkpoint["scaler_state"])

    return model, dataset.scaler, model_config


def score_features_h5(
    features_path: str | Path,
    output_csv: str | Path = "scores.csv",
    device: str = DEVICE,
) -> str:
    """
    Score all events in HDF5 file and save results to CSV.

    Args:
        features_path: Path to HDF5 features file.
        output_csv: Output CSV path.
        device: Device to run on.

    Returns:
        Path to output CSV file.
    """
    features_path = Path(features_path)
    output_csv = Path(output_csv)

    # Load model and scaler
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model, scaler, _ = load_model_and_scaler(device=device)

    # Load dataset
    print(f"Loading features from {features_path}...")
    dataset = LHCOFeatureDataset(features_path)
    dataset.scaler = scaler

    # Get features and labels
    features = dataset.get_features_only()
    labels = dataset.get_labels()
    scaled_features = dataset.transform_features(features)

    # Score
    print("Computing anomaly scores...")
    device_obj = torch.device(device)
    features_tensor = torch.from_numpy(scaled_features).float().to(device_obj)

    with torch.no_grad():
        anomaly_scores = model.reconstruction_error(features_tensor)
    anomaly_scores = anomaly_scores.cpu().numpy()

    # Create output DataFrame
    df = pd.DataFrame(features, columns=FEATURES)
    df["label"] = labels
    df["anomaly_score"] = anomaly_scores

    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Scores saved to {output_csv}")

    return str(output_csv)
