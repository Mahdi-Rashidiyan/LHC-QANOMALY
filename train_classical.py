"""Training pipeline for classical autoencoder."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    CHECKPOINT_PATH,
    DEFAULT_FEATURES_PATH,
    DEVICE,
    MODEL_HIDDEN_DIM,
    MODEL_INPUT_DIM,
    MODEL_LATENT_DIM,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LEARNING_RATE,
    TRAIN_SEED,
    TRAIN_VAL_SPLIT,
)
from .data_loader import LHCOFeatureDataset
from .model_classical import TabularAutoencoder


def train_autoencoder(features_path: Optional[str | Path] = None) -> None:
    """
    Train the tabular autoencoder on background events.

    Args:
        features_path: Path to HDF5 features file. Defaults to config value.
    """
    # Set seeds for reproducibility
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)

    # Load dataset
    if features_path is None:
        features_path = DEFAULT_FEATURES_PATH
    features_path = Path(features_path)

    print(f"Loading features from {features_path}...")
    dataset = LHCOFeatureDataset(features_path)

    # Get background events only
    print("Extracting background events (label=0)...")
    bg_features, _ = dataset.get_background_features()
    print(f"Found {len(bg_features)} background events")

    # Fit scaler on training data
    print("Fitting feature scaler...")
    dataset.fit_scaler(bg_features)
    scaled_features = dataset.transform_features(bg_features)

    # Train/val split
    n_samples = len(scaled_features)
    n_train = int(n_samples * (1 - TRAIN_VAL_SPLIT))
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_features = scaled_features[train_indices]
    val_features = scaled_features[val_indices]

    print(f"Train samples: {len(train_features)}, Val samples: {len(val_features)}")

    # Create DataLoader
    train_tensor = torch.from_numpy(train_features).float()
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    # Model, optimizer, loss
    device = torch.device(DEVICE)
    model = TabularAutoencoder(
        input_dim=MODEL_INPUT_DIM,
        hidden_dim=MODEL_HIDDEN_DIM,
        latent_dim=MODEL_LATENT_DIM,
    )
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=TRAIN_LEARNING_RATE)
    criterion = nn.MSELoss()

    # Training loop
    print(f"Training for {TRAIN_EPOCHS} epochs on device {device}...")
    for epoch in range(TRAIN_EPOCHS):
        train_loss = 0.0
        for batch in train_loader:
            x = batch[0].to(device)

            # Forward
            reconstructed, _ = model(x)
            loss = criterion(x, reconstructed)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_features)

        # Validation
        model.eval()
        with torch.no_grad():
            val_tensor = torch.from_numpy(val_features).float().to(device)
            val_reconstructed, _ = model(val_tensor)
            val_loss = criterion(val_tensor, val_reconstructed).item()
        model.train()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d}/{TRAIN_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

    print("Training complete.")

    # Save checkpoint
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": MODEL_INPUT_DIM,
            "hidden_dim": MODEL_HIDDEN_DIM,
            "latent_dim": MODEL_LATENT_DIM,
        },
        "scaler_state": dataset.get_scaler_state(),
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Checkpoint saved to {CHECKPOINT_PATH}")
