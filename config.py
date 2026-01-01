"""Configuration module for LHC Anomaly Detection."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_FEATURES_PATH = DATA_DIR / "events_anomalydetection_v2.features.h5"

# Feature list: high-level features for LHC Olympics 2020
FEATURES = [
    "pxj1", "pyj1", "pzj1", "mj1",
    "tau1j1", "tau2j1", "tau3j1",
    "pxj2", "pyj2", "pzj2", "mj2",
    "tau1j2", "tau2j2", "tau3j2",
]

# Model hyperparameters
MODEL_INPUT_DIM = len(FEATURES)
MODEL_HIDDEN_DIM = 32
MODEL_LATENT_DIM = 4

# Training hyperparameters
TRAIN_BATCH_SIZE = 512  # Larger batch for faster training
TRAIN_LEARNING_RATE = 1e-3
TRAIN_EPOCHS = 50  # Increased from 5 to 50 for better model convergence
TRAIN_VAL_SPLIT = 0.1
TRAIN_SEED = 42

# Device
DEVICE = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"

# HDF5 dataset
HDF5_EVENTS_GROUP = "events"  # Name of the HDF5 group containing event features

# Checkpoint
CHECKPOINT_PATH = MODELS_DIR / "autoencoder.pt"
