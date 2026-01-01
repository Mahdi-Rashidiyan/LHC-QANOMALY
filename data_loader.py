"""Data loading module for LHC anomaly detection."""

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, HDF5_EVENTS_GROUP


class LHCOFeatureDataset:
    """Loads and manages LHC Olympics features dataset."""

    def __init__(self, features_path: str | Path):
        """
        Initialize dataset loader.

        Args:
            features_path: Path to HDF5 features file.
        """
        self.features_path = Path(features_path)
        self.features_df: pd.DataFrame | None = None
        self.scaler: StandardScaler | None = None
        self._load_features()

    def _load_features(self) -> None:
        """Load features from HDF5 file into DataFrame."""
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")

        # Try to load as pandas DataFrame first (most common format)
        try:
            self.features_df = pd.read_hdf(self.features_path, "df")
            # Ensure label column exists as int
            if "label" in self.features_df.columns:
                self.features_df["label"] = self.features_df["label"].astype(int)
        except (KeyError, ValueError):
            # Fallback: try to load as raw HDF5 array
            with h5py.File(self.features_path, "r") as f:
                # Try different possible group names
                if HDF5_EVENTS_GROUP in f:
                    events_data = f[HDF5_EVENTS_GROUP][:]
                else:
                    # Use first available dataset
                    first_key = list(f.keys())[0]
                    events_data = f[first_key][:]

            # Create DataFrame with feature names + label
            n_features = len(FEATURES)
            feature_data = events_data[:, :n_features]
            label_data = events_data[:, n_features].astype(int)

            self.features_df = pd.DataFrame(feature_data, columns=FEATURES)
            self.features_df["label"] = label_data

    def get_features_df(self) -> pd.DataFrame:
        """Get the full features DataFrame."""
        if self.features_df is None:
            raise RuntimeError("Features not loaded")
        return self.features_df.copy()

    def get_features_only(self) -> np.ndarray:
        """Get only feature columns (no label) as numpy array."""
        if self.features_df is None:
            raise RuntimeError("Features not loaded")
        return self.features_df[FEATURES].values

    def get_labels(self) -> np.ndarray:
        """Get label column as numpy array."""
        if self.features_df is None:
            raise RuntimeError("Features not loaded")
        return self.features_df["label"].values

    def fit_scaler(self, features: np.ndarray | None = None) -> StandardScaler:
        """
        Fit StandardScaler on features.

        Args:
            features: Optional array to fit on. If None, uses all features.

        Returns:
            Fitted StandardScaler.
        """
        if features is None:
            features = self.get_features_only()

        self.scaler = StandardScaler()
        self.scaler.fit(features)
        return self.scaler

    def transform_features(self, features: np.ndarray | None = None) -> np.ndarray:
        """
        Transform features using fitted scaler.

        Args:
            features: Optional array to transform. If None, uses all features.

        Returns:
            Scaled features.
        """
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit_scaler() first.")

        if features is None:
            features = self.get_features_only()

        return self.scaler.transform(features)

    def get_scaled_tensor(
        self, features: np.ndarray | None = None, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Get scaled features as PyTorch tensor.

        Args:
            features: Optional array. If None, uses all features.
            dtype: PyTorch dtype.

        Returns:
            Scaled features as tensor.
        """
        scaled = self.transform_features(features)
        return torch.from_numpy(scaled).to(dtype)

    def get_background_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get only background (label=0) features.

        Returns:
            Tuple of (features, indices).
        """
        labels = self.get_labels()
        mask = labels == 0
        features = self.get_features_only()
        return features[mask], np.where(mask)[0]

    def get_scaler_state(self) -> dict:
        """Get scaler state for saving to checkpoint."""
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted.")
        return {
            "mean": self.scaler.mean_.copy(),
            "scale": self.scaler.scale_.copy(),
            "var": self.scaler.var_.copy(),
        }

    def set_scaler_state(self, state: dict) -> None:
        """Restore scaler from saved state."""
        self.scaler = StandardScaler()
        self.scaler.mean_ = state["mean"]
        self.scaler.scale_ = state["scale"]
        self.scaler.var_ = state["var"]
        self.scaler.n_features_in_ = len(state["mean"])
