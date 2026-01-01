"""Test suite for LHC anomaly detection pipeline."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from lhc_qanomaly.config import FEATURES, MODEL_HIDDEN_DIM, MODEL_LATENT_DIM
from lhc_qanomaly.data_loader import LHCOFeatureDataset
from lhc_qanomaly.model_classical import TabularAutoencoder
from lhc_qanomaly.train_classical import train_autoencoder
from lhc_qanomaly.infer_classical import score_features_h5, load_model_and_scaler


@pytest.fixture
def synthetic_features_file():
    """Create a temporary HDF5 file with synthetic features."""
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / "test_features.h5"

        # Generate synthetic data
        n_samples = 200
        n_features = len(FEATURES)

        # Background events (label=0): mostly low variance
        bg_data = np.random.normal(0, 1.0, (150, n_features))
        bg_labels = np.zeros(150)

        # Signal events (label=1): higher variance
        sig_data = np.random.normal(0, 2.0, (50, n_features))
        sig_labels = np.ones(50)

        # Combine and save
        data = np.vstack([bg_data, sig_data])
        labels = np.hstack([bg_labels, sig_labels])
        full_data = np.hstack([data, labels.reshape(-1, 1)])

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("events", data=full_data)

        yield h5_path


class TestDataLoader:
    """Test data loading functionality."""

    def test_load_features(self, synthetic_features_file):
        """Test loading features from HDF5 file."""
        dataset = LHCOFeatureDataset(synthetic_features_file)

        # Check shape
        df = dataset.get_features_df()
        assert df.shape[0] == 200
        assert list(df.columns) == FEATURES + ["label"]

        # Check data types
        assert df["label"].dtype == int
        assert all(df[col].dtype == np.float64 for col in FEATURES)

    def test_get_features_only(self, synthetic_features_file):
        """Test getting features without labels."""
        dataset = LHCOFeatureDataset(synthetic_features_file)

        features = dataset.get_features_only()
        assert features.shape == (200, len(FEATURES))

    def test_get_labels(self, synthetic_features_file):
        """Test getting labels."""
        dataset = LHCOFeatureDataset(synthetic_features_file)

        labels = dataset.get_labels()
        assert len(labels) == 200
        assert np.sum(labels == 0) == 150  # Background
        assert np.sum(labels == 1) == 50  # Signal

    def test_fit_scaler(self, synthetic_features_file):
        """Test fitting StandardScaler."""
        dataset = LHCOFeatureDataset(synthetic_features_file)

        scaler = dataset.fit_scaler()
        assert scaler is not None
        assert scaler.mean_ is not None
        assert scaler.scale_ is not None

    def test_transform_features(self, synthetic_features_file):
        """Test feature scaling."""
        dataset = LHCOFeatureDataset(synthetic_features_file)

        dataset.fit_scaler()
        scaled = dataset.transform_features()

        # Check that scaling worked
        assert np.abs(np.mean(scaled)) < 0.1
        assert np.abs(np.std(scaled) - 1.0) < 0.1

    def test_get_scaled_tensor(self, synthetic_features_file):
        """Test getting scaled tensor for PyTorch."""
        dataset = LHCOFeatureDataset(synthetic_features_file)

        dataset.fit_scaler()
        tensor = dataset.get_scaled_tensor()

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (200, len(FEATURES))
        assert tensor.dtype == torch.float32

    def test_get_background_features(self, synthetic_features_file):
        """Test extracting background events."""
        dataset = LHCOFeatureDataset(synthetic_features_file)

        bg_features, indices = dataset.get_background_features()

        assert bg_features.shape == (150, len(FEATURES))
        assert len(indices) == 150

    def test_scaler_state_save_restore(self, synthetic_features_file):
        """Test saving and restoring scaler state."""
        dataset = LHCOFeatureDataset(synthetic_features_file)
        dataset.fit_scaler()

        state = dataset.get_scaler_state()
        assert "mean" in state
        assert "scale" in state
        assert "var" in state

        # Restore to new dataset
        dataset2 = LHCOFeatureDataset.__new__(LHCOFeatureDataset)
        dataset2.set_scaler_state(state)

        # Check they produce same results
        features = dataset.get_features_only()
        scaled1 = dataset.transform_features(features)
        scaled2 = dataset2.transform_features(features)

        np.testing.assert_allclose(scaled1, scaled2)


class TestModel:
    """Test model architecture."""

    def test_model_initialization(self):
        """Test creating the autoencoder model."""
        model = TabularAutoencoder(
            input_dim=len(FEATURES),
            hidden_dim=MODEL_HIDDEN_DIM,
            latent_dim=MODEL_LATENT_DIM,
        )

        assert model.input_dim == len(FEATURES)
        assert model.hidden_dim == MODEL_HIDDEN_DIM
        assert model.latent_dim == MODEL_LATENT_DIM

    def test_encode(self):
        """Test encoder forward pass."""
        model = TabularAutoencoder(
            input_dim=len(FEATURES),
            hidden_dim=MODEL_HIDDEN_DIM,
            latent_dim=MODEL_LATENT_DIM,
        )

        x = torch.randn(32, len(FEATURES))
        z = model.encode(x)

        assert z.shape == (32, MODEL_LATENT_DIM)

    def test_decode(self):
        """Test decoder forward pass."""
        model = TabularAutoencoder(
            input_dim=len(FEATURES),
            hidden_dim=MODEL_HIDDEN_DIM,
            latent_dim=MODEL_LATENT_DIM,
        )

        z = torch.randn(32, MODEL_LATENT_DIM)
        reconstructed = model.decode(z)

        assert reconstructed.shape == (32, len(FEATURES))

    def test_forward(self):
        """Test full forward pass."""
        model = TabularAutoencoder(
            input_dim=len(FEATURES),
            hidden_dim=MODEL_HIDDEN_DIM,
            latent_dim=MODEL_LATENT_DIM,
        )

        x = torch.randn(32, len(FEATURES))
        reconstructed, z = model.forward(x)

        assert reconstructed.shape == (32, len(FEATURES))
        assert z.shape == (32, MODEL_LATENT_DIM)

    def test_reconstruction_error(self):
        """Test anomaly score computation."""
        model = TabularAutoencoder(
            input_dim=len(FEATURES),
            hidden_dim=MODEL_HIDDEN_DIM,
            latent_dim=MODEL_LATENT_DIM,
        )

        x = torch.randn(32, len(FEATURES))
        errors = model.reconstruction_error(x)

        assert errors.shape == (32,)
        assert torch.all(errors >= 0)  # MSE is non-negative


class TestTraining:
    """Test training pipeline."""

    def test_train_autoencoder(self, synthetic_features_file):
        """Test full training pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Monkey-patch config to use temp directory
            import lhc_qanomaly.train_classical as train_module

            original_checkpoint_path = train_module.CHECKPOINT_PATH
            original_device = train_module.DEVICE

            try:
                train_module.CHECKPOINT_PATH = Path(tmpdir) / "test_autoencoder.pt"
                train_module.DEVICE = "cpu"

                # Train
                train_autoencoder(features_path=synthetic_features_file)

                # Check checkpoint exists
                assert train_module.CHECKPOINT_PATH.exists()

                # Check checkpoint contents
                checkpoint = torch.load(
                    train_module.CHECKPOINT_PATH,
                    map_location="cpu",
                    weights_only=False,
                )
                assert "model_state_dict" in checkpoint
                assert "model_config" in checkpoint
                assert "scaler_state" in checkpoint

            finally:
                train_module.CHECKPOINT_PATH = original_checkpoint_path
                train_module.DEVICE = original_device


class TestInference:
    """Test inference pipeline."""

    def test_load_model_and_scaler(self, synthetic_features_file):
        """Test loading trained model and scaler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import lhc_qanomaly.train_classical as train_module
            import lhc_qanomaly.infer_classical as infer_module

            original_checkpoint_path_train = train_module.CHECKPOINT_PATH
            original_checkpoint_path_infer = infer_module.CHECKPOINT_PATH
            checkpoint_path = Path(tmpdir) / "test_autoencoder.pt"

            try:
                train_module.CHECKPOINT_PATH = checkpoint_path
                infer_module.CHECKPOINT_PATH = checkpoint_path

                # Train
                train_autoencoder(features_path=synthetic_features_file)

                # Load
                model, scaler, config = load_model_and_scaler(device="cpu")

                assert isinstance(model, TabularAutoencoder)
                assert scaler is not None
                assert config["input_dim"] == len(FEATURES)

            finally:
                train_module.CHECKPOINT_PATH = original_checkpoint_path_train
                infer_module.CHECKPOINT_PATH = original_checkpoint_path_infer

    def test_score_features_h5(self, synthetic_features_file):
        """Test scoring HDF5 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import lhc_qanomaly.train_classical as train_module
            import lhc_qanomaly.infer_classical as infer_module

            original_checkpoint_path_train = train_module.CHECKPOINT_PATH
            original_checkpoint_path_infer = infer_module.CHECKPOINT_PATH
            checkpoint_path = Path(tmpdir) / "test_autoencoder.pt"
            output_csv = Path(tmpdir) / "scores.csv"

            try:
                train_module.CHECKPOINT_PATH = checkpoint_path
                infer_module.CHECKPOINT_PATH = checkpoint_path

                # Train
                train_autoencoder(features_path=synthetic_features_file)

                # Score
                result_path = score_features_h5(
                    features_path=synthetic_features_file,
                    output_csv=output_csv,
                    device="cpu",
                )

                # Check output
                assert Path(result_path).exists()

                df = pd.read_csv(result_path)
                assert "anomaly_score" in df.columns
                assert len(df) == 200
                assert all(df["anomaly_score"] >= 0)

            finally:
                train_module.CHECKPOINT_PATH = original_checkpoint_path_train
                infer_module.CHECKPOINT_PATH = original_checkpoint_path_infer


class TestCLI:
    """Test command-line interface."""

    def test_cli_train_command(self, synthetic_features_file):
        """Test CLI train command."""
        from click.testing import CliRunner

        from lhc_qanomaly.cli import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            import lhc_qanomaly.train_classical as train_module

            original_checkpoint_path = train_module.CHECKPOINT_PATH

            try:
                train_module.CHECKPOINT_PATH = Path(tmpdir) / "test_autoencoder.pt"

                runner = CliRunner()
                result = runner.invoke(
                    cli,
                    ["train", "--features", str(synthetic_features_file)],
                )

                assert result.exit_code == 0
                assert train_module.CHECKPOINT_PATH.exists()

            finally:
                train_module.CHECKPOINT_PATH = original_checkpoint_path

    def test_cli_score_command(self, synthetic_features_file):
        """Test CLI score command."""
        from click.testing import CliRunner

        from lhc_qanomaly.cli import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            import lhc_qanomaly.train_classical as train_module
            import lhc_qanomaly.infer_classical as infer_module

            original_checkpoint_path_train = train_module.CHECKPOINT_PATH
            original_checkpoint_path_infer = infer_module.CHECKPOINT_PATH
            checkpoint_path = Path(tmpdir) / "test_autoencoder.pt"
            output_csv = Path(tmpdir) / "scores.csv"

            try:
                train_module.CHECKPOINT_PATH = checkpoint_path
                infer_module.CHECKPOINT_PATH = checkpoint_path

                # Train first
                runner = CliRunner()
                result = runner.invoke(
                    cli,
                    ["train", "--features", str(synthetic_features_file)],
                )
                assert result.exit_code == 0

                # Score
                result = runner.invoke(
                    cli,
                    [
                        "score",
                        "--features",
                        str(synthetic_features_file),
                        "--output",
                        str(output_csv),
                    ],
                )

                assert result.exit_code == 0
                assert output_csv.exists()

            finally:
                train_module.CHECKPOINT_PATH = original_checkpoint_path_train
                infer_module.CHECKPOINT_PATH = original_checkpoint_path_infer


class TestAPI:
    """Test FastAPI service."""

    def test_api_health_check(self):
        """Test health check endpoint."""
        from fastapi.testclient import TestClient

        from lhc_qanomaly.api import app

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert "status" in response.json()

    def test_api_score_without_model(self):
        """Test score endpoint with model loaded."""
        from fastapi.testclient import TestClient

        from lhc_qanomaly.api import app

        client = TestClient(app)
        payload = {"features": [1.0] * len(FEATURES)}

        response = client.post("/score", json=payload)

        # Model is loaded, should return 200 OK
        assert response.status_code == 200
        data = response.json()
        assert "anomaly_score" in data

    def test_api_score_wrong_dimensions(self):
        """Test score endpoint with wrong feature dimensions."""
        from fastapi.testclient import TestClient

        from lhc_qanomaly.api import app

        client = TestClient(app)
        payload = {"features": [1.0] * (len(FEATURES) - 1)}  # Wrong size

        response = client.post("/score", json=payload)

        assert response.status_code == 400
