"""Classical autoencoder model for anomaly detection."""

from typing import Tuple

import torch
import torch.nn as nn


class TabularAutoencoder(nn.Module):
    """Tabular autoencoder for anomaly detection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        latent_dim: int = 4,
    ):
        """
        Initialize TabularAutoencoder.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension for encoder/decoder.
            latent_dim: Latent space dimension.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Latent tensor of shape (batch_size, latent_dim).
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to reconstruction.

        Args:
            z: Latent tensor of shape (batch_size, latent_dim).

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim).
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple of (reconstructed, latent) tensors.
        """
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample MSE reconstruction error (anomaly score).

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Anomaly scores (MSE per sample), shape (batch_size,).
        """
        reconstructed, _ = self.forward(x)
        mse = torch.mean((x - reconstructed) ** 2, dim=1)
        return mse
