"""Command-line interface for LHC anomaly detection."""

from pathlib import Path

import click

from .config import DEFAULT_FEATURES_PATH
from .infer_classical import score_features_h5
from .train_classical import train_autoencoder


@click.group()
def cli():
    """LHC Anomaly Detection Platform."""
    pass


@cli.command()
@click.option(
    "--features",
    type=click.Path(exists=False),
    default=str(DEFAULT_FEATURES_PATH),
    help="Path to HDF5 features file.",
)
def train(features: str) -> None:
    """
    Train the autoencoder model.

    Expects the features file at the specified path and saves the trained
    model checkpoint to models/autoencoder.pt.
    """
    click.echo("Starting training pipeline...")
    try:
        train_autoencoder(features_path=features)
        click.echo("Training completed successfully!")
    except Exception as e:
        click.echo(f"Error during training: {e}", err=True)
        raise


@cli.command()
@click.option(
    "--features",
    type=click.Path(exists=True),
    required=True,
    help="Path to HDF5 features file.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="scores.csv",
    help="Output CSV file path.",
)
def score(features: str, output: str) -> None:
    """
    Score events in an HDF5 features file.

    Outputs a CSV with original features, labels, and anomaly_score column.
    """
    click.echo(f"Scoring features from {features}...")
    try:
        output_path = score_features_h5(features_path=features, output_csv=output)
        click.echo(f"Scoring completed! Results saved to {output_path}")
    except Exception as e:
        click.echo(f"Error during scoring: {e}", err=True)
        raise


if __name__ == "__main__":
    cli()
