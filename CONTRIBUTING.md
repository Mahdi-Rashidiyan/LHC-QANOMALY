# Development Guide

## Setup Development Environment

```bash
# Clone and enter directory
git clone https://github.com/yourusername/lhc-qanomaly.git
cd lhc-qanomaly

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Code Style

We follow PEP 8 with some customizations:

```bash
# Format with black
black src tests

# Lint with ruff
ruff check src tests --fix

# Type check (optional, some errors ignored)
mypy src --ignore-missing-imports
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/lhc_qanomaly --cov-report=html

# Run specific test
pytest tests/test_pipeline.py::TestDataLoader::test_load_features -v

# Run tests matching pattern
pytest tests/ -k "test_train" -v
```

## Project Architecture

### Core Components

- **`config.py`**: Centralized configuration. All hyperparameters here.
- **`data_loader.py`**: HDF5 loading and preprocessing. Handles scaling.
- **`model_classical.py`**: PyTorch model definition. No training logic.
- **`train_classical.py`**: Training pipeline. Stateless function.
- **`infer_classical.py`**: Inference and scoring. Checkpoint loading.
- **`cli.py`**: Command-line interface using Click.
- **`api.py`**: FastAPI service with lifespan management.

### Design Patterns

1. **Configuration**: All magic numbers in `config.py`, not scattered through code.
2. **Separation of Concerns**: Data, model, training, inference are separate modules.
3. **Functional Training**: `train_autoencoder()` is a pure function, no class state.
4. **Checkpoint Format**: Always save `{state_dict, config, scaler_state}`.
5. **Error Handling**: Explicit exceptions in data loaders; FastAPI handles API errors.

## Adding Features

### Adding a New Training Algorithm

1. Create `src/lhc_qanomaly/train_quantum.py`:

```python
from .config import DEFAULT_FEATURES_PATH
from .model_quantum import QuantumAutoencoder

def train_quantum_autoencoder(features_path=None):
    # Implementation here
    pass
```

2. Update `src/lhc_qanomaly/cli.py`:

```python
@cli.command()
@click.option("--features", ...)
def train_quantum(features):
    train_quantum_autoencoder(features_path=features)
```

3. Add tests to `tests/test_pipeline.py`.

### Adding API Endpoints

1. Edit `src/lhc_qanomaly/api.py`:

```python
@app.post("/score_batch")
async def score_batch(events: list[EventFeatures]):
    # Implementation
    pass
```

2. Add test in `tests/test_pipeline.py::TestAPI`.

## Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use IPython for interactive debugging
pip install ipython
from lhc_qanomaly.data_loader import LHCOFeatureDataset
dataset = LHCOFeatureDataset("path/to/features.h5")
# Explore interactively
```

## Performance Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile here
from lhc_qanomaly.train_classical import train_autoencoder
train_autoencoder()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

## Memory Profiling

```bash
pip install memory-profiler

python -m memory_profiler your_script.py
```

## Documentation

Docstrings follow NumPy style:

```python
def my_function(x: int, y: str) -> tuple[int, str]:
    """
    Short description.

    Longer description if needed.

    Parameters
    ----------
    x : int
        Description of x.
    y : str
        Description of y.

    Returns
    -------
    tuple[int, str]
        Description of return value.

    Examples
    --------
    >>> my_function(1, "hello")
    (1, 'hello')
    """
    pass
```

## Release Process

1. Update version in `pyproject.toml` and `src/lhc_qanomaly/__init__.py`
2. Add entry to CHANGELOG.md
3. Create git tag: `git tag v0.2.0`
4. Push: `git push --tags`
5. Build: `python -m build`
6. Upload to PyPI: `twine upload dist/*`

## Dependencies

### Core
- torch: Deep learning
- numpy/pandas: Data handling
- scikit-learn: Preprocessing
- h5py: HDF5 I/O
- click: CLI
- fastapi/uvicorn: API
- pydantic: Validation

### Dev
- pytest: Testing
- black: Formatting
- ruff: Linting
- mypy: Type checking

## Common Issues

**Import errors**: Make sure you've installed in editable mode: `pip install -e .`

**CUDA issues**: Disable GPU with `USE_CUDA=0` or install torch CPU version.

**HDF5 errors**: Install system dependencies: `apt-get install libhdf5-dev`

**Test failures**: Clean and reinstall: `pip install --force-reinstall -e .`

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes and test: `pytest tests/`
4. Format and lint: `black src tests && ruff check src tests`
5. Commit with clear message: `git commit -m "Add my feature"`
6. Push and create pull request
7. Wait for CI to pass and review

## Resources

- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Click Docs](https://click.palletsprojects.com/)
- [Pytest Docs](https://docs.pytest.org/)
- [LHC Olympics](https://lhcolympics.github.io/)
