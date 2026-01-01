.PHONY: install install-dev test lint format clean docker-build docker-run help

PYTHON := python
PIP := pip
VENV := venv

help:
	@echo "LHC Anomaly Detection Platform - Makefile Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install         - Install package in production mode"
	@echo "  make install-dev     - Install package with dev dependencies"
	@echo "  make venv            - Create virtual environment"
	@echo ""
	@echo "Development:"
	@echo "  make test            - Run test suite"
	@echo "  make test-cov        - Run tests with coverage report"
	@echo "  make lint            - Lint code (ruff + black check)"
	@echo "  make format          - Format code (black + ruff --fix)"
	@echo "  make typecheck       - Type check with mypy"
	@echo ""
	@echo "Execution:"
	@echo "  make train           - Train model"
	@echo "  make score           - Score features (requires trained model)"
	@echo "  make api             - Start FastAPI server"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    - Build Docker image"
	@echo "  make docker-run      - Run Docker container"
	@echo "  make docker-compose  - Run with Docker Compose"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean           - Remove build artifacts and cache"
	@echo "  make clean-all       - Remove everything including venv"

venv:
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created. Activate with: source $(VENV)/bin/activate"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/lhc_qanomaly --cov-report=html --cov-report=term
	@echo "Coverage report generated. Open htmlcov/index.html to view."

lint:
	ruff check src tests
	black --check src tests

format:
	black src tests
	ruff check src tests --fix

typecheck:
	mypy src --ignore-missing-imports

train:
	lhc_qanomaly train --features data/events_anomalydetection_v2.features.h5

score:
	lhc_qanomaly score --features data/events_anomalydetection_v2.features.h5 --output scores.csv

api:
	$(PYTHON) -m uvicorn lhc_qanomaly.api:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t lhc-qanomaly:latest .

docker-run:
	docker run -p 8000:8000 \
		-v $$(pwd)/models:/app/models \
		-v $$(pwd)/data:/app/data:ro \
		lhc-qanomaly:latest

docker-compose:
	docker-compose up

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name build -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name dist -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage coverage.xml 2>/dev/null || true

clean-all: clean
	rm -rf $(VENV) 2>/dev/null || true
	@echo "Everything cleaned!"
