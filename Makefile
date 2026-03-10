.PHONY: install train eval test report clean

## Install the package in editable mode
install:
	pip install -e ".[dev]"

## Run a full training session with default config
train:
	python scripts/train_default.py --config configs/default.yaml

## Run training with fast config (fewer steps, for debugging)
train-fast:
	python scripts/train_default.py --config configs/train_fast.yaml

## Evaluate the latest checkpoint
eval:
	python scripts/evaluate_checkpoint.py --config configs/eval.yaml

## Compare all baselines against the DQN agent
baselines:
	python scripts/compare_baselines.py --config configs/eval.yaml

## Generate performance report from last evaluation run
report:
	python scripts/generate_report.py

## Run all unit tests
test:
	pytest tests/ -v

## Run tests with coverage
test-cov:
	pytest tests/ -v --cov=src/market_making_agent --cov-report=term-missing

## Remove generated outputs (keep directory structure)
clean:
	find outputs/figures -name "*.png" -delete
	find outputs/checkpoints -name "*.pt" -delete
	find outputs/logs -name "*.csv" -o -name "*.json" | xargs rm -f
	find outputs/reports -name "*.txt" -o -name "*.md" | xargs rm -f
	@echo "Cleaned generated outputs."

## Show help
help:
	@grep -E '^##' Makefile | sed 's/## //'
