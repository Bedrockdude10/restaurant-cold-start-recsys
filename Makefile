.PHONY: install test lint preprocess splits train evaluate clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/ scripts/

preprocess:
	python scripts/preprocess_yelp.py --config configs/default.yaml

splits:
	python scripts/create_splits.py --config configs/default.yaml

train:
	python scripts/train.py --config configs/default.yaml

evaluate:
	python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt

clean:
	rm -rf data/processed/* data/splits/* checkpoints/* wandb/
	find . -type d -name __pycache__ -exec rm -rf {} +
