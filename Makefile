.PHONY: install dev lint format test jobs

install:
	pip install -e ".[train]"

dev:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest -q

jobs:     ## render all Slurm scripts into slurm/generated/
	python3 slurm/generate.py --all
