.PHONY: setup-dev setup-notebook-dev install-hooks lint format test clean

setup-dev:
	python -m pip install -e ".[dev]"
	pre-commit install

setup-notebook-dev:
	python -m pip install -e ".[dev,notebook]"
	pre-commit install

install-hooks:
	pre-commit install

lint:
	pre-commit run --all-files

format:
	ruff check --fix .
	ruff format .

test:
	pytest

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
