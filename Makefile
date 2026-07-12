VENV := .venv/bin
PYTHON := $(VENV)/python
PYTEST := $(VENV)/pytest
RUFF := $(VENV)/ruff
MYPY := $(VENV)/mypy

.PHONY: install test test-fast test-cov lint fmt typecheck check graph clean

install:
	python3 -m venv .venv
	$(VENV)/pip install -q -r requirements.txt
	$(VENV)/pip install -q -e .

test:
	$(PYTEST) tests/ -v --no-cov

test-fast:
	$(PYTEST) tests/ -v --no-cov -m "not slow"

test-cov:
	$(PYTEST) tests/ --cov --cov-report=term-missing

lint:
	$(RUFF) check .

fmt:
	$(RUFF) format .

typecheck:
	$(MYPY) core engine

check: lint typecheck test

graph:
	claude /graphify .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -rf .pytest_cache htmlcov .coverage
