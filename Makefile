# PYTHONPATH instead of an editable install: iCloud-synced folders silently
# hide .pth files (CPython skips them), and PYTHONPATH works everywhere.
PY := ./.venv/bin/python
export PYTHONPATH := src

.PHONY: help test lint invest fetch compare clean

help:
	@echo "make test     - run the test suite"
	@echo "make lint     - ruff check + format"
	@echo "make invest   - run the daily paper-trading loop locally"
	@echo "make fetch    - download market data into data/"
	@echo "make compare  - baseline comparison (research harness)"

test:
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m ruff check src/ tests/ --fix
	$(PY) -m ruff format src/ tests/

invest:
	$(PY) -m swingbot.cli invest --capital 100000

fetch:
	$(PY) -m swingbot.cli fetch

compare:
	$(PY) -m swingbot.cli compare

clean:
	rm -rf artifacts/ .pytest_cache/ .ruff_cache/
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
