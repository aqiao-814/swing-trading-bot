# Run everything through PYTHONPATH rather than the editable install.
#
# Why: this project lives under ~/Desktop, which is iCloud-synced. iCloud re-sets
# the macOS UF_HIDDEN flag on .pth files within seconds, and CPython's
# site.addpackage() silently skips hidden .pth files -- so `uv pip install -e .`
# appears to succeed and then does nothing. PYTHONPATH sidesteps the whole mess
# and works regardless of where the project lives.
#
# See docs/ENVIRONMENT.md for the full story.

PY := ./.venv/bin/python
export PYTHONPATH := src

.PHONY: help test lint fetch backtest compare clean unhide

help:
	@echo "make test      - run the test suite"
	@echo "make lint      - ruff check + format"
	@echo "make fetch     - download market data into data/"
	@echo "make compare   - run the baseline comparison"
	@echo "make unhide    - clear iCloud's UF_HIDDEN flag on .pth files (temporary)"

test:
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m ruff check src/ tests/ --fix
	$(PY) -m ruff format src/ tests/

fetch:
	$(PY) -m swingbot.cli fetch

compare:
	$(PY) -m swingbot.cli compare

# Temporary relief only: iCloud re-hides these within ~25s. Prefer PYTHONPATH.
unhide:
	chflags nohidden .venv/lib/python3.12/site-packages/*.pth || true
	@ls -lO .venv/lib/python3.12/site-packages/*.pth

clean:
	rm -rf artifacts/ .pytest_cache/ .ruff_cache/
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
