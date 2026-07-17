"""Append-only ledger of every configuration ever evaluated on the data.

The Deflated Sharpe Ratio is only honest if ``n_trials`` counts every model,
feature set, horizon, and hyperparameter you have ever scored -- including
the abandoned ones. Humans undercount that number in their own favour, every
time, so the runner writes this file and people do not: one JSON line per
evaluation, appended automatically, and ``n_trials`` is just the line count.
Deleting or editing this file invalidates every DSR computed after it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def log_trial(path: str | Path, record: dict) -> int:
    """Append one evaluated configuration; returns total trials on record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"utc": datetime.now(UTC).isoformat(), **record}, sort_keys=True)
    with path.open("a") as f:
        f.write(line + "\n")
    return n_trials(path)


def n_trials(path: str | Path) -> int:
    """How many configurations have ever been evaluated. Feed this to DSR."""
    path = Path(path)
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for line in f if line.strip())
