"""Export the paper portfolio's full state as one JSON blob for the live site.

Reads ``artifacts/paper/portfolio`` (the engine's source of truth) and writes
``site/data.json``. Runs after every ``swingbot invest`` in the nightly
workflow; the static dashboard fetches the JSON and renders it client-side.

Everything exported is SIMULATED capital.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
PORTFOLIO = Path(os.environ.get("SWINGBOT_PORTFOLIO", ROOT / "artifacts" / "paper" / "portfolio"))
OUT = Path(os.environ.get("SWINGBOT_SITE", ROOT / "site")) / "data.json"
# Newest fills kept in data.json. The full history stays in the parquet store.
TRADE_LIMIT = 1000


def _records(name: str, sort_by: str | None = None, descending: bool = False) -> list[dict]:
    path = PORTFOLIO / f"{name}.parquet"
    if not path.exists():
        return []
    df = pl.read_parquet(path)
    if df.is_empty():
        return []
    if sort_by and sort_by in df.columns:
        df = df.sort(sort_by, descending=descending)
    # Format timestamps and null out NaN/inf so json.dumps never chokes.
    for col in df.columns:
        if df[col].dtype == pl.Date:
            df = df.with_columns(pl.col(col).dt.strftime("%Y-%m-%d"))
        elif df[col].dtype == pl.Datetime:
            df = df.with_columns(pl.col(col).dt.strftime("%Y-%m-%d %H:%M"))
        elif df[col].dtype in (pl.Float32, pl.Float64):
            df = df.with_columns(pl.when(pl.col(col).is_finite()).then(pl.col(col)).alias(col))
    return df.to_dicts()


def main() -> None:
    state_path = PORTFOLIO / "state.json"
    if not state_path.exists():
        sys.exit("no state.json yet -- run `swingbot invest` first")
    state = json.loads(state_path.read_text())

    ledger = _records("ledger", sort_by="ts")
    trades = _records("trades", sort_by="ts", descending=True)
    positions = _records("positions", sort_by="symbol")
    decisions = _records("decisions", sort_by="ts")
    learning = _records("learning", sort_by="ts")

    # Win rate over the FULL history, before the list is truncated below --
    # otherwise the headline stat silently becomes "win rate over the last
    # TRADE_LIMIT fills" as history grows.
    closed = [t for t in trades if t.get("action") == "sell" and t.get("realized_pnl") is not None]
    wins = sum(1 for t in closed if t["realized_pnl"] > 0)
    n_trades = len(trades)
    # An uncapped book on 30m bars fills hundreds of times a session (~180/day
    # measured on a 670-name universe). Exporting every fill forever would grow
    # data.json into the tens of MB, and the publish step rewrites the whole
    # file into a git commit every 30 minutes. The page only renders the newest
    # 200; keep a working buffer and let the parquet history be the archive.
    trades = trades[:TRADE_LIMIT]

    last = ledger[-1] if ledger else {}
    last_day = last.get("ts")
    equity = last.get("equity", state["cash"])
    starting = state["starting_capital"]

    payload = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "simulated": True,
        "meta": {
            "universe": state["universe"],
            "interval": state.get("interval", "1d"),
            "inception": state["inception"],
            "last_processed": state["last_processed"],
            "starting_capital": starting,
            "n_fills": state["n_fills"],
        },
        "summary": {
            "equity": equity,
            "cash": state["cash"],
            "invested": last.get("invested", 0.0),
            "total_return": equity / starting - 1.0 if starting else 0.0,
            "realized_pnl": state["realized_pnl"],
            "unrealized_pnl": last.get("unrealized_pnl", 0.0),
            "explicit_costs": state["cumulative_costs"],
            "slippage_costs": state["cumulative_slippage"],
            "n_positions": last.get("n_positions", 0),
            "n_trades": n_trades,
            "n_closed_trades": len(closed),
            "win_rate": wins / len(closed) if closed else None,
        },
        "ledger": ledger,
        "positions": positions,
        "trades": trades,
        "pending_orders": state["pending_orders"],
        "decisions_last_day": [d for d in decisions if d.get("ts") == last_day],
        "learning": learning[-1] if learning else {},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(trades)} trades, {len(ledger)} days)")


if __name__ == "__main__":
    main()
