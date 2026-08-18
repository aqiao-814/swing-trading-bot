"""Export the v2 portfolio as one JSON blob for the live site.

Reads ``artifacts/v2/portfolio`` (the engine's source of truth) and writes
``site/data.json``; the static dashboard fetches it and renders client-side.

The payload changed shape at the v1 -> v2 cutover, because the book did. v1 was
long-only and cash-funded, so "cash" and "invested" described it completely. v2
is a margin book that can be short, so the honest summary is *balance* (cash
plus realized PnL), *gross* (how much is at risk) and *net* (how directional it
is) -- and a position now carries a sign.

Benchmarks are computed here rather than in the trading loop. A buy-and-hold
curve is a pure function of the bars and the inception instant, so recomputing
it on every export is both cheaper and less error-prone than carrying units in
the portfolio state, and it means adding a benchmark never requires a migration.

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
sys.path.insert(0, str(ROOT / "src"))

PORTFOLIO = Path(os.environ.get("SWINGBOT_PORTFOLIO", ROOT / "artifacts" / "v2" / "portfolio"))
DATA_ROOT = Path(os.environ.get("SWINGBOT_DATA", ROOT / "data30"))
OUT = Path(os.environ.get("SWINGBOT_SITE", ROOT / "site")) / "data.json"

# Newest fills kept in data.json. The full history stays in the parquet store.
# A long/short book on 30-minute bars fills hundreds of times a session, and
# this file is rewritten into a git commit every half hour.
TRADE_LIMIT = 800
BENCHMARKS = ("SPY", "QQQ")


def _records(name: str, sort_by: str | None = None, descending: bool = False) -> list[dict]:
    path = PORTFOLIO / f"{name}.parquet"
    if not path.exists():
        return []
    df = pl.read_parquet(path)
    if df.is_empty():
        return []
    if sort_by and sort_by in df.columns:
        df = df.sort(sort_by, descending=descending)
    for col in df.columns:
        if df[col].dtype == pl.Date:
            df = df.with_columns(pl.col(col).dt.strftime("%Y-%m-%d"))
        elif df[col].dtype == pl.Datetime:
            df = df.with_columns(pl.col(col).dt.strftime("%Y-%m-%d %H:%M"))
        elif df[col].dtype in (pl.Float32, pl.Float64):
            # NaN/inf would make json.dumps(allow_nan=False) throw; null is the
            # honest rendering of "not computed" anyway.
            df = df.with_columns(pl.when(pl.col(col).is_finite()).then(pl.col(col)).alias(col))
    return df.to_dicts()


def _benchmarks(ledger: list[dict], starting: float) -> dict[str, list[float | None]]:
    """Buy-and-hold curves for each benchmark, aligned to the ledger's bars.

    Each series starts at ``starting`` on the ledger's first bar, so the chart
    compares like with like. A benchmark with no stored bars is omitted rather
    than faked flat, which would read as "it went nowhere".
    """
    if not ledger:
        return {}
    try:
        from swingbot.data.store import BarStore
    except ImportError:
        return {}
    store = BarStore(DATA_ROOT)

    stamps = [r["ts"] for r in ledger]
    out: dict[str, list[float | None]] = {}
    for sym in BENCHMARKS:
        if sym not in store:
            continue
        df = store.read([sym])
        if df.is_empty():
            continue
        marks = {
            ts.strftime("%Y-%m-%d %H:%M"): close
            for ts, close in zip(df["ts"].to_list(), df["close"].to_list(), strict=True)
        }
        base = next((marks[s] for s in stamps if s in marks), None)
        if not base:
            continue
        last = starting
        series: list[float] = []
        for s in stamps:
            if s in marks:
                last = starting * marks[s] / base
            # Carry the previous mark through a bar the benchmark did not trade,
            # rather than dropping a point and shifting the line.
            series.append(round(last, 2))
        out[sym] = series
    return out


def main() -> None:
    state_path = PORTFOLIO / "state.json"
    if not state_path.exists():
        sys.exit("no v2 state.json yet -- run `swingbot trade` first")
    state = json.loads(state_path.read_text())

    ledger = _records("ledger", sort_by="ts")
    trades = _records("trades", sort_by="ts", descending=True)
    positions = _records("positions", sort_by="weight", descending=True)
    decisions = _records("decisions", sort_by="ts")

    starting = state["starting_capital"]
    last = ledger[-1] if ledger else {}
    equity = last.get("equity", state["account_balance"])
    n_trades = len(trades)
    trades = trades[:TRADE_LIMIT]

    last_ts = last.get("ts")
    longs = [p for p in positions if (p.get("quantity") or 0) > 0]
    shorts = [p for p in positions if (p.get("quantity") or 0) < 0]

    payload = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "simulated": True,
        "version": "v2",
        "engine": "NautilusTrader",
        "meta": {
            "universe": state["universe"],
            "interval": state.get("interval", "30m"),
            "inception": state["inception"],
            "last_processed": state["last_processed"],
            "starting_capital": starting,
            "n_fills": state["n_fills"],
        },
        "summary": {
            "equity": equity,
            # Cash plus realized PnL. NOT cash-on-hand: on a margin account an
            # open position locks margin instead of spending cash.
            "balance": state["account_balance"],
            "total_return": equity / starting - 1.0 if starting else 0.0,
            "gross": last.get("gross", 0.0),
            "net": last.get("net", 0.0),
            "unrealized_pnl": equity - state["account_balance"],
            "friction_costs": state.get("cumulative_friction", 0.0),
            "fee_costs": state.get("cumulative_fees", 0.0),
            "borrow_costs": state.get("cumulative_borrow", 0.0),
            "n_positions": len(positions),
            "n_long": len(longs),
            "n_short": len(shorts),
            "n_trades": n_trades,
        },
        "ledger": ledger,
        "benchmarks": _benchmarks(ledger, starting),
        "positions": positions,
        "trades": trades,
        "decisions_last_bar": [d for d in decisions if d.get("ts") == last_ts],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    print(
        f"wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(ledger)} bars, "
        f"{len(positions)} positions [{len(longs)}L/{len(shorts)}S], {n_trades:,} fills)"
    )


if __name__ == "__main__":
    main()
