"""Intraday live view: quotes, live P&L, and provisional open fills.

Runs every few minutes during market hours (dependencies: yfinance only --
it must NOT import swingbot, so the live workflow can install one package
and finish in seconds). Reads the committed portfolio state + nightly
data.json, fetches live quotes for every relevant symbol, and writes
``site/live.json`` for the dashboard's intraday layer.

The nightly engine run stays the source of truth: everything here is a
display-level estimate on top of it, and says so.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
STATE = (
    Path(os.environ.get("SWINGBOT_PORTFOLIO", ROOT / "artifacts" / "paper" / "portfolio"))
    / "state.json"
)
_SITE = Path(os.environ.get("SWINGBOT_SITE", ROOT / "site"))
DATA = _SITE / "data.json"
OUT = _SITE / "live.json"

ET = ZoneInfo("America/New_York")
BENCHMARKS = ["SPY", "QQQ"]


def fetch_daily(symbols: list[str]):
    """Last ~5 daily bars per symbol; today's partial bar carries the live price."""
    df = yf.download(
        symbols,
        period="5d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    out: dict[str, dict] = {}
    for sym in symbols:
        try:
            sub = df[sym] if len(symbols) > 1 else df
            sub = sub.dropna(subset=["Close"])
            if sub.empty:
                continue
            last = sub.iloc[-1]
            prev = sub.iloc[-2] if len(sub) > 1 else last
            out[sym] = {
                "date": sub.index[-1].date().isoformat(),
                "open": float(last["Open"]),
                "price": float(last["Close"]),
                "prev_close": float(prev["Close"]),
            }
        except Exception:  # one bad symbol must not sink the update
            continue
    return out


def main() -> None:
    if not STATE.exists() or not DATA.exists():
        sys.exit("state.json / data.json missing -- nightly run has not happened yet")
    state = json.loads(STATE.read_text())
    data = json.loads(DATA.read_text())

    held = {p["symbol"]: p for p in state.get("positions", [])}
    pending = state.get("pending_orders", [])
    symbols = sorted(set(held) | {o["symbol"] for o in pending} | set(BENCHMARKS))

    quotes = fetch_daily(symbols)
    if not quotes:
        sys.exit("no quotes returned; keeping previous live.json")

    now_et = datetime.now(ET)
    today = now_et.date().isoformat()
    spy = quotes.get("SPY", {})
    # A trading day is one where SPY printed a bar today; combined with the
    # clock this also gets holidays right.
    traded_today = spy.get("date") == today
    in_hours = now_et.weekday() < 5 and (9, 30) <= (now_et.hour, now_et.minute) < (16, 0)
    market_open = traded_today and in_hours

    # "Today's P&L" baselines at the previous *day's* last bar. On the hourly
    # loop the ledger has one row per bar, so the last row is only minutes old
    # on a trading day; on a quiet day the last row is the right baseline.
    ledger = data.get("ledger", [])
    prev_equity = ledger[-1]["equity"] if ledger else state["starting_capital"]
    if ledger and traded_today:
        before_today = [r for r in ledger if r["ts"][:10] < today]
        if before_today:
            prev_equity = before_today[-1]["equity"]

    positions = []
    invested = 0.0
    for sym, p in sorted(held.items()):
        q = quotes.get(sym)
        price = q["price"] if q else p["avg_price"]
        mv = p["quantity"] * price
        invested += mv
        positions.append(
            {
                "symbol": sym,
                "quantity": p["quantity"],
                "avg_price": p["avg_price"],
                "live_price": price,
                "market_value": mv,
                "unrealized_pnl": (price - p["avg_price"]) * p["quantity"],
                "day_change_pct": (price / q["prev_close"] - 1.0) if q else 0.0,
                "entry_ts": p.get("entry_ts"),
            }
        )
    equity = state["cash"] + invested
    for pos in positions:
        pos["weight"] = pos["market_value"] / equity if equity > 0 else 0.0

    # Provisional view of pending orders filling at today's open. Estimates
    # only -- the engine records the official fill (with costs) after close.
    provisional = []
    if traded_today and now_et.hour * 60 + now_et.minute >= 9 * 60 + 31 and not state.get("halted"):
        for o in pending:
            if o["decided_ts"] >= today:
                continue  # decided today; fills at the NEXT open
            q = quotes.get(o["symbol"])
            if not q or q["date"] != today:
                continue
            ref = q["open"]
            held_qty = held.get(o["symbol"], {}).get("quantity", 0.0)
            desired = math.floor(abs(o["target_weight"]) * prev_equity / ref)
            desired = desired if o["target_weight"] > 1e-9 else 0
            delta = desired - held_qty
            if abs(delta) < 1.0:
                continue
            provisional.append(
                {
                    "symbol": o["symbol"],
                    "action": "buy" if delta > 0 else "sell",
                    "est_quantity": abs(delta),
                    "est_price": ref,
                    "est_notional": abs(delta) * ref,
                    "reason": o["reason"],
                    "conviction": o["conviction"],
                    "decided_ts": o["decided_ts"],
                }
            )

    payload = {
        "asof_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "market_open": market_open,
        "traded_today": traded_today,
        "equity": equity,
        "cash": state["cash"],
        "invested": invested,
        "day_pnl": equity - prev_equity,
        "day_pnl_pct": equity / prev_equity - 1.0 if prev_equity > 0 else 0.0,
        "total_pnl": equity - state["starting_capital"],
        "total_return": equity / state["starting_capital"] - 1.0,
        "halted": state.get("halted"),
        "benchmarks": {
            s: {
                "price": quotes[s]["price"],
                "day_change_pct": quotes[s]["price"] / quotes[s]["prev_close"] - 1.0,
            }
            for s in BENCHMARKS
            if s in quotes
        },
        "positions": positions,
        "provisional_fills": provisional,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(
        f"wrote {OUT}: equity ${equity:,.2f}, {len(positions)} positions, "
        f"{len(provisional)} provisional fill(s), market_open={market_open}"
    )


if __name__ == "__main__":
    main()
