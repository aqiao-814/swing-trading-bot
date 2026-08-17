"""Freeze the retired v1 bot's final published record into the repository.

v1 (the long-only RRL day-trader) stopped trading at its last processed bar.
Its numbers are now history, and history must not be able to change: the
archive page renders from ``docs/v1-final.json`` -- a committed snapshot -- and
never from the live site's ``data.json``, which belongs to v2 and is rewritten
every thirty minutes.

Run once, against the last v1 payload:

    python scripts/freeze_v1.py https://aqiao-814.github.io/swingbot-live/data.json

Re-running against a v2 payload is refused: the snapshot is keyed to v1's
inception, so pointing this at the live URL after the cutover cannot silently
overwrite the archive with the successor's numbers.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "v1-final.json"

# v1's inception. A payload that does not carry it is not v1.
V1_INCEPTION = "2026-07-21T15:30:00"
# 30-minute bars: 13 regular-session bars a day, 252 sessions a year.
BARS_PER_YEAR = 13 * 252
# Fills kept for the archive's trade log. The full history stays in v1's
# parquet store; the page only ever rendered the newest couple of hundred.
TRADE_SAMPLE = 250
TRADE_FIELDS = (
    "ts",
    "symbol",
    "action",
    "quantity",
    "fill_price",
    "realized_pnl",
    "reason",
    "conviction",
)


def _load(src: str) -> dict:
    if src.startswith(("http://", "https://")):
        with urllib.request.urlopen(src, timeout=60) as fh:  # noqa: S310 - fixed https site
            return json.loads(fh.read())
    return json.loads(Path(src).read_text())


def _derive(ledger: list[dict]) -> dict:
    """Statistics the live exporter never computed, measured over all of v1."""
    equity = [r["equity"] for r in ledger]
    peak, max_dd = equity[0], 0.0
    for v in equity:
        peak = max(peak, v)
        max_dd = min(max_dd, v / peak - 1.0)

    rets = [r["daily_return"] for r in ledger if r.get("daily_return") is not None]
    sigma = statistics.pstdev(rets) if len(rets) > 1 else 0.0
    # Sharpe over nineteen sessions is a description of this sample, not an
    # estimate of v1's expected Sharpe. It is reported because the archive
    # states the sample size next to it.
    sharpe = (statistics.mean(rets) / sigma * math.sqrt(BARS_PER_YEAR)) if sigma else 0.0

    return {
        "bars": len(ledger),
        "sessions": len({r["ts"][:10] for r in ledger}),
        "max_drawdown": max_dd,
        "ann_sharpe": sharpe,
        "bar_return_std": sigma,
        "total_turnover": sum(r.get("turnover") or 0.0 for r in ledger),
        "peak_positions": max(r.get("n_positions") or 0 for r in ledger),
    }


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "https://aqiao-814.github.io/swingbot-live/data.json"
    payload = _load(src)

    inception = payload.get("meta", {}).get("inception")
    if inception != V1_INCEPTION:
        sys.exit(
            f"refusing to freeze: inception {inception!r} is not v1's {V1_INCEPTION!r}.\n"
            "This snapshot is v1's history; it must not be overwritten by v2's live payload."
        )

    ledger = payload["ledger"]
    trades = payload.get("trades", [])
    benches = [k for k in ledger[-1] if k.startswith("bench_")]
    start = payload["meta"]["starting_capital"]

    frozen = {
        "version": "v1",
        "retired": True,
        "source": src,
        "meta": payload["meta"],
        "summary": payload["summary"],
        "derived": _derive(ledger),
        "benchmarks": {
            b.removeprefix("bench_"): ledger[-1][b] / start - 1.0 for b in benches
        },
        # The full bar-by-bar equity curve: this is what the archive plots, and
        # at 235 rows it inlines into the page rather than being fetched.
        "curve": [
            {
                "ts": r["ts"],
                "eq": round(r["equity"], 2),
                "n": r.get("n_positions") or 0,
                **{b.removeprefix("bench_").lower(): round(r[b], 2) for b in benches},
            }
            for r in ledger
        ],
        # Only the columns the archive's trade log renders. v1's fills carried
        # fifteen fields; carrying the other seven would triple the page weight
        # to show nothing.
        "trades_sample": [
            {k: t.get(k) for k in TRADE_FIELDS} for t in trades[:TRADE_SAMPLE]
        ],
        "trades_sample_note": (
            f"newest {min(TRADE_SAMPLE, len(trades))} of {payload['summary']['n_trades']:,} fills"
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(frozen, indent=1, sort_keys=False) + "\n")
    d = frozen["derived"]
    print(
        f"froze v1 -> {OUT.relative_to(ROOT)}: "
        f"{d['bars']} bars over {d['sessions']} sessions, "
        f"equity ${frozen['summary']['equity']:,.2f} "
        f"({frozen['summary']['total_return']:+.2%}), "
        f"max DD {d['max_drawdown']:.2%}, {OUT.stat().st_size:,} bytes"
    )


if __name__ == "__main__":
    main()
