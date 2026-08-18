"""swingbot v2 — the live loop, rebuilt on the NautilusTrader engine.

v1 (``swingbot.paper``) was a bespoke Python bar loop: it owned its own order
queue, fill logic, position accounting and cash ledger. It worked, and it is
retired -- see ``docs/v1-final.json`` and the archive page for its record.

v2 hands all of that to NautilusTrader and keeps only the parts that are
actually *ours*: which stocks to want, how much of each, and when. The engine
owns orders, fills, commissions, margin, and position accounting, in a Rust
core that runs the same code path for a backtest and for the live loop.

Why the change was worth making is a strategy question, not an engineering one.
``docs/FINDINGS.md`` §10a measured a real, out-of-sample, regime-persistent edge
at a **3-day horizon, traded dollar-neutral long/short**, net-positive through
about 3 bp per side. v1 could not express it: it was long-only (so it could not
be neutral) and flat by every close (so it could not hold a 3-day signal). v2
exists to be able to hold a short overnight.

Every dollar is still SIMULATED. There are no brokerage credentials in this
repository and no code path that can place a real order.

Layout::

    instruments.py  the ~670-name US equity universe as Nautilus Equity objects
    costs.py        all-in execution costs as an explicit Nautilus FeeModel
    bars.py         yfinance OHLCV frames -> Nautilus Bar objects
    signals.py      the cross-sectional long/short alpha, and the news tilt
    strategy.py     the Nautilus Strategy: rank, size, and trade the book
    state.py        the portfolio that survives between 30-minute runs
    runner.py       restore -> replay -> extract, once per cron firing
"""

from __future__ import annotations

__all__ = ["VENUE_NAME"]

# Nasdaq's MIC. The bot trades US equities against one simulated venue; the
# name only matters in that instrument ids and persisted state are keyed by it,
# so changing it invalidates a saved book.
VENUE_NAME = "XNAS"
