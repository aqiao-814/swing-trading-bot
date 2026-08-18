"""v1, RETIRED — the long-only RRL day-trading loop.

This package is no longer the live bot. It ran 2026-07-21 to 2026-08-14 and
finished at $96,211.75 (-3.79%) against SPY +3.75% and its own universe's
+5.47%; its frozen record is ``docs/v1-final.json`` and the archive page is
``site/v1/``. See ``docs/FINDINGS.md`` §12 for the post-mortem.

**The live loop is now :mod:`tradingbot.nautilus`** — a cross-sectional long/short
book on the NautilusTrader engine, which can do the two things this one
structurally could not: hold a short, and hold it overnight. That mattered
because §10a measured the only real edge in this project at a 3-day horizon,
dollar-neutral — and a long-only book that liquidates every afternoon cannot
trade it.

The code stays because it is still the research harness the rest of the project
grew from, and because its continual-learning RRL implementation is the honest
record of an approach that was tried and measured. It is not maintained, and
``tradingbot invest`` should not be pointed at the live site's state.
"""
