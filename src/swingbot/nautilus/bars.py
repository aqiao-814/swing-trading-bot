"""Turn the bar store's polars frames into NautilusTrader ``Bar`` objects.

Small module, one genuinely dangerous detail.

**The timestamp is the bar's CLOSE, and nothing else will do.** Yahoo labels a
30-minute bar by the instant it *opened*: the bar covering 09:30-10:00 ET
arrives labelled 09:30, and that is what the bar store persists. NautilusTrader
does not interpret the index at all -- it copies it verbatim into ``ts_event``,
drives its clock from ``ts_init``, and feeds each bar to the matching engine
*before* calling the strategy. So a market order submitted while handling a bar
fills at that bar's own close.

Put the open time in ``ts_event`` and the engine will happily fill you, at
10:00's price, on a clock that reads 09:30 -- thirty minutes of look-ahead,
silently, on every single fill. Adding the interval is not a cosmetic
correction; it is the difference between a backtest and a fantasy. There is a
test for it (``tests/test_nautilus_invariants.py``).

Two pandas-3 traps are also handled here, both of which fail loudly in one
environment and silently in another:

* ``BarDataWrangler.process()`` is unusable -- under copy-on-write
  ``DataFrame.values`` is read-only and the Cython ``double[:]`` signature
  rejects it. The vectorised ``Bar.from_raw_arrays_to_list`` is used instead,
  which is also ~2.4x faster.
* Datetime indexes default to microsecond (or second) resolution, so casting
  straight to int64 yields microseconds and places every bar in 1970. Everything
  is normalised to nanoseconds explicitly.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.instruments import Equity

# Regular-session bar length in minutes, by the store's interval name.
INTERVAL_MINUTES = {"30m": 30, "60m": 60, "1d": 390}

# Nautilus bar specification per interval. The four-token suffix is mandatory
# and EXTERNAL is required: `add_data` rejects INTERNAL outright, because an
# internally-aggregated bar type means "the engine builds these itself".
_SPEC = {
    "30m": "30-MINUTE-LAST-EXTERNAL",
    "60m": "1-HOUR-LAST-EXTERNAL",
    "1d": "1-DAY-LAST-EXTERNAL",
}

# The bar store keeps intraday timestamps naive in exchange-local time.
_ET = "America/New_York"


def bar_type_for(instrument: Equity, interval: str = "30m") -> BarType:
    return BarType.from_str(f"{instrument.id}-{_SPEC[interval]}")


def close_times_ns(ts: pl.Series, interval: str) -> np.ndarray:
    """Naive-ET bar-open timestamps -> UTC nanoseconds at the bar's CLOSE.

    Localises to New York (which is what makes DST correct -- a fixed -4/-5
    offset would misplace every bar for several weeks a year), shifts by one
    bar length, and returns uint64 nanoseconds.
    """
    minutes = INTERVAL_MINUTES[interval]
    dt = (
        ts.cast(pl.Datetime("us"))
        .dt.replace_time_zone(_ET)
        .dt.convert_time_zone("UTC")
        .dt.cast_time_unit("ns")
    )
    ns = dt.to_numpy().astype("datetime64[ns]").astype(np.int64)
    return (ns + minutes * 60 * 1_000_000_000).astype(np.uint64)


def _f64(df: pl.DataFrame, col: str) -> np.ndarray:
    """A writable, contiguous float64 array.

    ``copy=True`` is load-bearing: numpy views of arrow-backed columns come back
    read-only, and the Cython ``double[:]`` memoryview refuses them with a
    thoroughly unhelpful ``ValueError: buffer source array is read-only``.
    """
    return np.array(df[col].to_numpy(), dtype=np.float64, copy=True)


def bars_from_frame(
    df: pl.DataFrame, instrument: Equity, *, interval: str = "30m"
) -> list[Bar]:
    """One symbol's bars, oldest first.

    ``df`` carries the store's schema (``ts``, ``open``, ``high``, ``low``,
    ``close``, ``volume``) for a single symbol.
    """
    if df.is_empty():
        return []
    df = df.sort("ts")
    ts = close_times_ns(df["ts"], interval)
    return Bar.from_raw_arrays_to_list(
        bar_type_for(instrument, interval),
        instrument.price_precision,
        0,  # Equity size_precision is fixed at 0 -- whole shares
        _f64(df, "open"),
        _f64(df, "high"),
        _f64(df, "low"),
        _f64(df, "close"),
        _f64(df, "volume"),
        ts,
        # ts_init == ts_event: the engine's clock follows ts_init, and any
        # offset here silently shifts every timestamp the state layer writes.
        ts,
    )


def synthetic_bars(
    bar_type: BarType, price_precision: int, prices: list[float], ts_ns: np.ndarray
) -> list[Bar]:
    """Flat bars at fixed prices -- the restore phase's scaffolding.

    Open, high, low and close are all the same value, so a market order filling
    against one of these lands on exactly that price and a rebuilt position
    inherits precisely its persisted average cost. Used for two things only:
    the portfolio-restore bars, and the clock instrument.
    """
    arr = np.array(prices, dtype=np.float64)
    return Bar.from_raw_arrays_to_list(
        bar_type,
        price_precision,
        0,
        arr.copy(),
        arr.copy(),
        arr.copy(),
        arr.copy(),
        np.full(len(prices), 1_000_000.0, dtype=np.float64),
        np.asarray(ts_ns, dtype=np.uint64),
        np.asarray(ts_ns, dtype=np.uint64),
    )
