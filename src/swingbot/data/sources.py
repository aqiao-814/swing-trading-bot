"""Market data sources.

Free-tier reality (verified by probing each endpoint, 2026-07):

* **Yahoo via yfinance** -- the default. No API key, ~30+ years of daily bars,
  and crucially it returns *both* raw ``Close`` and ``Adj Close``, which is what
  our point-in-time schema needs. Unofficial and rate-limited: heavy use trips
  ``YFRateLimitError`` (HTTP 429) and can get an IP blacklisted, so we throttle
  and cache. Not a TOS-sanctioned API -- fine for personal research, not for
  anything you'd build a business on.
* **Synthetic** -- a GBM/regime generator. Not a toy: it is how we test the
  engine without a network, and how we sanity-check that a strategy returns
  ~zero on data with no edge. If a strategy "profits" on synthetic noise, the
  backtest is broken.

Notably *unavailable*: **Stooq**, which the research blueprint recommended as
the best free bulk source, now sits behind a JavaScript proof-of-work anti-bot
challenge (probed 2026-07: every endpoint returns a SHA-256 PoW interstitial
rather than CSV). That challenge exists specifically to block automated clients,
so this module does not attempt to defeat it. If you want Stooq data, download
the bulk ZIP manually in a browser and load it via ``CsvSource``.

Deliberately absent: sources requiring a paid key. Alpha Vantage's free tier
(25 req/day) is too small to bulk-load, and Nasdaq/Quandl WIKI is frozen at
March 2018. Alpaca and Tiingo are good *keyed* free tiers -- add them here when
you have credentials; the ``BarSource`` interface is the only contract.
"""

from __future__ import annotations

import hashlib
import time
import warnings
from abc import ABC, abstractmethod
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from swingbot.data.schema import DataQualityError, normalize


def _stable_seed(seed: int, symbol: str) -> int:
    """Deterministic seed from (seed, symbol), stable across processes.

    Python's builtin ``hash()`` is salted per interpreter run for str inputs, so
    using it here would make "reproducible" synthetic data differ every run --
    quietly breaking both reproducibility and any test that depends on it.
    """
    digest = hashlib.blake2b(f"{seed}:{symbol}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32)


class BarSource(ABC):
    """Fetches bars for one symbol and normalises to BAR_SCHEMA."""

    @abstractmethod
    def fetch(
        self, symbol: str, start: date | str | None, end: date | str | None
    ) -> pl.DataFrame: ...

    def fetch_many(
        self,
        symbols: list[str],
        start: date | str | None = None,
        end: date | str | None = None,
        *,
        pause: float = 0.5,
        on_error: str = "warn",
    ) -> pl.DataFrame:
        """Fetch several symbols, tolerating individual failures.

        A universe build that dies on one delisted ticker is useless, so by
        default we collect what we can and report what we couldn't.
        """
        frames, failures = [], []
        for i, symbol in enumerate(symbols):
            try:
                df = self.fetch(symbol, start, end)
                if not df.is_empty():
                    frames.append(df)
            except Exception as exc:  # noqa: BLE001 - source errors are expected
                failures.append((symbol, str(exc)))
                if on_error == "raise":
                    raise
            if pause and i < len(symbols) - 1:
                time.sleep(pause)

        if failures and on_error == "warn":
            print(f"[data] {len(failures)} symbol(s) failed: {failures[:5]}")
        if not frames:
            raise DataQualityError(f"no data fetched for any of {symbols}")
        return normalize(pl.concat(frames, how="vertical"))


class YahooSource(BarSource):
    """Daily bars from Yahoo Finance via ``yfinance``.

    Returns raw OHLC plus a separate ``Adj Close``, so we can store both and
    defer adjustment to read time (see ``data.schema``).

    Yahoo is unofficial and rate-limits aggressively. ``fetch_many`` throttles
    between symbols; if you are pulling hundreds of tickers, expect to need a
    generous pause and to cache results (that is what the BarStore is for --
    fetch once, then read from Parquet forever).
    """

    def __init__(
        self, timeout: float = 30.0, auto_retry: bool = True, interval: str = "1d"
    ) -> None:
        self.timeout = timeout
        self.auto_retry = auto_retry
        self.interval = interval

    # Yahoo's intraday history caps depend on the interval: hourly reaches back
    # ~730 days, but sub-hour bars (30m and finer) only ~60. Clamp to just
    # inside each cap so an over-eager data_start doesn't get the whole request
    # rejected.
    _INTRADAY_MAX_DAYS = {"60m": 728, "1h": 728, "90m": 58, "30m": 58, "15m": 58, "5m": 58, "1m": 6}

    def _clamp_start(self, start: date | str | None) -> str | None:
        """Clamp an intraday start to just inside Yahoo's history cap."""
        if start is None or self.interval == "1d":
            return str(start) if start else None
        lo = date.today() - timedelta(days=self._INTRADAY_MAX_DAYS.get(self.interval, 58))
        s = date.fromisoformat(str(start)[:10])
        return str(max(s, lo))

    def _ts_series(self, values) -> pl.Series:
        """Vendor timestamps -> naive US/Eastern Datetime (intraday) or Date."""
        if self.interval == "1d":
            return pl.Series(values).cast(pl.Date)
        import pandas as pd

        idx = pd.DatetimeIndex(values)
        if idx.tz is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
        return pl.Series(idx.values.astype("datetime64[us]"))

    def fetch(
        self, symbol: str, start: date | str | None = None, end: date | str | None = None
    ) -> pl.DataFrame:
        import yfinance as yf  # imported lazily: only needed when actually fetching

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                symbol,
                start=self._clamp_start(start),
                end=str(end) if end else None,
                interval=self.interval,
                progress=False,
                auto_adjust=False,  # we want raw close AND adj close
                actions=False,
                timeout=self.timeout,
            )

        if raw is None or raw.empty:
            raise DataQualityError(f"no data returned for {symbol}")

        # yfinance returns MultiIndex columns (field, ticker) even for one ticker.
        if hasattr(raw.columns, "nlevels") and raw.columns.nlevels > 1:
            raw.columns = raw.columns.droplevel(-1)

        raw = raw.reset_index()
        cols = {c.lower().replace(" ", "_"): c for c in raw.columns}
        cols.setdefault("date", cols.get("datetime", cols.get("index")))
        if self.interval != "1d" and "close" in cols:
            # Intraday bars are as-traded; Yahoo may omit Adj Close there.
            cols.setdefault("adj_close", cols["close"])
        required = ("open", "high", "low", "close", "adj_close", "volume")
        missing = [c for c in required if c not in cols]
        if missing or cols["date"] is None:
            raise DataQualityError(
                f"yahoo schema for {symbol} missing {missing}: {list(raw.columns)}"
            )

        df = pl.DataFrame(
            {
                "symbol": [symbol.upper()] * len(raw),
                "ts": self._ts_series(raw[cols["date"]]),
                "open": pl.Series(raw[cols["open"]].astype(float)),
                "high": pl.Series(raw[cols["high"]].astype(float)),
                "low": pl.Series(raw[cols["low"]].astype(float)),
                "close": pl.Series(raw[cols["close"]].astype(float)),
                "adj_close": pl.Series(raw[cols["adj_close"]].astype(float)),
                "volume": pl.Series(raw[cols["volume"]].astype(float)),
            }
        )
        return normalize(df)


class YahooBulkSource(YahooSource):
    """YahooSource with a chunked multi-ticker ``fetch_many``.

    One yfinance request can carry ~50-100 tickers, which is the difference
    between refreshing an S&P-scale universe in a handful of requests and
    tripping Yahoo's rate limiter with 500 serial calls. Single-symbol
    ``fetch`` is inherited unchanged; symbols missing from a bulk response
    fall back to the throttled per-symbol path.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        chunk_size: int = 50,
        pause: float = 1.0,
        interval: str = "1d",
    ) -> None:
        super().__init__(timeout=timeout, interval=interval)
        self.chunk_size = chunk_size
        self.chunk_pause = pause

    def fetch_many(
        self,
        symbols: list[str],
        start: date | str | None = None,
        end: date | str | None = None,
        *,
        pause: float = 0.5,
        on_error: str = "warn",
    ) -> pl.DataFrame:
        import yfinance as yf

        wanted = [s.upper() for s in symbols]
        frames: list[pl.DataFrame] = []
        missing: list[str] = []

        for i in range(0, len(wanted), self.chunk_size):
            chunk = wanted[i : i + self.chunk_size]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    chunk,
                    start=self._clamp_start(start),
                    end=str(end) if end else None,
                    interval=self.interval,
                    progress=False,
                    auto_adjust=False,
                    actions=False,
                    group_by="ticker",
                    threads=True,
                    timeout=self.timeout,
                )
            if raw is None or raw.empty:
                missing.extend(chunk)
            else:
                for symbol in chunk:
                    df = self._extract_ticker(raw, symbol, single=len(chunk) == 1)
                    if df is None or df.is_empty():
                        missing.append(symbol)
                    else:
                        frames.append(df)
            if i + self.chunk_size < len(wanted) and self.chunk_pause:
                time.sleep(self.chunk_pause)

        # Per-symbol fallback for anything the bulk call dropped (delistings,
        # renames, transient holes). Failures there are tolerated as usual.
        if missing:
            try:
                frames.append(super().fetch_many(missing, start, end, pause=pause, on_error="warn"))
            except DataQualityError:
                if on_error == "raise":
                    raise
                print(f"[data] no data for {len(missing)} symbol(s): {missing[:5]}")

        if not frames:
            raise DataQualityError(f"no data fetched for any of {symbols}")
        return normalize(pl.concat(frames, how="vertical"))

    def _extract_ticker(self, raw, symbol: str, *, single: bool) -> pl.DataFrame | None:
        """Pull one ticker's frame out of a (possibly MultiIndex) bulk response."""
        try:
            sub = raw if single or not hasattr(raw.columns, "nlevels") else raw[symbol]
            if hasattr(sub.columns, "nlevels") and sub.columns.nlevels > 1:
                sub = sub.copy()
                sub.columns = sub.columns.droplevel(-1)
            sub = sub.dropna(how="all").reset_index()
        except (KeyError, AttributeError):
            return None

        cols = {c.lower().replace(" ", "_"): c for c in sub.columns}
        cols.setdefault("date", cols.get("datetime", cols.get("index")))
        if self.interval != "1d" and "close" in cols:
            cols.setdefault("adj_close", cols["close"])
        required = ("open", "high", "low", "close", "adj_close", "volume")
        if any(c not in cols for c in required) or cols["date"] is None:
            return None
        return pl.DataFrame(
            {
                "symbol": [symbol] * len(sub),
                "ts": self._ts_series(sub[cols["date"]]),
                "open": pl.Series(sub[cols["open"]].astype(float)),
                "high": pl.Series(sub[cols["high"]].astype(float)),
                "low": pl.Series(sub[cols["low"]].astype(float)),
                "close": pl.Series(sub[cols["close"]].astype(float)),
                "adj_close": pl.Series(sub[cols["adj_close"]].astype(float)),
                "volume": pl.Series(sub[cols["volume"]].astype(float)),
            }
        )


class CsvSource(BarSource):
    """Load bars from local CSV files -- the manual escape hatch.

    Use this for data you downloaded by hand (e.g. the Stooq bulk ZIP, which is
    free but browser-gated). Expects one file per symbol named ``<SYMBOL>.csv``
    with Date/Open/High/Low/Close/Volume columns, optionally ``Adj Close``.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def fetch(
        self, symbol: str, start: date | str | None = None, end: date | str | None = None
    ) -> pl.DataFrame:
        path = self.directory / f"{symbol.upper()}.csv"
        if not path.exists():
            raise DataQualityError(f"no CSV for {symbol} at {path}")

        df = pl.read_csv(path, try_parse_dates=True)
        df = df.rename({c: c.lower().replace(" ", "_") for c in df.columns})
        if "date" not in df.columns:
            raise DataQualityError(f"{path} has no Date column: {df.columns}")

        # Fall back to close when the file carries no adjusted series.
        adj = pl.col("adj_close") if "adj_close" in df.columns else pl.col("close")
        df = df.with_columns(
            [
                pl.lit(symbol.upper()).alias("symbol"),
                pl.col("date").cast(pl.Date).alias("ts"),
                adj.alias("adj_close"),
                pl.col("volume").cast(pl.Float64),
            ]
        )
        df = normalize(df)
        if start is not None:
            df = df.filter(pl.col("ts") >= pl.lit(start).cast(pl.Date))
        if end is not None:
            df = df.filter(pl.col("ts") <= pl.lit(end).cast(pl.Date))
        return df


class SyntheticSource(BarSource):
    """Deterministic GBM bars with optional regime switching.

    Used for tests and for the null hypothesis: on a pure random walk with costs,
    every strategy must lose money. That is the cheapest available check that the
    simulator isn't leaking future information.
    """

    def __init__(
        self,
        *,
        seed: int = 7,
        annual_drift: float = 0.07,
        annual_vol: float = 0.20,
        start_price: float = 100.0,
        regime_switching: bool = False,
    ) -> None:
        self.seed = seed
        self.annual_drift = annual_drift
        self.annual_vol = annual_vol
        self.start_price = start_price
        self.regime_switching = regime_switching

    def fetch(
        self,
        symbol: str,
        start: date | str | None = "2000-01-01",
        end: date | str | None = "2024-12-31",
    ) -> pl.DataFrame:
        days = pl.date_range(
            pl.lit(str(start or "2000-01-01")).str.to_date(),
            pl.lit(str(end or "2024-12-31")).str.to_date(),
            interval="1d",
            eager=True,
        )
        # Weekdays only: a crude but self-consistent trading calendar.
        days = days.filter(days.dt.weekday() <= 5)
        n = len(days)

        # Seed per symbol so each ticker differs but is reproducible.
        # NOT builtin hash(): it is salted per process (PYTHONHASHSEED), so it
        # would silently generate different "reproducible" data on every run.
        rng = np.random.default_rng(_stable_seed(self.seed, symbol))

        dt = 1.0 / 252.0
        vol = np.full(n, self.annual_vol)
        if self.regime_switching:
            # Two-state vol regime with sticky transitions (~2% daily switch rate).
            state, states = 0, np.zeros(n, dtype=int)
            for i in range(n):
                if rng.random() < 0.02:
                    state = 1 - state
                states[i] = state
            vol = np.where(states == 1, self.annual_vol * 2.5, self.annual_vol)

        shocks = rng.normal(0, 1, n) * vol * np.sqrt(dt)
        drift = (self.annual_drift - 0.5 * vol**2) * dt
        log_price = np.log(self.start_price) + np.cumsum(drift + shocks)
        close = np.exp(log_price)

        prev_close = np.concatenate([[self.start_price], close[:-1]])
        open_ = prev_close * np.exp(rng.normal(0, 0.002, n))  # overnight gap
        intrabar = np.abs(rng.normal(0, 0.005, n))
        high = np.maximum(open_, close) * (1 + intrabar)
        low = np.minimum(open_, close) * (1 - intrabar)
        volume = rng.lognormal(15, 0.4, n)

        return normalize(
            pl.DataFrame(
                {
                    "symbol": [symbol.upper()] * n,
                    "ts": days,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": close,
                    "volume": volume,
                }
            )
        )


def get_source(name: str, **kwargs) -> BarSource:
    match name.lower():
        case "yahoo" | "yfinance":
            return YahooSource(**kwargs)
        case "yahoo_bulk":
            return YahooBulkSource(**kwargs)
        case "csv":
            return CsvSource(**kwargs)
        case "synthetic":
            return SyntheticSource(**kwargs)
        case "stooq":
            raise ValueError(
                "stooq is behind a JavaScript proof-of-work anti-bot challenge as of 2026-07 "
                "and is not scriptable. Download the bulk ZIP in a browser and use "
                "source='csv', or use source='yahoo'."
            )
        case _:
            raise ValueError(f"unknown source '{name}' (have: yahoo, csv, synthetic)")
