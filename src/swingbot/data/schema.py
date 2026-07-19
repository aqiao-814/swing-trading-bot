"""Canonical bar schema and validation.

Every source normalises into this shape before it touches storage, so the rest
of the system never learns which vendor a bar came from.

Point-in-time policy
--------------------
We store **raw** OHLCV exactly as printed, plus the vendor's ``adj_close``. The
adjustment *factor* is derived (``adj_close / close``), never baked into the
stored prices. This matters: back-adjusting the whole series in storage means
today's split factor rewrites prices from 1998, which is look-ahead bias in
anything that keys off price levels (round numbers, dollar notional, tick size).
Callers ask for the adjustment they want at read time via ``AdjustmentMode``.
"""

from __future__ import annotations

from enum import StrEnum

import polars as pl

BAR_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.Utf8,
    "ts": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "adj_close": pl.Float64,
    "volume": pl.Float64,
}

REQUIRED_COLUMNS = tuple(BAR_SCHEMA.keys())


class AdjustmentMode(StrEnum):
    RAW = "raw"  # exactly as printed; correct for level-dependent logic
    ADJUSTED = "adjusted"  # back-adjusted; correct for return series
    NONE = "none"


class DataQualityError(ValueError):
    """Raised when bars violate an invariant that would corrupt a backtest."""


def normalize(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce a source frame into BAR_SCHEMA order, types, and sort.

    ``ts`` keeps its temporal resolution: intraday frames arrive with a
    Datetime column and must stay Datetime — casting to Date would collapse
    every bar of a day onto one key and silently drop all but the last.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise DataQualityError(f"missing required columns: {sorted(missing)}")

    schema = dict(BAR_SCHEMA)
    if df.schema["ts"] == pl.Datetime:
        schema["ts"] = pl.Datetime("us")

    return (
        df.select([pl.col(name).cast(dtype, strict=False) for name, dtype in schema.items()])
        .drop_nulls(subset=["symbol", "ts", "close"])
        .unique(subset=["symbol", "ts"], keep="last")
        .sort(["symbol", "ts"])
    )


def validate(df: pl.DataFrame, *, max_abs_daily_return: float = 0.75) -> list[str]:
    """Return a list of human-readable data-quality problems (empty == clean).

    These are the failures that silently produce beautiful, fictional backtests:
    negative prices, OHLC ordering violations, duplicate bars, and unadjusted
    splits masquerading as 50% single-day moves.
    """
    problems: list[str] = []
    if df.is_empty():
        return ["frame is empty"]

    for col in ("open", "high", "low", "close", "adj_close"):
        n_bad = df.filter(pl.col(col) <= 0).height
        if n_bad:
            problems.append(f"{n_bad} bars with non-positive {col}")

    n_neg_vol = df.filter(pl.col("volume") < 0).height
    if n_neg_vol:
        problems.append(f"{n_neg_vol} bars with negative volume")

    # High must bound the bar; low must floor it.
    n_hilo = df.filter(
        (pl.col("high") < pl.col("low"))
        | (pl.col("high") < pl.col("open"))
        | (pl.col("high") < pl.col("close"))
        | (pl.col("low") > pl.col("open"))
        | (pl.col("low") > pl.col("close"))
    ).height
    if n_hilo:
        problems.append(f"{n_hilo} bars violate OHLC ordering")

    n_dupes = df.height - df.unique(subset=["symbol", "ts"]).height
    if n_dupes:
        problems.append(f"{n_dupes} duplicate (symbol, ts) rows")

    # A >75% single-day move in a liquid name is almost always an unhandled
    # corporate action rather than a real print.
    jumps = (
        df.sort(["symbol", "ts"])
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1.0).alias("ret")
        )
        .filter(pl.col("ret").abs() > max_abs_daily_return)
    )
    if jumps.height:
        sample = jumps.select(["symbol", "ts", "ret"]).head(3).to_dicts()
        problems.append(
            f"{jumps.height} suspect jumps > {max_abs_daily_return:.0%} "
            f"(likely unadjusted splits), e.g. {sample}"
        )

    return problems


def apply_adjustment(df: pl.DataFrame, mode: AdjustmentMode) -> pl.DataFrame:
    """Apply split/dividend adjustment at read time.

    ADJUSTED scales OHLC by ``adj_close / close`` and inflates volume by the
    inverse, preserving traded notional across splits.
    """
    if mode in (AdjustmentMode.RAW, AdjustmentMode.NONE):
        return df

    factor = pl.col("adj_close") / pl.col("close")
    return df.with_columns(
        [
            (pl.col("open") * factor).alias("open"),
            (pl.col("high") * factor).alias("high"),
            (pl.col("low") * factor).alias("low"),
            pl.col("adj_close").alias("close"),
            (pl.col("volume") / factor).alias("volume"),
        ]
    )
