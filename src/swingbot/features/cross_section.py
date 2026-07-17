"""Cross-sectional feature panel for the excess-return ranker.

The ranker asks a *relative* question -- "will this name beat the benchmark
over the next ``horizon`` days?" -- so the unit of data is a (date, symbol)
panel, not a single symbol's history. Two disciplines are enforced here:

* **Trailing-only features.** Every feature at date ``t`` is computed from
  bars at ``t`` and earlier. The deliberately small starting set follows the
  seven-feature baseline that carries a real (0.03-0.07) RankIC in the open
  literature; resist adding more before the baseline has a measured IC.

* **The target is forward and honest about maturity.** ``target`` is the
  ``horizon``-trading-day forward return in excess of the benchmark, computed
  from dividend/split-adjusted closes on both legs. It is null until the
  window has fully matured, so "which rows may I train on as of date d" is a
  null-check plus a date cutoff, never a judgement call.
"""

from __future__ import annotations

import polars as pl

FEATURES = [
    "mom_1m",
    "mom_3m",
    "mom_12m",
    "vol_20d",
    "vol_60d",
    "adv_20d",
    "cross_sectional_rank",
]

# Trading-day window lengths.
_M1, _M3, _M12 = 21, 63, 252


def build_panel(
    bars: pl.DataFrame,
    bench: pl.DataFrame,
    *,
    horizon: int = 20,
) -> pl.DataFrame:
    """Build the (ts, symbol) feature/target panel.

    ``bars`` needs symbol/ts/close/volume; ``bench`` needs ts/close. Both must
    be adjusted prices (the BarStore's default read), otherwise the momentum
    features and the excess-return target are fiction around every ex-date.

    Rows appear once every feature is computable (~12 months of history);
    ``target`` stays null for the trailing ``horizon`` days, which is exactly
    the set of rows a walk-forward fit at the panel's edge must not train on.
    """
    ret = pl.col("close") / pl.col("close").shift(1) - 1.0
    over = {"partition_by": "symbol", "order_by": "ts"}

    panel = (
        bars.sort("symbol", "ts")
        .with_columns(
            mom_1m=(pl.col("close") / pl.col("close").shift(_M1) - 1.0).over(**over),
            mom_3m=(pl.col("close") / pl.col("close").shift(_M3) - 1.0).over(**over),
            # 12-1 momentum: skip the most recent month, which at the stock
            # level is reversal territory, not momentum.
            mom_12m=(pl.col("close").shift(_M1) / pl.col("close").shift(_M12) - 1.0).over(**over),
            vol_20d=ret.rolling_std(20).over(**over),
            vol_60d=ret.rolling_std(60).over(**over),
            adv_20d=(pl.col("close") * pl.col("volume")).rolling_mean(20).over(**over),
            fwd=(pl.col("close").shift(-horizon) / pl.col("close") - 1.0).over(**over),
        )
        .select("symbol", "ts", "close", *FEATURES[:-1], "fwd")
    )

    bench_fwd = bench.sort("ts").select(
        "ts",
        bench_fwd=pl.col("close").shift(-horizon) / pl.col("close") - 1.0,
    )

    return (
        panel.join(bench_fwd, on="ts", how="left")
        .with_columns(
            # Where this name's momentum sits in today's cross-section, in
            # [0, 1]. Itself predictive, and cheap.
            cross_sectional_rank=(
                (pl.col("mom_3m").rank("average").over("ts") - 1.0)
                / (pl.col("mom_3m").count().over("ts") - 1.0).clip(lower_bound=1)
            ),
            target=pl.col("fwd") - pl.col("bench_fwd"),
        )
        .drop("fwd", "bench_fwd", "close")
        .drop_nulls(subset=FEATURES)
        .sort("ts", "symbol")
    )
