"""Regime-trust gate: monitor the model, not the market.

Market-stress proxies (VIX percentile, realized vol) are near-useless -- and
sometimes *inverted* -- at predicting whether a specific ranker will have a
positive-IC day, because they measure how hard the environment is for a
generic investor, not whether this model's factor loadings still work. What
does work is the model's own trailing realized efficacy: an EWMA of RankIC
values whose forward windows have **matured**.

Design decisions, each empirically load-bearing:

* **The maturity lag is the honesty.** At date ``t`` only ICs from
  ``t - horizon`` and earlier are realized; using anything fresher is
  look-ahead. The cost is that a regime break is detected ~a month late --
  the gate is a drawdown *limiter*, not a *preventer*, and must be paired
  with the hard max-drawdown kill switch.

* **Binary abstention, not continuous throttling.** Scaling exposure by the
  health index destroys recovery convexity (you are half-off during the
  rebound that follows a regime failure). Trade or don't.

* **H_real alone does almost all the work.** Drift and cross-model
  disagreement terms buy a rounding error of AUROC for a lot of surface
  area; they are deliberately absent here.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def health_index(
    ic: pl.DataFrame,
    *,
    horizon: int,
    halflife: int = 30,
    min_periods: int = 20,
) -> pl.DataFrame:
    """EWMA of *matured* RankIC: ``ic`` rows lagged by ``horizon`` trading days.

    Input is the per-date IC frame from ``ranker.rank_ic`` (ts, ic). The lag is
    in rows of that frame -- i.e. trading days -- because an IC computed on a
    ``horizon``-day forward return is not knowable until ``horizon`` days later.
    Output: ts, h_real (null until enough matured history exists).
    """
    return ic.sort("ts").with_columns(
        h_real=pl.col("ic")
        .shift(horizon)
        .ewm_mean(half_life=float(halflife), min_samples=min_periods)
    )


def gate_signal(health: pl.DataFrame) -> pl.DataFrame:
    """Map ``h_real`` to a trust score G in [0, 1] via an expanding z-score.

    Expanding (never full-sample) statistics keep the mapping causal: the
    z-score at date t uses only h_real values up to t. Output adds ``g``.
    """
    h = health["h_real"].to_numpy().astype(np.float64)
    n = len(h)
    g = np.full(n, np.nan)
    count, s1, s2 = 0, 0.0, 0.0
    for i in range(n):
        if np.isnan(h[i]):
            continue
        count += 1
        s1 += h[i]
        s2 += h[i] * h[i]
        if count < 2:
            continue
        mean = s1 / count
        var = max(s2 / count - mean * mean, 0.0)
        std = np.sqrt(var)
        z = (h[i] - mean) / std if std > 1e-12 else 0.0
        sig = 1.0 / (1.0 + np.exp(-z))
        g[i] = float(np.clip((sig - 0.3) / 0.4, 0.0, 1.0))
    # NaN -> null so "no gate value yet" is a real null, not a float that
    # silently survives drop_nulls and poisons comparisons.
    return health.with_columns(g=pl.Series(g, dtype=pl.Float64).fill_nan(None))


def should_trade(g: float | None, *, threshold: float = 0.2) -> bool:
    """Binary abstention. No health history yet reads as 'trade' -- the gate
    only earns the right to say no once it has matured evidence."""
    if g is None or np.isnan(g):
        return True
    return g >= threshold
