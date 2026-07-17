"""Technical features.

**The one rule in this file:** every feature at bar ``t`` may use bars
``<= t`` and nothing else. Look-ahead bias here is invisible in a backtest and
fatal in production -- it is the single most common way a retail system produces
a spectacular equity curve that evaporates live.

Practically that means: rolling windows never centre, normalisation statistics
come from trailing windows only (never a full-sample mean/std), and the label
side is somebody else's problem. Bars with insufficient history yield NaN rather
than a partial-window guess.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from swingbot.config import FeatureConfig
from swingbot.features.fracdiff import frac_diff_ffd


def _rsi(close: pl.Expr, window: int) -> pl.Expr:
    """Wilder's RSI. Uses an EWM approximation of Wilder smoothing."""
    delta = close.diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    avg_gain = gain.ewm_mean(alpha=1.0 / window, adjust=False, ignore_nulls=False)
    avg_loss = loss.ewm_mean(alpha=1.0 / window, adjust=False, ignore_nulls=False)
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _garman_klass_vol(window: int) -> pl.Expr:
    """Garman-Klass volatility: uses OHLC, so ~7x more efficient than close-to-close.

    sigma^2 = 0.5*(ln(H/L))^2 - (2*ln2 - 1)*(ln(C/O))^2, averaged over the window.
    """
    hl = (pl.col("high") / pl.col("low")).log()
    co = (pl.col("close") / pl.col("open")).log()
    var = 0.5 * hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * co.pow(2)
    return var.rolling_mean(window).clip(lower_bound=0.0).sqrt()


def _rolling_zscore(expr: pl.Expr, window: int) -> pl.Expr:
    """Z-score against a *trailing* window.

    Using full-sample statistics here would leak the future into every bar --
    the classic silent look-ahead. The window is trailing and inclusive of t.
    """
    mean = expr.rolling_mean(window)
    std = expr.rolling_std(window)
    return (expr - mean) / (std + 1e-12)


def compute_features(bars: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    """Compute the full feature set for one symbol's bars.

    Expects a single symbol sorted ascending by ``ts``. Returns the input frame
    plus feature columns; rows lacking history carry NaN/null and are dropped by
    ``build_dataset``.
    """
    if bars.is_empty():
        return bars
    if bars["symbol"].n_unique() > 1:
        raise ValueError("compute_features expects one symbol at a time")

    df = bars.sort("ts")
    close, high, low, volume = (pl.col(c) for c in ("close", "high", "low", "volume"))
    exprs: list[pl.Expr] = []

    # --- multi-horizon momentum ---
    for h in cfg.return_horizons:
        exprs.append((close / close.shift(h)).log().alias(f"ret_{h}d"))

    # --- volatility ---
    log_ret = (close / close.shift(1)).log()
    for w in cfg.vol_windows:
        exprs.append((log_ret.rolling_std(w) * np.sqrt(252)).alias(f"vol_{w}d"))
        exprs.append((_garman_klass_vol(w) * np.sqrt(252)).alias(f"gk_vol_{w}d"))

    # --- oscillators ---
    exprs.append((_rsi(close, cfg.rsi_window) / 100.0).alias("rsi"))

    fast, slow, signal = cfg.macd
    ema_fast = close.ewm_mean(span=fast, adjust=False)
    ema_slow = close.ewm_mean(span=slow, adjust=False)
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm_mean(span=signal, adjust=False)
    # Scale by price so MACD is comparable across a $5 stock and a $500 one.
    exprs.append((macd_line / close).alias("macd"))
    exprs.append(((macd_line - macd_signal) / close).alias("macd_hist"))

    # --- mean reversion ---
    bb_mean = close.rolling_mean(cfg.bollinger_window)
    bb_std = close.rolling_std(cfg.bollinger_window)
    exprs.append(((close - bb_mean) / (2.0 * bb_std + 1e-12)).alias("bb_position"))
    for w in (21, 63, 200):
        exprs.append((close / close.rolling_mean(w) - 1.0).alias(f"dist_ma_{w}"))

    # --- volume ---
    exprs.append(_rolling_zscore(volume.log1p(), cfg.zscore_window).alias("volume_z"))
    exprs.append((volume / volume.rolling_mean(21) - 1.0).alias("volume_ratio"))

    # --- range ---
    true_range = pl.max_horizontal(
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    )
    exprs.append((true_range.rolling_mean(14) / close).alias("atr_pct"))

    df = df.with_columns(exprs)

    # --- fractional differentiation (needs materialised values, not an expr) ---
    log_close = np.log(df["close"].to_numpy())
    ffd = frac_diff_ffd(log_close, cfg.fracdiff_d, cfg.fracdiff_threshold)
    df = df.with_columns(pl.Series("ffd_log_close", ffd))
    # FFD output is a level, not a return: z-score it so scale is learnable.
    df = df.with_columns(_rolling_zscore(pl.col("ffd_log_close"), cfg.zscore_window).alias("ffd_z"))

    # Normalise the raw return columns so every feature is roughly unit-scale.
    df = df.with_columns(
        [
            _rolling_zscore(pl.col(f"ret_{h}d"), cfg.zscore_window).alias(f"ret_{h}d_z")
            for h in cfg.return_horizons
        ]
    )
    return df


def feature_columns(cfg: FeatureConfig) -> list[str]:
    """The exact model input columns, in a stable order.

    Order is part of the contract: a saved policy's weights are meaningless if
    column order shifts between training and inference.
    """
    cols = [f"ret_{h}d_z" for h in cfg.return_horizons]
    cols += [f"vol_{w}d" for w in cfg.vol_windows]
    cols += [f"gk_vol_{w}d" for w in cfg.vol_windows]
    cols += ["rsi", "macd", "macd_hist", "bb_position"]
    cols += [f"dist_ma_{w}" for w in (21, 63, 200)]
    cols += ["volume_z", "volume_ratio", "atr_pct", "ffd_z"]
    return cols


def build_dataset(
    bars: pl.DataFrame, cfg: FeatureConfig, *, drop_warmup: bool = True
) -> pl.DataFrame:
    """Compute features for every symbol and drop rows with incomplete history."""
    frames = []
    for _key, group in bars.group_by(["symbol"], maintain_order=True):
        feat = compute_features(group, cfg)
        if drop_warmup and feat.height > cfg.warmup:
            feat = feat.slice(cfg.warmup)
        frames.append(feat)

    if not frames:
        return bars
    out = pl.concat(frames, how="vertical")

    # Replace inf (division artifacts) with null, then drop any row that is not
    # fully populated. A model must never see a fabricated feature value.
    cols = feature_columns(cfg)
    out = out.with_columns(
        [
            pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in cols
        ]
    )
    return out.drop_nulls(subset=cols).sort(["symbol", "ts"])
