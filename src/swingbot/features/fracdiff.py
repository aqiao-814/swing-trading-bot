"""Fractional differentiation (Lopez de Prado, AFML Ch. 5).

The dilemma: raw prices are non-stationary (they break NN training), but simple
returns throw away *all* memory of the level. Fractional differencing ``(1-L)^d``
with real ``d`` in (0,1) interpolates -- enough differencing to reach
stationarity, minimal memory destroyed. AFML reports d < 0.6 suffices for 87 of
the most liquid global futures.

We implement the **fixed-width window** (FFD) variant: weights decay, so we
truncate them below a threshold and use the same finite window for every
observation. The expanding-window variant gives each observation a different
effective window, which silently makes early and late features incomparable.

Worth being blunt about: this is preprocessing, not alpha. It preserves
learnable memory. It does not create signal and it will not save an overfit model.
"""

from __future__ import annotations

import numpy as np


def ffd_weights(d: float, threshold: float = 1e-4, max_width: int = 10_000) -> np.ndarray:
    """Binomial weights for ``(1-L)^d``, truncated where |w| < threshold.

    Recurrence: ``w_0 = 1``, ``w_k = -w_{k-1} * (d - k + 1) / k``.
    Returned oldest-first so it can be convolved directly against a price window.
    """
    if d < 0:
        raise ValueError(f"d must be non-negative, got {d}")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    weights = [1.0]
    k = 1
    while k < max_width:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    # Reverse: index 0 is the oldest observation in the window.
    return np.array(weights[::-1], dtype=np.float64)


def frac_diff_ffd(series: np.ndarray, d: float, threshold: float = 1e-4) -> np.ndarray:
    """Fixed-width fractionally-differenced series.

    Returns an array the same length as ``series`` with the first
    ``len(weights)-1`` entries NaN (insufficient history -- never fabricated).

    Causality guarantee: entry ``i`` uses only ``series[i-width+1 : i+1]``. It is
    mathematically incapable of seeing the future.
    """
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"expected 1-D series, got shape {x.shape}")

    w = ffd_weights(d, threshold)
    width = len(w)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if width > len(x):
        return out

    # Sliding dot product of the weight window against the trailing history.
    windows = np.lib.stride_tricks.sliding_window_view(x, width)
    valid = windows @ w
    out[width - 1 :] = valid
    # Any NaN in the input window poisons its output; make that explicit.
    out[width - 1 :][np.isnan(windows).any(axis=1)] = np.nan
    return out


def min_ffd_order(
    series: np.ndarray,
    *,
    threshold: float = 1e-4,
    candidates: np.ndarray | None = None,
    pvalue_target: float = 0.05,
) -> tuple[float, float]:
    """Smallest ``d`` whose FFD series passes an ADF stationarity test.

    Returns ``(d, pvalue)``. This is the AFML recipe: search upward and stop at
    the first order that achieves stationarity, keeping maximum memory.

    Falls back to a variance-ratio heuristic if statsmodels isn't installed.
    """
    if candidates is None:
        candidates = np.linspace(0.0, 1.0, 11)

    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:  # pragma: no cover - optional dependency
        adfuller = None

    for d in candidates:
        diffed = frac_diff_ffd(series, float(d), threshold)
        clean = diffed[~np.isnan(diffed)]
        if len(clean) < 100:
            continue
        if adfuller is None:
            # Crude proxy: a stationary series' variance stops growing with n.
            first, second = clean[: len(clean) // 2], clean[len(clean) // 2 :]
            ratio = np.var(second) / max(np.var(first), 1e-12)
            if 0.5 < ratio < 2.0:
                return float(d), float("nan")
            continue
        pvalue = adfuller(clean, maxlag=1, regression="c", autolag=None)[1]
        if pvalue < pvalue_target:
            return float(d), float(pvalue)

    return float(candidates[-1]), float("nan")
