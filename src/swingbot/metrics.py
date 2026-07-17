"""Performance metrics, including the ones that tell you the truth.

Most of this file is standard fund analytics: Sharpe, Sortino, Calmar, max
drawdown, win rate, turnover. The important part is at the bottom.

``probabilistic_sharpe_ratio`` and ``deflated_sharpe_ratio`` (Bailey & Lopez de
Prado) exist because a plain Sharpe ratio from a backtest is not evidence. If
you try 1,000 configurations, the best one's Sharpe is inflated by selection
bias alone -- you would see an impressive number on pure noise. The DSR corrects
for exactly that, and it is the difference between research and self-deception.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

TRADING_DAYS = 252
_EULER_MASCHERONI = 0.5772156649015329


# ---- core return statistics ---------------------------------------------


def total_return(equity: np.ndarray) -> float:
    if len(equity) < 2 or equity[0] <= 0:
        return 0.0
    return float(equity[-1] / equity[0] - 1.0)


def cagr(equity: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    if len(equity) < 2 or equity[0] <= 0 or equity[-1] <= 0:
        return 0.0
    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0)


def annualized_volatility(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: np.ndarray, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualized Sharpe. Returns 0 for a degenerate (zero-variance) series."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods_per_year
    sd = np.std(excess, ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(periods_per_year))


def excess_sharpe(
    equity: np.ndarray, bench_equity: np.ndarray, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualized Sharpe of the strategy-minus-benchmark return stream.

    Deflating a long-only equity book against zero asks the wrong question:
    every long-only strategy has a positive Sharpe in a bull market. This
    asks the one that matters -- is anything left after the benchmark? -- and
    it is the number a "beat the market" claim must be judged on.
    """
    n = min(len(equity), len(bench_equity))
    if n < 3:
        return 0.0
    e, b = np.asarray(equity[:n], float), np.asarray(bench_equity[:n], float)
    if np.any(e[:-1] <= 0) or np.any(b[:-1] <= 0):
        return 0.0
    return sharpe_ratio(
        np.diff(e) / e[:-1] - np.diff(b) / b[:-1], periods_per_year=periods_per_year
    )


def sortino_ratio(
    returns: np.ndarray, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Like Sharpe but only punishes downside deviation."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if np.mean(excess) > 0 else 0.0
    dd = np.sqrt(np.mean(downside**2))
    if dd <= 1e-12:
        return 0.0
    return float(np.mean(excess) / dd * np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    """Worst peak-to-trough decline, as a positive fraction."""
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / np.where(peak <= 0, np.nan, peak)))


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return (peak - equity) / np.where(peak <= 0, np.nan, peak)


def max_drawdown_duration(equity: np.ndarray) -> int:
    """Longest stretch (in bars) spent below a prior high-water mark."""
    if len(equity) < 2:
        return 0
    peak = np.maximum.accumulate(equity)
    underwater = equity < peak
    longest = current = 0
    for u in underwater:
        current = current + 1 if u else 0
        longest = max(longest, current)
    return int(longest)


def calmar_ratio(equity: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(equity)
    if mdd <= 1e-12:
        return 0.0
    return float(cagr(equity, periods_per_year) / mdd)


def value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    if len(returns) == 0:
        return 0.0
    return float(np.quantile(returns, alpha))


def conditional_value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Expected loss conditional on breaching the VaR threshold (tail risk)."""
    if len(returns) == 0:
        return 0.0
    var = value_at_risk(returns, alpha)
    tail = returns[returns <= var]
    return float(np.mean(tail)) if len(tail) else var


# ---- trade statistics ----------------------------------------------------


def win_rate(trade_pnls: np.ndarray) -> float:
    if len(trade_pnls) == 0:
        return 0.0
    return float(np.mean(trade_pnls > 0))


def profit_factor(trade_pnls: np.ndarray) -> float:
    """Gross wins / gross losses. inf when there are no losses."""
    if len(trade_pnls) == 0:
        return 0.0
    wins = trade_pnls[trade_pnls > 0].sum()
    losses = -trade_pnls[trade_pnls < 0].sum()
    if losses <= 1e-12:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def turnover(positions: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualized position turnover -- the driver of cost drag."""
    if len(positions) < 2:
        return 0.0
    return float(np.abs(np.diff(positions)).sum() / len(positions) * periods_per_year)


# ---- overfitting-aware statistics ---------------------------------------


def probabilistic_sharpe_ratio(
    returns: np.ndarray, benchmark_sr: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """P(true Sharpe > benchmark), correcting for skew, kurtosis and sample size.

    A high Sharpe from 30 observations of a fat-tailed series is not the same
    evidence as the same Sharpe from 3,000 -- this is what quantifies that gap.
    """
    n = len(returns)
    if n < 3:
        return 0.0
    # A flat/constant series has no variance, so skew and kurtosis are undefined
    # and every downstream term becomes NaN. There is no evidence of skill here
    # either way, so report 0 rather than propagating NaN into the report.
    if np.std(returns, ddof=1) <= 1e-12:
        return 0.0

    sr = sharpe_ratio(returns, periods_per_year=periods_per_year) / np.sqrt(periods_per_year)
    bench = benchmark_sr / np.sqrt(periods_per_year)
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=False))

    # Variance of the Sharpe estimator under non-normal returns.
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 1e-12 or not np.isfinite(denom):
        return 0.0
    z = (sr - bench) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z)) if np.isfinite(z) else 0.0


def expected_max_sharpe(n_trials: int, sr_variance: float = 1.0) -> float:
    """Expected maximum Sharpe from ``n_trials`` independent *skill-less* trials.

    This is the bar a strategy must clear to be interesting. Try enough random
    strategies and one will look brilliant; this says how brilliant, by luck alone.
    """
    if n_trials < 2:
        return 0.0
    sd = np.sqrt(sr_variance)
    # Bailey & Lopez de Prado's Gumbel-based approximation.
    q1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    q2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sd * ((1.0 - _EULER_MASCHERONI) * q1 + _EULER_MASCHERONI * q2))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    *,
    sr_variance: float | None = None,
    periods_per_year: int = TRADING_DAYS,
    trial_sharpes: np.ndarray | None = None,
) -> float:
    """PSR against the Sharpe you'd expect from the best of ``n_trials`` flukes.

    Interpretation: DSR is the probability the strategy's true Sharpe exceeds
    what selection bias alone would have produced. Below ~0.95, you have not
    demonstrated skill -- you have demonstrated that you searched.

    ``trial_sharpes`` (the Sharpes of every configuration you tried) gives a far
    better variance estimate than the default; supply it when you have it.
    """
    if len(returns) < 3:
        return 0.0
    if sr_variance is None:
        if trial_sharpes is not None and len(trial_sharpes) > 1:
            sr_variance = float(np.var(np.asarray(trial_sharpes), ddof=1)) / periods_per_year
        else:
            # Fall back to the estimator's own sampling variance.
            sr_variance = 1.0 / (len(returns) - 1)

    sr0 = expected_max_sharpe(n_trials, sr_variance) * np.sqrt(periods_per_year)
    return probabilistic_sharpe_ratio(returns, sr0, periods_per_year)


def minimum_track_record_length(
    returns: np.ndarray,
    benchmark_sr: float = 0.0,
    confidence: float = 0.95,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Observations needed to claim SR > benchmark at ``confidence``.

    If this exceeds the data you actually have, your result is not yet a result.
    """
    n = len(returns)
    if n < 3:
        return float("inf")
    sr = sharpe_ratio(returns, periods_per_year=periods_per_year) / np.sqrt(periods_per_year)
    bench = benchmark_sr / np.sqrt(periods_per_year)
    if sr <= bench:
        return float("inf")
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=False))
    z = stats.norm.ppf(confidence)
    denom = (sr - bench) ** 2
    if denom <= 1e-12:
        return float("inf")
    return float(1.0 + (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) * (z / (sr - bench)) ** 2)


# ---- report --------------------------------------------------------------


@dataclass
class PerformanceReport:
    """The full institutional-style metric set for one run."""

    starting_capital: float
    ending_equity: float
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_days: int
    var_95: float
    cvar_95: float
    win_rate: float
    profit_factor: float
    n_trades: int
    turnover: float
    total_costs: float
    cost_drag: float  # costs as a fraction of starting capital
    psr: float
    dsr: float
    min_track_record: float
    halted_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_report(
    equity: np.ndarray,
    returns: np.ndarray,
    positions: np.ndarray,
    *,
    total_costs: float = 0.0,
    n_trials: int = 1,
    trade_pnls: np.ndarray | None = None,
    n_trades: int | None = None,
    periods_per_year: int = TRADING_DAYS,
    halted_reason: str | None = None,
) -> PerformanceReport:
    """Assemble every metric for a single equity curve.

    ``n_trades`` should be the count of actual fills. Do not infer it from the
    position series: a held position's *weight* drifts every bar as prices move,
    so counting weight changes reports buy-and-hold as thousands of trades.
    """
    equity = np.asarray(equity, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    if trade_pnls is None:
        # Fall back to per-bar P&L when discrete trades aren't supplied.
        trade_pnls = np.diff(equity) if len(equity) > 1 else np.array([])

    if n_trades is None:
        # Last resort: count only material weight jumps, not price drift.
        n_trades = (
            int(np.count_nonzero(np.abs(np.diff(positions)) > 0.01)) if len(positions) > 1 else 0
        )
    start = float(equity[0]) if len(equity) else 0.0

    return PerformanceReport(
        starting_capital=start,
        ending_equity=float(equity[-1]) if len(equity) else 0.0,
        total_return=total_return(equity),
        cagr=cagr(equity, periods_per_year),
        annual_volatility=annualized_volatility(returns, periods_per_year),
        sharpe=sharpe_ratio(returns, periods_per_year=periods_per_year),
        sortino=sortino_ratio(returns, periods_per_year=periods_per_year),
        calmar=calmar_ratio(equity, periods_per_year),
        max_drawdown=max_drawdown(equity),
        max_drawdown_days=max_drawdown_duration(equity),
        var_95=value_at_risk(returns),
        cvar_95=conditional_value_at_risk(returns),
        win_rate=win_rate(trade_pnls),
        profit_factor=profit_factor(trade_pnls),
        n_trades=int(n_trades),
        turnover=turnover(positions, periods_per_year),
        total_costs=float(total_costs),
        cost_drag=float(total_costs / start) if start > 0 else 0.0,
        psr=probabilistic_sharpe_ratio(returns, 0.0, periods_per_year),
        dsr=deflated_sharpe_ratio(returns, n_trials, periods_per_year=periods_per_year),
        min_track_record=minimum_track_record_length(returns, 0.0, 0.95, periods_per_year),
        halted_reason=halted_reason,
    )
