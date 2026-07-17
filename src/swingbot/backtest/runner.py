"""Backtest execution: run an agent through an environment, collect results."""

from __future__ import annotations

import numpy as np
import polars as pl

from swingbot.agents.baselines import Agent
from swingbot.config import EnvConfig
from swingbot.env.trading_env import EpisodeResult, SwingTradingEnv
from swingbot.metrics import PerformanceReport, build_report


def run_episode(env: SwingTradingEnv, agent: Agent, *, seed: int | None = None) -> EpisodeResult:
    """Play one full episode. Deterministic given a seed and a deterministic agent."""
    obs, _ = env.reset(seed=seed)
    agent.reset()

    while True:
        action = agent.act(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return env.result()


def evaluate(
    dataset: pl.DataFrame,
    feature_cols: list[str],
    agent: Agent,
    cfg: EnvConfig,
    *,
    seed: int | None = None,
    n_trials: int = 1,
) -> tuple[EpisodeResult, PerformanceReport]:
    """Run an agent over a dataset and produce the full metric report.

    ``n_trials`` should be the number of configurations you tried before picking
    this one -- it feeds the Deflated Sharpe. Leaving it at 1 while having tested
    hundreds is how people fool themselves.
    """
    env = SwingTradingEnv(dataset, feature_cols, cfg, random_start=False)
    result = run_episode(env, agent, seed=seed)

    # Count real fills, not weight drift: a held position's weight moves every
    # bar with price, which would report buy-and-hold as thousands of trades.
    n_fills = sum(1 for t in result.trades if abs(t.quantity) > 0)

    report = build_report(
        result.equity,
        result.returns,
        result.positions,
        total_costs=result.total_costs,
        n_trials=n_trials,
        n_trades=n_fills,
        trade_pnls=round_trip_pnls(result),
        periods_per_year=cfg.trading_days_per_year,
        halted_reason=result.halted_reason,
    )
    return result, report


def round_trip_pnls(result: EpisodeResult) -> np.ndarray:
    """P&L of each completed round trip, for win-rate and profit-factor.

    A "trade" for reporting purposes is a full position lifecycle: from leaving
    flat to returning to flat. Per-bar P&L would answer a different (and far
    less useful) question -- "what fraction of days were up?".
    """
    pnls: list[float] = []
    entry_equity: float | None = None

    for trade in result.trades:
        was_flat = abs(trade.prior_position) < 1e-6
        now_flat = abs(trade.target_position) < 1e-6

        if was_flat and not now_flat:
            entry_equity = trade.equity  # opened a position
        elif not was_flat and now_flat and entry_equity is not None:
            pnls.append(trade.equity - entry_equity)  # closed it
            entry_equity = None

    # An open position at the end is not a completed trade; exclude it.
    return np.array(pnls)


def train_rrl(
    dataset: pl.DataFrame,
    feature_cols: list[str],
    agent,
    cfg: EnvConfig,
    *,
    epochs: int = 20,
    cost_bps: float | None = None,
) -> list[float]:
    """Train an RRL agent by online gradient ascent on the differential Sharpe.

    Trains directly on the feature/return series rather than through the env:
    RRL needs the raw per-bar return to compute its gradient, and stepping the
    full execution simulator per epoch would be far slower for no benefit. The
    *evaluation* still goes through the env, so costs and delays are enforced
    where it counts.
    """
    x = dataset.select(feature_cols).to_numpy().astype(np.float64)
    close = dataset["close"].to_numpy().astype(np.float64)
    returns = np.diff(close) / close[:-1]

    if cost_bps is None:
        # Round-trip cost the agent should learn to respect.
        cost_bps = 2.0 * (cfg.costs.half_spread_bps + cfg.costs.slippage_bps)
    cost = cost_bps * 1e-4

    sharpes: list[float] = []
    for _ in range(epochs):
        agent.reset()
        for t in range(len(returns)):
            agent.update(x[t], float(returns[t]), cost)
        sharpes.append(agent.sharpe)
    return sharpes
