"""The trading environment: a Gymnasium POMDP over daily bars.

Timing contract (the part that makes or breaks realism)
-------------------------------------------------------
Bar ``t`` closes. The agent sees features built only from bars ``<= t`` and
chooses a *target position*. That order does **not** fill at bar ``t``'s close --
it fills at bar ``t+1``'s **open**, at a price degraded by spread, slippage and
impact. The position is then marked to market at bar ``t+1``'s close.

This one-bar delay is not a detail. Filling at the close you made the decision on
is the most common look-ahead bug in retail backtests and it manufactures
enormous fake alpha. Here it is structurally impossible: ``step()`` advances the
clock *before* it executes.

Overnight gap risk is modelled honestly too: a stop-loss does not guarantee the
stop price. If the market gaps through the stop overnight, the fill is at the
open. That asymmetry is precisely the risk a swing trader carries and cannot hedge.

The agent is never told the capital is simulated. Observations contain position
and P&L state, never a "this is fake" flag -- so a policy trained here makes the
same decisions it would make with real money.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium import spaces

from swingbot.config import ActionSpaceKind, EnvConfig
from swingbot.execution import ExecutionModel, MarketContext
from swingbot.portfolio import Portfolio
from swingbot.rewards import build_reward

# Discrete action index -> target position as a fraction of equity.
DISCRETE_POSITIONS = (-1.0, 0.0, 1.0)  # short, flat, long


@dataclass
class TradeRecord:
    """One decision and everything that followed from it, for later analysis."""

    ts: date
    symbol: str
    action: float
    target_position: float
    prior_position: float
    fill_price: float | None
    quantity: float
    commission: float
    fees: float
    slippage: float
    equity: float
    cash: float
    reward: float
    net_return: float
    exit_reason: str | None = None


@dataclass
class EpisodeResult:
    """Everything a run produced. Consumed by metrics/reporting."""

    equity: np.ndarray
    returns: np.ndarray
    positions: np.ndarray
    timestamps: list[date]
    trades: list[TradeRecord] = field(default_factory=list)
    starting_capital: float = 0.0
    total_costs: float = 0.0  # all-in: explicit + slippage
    explicit_costs: float = 0.0  # commission + fees + borrow (cash debits)
    slippage_costs: float = 0.0  # spread + slippage + impact (worse fills)
    halted_reason: str | None = None


class SwingTradingEnv(gym.Env):
    """Single-asset swing-trading environment over daily bars.

    Parameters
    ----------
    bars:
        One symbol's feature-complete frame (output of ``build_dataset``),
        sorted ascending by ``ts``. Must contain OHLC + the feature columns.
    feature_cols:
        Exact, ordered model inputs. Order is part of the policy contract.
    cfg:
        Environment configuration (capital, costs, risk, reward).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        bars: pl.DataFrame,
        feature_cols: list[str],
        cfg: EnvConfig | None = None,
        *,
        symbol: str | None = None,
        random_start: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.feature_cols = list(feature_cols)
        self.random_start = random_start

        if bars.is_empty():
            raise ValueError("bars frame is empty")
        if bars["symbol"].n_unique() > 1:
            raise ValueError("SwingTradingEnv handles one symbol; got several")
        missing = set(self.feature_cols) - set(bars.columns)
        if missing:
            raise ValueError(f"bars missing feature columns: {sorted(missing)}")

        bars = bars.sort("ts")
        self.symbol = symbol or str(bars["symbol"][0])

        # Materialise to numpy once: the env is stepped millions of times and
        # per-step dataframe indexing would dominate the runtime.
        self._features = bars.select(self.feature_cols).to_numpy().astype(np.float32)
        self._open = bars["open"].to_numpy().astype(np.float64)
        self._high = bars["high"].to_numpy().astype(np.float64)
        self._low = bars["low"].to_numpy().astype(np.float64)
        self._close = bars["close"].to_numpy().astype(np.float64)
        self._volume = bars["volume"].to_numpy().astype(np.float64)
        self._ts: list[date] = bars["ts"].to_list()
        # Realised vol drives the impact model; fall back to a sane default.
        vol_col = next((c for c in bars.columns if c.startswith("vol_")), None)
        daily_vol = (
            bars[vol_col].to_numpy() / np.sqrt(252) if vol_col else np.full(bars.height, 0.02)
        )
        self._daily_vol = np.nan_to_num(daily_vol, nan=0.02).astype(np.float64)

        self.n_bars = len(self._close)
        if self.n_bars < 3:
            raise ValueError(f"need at least 3 bars, got {self.n_bars}")

        self.execution = ExecutionModel(self.cfg.costs)
        self.portfolio = Portfolio(self.cfg.starting_capital)
        self.reward_fn = build_reward(self.cfg.reward, dsr_eta=self.cfg.dsr_eta)

        # --- spaces ---
        if self.cfg.action_space is ActionSpaceKind.DISCRETE:
            self.action_space = spaces.Discrete(len(DISCRETE_POSITIONS))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Features + [position, unrealized_pnl_pct, bars_held_scaled, drawdown]
        n_obs = len(self.feature_cols) + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )

        self._rng = np.random.default_rng()
        self.reset()

    # ---- gym API ---------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        length = self.cfg.episode_length
        if length is None or length >= self.n_bars - 1:
            self._start = 0
            self._end = self.n_bars - 1
        elif self.random_start:
            self._start = int(self._rng.integers(0, self.n_bars - length - 1))
            self._end = self._start + length
        else:
            self._start = 0
            self._end = length

        self._i = self._start
        self.portfolio.reset()
        self.reward_fn.reset()

        self._target_position = 0.0
        self._entry_price: float | None = None
        self._bars_held = 0
        self._peak_equity = self.cfg.starting_capital
        self._day_start_equity = self.cfg.starting_capital
        self._halted: str | None = None

        self._equity_curve = [self.cfg.starting_capital]
        self._returns: list[float] = []
        self._positions: list[float] = [0.0]
        self._episode_ts: list[date] = [self._ts[self._i]]
        self._trades: list[TradeRecord] = []

        return self._observe(), self._info()

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._i >= self._end:
            raise RuntimeError("step() called on a finished episode; call reset()")

        target = self._decode_action(action)
        if self._halted is not None:
            target = 0.0  # risk overlay has flattened us; stay flat

        prior_position = self._current_weight()
        equity_before = self.portfolio.equity({self.symbol: self._close[self._i]})

        # --- THE CLOCK ADVANCES BEFORE ANY FILL. This is the whole ballgame. ---
        self._i += 1
        fill_idx = self._i

        exit_reason = self._check_risk_exits(fill_idx)
        if exit_reason is not None:
            target = 0.0

        fill = self._rebalance_to(target, fill_idx)

        # Mark to market at the new bar's close.
        close = self._close[fill_idx]
        self._accrue_borrow(close)
        equity_after = self.portfolio.equity({self.symbol: close})

        net_return = (equity_after / equity_before - 1.0) if equity_before > 0 else -1.0
        reward = self.reward_fn.update(net_return)

        self._update_risk_state(equity_after)
        self._record(fill, target, prior_position, equity_after, reward, net_return, exit_reason)

        terminated = self._halted is not None or equity_after <= 0
        truncated = self._i >= self._end
        return self._observe(), float(reward), bool(terminated), bool(truncated), self._info()

    # ---- action / position -----------------------------------------------

    def _decode_action(self, action) -> float:
        """Map a raw agent action to a target position weight in [-1, 1]."""
        if self.cfg.action_space is ActionSpaceKind.DISCRETE:
            idx = int(np.asarray(action).item())
            if not 0 <= idx < len(DISCRETE_POSITIONS):
                raise ValueError(f"discrete action {idx} out of range")
            target = DISCRETE_POSITIONS[idx]
        else:
            target = float(np.clip(np.asarray(action).astype(np.float64).ravel()[0], -1.0, 1.0))

        risk = self.cfg.risk
        if not risk.allow_short:
            target = max(target, 0.0)
        target = float(np.clip(target, -risk.max_position_weight, risk.max_position_weight))
        return self._apply_vol_target(target)

    def _apply_vol_target(self, target: float) -> float:
        """Scale exposure inversely to realised vol, capped by gross limit.

        Volatility targeting holds *risk* constant rather than *notional*, which
        is what keeps a fixed stop-loss meaningful across calm and violent regimes.
        """
        risk = self.cfg.risk
        if risk.vol_target_annual is None:
            return target
        realized = self._daily_vol[self._i] * np.sqrt(252)
        if realized <= 1e-6:
            return target
        scale = min(risk.vol_target_annual / realized, risk.max_gross_exposure)
        return float(np.clip(target * scale * risk.kelly_fraction, -1.0, 1.0))

    def _current_weight(self) -> float:
        close = self._close[self._i]
        equity = self.portfolio.equity({self.symbol: close})
        if equity <= 0:
            return 0.0
        return self.portfolio.quantity(self.symbol) * close / equity

    def _rebalance_to(self, target_weight: float, idx: int):
        """Trade toward the target weight, filling at bar ``idx``'s open.

        Applies a no-trade band: small drifts are ignored. Rebalancing on every
        tick of noise generates thousands of trades whose costs dwarf any
        tracking benefit -- the band is what makes vol targeting affordable.
        """
        ref_price = self._open[idx]
        equity = self.portfolio.equity({self.symbol: ref_price})
        if equity <= 0:
            return None

        current_weight = self.portfolio.quantity(self.symbol) * ref_price / equity
        drift = abs(target_weight - current_weight)
        band = self.cfg.risk.rebalance_threshold
        # Always allow a full exit; never let the band strand a position we want
        # closed (a stop-loss must not be filtered out by a rebalancing rule).
        closing = abs(target_weight) < 1e-9 and abs(current_weight) > 1e-9
        if drift < band and not closing:
            return None

        desired_shares = float(round(target_weight * equity / ref_price))
        delta = desired_shares - self.portfolio.quantity(self.symbol)
        if abs(delta) < 1.0:  # never trade a fractional share
            return None

        ctx = MarketContext(
            ts=self._ts[idx],
            symbol=self.symbol,
            reference_price=ref_price,
            volatility=float(self._daily_vol[idx]),
            volume=float(self._volume[idx]),
        )
        fill = self.execution.build_fill(delta, ctx)
        if fill is None:
            return None

        was_flat = self.portfolio.position(self.symbol).is_flat
        self.portfolio.execute(fill)
        self._target_position = target_weight

        if self.portfolio.position(self.symbol).is_flat:
            self._entry_price, self._bars_held = None, 0
        elif was_flat:
            self._entry_price, self._bars_held = fill.price, 0
        return fill

    # ---- risk ------------------------------------------------------------

    def _check_risk_exits(self, idx: int) -> str | None:
        """Stop-loss / take-profit, evaluated against bar ``idx``'s range.

        Honest about gaps: we only report *whether* the level traded. The fill
        price comes from ``_rebalance_to`` at the open, so a stop that gaps is
        filled at the gapped open -- not at the stop price. That is what actually
        happens to swing traders and it is the risk that cannot be hedged away.
        """
        pos = self.portfolio.position(self.symbol)
        if pos.is_flat or self._entry_price is None:
            return None

        risk = self.cfg.risk
        low, high = self._low[idx], self._high[idx]
        entry = self._entry_price

        if pos.is_long:
            if risk.stop_loss_pct and low <= entry * (1 - risk.stop_loss_pct):
                return "stop_loss"
            if risk.take_profit_pct and high >= entry * (1 + risk.take_profit_pct):
                return "take_profit"
        else:
            if risk.stop_loss_pct and high >= entry * (1 + risk.stop_loss_pct):
                return "stop_loss"
            if risk.take_profit_pct and low <= entry * (1 - risk.take_profit_pct):
                return "take_profit"
        return None

    def _update_risk_state(self, equity: float) -> None:
        """Portfolio-level kill switches. These flatten and halt, permanently."""
        self._peak_equity = max(self._peak_equity, equity)
        risk = self.cfg.risk

        if risk.max_drawdown_pct is not None:
            dd = 1.0 - equity / self._peak_equity
            if dd >= risk.max_drawdown_pct:
                self._halted = f"max_drawdown ({dd:.1%})"
                return
        if risk.max_daily_loss_pct is not None:
            daily = 1.0 - equity / self._day_start_equity
            if daily >= risk.max_daily_loss_pct:
                self._halted = f"max_daily_loss ({daily:.1%})"
                return
        self._day_start_equity = equity
        if not self.portfolio.position(self.symbol).is_flat:
            self._bars_held += 1

    def _accrue_borrow(self, close: float) -> None:
        pos = self.portfolio.position(self.symbol)
        if pos.is_short:
            cost = self.execution.borrow_cost(pos.market_value(close), days=1.0)
            if cost > 0:
                self.portfolio.charge(cost, symbol=self.symbol)

    # ---- observation -----------------------------------------------------

    def _observe(self) -> np.ndarray:
        """Features plus the agent's own state.

        Including position is not optional: without it the process is not
        Markovian, because the cost of an action depends on what you already hold.
        """
        close = self._close[self._i]
        equity = self.portfolio.equity({self.symbol: close})
        pos = self.portfolio.position(self.symbol)

        weight = self._current_weight()
        unrealized = pos.unrealized_pnl(close) / equity if equity > 0 else 0.0
        held = min(self._bars_held / 21.0, 5.0)  # scaled to ~months
        drawdown = 1.0 - equity / self._peak_equity if self._peak_equity > 0 else 0.0

        obs = np.concatenate(
            [
                self._features[self._i],
                np.array([weight, unrealized, held, drawdown], dtype=np.float32),
            ]
        )
        # float32 throughout: MPS cannot handle float64 tensors at all.
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        close = self._close[self._i]
        return {
            "ts": self._ts[self._i],
            "equity": self.portfolio.equity({self.symbol: close}),
            "cash": self.portfolio.cash,
            "position": self._current_weight(),
            "costs": self.portfolio.cumulative_costs,
            "halted": self._halted,
        }

    def _record(self, fill, target, prior, equity, reward, net_return, exit_reason) -> None:
        self._equity_curve.append(equity)
        self._returns.append(net_return)
        self._positions.append(self._current_weight())
        self._episode_ts.append(self._ts[self._i])
        self._trades.append(
            TradeRecord(
                ts=self._ts[self._i],
                symbol=self.symbol,
                action=target,
                target_position=target,
                prior_position=prior,
                fill_price=fill.price if fill else None,
                quantity=fill.quantity if fill else 0.0,
                commission=fill.commission if fill else 0.0,
                fees=fill.fees if fill else 0.0,
                slippage=fill.slippage_cost if fill else 0.0,
                equity=equity,
                cash=self.portfolio.cash,
                reward=reward,
                net_return=net_return,
                exit_reason=exit_reason,
            )
        )

    # ---- results ---------------------------------------------------------

    def result(self) -> EpisodeResult:
        return EpisodeResult(
            equity=np.array(self._equity_curve),
            returns=np.array(self._returns),
            positions=np.array(self._positions),
            timestamps=list(self._episode_ts),
            trades=list(self._trades),
            starting_capital=self.cfg.starting_capital,
            # All-in: commission + fees + borrow, plus the slippage/spread that
            # is embedded in fill prices rather than charged as cash.
            total_costs=self.portfolio.all_in_costs,
            explicit_costs=self.portfolio.cumulative_costs,
            slippage_costs=self.portfolio.cumulative_slippage,
            halted_reason=self._halted,
        )

    def render(self) -> None:
        info = self._info()
        pnl = info["equity"] / self.cfg.starting_capital - 1.0
        print(
            f"{info['ts']} | equity ${info['equity']:>12,.2f} | {pnl:>+7.2%} "
            f"| pos {info['position']:>+.2f} | costs ${info['costs']:>8,.2f}"
        )
