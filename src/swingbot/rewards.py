"""Reward functions.

Reward design is the make-or-break decision in trading RL. Raw P&L produces
degenerate policies: the agent churns (it cannot "see" costs it isn't charged
for) or holds one lucky leveraged position. The defaults here are risk-adjusted
and cost-aware by construction.

The centrepiece is Moody & Saffell's **differential Sharpe ratio** (Learning to
Trade via Direct Reinforcement, IEEE TNN 2001), the exponentially-weighted
online derivative of the Sharpe ratio. It gives a dense, incremental,
risk-adjusted signal, which is exactly what an RL agent needs -- and Moody &
Saffell found maximising it yields more consistent results than maximising
profit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from swingbot.config import RewardKind

# Below this variance the Sharpe derivative is numerically meaningless.
_VAR_EPS = 1e-12


class Reward(ABC):
    """Maps a per-bar net return to a scalar learning signal."""

    @abstractmethod
    def update(self, net_return: float) -> float:
        """Consume one bar's net return, emit the reward for that bar."""

    @abstractmethod
    def reset(self) -> None: ...


class DifferentialSharpeRatio(Reward):
    """Moody & Saffell's online differential Sharpe ratio.

    Maintains exponentially-weighted first and second moments of returns::

        A_t = A_{t-1} + eta * (R_t - A_{t-1})
        B_t = B_{t-1} + eta * (R_t^2 - B_{t-1})

    and emits the derivative of the Sharpe ratio w.r.t. the adaptation rate::

        D_t = (B_{t-1} * dA - 0.5 * A_{t-1} * dB) / (B_{t-1} - A_{t-1}^2)^(3/2)

    where ``dA = R_t - A_{t-1}`` and ``dB = R_t^2 - B_{t-1}``.

    Intuition worth keeping: the numerator rewards return above the running mean
    but *penalises* it in proportion to how much it inflates running variance,
    so a big lucky gain scores worse than a steady one.
    """

    def __init__(self, eta: float = 0.01) -> None:
        if not 0.0 < eta < 1.0:
            raise ValueError(f"eta must be in (0, 1), got {eta}")
        self.eta = eta
        self.reset()

    def reset(self) -> None:
        self.a = 0.0  # EW mean of returns
        self.b = 0.0  # EW mean of squared returns
        self._steps = 0

    def update(self, net_return: float) -> float:
        r = float(net_return)
        if not np.isfinite(r):
            raise ValueError(f"net_return must be finite, got {net_return}")

        a_prev, b_prev = self.a, self.b
        d_a = r - a_prev
        d_b = r * r - b_prev

        variance = b_prev - a_prev * a_prev
        if self._steps == 0 or variance <= _VAR_EPS:
            # Not enough history for a meaningful risk adjustment yet. Emit no
            # signal rather than a divide-by-near-zero explosion.
            reward = 0.0
        else:
            reward = (b_prev * d_a - 0.5 * a_prev * d_b) / (variance**1.5)

        self.a = a_prev + self.eta * d_a
        self.b = b_prev + self.eta * d_b
        self._steps += 1
        return float(np.clip(reward, -1e6, 1e6))

    @property
    def sharpe(self) -> float:
        """Current EW Sharpe estimate (per-bar, not annualised)."""
        variance = self.b - self.a * self.a
        if variance <= _VAR_EPS:
            return 0.0
        return self.a / np.sqrt(variance)


class NetReturn(Reward):
    """Plain net return, optionally log-scaled. The honest naive baseline.

    Costs are already netted out by the environment before this sees them, so
    this is not *free* of cost-awareness -- but it has no risk adjustment, and
    will happily accept ruinous variance for a higher mean.
    """

    def __init__(self, *, log: bool = True, scale: float = 1.0) -> None:
        self.log = log
        self.scale = scale

    def reset(self) -> None:  # stateless
        pass

    def update(self, net_return: float) -> float:
        r = float(net_return)
        if self.log:
            # Guard against total-loss bars driving log to -inf.
            r = float(np.log(max(1.0 + r, 1e-8)))
        return r * self.scale


class DrawdownPenalized(Reward):
    """Net return minus a penalty proportional to current drawdown depth.

    Targets the failure mode DSR does not: a policy can have a fine Sharpe and
    still suffer an unacceptable peak-to-trough loss.
    """

    def __init__(self, penalty: float = 1.0, *, log: bool = True) -> None:
        self.penalty = penalty
        self.base = NetReturn(log=log)
        self.reset()

    def reset(self) -> None:
        self._cum = 0.0
        self._peak = 0.0
        self.base.reset()

    def update(self, net_return: float) -> float:
        r = self.base.update(net_return)
        self._cum += r
        self._peak = max(self._peak, self._cum)
        drawdown = self._peak - self._cum  # >= 0, in log-equity units
        return r - self.penalty * drawdown


def build_reward(kind: RewardKind, *, dsr_eta: float = 0.01) -> Reward:
    """Factory used by the environment so reward choice stays config-driven."""
    match kind:
        case RewardKind.DSR:
            return DifferentialSharpeRatio(eta=dsr_eta)
        case RewardKind.NET_LOG_RETURN:
            return NetReturn(log=True)
        case RewardKind.NET_SIMPLE_RETURN:
            return NetReturn(log=False)
        case _:
            raise ValueError(f"unknown reward kind: {kind}")
