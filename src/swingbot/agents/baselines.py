"""Baseline strategies -- the bar deep RL has to clear.

The blueprint is blunt about this: if you cannot beat buy-and-hold after costs,
stop and rethink. These exist so that claim is testable rather than rhetorical.

``RRLAgent`` is Moody & Saffell's Recurrent Reinforcement Learning (Learning to
Trade via Direct Reinforcement, IEEE TNN 2001): a single-layer recurrent policy
trained by gradient ascent directly on the differential Sharpe ratio, with no
value function and no bootstrapping. It has ~n_features parameters, which on the
low-SNR, small-sample data of daily bars is a feature, not a limitation -- it is
frequently the honest winner against far larger networks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    """Minimal policy interface shared by baselines and learned agents."""

    @abstractmethod
    def act(self, obs: np.ndarray) -> int | np.ndarray: ...

    def reset(self) -> None:
        """Clear any per-episode state. Stateless agents need not override."""
        return None


class BuyAndHold(Agent):
    """Always long. The benchmark that embarrasses most trading research."""

    def act(self, obs: np.ndarray) -> int:
        return 2  # LONG


class AlwaysFlat(Agent):
    """Never trades. The control: any strategy that loses to cash is worse than
    doing nothing, which is a surprisingly common outcome once costs are real."""

    def act(self, obs: np.ndarray) -> int:
        return 1  # FLAT


class RandomAgent(Agent):
    """Uniform random actions. Establishes the cost of churning for no reason."""

    def __init__(self, n_actions: int = 3, seed: int = 0) -> None:
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(0, self.n_actions))


class MovingAverageCrossover(Agent):
    """Classic trend following on the fast/slow MA distance features.

    Reads ``dist_ma_21`` and ``dist_ma_63`` out of the observation, so it needs
    to be told where they sit in the feature vector.
    """

    def __init__(self, fast_idx: int, slow_idx: int, *, allow_short: bool = True) -> None:
        self.fast_idx = fast_idx
        self.slow_idx = slow_idx
        self.allow_short = allow_short

    def act(self, obs: np.ndarray) -> int:
        fast, slow = obs[self.fast_idx], obs[self.slow_idx]
        if fast > slow:
            return 2  # LONG
        if self.allow_short:
            return 0  # SHORT
        return 1  # FLAT


class RRLAgent(Agent):
    """Moody & Saffell direct reinforcement, trained on the differential Sharpe.

    Policy (continuous position in [-1, 1])::

        F_t = tanh(w . x_t + u * F_{t-1} + b)

    The recurrent term ``u * F_{t-1}`` is what makes the policy cost-aware: the
    agent's own prior position enters the decision, so it can learn that holding
    is cheaper than flipping. Trained by gradient ascent on the DSR with the
    recurrence approximated (the standard online RRL simplification -- full BPTT
    through the position path buys little on noisy data).

    Continuous output is discretised only when the env asks for discrete actions.

    Note on dimensionality: RRL consumes the *market* features only. The env's
    observation appends agent-state fields (position, unrealised P&L, ...), which
    we deliberately slice off -- in Moody & Saffell's formulation the position
    enters through the recurrent term ``u * F_{t-1}``, not through the feature
    vector. Feeding it in twice would double-count it.
    """

    def __init__(
        self,
        n_features: int,
        *,
        learning_rate: float = 0.01,
        eta: float = 0.01,
        seed: int = 0,
        discrete: bool = True,
        threshold: float = 0.33,
        l2: float = 1e-3,
        max_weight_norm: float = 1.0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        # Small init: start near flat and let the data move us.
        self.w = rng.normal(0, 0.01, n_features)
        self.u = 0.0
        self.b = 0.0
        self.lr = learning_rate
        self.eta = eta
        self.discrete = discrete
        self.threshold = threshold
        # Saturation guards. Once |w.x| routinely exceeds ~2, tanh pins at +/-1,
        # its gradient vanishes, and every conviction ties at 1.0 -- so ranking
        # degenerates to the sort's tiebreak. L2 shrinks; the norm cap is a
        # hard stop against the slow monotonic drift L2 alone permits.
        self.l2 = l2
        self.max_weight_norm = max_weight_norm
        self.reset()

    def reset(self) -> None:
        self.f_prev = 0.0
        self.a = 0.0  # EW first moment of returns
        self.b_moment = 0.0  # EW second moment
        self._steps = 0
        # Running gradient of F_prev w.r.t. weights, for the recurrent term.
        self._dfprev_dw = np.zeros_like(self.w)
        self._dfprev_du = 0.0

    def _market_features(self, obs: np.ndarray) -> np.ndarray:
        """Take only the market features, dropping any appended agent state."""
        x = np.asarray(obs, dtype=np.float64).ravel()
        if len(x) < self.n_features:
            raise ValueError(
                f"observation has {len(x)} values but agent expects at least {self.n_features}"
            )
        return x[: self.n_features]

    def position(self, x: np.ndarray) -> float:
        """Raw continuous position in [-1, 1]."""
        z = float(self.w @ self._market_features(x) + self.u * self.f_prev + self.b)
        return float(np.tanh(z))

    def act(self, obs: np.ndarray) -> int | np.ndarray:
        f = self.position(obs)
        self.f_prev = f
        if not self.discrete:
            return np.array([f], dtype=np.float32)
        if f > self.threshold:
            return 2  # LONG
        if f < -self.threshold:
            return 0  # SHORT
        return 1  # FLAT

    def update(self, x: np.ndarray, ret: float, cost: float = 0.0) -> float:
        """One online DSR gradient-ascent step.

        ``ret`` is the asset's return this bar and ``cost`` the per-unit cost of
        changing position, so the gradient sees the cost of trading directly --
        which is what stops RRL from churning.
        """
        x = self._market_features(x)
        z = float(self.w @ x + self.u * self.f_prev + self.b)
        f = float(np.tanh(z))
        dtanh = 1.0 - f * f

        # Realised net return for this bar, given the position we just took.
        r = self.f_prev * ret - cost * abs(f - self.f_prev)

        a_prev, b_prev = self.a, self.b_moment
        variance = b_prev - a_prev * a_prev

        if self._steps == 0 or variance <= 1e-12:
            d_sharpe_d_r = 0.0
        else:
            # dD_t/dR_t for the differential Sharpe ratio.
            d_sharpe_d_r = (b_prev - a_prev * r) / (variance**1.5)

        # dR/dF_t is the cost term; dR/dF_{t-1} carries the return.
        dr_df = -cost * np.sign(f - self.f_prev)
        dr_dfprev = ret + cost * np.sign(f - self.f_prev)

        # Total derivative through the recurrence.
        df_dw = dtanh * (x + self.u * self._dfprev_dw)
        df_du = dtanh * (self.f_prev + self.u * self._dfprev_du)
        df_db = dtanh

        grad_w = d_sharpe_d_r * (dr_df * df_dw + dr_dfprev * self._dfprev_dw)
        grad_u = d_sharpe_d_r * (dr_df * df_du + dr_dfprev * self._dfprev_du)
        grad_b = d_sharpe_d_r * dr_df * df_db

        # Ascent: we are maximising the Sharpe ratio, not minimising a loss.
        # L2 shrinkage rides along inside the step so the penalty is felt even
        # when the DSR gradient has vanished into a saturated tanh.
        self.w += self.lr * (np.clip(grad_w, -1.0, 1.0) - self.l2 * self.w)
        self.u += self.lr * (float(np.clip(grad_u, -1.0, 1.0)) - self.l2 * self.u)
        self.b += self.lr * float(np.clip(grad_b, -1.0, 1.0))
        if self.max_weight_norm > 0:
            norm = float(np.linalg.norm(self.w))
            if norm > self.max_weight_norm:
                self.w *= self.max_weight_norm / norm

        self._dfprev_dw, self._dfprev_du = df_dw, df_du

        self.a = a_prev + self.eta * (r - a_prev)
        self.b_moment = b_prev + self.eta * (r * r - b_prev)
        self._steps += 1
        self.f_prev = f
        return r

    @property
    def sharpe(self) -> float:
        var = self.b_moment - self.a * self.a
        return float(self.a / np.sqrt(var)) if var > 1e-12 else 0.0
