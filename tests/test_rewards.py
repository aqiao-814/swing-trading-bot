"""Reward semantics -- especially that DSR actually prices risk."""

from __future__ import annotations

import numpy as np
import pytest

from swingbot.config import RewardKind
from swingbot.rewards import DifferentialSharpeRatio, DrawdownPenalized, NetReturn, build_reward


class TestDifferentialSharpe:
    def test_first_step_emits_no_signal(self):
        """With no variance history, any reward would be a divide-by-zero artifact."""
        dsr = DifferentialSharpeRatio(eta=0.01)
        assert dsr.update(0.01) == 0.0

    def test_moments_track_returns(self):
        dsr = DifferentialSharpeRatio(eta=0.1)
        for _ in range(500):
            dsr.update(0.01)
        assert dsr.a == pytest.approx(0.01, abs=1e-6)
        assert dsr.b == pytest.approx(0.0001, abs=1e-8)

    def test_positive_surprise_rewarded_negative_punished(self):
        rng = np.random.default_rng(0)
        dsr = DifferentialSharpeRatio(eta=0.05)
        for _ in range(200):
            dsr.update(float(rng.normal(0.001, 0.01)))
        good = DifferentialSharpeRatio(eta=0.05)
        good.a, good.b, good._steps = dsr.a, dsr.b, dsr._steps
        bad = DifferentialSharpeRatio(eta=0.05)
        bad.a, bad.b, bad._steps = dsr.a, dsr.b, dsr._steps

        assert good.update(0.02) > 0
        assert bad.update(-0.02) < 0

    def test_prefers_steady_returns_over_volatile_at_equal_mean(self):
        """The whole point of DSR: same mean, lower variance, higher score."""
        steady_returns = [0.001] * 300
        rng = np.random.default_rng(42)
        volatile_returns = [0.001 + float(rng.normal(0, 0.03)) for _ in range(300)]
        # Force identical realised mean so only variance differs.
        volatile_returns = list(np.array(volatile_returns) - np.mean(volatile_returns) + 0.001)

        steady = DifferentialSharpeRatio(eta=0.01)
        volatile = DifferentialSharpeRatio(eta=0.01)
        for r in steady_returns:
            steady.update(r)
        for r in volatile_returns:
            volatile.update(r)

        assert steady.sharpe > volatile.sharpe

    def test_reset_clears_state(self):
        dsr = DifferentialSharpeRatio()
        for _ in range(10):
            dsr.update(0.01)
        dsr.reset()
        assert (dsr.a, dsr.b, dsr._steps) == (0.0, 0.0, 0)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            DifferentialSharpeRatio(eta=0.0)
        with pytest.raises(ValueError):
            DifferentialSharpeRatio(eta=1.5)
        with pytest.raises(ValueError):
            DifferentialSharpeRatio().update(float("nan"))

    def test_survives_a_total_loss_bar(self):
        dsr = DifferentialSharpeRatio(eta=0.01)
        for _ in range(50):
            dsr.update(0.001)
        assert np.isfinite(dsr.update(-1.0))


class TestNetReturn:
    def test_log_scaling(self):
        assert NetReturn(log=True).update(0.10) == pytest.approx(np.log(1.10))

    def test_simple_passthrough(self):
        assert NetReturn(log=False).update(0.10) == pytest.approx(0.10)

    def test_total_loss_does_not_produce_negative_infinity(self):
        assert np.isfinite(NetReturn(log=True).update(-1.0))


class TestDrawdownPenalized:
    def test_no_penalty_while_making_new_highs(self):
        r = DrawdownPenalized(penalty=1.0)
        first = r.update(0.01)
        assert first == pytest.approx(np.log(1.01))

    def test_penalizes_returns_taken_inside_a_drawdown(self):
        deep = DrawdownPenalized(penalty=1.0)
        deep.update(0.10)  # peak
        deep.update(-0.15)  # drop into drawdown
        recovering = deep.update(0.01)
        # Same +1% bar scores worse underwater than at a high-water mark.
        assert recovering < DrawdownPenalized(penalty=1.0).update(0.01)


def test_factory_builds_each_kind():
    assert isinstance(build_reward(RewardKind.DSR, dsr_eta=0.02), DifferentialSharpeRatio)
    assert isinstance(build_reward(RewardKind.NET_LOG_RETURN), NetReturn)
    assert isinstance(build_reward(RewardKind.NET_SIMPLE_RETURN), NetReturn)
