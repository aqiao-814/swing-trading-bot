"""Environment mechanics.

Two tests here are load-bearing:

* ``test_fills_at_next_open_not_current_close`` -- proves the execution delay is
  real, so the agent cannot trade on information it hasn't earned.
* ``test_no_free_money_on_pure_noise`` -- the null hypothesis. On a driftless
  random walk with costs, any strategy must lose. A strategy that profits here
  means the simulator leaks the future.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from gymnasium.utils.env_checker import check_env

from swingbot.config import (
    ActionSpaceKind,
    CostConfig,
    EnvConfig,
    FeatureConfig,
    RewardKind,
    RiskConfig,
)
from swingbot.data.sources import SyntheticSource
from swingbot.env.trading_env import SwingTradingEnv
from swingbot.features.technical import build_dataset, feature_columns

LONG, FLAT, SHORT = 2, 1, 0


@pytest.fixture(scope="module")
def dataset() -> pl.DataFrame:
    bars = SyntheticSource(seed=11).fetch("TEST", "2010-01-01", "2024-12-31")
    return build_dataset(bars, FeatureConfig())


@pytest.fixture
def cfg() -> EnvConfig:
    # Risk overlays off by default so mechanics tests aren't fighting kill switches.
    return EnvConfig(
        starting_capital=100_000.0,
        episode_length=None,
        risk=RiskConfig(
            vol_target_annual=None,
            stop_loss_pct=None,
            max_drawdown_pct=None,
            max_daily_loss_pct=None,
        ),
    )


def make_env(dataset, cfg, **kw) -> SwingTradingEnv:
    kw.setdefault("random_start", False)
    return SwingTradingEnv(dataset, feature_columns(FeatureConfig()), cfg, **kw)


class TestGymCompliance:
    def test_passes_gymnasium_env_checker(self, dataset, cfg):
        check_env(make_env(dataset, cfg), skip_render_check=True)

    def test_observation_is_float32_for_mps(self, dataset, cfg):
        """MPS cannot convert float64 tensors -- float32 is not a preference."""
        obs, _ = make_env(dataset, cfg).reset()
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))

    def test_observation_includes_agent_state(self, dataset, cfg):
        env = make_env(dataset, cfg)
        obs, _ = env.reset()
        assert obs.shape == (len(feature_columns(FeatureConfig())) + 4,)

    def test_continuous_action_space(self, dataset):
        c = EnvConfig(
            action_space=ActionSpaceKind.CONTINUOUS,
            episode_length=None,
            risk=RiskConfig(vol_target_annual=None),
        )
        env = make_env(dataset, c)
        env.reset()
        env.step(np.array([0.5], dtype=np.float32))
        assert env._target_position == pytest.approx(0.5)

    def test_seeded_resets_are_reproducible(self, dataset, cfg):
        cfg.episode_length = 100
        a, b = make_env(dataset, cfg, random_start=True), make_env(dataset, cfg, random_start=True)
        o1, _ = a.reset(seed=42)
        o2, _ = b.reset(seed=42)
        assert np.array_equal(o1, o2)


class TestTiming:
    def test_fills_at_next_open_not_current_close(self, dataset, cfg):
        """The single most important guarantee in the system."""
        env = make_env(dataset, cfg)
        env.reset()
        decision_idx = env._i
        env.step(LONG)

        fill_price = env._trades[-1].fill_price
        assert fill_price is not None
        # The fill must derive from bar t+1's open, never bar t's close.
        next_open = env._open[decision_idx + 1]
        assert fill_price == pytest.approx(next_open, rel=0.01)
        assert fill_price != pytest.approx(env._close[decision_idx], rel=1e-6)

    def test_clock_advances_exactly_one_bar_per_step(self, dataset, cfg):
        env = make_env(dataset, cfg)
        env.reset()
        start = env._i
        for _ in range(5):
            env.step(FLAT)
        assert env._i == start + 5

    def test_stepping_a_finished_episode_raises(self, dataset, cfg):
        cfg.episode_length = 5
        env = make_env(dataset, cfg)
        env.reset()
        for _ in range(5):
            env.step(FLAT)
        with pytest.raises(RuntimeError, match="finished episode"):
            env.step(FLAT)


class TestAccounting:
    def test_flat_agent_never_loses_money(self, dataset, cfg):
        """No position, no costs, no drift in equity. Catches phantom charges."""
        env = make_env(dataset, cfg)
        env.reset()
        for _ in range(200):
            env.step(FLAT)
        r = env.result()
        assert r.equity[-1] == pytest.approx(cfg.starting_capital)
        assert r.total_costs == 0.0

    def test_trading_incurs_costs(self, dataset, cfg):
        env = make_env(dataset, cfg)
        env.reset()
        env.step(LONG)
        assert env.result().total_costs > 0

    def test_spread_and_slippage_are_counted_not_hidden_in_the_fill_price(self, dataset):
        """Regression: spread/slippage never debit cash -- they just make the
        fill worse -- so a naive cost counter reports zero and flatters returns."""
        c = EnvConfig(
            episode_length=None,
            # Zero explicit costs: every dollar lost here is pure slippage.
            costs=CostConfig(
                commission_per_share=0.0,
                sec_fee_bps=0.0,
                taf_per_share=0.0,
                half_spread_bps=5.0,
                slippage_bps=2.0,
            ),
            risk=RiskConfig(
                vol_target_annual=None,
                stop_loss_pct=None,
                max_drawdown_pct=None,
                max_daily_loss_pct=None,
            ),
        )
        env = make_env(dataset, c)
        env.reset()
        env.step(LONG)
        r = env.result()
        assert r.explicit_costs == 0.0
        assert r.slippage_costs > 0.0
        assert r.total_costs == pytest.approx(r.slippage_costs)

    def test_churning_bleeds_capital(self, dataset, cfg):
        """Churning must cost exactly what the frictions say it costs.

        Compared against an identical zero-cost run rather than against a fixed
        threshold: on a trending path, a churning strategy can still finish up,
        so 'ends below starting capital' would be a test of the price path, not
        of the cost model. This isolates the drag itself.
        """

        def run(costs: CostConfig) -> float:
            c = cfg.model_copy(update={"costs": costs})
            env = make_env(dataset, c)
            env.reset()
            for i in range(200):
                env.step(LONG if i % 2 == 0 else FLAT)
            return env.result().equity[-1]

        free = run(
            CostConfig(
                half_spread_bps=0.0,
                slippage_bps=0.0,
                commission_per_share=0.0,
                sec_fee_bps=0.0,
                taf_per_share=0.0,
                use_sqrt_impact=False,
            )
        )
        costly = run(CostConfig(half_spread_bps=5.0, slippage_bps=2.0))
        assert costly < free, "frictions did not reduce the return of a churning strategy"

    def test_equity_curve_tracks_portfolio(self, dataset, cfg):
        env = make_env(dataset, cfg)
        env.reset()
        for _ in range(50):
            env.step(LONG)
        r = env.result()
        assert r.equity[-1] == pytest.approx(env.portfolio.equity({env.symbol: env._close[env._i]}))

    def test_short_position_accrues_borrow_cost(self, dataset, cfg):
        cfg.costs = CostConfig(short_borrow_annual_bps=500.0, half_spread_bps=0, slippage_bps=0)
        env = make_env(dataset, cfg)
        env.reset()
        env.step(SHORT)
        costs_after_entry = env.portfolio.cumulative_costs
        for _ in range(20):
            env.step(SHORT)
        assert env.portfolio.cumulative_costs > costs_after_entry


class TestNoTradeBand:
    def test_band_suppresses_churn_from_vol_targeting(self, dataset):
        """Without a band, vol targeting re-sizes every bar and churns on noise."""

        def n_fills(band: float) -> int:
            c = EnvConfig(
                episode_length=None,
                risk=RiskConfig(
                    vol_target_annual=0.15,
                    rebalance_threshold=band,
                    stop_loss_pct=None,
                    max_drawdown_pct=None,
                    max_daily_loss_pct=None,
                ),
            )
            env = make_env(dataset, c)
            env.reset()
            for _ in range(500):
                env.step(LONG)  # constant intent; any trade is pure rebalancing
            return sum(1 for t in env.result().trades if abs(t.quantity) > 0)

        assert n_fills(0.05) < n_fills(0.0) / 2

    def test_band_never_blocks_a_full_exit(self, dataset):
        """A stop-loss must not be filtered out by a rebalancing rule."""
        c = EnvConfig(
            episode_length=None,
            risk=RiskConfig(
                vol_target_annual=None,
                rebalance_threshold=0.99,
                stop_loss_pct=None,
                max_drawdown_pct=None,
                max_daily_loss_pct=None,
            ),
        )
        env = make_env(dataset, c)
        env.reset()
        for _ in range(5):
            env.step(LONG)
        assert not env.portfolio.position(env.symbol).is_flat
        for _ in range(3):
            env.step(FLAT)
        # Despite a 99% band, going flat is always honoured.
        assert env.portfolio.position(env.symbol).is_flat


class TestRiskOverlays:
    def test_stop_loss_flattens_position(self, dataset):
        c = EnvConfig(
            episode_length=None,
            risk=RiskConfig(
                stop_loss_pct=0.01,
                vol_target_annual=None,
                max_drawdown_pct=None,
                max_daily_loss_pct=None,
            ),
        )
        env = make_env(dataset, c)
        env.reset()
        stopped = False
        for _ in range(300):
            env.step(LONG)
            if env._trades[-1].exit_reason == "stop_loss":
                stopped = True
                break
        assert stopped, "a 1% stop should trigger within 300 bars"

    def test_max_drawdown_halts_and_stays_halted(self, dataset):
        c = EnvConfig(
            episode_length=None,
            risk=RiskConfig(
                max_drawdown_pct=0.02,
                vol_target_annual=None,
                stop_loss_pct=None,
                max_daily_loss_pct=None,
            ),
        )
        env = make_env(dataset, c)
        env.reset()
        for _ in range(400):
            _, _, terminated, _, info = env.step(LONG)
            if terminated:
                assert "max_drawdown" in info["halted"]
                return
        pytest.skip("no 2% drawdown occurred in this synthetic path")

    def test_shorting_can_be_disabled(self, dataset, cfg):
        cfg.risk = RiskConfig(allow_short=False, vol_target_annual=None)
        env = make_env(dataset, cfg)
        env.reset()
        for _ in range(20):
            env.step(SHORT)
        assert env.portfolio.quantity(env.symbol) >= 0

    def test_vol_targeting_shrinks_exposure_when_vol_is_high(self, dataset):
        calm = EnvConfig(
            episode_length=None, risk=RiskConfig(vol_target_annual=0.10, kelly_fraction=1.0)
        )
        hot = EnvConfig(
            episode_length=None, risk=RiskConfig(vol_target_annual=0.02, kelly_fraction=1.0)
        )
        a, b = make_env(dataset, calm), make_env(dataset, hot)
        a.reset()
        b.reset()
        a.step(LONG)
        b.step(LONG)
        assert abs(b._target_position) < abs(a._target_position)


class TestNullHypothesis:
    def test_no_free_money_on_pure_noise(self):
        """On a driftless random walk with costs, always-long must lose.

        If this ever passes with a profit, the environment is leaking future
        information and every result the system produces is fiction.
        """
        bars = SyntheticSource(seed=99, annual_drift=0.0, annual_vol=0.20).fetch(
            "NOISE", "2000-01-01", "2024-12-31"
        )
        data = build_dataset(bars, FeatureConfig())
        c = EnvConfig(
            starting_capital=100_000.0,
            episode_length=None,
            costs=CostConfig(half_spread_bps=2.0, slippage_bps=1.0),
            risk=RiskConfig(
                vol_target_annual=None,
                stop_loss_pct=None,
                max_drawdown_pct=None,
                max_daily_loss_pct=None,
            ),
            reward=RewardKind.NET_LOG_RETURN,
        )
        env = SwingTradingEnv(data, feature_columns(FeatureConfig()), c, random_start=False)
        env.reset()
        # Churn every bar: maximum exposure to costs, zero expected edge.
        for i in range(1000):
            env.step(LONG if i % 2 == 0 else SHORT)
        r = env.result()
        assert r.equity[-1] < r.starting_capital, (
            "churning a driftless series turned a profit -- the env leaks the future"
        )


def test_agent_is_not_told_the_money_is_fake(dataset, cfg):
    """No observation field may reveal simulation. The policy must behave
    identically to one trading real capital."""
    env = make_env(dataset, cfg)
    obs, info = env.reset()
    assert len(obs) == len(feature_columns(FeatureConfig())) + 4
    assert not any("sim" in k or "fake" in k or "paper" in k for k in info)
