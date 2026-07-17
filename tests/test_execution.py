"""Frictions must always hurt, never help."""

from __future__ import annotations

from datetime import date

import pytest

from swingbot.config import CostConfig
from swingbot.execution import ExecutionModel, MarketContext

TS = date(2024, 1, 2)


def ctx(price: float = 100.0, **kw) -> MarketContext:
    return MarketContext(ts=TS, symbol="AAPL", reference_price=price, **kw)


class TestFillPrice:
    def test_buys_fill_above_and_sells_below_reference(self):
        em = ExecutionModel(CostConfig(half_spread_bps=1.0, slippage_bps=0.5))
        assert em.fill_price(100, ctx()) > 100.0
        assert em.fill_price(-100, ctx()) < 100.0

    def test_spread_and_slippage_add_up_exactly(self):
        em = ExecutionModel(
            CostConfig(half_spread_bps=2.0, slippage_bps=1.0, use_sqrt_impact=False)
        )
        # 3bps adverse on a 100.00 reference.
        assert em.fill_price(100, ctx()) == pytest.approx(100.0 * 1.0003)
        assert em.fill_price(-100, ctx()) == pytest.approx(100.0 * 0.9997)

    def test_zero_quantity_is_costless(self):
        em = ExecutionModel(CostConfig())
        assert em.fill_price(0, ctx()) == 100.0

    def test_impact_grows_with_participation(self):
        em = ExecutionModel(CostConfig(use_sqrt_impact=True, impact_coef=0.1))
        small = em.fill_price(100, ctx(volume=1_000_000))
        large = em.fill_price(100_000, ctx(volume=1_000_000))
        assert large > small > 100.0

    def test_impact_is_sublinear_in_size(self):
        """Square-root law: 4x the size must cost less than 4x the impact."""
        em = ExecutionModel(CostConfig(half_spread_bps=0, slippage_bps=0, impact_coef=0.1))
        base = em.fill_price(1_000, ctx(volume=1_000_000)) - 100.0
        quad = em.fill_price(4_000, ctx(volume=1_000_000)) - 100.0
        assert quad == pytest.approx(2.0 * base, rel=1e-6)

    def test_missing_volume_disables_impact(self):
        em = ExecutionModel(CostConfig(half_spread_bps=0, slippage_bps=0, impact_coef=0.1))
        assert em.fill_price(100_000, ctx(volume=None)) == pytest.approx(100.0)

    def test_rejects_nonpositive_reference_price(self):
        with pytest.raises(ValueError):
            ctx(price=0.0)


class TestExplicitCosts:
    def test_per_share_commission(self):
        em = ExecutionModel(CostConfig(commission_per_share=0.005))
        assert em.commission(100, 100.0) == pytest.approx(0.5)

    def test_bps_commission_scales_with_notional(self):
        em = ExecutionModel(CostConfig(commission_bps=1.0))
        assert em.commission(100, 100.0) == pytest.approx(10_000 * 1e-4)

    def test_minimum_commission_floor(self):
        em = ExecutionModel(CostConfig(commission_per_share=0.005, min_commission=1.0))
        assert em.commission(10, 100.0) == pytest.approx(1.0)

    def test_sec_fees_charged_on_sells_only(self):
        em = ExecutionModel(CostConfig())
        assert em.fees(100, 100.0) == 0.0
        assert em.fees(-100, 100.0) > 0.0

    def test_sell_fees_have_realistic_magnitude(self):
        """Regression: sec_fee_bps was once a *fraction* fed through a second
        bps conversion, so every sell fee in the ledger rendered as $0.00.
        A `> 0` assertion passes right through that bug; magnitude does not.
        """
        em = ExecutionModel(CostConfig())
        fee = em.fees(-100, 100.0)  # 100 shares at $100 = $10,000 sell
        sec = 10_000 * 0.278 * 1e-4  # Section 31: ~$27.80 per $1M notional
        taf = 100 * 0.000166  # FINRA TAF per share
        assert fee == pytest.approx(sec + taf)
        assert fee > 0.10  # visible in a two-decimal ledger, never $0.00

    def test_taf_is_capped_per_trade(self):
        em = ExecutionModel(CostConfig(sec_fee_bps=0.0))
        assert em.fees(-1_000_000, 1.0) == pytest.approx(8.30)

    def test_borrow_accrues_on_shorts_only(self):
        em = ExecutionModel(CostConfig(short_borrow_annual_bps=30.0))
        assert em.borrow_cost(10_000.0) == 0.0  # long
        cost = em.borrow_cost(-10_000.0, days=1.0)
        assert cost == pytest.approx(10_000 * 0.003 / 360.0)

    def test_borrow_scales_with_days_held(self):
        em = ExecutionModel(CostConfig(short_borrow_annual_bps=30.0))
        one = em.borrow_cost(-10_000.0, days=1.0)
        three = em.borrow_cost(-10_000.0, days=3.0)
        assert three == pytest.approx(3.0 * one)


class TestBuildFill:
    def test_produces_fully_costed_fill(self):
        em = ExecutionModel(CostConfig(commission_per_share=0.005, half_spread_bps=1.0))
        fill = em.build_fill(100, ctx())
        assert fill is not None
        assert fill.quantity == 100
        assert fill.price > 100.0
        assert fill.commission > 0
        assert fill.reference_price == 100.0
        assert fill.slippage_cost > 0  # buying cost us money vs reference

    def test_no_fill_for_no_trade(self):
        assert ExecutionModel(CostConfig()).build_fill(0, ctx()) is None

    def test_round_trip_cost_is_symmetric_two_way(self):
        em = ExecutionModel(CostConfig(half_spread_bps=1.0, slippage_bps=0.5))
        assert em.round_trip_cost_bps(ctx()) == pytest.approx(3.0)
