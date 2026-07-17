"""Accounting invariants. If these break, every downstream metric is fiction."""

from __future__ import annotations

from datetime import date

import pytest

from swingbot.portfolio import Fill, Portfolio, Position

TS = date(2024, 1, 2)


def _fill(qty: float, price: float, **kw) -> Fill:
    return Fill(ts=TS, symbol="AAPL", quantity=qty, price=price, **kw)


class TestPosition:
    def test_open_long_sets_basis(self):
        p = Position("AAPL")
        assert p.apply(100, 50.0) == 0.0
        assert p.quantity == 100
        assert p.avg_price == 50.0

    def test_averaging_up_reweights_basis(self):
        p = Position("AAPL")
        p.apply(100, 50.0)
        p.apply(100, 60.0)
        assert p.quantity == 200
        assert p.avg_price == pytest.approx(55.0)

    def test_partial_close_realizes_only_closed_shares(self):
        p = Position("AAPL")
        p.apply(100, 50.0)
        realized = p.apply(-40, 60.0)
        assert realized == pytest.approx(40 * 10.0)
        assert p.quantity == 60
        # Basis of the surviving shares is untouched by a partial exit.
        assert p.avg_price == pytest.approx(50.0)

    def test_short_profits_when_price_falls(self):
        p = Position("AAPL")
        p.apply(-100, 50.0)
        assert p.is_short
        assert p.unrealized_pnl(40.0) == pytest.approx(1000.0)
        realized = p.apply(100, 40.0)
        assert realized == pytest.approx(1000.0)
        assert p.is_flat

    def test_flip_long_to_short_realizes_old_and_rebases(self):
        p = Position("AAPL")
        p.apply(100, 50.0)
        # Sell 150: closes 100 long (+$1000), opens 50 short at 60.
        realized = p.apply(-150, 60.0)
        assert realized == pytest.approx(1000.0)
        assert p.quantity == -50
        assert p.avg_price == pytest.approx(60.0)

    def test_full_close_returns_to_exact_flat(self):
        p = Position("AAPL")
        p.apply(100, 50.0)
        p.apply(-100, 55.0)
        assert p.quantity == 0.0
        assert p.avg_price == 0.0
        assert p.is_flat


class TestPortfolio:
    def test_trading_alone_does_not_change_equity(self):
        """A fill at the mark with no costs must be equity-neutral."""
        pf = Portfolio(100_000)
        pf.execute(_fill(100, 50.0))
        assert pf.cash == pytest.approx(95_000)
        assert pf.equity({"AAPL": 50.0}) == pytest.approx(100_000)

    def test_costs_are_the_only_leak(self):
        pf = Portfolio(100_000)
        pf.execute(_fill(100, 50.0, commission=1.0, fees=0.5))
        assert pf.equity({"AAPL": 50.0}) == pytest.approx(100_000 - 1.5)
        assert pf.cumulative_costs == pytest.approx(1.5)

    def test_short_credits_cash_and_keeps_equity_flat(self):
        pf = Portfolio(100_000)
        pf.execute(_fill(-100, 50.0))
        assert pf.cash == pytest.approx(105_000)
        assert pf.equity({"AAPL": 50.0}) == pytest.approx(100_000)
        # Short gains as price falls.
        assert pf.equity({"AAPL": 45.0}) == pytest.approx(100_500)

    def test_equity_invariant_holds_across_a_flip(self):
        pf = Portfolio(100_000)
        pf.execute(_fill(100, 50.0))
        pf.execute(_fill(-150, 60.0))
        prices = {"AAPL": 60.0}
        assert pf.equity(prices) == pytest.approx(pf.cash + pf.market_value(prices))
        # +$1000 realized on the closed long, short opened at the mark.
        assert pf.equity(prices) == pytest.approx(101_000)
        assert pf.realized_pnl == pytest.approx(1000.0)

    def test_round_trip_pnl_lands_in_equity(self):
        pf = Portfolio(100_000)
        pf.execute(_fill(100, 50.0))
        pf.execute(_fill(-100, 55.0))
        assert pf.realized_pnl == pytest.approx(500.0)
        assert pf.equity({"AAPL": 55.0}) == pytest.approx(100_500)
        assert pf.cash == pytest.approx(100_500)

    def test_exposure_metrics_distinguish_gross_from_net(self):
        pf = Portfolio(100_000)
        pf.execute(Fill(ts=TS, symbol="AAPL", quantity=100, price=50.0))
        pf.execute(Fill(ts=TS, symbol="MSFT", quantity=-50, price=100.0))
        prices = {"AAPL": 50.0, "MSFT": 100.0}
        assert pf.gross_exposure(prices) == pytest.approx(10_000)
        assert pf.net_exposure(prices) == pytest.approx(0.0)

    def test_fractional_shares_rejected_by_default(self):
        pf = Portfolio(100_000)
        pf.execute(_fill(10.7, 50.0))
        assert pf.quantity("AAPL") == 11.0

    def test_borrow_charge_reduces_equity(self):
        pf = Portfolio(100_000)
        pf.charge(12.34, symbol="AAPL")
        assert pf.equity({}) == pytest.approx(100_000 - 12.34)
        assert pf.cumulative_costs == pytest.approx(12.34)

    def test_reset_restores_initial_state(self):
        pf = Portfolio(100_000)
        pf.execute(_fill(100, 50.0, commission=1.0))
        pf.snapshot(TS, {"AAPL": 50.0})
        pf.reset()
        assert pf.cash == 100_000
        assert pf.positions == {}
        assert pf.fills == []
        assert pf.history == []

    def test_rejects_nonsense_construction(self):
        with pytest.raises(ValueError):
            Portfolio(0)
        with pytest.raises(ValueError):
            _fill(10, -5.0)
        with pytest.raises(ValueError):
            _fill(10, 50.0, commission=-1.0)


def test_slippage_attribution_signs_both_directions():
    buy = _fill(100, 50.10, reference_price=50.0)
    assert buy.slippage_cost == pytest.approx(10.0)  # paid up
    sell = _fill(-100, 49.90, reference_price=50.0)
    assert sell.slippage_cost == pytest.approx(10.0)  # sold down
