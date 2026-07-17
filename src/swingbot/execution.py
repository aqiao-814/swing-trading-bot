"""Execution and friction modelling.

Turns an *intent* ("I want to be 60% long AAPL") into a *fill* at a price that
is deliberately worse than the reference price, because that is what happens in
reality. Every cost the research blueprint calls for is modelled explicitly:
spread, slippage, square-root market impact, commission, regulatory fees, and
overnight short borrow.

The single most important rule enforced here is **execution delay**: a decision
made on bar *t*'s close cannot fill at bar *t*'s close. It fills at bar *t+1*'s
open. Violating that is the most common source of look-ahead bias in retail
backtests, and it is why this module never sees a price it wasn't handed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from swingbot.config import CostConfig
from swingbot.portfolio import Fill

_BPS = 1e-4


@dataclass(frozen=True)
class MarketContext:
    """Everything the execution model may legally know at fill time.

    Deliberately narrow: if a field isn't here, the cost model cannot depend on
    it, which makes look-ahead bias a type error rather than a subtle bug.
    """

    ts: date
    symbol: str
    reference_price: float  # the price we execute against (e.g. next open)
    volatility: float = 0.02  # daily return stdev, for impact scaling
    volume: float | None = None  # shares traded that bar, for participation

    def __post_init__(self) -> None:
        if self.reference_price <= 0:
            raise ValueError(f"reference_price must be positive, got {self.reference_price}")


class ExecutionModel:
    """Computes realistic fill prices and costs from a target trade."""

    def __init__(self, costs: CostConfig) -> None:
        self.costs = costs

    # ---- price impact ----------------------------------------------------

    def _impact_fraction(self, quantity: float, ctx: MarketContext) -> float:
        """Square-root market impact (Almgren-Chriss): impact ~ coef * sigma * sqrt(Q/V).

        Returns a positive fraction of price. Falls back to zero when volume is
        unknown -- an optimistic assumption, so callers should supply volume for
        anything but the most liquid names.
        """
        if not self.costs.use_sqrt_impact:
            return 0.0
        if not ctx.volume or ctx.volume <= 0:
            return 0.0
        participation = abs(quantity) / ctx.volume
        return self.costs.impact_coef * ctx.volatility * (participation**0.5)

    def fill_price(self, quantity: float, ctx: MarketContext) -> float:
        """Reference price degraded by spread, slippage, and impact.

        Costs are always *adverse*: buys fill above the reference, sells below.
        """
        if quantity == 0:
            return ctx.reference_price

        adverse = (
            self.costs.half_spread_bps * _BPS
            + self.costs.slippage_bps * _BPS
            + self._impact_fraction(quantity, ctx)
        )
        direction = 1.0 if quantity > 0 else -1.0
        price = ctx.reference_price * (1.0 + direction * adverse)
        # A cost model must never produce a non-positive price.
        return max(price, 1e-8)

    # ---- explicit costs --------------------------------------------------

    def commission(self, quantity: float, price: float) -> float:
        if quantity == 0:
            return 0.0
        shares = abs(quantity)
        comm = shares * self.costs.commission_per_share
        comm += shares * price * self.costs.commission_bps * _BPS
        return max(comm, self.costs.min_commission)

    def fees(self, quantity: float, price: float) -> float:
        """Regulatory fees, charged on sells only.

        SEC Section 31 is a bps rate on notional; FINRA TAF is per share with
        a per-trade cap. Sells are never free: a $0.00 here means a unit bug.
        """
        if quantity >= 0:
            return 0.0
        shares = abs(quantity)
        sec = shares * price * self.costs.sec_fee_bps * _BPS
        taf = min(shares * self.costs.taf_per_share, self.costs.taf_cap_per_trade)
        return sec + taf

    def borrow_cost(self, short_value: float, days: float = 1.0) -> float:
        """Overnight borrow on short notional, accrued on a 360-day basis."""
        if short_value >= 0:
            return 0.0
        notional = abs(short_value)
        return notional * (self.costs.short_borrow_annual_bps * _BPS) * (days / 360.0)

    # ---- assembly --------------------------------------------------------

    def build_fill(self, quantity: float, ctx: MarketContext) -> Fill | None:
        """Produce a fully-costed Fill, or None when there is nothing to trade."""
        if abs(quantity) < 1e-9:
            return None
        price = self.fill_price(quantity, ctx)
        return Fill(
            ts=ctx.ts,
            symbol=ctx.symbol,
            quantity=quantity,
            price=price,
            commission=self.commission(quantity, price),
            fees=self.fees(quantity, price),
            reference_price=ctx.reference_price,
        )

    def round_trip_cost_bps(self, ctx: MarketContext, quantity: float = 0.0) -> float:
        """Estimated cost of entering and exiting, in bps of notional.

        Useful as a sanity check: if the strategy's expected edge per trade is
        smaller than this, it cannot be profitable no matter how good the model.
        """
        one_way = self.costs.half_spread_bps + self.costs.slippage_bps
        if quantity:
            one_way += self._impact_fraction(quantity, ctx) / _BPS
        one_way += self.costs.commission_bps
        return 2.0 * one_way
