"""All-in execution costs, as a NautilusTrader ``FeeModel``.

**The design decision worth reading.** v1 split its costs in two: spread,
slippage and market impact were baked into the *fill price*, while commissions
and regulatory fees were charged as explicit cash. That split is defensible --
it is what really happens -- but it made the bot's own cost reporting wrong.
Money lost to a spread never debits cash, it just makes the fill worse, so a
report that sums the explicit costs reports about $0 on a trade that cost real
money. v1's final record shows the symptom plainly: $227 of "costs" against
$3,768 of losses on 13.8x turnover.

v2 charges **every** friction as an explicit commission and fills at the bar's
own price. The P&L is identical -- a dollar of spread and a dollar of
commission cost a dollar either way -- but now every dollar of friction is
visible, attributable to a fill, and summable. If v2 loses money to costs, the
cost column will say so.

Modelled, per fill:

* half-spread and slippage, in bps of notional (adverse by construction: a cost
  is a cost whether you are buying or selling)
* square-root market impact, ``coef * sigma * sqrt(shares / bar volume)``
  (Almgren-Chriss), which is what stops a 670-name book from pretending it can
  move size in a thin name for free
* SEC Section 31 fee and FINRA TAF, on sells only, with the TAF's per-trade cap

Short borrow is **not** here: it accrues per day on an open short rather than
per fill, so it is charged by the runner against the persisted cash. See
:func:`borrow_accrual`.
"""

from __future__ import annotations

from decimal import Decimal

from nautilus_trader.backtest.models import FeeModel
from nautilus_trader.model.objects import Money

from tradingbot.config import CostConfig

_BPS = 1e-4


class TradingBotFeeModel(FeeModel):
    """v1's ``ExecutionModel`` frictions, charged as Nautilus commission.

    The venue's fee model **cannot be replaced after ``add_venue()``** --
    ``SimulatedExchange.fee_model`` is a read-only Cython attribute and
    ``BacktestEngine`` exposes ``change_fill_model`` but no fee equivalent. Each
    run therefore replays two phases through one engine: a restore phase that
    rebuilds the previous run's book and must be free, then the real bars. The
    phase flag lives *inside* this object and the strategy flips it; see
    :meth:`set_free`.
    """

    def __init__(self, costs: CostConfig, *, free: bool = False) -> None:
        super().__init__()
        self.costs = costs
        self._free = free
        # Bar volume for the fill currently being priced, set by the strategy
        # immediately before it submits. Impact needs the bar's volume and a
        # FeeModel is handed only the order, so it has to be pushed in.
        self._volume: dict[str, float] = {}
        self._sigma: dict[str, float] = {}
        # Running total of the impact/spread/slippage component, so the runner
        # can report friction separately from regulatory fees.
        self.charged_friction: float = 0.0
        self.charged_fees: float = 0.0

    # ---- phase control ---------------------------------------------------

    def set_free(self, free: bool) -> None:
        """Make every subsequent fill cost nothing (the restore phase), or not."""
        self._free = free

    def observe(self, symbol: str, *, volume: float, sigma: float) -> None:
        """Record the market context impact will be computed against."""
        self._volume[symbol] = volume
        self._sigma[symbol] = sigma

    # ---- the FeeModel contract -------------------------------------------

    def get_commission(self, order, fill_qty, fill_px, instrument) -> Money:
        ccy = instrument.quote_currency
        if self._free:
            return Money(0, ccy)

        shares = float(fill_qty.as_double())
        price = float(fill_px.as_double())
        if shares <= 0.0 or price <= 0.0:
            return Money(0, ccy)

        symbol = instrument.id.symbol.value
        notional = shares * price

        friction = notional * (self.costs.half_spread_bps + self.costs.slippage_bps) * _BPS
        friction += notional * self._impact_fraction(symbol, shares)

        commission = shares * self.costs.commission_per_share
        commission += notional * self.costs.commission_bps * _BPS
        commission = max(commission, self.costs.min_commission) if commission else 0.0

        fees = 0.0
        if order.side == 2:  # OrderSide.SELL. Regulatory fees are sell-side only.
            fees = notional * self.costs.sec_fee_bps * _BPS
            fees += min(shares * self.costs.taf_per_share, self.costs.taf_cap_per_trade)

        self.charged_friction += friction
        self.charged_fees += commission + fees
        # Quantise once, at the end. Money rounds to the currency's precision, so
        # summing pre-rounded components would round every term to the cent --
        # which is how a $0.005/share commission becomes $0.01/share.
        return Money(Decimal(repr(friction + commission + fees)), ccy)

    # ---- impact ----------------------------------------------------------

    def _impact_fraction(self, symbol: str, shares: float) -> float:
        """Square-root impact as a fraction of price.

        Zero when the bar's volume is unknown, which is optimistic; the
        alternative -- inventing a volume -- is worse, and the universe screen
        exists so that "unknown" is rare and never a $0.40 stock.
        """
        if not self.costs.use_sqrt_impact:
            return 0.0
        volume = self._volume.get(symbol, 0.0)
        if volume <= 0.0:
            return 0.0
        sigma = self._sigma.get(symbol, 0.02)
        return self.costs.impact_coef * sigma * (shares / volume) ** 0.5


def borrow_accrual(short_notional: float, costs: CostConfig, days: float) -> float:
    """Overnight borrow owed on an open short, on a 360-day basis.

    v1 never paid this: it was long-only and flat by every close, so no short
    ever survived a night. v2 is neither, which makes borrow a real cost for the
    first time -- and one Nautilus will not charge, because a ``FeeModel`` is
    only consulted on a fill. The runner accrues it against persisted cash
    instead, once per calendar day a short is held.
    """
    if short_notional <= 0.0 or days <= 0.0:
        return 0.0
    return short_notional * (costs.short_borrow_annual_bps * _BPS) * (days / 360.0)
