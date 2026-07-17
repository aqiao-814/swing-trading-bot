"""Portfolio accounting.

This module is deliberately dumb about strategy and smart about money. It knows
nothing of agents, features, or rewards; it tracks cash, positions, cost basis,
and realized/unrealized P&L with the same rules a real broker would apply.

Conventions
-----------
* Quantities are signed: positive = long, negative = short.
* ``avg_price`` is always a positive per-share cost basis.
* Short sale proceeds are credited to cash and the position carries negative
  market value, so ``equity = cash + market_value`` holds for both directions.
* Costs are passed in already-computed by the execution model; this class never
  invents a fee.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date

# Quantities below this are treated as flat, guarding against float dust
# accumulating into a phantom position.
_QTY_EPS = 1e-9


@dataclass(frozen=True)
class Fill:
    """A single executed transaction, after slippage and fees are known."""

    ts: date
    symbol: str
    quantity: float  # signed: +buy / -sell
    price: float  # execution price, already includes slippage/spread
    commission: float = 0.0
    fees: float = 0.0
    # Price the decision was made at, for post-hoc slippage attribution.
    reference_price: float | None = None

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError(f"fill price must be positive, got {self.price}")
        if self.commission < 0 or self.fees < 0:
            raise ValueError("commission and fees must be non-negative")

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.price

    @property
    def total_cost(self) -> float:
        return self.commission + self.fees

    @property
    def slippage_cost(self) -> float:
        """Dollar cost of executing away from the reference price."""
        if self.reference_price is None:
            return 0.0
        # Buys filled above reference and sells filled below both cost money.
        return self.quantity * (self.price - self.reference_price)


@dataclass
class Position:
    """A single symbol's holding with average-cost basis."""

    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    # Cumulative frictions attributed to this symbol.
    total_costs: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < _QTY_EPS

    @property
    def is_long(self) -> bool:
        return self.quantity > _QTY_EPS

    @property
    def is_short(self) -> bool:
        return self.quantity < -_QTY_EPS

    def market_value(self, price: float) -> float:
        """Signed mark-to-market value. Negative for shorts."""
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        if self.is_flat:
            return 0.0
        # Long gains when price > basis; short gains when price < basis.
        return self.quantity * (price - self.avg_price)

    def apply(self, quantity: float, price: float) -> float:
        """Apply a signed quantity change at ``price``. Returns realized P&L.

        Handles the three cases a broker must: increasing an existing position
        (re-average the basis), reducing it (realize P&L on the closed shares),
        and flipping through zero (realize the whole old position, then open a
        new one at ``price`` for the remainder).
        """
        if abs(quantity) < _QTY_EPS:
            return 0.0

        realized = 0.0
        opening = self.is_flat or (quantity > 0) == self.is_long

        if opening:
            # Same direction (or from flat): weighted-average the cost basis.
            new_qty = self.quantity + quantity
            total_cost = self.avg_price * abs(self.quantity) + price * abs(quantity)
            self.avg_price = total_cost / abs(new_qty)
            self.quantity = new_qty
        else:
            closing_qty = min(abs(quantity), abs(self.quantity))
            # Long: profit = (exit - basis); short: profit = (basis - exit).
            direction = 1.0 if self.is_long else -1.0
            realized = closing_qty * (price - self.avg_price) * direction
            self.realized_pnl += realized

            remaining = abs(quantity) - closing_qty
            self.quantity += quantity

            if abs(self.quantity) < _QTY_EPS:
                self.quantity = 0.0
                self.avg_price = 0.0
            elif remaining > _QTY_EPS:
                # Flipped through zero: the remainder is a fresh position.
                self.avg_price = price

        return realized


@dataclass
class PortfolioSnapshot:
    """Immutable record of portfolio state at one point in time."""

    ts: date
    cash: float
    equity: float
    gross_exposure: float
    net_exposure: float
    unrealized_pnl: float
    realized_pnl: float
    cumulative_costs: float
    positions: dict[str, float] = field(default_factory=dict)


class Portfolio:
    """Cash + positions with broker-grade accounting.

    The invariant that matters: ``equity == cash + sum(market values)`` at all
    times, and equity only changes through P&L and costs -- never through the
    act of trading itself.
    """

    def __init__(self, starting_capital: float, *, allow_fractional: bool = False) -> None:
        if starting_capital <= 0:
            raise ValueError("starting_capital must be positive")
        self.starting_capital = float(starting_capital)
        self.cash = float(starting_capital)
        self.allow_fractional = allow_fractional
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0
        # Explicit costs that debit cash directly (commission, fees, borrow).
        self.cumulative_costs = 0.0
        # Implicit cost embedded in the fill price (spread + slippage + impact).
        # It never appears as a cash debit -- it is simply a worse fill -- but it
        # is real money and understating it flatters every backtest.
        self.cumulative_slippage = 0.0
        self.fills: list[Fill] = []
        self._history: list[PortfolioSnapshot] = []

    @property
    def all_in_costs(self) -> float:
        """Total economic cost of trading: explicit charges plus slippage."""
        return self.cumulative_costs + self.cumulative_slippage

    # ---- position access -------------------------------------------------

    def position(self, symbol: str) -> Position:
        return self.positions.setdefault(symbol, Position(symbol))

    def quantity(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        return pos.quantity if pos else 0.0

    # ---- valuation -------------------------------------------------------

    def market_value(self, prices: dict[str, float]) -> float:
        return sum(p.market_value(prices[s]) for s, p in self.positions.items() if not p.is_flat)

    def equity(self, prices: dict[str, float]) -> float:
        """Total account value: cash plus signed mark-to-market of holdings."""
        return self.cash + self.market_value(prices)

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        return sum(p.unrealized_pnl(prices[s]) for s, p in self.positions.items() if not p.is_flat)

    def gross_exposure(self, prices: dict[str, float]) -> float:
        """Sum of absolute position values -- what leverage limits bind on."""
        return sum(
            abs(p.market_value(prices[s])) for s, p in self.positions.items() if not p.is_flat
        )

    def net_exposure(self, prices: dict[str, float]) -> float:
        return self.market_value(prices)

    # ---- mutation --------------------------------------------------------

    def execute(self, fill: Fill) -> float:
        """Apply a fill: move cash, update the position, book costs.

        Returns realized P&L from this fill (excluding its costs).
        """
        qty = fill.quantity if self.allow_fractional else float(round(fill.quantity))
        if abs(qty) < _QTY_EPS:
            return 0.0
        if qty != fill.quantity:
            fill = replace(fill, quantity=qty)

        pos = self.position(fill.symbol)
        realized = pos.apply(qty, fill.price)

        # Buying consumes cash, selling/shorting raises it; costs always drain it.
        self.cash -= qty * fill.price
        self.cash -= fill.total_cost

        self.realized_pnl += realized
        self.cumulative_costs += fill.total_cost
        self.cumulative_slippage += fill.slippage_cost
        pos.total_costs += fill.total_cost
        self.fills.append(fill)
        return realized

    def charge(self, amount: float, symbol: str | None = None) -> None:
        """Debit a non-trade cost such as overnight short borrow."""
        if amount < 0:
            raise ValueError("charge amount must be non-negative")
        self.cash -= amount
        self.cumulative_costs += amount
        if symbol is not None:
            self.position(symbol).total_costs += amount

    # ---- history ---------------------------------------------------------

    def snapshot(self, ts: date, prices: dict[str, float]) -> PortfolioSnapshot:
        snap = PortfolioSnapshot(
            ts=ts,
            cash=self.cash,
            equity=self.equity(prices),
            gross_exposure=self.gross_exposure(prices),
            net_exposure=self.net_exposure(prices),
            unrealized_pnl=self.unrealized_pnl(prices),
            realized_pnl=self.realized_pnl,
            cumulative_costs=self.cumulative_costs,
            positions={s: p.quantity for s, p in self.positions.items() if not p.is_flat},
        )
        self._history.append(snap)
        return snap

    @property
    def history(self) -> list[PortfolioSnapshot]:
        return list(self._history)

    def reset(self) -> None:
        self.cash = self.starting_capital
        self.positions.clear()
        self.realized_pnl = 0.0
        self.cumulative_costs = 0.0
        self.cumulative_slippage = 0.0
        self.fills.clear()
        self._history.clear()
