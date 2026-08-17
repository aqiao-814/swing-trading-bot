"""The trading universe as NautilusTrader ``Equity`` instruments.

One ``Equity`` per ticker, built directly rather than through
``TestInstrumentProvider.equity()``, which hard-codes Apple's ISIN for every
symbol, a 100-share lot, and zero margin requirements -- all three wrong for a
670-name margin book.

Two properties of ``Equity`` drive the rest of v2 and are not configurable:

* **Whole shares only.** ``Equity`` fixes ``size_precision=0`` /
  ``size_increment=1`` in its constructor. A fractional order is not rounded,
  it is *denied* by the risk engine before it reaches the venue, so every size
  this package computes goes through :func:`share_qty`.
* **Two-decimal prices.** Fine for this universe precisely because v1's screen
  already floors it at a $5 share price; sub-dollar names would need sub-penny
  increments and, far more importantly, a cost model that does not pretend a
  1 bp half-spread is honest for them.
"""

from __future__ import annotations

from decimal import Decimal

from nautilus_trader.model.currencies import USD
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity

from swingbot.nautilus import VENUE_NAME

VENUE = Venue(VENUE_NAME)

# Regulation T. `margin_init` is what a new position locks, `margin_maint` what
# it keeps locked. Nautilus's default LeveragedMarginModel divides these by the
# account leverage, so with default_leverage=2 a $100k short locks $12.5k
# maintenance rather than $25k. Leaving them at Equity's default of 0 -- which
# is what TestInstrumentProvider does -- means an unlimited book locks nothing
# and the margin account is decorative.
REG_T_INIT = Decimal("0.50")
REG_T_MAINT = Decimal("0.25")

# US equities quote in pennies above $1.00 (Reg NMS Rule 612).
PRICE_PRECISION = 2
PRICE_INCREMENT = Price.from_str("0.01")


def make_equity(symbol: str, venue: Venue = VENUE) -> Equity:
    """One US equity instrument.

    ``lot_size=1`` is deliberate: it is not what constrains order size (that is
    ``size_increment``, which ``Equity`` fixes at 1 regardless), but a lot size
    of 100 makes round-trip quantity arithmetic read as if odd lots were
    illegal, which they are not.
    """
    sym = Symbol(symbol.upper())
    return Equity(
        instrument_id=InstrumentId(sym, venue),
        raw_symbol=sym,
        currency=USD,
        price_precision=PRICE_PRECISION,
        price_increment=PRICE_INCREMENT,
        lot_size=Quantity.from_int(1),
        ts_event=0,
        ts_init=0,
        margin_init=REG_T_INIT,
        margin_maint=REG_T_MAINT,
        # Commission is charged by SwingbotFeeModel, which models spread,
        # slippage, impact and regulatory fees together. Leaving these at zero
        # keeps the default MakerTakerFeeModel from double-charging if the fee
        # model is ever omitted.
        maker_fee=Decimal(0),
        taker_fee=Decimal(0),
        isin=None,
    )


def build_universe(symbols: list[str], venue: Venue = VENUE) -> dict[str, Equity]:
    """``{symbol: Equity}`` for the whole universe, deduplicated and sorted.

    Sorted so that instrument-registration order -- and therefore the order the
    engine reports things in -- is a function of the universe alone, not of how
    the symbol list happened to be assembled.
    """
    return {s: make_equity(s, venue) for s in sorted({x.upper() for x in symbols})}


def share_qty(instrument: Equity, shares: float) -> Quantity | None:
    """Round a desired share count down to a legal, non-zero ``Quantity``.

    Rounds **down**, never up: rounding a 0.6-share target up to 1 would let a
    book of several hundred names each overshoot its target by up to half a
    share, and at 670 names that is real money spent on positions nobody asked
    for. ``None`` means "too small to trade" -- the caller drops the order
    rather than queueing one that would be denied.
    """
    n = abs(shares)
    if n < 1.0:
        return None
    # make_qty(round_down=True) still raises if the result would be zero, which
    # the guard above already precludes; the guard is kept because relying on an
    # exception for ordinary control flow here would be silly.
    return instrument.make_qty(n, round_down=True)
