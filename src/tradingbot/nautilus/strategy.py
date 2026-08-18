"""The v2 Nautilus strategy: rank the cross-section, size it, trade it.

**How a cross-sectional strategy fits an event-driven engine.** Nautilus hands
the strategy one bar at a time. A ranking strategy cannot act on one bar -- it
needs every symbol's bar for the same instant before it can say which names are
strong *relative to the others*. The engine delivers all bars sharing a
timestamp contiguously, so buffering until the timestamp changes is reliable;
the difficulty is that acting on the *first* bar of the new timestamp means one
symbol's matching engine has already advanced while the other 669 have not, and
that symbol would fill at a different bar's price than everyone else.

The fix is a **clock instrument**: a synthetic, never-traded symbol whose bars
are inserted first, so its bar always arrives before any real symbol's bar for
the same timestamp. On the clock bar for timestamp T:

* every real symbol has delivered its bar for T-1, so the slice for T-1 is
  complete and a market order fills at T-1's close;
* no real symbol has delivered T yet, so every fill in the book lands on the
  *same* bar. No per-symbol skew.

That gives the execution-delay invariant for free. At clock tick T the strategy
submits the decision it computed at clock tick T-1 (from data through T-2), so
those orders fill at T-1's close: **a decision is always filled on a bar
strictly later than the one it was computed from.** Measured across every
symbol in ``tests/test_nautilus_invariants.py``, not assumed.

The strategy never uses ``StrategyConfig``. That class is a frozen msgspec
Struct meant for config-driven ``BacktestNode`` runs; this bot builds its engine
in-process, so plain ``__init__`` arguments carry the alpha configuration
without forcing every dataclass through msgspec's serialisation rules.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.trading.strategy import Strategy

from tradingbot.nautilus.costs import TradingBotFeeModel
from tradingbot.nautilus.instruments import share_qty
from tradingbot.nautilus.signals import (
    DEFAULT_ALPHA,
    DEFAULT_NEWS,
    DEFAULT_SIZING,
    AlphaConfig,
    AlphaInputs,
    NewsConfig2,
    SizingConfig,
    apply_news,
    rebalance_needed,
    score_cross_section,
    target_weights,
)


@dataclass
class DecisionRow:
    """One symbol's decision on one bar, for the record."""

    ts: str
    symbol: str
    score: float
    news: float
    target_weight: float
    current_weight: float
    traded: bool
    reason: str


@dataclass
class LedgerRow:
    ts: str
    equity: float
    balance: float
    gross: float
    net: float
    n_positions: int
    n_long: int
    n_short: int
    turnover: float


@dataclass
class NewsView:
    """The weekend signal, reduced to what the strategy is allowed to know.

    ``as_of_ns`` is the instant the signal was collected, not an age. Age is
    computed per bar against the bar's own timestamp, so a signal's staleness is
    a property of the data being processed rather than of when the process
    happened to run. Measuring it against wall-clock instead would mean a replay
    of last month's bars applied whatever decay today's date implied -- the
    same bars would produce different trades on different days, which is exactly
    the kind of irreproducibility the rest of this loop is built to avoid.
    """

    company: dict[str, float] = field(default_factory=dict)
    macro: float = 0.0
    as_of_ns: int | None = None

    def age_days_at(self, bar_ts_ns: int) -> float:
        if self.as_of_ns is None:
            return 0.0
        return max(0.0, (bar_ts_ns - self.as_of_ns) / 86_400_000_000_000)


class TradingBotV2(Strategy):
    """Cross-sectional long/short, vol-targeted, held overnight.

    No kill switch and no exposure limits: book size comes from the volatility
    target in :class:`~tradingbot.nautilus.signals.SizingConfig`, not from a
    gross cap. The execution-delay and no-look-ahead properties below are the
    part that is not negotiable.
    """

    def __init__(
        self,
        *,
        clock_bar_type: BarType,
        bar_types: dict[str, BarType],
        fee_model: TradingBotFeeModel,
        restore_book: dict[str, tuple[float, float]] | None = None,
        trade_from_ns: int = 0,
        decide_from_ns: int | None = None,
        news: NewsView | None = None,
        alpha: AlphaConfig = DEFAULT_ALPHA,
        sizing: SizingConfig = DEFAULT_SIZING,
        news_cfg: NewsConfig2 = DEFAULT_NEWS,
        min_names: int = 20,
        borrow_annual_bps: float = 0.0,
    ) -> None:
        super().__init__()
        self.clock_bar_type = clock_bar_type
        self.bar_types = bar_types
        self.fee_model = fee_model
        self.restore_book = restore_book or {}
        self.trade_from_ns = trade_from_ns
        # A run's first tradable bar must submit the decision that the *previous*
        # bar produced -- otherwise one decision is dropped at every run
        # boundary, and the book a resumed run ends up with is not the book an
        # uninterrupted run would have. Rather than persisting that decision,
        # the resumed run recomputes it: it starts deciding one bar early, from
        # warmup data it already has, which reproduces it exactly.
        self.decide_from_ns = trade_from_ns if decide_from_ns is None else decide_from_ns
        self.news = news or NewsView()
        self.alpha = alpha
        self.sizing = sizing
        self.news_cfg = news_cfg
        self.min_names = min_names
        self.borrow_annual_bps = borrow_annual_bps
        # Borrow accrues on the BARS processed, never on the wall-clock gap
        # between cron firings. Charging per run would make the cost depend on
        # how often GitHub Actions happened to fire, so a bot that ran twice
        # would owe twice as much for holding the same short over the same
        # night -- and a resumed run would not reproduce an uninterrupted one.
        self.borrow_accrued = 0.0
        self._prev_mark_ns: int | None = None

        history = alpha.warmup_bars + 4
        self._closes: dict[str, deque[float]] = {s: deque(maxlen=history) for s in bar_types}
        self._volumes: dict[str, deque[float]] = {s: deque(maxlen=history) for s in bar_types}
        self._last_close: dict[str, float] = {}

        self._tick = 0
        self._restored = not self.restore_book
        self._restore_tick: int | None = None
        # The decision computed on the previous clock tick, awaiting submission.
        self._pending: dict[str, float] | None = None

        self.decisions: list[DecisionRow] = []
        self.ledger: list[LedgerRow] = []
        self.fills: list[dict] = []
        self._turnover_this_bar = 0.0

    # ---- lifecycle -------------------------------------------------------

    def on_start(self) -> None:
        # Free fills until the book is rebuilt; a restore is bookkeeping, not a
        # trade, and charging it would bleed the account a little on every run.
        self.fee_model.set_free(bool(self.restore_book))
        self.subscribe_bars(self.clock_bar_type)
        for bt in self.bar_types.values():
            self.subscribe_bars(bt)

    # ---- the bar loop ----------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        if bar.bar_type != self.clock_bar_type:
            self._absorb(bar)
            return
        self._tick += 1

        if not self._restored:
            # Restore bars are ingested by tick 2; rebuild the book against them.
            if self._tick >= 2:
                self._rebuild_book()
                self._restored = True
                self._restore_tick = self._tick
            return

        if self._restore_tick is not None and self._tick > self._restore_tick:
            # Deliberately a LATER bar than the one that submitted the rebuild:
            # fills are processed after the callback returns, so clearing this
            # inline would charge the restore. Verified to cost exactly the
            # book's notional x the fee rate when done wrong.
            self.fee_model.set_free(False)
            self._restore_tick = None

        ts = int(bar.ts_event)
        live = ts >= self.trade_from_ns

        # 1. Submit the previous tick's decision. It was computed from the slice
        #    two timestamps back, so it fills on the bar after the one it saw.
        if live and self._pending is not None:
            self._submit(self._pending, bar)
        # 2. Rank the slice that just completed; it trades on the NEXT tick.
        #    This begins one bar BEFORE trading does, so the first tradable bar
        #    has a decision waiting exactly as it would mid-run.
        if ts >= self.decide_from_ns:
            self._pending = self._decide(bar, record=live)

        if live:
            self._mark(bar)

    def on_stop(self) -> None:
        # The final timestamp never gets a following clock bar, so its decision
        # would otherwise be lost. It is deliberately NOT submitted: there is no
        # later bar for it to fill against, and filling it here would fill it on
        # the very bar it was computed from -- exactly the look-ahead the whole
        # clock-instrument arrangement exists to prevent. It is recorded as a
        # queued order and submitted by the next run instead.
        pass

    # ---- steps -----------------------------------------------------------

    def _absorb(self, bar: Bar) -> None:
        sym = bar.bar_type.instrument_id.symbol.value
        if sym not in self._closes:
            return
        close = float(bar.close.as_double())
        self._closes[sym].append(close)
        self._volumes[sym].append(float(bar.volume.as_double()))
        self._last_close[sym] = close

    def _rebuild_book(self) -> None:
        """Re-establish the persisted positions at their persisted cost basis."""
        for sym, (qty, _avg) in self.restore_book.items():
            bt = self.bar_types.get(sym)
            if bt is None or qty == 0:
                continue
            instrument = self.cache.instrument(bt.instrument_id)
            q = share_qty(instrument, abs(qty))
            if q is None:
                continue
            self.submit_order(
                self.order_factory.market(
                    instrument_id=bt.instrument_id,
                    order_side=OrderSide.BUY if qty > 0 else OrderSide.SELL,
                    quantity=q,
                )
            )

    def _equity(self) -> float:
        venue = self.clock_bar_type.instrument_id.venue
        eq = self.portfolio.equity(venue)
        if not eq:
            return 0.0
        return float(next(iter(eq.values())).as_double())

    def _current_weights(self, equity: float) -> dict[str, float]:
        """Signed position value as a fraction of NAV.

        ``portfolio.net_exposure`` is unusable here: despite the name it returns
        an absolute notional, so a short reports positive. The signed position
        has to be multiplied out by hand.
        """
        if equity <= 0.0:
            return {}
        out = {}
        for sym, bt in self.bar_types.items():
            qty = float(self.portfolio.net_position(bt.instrument_id))
            if qty:
                out[sym] = qty * self._last_close.get(sym, 0.0) / equity
        return out

    def _decide(self, clock_bar: Bar, *, record: bool = True) -> dict[str, float] | None:
        closes = {
            s: np.asarray(d, dtype=float) for s, d in self._closes.items() if len(d) >= 2
        }
        if len(closes) < self.min_names:
            return None
        volumes = {s: np.asarray(self._volumes[s], dtype=float) for s in closes}

        scores, vols = score_cross_section(AlphaInputs(closes, volumes), self.alpha)
        if len(scores) < self.min_names:
            return None

        tilted, net_tilt = apply_news(
            scores,
            company=self.news.company,
            macro=self.news.macro,
            age_days=self.news.age_days_at(int(clock_bar.ts_event)),
            cfg=self.news_cfg,
        )
        dollar_volume = {
            s: float(np.median(volumes[s][-26:] * closes[s][-26:]))
            for s in tilted
            if s in volumes and volumes[s].size >= 26
        }
        targets = target_weights(
            tilted, vols, net_tilt=net_tilt, dollar_volume=dollar_volume, cfg=self.sizing
        )

        ts = str(clock_bar.ts_event)
        equity = self._equity()
        current = self._current_weights(equity)
        # Anything currently held that the ranking no longer selects is an exit,
        # not an omission. Without this a name simply stops being mentioned and
        # is held forever.
        for sym in current:
            targets.setdefault(sym, 0.0)

        if not record:
            # A warmup decision: real, and submitted on the first live bar, but
            # it belongs to the previous run's log, not this one's.
            return targets

        for sym, w in targets.items():
            cur = current.get(sym, 0.0)
            trade = rebalance_needed(w, cur, self.sizing) or (w == 0.0 and cur != 0.0)
            self.decisions.append(
                DecisionRow(
                    ts=ts,
                    symbol=sym,
                    score=float(tilted.get(sym, 0.0)),
                    news=float(self.news.company.get(sym, 0.0)),
                    target_weight=w,
                    current_weight=cur,
                    traded=trade,
                    reason="exit" if w == 0.0 else ("entry" if cur == 0.0 else "rebalance"),
                )
            )
        return targets

    def _submit(self, targets: dict[str, float], clock_bar: Bar) -> None:
        equity = self._equity()
        if equity <= 0.0:
            return
        current = self._current_weights(equity)
        self._turnover_this_bar = 0.0

        for sym, target in targets.items():
            bt = self.bar_types.get(sym)
            price = self._last_close.get(sym, 0.0)
            if bt is None or price <= 0.0:
                continue
            cur = current.get(sym, 0.0)
            if not (rebalance_needed(target, cur, self.sizing) or (target == 0.0 and cur != 0.0)):
                continue

            instrument = self.cache.instrument(bt.instrument_id)
            want_shares = target * equity / price
            have_shares = float(self.portfolio.net_position(bt.instrument_id))
            delta = want_shares - have_shares
            q = share_qty(instrument, delta)
            if q is None:
                continue
            self._turnover_this_bar += abs(float(q.as_double())) * price
            # The fee model prices impact against the bar's own volume, which it
            # is not handed, so push the market context in before submitting.
            vol_hist = self._volumes.get(sym)
            sigma = 0.02
            closes = self._closes.get(sym)
            if closes and len(closes) > 3:
                arr = np.asarray(closes, dtype=float)[-27:]
                if np.all(arr > 0):
                    r = np.diff(np.log(arr))
                    if r.size:
                        sigma = float(np.std(r)) or 0.02
            self.fee_model.observe(
                sym, volume=(vol_hist[-1] if vol_hist else 0.0), sigma=sigma
            )
            self.submit_order(
                self.order_factory.market(
                    instrument_id=bt.instrument_id,
                    order_side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                    quantity=q,
                )
            )

    def _mark(self, clock_bar: Bar) -> None:
        equity = self._equity()
        venue = self.clock_bar_type.instrument_id.venue
        account = self.portfolio.account(venue)
        balance = (
            float(account.balance_total(account.base_currency).as_double()) if account else 0.0
        )

        gross = net = 0.0
        n_long = n_short = 0
        for sym, bt in self.bar_types.items():
            qty = float(self.portfolio.net_position(bt.instrument_id))
            if not qty:
                continue
            value = qty * self._last_close.get(sym, 0.0)
            gross += abs(value)
            net += value
            n_long += qty > 0
            n_short += qty < 0

        # Short borrow, accrued over the elapsed time since the previous mark.
        # Using elapsed bar time rather than sessions is what makes an overnight
        # or weekend hold cost what it should: the timestamps jump, so the gap
        # does the work.
        ts_ns = int(clock_bar.ts_event)
        if self.borrow_annual_bps > 0.0 and self._prev_mark_ns is not None:
            short_notional = 0.0
            for sym, bt in self.bar_types.items():
                q = float(self.portfolio.net_position(bt.instrument_id))
                if q < 0:
                    short_notional += abs(q) * self._last_close.get(sym, 0.0)
            days = max(0.0, (ts_ns - self._prev_mark_ns) / 86_400_000_000_000)
            self.borrow_accrued += (
                short_notional * (self.borrow_annual_bps * 1e-4) * (days / 360.0)
            )
        self._prev_mark_ns = ts_ns

        self.ledger.append(
            LedgerRow(
                ts=str(clock_bar.ts_event),
                equity=equity,
                balance=balance,
                gross=gross / equity if equity > 0 else 0.0,
                net=net / equity if equity > 0 else 0.0,
                n_positions=n_long + n_short,
                n_long=n_long,
                n_short=n_short,
                turnover=self._turnover_this_bar / equity if equity > 0 else 0.0,
            )
        )
        self._turnover_this_bar = 0.0

    # ---- events ----------------------------------------------------------

    def on_order_filled(self, event) -> None:
        if self.fee_model._free:  # restore fills are bookkeeping, not trades
            return
        self.fills.append(
            {
                "ts_ns": int(event.ts_event),
                "symbol": event.instrument_id.symbol.value,
                "side": "buy" if event.order_side == OrderSide.BUY else "sell",
                "quantity": float(event.last_qty.as_double()),
                "fill_price": float(event.last_px.as_double()),
                "commission": float(event.commission.as_double()) if event.commission else 0.0,
            }
        )

    @property
    def pending_targets(self) -> dict[str, float]:
        """The decision computed on the final bar, which no bar could fill.

        The next run submits it. This is what makes the loop resumable without
        either losing a decision or filling it on the bar it was computed from.
        """
        return dict(self._pending or {})

    def last_prices(self) -> dict[str, float]:
        return dict(self._last_close)
