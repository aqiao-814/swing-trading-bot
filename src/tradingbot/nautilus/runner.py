"""One cron firing: restore the book, replay the new bars, persist the result.

Every thirty minutes this does the same four things, and the order matters:

1. **Refresh** bars for the universe into the store (incremental, bulk).
2. **Replay** a *bounded* window through a fresh ``BacktestEngine``: enough
   trailing bars to warm the alpha's lookbacks, then every bar completed since
   the last run. Only the second part is allowed to trade.
3. **Extract** the resulting balance, positions and fills.
4. **Persist**, so the next firing resumes exactly here.

The window is bounded rather than replayed from inception, and that is a
deliberate trade. Replaying everything would make state a pure function of the
data -- lovely -- but it needs a durable archive of every bar ever seen, and
Yahoo only serves ~60 days of 30-minute history, so such an archive would have
to be carried in the repository and would grow without limit. A bounded window
plus an exact state round-trip gets the same answer at constant cost per run:
resuming from persisted state reproduces an uninterrupted run to the cent, and
there is a test that asserts it.

**Inception is a floor, not a suggestion.** ``paper.start`` fixes the first
tradable bar. Bars before it are still read -- the alpha needs ~20 sessions of
lookback -- but they cannot be traded, so a book with no watermark cannot mistake
the contents of the bar store for its own history. Moving the floor forward is
also how the record is reset: the run retires the old book under ``retired/``
and incepts a new one at full capital. Both halves matter, and the first live
book proved it, incepting itself at ``data_start`` and back-filling three months
of trading into what was supposed to be a forward record.

**No kill switch, and no exposure limits.** There is no drawdown halt, no
daily-loss halt, no model-health halt, and no state flag that can stop the bot.
There is also no gross cap, no per-name cap and no net-exposure clamp unless a
caller sets one: book size is decided by the vol target in
:class:`~tradingbot.nautilus.signals.SizingConfig`, so the bot levers into a
calm cross-section and out of a violent one. What remains is the part that
makes the number mean something -- de-meaned scores, one-bar execution delay,
full transaction costs, and a liquidity floor below which a fill would be
fiction. Risk is *priced*, not *forbidden*. The stated objective is to maximise
simulated PnL, and the honest cost of that objective is that the equity curve
can go a very long way down before it goes up.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.objects import Money

from tradingbot.config import Config
from tradingbot.data.schema import DataQualityError
from tradingbot.data.sources import get_source
from tradingbot.data.store import BarStore
from tradingbot.nautilus.bars import (
    INTERVAL_MINUTES,
    bar_type_for,
    bars_from_frame,
    close_times_ns,
    synthetic_bars,
)
from tradingbot.nautilus.costs import TradingBotFeeModel
from tradingbot.nautilus.instruments import VENUE, build_universe, make_equity
from tradingbot.nautilus.signals import AlphaConfig, NewsConfig2, SizingConfig
from tradingbot.nautilus.state import HeldPosition, V2State
from tradingbot.nautilus.strategy import NewsView, TradingBotV2
from tradingbot.news.signal import NewsSignal
from tradingbot.paper.state import PaperStore
from tradingbot.paper.universe import resolve_universe

_ET = ZoneInfo("America/New_York")

# The clock instrument. Never traded, never scored; its only job is to be the
# first bar delivered at each timestamp so the whole book fills on the same bar.
# See strategy.py for why that is what buys the execution-delay invariant.
CLOCK_SYMBOL = "__CLOCK"


def _inception_instant(value: str | None) -> datetime | None:
    """``paper.start`` as a naive-ET instant: ``2026-08-18`` or with a time."""
    return datetime.fromisoformat(value) if value else None


@dataclass
class RunReport:
    bars_processed: int
    fills: int
    equity: float
    balance: float
    gross: float
    net: float
    n_positions: int
    n_long: int
    n_short: int
    borrow_charged: float
    last_processed: str | None
    incepted: bool


class V2Runner:
    """The NautilusTrader-backed live loop."""

    def __init__(
        self,
        cfg: Config,
        *,
        universe: str | None = None,
        symbols: list[str] | None = None,
        artifacts: str | Path = "artifacts/v2",
        alpha: AlphaConfig | None = None,
        sizing: SizingConfig | None = None,
        news_cfg: NewsConfig2 | None = None,
        # 20x, not 2x. This is the ceiling the margin account will honour, not
        # a target: gross is set by the vol target in SizingConfig, and this
        # only has to be high enough not to be what decides book size. Set it
        # too low and the exchange silently refuses the back half of a
        # rebalance, which looks like alpha decay and is not.
        leverage: Decimal = Decimal("20"),
    ) -> None:
        self.cfg = cfg
        self.paper = cfg.paper
        self.interval = self.paper.interval
        self.universe_name = universe or self.paper.universe
        resolved = symbols if symbols is not None else resolve_universe(self.universe_name, cfg)
        self.symbols = [s for s in resolved if s != CLOCK_SYMBOL]
        self.benchmarks = list(self.paper.benchmark_symbols)
        self.bar_store = BarStore(cfg.data.root)
        self.store = PaperStore(artifacts)
        self.state_path = Path(artifacts) / "portfolio" / "state.json"
        # Inception floor (``paper.start``). Two jobs, and they are the same
        # job: no bar before this instant is ever traded, and a book that
        # incepted before it is retired and re-incepted at full capital. See
        # `_load_or_incept`.
        self.inception = _inception_instant(self.paper.start)
        self.alpha = alpha or AlphaConfig()
        self.sizing = sizing or SizingConfig()
        self.news_cfg = news_cfg or NewsConfig2()
        self.leverage = leverage

    # ---- data ------------------------------------------------------------

    def refresh_data(self, *, log=print) -> None:
        """Incremental bulk refresh for the universe plus benchmarks."""
        source_name = self.cfg.data.source
        if source_name in ("yahoo", "yfinance"):
            source_name = "yahoo_bulk"
        kwargs = {"interval": self.interval} if source_name.startswith("yahoo") else {}
        source = get_source(source_name, **kwargs)

        wanted = sorted(set(self.symbols) | set(self.benchmarks))
        cached = [s for s in wanted if s in self.bar_store]
        fresh = [s for s in wanted if s not in self.bar_store]

        batches: list[tuple[list[str], str]] = []
        if fresh:
            batches.append((fresh, self.paper.data_start))
        if cached:
            coverage = self.bar_store.coverage().filter(pl.col("symbol").is_in(cached))
            stalest = coverage["end"].min() if not coverage.is_empty() else None
            if isinstance(stalest, datetime):
                stalest = stalest.date()
            start = max(
                date.fromisoformat(self.paper.data_start),
                (stalest or date.min) - timedelta(days=5),
            )
            batches.append((cached, start.isoformat()))

        for syms, start in batches:
            log(f"[data] fetching {len(syms)} symbol(s) from {start}")
            try:
                df = source.fetch_many(syms, start, None, on_error="warn")
            except DataQualityError as exc:
                log(f"[data] fetch failed: {exc}")
                continue
            for (sym,), group in df.group_by(["symbol"], maintain_order=True):
                try:
                    self.bar_store.write(group)
                except DataQualityError as exc:
                    log(f"[data] skipping {sym}: {str(exc)[:80]}")

    def _completed_bars(self, as_of: datetime | None = None) -> pl.DataFrame:
        """Every stored bar whose close has already happened.

        A 30-minute bar labelled 15:30 completes at 16:00; a bar still forming
        must never enter the loop, because its close is not yet the close.
        """
        now = as_of or datetime.now(UTC).astimezone(_ET).replace(tzinfo=None)
        bars = self.bar_store.read(self.symbols, start=self.paper.data_start)
        if bars.is_empty():
            return bars
        minutes = INTERVAL_MINUTES[self.interval]
        return bars.filter(pl.col("ts") + pl.duration(minutes=minutes) <= pl.lit(now))

    # ---- the run ---------------------------------------------------------

    def run(self, *, as_of: datetime | None = None, log=print) -> RunReport:
        state, incepted = self._load_or_incept(log=log)
        if incepted:
            # Persist the empty book at once. Inception happens on the first
            # firing after the floor, which is generally hours before the first
            # bar closes below it; without this the dashboard export -- which
            # reads state.json -- would fail on every firing in between.
            state.save(self.state_path)
        bars = self._completed_bars(as_of)
        if bars.is_empty():
            log("[run] no completed bars in the store")
            return self._empty_report(state, incepted)

        grid = sorted(bars["ts"].unique().to_list())
        last = datetime.fromisoformat(state.last_processed) if state.last_processed else None
        # Two independent gates, and the second one only ever matters at
        # inception. The watermark stops a bar being traded twice. The
        # inception floor stops a *fresh* book trading bars that predate it --
        # without it, a book with no watermark treats the entire bar store as
        # "new" and back-fills months of simulated trading into what is
        # supposed to be a forward record, dating its own inception to
        # whatever `data_start` happens to be. Bars below the floor are still
        # read: they warm the alpha's lookbacks, they are just not tradable.
        new_grid = [
            t
            for t in grid
            if (last is None or t > last) and (self.inception is None or t >= self.inception)
        ]
        if not new_grid:
            if self.inception is not None and grid and grid[-1] < self.inception:
                log(f"[run] waiting for inception at {self.inception.isoformat()}")
            else:
                log(f"[run] nothing new; watermark still {state.last_processed}")
            return self._empty_report(state, incepted)

        # Bounded replay: enough trailing bars to warm the lookbacks, then the
        # new ones. One extra timestamp because the clock arrangement spends the
        # first live tick submitting the previous decision.
        warmup = self.alpha.warmup_bars + 2
        first_new_idx = grid.index(new_grid[0])
        window = grid[max(0, first_new_idx - warmup) :]
        frame = bars.filter(pl.col("ts").is_in(window))
        trade_from_ns = int(close_times_ns(pl.Series("ts", [new_grid[0]]), self.interval)[0])
        # Start deciding one bar earlier than trading, so the first tradable bar
        # inherits a pending decision instead of dropping one. At inception
        # there is no earlier bar and none is dropped.
        window_idx = window.index(new_grid[0])
        decide_from_ns = (
            int(close_times_ns(pl.Series("ts", [window[window_idx - 1]]), self.interval)[0])
            if window_idx > 0
            else trade_from_ns
        )

        log(
            f"[run] {len(new_grid)} new bar(s) through {new_grid[-1]}; "
            f"replaying {len(window)} timestamps over {frame['symbol'].n_unique()} symbols"
        )

        engine, strategy, instruments = self._build_engine(
            state, frame, window, trade_from_ns, decide_from_ns
        )
        engine.run()
        report = self._extract(engine, strategy, state, new_grid, incepted, log=log)
        engine.dispose()
        return report

    # ---- engine assembly -------------------------------------------------

    def _build_engine(
        self,
        state: V2State,
        frame: pl.DataFrame,
        window,
        trade_from_ns: int,
        decide_from_ns: int,
    ):
        fee_model = TradingBotFeeModel(self.cfg.env.costs, free=bool(state.positions))
        engine = BacktestEngine(
            config=BacktestEngineConfig(
                trader_id=TraderId("TRADINGBOT-V2"),
                logging=LoggingConfig(bypass_logging=True),
                # The RiskEngine's default 100-orders-per-second cap silently
                # DENIES most of a cross-sectional rebalance: at 670 names one
                # bar can legitimately submit several hundred orders at the same
                # instant. Position and exposure limits are enforced in sizing,
                # where they can be reasoned about, not by dropping orders.
                risk_engine=RiskEngineConfig(bypass=True),
            )
        )
        engine.add_venue(
            venue=VENUE,
            oms_type=OmsType.NETTING,  # one net position per symbol; flips through zero
            account_type=AccountType.MARGIN,  # what makes shorting and leverage possible
            base_currency=USD,
            starting_balances=[Money(state.account_balance, USD)],
            default_leverage=self.leverage,  # must be Decimal; int/float raises
            fee_model=fee_model,
            bar_execution=True,
        )

        by_symbol = {s: g for (s,), g in frame.group_by(["symbol"], maintain_order=True)}
        symbols = sorted(by_symbol)
        instruments = build_universe(symbols)

        ts_ns = close_times_ns(pl.Series("ts", list(window)), self.interval)
        step = INTERVAL_MINUTES[self.interval] * 60 * 1_000_000_000
        restore_ts = np.array([ts_ns[0] - 2 * step, ts_ns[0] - step], dtype=np.uint64)
        needs_restore = bool(state.positions)

        # THE CLOCK INSTRUMENT GOES FIRST. Same-timestamp bars are delivered in
        # insertion order, so inserting this first is what guarantees it is seen
        # before any tradable symbol at every timestamp.
        clock = make_equity(CLOCK_SYMBOL)
        engine.add_instrument(clock)
        clock_bt = bar_type_for(clock, self.interval)
        clock_ts = np.concatenate([restore_ts, ts_ns]) if needs_restore else ts_ns
        engine.add_data(
            synthetic_bars(clock_bt, clock.price_precision, [1.0] * len(clock_ts), clock_ts),
            sort=False,
        )

        book = state.book
        bar_types = {}
        for sym in symbols:
            inst = instruments[sym]
            engine.add_instrument(inst)
            bt = bar_type_for(inst, self.interval)
            bar_types[sym] = bt
            data = bars_from_frame(by_symbol[sym], inst, interval=self.interval)
            if needs_restore:
                # Two flat bars at the position's own average cost, so the
                # rebuild fill inherits exactly that basis. A symbol with no
                # position still needs bars here to keep the stream aligned;
                # its price is irrelevant because it is never traded on them.
                avg = book.get(sym, (0.0, 0.0))[1] or float(by_symbol[sym]["close"][0])
                data = synthetic_bars(bt, inst.price_precision, [avg, avg], restore_ts) + data
            engine.add_data(data, sort=False)

        engine.sort_data()

        strategy = TradingBotV2(
            clock_bar_type=clock_bt,
            bar_types=bar_types,
            fee_model=fee_model,
            restore_book={s: v for s, v in book.items() if s in bar_types},
            trade_from_ns=trade_from_ns,
            decide_from_ns=decide_from_ns,
            news=self._news_view(),
            alpha=self.alpha,
            sizing=self.sizing,
            news_cfg=self.news_cfg,
            borrow_annual_bps=self.cfg.env.costs.short_borrow_annual_bps,
        )
        engine.add_strategy(strategy)
        return engine, strategy, instruments

    # ---- news ------------------------------------------------------------

    def _news_view(self) -> NewsView:
        """Load the weekend signal; anything unreadable degrades to no-news."""
        news_cfg = getattr(self.paper, "news", None)
        if news_cfg is None or not getattr(news_cfg, "enabled", False):
            return NewsView()
        signal = NewsSignal.read(Path(news_cfg.signal_path))
        if signal is None:
            return NewsView()
        try:
            as_of = datetime.fromisoformat(signal.as_of)
        except (TypeError, ValueError):
            return NewsView()
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)
        # `relative` is the cross-sectionally de-meaned company score -- the only
        # form that orders anything, because financial copy is overwhelmingly
        # bullish (measured: mean tone +0.346, only 77 of 635 names negative).
        return NewsView(
            company={s: v.relative for s, v in signal.symbols.items()},
            macro=signal.macro,
            as_of_ns=int(as_of.timestamp() * 1_000_000_000),
        )

    # ---- persistence -----------------------------------------------------

    def _load_or_incept(self, *, log=print) -> tuple[V2State, bool]:
        """The persisted book, or a fresh one at full capital.

        A stored book is *retired* rather than restored when it incepted before
        the configured floor (``paper.start``). That is the reset switch, and it
        is deliberately the same knob that bounds the first tradable bar:
        moving the floor forward both ends the old record and stops the new one
        back-filling itself out of whatever history the bar store holds.

        Retiring moves the old book aside under ``retired/<inception>`` instead
        of deleting it. It was published; a record that has been published is
        not something to overwrite silently, and the equity curve on the site
        is rebuilt from these parquets -- leaving them in place would splice the
        old book's history onto the new book's and call the join an equity
        curve.
        """
        if self.state_path.exists():
            state = V2State.load(self.state_path)
            started = _inception_instant(state.inception)
            if self.inception is None or started is None or started >= self.inception:
                return state, False
            self._retire(state, log=log)
            reason = (
                f"book incepted {state.inception} predates the configured "
                f"inception {self.inception.isoformat()}"
            )
        else:
            reason = "no v2 state"
        log(
            f"[run] {reason} -- incepting at "
            f"${self.cfg.env.starting_capital:,.0f}, all cash, no positions"
        )
        return (
            V2State.incept(
                universe=self.universe_name,
                capital=self.cfg.env.starting_capital,
                interval=self.interval,
                seed=self.cfg.seed,
            ),
            True,
        )

    def _retire(self, state: V2State, *, log=print) -> None:
        """Move a superseded book's state and history under ``retired/``."""
        stamp = (state.inception or "unknown").replace(":", "").replace("-", "")
        dest = self.store.root / "retired" / stamp
        dest.mkdir(parents=True, exist_ok=True)
        moved = []
        for path in sorted(self.store.portfolio_dir.iterdir()):
            if path.is_file():
                path.replace(dest / path.name)
                moved.append(path.name)
        log(f"[run] retired the book incepted {state.inception} -> {dest} ({len(moved)} files)")

    def _extract(self, engine, strategy, state: V2State, new_grid, incepted, *, log=print):
        account = engine.cache.account_for_venue(VENUE)
        balance = float(account.balance_total(USD).as_double())
        prices = strategy.last_prices()

        positions = []
        for p in engine.cache.positions_open():
            sym = p.instrument_id.symbol.value
            if sym == CLOCK_SYMBOL:
                continue
            positions.append(
                HeldPosition(
                    symbol=sym,
                    quantity=float(p.signed_qty),
                    avg_price=float(p.avg_px_open),
                    entry_ts=datetime.fromtimestamp(p.ts_opened / 1e9, UTC).isoformat(
                        timespec="seconds"
                    ),
                )
            )

        # Short borrow. Nautilus never charges this -- a FeeModel only sees
        # fills -- and v1 never owed it, being long-only and flat by every close.
        # The strategy accrued it bar by bar (see TradingBotV2._mark); it is
        # debited from the balance here, which is the same account the next run
        # restores from, so it compounds correctly.
        borrow = strategy.borrow_accrued
        balance -= borrow

        state.account_balance = balance
        state.positions = positions
        state.last_processed = new_grid[-1].isoformat()
        state.inception = state.inception or new_grid[0].isoformat()
        state.n_fills += len(strategy.fills)
        state.cumulative_friction = fee_friction = strategy.fee_model.charged_friction
        state.cumulative_fees = strategy.fee_model.charged_fees
        state.cumulative_borrow += borrow

        equity = state.equity(prices)
        gross = state.gross_exposure(prices) / equity if equity else 0.0
        net = state.net_exposure(prices) / equity if equity else 0.0
        n_long = sum(1 for p in positions if p.quantity > 0)
        n_short = sum(1 for p in positions if p.quantity < 0)

        self._append_history(strategy, state, prices)
        state.save(self.state_path)

        log(
            f"[run] equity ${equity:,.2f} | balance ${balance:,.2f} | "
            f"gross {gross:.2f}x net {net:+.2f}x | {n_long}L/{n_short}S | "
            f"{len(strategy.fills)} fills | friction ${fee_friction:,.2f} borrow ${borrow:,.2f}"
        )
        return RunReport(
            bars_processed=len(new_grid),
            fills=len(strategy.fills),
            equity=equity,
            balance=balance,
            gross=gross,
            net=net,
            n_positions=len(positions),
            n_long=n_long,
            n_short=n_short,
            borrow_charged=borrow,
            last_processed=state.last_processed,
            incepted=incepted,
        )

    def _append_history(self, strategy, state: V2State, prices: dict[str, float]) -> None:
        """Append this run's ledger, fills, decisions and positions to parquet."""

        def _ts(ns: int) -> datetime:
            return datetime.fromtimestamp(int(ns) / 1e9, UTC).astimezone(_ET).replace(tzinfo=None)

        if strategy.ledger:
            self.store.append(
                "ledger",
                pl.DataFrame(
                    [
                        {
                            "ts": _ts(r.ts),
                            "equity": r.equity,
                            "balance": r.balance,
                            "gross": r.gross,
                            "net": r.net,
                            "n_positions": r.n_positions,
                            "n_long": r.n_long,
                            "n_short": r.n_short,
                            "turnover": r.turnover,
                        }
                        for r in strategy.ledger
                    ]
                ),
            )
        if strategy.fills:
            self.store.append(
                "trades",
                pl.DataFrame(
                    [
                        {
                            "ts": _ts(f["ts_ns"]),
                            "symbol": f["symbol"],
                            "action": f["side"],
                            "quantity": f["quantity"],
                            "fill_price": f["fill_price"],
                            "cost": f["commission"],
                        }
                        for f in strategy.fills
                    ]
                ),
            )
        traded = [d for d in strategy.decisions if d.traded]
        if traded:
            self.store.append(
                "decisions",
                pl.DataFrame(
                    [
                        {
                            "ts": _ts(d.ts),
                            "symbol": d.symbol,
                            "score": d.score,
                            "news": d.news,
                            "target_weight": d.target_weight,
                            "current_weight": d.current_weight,
                            "reason": d.reason,
                        }
                        for d in traded
                    ]
                ),
            )
        equity = state.equity(prices)
        self.store.replace(
            "positions",
            pl.DataFrame(
                [
                    {
                        "symbol": p.symbol,
                        "quantity": p.quantity,
                        "avg_price": p.avg_price,
                        "last_price": prices.get(p.symbol, p.avg_price),
                        "value": p.quantity * prices.get(p.symbol, p.avg_price),
                        "weight": (
                            p.quantity * prices.get(p.symbol, p.avg_price) / equity
                            if equity
                            else 0.0
                        ),
                        "unrealized_pnl": p.quantity
                        * (prices.get(p.symbol, p.avg_price) - p.avg_price),
                        "side": "short" if p.quantity < 0 else "long",
                        "entry_ts": p.entry_ts,
                    }
                    for p in state.positions
                ]
                or [],
            ),
        )

    def _empty_report(self, state: V2State, incepted: bool) -> RunReport:
        prices = {p.symbol: p.avg_price for p in state.positions}
        equity = state.equity(prices)
        return RunReport(
            bars_processed=0,
            fills=0,
            equity=equity,
            balance=state.account_balance,
            gross=state.gross_exposure(prices) / equity if equity else 0.0,
            net=state.net_exposure(prices) / equity if equity else 0.0,
            n_positions=len(state.positions),
            n_long=sum(1 for p in state.positions if p.quantity > 0),
            n_short=sum(1 for p in state.positions if p.quantity < 0),
            borrow_charged=0.0,
            last_processed=state.last_processed,
            incepted=incepted,
        )
