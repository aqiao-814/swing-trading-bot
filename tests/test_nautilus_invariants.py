"""The invariants v2 inherits from v1, re-proved against the NautilusTrader engine.

v1 had three tests it was never allowed to break. Rewriting the engine does not
retire them -- it obliges them to be proved again, because every one of them is
now the responsibility of different code:

1. **Execution delay.** A decision computed from bar *t* may only be filled on a
   bar strictly later than *t*, and every symbol in the book must fill on the
   *same* bar. In v1 this was arithmetic in a Python loop. In v2 it is an
   emergent property of the clock-instrument arrangement, which is exactly the
   kind of thing that works until someone reorders an ``add_data`` call.

2. **No look-ahead in the data.** Yahoo labels a 30-minute bar by its OPEN;
   Nautilus fills a market order at the bar's own CLOSE. If the bar bridge
   forgets to shift the timestamp, the engine fills every order thirty minutes
   into its own future and the equity curve becomes fiction.

3. **No free money on noise.** Churning a driftless random walk while paying
   costs has to lose money. A backtest that turns noise into profit has a bug
   in its cost model, and this is the test that finds it.

A fourth is new to v2, because v2 has something v1 did not -- a book that
survives between processes:

4. **State round-trip exactness.** Stopping, persisting, and resuming must
   reproduce an uninterrupted run to the cent.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import polars as pl
import pytest
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.objects import Money
from nautilus_trader.trading.strategy import Strategy

from tradingbot.config import CostConfig
from tradingbot.nautilus.bars import (
    bar_type_for,
    bars_from_frame,
    close_times_ns,
    synthetic_bars,
)
from tradingbot.nautilus.costs import TradingBotFeeModel
from tradingbot.nautilus.instruments import VENUE, make_equity

CLOCK = "__CLOCK"
STEP_NS = 30 * 60 * 1_000_000_000


def _grid(n: int, start=datetime(2026, 5, 4, 9, 30)) -> list[datetime]:
    """A naive-ET 30-minute grid, 13 bars per weekday session."""
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            for k in range(13):
                if len(out) < n:
                    out.append(d + timedelta(minutes=30 * k))
        d = (d + timedelta(days=1)).replace(hour=9, minute=30)
    return out


def _engine(balance: float = 1_000_000.0, fee_model=None) -> BacktestEngine:
    eng = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("TEST-001"),
            logging=LoggingConfig(bypass_logging=True),
            risk_engine=RiskEngineConfig(bypass=True),
        )
    )
    eng.add_venue(
        venue=VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(balance, USD)],
        default_leverage=Decimal("2"),
        fee_model=fee_model,
        bar_execution=True,
    )
    return eng


# ---------------------------------------------------------------------------
# 1. Execution delay
# ---------------------------------------------------------------------------


class _DelayProbe(Strategy):
    """Buys every symbol on one clock tick and records what price it got."""

    def __init__(self, clock_bt, bar_types, trade_at_tick):
        super().__init__()
        self.clock_bt = clock_bt
        self.bar_types = bar_types
        self.trade_at = trade_at_tick
        self.tick = 0
        self.fills: dict[str, float] = {}

    def on_start(self):
        self.subscribe_bars(self.clock_bt)
        for bt in self.bar_types.values():
            self.subscribe_bars(bt)

    def on_bar(self, bar):
        if bar.bar_type != self.clock_bt:
            return
        self.tick += 1
        if self.tick != self.trade_at:
            return
        for bt in self.bar_types.values():
            inst = self.cache.instrument(bt.instrument_id)
            self.submit_order(
                self.order_factory.market(
                    instrument_id=bt.instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=inst.make_qty(10),
                )
            )

    def on_order_filled(self, event):
        self.fills[event.instrument_id.symbol.value] = float(event.last_px.as_double())


def test_execution_delay_is_uniform_and_strictly_after_the_decision_bar():
    """Every symbol fills on the SAME bar, and it is the bar before the clock tick.

    The price of symbol *i* at bar *k* is ``1000*(i+1) + k``, so a fill price
    names the exact bar it came from. Acting on clock tick T must fill the whole
    book at bar T-1 -- never at T (which the strategy has not seen), and never
    at different bars for different symbols.
    """
    n_sym, n_bars, trade_at = 6, 8, 5
    eng = _engine()

    grid = _grid(n_bars)
    ts = close_times_ns(pl.Series("ts", grid), "30m")

    clock = make_equity(CLOCK)
    eng.add_instrument(clock)
    clock_bt = bar_type_for(clock, "30m")
    eng.add_data(synthetic_bars(clock_bt, 2, [1.0] * n_bars, ts), sort=False)

    bar_types = {}
    for i in range(n_sym):
        sym = f"S{i:02d}"
        inst = make_equity(sym)
        eng.add_instrument(inst)
        bt = bar_type_for(inst, "30m")
        bar_types[sym] = bt
        closes = [1000.0 * (i + 1) + k for k in range(n_bars)]
        eng.add_data(synthetic_bars(bt, 2, closes, ts), sort=False)

    eng.sort_data()
    probe = _DelayProbe(clock_bt, bar_types, trade_at)
    eng.add_strategy(probe)
    eng.run()

    assert len(probe.fills) == n_sym, "every symbol should have filled"
    expected_bar = trade_at - 2  # clock tick is 1-based; it fills at bar T-1
    for i in range(n_sym):
        got = probe.fills[f"S{i:02d}"]
        want = 1000.0 * (i + 1) + expected_bar
        assert got == pytest.approx(want), (
            f"S{i:02d} filled at {got}, expected bar {expected_bar} price {want} -- "
            "the book did not fill uniformly one bar behind the decision"
        )
    eng.dispose()


# ---------------------------------------------------------------------------
# 2. No look-ahead: the bar timestamp is the CLOSE
# ---------------------------------------------------------------------------


def test_bar_timestamps_are_shifted_to_the_bar_close():
    """A bar labelled 09:30 must reach the engine stamped 10:00.

    Yahoo labels by the open. Nautilus fills at the bar's close on the clock it
    was stamped with, so leaving the open label in place hands the strategy a
    price thirty minutes ahead of its own clock.
    """
    ts = pl.Series("ts", [datetime(2026, 5, 4, 9, 30), datetime(2026, 5, 4, 10, 0)])
    ns = close_times_ns(ts, "30m")

    got = [datetime.fromtimestamp(int(v) / 1e9, UTC).replace(tzinfo=None) for v in ns]
    # 09:30 ET on 2026-05-04 is 13:30 UTC (EDT); the bar CLOSES at 14:00 UTC.
    assert got[0] == datetime(2026, 5, 4, 14, 0)
    assert got[1] == datetime(2026, 5, 4, 14, 30)
    assert int(ns[1]) - int(ns[0]) == STEP_NS


def test_bar_close_shift_survives_a_daylight_saving_transition():
    """The shift is a timezone conversion, not a fixed offset.

    US DST ended 2026-11-01. A hard-coded -4h would misplace every bar after it
    by an hour, which is a whole bar on this cadence.
    """
    ts = pl.Series("ts", [datetime(2026, 10, 30, 9, 30), datetime(2026, 11, 3, 9, 30)])
    ns = close_times_ns(ts, "30m")
    got = [datetime.fromtimestamp(int(v) / 1e9, UTC).replace(tzinfo=None) for v in ns]
    assert got[0] == datetime(2026, 10, 30, 14, 0)  # EDT: 09:30 -> 13:30, close 14:00
    assert got[1] == datetime(2026, 11, 3, 15, 0)  # EST: 09:30 -> 14:30, close 15:00


def test_bars_from_frame_preserves_prices_and_orders_oldest_first():
    inst = make_equity("AAPL")
    grid = _grid(5)
    df = pl.DataFrame(
        {
            "ts": grid,
            "open": [10.0, 11.0, 12.0, 13.0, 14.0],
            "high": [10.5, 11.5, 12.5, 13.5, 14.5],
            "low": [9.5, 10.5, 11.5, 12.5, 13.5],
            "close": [10.2, 11.2, 12.2, 13.2, 14.2],
            "volume": [1e6] * 5,
        }
    ).with_columns(pl.col("ts").cast(pl.Datetime("us")))

    bars = bars_from_frame(df, inst, interval="30m")
    assert len(bars) == 5
    assert [float(b.close.as_double()) for b in bars] == [10.2, 11.2, 12.2, 13.2, 14.2]
    assert all(bars[i].ts_event < bars[i + 1].ts_event for i in range(4))
    # ts_init must equal ts_event: the engine's clock follows ts_init, and any
    # skew silently shifts every timestamp the state layer writes.
    assert all(b.ts_event == b.ts_init for b in bars)


# ---------------------------------------------------------------------------
# 3. No free money on noise
# ---------------------------------------------------------------------------


class _Churner(Strategy):
    """Flips a market-neutral book long/short every bar. Pure turnover, no view.

    Half the names are held long and half short, and every bar reverses all of
    them. Net exposure is ~zero throughout, so the P&L is dominated by what the
    trading *costs* rather than by which way the noise happened to go.
    """

    def __init__(self, clock_bt, bar_types, shares=200):
        super().__init__()
        self.clock_bt = clock_bt
        self.bar_types = bar_types
        self.shares = shares
        self.tick = 0

    def on_start(self):
        self.subscribe_bars(self.clock_bt)
        for bt in self.bar_types.values():
            self.subscribe_bars(bt)

    def on_bar(self, bar):
        if bar.bar_type != self.clock_bt:
            return
        self.tick += 1
        if self.tick < 2:
            return
        flip = 1 if self.tick % 2 == 0 else -1
        for i, bt in enumerate(self.bar_types.values()):
            inst = self.cache.instrument(bt.instrument_id)
            long = (i % 2 == 0) == (flip == 1)
            target = self.shares if long else -self.shares
            have = int(self.portfolio.net_position(bt.instrument_id))
            delta = target - have
            if delta == 0:
                continue
            self.submit_order(
                self.order_factory.market(
                    instrument_id=bt.instrument_id,
                    order_side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                    quantity=inst.make_qty(abs(delta)),
                )
            )


def _run_churn(seed: int, *, charge_costs: bool) -> tuple[float, float]:
    """Identical trades, with and without the cost model. Returns (balance, charged)."""
    n_bars, n_sym = 60, 6
    start_balance = 1_000_000.0
    fee = TradingBotFeeModel(CostConfig(), free=not charge_costs)
    eng = _engine(start_balance, fee_model=fee)

    grid = _grid(n_bars)
    ts = close_times_ns(pl.Series("ts", grid), "30m")

    clock = make_equity(CLOCK)
    eng.add_instrument(clock)
    clock_bt = bar_type_for(clock, "30m")
    eng.add_data(synthetic_bars(clock_bt, 2, [1.0] * n_bars, ts), sort=False)

    rng = np.random.default_rng(seed)
    bar_types = {}
    for i in range(n_sym):
        sym = f"N{i:02d}"
        inst = make_equity(sym)
        eng.add_instrument(inst)
        bt = bar_type_for(inst, "30m")
        bar_types[sym] = bt
        # Driftless: zero-mean log returns, so E[return] is exactly zero.
        closes = (100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.004, n_bars)))).tolist()
        eng.add_data(synthetic_bars(bt, 2, closes, ts), sort=False)

    eng.sort_data()
    eng.add_strategy(_Churner(clock_bt, bar_types))
    eng.run()
    account = eng.cache.account_for_venue(VENUE)
    balance = float(account.balance_total(USD).as_double())
    charged = fee.charged_friction + fee.charged_fees
    eng.dispose()
    return balance - start_balance, charged


def test_churning_noise_costs_exactly_what_the_cost_model_says():
    """The oldest rule in the repository, stated so that luck cannot pass it.

    Comparing a costed run against a *free* run of the identical trades isolates
    the cost exactly: the price path, the fills and the position sequence are
    all the same, so the entire difference is friction. Asserting only "the
    costed run lost money" would be a coin flip on the noise -- and on the first
    seed tried, it was: a directional churner ended $364 UP.
    """
    for seed in (4, 11, 23):
        costed, charged = _run_churn(seed, charge_costs=True)
        free, zero = _run_churn(seed, charge_costs=False)

        assert zero == 0.0, "the free run must charge nothing"
        assert charged > 0.0, "the cost model charged nothing on a churning book"
        assert costed < free, (
            f"seed {seed}: churning cost nothing -- costed {costed:,.2f} "
            f"vs free {free:,.2f}. Free money on noise means the cost model is not charging."
        )
        # The whole gap is friction, so it must equal what the model says it
        # charged. Money quantises each fill's commission to the cent, so the
        # account's total and the model's running sum differ by up to a cent per
        # fill -- pennies over a few thousand fills. Real leakage would be orders
        # of magnitude larger than this tolerance.
        assert (free - costed) == pytest.approx(charged, rel=1e-4), (
            f"seed {seed}: the account lost {free - costed:,.2f} but the cost model "
            f"reports charging {charged:,.2f} -- costs are leaking somewhere unreported"
        )


def test_churning_a_neutral_book_on_driftless_noise_loses_on_average():
    """And the economic statement: constant turnover with no view is a losing game."""
    pnls = [_run_churn(seed, charge_costs=True)[0] for seed in (4, 11, 23, 37, 51)]
    assert sum(pnls) / len(pnls) < 0.0, (
        f"a market-neutral churner on driftless noise averaged {sum(pnls)/len(pnls):,.2f} "
        "across seeds -- it must lose"
    )


def test_fee_model_charges_every_friction_and_can_be_switched_free():
    """Spread, slippage and impact are explicit commission, not a worse fill price.

    v1 buried them in the fill price, where they never debited cash and so
    reported as ~$0 on trades that cost real money. Here they are commission,
    which is what makes the cost column trustworthy.
    """
    costs = CostConfig()
    fee = TradingBotFeeModel(costs)

    class _Order:
        side = 1  # BUY

    class _Inst:
        quote_currency = USD
        multiplier = None

        class id:
            class symbol:
                value = "AAPL"

    class _Qty:
        def as_double(self):
            return 1000.0

    class _Px:
        def as_double(self):
            return 100.0

    charged = fee.get_commission(_Order(), _Qty(), _Px(), _Inst())
    notional = 100_000.0
    expected = notional * (costs.half_spread_bps + costs.slippage_bps) * 1e-4
    assert float(charged.as_double()) == pytest.approx(expected, rel=1e-6)

    fee.set_free(True)
    assert float(fee.get_commission(_Order(), _Qty(), _Px(), _Inst()).as_double()) == 0.0


def test_sell_side_pays_regulatory_fees_and_buy_side_does_not():
    costs = CostConfig()
    fee = TradingBotFeeModel(costs)

    class _Inst:
        quote_currency = USD
        multiplier = None

        class id:
            class symbol:
                value = "AAPL"

    class _Qty:
        def as_double(self):
            return 1000.0

    class _Px:
        def as_double(self):
            return 100.0

    class _Buy:
        side = 1

    class _Sell:
        side = 2

    buy = float(fee.get_commission(_Buy(), _Qty(), _Px(), _Inst()).as_double())
    sell = float(fee.get_commission(_Sell(), _Qty(), _Px(), _Inst()).as_double())
    # SEC Section 31 plus FINRA TAF land on sells only; a sell is never free.
    assert sell > buy > 0.0


# ---------------------------------------------------------------------------
# 4. State round-trip exactness
# ---------------------------------------------------------------------------


def _seed_store(root, symbols, grid, seed=11):
    """A synthetic bar store on the given naive-ET grid."""
    from tradingbot.data.store import BarStore

    rng = np.random.default_rng(seed)
    frames = []
    n = len(grid)
    for _ in symbols:
        px = 50 + 150 * rng.random()
        prices = px * np.exp(np.cumsum(rng.normal(rng.normal(0, 4e-4), 0.004, n)))
        frames.append(
            pl.DataFrame(
                {
                    "symbol": [_] * n,
                    "ts": grid,
                    "open": prices * (1 + rng.normal(0, 5e-4, n)),
                    "high": prices * 1.003,
                    "low": prices * 0.997,
                    "close": prices,
                    "adj_close": prices,
                    "volume": rng.lognormal(11.5, 0.6, n),
                }
            )
        )
    frame = pl.concat(frames).with_columns(pl.col("ts").cast(pl.Datetime("us")))
    store = BarStore(root)
    store.write(frame, validate_quality=False)
    return store


def test_resuming_from_persisted_state_reproduces_an_uninterrupted_run(tmp_path):
    """Stop, persist, resume -- and land on the same book, to the cent.

    This is what lets the loop run as a bounded replay window instead of
    re-simulating from inception on every cron firing. If it ever stops holding,
    the bot's equity curve silently depends on how often GitHub Actions happened
    to fire, which is not a property any backtest should have.
    """
    from tradingbot.config import Config
    from tradingbot.nautilus.runner import V2Runner
    from tradingbot.nautilus.signals import AlphaConfig, SizingConfig

    symbols = [f"SYM{i:03d}" for i in range(30)]
    grid = _grid(13 * 30)
    alpha = AlphaConfig(reversal_days=1.0, momentum_days=4.0, vol_days=2.0)
    sizing = SizingConfig(target_gross=1.4, max_position_weight=0.08)

    def make_cfg(data_root):
        cfg = Config()
        cfg.data.root = data_root
        cfg.paper.interval = "30m"
        cfg.paper.data_start = "2026-01-01"
        cfg.paper.benchmark_symbols = []
        cfg.env.starting_capital = 100_000.0
        return cfg

    split_at = grid[int(len(grid) * 0.75)] + timedelta(minutes=31)
    end_at = grid[-1] + timedelta(minutes=31)

    # --- one uninterrupted run -------------------------------------------
    root_a = tmp_path / "a"
    _seed_store(root_a / "data", symbols, grid)
    cfg_a = make_cfg(root_a / "data")
    runner_a = V2Runner(
        cfg_a, symbols=symbols, artifacts=root_a / "art", alpha=alpha, sizing=sizing
    )
    whole = runner_a.run(as_of=end_at, log=lambda _m: None)

    # --- the same window, split across two runs ---------------------------
    root_b = tmp_path / "b"
    _seed_store(root_b / "data", symbols, grid)  # identical seed -> identical bars
    cfg_b = make_cfg(root_b / "data")

    def runner_b():
        return V2Runner(
            cfg_b, symbols=symbols, artifacts=root_b / "art", alpha=alpha, sizing=sizing
        )

    runner_b().run(as_of=split_at, log=lambda _m: None)
    resumed = runner_b().run(as_of=end_at, log=lambda _m: None)

    assert resumed.last_processed == whole.last_processed
    assert resumed.balance == pytest.approx(whole.balance, abs=0.01), (
        f"resumed balance {resumed.balance:,.2f} != uninterrupted {whole.balance:,.2f}"
    )
    assert resumed.equity == pytest.approx(whole.equity, abs=0.01)
    assert resumed.n_positions == whole.n_positions
    assert resumed.n_long == whole.n_long
    assert resumed.n_short == whole.n_short
