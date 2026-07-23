"""Forward paper-trading invariants.

The paper engine manages a *persistent* simulated portfolio, so the failure
modes worth testing are the stateful ones: double-processing a day, leaking
tomorrow's bar into today's decision, cash going negative, state or model not
surviving a restart, and two identical runs disagreeing.

Everything runs on the synthetic source: deterministic, offline, and by
construction free of edge -- which also makes look-ahead detectable.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from swingbot.config import Config
from swingbot.data.sources import SyntheticSource
from swingbot.data.store import BarStore
from swingbot.paper.dashboard import build_paper_dashboard
from swingbot.paper.engine import PaperEngine, stop_cooldown_active, target_gross_exposure
from swingbot.paper.learner import ContinualRRL
from swingbot.paper.state import PaperState
from swingbot.paper.universe import resolve_universe

SYMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
AS_OF = date(2024, 8, 30)


def make_cfg(tmp: Path) -> Config:
    cfg = Config()
    cfg.data.root = tmp / "data"
    cfg.data.source = "synthetic"
    cfg.data.universe = list(SYMS)
    cfg.artifacts_root = tmp / "artifacts"
    cfg.paper.universe = "config"
    cfg.paper.start = "2024-06-03"
    cfg.paper.data_start = "2019-01-01"
    cfg.paper.pretrain_years = 1.0
    # Synthetic signals are weak; lower the bar so the tests exercise trading.
    cfg.paper.min_conviction = 0.02
    cfg.paper.exit_conviction = 0.005
    # Kill switches off by default in tests: the synthetic policy's conviction
    # spread is legitimately tiny, and most tests exercise normal trading.
    # TestKillSwitches turns them on selectively.
    cfg.paper.kill_max_drawdown = None
    cfg.paper.kill_daily_loss = None
    cfg.paper.kill_rolling_20d_loss = None
    cfg.paper.kill_conviction_std = None
    return cfg


def seed_store(cfg: Config, end: str = "2024-08-30") -> BarStore:
    src = SyntheticSource(seed=7, regime_switching=True)
    store = BarStore(cfg.data.root)
    for sym in SYMS + cfg.paper.benchmark_symbols:
        store.write(src.fetch(sym, "2019-01-01", end))
    return store


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory) -> tuple[Config, PaperEngine, object]:
    """One full engine run shared by read-only assertions."""
    tmp = tmp_path_factory.mktemp("paper")
    cfg = make_cfg(tmp)
    seed_store(cfg)
    engine = PaperEngine(cfg)
    summary = engine.run(capital=100_000, as_of=AS_OF, refresh=False, log=lambda m: None)
    return cfg, engine, summary


class TestIdempotency:
    def test_second_run_same_day_is_a_noop(self, tmp_path):
        cfg = make_cfg(tmp_path)
        seed_store(cfg)
        engine = PaperEngine(cfg)
        s1 = engine.run(capital=100_000, as_of=AS_OF, refresh=False, log=lambda m: None)
        trades_before = engine.store.read("trades")
        learn_before = engine.store.read("learning")

        s2 = engine.run(capital=100_000, as_of=AS_OF, refresh=False, log=lambda m: None)
        assert s2.days == []  # nothing re-processed
        assert s2.equity == pytest.approx(s1.equity)
        # No duplicate trades and no duplicate training.
        assert engine.store.read("trades").height == trades_before.height
        assert engine.store.read("learning").height == learn_before.height
        assert s2.learning["n_updates"] == s1.learning["n_updates"]

    def test_catchup_equals_day_by_day(self, tmp_path):
        """Processing N days in one run == processing them one at a time."""
        cfg_a, cfg_b = make_cfg(tmp_path / "a"), make_cfg(tmp_path / "b")
        seed_store(cfg_a)
        seed_store(cfg_b)

        eng_a = PaperEngine(cfg_a)
        eng_a.run(capital=100_000, as_of=AS_OF, refresh=False, log=lambda m: None)

        eng_b = PaperEngine(cfg_b)
        for cut in [date(2024, 7, 1), date(2024, 8, 1), AS_OF]:
            eng_b.run(capital=100_000, as_of=cut, refresh=False, log=lambda m: None)

        la = eng_a.store.read("ledger").sort("ts")
        lb = eng_b.store.read("ledger").sort("ts")
        assert la["ts"].to_list() == lb["ts"].to_list()
        np.testing.assert_allclose(la["equity"].to_numpy(), lb["equity"].to_numpy(), rtol=1e-12)


class TestNoLookahead:
    def test_decisions_unchanged_by_future_bars(self, tmp_path):
        """Adding future bars must not change any already-made decision.

        The two stores share bit-identical bars up to the cutoff; the long one
        additionally contains six more weeks of future data. If any decision or
        equity value differs, something read a bar it should not have seen.
        """
        cfg_short, cfg_long = make_cfg(tmp_path / "s"), make_cfg(tmp_path / "l")
        cut = date(2024, 7, 15)
        src = SyntheticSource(seed=7, regime_switching=True)
        short_store, long_store = BarStore(cfg_short.data.root), BarStore(cfg_long.data.root)
        for sym in SYMS + cfg_long.paper.benchmark_symbols:
            bars = src.fetch(sym, "2019-01-01", "2024-08-30")
            long_store.write(bars)
            short_store.write(bars.filter(pl.col("ts") <= cut))

        eng_s = PaperEngine(cfg_short)
        eng_s.run(capital=100_000, as_of=cut, refresh=False, log=lambda m: None)
        eng_l = PaperEngine(cfg_long)
        eng_l.run(capital=100_000, as_of=cut, refresh=False, log=lambda m: None)

        ds = eng_s.store.read("decisions").sort(["ts", "symbol"])
        dl = eng_l.store.read("decisions").sort(["ts", "symbol"])
        assert ds.drop("result").equals(dl.drop("result"))
        ls = eng_s.store.read("ledger").sort("ts")
        ll = eng_l.store.read("ledger").sort("ts")
        np.testing.assert_allclose(ls["equity"].to_numpy(), ll["equity"].to_numpy(), rtol=1e-12)

    def test_orders_fill_strictly_after_decision(self, completed_run):
        _, engine, _ = completed_run
        trades = engine.store.read("trades")
        assert trades.height > 0
        assert (trades["ts"] > trades["decided_ts"]).all()

    def test_latest_completed_never_returns_the_future(self):
        assert PaperEngine.latest_completed(date(2020, 5, 4)) == date(2020, 5, 4)
        assert PaperEngine.latest_completed(None) <= date.today()


class TestCostsAndAccounting:
    def test_buys_fill_above_reference_sells_below(self, completed_run):
        _, engine, _ = completed_run
        trades = engine.store.read("trades")
        buys = trades.filter(pl.col("quantity") > 0)
        sells = trades.filter(pl.col("quantity") < 0)
        assert buys.height > 0
        assert (buys["fill_price"] > buys["reference_price"]).all()
        if sells.height:
            assert (sells["fill_price"] < sells["reference_price"]).all()

    def test_slippage_and_fees_are_recorded(self, completed_run):
        _, engine, _ = completed_run
        trades = engine.store.read("trades")
        assert (trades["slippage"] > 0).all()  # adverse by construction
        ledger = engine.store.read("ledger").sort("ts")
        assert float(ledger["slippage_costs"][-1]) > 0
        # SEC fees only exist on sells.
        assert (trades.filter(pl.col("quantity") > 0)["fees"] == 0).all()

    def test_equity_equals_cash_plus_market_value(self, completed_run):
        _, engine, summary = completed_run
        state = PaperState.load(engine.store.state_path)
        pf = state.to_portfolio()
        prices = {p["symbol"]: p["current_price"] for p in summary.positions}
        assert pf.equity(prices) == pytest.approx(summary.equity)
        assert state.cash == pytest.approx(summary.cash)

    def test_cash_never_negative(self, completed_run):
        _, engine, _ = completed_run
        ledger = engine.store.read("ledger")
        assert (ledger["cash"] >= 0).all()

    def test_ledger_equity_is_consistent_with_daily_returns(self, completed_run):
        _, engine, _ = completed_run
        ledger = engine.store.read("ledger").sort("ts")
        eq = np.array([100_000.0, *ledger["equity"].to_list()])
        rets = ledger["daily_return"].to_numpy()
        np.testing.assert_allclose(eq[1:] / eq[:-1] - 1.0, rets, atol=1e-9)


class TestAllocation:
    def test_weights_bounded_and_gross_within_limit(self, completed_run):
        cfg, engine, _ = completed_run
        decisions = engine.store.read("decisions")
        opens = decisions.filter(pl.col("action").is_in(["buy", "rebalance"]))
        assert (opens["allocation"].abs() <= cfg.paper.max_position_weight + 1e-9).all()
        by_day = opens.group_by("ts").agg(pl.col("allocation").abs().sum().alias("gross"))
        assert (by_day["gross"] <= cfg.paper.max_gross_exposure + 1e-9).all()

    def test_never_forced_fully_invested(self, completed_run):
        _, engine, _ = completed_run
        ledger = engine.store.read("ledger")
        assert (ledger["cash"] > 0).all()  # some cash held every single day


class TestStopDiscipline:
    """A stop-out must convert risk to cash, not rotate it into the next name."""

    @staticmethod
    def fresh_state() -> PaperState:
        return PaperState(universe="config", starting_capital=1e5, cash=1e5, seed=7)

    def test_stopped_symbol_is_locked_out_then_eligible(self):
        p = Config().paper
        state = self.fresh_state()
        state.last_stop_out["AAA"] = "2024-06-03"
        assert stop_cooldown_active(p, state, "AAA", date(2024, 6, 4))
        assert stop_cooldown_active(p, state, "AAA", date(2024, 6, 12))
        assert not stop_cooldown_active(p, state, "AAA", date(2024, 6, 13))
        assert not stop_cooldown_active(p, state, "BBB", date(2024, 6, 4))

    def test_stop_outs_de_gross_the_book(self):
        p = Config().paper  # gross 0.90, -0.10 per stop, floor 0.30
        state = self.fresh_state()
        d = date(2024, 6, 10)
        assert target_gross_exposure(p, state, d) == pytest.approx(0.90)
        state.last_stop_out = {"AAA": "2024-06-05", "BBB": "2024-06-07"}
        assert target_gross_exposure(p, state, d) == pytest.approx(0.70)
        # Stops age out of the window; the cap recovers.
        assert target_gross_exposure(p, state, date(2024, 7, 1)) == pytest.approx(0.90)
        # The floor holds no matter how many stops fire.
        state.last_stop_out = {f"S{i}": "2024-06-09" for i in range(10)}
        assert target_gross_exposure(p, state, d) == pytest.approx(0.30)

    def test_stop_out_recorded_and_no_reentry_within_cooldown(self, tmp_path):
        cfg = make_cfg(tmp_path)
        cfg.paper.stop_loss_sigma = 0.05  # hair trigger: synthetic noise must stop out
        seed_store(cfg)
        engine = PaperEngine(cfg)
        engine.run(capital=100_000, as_of=AS_OF, refresh=False, log=lambda m: None)

        trades = engine.store.read("trades")
        stops = trades.filter(pl.col("reason") == "stop_loss")
        assert not stops.is_empty()
        state = PaperState.load(engine.store.state_path)
        assert state.last_stop_out  # every stop fill left a cooldown record

        cooldown = cfg.paper.stop_cooldown_days
        for stop in stops.iter_rows(named=True):
            after = trades.filter(
                (pl.col("symbol") == stop["symbol"])
                & (pl.col("action") == "buy")
                & (pl.col("ts") > stop["ts"])
            )
            for buy in after.iter_rows(named=True):
                days_out = (buy["ts"] - stop["ts"]).days
                assert days_out > cooldown, (
                    f"{stop['symbol']} re-bought {days_out}d after its stop-out"
                )


class TestKillSwitches:
    """A fired switch flattens the book, halts entries, and survives restarts."""

    def test_daily_loss_kill_flattens_and_halts(self, tmp_path):
        cfg = make_cfg(tmp_path)
        cfg.paper.kill_daily_loss = 0.0001  # any down day fires
        seed_store(cfg)
        engine = PaperEngine(cfg)
        engine.run(capital=100_000, as_of=AS_OF, refresh=False, log=lambda m: None)

        state = PaperState.load(engine.store.state_path)
        assert state.halted and "daily_loss" in state.halted
        assert not state.positions  # the flatten orders actually filled
        halted_day = date.fromisoformat(state.halted_ts)
        buys_after = engine.store.read("trades").filter(
            (pl.col("action") == "buy") & (pl.col("ts") > halted_day)
        )
        assert buys_after.is_empty()

    def test_model_health_kill_never_lets_the_book_open(self, tmp_path):
        """conviction_std below the bar means the scores are degenerate; a
        book allocated by a degenerate ranking has no reason to exist."""
        cfg = make_cfg(tmp_path)
        cfg.paper.kill_conviction_std = 10.0  # impossible bar: fires on day one
        seed_store(cfg)
        engine = PaperEngine(cfg)
        summary = engine.run(capital=100_000, as_of=AS_OF, refresh=False, log=lambda m: None)

        state = PaperState.load(engine.store.state_path)
        assert state.halted and "conviction_std" in state.halted
        assert engine.store.read("trades").is_empty()  # never traded at all
        assert summary.equity == pytest.approx(100_000)
        assert summary.halted == state.halted

    def test_clear_halt_is_an_explicit_operator_action(self, tmp_path):
        cfg = make_cfg(tmp_path)
        cfg.paper.kill_conviction_std = 10.0
        seed_store(cfg)
        PaperEngine(cfg).run(capital=100_000, as_of=AS_OF, refresh=False, log=lambda m: None)

        # A plain re-run must NOT clear the halt.
        engine2 = PaperEngine(cfg)
        engine2.run(capital=100_000, as_of=AS_OF, refresh=False, log=lambda m: None)
        assert PaperState.load(engine2.store.state_path).halted

        engine3 = PaperEngine(cfg)
        engine3.run(
            capital=100_000, as_of=AS_OF, refresh=False, clear_halt=True, log=lambda m: None
        )
        assert PaperState.load(engine3.store.state_path).halted is None


class TestPersistence:
    def test_state_survives_restart(self, completed_run):
        cfg, engine, summary = completed_run
        # A brand-new engine instance must see the exact same portfolio.
        engine2 = PaperEngine(cfg)
        s2 = engine2.run(capital=100_000, as_of=AS_OF, refresh=False, log=lambda m: None)
        assert s2.days == []
        assert s2.equity == pytest.approx(summary.equity)
        assert [p["symbol"] for p in s2.positions] == [p["symbol"] for p in summary.positions]

    def test_state_file_is_flagged_simulated(self, completed_run):
        _, engine, _ = completed_run
        state = PaperState.load(engine.store.state_path)
        assert state.simulated_capital is True

    def test_model_checkpoints_persist_and_roundtrip(self, completed_run):
        cfg, engine, _ = completed_run
        latest = cfg.artifacts_root / "models" / "rrl_latest.bin"
        assert latest.exists()
        assert list((cfg.artifacts_root / "models" / "checkpoints").glob("rrl_*.bin"))

        a = ContinualRRL.load(latest)
        b = ContinualRRL.load(latest)
        np.testing.assert_array_equal(a.agent.w, b.agent.w)
        assert a.n_updates == b.n_updates
        # Round-trip through save/load preserves everything bit-for-bit.
        p = latest.parent / "roundtrip.bin"
        a.save(p)
        c = ContinualRRL.load(p)
        np.testing.assert_array_equal(a.agent.w, c.agent.w)
        assert c._states.keys() == a._states.keys()


class TestContinualLearning:
    def test_updates_move_weights_and_accumulate(self, tmp_path):
        learner = ContinualRRL([f"f{i}" for i in range(4)], seed=3)
        w0 = learner.agent.w.copy()
        rng = np.random.default_rng(0)
        for _ in range(200):
            learner.observe("AAA", rng.normal(size=4), float(rng.normal(0, 0.01)), 0.0003)
        assert learner.n_updates == 200
        assert not np.allclose(learner.agent.w, w0)

    def test_engine_learns_once_per_symbol_day(self, completed_run):
        _, engine, summary = completed_run
        learning = engine.store.read("learning").sort("ts")
        # Every processed day trained on every feature-complete symbol exactly once.
        assert (learning["day_updates"] == len(SYMS)).all()
        pretrain = int(learning["n_updates"][0]) - int(learning["day_updates"][0])
        assert summary.learning["n_updates"] == pretrain + int(learning["day_updates"].sum())

    def test_per_symbol_recurrent_state_is_isolated(self):
        learner = ContinualRRL([f"f{i}" for i in range(3)], seed=1)
        x = np.ones(3)
        learner.observe("AAA", x, 0.01, 0.0)
        st_bbb_before = learner._state("BBB").f_prev
        assert learner._state("AAA").f_prev != 0.0
        assert st_bbb_before == 0.0  # AAA's update never touches BBB's recurrence

    def test_weight_norm_never_exceeds_cap(self):
        """The saturation guard: once ||w|| drifts past ~2, tanh pins at +/-1
        and conviction ranking degenerates. The cap is a hard invariant."""
        learner = ContinualRRL([f"f{i}" for i in range(6)], seed=2, max_weight_norm=0.5)
        rng = np.random.default_rng(0)
        for _ in range(500):
            learner.observe("AAA", rng.normal(0, 3, 6), float(rng.normal(0.01, 0.05)), 0.0)
            assert learner.weight_norm() <= 0.5 + 1e-9

    def test_l2_decays_weights_absent_signal(self):
        """With no reward gradient, L2 alone must pull weights toward zero --
        that is what stops the slow monotonic norm drift of the online loop."""
        learner = ContinualRRL([f"f{i}" for i in range(6)], seed=2, l2=0.05)
        learner.agent.w[:] = 1.0
        n0 = learner.weight_norm()
        for _ in range(50):
            learner.observe("AAA", np.zeros(6), 0.0, 0.0)
        assert learner.weight_norm() < n0

    def test_saturation_metrics_are_logged(self, completed_run):
        _, engine, _ = completed_run
        learning = engine.store.read("learning")
        assert {"frac_saturated", "conviction_std"} <= set(learning.columns)
        assert learning["frac_saturated"].null_count() == 0
        assert (learning["frac_saturated"] <= 1.0).all()


class TestDeterminism:
    def test_two_fresh_runs_are_identical(self, tmp_path):
        results = []
        for name in ("x", "y"):
            cfg = make_cfg(tmp_path / name)
            seed_store(cfg)
            engine = PaperEngine(cfg)
            engine.run(capital=100_000, as_of=AS_OF, refresh=False, log=lambda m: None)
            results.append(
                (
                    engine.store.read("ledger").sort("ts"),
                    engine.store.read("trades").sort(["ts", "symbol"]),
                    ContinualRRL.load(cfg.artifacts_root / "models" / "rrl_latest.bin"),
                )
            )
        (la, ta, ma), (lb, tb, mb) = results
        assert la.equals(lb)
        assert ta.equals(tb)
        np.testing.assert_array_equal(ma.agent.w, mb.agent.w)


class TestDashboardAndUniverse:
    def test_dashboard_builds_self_contained(self, completed_run):
        _, engine, _ = completed_run
        path = build_paper_dashboard(engine.paper_root)
        html = path.read_text()
        assert "SIMULATED CAPITAL" in html
        for section in ("Daily decisions", "Trade history", "Learning progress", "Benchmark"):
            assert section in html
        assert "https://" not in html and "http://" not in html  # no CDN, no network

    def test_universe_resolution(self, tmp_path):
        assert len(resolve_universe("nasdaq100")) > 90
        assert len(resolve_universe("sp500")) > 450
        watchlist = tmp_path / "list.txt"
        watchlist.write_text("# mine\naapl\nMSFT\n\nmsft\n")
        assert resolve_universe(str(watchlist)) == ["AAPL", "MSFT"]
        with pytest.raises(ValueError):
            resolve_universe("nope")


# ---- day trading (intraday, flat by close) -----------------------------------

# The regular 30-minute session: 09:30..15:30 ET, thirteen bars. The flatten
# decision bar is 15:00 (its next-open fill is the 15:30 open); the final bar is
# 15:30. Naive-ET datetimes, exactly as the intraday loop stores them.
SESSION_TIMES = [time(9, 30)] + [time(10 + (i // 2), (i % 2) * 30) for i in range(12)]
FLATTEN_TIME = SESSION_TIMES[-2]  # 15:00
LAST_TIME = SESSION_TIMES[-1]  # 15:30


def _trading_days(start: date, n: int) -> list[date]:
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def make_intraday_cfg(tmp: Path) -> Config:
    """A 30-minute day-trading config with short feature windows so a few weeks
    of synthetic bars are enough to warm the features up."""
    cfg = Config()
    cfg.data.root = tmp / "data"
    cfg.data.source = "synthetic"
    cfg.data.universe = list(SYMS)
    cfg.artifacts_root = tmp / "artifacts"
    # Short windows: ~13 bars/day means default 252-bar windows would need a year
    # of intraday history. These keep the feature COLUMN set unchanged.
    cfg.features.return_horizons = [1, 5, 10]
    cfg.features.vol_windows = [5, 10]
    cfg.features.rsi_window = 10
    cfg.features.macd = (6, 13, 5)
    cfg.features.bollinger_window = 10
    cfg.features.zscore_window = 20
    cfg.features.fracdiff_threshold = 0.05
    cfg.features.warmup = 20
    cfg.paper.universe = "config"
    cfg.paper.interval = "30m"
    cfg.paper.day_trading = True
    cfg.paper.benchmark_symbols = []
    cfg.paper.data_start = "2026-01-01"
    cfg.paper.pretrain_years = 0.2
    cfg.paper.min_conviction = 0.01  # synthetic edge is weak; keep the book busy
    cfg.paper.exit_conviction = 0.005
    cfg.paper.kill_max_drawdown = None
    cfg.paper.kill_daily_loss = None
    cfg.paper.kill_rolling_20d_loss = None
    cfg.paper.kill_conviction_std = None
    return cfg


def seed_intraday_store(cfg: Config, days: list[date]) -> None:
    """Write deterministic 30-minute OHLCV bars for every SYM across ``days``."""
    store = BarStore(cfg.data.root)
    stamps = [
        datetime.combine(d, t) for d in days for t in SESSION_TIMES
    ]  # continuous intraday series, as Yahoo 30m bars arrive
    n = len(stamps)
    for k, sym in enumerate(SYMS):
        rng = np.random.default_rng(1000 + k)
        # A gentle geometric walk per 30m bar (~0.3% bar vol).
        steps = rng.normal(0.0, 0.003, n)
        close = 100.0 * np.exp(np.cumsum(steps))
        prev = np.concatenate([[100.0], close[:-1]])
        openp = prev * np.exp(rng.normal(0, 0.0005, n))
        wick = np.abs(rng.normal(0, 0.001, n))
        high = np.maximum(openp, close) * (1 + wick)
        low = np.minimum(openp, close) * (1 - wick)
        vol = rng.lognormal(12, 0.3, n)
        store.write(
            pl.DataFrame(
                {
                    "symbol": [sym] * n,
                    "ts": stamps,
                    "open": openp,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": close,
                    "volume": vol,
                }
            )
        )


@pytest.fixture(scope="module")
def intraday_run(tmp_path_factory):
    """One full 30-minute day-trading run shared by the read-only assertions."""
    tmp = tmp_path_factory.mktemp("intraday")
    cfg = make_intraday_cfg(tmp)
    days = _trading_days(date(2026, 3, 2), 45)
    seed_intraday_store(cfg, days)
    # Incept flat on a session's last bar; trade forward ~6 sessions.
    cfg.paper.start = datetime.combine(days[-7], LAST_TIME).isoformat()
    engine = PaperEngine(cfg)
    summary = engine.run(capital=100_000, as_of=days[-1], refresh=False, log=lambda m: None)
    return cfg, engine, summary


class TestDayTradingFlatByClose:
    """A day-trading bot never carries a position overnight: on the flatten bar
    the whole book is sold to zero, filling at the session's final bar open."""

    def test_engine_recognizes_the_flatten_bar(self, intraday_run):
        _, engine, _ = intraday_run
        assert engine.day_trading
        assert engine._flatten_time == FLATTEN_TIME

    def test_book_actually_trades_intraday(self, intraday_run):
        """Guards against a vacuous pass: the bot must open real positions."""
        _, engine, _ = intraday_run
        trades = engine.store.read("trades")
        buys = trades.filter(pl.col("action") == "buy")
        assert buys.height > 0
        ledger = engine.store.read("ledger")
        assert (ledger["n_positions"] > 0).any()  # held something intraday

    def test_flat_at_every_session_close(self, intraday_run):
        """The load-bearing invariant: at each session's final bar the book is
        already flat -- nothing to mark, nothing carried overnight."""
        _, engine, _ = intraday_run
        ledger = engine.store.read("ledger").with_columns(
            pl.col("ts").dt.time().alias("tod")
        )
        closes = ledger.filter(pl.col("tod") == LAST_TIME)
        assert closes.height >= 3  # several sessions were processed
        assert (closes["n_positions"] == 0).all()
        assert (closes["invested"].abs() < 1e-6).all()

    def test_no_entries_into_the_close(self, intraday_run):
        """No position is opened on the flatten bar or the final bar."""
        _, engine, _ = intraday_run
        decisions = engine.store.read("decisions").with_columns(
            pl.col("ts").dt.time().alias("tod")
        )
        late_entries = decisions.filter(
            (pl.col("action") == "buy") & (pl.col("tod") >= FLATTEN_TIME)
        )
        assert late_entries.is_empty()

    def test_flatten_orders_are_recorded_and_fill(self, intraday_run):
        _, engine, _ = intraday_run
        trades = engine.store.read("trades").with_columns(
            pl.col("ts").dt.time().alias("tod")
        )
        eod = trades.filter(pl.col("reason") == "eod_flat")
        assert eod.height > 0
        assert (eod["action"] == "sell").all()
        # eod_flat fills land at the session's final bar open.
        assert (eod["tod"] == LAST_TIME).all()

    def test_final_state_holds_nothing_overnight(self, intraday_run):
        _, engine, summary = intraday_run
        # The run ends on a session's last bar, so the persisted book is flat.
        state = PaperState.load(engine.store.state_path)
        assert state.positions == []
        assert summary.positions == []
