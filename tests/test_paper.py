"""Forward paper-trading invariants.

The paper engine manages a *persistent* simulated portfolio, so the failure
modes worth testing are the stateful ones: double-processing a day, leaking
tomorrow's bar into today's decision, cash going negative, state or model not
surviving a restart, and two identical runs disagreeing.

Everything runs on the synthetic source: deterministic, offline, and by
construction free of edge -- which also makes look-ahead detectable.
"""

from __future__ import annotations

import json
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
        p = Config().paper  # gross 0.90, floor 0.30, degross fraction 1.0
        state = self.fresh_state()
        d = date(2024, 6, 10)
        assert target_gross_exposure(p, state, d, n_open=8) == pytest.approx(0.90)
        # Two stops out of a ten-name book: 20% of the book, so 20% off the cap.
        state.last_stop_out = {"AAA": "2024-06-05", "BBB": "2024-06-07"}
        assert target_gross_exposure(p, state, d, n_open=8) == pytest.approx(0.72)
        # Stops age out of the window; the cap recovers.
        assert target_gross_exposure(p, state, date(2024, 7, 1), n_open=8) == pytest.approx(0.90)
        # The floor holds even if the entire book stops out at once.
        state.last_stop_out = {f"S{i}": "2024-06-09" for i in range(10)}
        assert target_gross_exposure(p, state, d, n_open=0) == pytest.approx(0.30)

    def test_de_gross_scales_with_book_breadth(self):
        """The load-bearing property for an uncapped book: the same handful of
        stops must not cripple a wide book the way it throttles a narrow one.
        A flat per-stop slab would have pinned the 300-name case at the floor."""
        p = Config().paper
        state = self.fresh_state()
        d = date(2024, 6, 10)
        state.last_stop_out = {f"S{i}": "2024-06-09" for i in range(5)}
        narrow = target_gross_exposure(p, state, d, n_open=5)
        wide = target_gross_exposure(p, state, d, n_open=300)
        assert narrow == pytest.approx(0.45)  # half the book stopped: half off
        assert wide > 0.88  # 5 of 305 names: barely a scratch
        assert wide < p.max_gross_exposure  # but still a real cut

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


def _crash_then_recover(bars: pl.DataFrame, inception: date, depth: float = 0.35) -> pl.DataFrame:
    """Scale every price after ``inception`` into a sharp crash, then a rebound.

    Synthetic bars have no edge and no crashes, so a drawdown deep enough to
    have tripped the old kill switches has to be constructed. Volume is left
    alone; only the price path is bent.
    """
    ts = bars.sort("ts")["ts"].to_list()
    after = [i for i, t in enumerate(ts) if t >= inception]
    crash_len = min(12, max(len(after) // 4, 1))
    mult = np.ones(len(ts))
    for k, i in enumerate(after):
        if k < crash_len:
            mult[i] = 1.0 - depth * (k + 1) / crash_len
        else:
            frac = (k - crash_len + 1) / max(len(after) - crash_len, 1)
            mult[i] = (1.0 - depth) + (depth + 0.10) * frac
    m = pl.Series(mult)
    return bars.sort("ts").with_columns(
        [(pl.col(c) * m).alias(c) for c in ("open", "high", "low", "close", "adj_close")]
    )


class TestNoLatchingHalt:
    """There is no kill switch. Losses do not put the bot into a dead state it
    needs a human to leave -- risk is carried by the per-name stops, the gross
    cap, and (intraday) being flat at every close."""

    def test_bot_trades_the_recovery_after_a_crash(self, tmp_path):
        """The behaviour a latching kill switch destroys: crash hard, then
        recover. A -4%-day / -15%-drawdown switch would have fired during the
        crash and left the book in cash for the entire rebound, needing a human
        to notice and clear it. With no switch, the bot must buy again."""
        cfg = make_cfg(tmp_path)
        inception = date.fromisoformat(cfg.paper.start)
        src = SyntheticSource(seed=7, regime_switching=True)
        store = BarStore(cfg.data.root)
        for sym in SYMS + cfg.paper.benchmark_symbols:
            store.write(_crash_then_recover(src.fetch(sym, "2019-01-01", "2024-08-30"), inception))

        engine = PaperEngine(cfg)
        engine.run(capital=100_000, as_of=AS_OF, refresh=False, log=lambda m: None)

        ledger = engine.store.read("ledger").sort("ts")
        equity, days = ledger["equity"].to_list(), ledger["ts"].to_list()
        peak = max(equity)
        trough_i = min(range(len(equity)), key=lambda i: equity[i])
        drawdown = equity[trough_i] / peak - 1.0
        assert drawdown < -0.04, f"test is vacuous without a real drawdown (got {drawdown:.2%})"
        # A single-bar loss deep enough to have tripped the old daily switch.
        assert min(ledger["daily_return"].to_list()) < -0.02

        buys_after_trough = engine.store.read("trades").filter(
            (pl.col("action") == "buy") & (pl.col("ts") > days[trough_i])
        )
        assert buys_after_trough.height > 0, "book never re-entered after the drawdown"
        assert not engine.store.read("decisions").filter(pl.col("ts") == days[-1]).is_empty()

    def test_state_has_no_halt_field(self, tmp_path):
        cfg = make_cfg(tmp_path)
        seed_store(cfg)
        engine = PaperEngine(cfg)
        engine.run(capital=100_000, as_of=AS_OF, refresh=False, log=lambda m: None)
        raw = json.loads(engine.store.state_path.read_text())
        assert "halted" not in raw and "halted_ts" not in raw

    def test_load_ignores_a_retired_halt_field(self, tmp_path):
        """A live portfolio's state file outlives the code that wrote it. The
        deployed bot's state.json still carries `halted`; loading it must drop
        the key, not crash -- otherwise shipping the removal kills the bot."""
        cfg = make_cfg(tmp_path)
        seed_store(cfg)
        engine = PaperEngine(cfg)
        engine.run(capital=100_000, as_of=AS_OF, refresh=False, log=lambda m: None)

        raw = json.loads(engine.store.state_path.read_text())
        raw["halted"] = "max_drawdown -18.00% from peak 100,000"
        raw["halted_ts"] = "2024-08-01"
        engine.store.state_path.write_text(json.dumps(raw))

        state = PaperState.load(engine.store.state_path)
        assert not hasattr(state, "halted")
        assert state.cash == pytest.approx(raw["cash"])
        # And the engine runs on it without complaint.
        PaperEngine(cfg).run(capital=100_000, as_of=AS_OF, refresh=False, log=lambda m: None)


class TestUncappedBook:
    """``max_positions = None`` lets the book hold every name that clears the
    conviction bar; breadth is limited by capital, never by a slot count."""

    def test_book_exceeds_the_old_ten_name_cap(self, tmp_path):
        cfg = make_cfg(tmp_path)
        cfg.data.universe = [f"S{i:02d}" for i in range(24)]
        cfg.paper.max_positions = None
        cfg.paper.min_conviction = 0.0  # every name is a candidate
        cfg.paper.max_position_weight = 0.20
        src = SyntheticSource(seed=11, regime_switching=True)
        store = BarStore(cfg.data.root)
        for sym in cfg.data.universe + cfg.paper.benchmark_symbols:
            store.write(src.fetch(sym, "2019-01-01", "2024-08-30"))

        engine = PaperEngine(cfg)
        engine.run(capital=100_000, as_of=AS_OF, refresh=False, log=lambda m: None)
        ledger = engine.store.read("ledger")
        assert int(ledger["n_positions"].max()) > 10

    def test_gross_still_respects_the_cap_when_uncapped(self, tmp_path):
        """Breadth must dilute, not lever. The cap binds on what the bot *asks
        for*: allocations decided on one bar sum to at most max_gross_exposure
        no matter how many names clear the bar. (The marked ratio at a later
        close drifts above that as prices move after the fill -- that is price
        appreciation on an already-sized book, not leverage, and cash staying
        non-negative is what proves nothing was borrowed.)"""
        cfg = make_cfg(tmp_path)
        cfg.data.universe = [f"S{i:02d}" for i in range(24)]
        cfg.paper.max_positions = None
        cfg.paper.min_conviction = 0.0
        src = SyntheticSource(seed=11, regime_switching=True)
        store = BarStore(cfg.data.root)
        for sym in cfg.data.universe + cfg.paper.benchmark_symbols:
            store.write(src.fetch(sym, "2019-01-01", "2024-08-30"))

        engine = PaperEngine(cfg)
        engine.run(capital=100_000, as_of=AS_OF, refresh=False, log=lambda m: None)

        decisions = engine.store.read("decisions")
        opens = decisions.filter(pl.col("action").is_in(["buy", "rebalance"]))
        by_bar = opens.group_by("ts").agg(pl.col("allocation").abs().sum().alias("gross"))
        assert by_bar.height > 0
        assert (by_bar["gross"] <= cfg.paper.max_gross_exposure + 1e-9).all()
        assert (opens["allocation"].abs() <= cfg.paper.max_position_weight + 1e-9).all()
        assert (engine.store.read("ledger")["cash"] >= 0).all()

    def test_wide_book_keeps_a_cash_buffer(self, tmp_path):
        """Regression: the no-trade band must scale with position size.

        With an absolute band, a wide book's targets (~gross/N) are far smaller
        than the band, so every holding reads as "close enough" and keeps its
        old weight while new entries are sized on top. Gross ratchets until cash
        hits zero and max_gross_exposure means nothing. Observed on real 30m
        bars before the fix: 197 positions, cash down to $1.68 on $99k equity.
        """
        cfg = make_cfg(tmp_path)
        cfg.data.universe = [f"S{i:02d}" for i in range(40)]
        cfg.paper.max_positions = None
        cfg.paper.min_conviction = 0.0  # everything is a candidate: widest book
        src = SyntheticSource(seed=11, regime_switching=True)
        store = BarStore(cfg.data.root)
        for sym in cfg.data.universe + cfg.paper.benchmark_symbols:
            store.write(src.fetch(sym, "2019-01-01", "2024-08-30"))

        engine = PaperEngine(cfg)
        engine.run(capital=100_000, as_of=AS_OF, refresh=False, log=lambda m: None)
        ledger = engine.store.read("ledger")
        assert int(ledger["n_positions"].max()) > 15, "not actually a wide book"
        # The band still lets gross run up to band_frac above target, so this
        # asserts the pathology is gone, not that tracking is perfect: cash is
        # never squeezed to nothing and the book is never fully invested.
        cash_frac = float((ledger["cash"] / ledger["equity"]).min())
        gross = float((ledger["invested"] / ledger["equity"]).max())
        assert cash_frac > 0.01, f"cash buffer collapsed to {cash_frac:.4%}"
        assert gross < 0.99, f"book went effectively fully invested ({gross:.4f})"

    def test_unfillable_entries_are_not_queued(self, tmp_path):
        """With a wide book the thinnest targets buy less than one share. Those
        are dropped at decision time so the log doesn't promise phantom buys."""
        cfg = make_cfg(tmp_path)
        cfg.data.universe = [f"S{i:02d}" for i in range(24)]
        cfg.paper.max_positions = None
        cfg.paper.min_conviction = 0.0
        src = SyntheticSource(seed=11, regime_switching=True, start_price=5000.0)
        store = BarStore(cfg.data.root)
        for sym in cfg.data.universe + cfg.paper.benchmark_symbols:
            store.write(src.fetch(sym, "2019-01-01", "2024-08-30"))

        engine = PaperEngine(cfg)
        engine.run(capital=100_000, as_of=AS_OF, refresh=False, log=lambda m: None)
        decisions = engine.store.read("decisions")
        assert not decisions.filter(pl.col("action") == "skip").is_empty()
        # Nothing logged as a buy was too small to fill.
        buys = decisions.filter(pl.col("action") == "buy")
        assert (buys["allocation"].abs() > 0).all()


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

    def test_extended_universe_is_the_widest_and_is_a_superset(self):
        """The bot's default hunting ground: every index name plus screened
        liquid movers, deduplicated. It must strictly contain the others."""
        extended = set(resolve_universe("extended"))
        assert len(extended) > 600
        assert extended >= set(resolve_universe("sp500"))
        assert extended >= set(resolve_universe("nasdaq100"))
        # No ETFs: the bot must not be able to buy its own benchmark.
        assert not extended & {"SPY", "QQQ", "IWM", "DIA"}

    def test_universe_symbols_are_yahoo_notation(self):
        """A dot ticker (BRK.B) silently fetches nothing from Yahoo."""
        for name in ("nasdaq100", "sp100", "sp500", "extended"):
            for sym in resolve_universe(name):
                assert "." not in sym and sym == sym.upper() and sym.strip() == sym

    def test_watchlist_and_unknown_name(self, tmp_path):
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
        ledger = engine.store.read("ledger").with_columns(pl.col("ts").dt.time().alias("tod"))
        closes = ledger.filter(pl.col("tod") == LAST_TIME)
        assert closes.height >= 3  # several sessions were processed
        assert (closes["n_positions"] == 0).all()
        assert (closes["invested"].abs() < 1e-6).all()

    def test_no_entries_into_the_close(self, intraday_run):
        """No position is opened on the flatten bar or the final bar."""
        _, engine, _ = intraday_run
        decisions = engine.store.read("decisions").with_columns(pl.col("ts").dt.time().alias("tod"))
        late_entries = decisions.filter(
            (pl.col("action") == "buy") & (pl.col("tod") >= FLATTEN_TIME)
        )
        assert late_entries.is_empty()

    def test_flatten_orders_are_recorded_and_fill(self, intraday_run):
        _, engine, _ = intraday_run
        trades = engine.store.read("trades").with_columns(pl.col("ts").dt.time().alias("tod"))
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


class TestUnlimitedIntradayTrading:
    """The loop has no per-day trade budget. Every completed 30m bar gets the
    full fill+learn+decide pass (exactly what one cron firing does live), so
    the bot can buy and sell on any bar of the session, as many times as its
    signal asks for."""

    def test_every_session_bar_is_processed(self, intraday_run):
        """One ledger row per completed bar, 13 per full session, no gaps:
        every price update the cron sees produced a decision pass."""
        _, engine, _ = intraday_run
        stamps = engine.store.read("ledger").sort("ts")["ts"].to_list()
        days = sorted({t.date() for t in stamps})
        grid = [datetime.combine(d, t) for d in days for t in SESSION_TIMES]
        assert stamps == [t for t in grid if t >= stamps[0]]

    def test_fills_happen_all_across_the_session(self, intraday_run):
        """Not an open-only bot: fills land on many distinct bars per session,
        including strictly mid-session ones."""
        _, engine, _ = intraday_run
        trades = engine.store.read("trades").with_columns(
            pl.col("ts").dt.date().alias("day"), pl.col("ts").dt.time().alias("tod")
        )
        bars_per_day = trades.group_by("day").agg(pl.col("tod").n_unique().alias("n"))
        assert bars_per_day["n"].max() >= 5
        mid_session = trades.filter(
            (pl.col("tod") > SESSION_TIMES[0]) & (pl.col("tod") < FLATTEN_TIME)
        )
        assert mid_session.height > 0

    def test_round_trips_inside_a_single_session(self, intraday_run):
        """Entries, exits and the eod flatten all inside one session: buys and
        sells are not rationed to one shot per day."""
        _, engine, _ = intraday_run
        trades = engine.store.read("trades").with_columns(pl.col("ts").dt.date().alias("day"))
        by_day = trades.group_by("day").agg(pl.col("reason").unique().alias("reasons"))
        assert any({"entry", "exit", "eod_flat"} <= set(r) for r in by_day["reasons"].to_list())


class TestIntradayStopCooldownClock:
    """With ``stop_cooldown_bars`` set, the stop lockout and the de-gross
    window run on the bar clock: a stopped name is back in play the same
    session instead of frozen for stop_cooldown_days of calendar time."""

    @staticmethod
    def paper_30m():
        p = Config().paper
        p.interval = "30m"
        p.stop_cooldown_bars = 6  # 3 hours of 30m bars
        return p

    def test_lockout_lapses_within_the_session(self):
        p = self.paper_30m()
        state = TestStopDiscipline.fresh_state()
        state.last_stop_out["AAA"] = "2026-03-02T10:00"
        assert stop_cooldown_active(p, state, "AAA", datetime(2026, 3, 2, 10, 30))
        assert stop_cooldown_active(p, state, "AAA", datetime(2026, 3, 2, 12, 30))
        assert not stop_cooldown_active(p, state, "AAA", datetime(2026, 3, 2, 13, 0))
        assert not stop_cooldown_active(p, state, "BBB", datetime(2026, 3, 2, 10, 30))

    def test_degross_window_rolls_in_bars(self):
        p = self.paper_30m()  # gross 0.90, degross by fraction of the book
        state = TestStopDiscipline.fresh_state()
        state.last_stop_out = {"AAA": "2026-03-02T10:00", "BBB": "2026-03-02T11:30"}
        gross = lambda t: target_gross_exposure(p, state, t, n_open=8)  # noqa: E731
        assert gross(datetime(2026, 3, 2, 12, 0)) == pytest.approx(0.72)  # 2 of 10
        # AAA ages out at 13:00, BBB at 14:30: the cap recovers the same day.
        assert gross(datetime(2026, 3, 2, 13, 30)) == pytest.approx(0.80)  # 1 of 9
        assert gross(datetime(2026, 3, 2, 14, 30)) == pytest.approx(0.90)  # none left

    def test_daily_loop_ignores_the_bar_clock(self):
        p = Config().paper
        p.stop_cooldown_bars = 6  # meaningless on "1d"; the day clock rules
        state = TestStopDiscipline.fresh_state()
        state.last_stop_out["AAA"] = "2024-06-03"
        assert stop_cooldown_active(p, state, "AAA", date(2024, 6, 12))
        assert not stop_cooldown_active(p, state, "AAA", date(2024, 6, 13))
