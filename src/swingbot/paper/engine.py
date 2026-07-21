"""The autonomous daily paper-investing loop.

Every run does the same thing, no matter how long it has been since the last
one: find every completed trading day that has not been processed yet and
process them **in order**, one day at a time. For each day ``d``:

1. **Fill** pending orders (decided on the previous processed bar) at day
   ``d``'s open, through the existing ``ExecutionModel`` -- spread, slippage,
   sqrt impact, commissions, fees, borrow. Sells first, buys by conviction,
   and buys are capped so simulated cash can never go negative.
2. **Mark to market** at day ``d``'s close and append the ledger row,
   including benchmark equity (buy-and-hold SPY / QQQ / equal-weight universe).
3. **Learn**: every symbol's realized bar return becomes one online RRL
   update -- the continual-learning step. This happens exactly once per
   (symbol, day) because days are processed exactly once.
4. **Decide**: score every symbol in the universe on day ``d``'s close,
   rank by conviction, allocate capital, and queue orders for the *next*
   open. Deciding nothing is always allowed.

Idempotency is structural: ``state.last_processed`` is the watermark, so
re-running on the same day is a no-op -- no duplicate trades, no duplicate
training. Look-ahead is structurally impossible: a decision at ``d`` can only
ever be filled at a bar strictly after ``d``, and features are trailing-only.

All capital is SIMULATED. Nothing here can place a real order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from swingbot.config import Config, PaperConfig
from swingbot.data.schema import DataQualityError
from swingbot.data.sources import get_source
from swingbot.data.store import BarStore
from swingbot.execution import ExecutionModel, MarketContext
from swingbot.features.technical import build_dataset, feature_columns
from swingbot.paper.learner import ContinualRRL
from swingbot.paper.state import PaperState, PaperStore, PendingOrder
from swingbot.paper.universe import resolve_universe

_ET = ZoneInfo("America/New_York")
_MARKET_CLOSE_HOUR = 16  # 16:15 ET: bar for `today` is considered complete
_LAST_HOURLY_BAR = time(15, 30)  # the half-hour 15:30 bar; completes at 16:00

# One bar timestamp: a date on the daily loop, a naive-ET datetime intraday.
BarTs = date | datetime


def _parse_like(iso: str, like: BarTs) -> BarTs:
    """Parse a stored ISO timestamp with the same resolution as ``like``."""
    return datetime.fromisoformat(iso) if isinstance(like, datetime) else date.fromisoformat(iso)


def stop_cooldown_active(paper: PaperConfig, state: PaperState, symbol: str, d: BarTs) -> bool:
    """True while a stopped-out symbol is still barred from re-entry."""
    last = state.last_stop_out.get(symbol)
    if last is None:
        return False
    return (d - _parse_like(last, d)).days < paper.stop_cooldown_days


def target_gross_exposure(paper: PaperConfig, state: PaperState, d: BarTs) -> float:
    """Gross-exposure cap after recent stop-outs.

    Each stop inside the cooldown window de-grosses the book, so the cash a
    stop frees stays cash instead of rotating into the next correlated name.
    """
    recent = sum(
        1
        for iso in state.last_stop_out.values()
        if 0 <= (d - _parse_like(iso, d)).days < paper.stop_cooldown_days
    )
    return max(
        paper.min_gross_exposure,
        paper.max_gross_exposure - paper.stop_degross_per_stop * recent,
    )


@dataclass
class _SymbolData:
    """One symbol's feature-complete history, materialised for fast lookup."""

    ts: list[date]
    idx: dict[date, int]
    x: np.ndarray  # (n, n_features)
    open: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    daily_vol: np.ndarray


@dataclass
class DayReport:
    """What one processed trading day produced -- the CLI's raw material."""

    ts: date
    fills: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    equity: float = 0.0
    cash: float = 0.0
    daily_return: float = 0.0
    n_positions: int = 0
    learn_updates: int = 0
    learn_mean_reward: float = 0.0


@dataclass
class RunSummary:
    """Everything the caller needs to report a run."""

    universe: str
    n_symbols: int
    days: list[DayReport] = field(default_factory=list)
    last_processed: date | None = None
    equity: float = 0.0
    cash: float = 0.0
    starting_capital: float = 0.0
    total_return: float = 0.0
    benchmarks: dict[str, float] = field(default_factory=dict)  # name -> total return
    positions: list[dict] = field(default_factory=list)
    learning: dict = field(default_factory=dict)
    dashboard_path: Path | None = None
    halted: str | None = None  # fired kill switch, verbatim from state

    @property
    def today(self) -> DayReport | None:
        return self.days[-1] if self.days else None


class PaperEngine:
    """Owns one persistent simulated portfolio over a stock universe."""

    def __init__(self, cfg: Config, *, universe: str | None = None) -> None:
        self.cfg = cfg
        self.paper = cfg.paper
        self.universe_name = universe or self.paper.universe
        self.symbols = resolve_universe(self.universe_name, cfg)
        self.benchmarks = [s.upper() for s in self.paper.benchmark_symbols]
        self.hourly = self.paper.interval == "60m"
        self.bars_per_day = 7 if self.hourly else 1  # regular-session hourly bars

        self.bar_store = BarStore(cfg.data.root)
        self.paper_root = cfg.artifacts_root / "paper"
        self.store = PaperStore(self.paper_root)
        self.models_root = cfg.artifacts_root / "models"
        self.execution = ExecutionModel(cfg.env.costs)
        self.feature_cols = feature_columns(cfg.features)
        # Round-trip friction the learner is charged per unit of position change.
        self._learn_cost = 2.0 * (cfg.env.costs.half_spread_bps + cfg.env.costs.slippage_bps) * 1e-4

    def _iso(self, iso: str) -> BarTs:
        """Parse a stored ISO timestamp at this loop's bar resolution."""
        return datetime.fromisoformat(iso) if self.hourly else date.fromisoformat(iso)

    # ---- data --------------------------------------------------------------

    def refresh_data(self, *, log=print) -> None:
        """Pull missing daily bars for the universe + benchmarks into the store.

        Incremental: cached symbols refetch a 30-day overlap window (so recent
        splits/dividends re-adjust cleanly); new symbols fetch full history.
        """
        source_name = self.cfg.data.source
        if source_name in ("yahoo", "yfinance"):
            source_name = "yahoo_bulk"  # universe-scale refresh in a few requests
        kwargs = {"interval": self.paper.interval} if source_name.startswith("yahoo") else {}
        source = get_source(source_name, **kwargs)

        wanted = sorted(set(self.symbols) | set(self.benchmarks))
        cached, new = [], []
        for sym in wanted:
            (cached if sym in self.bar_store else new).append(sym)

        batches: list[tuple[list[str], str]] = []
        if new:
            batches.append((new, self.paper.data_start))
        if cached:
            coverage = self.bar_store.coverage().filter(pl.col("symbol").is_in(cached))
            # Refresh from the *stalest* cached symbol, minus a 30-day overlap
            # so recent splits/dividends re-adjust the boundary cleanly.
            stalest = coverage["end"].min() if not coverage.is_empty() else None
            if isinstance(stalest, datetime):
                stalest = stalest.date()
            start = max(
                date.fromisoformat(self.paper.data_start),
                (stalest or date.min) - timedelta(days=30),
            )
            batches.append((cached, start.isoformat()))

        for syms, start in batches:
            log(f"[data] fetching {len(syms)} symbol(s) from {start}")
            try:
                df = source.fetch_many(syms, start, None, on_error="warn")
            except DataQualityError as exc:
                log(f"[data] fetch failed: {exc}")
                continue
            # Write per symbol: one bad ticker must not sink the whole batch.
            for (sym,), group in df.group_by(["symbol"], maintain_order=True):
                try:
                    self.bar_store.write(group)
                except DataQualityError as exc:
                    log(f"[data] skipping {sym}: {str(exc)[:80]}")

    def _read_completed(self, symbols: list[str], cutoff: BarTs) -> pl.DataFrame:
        """Bars whose *completion time* is at or before the cutoff.

        Daily bars complete at their own date (the caller's cutoff rule already
        excluded today's forming bar). Hourly bars start on the stamp and
        complete an hour later — 30 minutes for the half-length 15:30 bar — so
        a partial bar the vendor returns mid-hour can never enter the loop.
        """
        if not self.hourly:
            return self.bar_store.read(symbols, start=self.paper.data_start, end=cutoff)
        bars = self.bar_store.read(symbols, start=self.paper.data_start)
        if bars.is_empty():
            return bars
        completes = (
            pl.when(pl.col("ts").dt.time() == _LAST_HOURLY_BAR)
            .then(pl.col("ts") + pl.duration(minutes=30))
            .otherwise(pl.col("ts") + pl.duration(minutes=60))
        )
        return bars.filter(completes <= pl.lit(cutoff))

    def _load_symbol_data(self, as_of: BarTs) -> dict[str, _SymbolData]:
        """Feature-complete per-symbol arrays, truncated to completed bars."""
        bars = self._read_completed(self.symbols, as_of)
        if bars.is_empty():
            return {}
        features = build_dataset(bars, self.cfg.features)
        vol_col = f"vol_{self.cfg.features.vol_windows[0]}d"

        out: dict[str, _SymbolData] = {}
        for (sym,), grp in features.group_by(["symbol"], maintain_order=True):
            ts = grp["ts"].to_list()
            out[str(sym)] = _SymbolData(
                ts=ts,
                idx={t: i for i, t in enumerate(ts)},
                x=grp.select(self.feature_cols).to_numpy().astype(np.float64),
                open=grp["open"].to_numpy().astype(np.float64),
                close=grp["close"].to_numpy().astype(np.float64),
                volume=grp["volume"].to_numpy().astype(np.float64),
                daily_vol=np.nan_to_num(
                    grp[vol_col].to_numpy().astype(np.float64) / math.sqrt(252.0), nan=0.02
                ),
            )
        return out

    def _benchmark_closes(self, as_of: BarTs) -> dict[str, dict[BarTs, float]]:
        out: dict[str, dict[BarTs, float]] = {}
        for sym in self.benchmarks:
            bars = self._read_completed([sym], as_of)
            if not bars.is_empty():
                out[sym] = dict(zip(bars["ts"].to_list(), bars["close"].to_list(), strict=True))
        return out

    @staticmethod
    def latest_completed(as_of: date | None) -> date:
        """The most recent date whose daily bar can be considered complete.

        Decisions may only use completed bars. Before the close (16:15 ET),
        today's bar is still forming, so the latest completed bar is
        yesterday's -- even if the vendor already returns a partial bar.
        """
        now = datetime.now(_ET)
        if as_of is not None and as_of < now.date():
            return as_of  # explicit historical cutoff (tests, replays)
        today = now.date()
        market_closed = (now.hour, now.minute) >= (_MARKET_CLOSE_HOUR, 15)
        return today if market_closed else today - timedelta(days=1)

    def _cutoff(self, as_of: date | None) -> BarTs:
        """The completion-time cutoff for this run, per the loop's interval."""
        if not self.hourly:
            return self.latest_completed(as_of)
        if as_of is not None and as_of < datetime.now(_ET).date():
            return datetime.combine(as_of, time(23, 59))  # replay a full day
        return datetime.now(_ET).replace(tzinfo=None)

    # ---- the daily loop -----------------------------------------------------

    def run(
        self,
        *,
        capital: float = 100_000.0,
        as_of: date | None = None,
        refresh: bool = True,
        clear_halt: bool = False,
        log=print,
    ) -> RunSummary:
        """Process every unprocessed completed trading day. Idempotent."""
        if refresh:
            self.refresh_data(log=log)

        cutoff = self._cutoff(as_of)
        data = self._load_symbol_data(cutoff)
        if not data:
            raise DataQualityError("no feature-complete bars for the universe; fetch data first")
        bench_closes = self._benchmark_closes(cutoff)

        calendar = sorted({t for sd in data.values() for t in sd.ts})
        state, learner = self._load_or_init(capital, calendar, data, bench_closes, log=log)
        if clear_halt and state.halted:
            log(f"[risk] halt cleared by operator (was: {state.halted} on {state.halted_ts})")
            state.halted = None
            state.halted_ts = None
            state.save(self.store.state_path)

        last = self._iso(state.last_processed) if state.last_processed else None
        todo = [d for d in calendar if (last is None or d > last)]
        if state.inception:
            todo = [d for d in todo if d >= self._iso(state.inception)]

        summary = RunSummary(
            universe=self.universe_name,
            n_symbols=len(self.symbols),
            starting_capital=state.starting_capital,
        )

        pf = state.to_portfolio()
        entry_ts = {p.symbol: p.entry_ts for p in state.positions if p.entry_ts}
        # Prime carry-forward prices strictly *before* the first day we will
        # process -- never from bars the loop has not reached yet.
        boundary = last or (self._iso(state.inception) - timedelta(days=1))
        last_close: dict[str, float] = {}
        for sym, sd in data.items():
            past = [i for i, t in enumerate(sd.ts) if t <= boundary]
            if past:
                last_close[sym] = float(sd.close[past[-1]])

        prev_equity = self._ledger_last_equity(state)
        equity_hist = self._ledger_equity_history()
        if todo:
            log(f"[paper] processing {len(todo)} trading day(s): {todo[0]} .. {todo[-1]}")
        ledger_rows, trade_rows, decision_rows, learning_rows = [], [], [], []
        prev_day: date | None = last

        for d in todo:
            report, prev_equity = self._process_day(
                d,
                prev_day,
                state,
                pf,
                learner,
                data,
                bench_closes,
                last_close,
                entry_ts,
                prev_equity,
                equity_hist,
                ledger_rows,
                trade_rows,
                decision_rows,
                learning_rows,
            )
            summary.days.append(report)
            prev_day = d

        if state.halted:
            log(
                f"[risk] HALTED since {state.halted_ts} ({state.halted}) -- "
                f"book is being flattened; no entries until 'invest --clear-halt'"
            )

        if learning_rows:
            lr = learning_rows[-1]
            if lr["frac_saturated"] > 0.5 or lr["conviction_std"] < 0.05:
                log(
                    f"[learn] WARNING: policy saturated "
                    f"(frac |f|>0.99 = {lr['frac_saturated']:.2f}, "
                    f"conviction std = {lr['conviction_std']:.3f}) -- "
                    f"conviction ranking is degenerate; sizing is the tiebreak"
                )

        # ---- persist everything exactly once ----
        if todo:
            state.capture_portfolio(pf, entry_ts)
            state.last_processed = todo[-1].isoformat()
            ckpt_ts = todo[-1].date() if isinstance(todo[-1], datetime) else todo[-1]
            self.store.append("ledger", pl.DataFrame(ledger_rows))
            if trade_rows:
                self.store.append("trades", pl.DataFrame(trade_rows))
            if decision_rows:
                self.store.append(
                    "decisions",
                    pl.DataFrame(decision_rows).with_columns(pl.col("result").cast(pl.Float64)),
                )
            self.store.append("learning", pl.DataFrame(learning_rows))
            self._write_positions(pf, entry_ts, last_close)
            learner.checkpoint(self.models_root, ckpt_ts, max_keep=self.paper.max_checkpoints)
            state.save(self.store.state_path)
        else:
            log("[paper] no new completed trading day; nothing to do")

        self._fill_summary(summary, state, pf, learner, last_close, entry_ts)
        return summary

    # ---- one trading day -----------------------------------------------------

    def _process_day(
        self,
        d: date,
        prev_day: date | None,
        state: PaperState,
        pf,
        learner: ContinualRRL,
        data: dict[str, _SymbolData],
        bench_closes: dict[str, dict[date, float]],
        last_close: dict[str, float],
        entry_ts: dict[str, str],
        prev_equity: float,
        equity_hist: list[float],
        ledger_rows: list,
        trade_rows: list,
        decision_rows: list,
        learning_rows: list,
    ) -> tuple[DayReport, float]:
        report = DayReport(ts=d)

        # 1. fill pending orders at today's open
        day_turnover = self._fill_orders(d, state, pf, data, last_close, entry_ts, trade_rows)
        report.fills = [t for t in trade_rows if t["ts"] == d]

        # 2. borrow accrual on shorts, then mark to market at the close
        for sym, sd in data.items():
            i = sd.idx.get(d)
            if i is not None:
                last_close[sym] = float(sd.close[i])
        for sym in list(pf.positions):
            pos = pf.positions[sym]
            if pos.is_short and sym in last_close:
                cost = self.execution.borrow_cost(
                    pos.market_value(last_close[sym]), days=1.0 / self.bars_per_day
                )
                if cost > 0:
                    pf.charge(cost, symbol=sym)

        equity = pf.equity(last_close)
        daily_ret = equity / prev_equity - 1.0 if prev_equity > 0 else 0.0

        # 3. continual learning: every symbol's realized bar return is experience
        day_rewards, results_prev = [], {}
        day_z, day_grad = [], []
        for sym in sorted(data):
            sd = data[sym]
            i = sd.idx.get(d)
            if i is None or i == 0:
                continue
            ret = float(sd.close[i] / sd.close[i - 1] - 1.0)
            r = learner.observe(sym, sd.x[i - 1], ret, self._learn_cost)
            day_rewards.append(r)
            day_z.append(abs(learner.agent.last_z))
            day_grad.append(learner.agent.last_grad_norm)
            if prev_day is not None and sd.ts[i - 1] == prev_day:
                results_prev[sym] = ret
        if prev_day is not None and results_prev:
            # Yesterday's decisions may still be in this batch's memory buffer
            # (multi-day catch-up) or already on disk (normal daily cadence).
            in_memory = False
            for row in decision_rows:
                if row["ts"] == prev_day and row["symbol"] in results_prev:
                    row["result"] = results_prev[row["symbol"]]
                    in_memory = True
            if not in_memory:
                self.store.set_decision_results(prev_day, results_prev)

        # 4. decide: score the whole universe on today's close
        orders, decisions, sat = self._decide(d, state, pf, learner, data, last_close, equity)

        # 5. kill switches: P&L first, then model health. A fired switch
        # replaces today's decisions wholesale with "flatten everything" and
        # persists until an operator clears it.
        if state.halted is None:
            reason = self._kill_reason(daily_ret, equity, equity_hist, sat)
            if reason:
                state.halted = reason
                state.halted_ts = d.isoformat()
        if state.halted:
            orders = []
            decisions = []
            for sym in sorted(s for s in pf.positions if not pf.positions[s].is_flat):
                orders.append(
                    PendingOrder(
                        symbol=sym,
                        decided_ts=d.isoformat(),
                        target_weight=0.0,
                        conviction=0.0,
                        expected_reward=0.0,
                        reason="kill_switch",
                    )
                )
                decisions.append(
                    {
                        "ts": d,
                        "symbol": sym,
                        "action": "sell",
                        "conviction": 0.0,
                        "expected_reward": 0.0,
                        "allocation": 0.0,
                        "current_weight": (
                            pf.quantity(sym) * last_close.get(sym, 0.0) / equity
                            if equity > 0
                            else 0.0
                        ),
                        "result": None,
                    }
                )
            # While halted nothing carries over: a stale buy filling tomorrow
            # would defeat the entire point of the switch.
            state.pending_orders = orders
        else:
            # New decisions supersede same-symbol pending orders; orders for
            # symbols that printed no bar today (not re-scored) carry over.
            new_syms = {o.symbol for o in orders}
            state.pending_orders = orders + [
                o for o in state.pending_orders if o.symbol not in new_syms
            ]
        decision_rows.extend(decisions)
        equity_hist.append(equity)

        # ---- record the day ----
        # float(): market_value / unrealized_pnl are sum() over positions, so a
        # flat day yields int 0 and would type the column Int64; keep the money
        # columns Float64 so the ledger schema never drifts between days.
        invested = float(pf.market_value(last_close))
        ledger_rows.append(
            {
                "ts": d,
                "cash": float(pf.cash),
                "equity": float(equity),
                "invested": invested,
                "n_positions": sum(1 for p in pf.positions.values() if not p.is_flat),
                "daily_return": float(daily_ret),
                "realized_pnl": float(pf.realized_pnl),
                "unrealized_pnl": float(pf.unrealized_pnl(last_close)),
                "explicit_costs": pf.cumulative_costs,
                "slippage_costs": pf.cumulative_slippage,
                "turnover": day_turnover / equity if equity > 0 else 0.0,
                **self._benchmark_equity(state, d, bench_closes, data),
            }
        )
        learning_rows.append(
            {
                "ts": d,
                "n_updates": learner.n_updates,
                "day_updates": len(day_rewards),
                "day_mean_reward": float(np.mean(day_rewards)) if day_rewards else 0.0,
                "cum_reward": learner.cum_reward,
                "policy_loss": -float(np.mean(day_rewards)) if day_rewards else 0.0,
                "ew_sharpe": learner.sharpe,
                "weight_norm": learner.weight_norm(),
                "grad_norm": float(np.mean(day_grad)) if day_grad else 0.0,
                "z_abs_mean": float(np.mean(day_z)) if day_z else 0.0,
                # The saturation alarms. frac_saturated > 0.5 or
                # conviction_std < 0.05 means ranking has degenerated into the
                # sort's tiebreak; run() shouts when that happens.
                "frac_saturated": sat["frac_saturated"],
                "conviction_std": sat["conviction_std"],
            }
        )

        report.decisions = decisions
        report.equity = equity
        report.cash = pf.cash
        report.daily_return = daily_ret
        report.n_positions = sum(1 for p in pf.positions.values() if not p.is_flat)
        report.learn_updates = len(day_rewards)
        report.learn_mean_reward = float(np.mean(day_rewards)) if day_rewards else 0.0
        return report, equity

    # ---- kill switches ---------------------------------------------------------

    def _kill_reason(
        self,
        daily_ret: float,
        equity: float,
        equity_hist: list[float],
        sat: dict[str, float],
    ) -> str | None:
        """First kill switch that fires today, or None.

        ``equity_hist`` is strictly *prior* days' equity, so the drawdown
        compares today against the historical peak and the rolling window
        against the close 20 trading days back.
        """
        p = self.paper
        if p.kill_daily_loss is not None and daily_ret <= -p.kill_daily_loss:
            return f"daily_loss {daily_ret:.2%}"
        peak = max(equity_hist) if equity_hist else equity
        if p.kill_max_drawdown is not None and peak > 0:
            dd = equity / peak - 1.0
            if dd <= -p.kill_max_drawdown:
                return f"max_drawdown {dd:.2%} from peak {peak:,.0f}"
        if p.kill_rolling_20d_loss is not None and len(equity_hist) >= 20:
            roll = equity / equity_hist[-20] - 1.0
            if roll <= -p.kill_rolling_20d_loss:
                return f"rolling_20d_loss {roll:.2%}"
        if p.kill_conviction_std is not None and sat["conviction_std"] < p.kill_conviction_std:
            return (
                f"conviction_std {sat['conviction_std']:.4f} < {p.kill_conviction_std} "
                f"(model health: scores are degenerate)"
            )
        return None

    # ---- fills -----------------------------------------------------------------

    def _fill_orders(
        self,
        d: date,
        state: PaperState,
        pf,
        data: dict[str, _SymbolData],
        last_close: dict[str, float],
        entry_ts: dict[str, str],
        trade_rows: list,
    ) -> float:
        """Execute pending orders at day ``d``'s open. Returns traded notional."""
        keep: list[PendingOrder] = []
        executable: list[PendingOrder] = []
        for order in state.pending_orders:
            sd = data.get(order.symbol)
            if sd is not None and d in sd.idx:
                executable.append(order)
            elif (d - _parse_like(order.decided_ts, d)).days <= self.paper.cancel_after_days:
                keep.append(order)  # no bar today; wait, then expire
        state.pending_orders = keep

        open_prices = dict(last_close)
        for order in executable:
            sd = data[order.symbol]
            open_prices[order.symbol] = float(sd.open[sd.idx[d]])

        def current_weight(order: PendingOrder, equity: float) -> float:
            price = open_prices.get(order.symbol, 0.0)
            return pf.quantity(order.symbol) * price / equity if equity > 0 else 0.0

        # Sells first (they free cash), then buys in conviction order.
        equity0 = pf.equity(open_prices)
        sells = [o for o in executable if o.target_weight < current_weight(o, equity0)]
        buys = [o for o in executable if o not in sells]
        sells.sort(key=lambda o: o.symbol)
        buys.sort(key=lambda o: (-abs(o.conviction), o.symbol))

        notional = 0.0
        for order in sells + buys:
            sd = data[order.symbol]
            i = sd.idx[d]
            ref = float(sd.open[i])
            equity = pf.equity(open_prices)
            if equity <= 0:
                break
            desired = math.floor(abs(order.target_weight) * equity / ref) * (
                1 if order.target_weight >= 0 else -1
            )
            if abs(order.target_weight) < 1e-9:
                desired = 0
            delta = desired - pf.quantity(order.symbol)
            if abs(delta) < 1.0:
                continue

            ctx = MarketContext(
                ts=d,
                symbol=order.symbol,
                reference_price=ref,
                volatility=float(sd.daily_vol[i]),
                volume=float(sd.volume[i]),
            )
            fill = self._build_affordable_fill(delta, ctx, pf)
            if fill is None:
                continue

            was_flat = pf.position(order.symbol).is_flat
            realized = pf.execute(fill)
            notional += fill.notional
            if pf.position(order.symbol).is_flat:
                entry_ts.pop(order.symbol, None)
                if order.reason == "stop_loss":
                    state.last_stop_out[order.symbol] = d.isoformat()
            elif was_flat:
                entry_ts[order.symbol] = d.isoformat()
            state.n_fills += 1
            trade_rows.append(
                {
                    "ts": d,
                    "symbol": order.symbol,
                    "action": "buy" if fill.quantity > 0 else "sell",
                    "quantity": fill.quantity,
                    "fill_price": fill.price,
                    "reference_price": ref,
                    "notional": fill.notional,
                    "commission": fill.commission,
                    "fees": fill.fees,
                    "slippage": fill.slippage_cost,
                    "realized_pnl": realized,
                    "reason": order.reason,
                    "conviction": order.conviction,
                    "target_weight": order.target_weight,
                    "decided_ts": _parse_like(order.decided_ts, d),
                }
            )
        return notional

    def _build_affordable_fill(self, delta: float, ctx: MarketContext, pf):
        """Build a fill, shrinking a buy until simulated cash covers it fully."""
        fill = self.execution.build_fill(delta, ctx)
        if fill is None or fill.quantity < 0:
            return fill  # sells only raise cash
        for _ in range(8):
            total = fill.quantity * fill.price + fill.total_cost
            if total <= pf.cash:
                return fill
            affordable = math.floor(pf.cash / (fill.price * 1.001))
            new_qty = min(affordable, int(fill.quantity) - 1)
            if new_qty < 1:
                return None
            fill = self.execution.build_fill(float(new_qty), ctx)
            if fill is None:
                return None
        return None

    # ---- decisions -----------------------------------------------------------

    def _decide(
        self,
        d: date,
        state: PaperState,
        pf,
        learner: ContinualRRL,
        data: dict[str, _SymbolData],
        last_close: dict[str, float],
        equity: float,
    ) -> tuple[list[PendingOrder], list[dict], dict[str, float]]:
        """Score the universe, rank, allocate.

        Returns (orders, decision log, saturation metrics). The metrics are
        computed over the *whole* scored universe, before any conviction
        threshold, because the failure they detect -- every conviction pinned
        near +/-1 -- is only visible in the full cross-section.
        """
        p = self.paper
        raw_scores: list[float] = []
        scores: dict[str, tuple[float, float]] = {}  # sym -> (f, daily_vol)
        for sym in sorted(data):
            sd = data[sym]
            i = sd.idx.get(d)
            if i is None:
                continue
            f = learner.score(sym, sd.x[i])
            raw_scores.append(f)
            if not p.allow_short:
                f = max(f, 0.0)
            scores[sym] = (f, float(sd.daily_vol[i]))
        fs = np.abs(np.asarray(raw_scores)) if raw_scores else np.zeros(0)
        sat = {
            "frac_saturated": float((fs > 0.99).mean()) if fs.size else 0.0,
            "conviction_std": float(fs.std()) if fs.size else 0.0,
        }

        held = {s for s in pf.positions if not pf.positions[s].is_flat}
        orders: list[PendingOrder] = []
        decisions: list[dict] = []

        def weight_of(sym: str) -> float:
            price = last_close.get(sym, 0.0)
            return pf.quantity(sym) * price / equity if equity > 0 else 0.0

        def log_decision(sym: str, action: str, f: float, dvol: float, target: float) -> None:
            decisions.append(
                {
                    "ts": d,
                    "symbol": sym,
                    "action": action,
                    "conviction": f,
                    "expected_reward": f * dvol,
                    "allocation": target,
                    "current_weight": weight_of(sym),
                    "result": None,
                }
            )

        # ---- exits: conviction decay and stop-loss ----
        keep_holds: list[str] = []
        for sym in sorted(held):
            f, dvol = scores.get(sym, (0.0, 0.02))
            pos = pf.positions[sym]
            price = last_close.get(sym)
            # Vol-scaled stop: the barrier sits at stop_loss_sigma standard
            # deviations of THIS name's horizon vol below cost basis, so every
            # position carries the same noise-touch probability. The fixed
            # percentage is only a fallback.
            if p.stop_loss_sigma is not None:
                stop_frac = p.stop_loss_sigma * dvol * math.sqrt(p.stop_horizon_days)
            else:
                stop_frac = p.stop_loss_pct
            stopped = (
                stop_frac is not None
                and price is not None
                and pos.is_long
                and price <= pos.avg_price * (1.0 - stop_frac)
            )
            if stopped or abs(f) < p.exit_conviction:
                reason = "stop_loss" if stopped else "exit"
                orders.append(
                    PendingOrder(
                        symbol=sym,
                        decided_ts=d.isoformat(),
                        target_weight=0.0,
                        conviction=f,
                        expected_reward=f * dvol,
                        reason=reason,
                    )
                )
                log_decision(sym, "sell", f, dvol, 0.0)
            else:
                keep_holds.append(sym)

        # ---- entries: rank the rest of the universe by conviction ----
        candidates = sorted(
            (
                (sym, f, dvol)
                for sym, (f, dvol) in scores.items()
                if sym not in held
                and abs(f) >= p.min_conviction
                and not stop_cooldown_active(p, state, sym, d)
            ),
            key=lambda t: (-abs(t[1]), t[0]),
        )
        slots = max(p.max_positions - len(keep_holds), 0)
        entries = candidates[:slots]

        # ---- size everything: weight proportional to conviction, capped ----
        targets: dict[str, tuple[float, float, float, str]] = {}
        for sym in keep_holds:
            f, dvol = scores[sym]
            targets[sym] = (f * p.max_position_weight, f, dvol, "rebalance")
        for sym, f, dvol in entries:
            targets[sym] = (f * p.max_position_weight, f, dvol, "entry")

        gross = sum(abs(w) for w, *_ in targets.values())
        gross_cap = target_gross_exposure(p, state, d)
        scale = min(gross_cap / gross, 1.0) if gross > 0 else 1.0

        for sym in sorted(targets):
            target, f, dvol, reason = targets[sym]
            target = float(np.clip(target * scale, -p.max_position_weight, p.max_position_weight))
            drift = abs(target - weight_of(sym))
            if reason == "rebalance" and drift < p.rebalance_threshold:
                log_decision(sym, "hold", f, dvol, weight_of(sym))
                continue
            orders.append(
                PendingOrder(
                    symbol=sym,
                    decided_ts=d.isoformat(),
                    target_weight=target,
                    conviction=f,
                    expected_reward=f * dvol,
                    reason=reason,
                )
            )
            log_decision(sym, "buy" if reason == "entry" else "rebalance", f, dvol, target)

        return orders, decisions, sat

    # ---- benchmarks ------------------------------------------------------------

    def _benchmark_equity(
        self,
        state: PaperState,
        d: date,
        bench_closes: dict[str, dict[date, float]],
        data: dict[str, _SymbolData],
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for sym, units in state.benchmark_units.items():
            closes = bench_closes.get(sym, {})
            price = closes.get(d)
            if price is None:  # holiday for the ETF but not the universe: carry
                past = [t for t in closes if t <= d]
                price = closes[max(past)] if past else 0.0
            out[f"bench_{sym}"] = units * price
        base = state.equal_weight_base
        if base:
            rels = []
            for sym, p0 in base.items():
                sd = data.get(sym)
                if sd is None or p0 <= 0:
                    continue
                i = sd.idx.get(d)
                price = float(sd.close[i]) if i is not None else None
                if price is None:
                    past = [t for t in sd.ts if t <= d]
                    price = float(sd.close[sd.idx[max(past)]]) if past else None
                if price is not None:
                    rels.append(price / p0)
            out["bench_EW"] = state.starting_capital * float(np.mean(rels)) if rels else 0.0
        return out

    # ---- bootstrap ---------------------------------------------------------------

    def _load_or_init(
        self,
        capital: float,
        calendar: list[date],
        data: dict[str, _SymbolData],
        bench_closes: dict[str, dict[date, float]],
        *,
        log=print,
    ) -> tuple[PaperState, ContinualRRL]:
        model_path = self.models_root / "rrl_latest.bin"
        if self.store.state_path.exists():
            state = PaperState.load(self.store.state_path)
            if state.interval != self.paper.interval:
                raise ValueError(
                    f"saved portfolio state is on interval '{state.interval}' but the config "
                    f"says '{self.paper.interval}'; delete artifacts/paper (and artifacts/"
                    f"models) to start a fresh portfolio at the new cadence"
                )
            if model_path.exists():
                learner = ContinualRRL.load(model_path)
                if learner.feature_cols != self.feature_cols:
                    raise ValueError(
                        "saved model's feature columns do not match the current config; "
                        "delete artifacts/models to retrain"
                    )
            else:
                learner = self._new_learner()
            return state, learner

        # ---- first run: inception ----
        # On the hourly loop `start` may carry a time ("2026-07-17T15:30") to
        # incept at a specific bar -- e.g. a session's last bar, so the first
        # decisions are made on a flat book and fill at the next open.
        start: BarTs = self._iso(self.paper.start) if self.paper.start else calendar[-1]
        inception_days = [t for t in calendar if t >= start]
        if not inception_days:
            raise DataQualityError(f"no completed trading days on/after {start}")
        inception = inception_days[0]

        state = PaperState(
            universe=self.universe_name,
            starting_capital=capital,
            cash=capital,
            seed=self.cfg.seed,
            inception=inception.isoformat(),
            interval=self.paper.interval,
        )
        # Benchmarks: buy-and-hold units purchased at the first close *before*
        # inception (the last price knowable when the portfolio went live).
        base_day = max((t for t in calendar if t < inception), default=inception)
        for sym, closes in bench_closes.items():
            past = [t for t in closes if t <= base_day]
            if past:
                state.benchmark_units[sym] = capital / closes[max(past)]
        for sym, sd in data.items():
            past = [i for i, t in enumerate(sd.ts) if t <= base_day]
            if past:
                state.equal_weight_base[sym] = float(sd.close[past[-1]])

        learner = self._new_learner()
        if self.paper.pretrain_years > 0:
            self._pretrain(learner, data, inception, log=log)
        learner.save(model_path)
        state.save(self.store.state_path)
        log(
            f"[paper] inception {inception} · ${capital:,.0f} simulated capital · "
            f"{len(data)} tradeable symbols"
        )
        return state, learner

    def _new_learner(self) -> ContinualRRL:
        return ContinualRRL(
            self.feature_cols,
            learning_rate=self.paper.learning_rate,
            eta=self.paper.dsr_eta,
            seed=self.cfg.seed,
            l2=self.paper.learn_l2,
            max_weight_norm=self.paper.learn_max_weight_norm,
        )

    def _pretrain(
        self, learner: ContinualRRL, data: dict[str, _SymbolData], inception: date, *, log=print
    ) -> None:
        """Warm-start on history strictly before inception -- never on the
        forward period the portfolio will be judged on."""
        window = int(self.paper.pretrain_years * 252 * self.bars_per_day)
        feats: dict[str, np.ndarray] = {}
        rets: dict[str, np.ndarray] = {}
        for sym, sd in data.items():
            hist = [i for i, t in enumerate(sd.ts) if t < inception]
            if len(hist) < 2:
                continue
            hist = hist[-window:]
            close = sd.close[hist]
            feats[sym] = sd.x[hist][:-1]
            rets[sym] = np.diff(close) / close[:-1]
        n = sum(len(v) for v in rets.values())
        if n == 0:
            return
        log(f"[learn] pretraining on {n:,} historical bars across {len(feats)} symbols")
        learner.pretrain(feats, rets, cost=self._learn_cost, epochs=self.paper.pretrain_epochs)

    # ---- persistence helpers -------------------------------------------------------

    def _ledger_last_equity(self, state: PaperState) -> float:
        ledger = self.store.read("ledger")
        if ledger.is_empty():
            return state.starting_capital
        return float(ledger.sort("ts")["equity"][-1])

    def _ledger_equity_history(self) -> list[float]:
        """Prior days' equity, oldest first -- the kill switches' memory."""
        ledger = self.store.read("ledger")
        if ledger.is_empty():
            return []
        return ledger.sort("ts")["equity"].to_list()

    def _write_positions(self, pf, entry_ts: dict[str, str], last_close: dict[str, float]) -> None:
        rows = []
        equity = pf.equity(last_close)
        for sym in sorted(pf.positions):
            pos = pf.positions[sym]
            if pos.is_flat:
                continue
            price = last_close.get(sym, pos.avg_price)
            rows.append(
                {
                    "symbol": sym,
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "current_price": price,
                    "market_value": pos.market_value(price),
                    "unrealized_pnl": pos.unrealized_pnl(price),
                    "weight": pos.market_value(price) / equity if equity > 0 else 0.0,
                    "entry_ts": entry_ts.get(sym),
                    "total_costs": pos.total_costs,
                }
            )
        self.store.replace(
            "positions",
            pl.DataFrame(
                rows,
                schema={
                    "symbol": pl.Utf8,
                    "quantity": pl.Float64,
                    "avg_price": pl.Float64,
                    "current_price": pl.Float64,
                    "market_value": pl.Float64,
                    "unrealized_pnl": pl.Float64,
                    "weight": pl.Float64,
                    "entry_ts": pl.Utf8,
                    "total_costs": pl.Float64,
                },
            ),
        )

    def _fill_summary(
        self,
        summary: RunSummary,
        state: PaperState,
        pf,
        learner: ContinualRRL,
        last_close: dict[str, float],
        entry_ts: dict[str, str],
    ) -> None:
        equity = pf.equity(last_close)
        summary.last_processed = self._iso(state.last_processed) if state.last_processed else None
        summary.equity = equity
        summary.cash = pf.cash
        summary.halted = state.halted
        summary.total_return = equity / state.starting_capital - 1.0
        ledger = self.store.read("ledger")
        if not ledger.is_empty():
            row = ledger.sort("ts").tail(1)
            for col in row.columns:
                if col.startswith("bench_"):
                    base = state.starting_capital
                    summary.benchmarks[col.removeprefix("bench_")] = (
                        float(row[col][0]) / base - 1.0 if base > 0 else 0.0
                    )
        for sym in sorted(pf.positions):
            pos = pf.positions[sym]
            if pos.is_flat:
                continue
            price = last_close.get(sym, pos.avg_price)
            summary.positions.append(
                {
                    "symbol": sym,
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "current_price": price,
                    "unrealized_pnl": pos.unrealized_pnl(price),
                    "weight": pos.market_value(price) / equity if equity > 0 else 0.0,
                    "entry_ts": entry_ts.get(sym),
                }
            )
        summary.learning = {
            "n_updates": learner.n_updates,
            "cum_reward": learner.cum_reward,
            "avg_reward": learner.avg_reward,
            "ew_sharpe": learner.sharpe,
            "weight_norm": learner.weight_norm(),
            "checkpoint": str(self.models_root / "rrl_latest.bin"),
        }
