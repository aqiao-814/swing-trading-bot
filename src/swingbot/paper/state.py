"""Persistent paper-portfolio state.

One JSON file is the source of truth for *money* (cash, positions, pending
orders, watermark of the last processed day); Parquet files carry the *history*
(ledger, trades, decisions, learning metrics). The JSON is small and atomic;
the Parquet files are append-only per processed day, so idempotency reduces to
one check: has ``last_processed`` already reached this bar date?

Everything in here is SIMULATED CAPITAL. The state file says so explicitly and
the flag is asserted on load, so a file from some future real-money system can
never be silently mistaken for this one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from swingbot.portfolio import Portfolio, Position

STATE_VERSION = 1


@dataclass
class PendingOrder:
    """A decision made on bar t, waiting for bar t+1's open to fill."""

    symbol: str
    decided_ts: str  # ISO date of the bar the decision was made on
    target_weight: float  # fraction of equity, signed
    conviction: float  # policy output f in [-1, 1]
    expected_reward: float  # predicted next-day net return
    reason: str  # entry | exit | rebalance | stop_loss


@dataclass
class HeldPosition:
    """Broker-grade position record, JSON-serialisable."""

    symbol: str
    quantity: float
    avg_price: float
    realized_pnl: float = 0.0
    total_costs: float = 0.0
    entry_ts: str | None = None  # first fill date of the current lifecycle


@dataclass
class PaperState:
    """Complete persistent state of the simulated portfolio."""

    universe: str
    starting_capital: float
    cash: float
    seed: int
    # Bar interval this portfolio trades on ("1d" | "60m"). Timestamps below
    # are date-ISO on the daily loop and datetime-ISO intraday; the engine
    # refuses to run a state file at the wrong resolution.
    interval: str = "1d"
    inception: str | None = None  # first processed trading day
    last_processed: str | None = None  # idempotency watermark
    positions: list[HeldPosition] = field(default_factory=list)
    pending_orders: list[PendingOrder] = field(default_factory=list)
    realized_pnl: float = 0.0
    cumulative_costs: float = 0.0
    cumulative_slippage: float = 0.0
    n_fills: int = 0
    # Benchmark units bought (virtually) at inception, for the comparison series.
    benchmark_units: dict[str, float] = field(default_factory=dict)
    equal_weight_base: dict[str, float] = field(default_factory=dict)
    # Last stop-out fill date per symbol (ISO), for re-entry cooldown and
    # post-stop de-grossing.
    last_stop_out: dict[str, str] = field(default_factory=dict)
    # A fired kill switch: reason string plus the day it tripped. While set,
    # the engine only ever flattens -- no entries, no rebalances -- until a
    # human clears it. Surviving restarts is the entire point.
    halted: str | None = None
    halted_ts: str | None = None
    version: int = STATE_VERSION
    simulated_capital: bool = True  # always; asserted on load
    updated_utc: str = ""

    # ---- portfolio bridge ---------------------------------------------------

    def to_portfolio(self) -> Portfolio:
        """Rehydrate the accounting engine from persisted state."""
        pf = Portfolio(self.starting_capital)
        pf.cash = self.cash
        pf.realized_pnl = self.realized_pnl
        pf.cumulative_costs = self.cumulative_costs
        pf.cumulative_slippage = self.cumulative_slippage
        for hp in self.positions:
            pf.positions[hp.symbol] = Position(
                symbol=hp.symbol,
                quantity=hp.quantity,
                avg_price=hp.avg_price,
                realized_pnl=hp.realized_pnl,
                total_costs=hp.total_costs,
            )
        return pf

    def capture_portfolio(self, pf: Portfolio, entry_ts: dict[str, str]) -> None:
        """Write the accounting engine's truth back into serialisable state."""
        self.cash = pf.cash
        self.realized_pnl = pf.realized_pnl
        self.cumulative_costs = pf.cumulative_costs
        self.cumulative_slippage = pf.cumulative_slippage
        self.positions = [
            HeldPosition(
                symbol=p.symbol,
                quantity=p.quantity,
                avg_price=p.avg_price,
                realized_pnl=p.realized_pnl,
                total_costs=p.total_costs,
                entry_ts=entry_ts.get(p.symbol),
            )
            for p in pf.positions.values()
            if not p.is_flat
        ]

    # ---- io -------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_utc = datetime.now(UTC).isoformat()
        payload = asdict(self)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)  # atomic on POSIX: a crash never leaves half a state
        return path

    @classmethod
    def load(cls, path: str | Path) -> PaperState:
        raw = json.loads(Path(path).read_text())
        if not raw.get("simulated_capital", False):
            raise ValueError(f"{path} is not flagged simulated_capital; refusing to load")
        raw["positions"] = [HeldPosition(**p) for p in raw.get("positions", [])]
        raw["pending_orders"] = [PendingOrder(**o) for o in raw.get("pending_orders", [])]
        return cls(**raw)


# ---- parquet history ---------------------------------------------------------


class PaperStore:
    """Append-only Parquet history under ``<root>/portfolio/``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.portfolio_dir = self.root / "portfolio"
        self.portfolio_dir.mkdir(parents=True, exist_ok=True)

    @property
    def state_path(self) -> Path:
        return self.portfolio_dir / "state.json"

    def _path(self, name: str) -> Path:
        return self.portfolio_dir / f"{name}.parquet"

    def read(self, name: str) -> pl.DataFrame:
        path = self._path(name)
        return pl.read_parquet(path) if path.exists() else pl.DataFrame()

    def append(self, name: str, rows: pl.DataFrame) -> None:
        """Append rows; caller guarantees the day has not been written before.

        Columns added by newer code backfill as null on old files instead of
        being silently dropped -- history written before a metric existed
        stays honest about not having it.
        """
        if rows.is_empty():
            return
        existing = self.read(name)
        if not existing.is_empty():
            for col in rows.columns:
                if col not in existing.columns:
                    existing = existing.with_columns(pl.lit(None).cast(rows[col].dtype).alias(col))
            # vertical_relaxed upcasts to a common supertype instead of raising:
            # a flat day writes `invested`/`unrealized_pnl` as Int64 (sum([]) is
            # int 0), while days with holdings produce Float64. Strict "vertical"
            # rejects that widening; relaxed promotes Int64 -> Float64 and rewrites
            # the file consistently. Applies to every table and numeric column.
            rows = pl.concat([existing, rows.select(existing.columns)], how="vertical_relaxed")
        rows.write_parquet(self._path(name), compression="zstd")

    def replace(self, name: str, rows: pl.DataFrame) -> None:
        rows.write_parquet(self._path(name), compression="zstd")

    def set_decision_results(self, ts: date, results: dict[str, float]) -> None:
        """Backfill the realized next-day return onto an earlier day's decisions."""
        decisions = self.read("decisions")
        if decisions.is_empty() or not results:
            return
        updated = decisions.with_columns(
            pl.when(pl.col("ts") == pl.lit(ts))
            .then(
                pl.col("symbol").replace_strict(
                    results, default=pl.col("result"), return_dtype=pl.Float64
                )
            )
            .otherwise(pl.col("result"))
            .alias("result")
        )
        self.replace("decisions", updated)
