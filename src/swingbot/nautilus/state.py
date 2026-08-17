"""The v2 portfolio, as it survives between thirty-minute runs.

Each cron firing is a *fresh* ``BacktestEngine``. Nothing in Nautilus persists
across processes without a Redis-backed cache, which this bot does not have and
does not want. So the book is serialised to JSON, and the next run rebuilds it.

**What has to be persisted is exactly two things**, and finding that out took an
experiment rather than a reading of the docs. On a Nautilus *margin* account,
``balance_total`` is cash plus realized PnL and is **not** reduced by opening a
position -- a position locks margin (``balance_locked``) instead of consuming
cash. So the complete state of the book is::

    (balance_total, {symbol: (signed_qty, avg_px_open)})

and net asset value is a derived quantity: ``balance_total`` plus unrealized PnL
marked at current prices. Persisting "cash" in the ordinary sense, as v1 did,
would be wrong here -- it would double-count every open position.

That serialisation is *exact*, not approximate. A run that stops at bar N,
persists, and resumes reproduces the balance, the NAV and every position of an
uninterrupted run to the cent (``tests/test_nautilus_invariants.py`` asserts
it). Two details make it exact and both are easy to get wrong:

* the restore fills must be **free**, and the zero-fee flag must be cleared on a
  *later* bar than the one that submits them -- fills are processed after the
  callback returns, so clearing it inline charges the rebuild;
* each restored position needs a synthetic bar at its own average price, so the
  rebuild fill lands exactly on the cost basis it is supposed to inherit.

Everything here is SIMULATED CAPITAL. The flag is written into the file and
asserted on load, so a state file from some hypothetical real-money system can
never be mistaken for this one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path

STATE_VERSION = 2


@dataclass
class HeldPosition:
    """One open position. ``quantity`` is signed: negative is a short."""

    symbol: str
    quantity: float
    avg_price: float
    entry_ts: str | None = None

    @property
    def is_short(self) -> bool:
        return self.quantity < 0.0


@dataclass
class V2State:
    """Everything the next run needs to continue exactly where this one stopped."""

    universe: str
    starting_capital: float
    # Nautilus `account.balance_total()`: cash + realized PnL. NOT cash-on-hand.
    account_balance: float
    interval: str = "30m"
    seed: int = 7

    inception: str | None = None
    last_processed: str | None = None  # ISO, naive ET, the bar's OPEN label

    positions: list[HeldPosition] = field(default_factory=list)

    realized_pnl: float = 0.0
    # Spread + slippage + impact, charged as explicit commission (see costs.py).
    cumulative_friction: float = 0.0
    # Commissions and regulatory fees.
    cumulative_fees: float = 0.0
    # Short borrow, which Nautilus does not charge because it is not a fill
    # event. v1 never paid this; v2 holds shorts overnight, so it must.
    cumulative_borrow: float = 0.0
    n_fills: int = 0

    # Units of each benchmark bought at inception, for the comparison series.
    benchmark_units: dict[str, float] = field(default_factory=dict)

    version: int = STATE_VERSION
    simulated_capital: bool = True  # always; asserted on load
    updated_utc: str = ""

    # ---- derived ---------------------------------------------------------

    @property
    def book(self) -> dict[str, tuple[float, float]]:
        """``{symbol: (signed_qty, avg_px)}`` -- the restore instruction."""
        return {p.symbol: (p.quantity, p.avg_price) for p in self.positions}

    def gross_exposure(self, prices: dict[str, float]) -> float:
        return sum(abs(p.quantity) * prices.get(p.symbol, p.avg_price) for p in self.positions)

    def net_exposure(self, prices: dict[str, float]) -> float:
        return sum(p.quantity * prices.get(p.symbol, p.avg_price) for p in self.positions)

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        return sum(
            p.quantity * (prices.get(p.symbol, p.avg_price) - p.avg_price) for p in self.positions
        )

    def equity(self, prices: dict[str, float]) -> float:
        """Net asset value: balance plus unrealized PnL at current marks."""
        return self.account_balance + self.unrealized_pnl(prices)

    def short_notional(self, prices: dict[str, float]) -> float:
        return sum(
            abs(p.quantity) * prices.get(p.symbol, p.avg_price)
            for p in self.positions
            if p.is_short
        )

    # ---- io --------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_utc = datetime.now(UTC).isoformat(timespec="seconds")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(path)  # atomic on POSIX: a crash never leaves half a book
        return path

    @classmethod
    def load(cls, path: str | Path) -> V2State:
        raw = json.loads(Path(path).read_text())
        if not raw.get("simulated_capital", False):
            raise ValueError(f"{path} is not flagged simulated_capital; refusing to load")
        if raw.get("version") != STATE_VERSION:
            raise ValueError(
                f"{path} is state version {raw.get('version')}, this is v{STATE_VERSION}. "
                "v1 state is not upgradeable -- v2 incepts fresh at $100k by design."
            )
        raw["positions"] = [HeldPosition(**p) for p in raw.get("positions", [])]
        # Drop keys this version no longer models. A live book's state file
        # outlives the code that wrote it, so a removed field has to degrade to
        # "ignored" -- otherwise shipping the removal is what breaks the bot.
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    @classmethod
    def incept(
        cls, *, universe: str, capital: float = 100_000.0, interval: str = "30m", seed: int = 7
    ) -> V2State:
        """A brand-new book: all cash, no positions, no history.

        This is what the v1 -> v2 cutover produces. v1's final $96,211.75 is not
        carried over; v2 starts from the same $100,000 v1 started from, so the
        two records are directly comparable rather than one continuing the
        other's drawdown.
        """
        return cls(
            universe=universe,
            starting_capital=capital,
            account_balance=capital,
            interval=interval,
            seed=seed,
        )
