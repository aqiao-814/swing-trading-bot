"""Typed configuration for the whole system.

Every knob that affects a backtest lives here so a run is reproducible from a
single YAML file plus a seed. Nothing reads loose globals.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ActionSpaceKind(StrEnum):
    DISCRETE = "discrete"  # {short, flat, long}
    CONTINUOUS = "continuous"  # position in [-1, +1]


class RewardKind(StrEnum):
    DSR = "dsr"  # Moody & Saffell differential Sharpe ratio
    NET_LOG_RETURN = "net_log_return"  # log return minus costs
    NET_SIMPLE_RETURN = "net_simple_return"


class DataConfig(BaseModel):
    """Where price data comes from and lives."""

    root: Path = Path("data")
    # yahoo | csv | synthetic. Not stooq: it sits behind a proof-of-work
    # anti-bot challenge as of 2026-07 and is not scriptable (see sources.py).
    source: str = "yahoo"
    universe: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "SPY"])
    start: str = "1995-01-01"
    end: str | None = None
    # Bars whose adjusted close moves more than this in one day are treated as
    # suspect (usually an unhandled split) and flagged by validation.
    max_abs_daily_return: float = 0.75


class FeatureConfig(BaseModel):
    return_horizons: list[int] = Field(default_factory=lambda: [1, 5, 10, 21, 63])
    vol_windows: list[int] = Field(default_factory=lambda: [21, 63])
    rsi_window: int = 14
    macd: tuple[int, int, int] = (12, 26, 9)
    bollinger_window: int = 20
    zscore_window: int = 252
    # Fractional differentiation order; d<0.6 achieves stationarity for most
    # liquid series (Lopez de Prado, AFML Ch.5).
    fracdiff_d: float = 0.4
    fracdiff_threshold: float = 1e-4
    # Warmup bars discarded so no feature is computed from a partial window.
    warmup: int = 300


class CostConfig(BaseModel):
    """Trading frictions. Defaults model a retail zero-commission US equity broker.

    All rates are fractions of notional unless stated otherwise.
    """

    commission_per_share: float = 0.0
    commission_bps: float = 0.0
    min_commission: float = 0.0
    # Half-spread paid on entry and exit. 1bp ~ liquid megacap; 5-10bp small cap.
    half_spread_bps: float = 1.0
    # Fixed slippage floor applied on top of spread.
    slippage_bps: float = 0.5
    # Square-root market impact (Almgren-Chriss): impact = coef * sigma * sqrt(Q/V)
    impact_coef: float = 0.1
    use_sqrt_impact: bool = True
    # Annualized borrow rate charged on short notional, accrued daily.
    short_borrow_annual_bps: float = 30.0
    # SEC Section 31 fee on sell notional, in bps (~$27.80 per $1M notional).
    sec_fee_bps: float = 0.278
    # FINRA Trading Activity Fee, per share sold, capped per trade.
    taf_per_share: float = 0.000166
    taf_cap_per_trade: float = 8.30
    # Days between decision and fill. 1 = decide on close, fill next open.
    execution_delay_bars: int = 1


class RiskConfig(BaseModel):
    max_gross_exposure: float = 1.0  # 1.0 = no leverage
    max_position_weight: float = 1.0
    allow_short: bool = True
    stop_loss_pct: float | None = 0.08
    take_profit_pct: float | None = None
    max_daily_loss_pct: float | None = 0.05
    max_drawdown_pct: float | None = 0.25
    # Fractional Kelly / vol targeting
    vol_target_annual: float | None = 0.15
    kelly_fraction: float = 0.5
    # No-trade band: don't rebalance until the position drifts this far (in
    # fraction of equity) from target. Without it, vol targeting recomputes the
    # target every bar and churns on noise -- thousands of tiny trades whose
    # costs swamp the benefit. Real desks always band their rebalancing.
    rebalance_threshold: float = 0.05


class EnvConfig(BaseModel):
    starting_capital: float = 100_000.0
    action_space: ActionSpaceKind = ActionSpaceKind.DISCRETE
    reward: RewardKind = RewardKind.DSR
    # DSR adaptation rate. Smaller = longer memory.
    dsr_eta: float = 0.01
    episode_length: int | None = 252
    trading_days_per_year: int = 252
    costs: CostConfig = Field(default_factory=CostConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)


class BacktestConfig(BaseModel):
    n_train_folds: int = 10
    n_test_folds: int = 8
    purge_days: int = 21
    embargo_days: int = 21


class PaperConfig(BaseModel):
    """Forward paper-trading: one persistent simulated portfolio over a universe.

    Everything here is about *live-like* operation: decisions on the latest
    completed bar, fills at the next open, continual learning, and a state file
    that makes the daily run idempotent.
    """

    # Universe name (sp500 | nasdaq100 | sp100 | config | a watchlist file path).
    universe: str = "nasdaq100"
    # Bar interval the loop trades on: "1d" (decide at close, fill next open),
    # "60m" (decide each completed hourly bar), or "30m" (each half-hour bar).
    # Intraday history is capped by the data source: ~730 days for 60m, only
    # ~60 days for 30m -- so a 30m loop is meant to be *seeded* from a model
    # trained offline on longer history rather than pretrained on 30m bars.
    interval: Literal["1d", "60m", "30m"] = "1d"
    # Paper-trading inception. None = today (portfolio starts with today's run).
    # A past date makes the engine replay forward day-by-day from there, which
    # is how a fresh install builds a real forward track record. On the hourly
    # loop this may include a time ("2026-07-17T15:30") to incept at one bar.
    start: str | None = None
    # How far back to fetch bars. Features need ~550 bars of warmup, and the
    # learner pretrains on history *before* inception, so keep several years.
    data_start: str = "2018-01-01"
    # Benchmarks tracked in the ledger. First one is the headline comparison.
    benchmark_symbols: list[str] = Field(default_factory=lambda: ["SPY", "QQQ"])

    # ---- allocation ----
    # Cap on the number of names held at once. None = uncapped: the book holds
    # every name that clears ``min_conviction``, and breadth is limited only by
    # capital. Sizing is already self-normalising -- targets are conviction-
    # proportional and then scaled so gross lands on ``max_gross_exposure`` --
    # so widening the book dilutes each name instead of levering the portfolio.
    max_positions: int | None = None
    max_position_weight: float = 0.20  # fraction of equity per name
    max_gross_exposure: float = 0.90  # never forced fully invested
    min_conviction: float = 0.15  # |policy output| needed to open
    exit_conviction: float = 0.05  # close when conviction decays below this
    # No-trade band, as a fraction of the position's own size: rebalance only
    # once a holding drifts this far from its target relative to
    # max(|target|, |current|). It MUST be relative rather than an absolute
    # slice of equity. With an uncapped book a target is ~gross/N -- a few
    # tenths of a percent across a few hundred names -- so the old absolute
    # 0.05 band was an order of magnitude wider than the positions it governed:
    # every holding read as "close enough, don't trade" and kept its full old
    # weight while fresh entries were sized on top, ratcheting gross to 100% of
    # cash and quietly voiding max_gross_exposure. At the former 20% position
    # size 0.25 x 0.20 reproduces exactly the old 0.05 band.
    # Note the band's inherent cost, which is not a bug and not new: holdings
    # may sit up to this fraction above target without trading, so realized
    # gross can exceed max_gross_exposure by up to that same fraction before
    # cash (never negative) becomes the binding constraint. Lower it to track
    # the gross target more tightly at the price of more turnover.
    rebalance_band_frac: float = 0.25
    # Vol-scaled stop: exit when price falls stop_loss_sigma standard
    # deviations of the name's own horizon volatility below cost basis.
    # A fixed-percentage stop is a different distance in sigma for every name
    # (the old 10% stop was ~1.2 sigma on high-vol names -- a coin flip that
    # loses 10%, hit ~20% of the time by pure noise). Set to None to fall
    # back to the fixed stop_loss_pct.
    stop_loss_sigma: float | None = 2.0
    stop_horizon_days: int = 20  # holding horizon the stop is scaled to
    stop_loss_pct: float | None = 0.10  # fixed fallback when sigma stop is off
    allow_short: bool = False  # long-only keeps simulated cash non-negative
    # Cancel a pending order if the symbol prints no bar for this many days.
    cancel_after_days: int = 5

    # ---- day trading (intraday only) ----
    # When True on an intraday loop ("60m"/"30m"), the book is forced flat
    # before every session close: no position is ever carried overnight. On the
    # flatten bar (the last bar whose next-open fill still lands inside the same
    # session) every holding is sold to zero, and no new position is opened on
    # the flatten bar or the final bar because it could not be closed again the
    # same day. Ignored on the "1d" loop, where a bar *is* a day. Off by default
    # so the daily research/backtest path is unchanged.
    day_trading: bool = False

    # ---- risk ----
    # There is deliberately NO portfolio kill switch here. A latching halt (fire
    # on a drawdown, flatten, stay dead until an operator clears it) is a swing-
    # trading control: it assumes the book is meant to persist and that a human
    # is on hand to restart it. On a day-trading loop the book is already flat at
    # every close, the worst single-bar loss is bounded by the per-name stop plus
    # the gross cap, and a latched halt just means the bot silently stops trading
    # for days until someone notices. Risk is controlled continuously instead:
    # per-name vol-scaled stops, post-stop de-grossing, and the gross cap.
    # Model health is still measured every bar (conviction_std / frac_saturated
    # in the learning table) and shouted about in the run log -- it just no
    # longer halts trading.

    # ---- stop-loss discipline ----
    # A stopped-out symbol may not be re-entered for this many calendar days.
    # The stop just said the model was wrong about this name; instantly
    # re-buying it (or churning back in on the next signal) repeats the loss.
    stop_cooldown_days: int = 10
    # On an intraday loop the day-based cooldown is the wrong clock: 10 calendar
    # days is ~130 thirty-minute bars, freezing a name for two weeks of
    # decisions. When set (and the interval is intraday), the re-entry lockout
    # and the de-gross window below are this many BARS of elapsed time instead,
    # so the same discipline runs on the day-trader's clock and a stopped name
    # becomes tradeable again the same session once the lockout lapses.
    stop_cooldown_bars: int | None = None
    # Stops inside the cooldown window lower the gross-exposure cap: cash a stop
    # frees should stay cash, not rotate straight into the next correlated name.
    # The cut is proportional to the FRACTION of the book that stopped out, not
    # a flat slab per stop -- with ``max_positions`` uncapped a 300-name book
    # will always have a few names stopped out somewhere, and a per-stop slab
    # would pin it at the floor permanently. At the old 10-name book size one
    # stop still de-grosses ~10% of the cap, so this preserves the original
    # behaviour where it was calibrated. 0 disables de-grossing.
    stop_degross_fraction: float = 1.0
    min_gross_exposure: float = 0.30

    # ---- learning ----
    learning_rate: float = 0.01
    dsr_eta: float = 0.01
    # Saturation guards on the online policy: L2 shrinkage plus a hard cap on
    # ||w||. Without them the weight norm drifts up until tanh pins at +/-1,
    # every conviction ties at 1.0, and "conviction-ranked" sizing degenerates
    # into the sort's tiebreak (i.e. alphabetical).
    learn_l2: float = 1e-3
    learn_max_weight_norm: float = 1.0
    # Hard cap on the recurrent weight |u| in F_t = tanh(w.x + u*F_{t-1} + b).
    # The recurrence only contracts while |u| < 1; past that it is explosive and
    # convictions saturate to +/-1 within a few bars no matter the features.
    # None keeps the original uncapped behavior; a live intraday loop seeded
    # from an offline model wants this set (e.g. 0.7) so the seed's recurrence
    # can't re-saturate during pretraining or forward trading.
    learn_max_recurrence: float | None = None
    # Pretrain the policy on this many years of history before inception, so
    # day one is not a random coin-flip. 0 disables pretraining.
    pretrain_years: float = 3.0
    pretrain_epochs: int = 1
    # Keep at most this many dated checkpoints (latest is always kept).
    max_checkpoints: int = 30


class Config(BaseModel):
    seed: int = 7
    run_name: str = "default"
    artifacts_root: Path = Path("artifacts")
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    paper: PaperConfig = Field(default_factory=PaperConfig)

    @model_validator(mode="after")
    def _check_risk_coherence(self) -> Config:
        r = self.env.risk
        if r.max_position_weight > r.max_gross_exposure:
            raise ValueError(
                f"max_position_weight ({r.max_position_weight}) exceeds "
                f"max_gross_exposure ({r.max_gross_exposure})"
            )
        if self.env.starting_capital <= 0:
            raise ValueError("starting_capital must be positive")
        return self

    @classmethod
    def load(cls, path: str | Path) -> Config:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(raw)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))
