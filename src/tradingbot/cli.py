"""Command-line interface.

python -m tradingbot.cli fetch --symbols AAPL,MSFT,SPY
python -m tradingbot.cli compare --symbol AAPL
python -m tradingbot.cli backtest --symbol AAPL --capital 50000
python -m tradingbot.cli dashboard --symbol AAPL --open
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from tradingbot.agents.baselines import (
    AlwaysFlat,
    BuyAndHold,
    MovingAverageCrossover,
    RandomAgent,
    RRLAgent,
)
from tradingbot.config import Config
from tradingbot.dashboard import StrategyResult, build_dashboard
from tradingbot.data.sources import get_source
from tradingbot.data.store import BarStore
from tradingbot.features.technical import build_dataset, feature_columns
from tradingbot.metrics import excess_sharpe
from tradingbot.reporting import format_report, write_run

app = typer.Typer(
    add_completion=False,
    help="Simulated-capital trading bot: maximise simulated PnL, no exposure limits.",
)
console = Console()
warnings.filterwarnings("ignore")


def _load_config(path: Path | None) -> Config:
    return Config.load(path) if path else Config()


@app.command()
def fetch(
    symbols: str = typer.Option("SPY,AAPL,MSFT,JPM,XOM", help="Comma-separated tickers"),
    start: str = typer.Option("1995-01-01"),
    end: str | None = typer.Option(None),
    source: str = typer.Option("yahoo", help="yahoo | csv | synthetic"),
    root: Path = typer.Option(Path("data")),
) -> None:
    """Download bars into the local Parquet store."""
    store = BarStore(root)
    src = get_source(source)
    tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    table = Table("symbol", "bars", "start", "end", title="Fetched")
    for sym in tickers:
        try:
            df = src.fetch(sym, start, end)
            store.write(df)
            table.add_row(sym, str(df.height), str(df["ts"].min()), str(df["ts"].max()))
        except Exception as exc:  # noqa: BLE001
            table.add_row(sym, "[red]FAILED[/red]", str(exc)[:40], "")
    console.print(table)


@app.command()
def backtest(
    symbol: str = typer.Option("AAPL"),
    capital: float = typer.Option(100_000.0, help="Starting capital (simulated)"),
    strategy: str = typer.Option("buy_and_hold", help="buy_and_hold | flat | random | ma | rrl"),
    start: str = typer.Option("2016-01-01", help="Out-of-sample start"),
    config: Path | None = typer.Option(None),
    save: bool = typer.Option(True, help="Write artifacts to disk"),
) -> None:
    """Run one strategy over one symbol and report performance."""
    from tradingbot.backtest.runner import evaluate  # RL extras load only when backtesting

    cfg = _load_config(config)
    cfg.env.starting_capital = capital
    cfg.env.episode_length = None

    cols = feature_columns(cfg.features)
    store = BarStore(cfg.data.root)
    if symbol.upper() not in store:
        console.print(f"[red]No data for {symbol}. Run 'fetch' first.[/red]")
        raise typer.Exit(1)

    data = build_dataset(store.read(symbol), cfg.features)
    test = data.filter(pl.col("ts") >= pl.lit(start).str.to_date())
    agent = _build_agent(strategy, cols, data, cfg, start)

    _, report = evaluate(test, cols, agent, cfg.env, n_trials=1)
    console.print(format_report(report, title=f"{symbol} / {strategy} (simulated capital)"))

    if save:
        out = write_run(
            cfg.artifacts_root / f"{symbol}_{strategy}",
            config=cfg,
            result=evaluate(test, cols, agent, cfg.env)[0],
            report=report,
            label=f"{symbol}/{strategy}",
        )
        console.print(f"\n  artifacts -> {out}")


@app.command()
def compare(
    symbol: str = typer.Option("AAPL"),
    capital: float = typer.Option(100_000.0),
    start: str = typer.Option("2016-01-01", help="Out-of-sample start"),
    config: Path | None = typer.Option(None),
) -> None:
    """Compare every strategy under identical market conditions."""
    from tradingbot.backtest.runner import evaluate  # RL extras load only when backtesting

    cfg = _load_config(config)
    cfg.env.starting_capital = capital
    cfg.env.episode_length = None

    cols = feature_columns(cfg.features)
    store = BarStore(cfg.data.root)
    if symbol.upper() not in store:
        console.print(f"[red]No data for {symbol}. Run 'fetch' first.[/red]")
        raise typer.Exit(1)

    data = build_dataset(store.read(symbol), cfg.features)
    split = pl.lit(start).str.to_date()
    train, test = data.filter(pl.col("ts") < split), data.filter(pl.col("ts") >= split)
    console.print(
        f"[dim]{symbol}: {train.height} train bars, {test.height} out-of-sample bars "
        f"(OOS from {start}). All capital simulated.[/dim]"
    )

    strategies = ["buy_and_hold", "flat", "random", "ma", "rrl"]
    agents = {s: _build_agent(s, cols, train, cfg, start) for s in strategies}

    table = Table(title=f"{symbol} out-of-sample ({start} onward)")
    for c in (
        "strategy",
        "total ret",
        "CAGR",
        "Sharpe",
        "xSharpe",
        "max DD",
        "trades",
        "costs",
        "DSR",
    ):
        table.add_column(c, justify="right" if c != "strategy" else "left")

    # n_trials = number of configurations tried, which is what DSR deflates by.
    runs = []
    for name, agent in agents.items():
        agent.reset()
        result, r = evaluate(test, cols, agent, cfg.env, n_trials=len(strategies))
        runs.append((name, result, r))

    # Excess Sharpe vs buy-and-hold: the headline for a long-only book. A
    # positive raw Sharpe in a bull market proves nothing; xSharpe asks
    # whether anything is left after the benchmark.
    bench_equity = next(res.equity for name, res, _ in runs if name == "buy_and_hold")
    for name, result, r in runs:
        xs = excess_sharpe(np.asarray(result.equity), np.asarray(bench_equity))
        table.add_row(
            name,
            f"{r.total_return:.1%}",
            f"{r.cagr:.1%}",
            f"{r.sharpe:.2f}",
            "--" if name == "buy_and_hold" else f"{xs:+.2f}",
            f"{r.max_drawdown:.1%}",
            f"{r.n_trades:,}",
            f"${r.total_costs:,.0f}",
            f"{r.dsr:.2f}",
        )
    console.print(table)
    console.print(
        "\n[dim]If nothing beats buy_and_hold after costs, that is the finding.\n"
        "xSharpe is the Sharpe of (strategy - buy_and_hold); negative means no alpha.\n"
        "Deflated Sharpe below 0.95 means the result is consistent with luck.[/dim]"
    )


@app.command()
def dashboard(
    symbol: str = typer.Option("AAPL"),
    capital: float = typer.Option(100_000.0),
    start: str = typer.Option("2016-01-01", help="Out-of-sample start"),
    out: Path = typer.Option(Path("artifacts/dashboard.html")),
    config: Path | None = typer.Option(None),
    open_browser: bool = typer.Option(False, "--open", help="Open when done"),
) -> None:
    """Build a self-contained HTML analytics dashboard for every strategy."""
    from tradingbot.backtest.runner import evaluate  # RL extras load only when backtesting

    cfg = _load_config(config)
    cfg.env.starting_capital = capital
    cfg.env.episode_length = None

    cols = feature_columns(cfg.features)
    store = BarStore(cfg.data.root)
    if symbol.upper() not in store:
        console.print(f"[red]No data for {symbol}. Run 'fetch' first.[/red]")
        raise typer.Exit(1)

    data = build_dataset(store.read(symbol), cfg.features)
    split = pl.lit(start).str.to_date()
    train, test = data.filter(pl.col("ts") < split), data.filter(pl.col("ts") >= split)

    strategies = ["buy_and_hold", "flat", "random", "ma", "rrl"]
    results = []
    for name in strategies:
        agent = _build_agent(name, cols, train, cfg, start)
        agent.reset()
        result, report = evaluate(test, cols, agent, cfg.env, n_trials=len(strategies))
        results.append(StrategyResult(name=name, result=result, report=report))
        console.print(f"  [dim]{name:14} {report.total_return:>8.1%}  DSR {report.dsr:.2f}[/dim]")

    period = f"{test['ts'].min()} to {test['ts'].max()}"
    path = build_dashboard(
        results, out, symbol=symbol.upper(), period=period, starting_capital=capital
    )
    console.print(f"\n  dashboard -> {path}")
    if open_browser:
        import webbrowser

        webbrowser.open(f"file://{path.resolve()}")


@app.command()
def news(
    out: Path = typer.Option(Path("news"), help="Directory for signal.json + articles.parquet"),
    universe: str | None = typer.Option(
        None, help="Universe to resolve symbols against (default: config paper.universe)"
    ),
    config: Path | None = typer.Option(None),
    company: bool = typer.Option(True, help="Also sweep per-company news (slow, best-effort)"),
    max_symbols: int | None = typer.Option(
        None, help="Cap the per-company sweep (default: config paper.news.max_symbols)"
    ),
    pause: float | None = typer.Option(None, help="Seconds between per-company requests"),
) -> None:
    """Collect free economy + company news and publish a sentiment signal.

    Macro feeds (CNBC, MarketWatch/Dow Jones, the Fed press wire) always run;
    the per-company sweep goes through yfinance and is best-effort. Writes an
    append-only article archive plus ``signal.json``, which the paper loop reads
    as a bounded tilt on model conviction.
    """
    from tradingbot.news import collect as collect_news
    from tradingbot.news import summarize
    from tradingbot.paper.universe import resolve_universe

    cfg = _load_config(config)
    ncfg = cfg.paper.news
    symbols = resolve_universe(universe or cfg.paper.universe, cfg)
    console.print(f"[bold]news[/bold] universe={universe or cfg.paper.universe} ({len(symbols)})")

    sig = collect_news(
        out_dir=out,
        universe=symbols,
        company_news=company,
        max_symbols=max_symbols if max_symbols is not None else ncfg.max_symbols,
        pause=pause if pause is not None else ncfg.fetch_pause_seconds,
        half_life_days=ncfg.half_life_days,
        prior_count=ncfg.prior_count,
        lookback_days=ncfg.lookback_days,
        log_fn=lambda m: console.print(f"  {m}"),
    )
    console.print()
    console.print(summarize(sig, top=15))


@app.command()
def invest(
    strategy: str = typer.Option("rrl", help="Only 'rrl' is implemented"),
    capital: float = typer.Option(100_000.0, help="Simulated starting capital (first run only)"),
    universe: str | None = typer.Option(
        None,
        help=(
            "extended | sp500 | nasdaq100 | sp100 | config | watchlist file "
            "(default: config paper.universe)"
        ),
    ),
    start: str | None = typer.Option(
        None, help="Paper-trading inception date (first run only); default: config paper.start"
    ),
    config: Path | None = typer.Option(None),
    refresh: bool = typer.Option(True, help="Refresh market data before running"),
    as_of: str | None = typer.Option(None, help="Process bars up to this date (for replays/tests)"),
    open_browser: bool = typer.Option(False, "--open", help="Open the dashboard when done"),
) -> None:
    """Run the autonomous daily paper-investing loop (idempotent, simulated capital).

    Scans the whole universe, ranks opportunities by conviction, queues orders
    that fill at the next open, learns from every realized return, and updates
    the persistent portfolio + dashboard. Safe to run any number of times per
    day: a completed trading day is processed exactly once.
    """
    from datetime import date as _date

    from tradingbot.paper.dashboard import build_paper_dashboard
    from tradingbot.paper.engine import PaperEngine

    if strategy != "rrl":
        raise typer.BadParameter(f"unknown strategy '{strategy}' (only 'rrl' is implemented)")
    cfg = _load_config(config)
    if start:
        cfg.paper.start = start

    engine = PaperEngine(cfg, universe=universe)
    console.print(
        f"[dim]universe {engine.universe_name} ({len(engine.symbols)} symbols) · "
        f"all capital simulated[/dim]"
    )
    summary = engine.run(
        capital=capital,
        as_of=_date.fromisoformat(as_of) if as_of else None,
        refresh=refresh,
        log=lambda m: console.print(f"[dim]{m}[/dim]"),
    )

    today = summary.today
    if today is not None:
        _print_invest_day(today)
    else:
        console.print("[yellow]No new completed trading day - portfolio unchanged.[/yellow]")

    # ---- portfolio ----
    ret = summary.total_return
    tone = "green" if ret >= 0 else "red"
    console.print(
        f"\n[bold]Portfolio[/bold] (SIMULATED)  equity [bold]${summary.equity:,.2f}[/bold] "
        f"([{tone}]{ret:+.2%}[/{tone}] since inception) · cash ${summary.cash:,.2f} · "
        f"{len(summary.positions)} position(s)"
    )
    if summary.positions:
        table = Table("symbol", "shares", "basis", "price", "unrealized", "weight", "entered")
        for p in summary.positions:
            table.add_row(
                p["symbol"],
                f"{p['quantity']:,.0f}",
                f"${p['avg_price']:.2f}",
                f"${p['current_price']:.2f}",
                f"${p['unrealized_pnl']:+,.2f}",
                f"{p['weight']:.1%}",
                str(p["entry_ts"] or "—"),
            )
        console.print(table)

    if summary.benchmarks:
        table = Table("series", "total return", "vs portfolio", title="Benchmarks (same period)")
        table.add_row("paper portfolio", f"{ret:+.2%}", "—")
        for name, bret in summary.benchmarks.items():
            table.add_row(f"buy & hold {name}", f"{bret:+.2%}", f"{ret - bret:+.2%}")
        console.print(table)

    lrn = summary.learning
    console.print(
        f"[dim]learning: {lrn['n_updates']:,} updates · cum reward {lrn['cum_reward']:.3f} · "
        f"EW Sharpe {lrn['ew_sharpe']:.4f} · |w| {lrn['weight_norm']:.3f} · "
        f"checkpoint {lrn['checkpoint']}[/dim]"
    )

    path = build_paper_dashboard(engine.paper_root)
    console.print(f"\n  dashboard -> {path}")
    if open_browser:
        import webbrowser

        webbrowser.open(f"file://{path.resolve()}")


def _print_invest_day(today) -> None:
    """Render the newest processed day's fills and decisions."""
    console.print(f"\n[bold]Trading day {today.ts}[/bold]")
    if today.fills:
        table = Table(
            "symbol",
            "action",
            "qty",
            "fill",
            "ref open",
            "costs",
            "reason",
            title="Trades executed",
        )
        for f in today.fills:
            table.add_row(
                f["symbol"],
                f["action"],
                f"{abs(f['quantity']):,.0f}",
                f"${f['fill_price']:.2f}",
                f"${f['reference_price']:.2f}",
                f"${f['commission'] + f['fees'] + f['slippage']:.2f}",
                f["reason"],
            )
        console.print(table)
    else:
        console.print("[dim]no fills today[/dim]")

    actionable = [d for d in today.decisions if d["action"] != "hold"]
    holds = [d for d in today.decisions if d["action"] == "hold"]
    if actionable or holds:
        table = Table(
            "symbol",
            "action",
            "conviction",
            "reward pred",
            "allocation",
            title="Decisions (fill at next open)",
        )
        for dec in actionable + holds:
            table.add_row(
                dec["symbol"],
                dec["action"],
                f"{dec['conviction']:+.3f}",
                f"{dec['expected_reward']:+.3%}",
                f"{dec['allocation']:.1%}",
            )
        console.print(table)
    else:
        console.print(
            "[dim]no stock cleared the conviction bar today — staying in cash is a decision[/dim]"
        )


@app.command()
def rank(
    universe: str | None = typer.Option(
        None, help="sp500 | nasdaq100 | sp100 | config | watchlist file (default: paper.universe)"
    ),
    benchmark: str = typer.Option("QQQ", help="Excess-return benchmark"),
    horizon: int = typer.Option(20, help="Forward-return horizon in trading days"),
    refit_every: int = typer.Option(21, help="Refit cadence in trading days"),
    start: str | None = typer.Option(None, help="Bars start (default: paper.data_start)"),
    config: Path | None = typer.Option(None),
    out: Path = typer.Option(Path("artifacts/ranker"), help="Where scores/IC land"),
    shuffle_null: bool = typer.Option(
        False,
        "--shuffle-null",
        help="Also run the cross-sectional shuffle null (must produce IC ~ 0, else leakage)",
    ),
) -> None:
    """Walk-forward evaluation of the cross-sectional excess-return ranker.

    Trains LightGBM on excess total return vs the benchmark with purged,
    embargoed expanding windows, then reports per-date RankIC. This is the
    go/no-go gate for the ranker: mean IC >= 0.02 with stability > 0.15 or
    nothing gets built on top of it. Every invocation is appended to
    artifacts/trials.jsonl -- the n_trials that keeps DSR honest.
    """
    from tradingbot.agents.ranker import (
        ic_summary,
        rank_ic,
        shuffle_targets_within_date,
        walk_forward_scores,
    )
    from tradingbot.features.cross_section import build_panel
    from tradingbot.paper.gate import gate_signal, health_index
    from tradingbot.paper.universe import resolve_universe
    from tradingbot.trials import log_trial

    cfg = _load_config(config)
    if universe:
        cfg.paper.universe = universe
    symbols = resolve_universe(cfg.paper.universe, cfg)
    data_start = start or cfg.paper.data_start
    store = BarStore(cfg.data.root)

    bench_bars = store.read([benchmark.upper()], start=data_start)
    bars = store.read([s for s in symbols if s in store], start=data_start)
    if bars.is_empty() or bench_bars.is_empty():
        console.print(
            "[red]No cached bars for the universe/benchmark. Run 'invest' or 'fetch' first.[/red]"
        )
        raise typer.Exit(1)

    panel = build_panel(bars, bench_bars, horizon=horizon)
    console.print(
        f"[dim]panel: {panel.height:,} rows · {panel['symbol'].n_unique()} symbols · "
        f"{panel['ts'].min()} .. {panel['ts'].max()}[/dim]"
    )
    result = walk_forward_scores(panel, horizon=horizon, refit_every=refit_every, seed=cfg.seed)
    ic = rank_ic(result.scores)
    s = ic_summary(ic)

    table = Table("metric", "value", title=f"RankIC · {horizon}d excess vs {benchmark.upper()}")
    table.add_row("days scored", f"{s['n_days']}")
    table.add_row("mean IC", f"{s['mean']:+.4f}")
    table.add_row("IC stability (mean/std)", f"{s['stability']:+.3f}")
    table.add_row("t-stat", f"{s['t_stat']:+.2f}")
    table.add_row("frac positive days", f"{s['frac_positive']:.1%}")
    console.print(table)

    year_table = Table("year", "mean IC", "days", title="By year")
    by_year = (
        ic.with_columns(year=pl.col("ts").dt.year())
        .group_by("year")
        .agg(pl.col("ic").mean(), pl.len())
        .sort("year")
    )
    for row in by_year.iter_rows():
        year_table.add_row(str(row[0]), f"{row[1]:+.4f}", str(row[2]))
    console.print(year_table)

    # Gate preview: how often would the realized-efficacy gate have abstained?
    g = gate_signal(health_index(ic, horizon=horizon))
    live = g.drop_nulls(subset=["g"])
    if not live.is_empty():
        abstain = float((live["g"] < 0.2).mean())
        console.print(f"[dim]gate preview: abstains {abstain:.1%} of days at threshold 0.2[/dim]")

    out.mkdir(parents=True, exist_ok=True)
    result.scores.write_parquet(out / "scores.parquet", compression="zstd")
    ic.write_parquet(out / "rank_ic.parquet", compression="zstd")
    console.print(f"\n  scores -> {out / 'scores.parquet'}")

    if shuffle_null:
        null_res = walk_forward_scores(
            shuffle_targets_within_date(panel, seed=cfg.seed),
            horizon=horizon,
            refit_every=refit_every,
            seed=cfg.seed,
        )
        ns = ic_summary(rank_ic(null_res.scores))
        clean = abs(ns["mean"]) < 0.01
        tone = "green" if clean else "red"
        console.print(
            f"[{tone}]shuffle null: mean IC {ns['mean']:+.4f} over {ns['n_days']} days -- "
            f"{'pipeline is leak-free' if clean else 'LEAKAGE: shuffled targets scored'}[/{tone}]"
        )

    n = log_trial(
        Path("artifacts/trials.jsonl"),
        {
            "command": "rank",
            "universe": cfg.paper.universe,
            "benchmark": benchmark.upper(),
            "horizon": horizon,
            "refit_every": refit_every,
            "data_start": data_start,
            "n_days": s["n_days"],
            "mean_ic": s["mean"],
            "stability": s["stability"],
        },
    )
    console.print(
        f"[dim]trials on record: {n} (artifacts/trials.jsonl -- feeds DSR n_trials)[/dim]"
    )

    verdict = s["mean"] >= 0.02 and s["stability"] > 0.15
    color = "green" if verdict else "yellow"
    console.print(
        f"[{color}]sanity gate {'PASSED' if verdict else 'NOT met'}: "
        f"need mean IC >= 0.02 and stability > 0.15 before building on this signal[/{color}]"
    )


def _build_agent(strategy: str, cols: list[str], train: pl.DataFrame, cfg: Config, start: str):
    from tradingbot.backtest.runner import train_rrl  # RL extras load only when backtesting

    match strategy:
        case "buy_and_hold":
            return BuyAndHold()
        case "flat":
            return AlwaysFlat()
        case "random":
            return RandomAgent(seed=cfg.seed)
        case "ma":
            return MovingAverageCrossover(cols.index("dist_ma_21"), cols.index("dist_ma_63"))
        case "rrl":
            agent = RRLAgent(len(cols), seed=cfg.seed)
            fit = train.filter(pl.col("ts") < pl.lit(start).str.to_date())
            train_rrl(fit if fit.height > 100 else train, cols, agent, cfg.env, epochs=30)
            return agent
        case _:
            raise typer.BadParameter(f"unknown strategy '{strategy}'")



@app.command("trade")
def trade(
    capital: float = typer.Option(
        100_000.0, help="Simulated capital at inception (first run only)"
    ),
    universe: str | None = typer.Option(None, help="Default: config paper.universe"),
    config: Path | None = typer.Option(None),
    artifacts: Path = typer.Option(Path("artifacts/v2"), help="Where v2 keeps its book"),
    refresh: bool = typer.Option(True, help="Refresh market data before running"),
    as_of: str | None = typer.Option(None, help="Process bars completed by this ET instant"),
    gross: float | None = typer.Option(
        None, help="Pin gross exposure (multiple of equity). Default: size by --vol-target"
    ),
    vol_target: float = typer.Option(
        0.35, help="Annualised volatility the book is sized to when --gross is unset"
    ),
    max_gross: float | None = typer.Option(
        None, help="Ceiling on gross after vol targeting. Default: none"
    ),
    max_weight: float | None = typer.Option(
        None, help="Cap on any single name, as a fraction of equity. Default: none"
    ),
    max_net: float | None = typer.Option(
        None, help="Clamp on net exposure after the macro tilt. Default: none"
    ),
    leverage: float = typer.Option(20.0, help="Account leverage available to the margin book"),
) -> None:
    """Run the v2 loop: cross-sectional long/short on the NautilusTrader engine.

    Idempotent. Each firing restores the persisted book, replays every bar
    completed since the last run, and persists the result; running it twice on
    the same bars is a no-op. All capital is SIMULATED.

    The objective is to maximise simulated PnL, so **no exposure limit is set
    by default**: gross floats to whatever hits --vol-target, and there is no
    per-name cap and no net clamp unless you pass one. What is not optional is
    the honesty of the simulation -- one-bar execution delay, no look-ahead,
    full transaction costs, and a liquidity floor -- because a number produced
    without those is not a result.

    v2 replaces the retired v1 loop (`tradingbot invest`), which was long-only and
    flat by every close. See docs/FINDINGS.md for why.
    """
    from datetime import datetime as _dt
    from decimal import Decimal as _D

    from tradingbot.nautilus.runner import V2Runner
    from tradingbot.nautilus.signals import SizingConfig

    cfg = _load_config(config)
    cfg.env.starting_capital = capital
    runner = V2Runner(
        cfg,
        universe=universe,
        artifacts=artifacts,
        sizing=SizingConfig(
            target_gross=gross,
            vol_target_annual=vol_target,
            max_gross=max_gross,
            max_position_weight=max_weight,
            max_net_exposure=max_net,
        ),
        leverage=_D(str(leverage)),
    )
    if refresh:
        runner.refresh_data(log=lambda m: console.print(f"[dim]{m}[/dim]"))

    report = runner.run(
        as_of=_dt.fromisoformat(as_of) if as_of else None,
        log=lambda m: console.print(f"[dim]{m}[/dim]"),
    )

    if report.incepted:
        console.print(f"[bold]Incepted[/bold] at ${capital:,.0f}, all cash, no positions.")
    if report.bars_processed == 0:
        console.print("[yellow]No new completed bars — nothing to do.[/yellow]")
        return

    colour = "green" if report.equity >= cfg.env.starting_capital else "red"
    console.print(
        f"\n[bold]{report.bars_processed}[/bold] bar(s) → "
        f"equity [{colour}]${report.equity:,.2f}[/{colour}] "
        f"(balance ${report.balance:,.2f})"
    )
    console.print(
        f"book: {report.n_long} long / {report.n_short} short · "
        f"gross {report.gross:.2f}x · net {report.net:+.2f}x · "
        f"{report.fills} fills · borrow ${report.borrow_charged:,.2f}"
    )
    console.print(f"[dim]watermark {report.last_processed}[/dim]")

if __name__ == "__main__":
    app()
