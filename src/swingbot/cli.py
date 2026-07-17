"""Command-line interface.

python -m swingbot.cli fetch --symbols AAPL,MSFT,SPY
python -m swingbot.cli compare --symbol AAPL
python -m swingbot.cli backtest --symbol AAPL --capital 50000
python -m swingbot.cli dashboard --symbol AAPL --open
"""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from swingbot.agents.baselines import (
    AlwaysFlat,
    BuyAndHold,
    MovingAverageCrossover,
    RandomAgent,
    RRLAgent,
)
from swingbot.backtest.runner import evaluate, train_rrl
from swingbot.config import Config
from swingbot.dashboard import StrategyResult, build_dashboard
from swingbot.data.sources import get_source
from swingbot.data.store import BarStore
from swingbot.features.technical import build_dataset, feature_columns
from swingbot.reporting import format_report, write_run

app = typer.Typer(add_completion=False, help="Simulated-capital swing-trading research system.")
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
    for c in ("strategy", "total ret", "CAGR", "Sharpe", "max DD", "trades", "costs", "DSR"):
        table.add_column(c, justify="right" if c != "strategy" else "left")

    # n_trials = number of configurations tried, which is what DSR deflates by.
    for name, agent in agents.items():
        agent.reset()
        _, r = evaluate(test, cols, agent, cfg.env, n_trials=len(strategies))
        table.add_row(
            name,
            f"{r.total_return:.1%}",
            f"{r.cagr:.1%}",
            f"{r.sharpe:.2f}",
            f"{r.max_drawdown:.1%}",
            f"{r.n_trades:,}",
            f"${r.total_costs:,.0f}",
            f"{r.dsr:.2f}",
        )
    console.print(table)
    console.print(
        "\n[dim]If nothing beats buy_and_hold after costs, that is the finding.\n"
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
def invest(
    strategy: str = typer.Option("rrl", help="Only 'rrl' is implemented"),
    capital: float = typer.Option(100_000.0, help="Simulated starting capital (first run only)"),
    universe: str | None = typer.Option(
        None,
        help="sp500 | nasdaq100 | sp100 | config | watchlist file (default: config paper.universe)",
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

    from swingbot.paper.dashboard import build_paper_dashboard
    from swingbot.paper.engine import PaperEngine

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
) -> None:
    """Walk-forward evaluation of the cross-sectional excess-return ranker.

    Trains LightGBM on excess total return vs the benchmark with purged,
    embargoed expanding windows, then reports per-date RankIC. This is the
    go/no-go gate for the ranker: mean IC >= 0.02 with stability > 0.15 or
    nothing gets built on top of it.
    """
    from swingbot.agents.ranker import ic_summary, rank_ic, walk_forward_scores
    from swingbot.features.cross_section import build_panel
    from swingbot.paper.gate import gate_signal, health_index
    from swingbot.paper.universe import resolve_universe

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

    verdict = s["mean"] >= 0.02 and s["stability"] > 0.15
    color = "green" if verdict else "yellow"
    console.print(
        f"[{color}]sanity gate {'PASSED' if verdict else 'NOT met'}: "
        f"need mean IC >= 0.02 and stability > 0.15 before building on this signal[/{color}]"
    )


def _build_agent(strategy: str, cols: list[str], train: pl.DataFrame, cfg: Config, start: str):
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


if __name__ == "__main__":
    app()
