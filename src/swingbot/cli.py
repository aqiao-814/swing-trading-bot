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
