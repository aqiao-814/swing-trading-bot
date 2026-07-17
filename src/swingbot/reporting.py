"""Artifact writing: trade logs, equity curves, run manifests.

Every run writes a self-describing directory: the exact config used, the metrics,
the full trade log, and the equity curve. Reproducibility means a result you can
re-derive six months later without remembering anything.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from swingbot.config import Config
from swingbot.env.trading_env import EpisodeResult
from swingbot.metrics import PerformanceReport, drawdown_series


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - absence of git is not an error
        return None


def trades_frame(result: EpisodeResult) -> pl.DataFrame:
    """Full decision log: every bar, not just the ones that traded.

    Keeping non-trading bars is deliberate -- 'why did it do nothing here?' is
    as important a question as 'why did it buy here?'.
    """
    if not result.trades:
        return pl.DataFrame()
    return pl.DataFrame([asdict(t) for t in result.trades])


def equity_frame(result: EpisodeResult) -> pl.DataFrame:
    equity = np.asarray(result.equity)
    n = len(equity)
    returns = np.concatenate([[0.0], result.returns]) if len(result.returns) else np.zeros(n)
    return pl.DataFrame(
        {
            "ts": result.timestamps[:n],
            "equity": equity,
            "returns": returns[:n],
            "position": np.asarray(result.positions)[:n],
            "drawdown": drawdown_series(equity),
        }
    )


def write_run(
    outdir: str | Path,
    *,
    config: Config,
    result: EpisodeResult,
    report: PerformanceReport,
    label: str = "run",
) -> Path:
    """Persist a complete, self-describing run directory."""
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)

    config.dump(path / "config.yaml")
    (path / "metrics.json").write_text(json.dumps(report.to_dict(), indent=2, default=str))

    trades = trades_frame(result)
    if not trades.is_empty():
        trades.write_parquet(path / "trades.parquet")
        trades.write_csv(path / "trades.csv")
    equity_frame(result).write_parquet(path / "equity.parquet")

    # The manifest is what makes a result re-derivable months later.
    manifest = {
        "label": label,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        "seed": config.seed,
        "starting_capital": config.env.starting_capital,
        "simulated_capital": True,  # this system never touches real money
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return path


def format_report(report: PerformanceReport, *, title: str = "Performance") -> str:
    """Human-readable metric block for the terminal."""
    r = report
    inf = float("inf")
    mtrl = "inf" if r.min_track_record == inf else f"{r.min_track_record:,.0f}"
    lines = [
        f"\n{title}",
        "=" * len(title),
        f"  Starting capital   ${r.starting_capital:>14,.2f}   (SIMULATED)",
        f"  Ending equity      ${r.ending_equity:>14,.2f}",
        f"  Total return       {r.total_return:>15.2%}",
        f"  CAGR               {r.cagr:>15.2%}",
        "",
        f"  Sharpe             {r.sharpe:>15.2f}",
        f"  Sortino            {'inf' if r.sortino == inf else f'{r.sortino:.2f}':>15}",
        f"  Calmar             {r.calmar:>15.2f}",
        f"  Annual vol         {r.annual_volatility:>15.2%}",
        "",
        f"  Max drawdown       {r.max_drawdown:>15.2%}",
        f"  Max DD duration    {r.max_drawdown_days:>13} bars",
        f"  VaR (95%)          {r.var_95:>15.2%}",
        f"  CVaR (95%)         {r.cvar_95:>15.2%}",
        "",
        f"  Trades             {r.n_trades:>15,}",
        f"  Win rate           {r.win_rate:>15.2%}",
        f"  Profit factor      {'inf' if r.profit_factor == inf else f'{r.profit_factor:.2f}':>15}",
        f"  Turnover (ann.)    {r.turnover:>15.2f}x",
        f"  Total costs        ${r.total_costs:>14,.2f}",
        f"  Cost drag          {r.cost_drag:>15.2%}",
        "",
        "  -- overfitting-aware --",
        f"  PSR                {r.psr:>15.2%}",
        f"  Deflated Sharpe    {r.dsr:>15.2%}",
        f"  Min track record   {mtrl:>15} bars",
    ]
    if r.halted_reason:
        lines.append(f"\n  !! HALTED: {r.halted_reason}")
    if r.dsr < 0.95:
        lines.append(
            "\n  NOTE: Deflated Sharpe < 95% -- this has not demonstrated skill\n"
            "        beyond what searching over configurations would produce."
        )
    return "\n".join(lines)
