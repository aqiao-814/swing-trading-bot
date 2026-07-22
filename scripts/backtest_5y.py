"""Five-year walk-forward backtest of the live paper-trading model.

Runs the *actual* ``PaperEngine`` -- the same continual-learning RRL policy the
hosted bot trades -- over the last five years of daily bars for the nasdaq100
universe. Because the engine learns online from every realized bar as it walks
forward, this run is simultaneously the evaluation ("how does the model
perform, year by year?") and the training ("let the model learn from those
results"): the checkpoint it leaves behind has lived five years of experience
and becomes the seed for the live 30-minute loop.

No look-ahead: the learner pretrains only on history strictly before inception,
then every forward decision uses trailing features and fills at the next open.
The live kill switches are disabled here on purpose -- they are a production
safety valve that would freeze a five-year measurement at the first drawdown
halt; we want to see the model's own multi-year behavior.

Outputs (under ``artifacts/backtest5y``):
  * ``paper/portfolio/*.parquet`` -- full ledger, trades, learning diagnostics
  * ``models/rrl_latest.bin``     -- the five-year-trained policy (deploy seed)
  * ``findings.json``             -- per-year + overall stats and chart series

Usage:
  PYTHONPATH=src ./.venv/bin/python scripts/backtest_5y.py [--fresh]
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from swingbot.config import Config
from swingbot.paper.engine import PaperEngine

ROOT = Path(__file__).resolve().parent.parent
BT_ROOT = ROOT / "artifacts" / "backtest5y"
YEARS = 5
INCEPTION = "2021-07-01"
DATA_START = "2018-01-01"
AS_OF = date(2026, 7, 17)  # last completed daily bar in the cached store
STARTING_CAPITAL = 100_000.0
BENCH_KEYS = {"bench_QQQ": "QQQ", "bench_SPY": "SPY", "bench_EW": "EW"}


def build_config() -> Config:
    """Daily-bar config for the research backtest, kill switches off."""
    cfg = Config()
    cfg.run_name = "backtest5y"
    cfg.artifacts_root = BT_ROOT
    cfg.data.root = ROOT / "data"
    cfg.data.source = "yahoo"
    p = cfg.paper
    p.interval = "1d"
    p.universe = "nasdaq100"
    p.data_start = DATA_START
    p.start = INCEPTION
    p.pretrain_years = 3.0
    p.benchmark_symbols = ["SPY", "QQQ"]
    # Disable the production kill switches for the measurement (see module docs).
    p.kill_max_drawdown = None
    p.kill_daily_loss = None
    p.kill_rolling_20d_loss = None
    p.kill_conviction_std = None
    return cfg


def run_backtest(fresh: bool) -> PaperEngine:
    if fresh and BT_ROOT.exists():
        shutil.rmtree(BT_ROOT)
    cfg = build_config()
    engine = PaperEngine(cfg)
    print(f"universe {engine.universe_name} ({len(engine.symbols)} symbols) · daily bars")
    engine.run(
        capital=STARTING_CAPITAL,
        as_of=AS_OF,
        refresh=False,  # use cached daily data; no network
        log=lambda m: print(f"  {m}"),
    )
    return engine


# ---- metrics ---------------------------------------------------------------


def _sharpe(daily_returns: np.ndarray) -> float:
    if daily_returns.size < 2:
        return 0.0
    sd = float(daily_returns.std(ddof=1))
    return float(daily_returns.mean() / sd * np.sqrt(252)) if sd > 1e-12 else 0.0


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def _series(values: np.ndarray, base: float) -> list[float]:
    """Rebase a value series to 100 at ``base``, rounded for a compact payload."""
    if base <= 0:
        base = values[0] if values.size else 1.0
    return [round(float(v) / base * 100.0, 3) for v in values]


def _window_stats(
    ts: list[str],
    equity: np.ndarray,
    rets: np.ndarray,
    benches: dict[str, np.ndarray],
    base_equity: float,
    base_bench: dict[str, float],
    n_trades: int,
    label: str,
) -> dict:
    total_ret = float(equity[-1] / base_equity - 1.0) if base_equity > 0 else 0.0
    bench_returns = {
        name: (float(vals[-1] / base_bench[key] - 1.0) if base_bench.get(key, 0) > 0 else None)
        for key, name in BENCH_KEYS.items()
        if (vals := benches.get(key)) is not None
    }
    return {
        "label": label,
        "total_return": total_ret,
        "sharpe": _sharpe(rets),
        "max_drawdown": _max_drawdown(np.concatenate([[base_equity], equity])),
        "ann_vol": float(rets.std(ddof=1) * np.sqrt(252)) if rets.size > 1 else 0.0,
        "n_trades": n_trades,
        "benchmarks": bench_returns,
        "excess_vs_qqq": (
            total_ret - bench_returns["QQQ"] if bench_returns.get("QQQ") is not None else None
        ),
        "series": {
            "ts": ts,
            "port": _series(equity, base_equity),
            **{name.lower(): _series(vals, base_bench[key]) for key, name in BENCH_KEYS.items()
               if (vals := benches.get(key)) is not None and base_bench.get(key, 0) > 0},
        },
    }


def build_findings(engine: PaperEngine) -> dict:
    port = engine.store.read("ledger").sort("ts")
    trades = engine.store.read("trades")
    learning = engine.store.read("learning").sort("ts")
    if port.is_empty():
        raise SystemExit("no ledger rows -- backtest produced nothing")

    ts = [str(t) for t in port["ts"].to_list()]
    years = [int(str(t)[:4]) for t in ts]
    equity = port["equity"].to_numpy().astype(float)
    rets = port["daily_return"].to_numpy().astype(float)
    bench_cols = {k: port[k].to_numpy().astype(float) for k in BENCH_KEYS if k in port.columns}

    # Trade counts per calendar year (a trade = one fill row).
    trade_years = (
        trades.with_columns(pl.col("ts").cast(pl.Utf8).str.slice(0, 4).alias("yr"))
        if not trades.is_empty()
        else pl.DataFrame({"yr": []})
    )
    trades_by_year = (
        dict(trade_years.group_by("yr").len().iter_rows()) if not trades.is_empty() else {}
    )
    n_trades_total = 0 if trades.is_empty() else trades.height

    # ---- overall (rebased to inception) ----
    overall = _window_stats(
        ts, equity, rets, bench_cols,
        base_equity=STARTING_CAPITAL,
        base_bench={k: STARTING_CAPITAL for k in bench_cols},
        n_trades=n_trades_total,
        label="Full backtest",
    )
    n = len(equity)
    overall["cagr"] = float((equity[-1] / STARTING_CAPITAL) ** (252.0 / max(n, 1)) - 1.0)
    overall["start"] = ts[0]
    overall["end"] = ts[-1]
    overall["n_days"] = n

    # ---- per-year windows ----
    year_reports = []
    for y in sorted(set(years)):
        idx = [i for i, yr in enumerate(years) if yr == y]
        first = idx[0]
        base_equity = float(equity[first - 1]) if first > 0 else STARTING_CAPITAL
        base_bench = {
            k: (float(v[first - 1]) if first > 0 else STARTING_CAPITAL)
            for k, v in bench_cols.items()
        }
        year_reports.append(
            _window_stats(
                [ts[i] for i in idx],
                equity[idx],
                rets[idx],
                {k: v[idx] for k, v in bench_cols.items()},
                base_equity,
                base_bench,
                trades_by_year.get(str(y), 0),
                label=str(y),
            )
            | {"year": y}
        )

    # ---- model-health diagnostics over the walk ----
    lr = learning
    health = {}
    if not lr.is_empty():
        health = {
            "n_updates": int(lr["n_updates"][-1]),
            "final_weight_norm": float(lr["weight_norm"][-1]),
            "final_ew_sharpe": float(lr["ew_sharpe"][-1]),
            "mean_conviction_std": float(lr["conviction_std"].mean()),
            "mean_frac_saturated": float(lr["frac_saturated"].mean()),
            "ts": [str(t) for t in lr["ts"].to_list()],
            "conviction_std": [round(float(v), 4) for v in lr["conviction_std"].to_list()],
        }

    return {
        "meta": {
            "universe": engine.universe_name,
            "n_symbols": len(engine.symbols),
            "interval": "1d",
            "inception": ts[0],
            "end": ts[-1],
            "starting_capital": STARTING_CAPITAL,
            "pretrain_years": engine.paper.pretrain_years,
            "kill_switches": "disabled (research measurement)",
            "model": "ContinualRRL (Moody & Saffell direct reinforcement, shared weights)",
        },
        "overall": overall,
        "years": year_reports,
        "health": health,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="wipe artifacts/backtest5y and rerun")
    ap.add_argument(
        "--skip-run", action="store_true", help="reuse an existing ledger; only rebuild findings"
    )
    args = ap.parse_args()

    if args.skip_run and (BT_ROOT / "paper" / "portfolio" / "ledger.parquet").exists():
        cfg = build_config()
        engine = PaperEngine(cfg)
    else:
        engine = run_backtest(fresh=args.fresh)

    findings = build_findings(engine)
    out = BT_ROOT / "findings.json"
    out.write_text(json.dumps(findings, separators=(",", ":"), allow_nan=False))

    o = findings["overall"]
    print("\n=== FIVE-YEAR BACKTEST ===")
    print(f"period {o['start']} .. {o['end']}  ({o['n_days']} trading days)")
    print(f"portfolio total return {o['total_return']:+.1%}  CAGR {o['cagr']:+.1%}  "
          f"Sharpe {o['sharpe']:.2f}  maxDD {o['max_drawdown']:.1%}  trades {o['n_trades']:,}")
    for name, r in o["benchmarks"].items():
        if r is not None:
            print(f"   buy&hold {name:3} {r:+.1%}   (excess {o['total_return'] - r:+.1%})")
    print(f"{'year':>6} {'return':>9} {'sharpe':>7} {'maxDD':>8} {'vs QQQ':>8} {'trades':>7}")
    for y in findings["years"]:
        vq = y["excess_vs_qqq"]
        print(f"{y['year']:>6} {y['total_return']:>+8.1%} {y['sharpe']:>7.2f} "
              f"{y['max_drawdown']:>+7.1%} {('' if vq is None else f'{vq:+.1%}'):>8} "
              f"{y['n_trades']:>7,}")
    h = findings["health"]
    if h:
        print(f"\nmodel health: {h['n_updates']:,} updates "
              f"· final |w| {h['final_weight_norm']:.3f} "
              f"· mean conviction σ {h['mean_conviction_std']:.3f} "
              f"· mean frac saturated {h['mean_frac_saturated']:.1%}")
    print(f"\nfindings -> {out}")
    print(f"trained model -> {BT_ROOT / 'models' / 'rrl_latest.bin'}")


if __name__ == "__main__":
    main()
