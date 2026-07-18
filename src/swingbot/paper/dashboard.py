"""Self-contained HTML dashboard for the persistent paper portfolio.

Same design system as the backtest dashboard (no CDN, no network, both themes,
direct-labeled series, tables as the accessible complement to every chart),
with the sections a *forward* track record needs: holdings, trade history,
daily decisions, learning progress, and benchmark comparison.

Every figure on the page is SIMULATED CAPITAL and the page says so.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from swingbot.metrics import (
    cagr,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from swingbot.metrics import (
    drawdown_series as _dd,
)
from swingbot.paper.state import PaperState, PaperStore

_BENCH_LABELS = {"SPY": "Buy & Hold S&P 500", "QQQ": "Buy & Hold Nasdaq", "EW": "Equal Weight"}


def _series(ledger: pl.DataFrame, col: str, start: float) -> list[float]:
    vals = [start, *ledger[col].fill_null(strategy="forward").to_list()]
    return [round(float(v or 0.0), 2) for v in vals]


def _bench_metrics(equity: np.ndarray) -> dict:
    equity = equity[equity > 0]
    if len(equity) < 2:
        return {"total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    rets = np.diff(equity) / equity[:-1]
    return {
        "total_return": float(equity[-1] / equity[0] - 1.0),
        "cagr": cagr(equity),
        "sharpe": sharpe_ratio(rets),
        "max_drawdown": max_drawdown(equity),
    }


def _monthly_returns(ledger: pl.DataFrame) -> list[dict]:
    if ledger.is_empty():
        return []
    monthly = (
        ledger.with_columns(pl.col("ts").dt.strftime("%Y-%m").alias("month"))
        .group_by("month", maintain_order=True)
        .agg(((pl.col("daily_return") + 1.0).product() - 1.0).alias("ret"))
    )
    return [{"month": m, "ret": round(float(r), 6)} for m, r in monthly.iter_rows()]


def build_paper_dashboard(paper_root: str | Path, out: str | Path | None = None) -> Path:
    """Render the paper-portfolio dashboard from persisted state. Returns path."""
    root = Path(paper_root)
    store = PaperStore(root)
    state = PaperState.load(store.state_path)
    out = Path(out) if out else root / "dashboard.html"

    def _sorted(name: str) -> pl.DataFrame:
        # Day one: a table with no rows yet reads back column-less; sorting it
        # would raise, and inception day has no trades by construction.
        df = store.read(name)
        return df.sort("ts") if "ts" in df.columns else df

    ledger = _sorted("ledger")
    trades = _sorted("trades")
    decisions = _sorted("decisions")
    learning = _sorted("learning")
    positions = store.read("positions")

    cap = state.starting_capital
    equity = (
        np.array([cap, *ledger["equity"].to_list()]) if not ledger.is_empty() else np.array([cap])
    )
    rets = ledger["daily_return"].to_numpy() if not ledger.is_empty() else np.array([])

    # ---- headline metrics -------------------------------------------------
    sells = trades.filter(pl.col("quantity") < 0) if not trades.is_empty() else pl.DataFrame()
    pnls = sells["realized_pnl"].to_numpy() if not sells.is_empty() else np.array([])
    wins, losses = pnls[pnls > 0], pnls[pnls < 0]
    explicit = float(ledger["explicit_costs"][-1]) if not ledger.is_empty() else 0.0
    slippage = float(ledger["slippage_costs"][-1]) if not ledger.is_empty() else 0.0

    metrics = {
        "equity": float(equity[-1]),
        "cash": state.cash,
        "invested": float(ledger["invested"][-1]) if not ledger.is_empty() else 0.0,
        "daily_return": float(rets[-1]) if len(rets) else 0.0,
        "total_return": float(equity[-1] / cap - 1.0),
        "annualized_return": cagr(equity),
        "sharpe": sharpe_ratio(rets),
        "sortino": min(sortino_ratio(rets), 99.0),
        "max_drawdown": max_drawdown(equity),
        "volatility": float(np.std(rets, ddof=1) * np.sqrt(252)) if len(rets) > 1 else 0.0,
        "win_rate": float(np.mean(pnls > 0)) if len(pnls) else 0.0,
        "avg_gain": float(np.mean(wins)) if len(wins) else 0.0,
        "avg_loss": float(np.mean(losses)) if len(losses) else 0.0,
        "profit_factor": (
            min(float(wins.sum() / -losses.sum()), 99.0) if losses.sum() < 0 else 0.0
        ),
        "explicit_costs": explicit,
        "slippage_costs": slippage,
        "total_costs": explicit + slippage,
        "n_fills": state.n_fills,
        "turnover": (float(ledger["turnover"].mean() * 252) if not ledger.is_empty() else 0.0),
    }

    # ---- chart series -------------------------------------------------------
    dates = [state.inception or str(date.today())] + (
        [str(t) for t in ledger["ts"].to_list()] if not ledger.is_empty() else []
    )
    bench_cols = (
        [c for c in ledger.columns if c.startswith("bench_")] if not ledger.is_empty() else []
    )
    benches = {}
    for col in bench_cols:
        name = col.removeprefix("bench_")
        series = _series(ledger, col, cap)
        benches[name] = {
            "label": _BENCH_LABELS.get(name, name),
            "series": series,
            "metrics": _bench_metrics(np.array(series)),
        }

    payload = {
        "meta": {
            "universe": state.universe,
            "inception": state.inception,
            "lastProcessed": state.last_processed,
            "startingCapital": cap,
            "nFills": state.n_fills,
        },
        "metrics": metrics,
        "dates": dates,
        "equity": [round(float(v), 2) for v in equity],
        "drawdown": [round(float(v) * 100, 2) for v in _dd(equity)],
        "dailyPnl": ([round(float(v), 2) for v in np.diff(equity)] if len(equity) > 1 else []),
        "benchmarks": benches,
        "monthly": _monthly_returns(ledger),
        "holdings": positions.to_dicts() if not positions.is_empty() else [],
        "trades": trades.sort("ts", descending=True).head(300).to_dicts()
        if not trades.is_empty()
        else [],
        "decisions": decisions.sort("ts", descending=True).head(400).to_dicts()
        if not decisions.is_empty()
        else [],
        "learning": {
            "dates": [str(t) for t in learning["ts"].to_list()] if not learning.is_empty() else [],
            "cumReward": learning["cum_reward"].to_list() if not learning.is_empty() else [],
            "dayReward": learning["day_mean_reward"].to_list() if not learning.is_empty() else [],
            "policyLoss": learning["policy_loss"].to_list() if not learning.is_empty() else [],
            "ewSharpe": learning["ew_sharpe"].to_list() if not learning.is_empty() else [],
            "weightNorm": learning["weight_norm"].to_list() if not learning.is_empty() else [],
            "nUpdates": int(learning["n_updates"][-1]) if not learning.is_empty() else 0,
        },
    }

    html = _TEMPLATE.replace("__DATA__", json.dumps(payload, default=str))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>swingbot — paper portfolio</title>
<style>
  :root {
    color-scheme: light dark;
    --surface-0: #f6f6f4; --surface-1: #fcfcfb; --border: #e2e1dc;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #85837c;
    --grid: #ebeae6; --good: #0ca30c; --warning: #b98300; --critical: #d03b3b;
    --accent: #2a78d6;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      --surface-0: #111110; --surface-1: #1a1a19; --border: #35342f;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8d8b80;
      --grid: #2a2a27; --good: #21c55d; --warning: #fab219; --critical: #f26d6d;
      --accent: #3987e5;
    }
  }
  :root[data-theme="dark"] {
    --surface-0: #111110; --surface-1: #1a1a19; --border: #35342f;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8d8b80;
    --grid: #2a2a27; --good: #21c55d; --warning: #fab219; --critical: #f26d6d;
    --accent: #3987e5;
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 32px 24px 64px; background: var(--surface-0); color: var(--text-primary);
    font: 14px/1.5 ui-sans-serif, -apple-system, "Segoe UI", system-ui, sans-serif; }
  .wrap { max-width: 1120px; margin: 0 auto; }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
  h2 { font-size: 14px; font-weight: 600; margin: 0 0 2px; }
  .sub { color: var(--text-secondary); font-size: 13px; }
  .sim-badge { display: inline-block; margin-left: 8px; padding: 2px 8px; border-radius: 999px;
    background: color-mix(in srgb, var(--warning) 18%, transparent); font-size: 11px; font-weight: 600;
    border: 1px solid color-mix(in srgb, var(--warning) 45%, transparent); }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 24px 0; }
  .tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
  .tile .label { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--text-muted); margin-bottom: 4px; }
  .tile .value { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .tile .note { font-size: 12px; color: var(--text-secondary); margin-top: 2px; }
  .pos { color: var(--good); } .neg { color: var(--critical); }
  .card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 18px; margin-bottom: 20px; }
  .card .hint { font-size: 12px; color: var(--text-secondary); margin-bottom: 12px; }
  .legend { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 8px; font-size: 12px; color: var(--text-secondary); }
  .legend .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; }
  .chart { position: relative; overflow-x: auto; }
  svg { display: block; width: 100%; height: auto; }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
  th, td { padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--border); white-space: nowrap; }
  thead th { color: var(--text-muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }
  th:first-child, td:first-child { text-align: left; }
  tbody tr:hover { background: var(--surface-0); }
  .scroll { overflow-x: auto; max-height: 420px; overflow-y: auto; }
  .pill { padding: 1px 7px; border-radius: 999px; font-size: 11px; font-weight: 600; }
  .pill.buy { background: color-mix(in srgb, var(--good) 15%, transparent); color: var(--good); }
  .pill.sell, .pill.stop_loss { background: color-mix(in srgb, var(--critical) 14%, transparent); color: var(--critical); }
  .pill.hold, .pill.rebalance { background: color-mix(in srgb, var(--accent) 13%, transparent); color: var(--accent); }
  .toggle { position: fixed; top: 14px; right: 16px; background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 8px; padding: 6px 10px; cursor: pointer; font: inherit; font-size: 12px; color: var(--text-secondary); }
  .tooltip { position: absolute; pointer-events: none; opacity: 0; transition: opacity .1s; background: var(--surface-1);
    border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 12px;
    box-shadow: 0 4px 16px rgb(0 0 0 / .14); min-width: 150px; z-index: 5; }
  .tooltip .tt-date { color: var(--text-muted); margin-bottom: 4px; font-size: 11px; }
  .tooltip .tt-row { display: flex; gap: 10px; justify-content: space-between; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 860px) { .grid2 { grid-template-columns: 1fr; } }
  .note-block { background: color-mix(in srgb, var(--warning) 10%, var(--surface-1));
    border: 1px solid color-mix(in srgb, var(--warning) 35%, transparent);
    border-radius: 10px; padding: 12px 16px; font-size: 13px; color: var(--text-secondary); margin-bottom: 20px; }
</style>
</head>
<body>
<button class="toggle" id="themeToggle">◐ theme</button>
<div class="wrap">
  <header>
    <h1>Paper portfolio <span class="sim-badge">SIMULATED CAPITAL</span></h1>
    <div class="sub" id="subline"></div>
  </header>

  <div class="tiles" id="tiles"></div>

  <div class="card">
    <h2>Equity vs benchmarks</h2>
    <div class="hint">Portfolio equity against buy-and-hold benchmarks bought at inception with the same simulated capital. All fills pay spread, slippage, impact and fees, one bar after each decision.</div>
    <div class="legend" id="eqLegend"></div>
    <div class="chart" id="equityChart"></div>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>Drawdown</h2>
      <div class="hint">Decline from the portfolio's running peak.</div>
      <div class="chart" id="ddChart"></div>
    </div>
    <div class="card">
      <h2>Daily P&amp;L</h2>
      <div class="hint">Dollar change in equity per trading day.</div>
      <div class="chart" id="pnlChart"></div>
    </div>
  </div>

  <div class="card">
    <h2>Benchmark comparison</h2>
    <div class="hint">Same period, same starting capital. The bar to clear.</div>
    <div class="scroll" id="benchTable"></div>
  </div>

  <div class="card">
    <h2>Monthly returns</h2>
    <div class="scroll" id="monthlyTable"></div>
  </div>

  <div class="card">
    <h2>Current holdings</h2>
    <div class="scroll" id="holdingsTable"></div>
  </div>

  <div class="card">
    <h2>Daily decisions</h2>
    <div class="hint">What the agent chose each day, with conviction, predicted reward, allocation, and the realized next-day return once known. Choosing nothing is a valid decision.</div>
    <div class="scroll" id="decisionsTable"></div>
  </div>

  <div class="card">
    <h2>Trade history</h2>
    <div class="hint">Most recent fills first. Slippage is the cost of executing away from the reference open.</div>
    <div class="scroll" id="tradesTable"></div>
  </div>

  <div class="card">
    <h2>Learning progress</h2>
    <div class="hint">Continual RRL: every symbol's realized daily return is one online update to the shared policy. Policy loss is the negated mean differential-Sharpe reward; the policy is deterministic, so there is no exploration-rate schedule.</div>
    <div class="legend" id="learnLegend"></div>
    <div class="grid2">
      <div class="chart" id="rewardChart"></div>
      <div class="chart" id="sharpeChart"></div>
    </div>
    <div class="scroll" id="learnStats" style="margin-top:12px"></div>
  </div>

  <div class="note-block"><strong>Every dollar on this page is simulated.</strong>
    Decisions use only the latest completed daily bar; orders fill at the next day's open through the
    full execution model (spread, slippage, square-root market impact, commissions, fees, borrow).
    No order was ever placed with a broker.</div>
</div>

<script>
const D = __DATA__;
const isDark = () => (document.documentElement.dataset.theme || "")
  ? document.documentElement.dataset.theme === "dark"
  : matchMedia("(prefers-color-scheme: dark)").matches;
const COLORS = {
  portfolio: ["#2a78d6", "#3987e5"], SPY: ["#008300", "#1fae4d"],
  QQQ: ["#e87ba4", "#d55181"], EW: ["#eda100", "#c98500"],
};
const colorOf = k => (COLORS[k] || COLORS.EW)[isDark() ? 1 : 0];
const fmtMoney = v => (v < 0 ? "-$" : "$") + Math.round(Math.abs(v)).toLocaleString();
const fmtPct = v => (v * 100).toFixed(2) + "%";
const fmtPct1 = v => (v * 100).toFixed(1) + "%";
const cls = v => v > 0 ? "pos" : v < 0 ? "neg" : "";

document.getElementById("subline").textContent =
  `Universe: ${D.meta.universe} · inception ${D.meta.inception} · last processed ${D.meta.lastProcessed}` +
  ` · starting capital $${Math.round(D.meta.startingCapital).toLocaleString()} · ${D.meta.nFills} fills`;

/* ---------- tiles ---------- */
const M = D.metrics;
const tiles = [
  ["Total equity", fmtMoney(M.equity), (M.total_return>=0?"+":"") + fmtPct(M.total_return) + " total", cls(M.total_return)],
  ["Cash", fmtMoney(M.cash), fmtPct1(M.cash / M.equity) + " of equity", ""],
  ["Invested", fmtMoney(M.invested), fmtPct1(M.invested / M.equity) + " of equity", ""],
  ["Daily return", fmtPct(M.daily_return), "latest processed day", cls(M.daily_return)],
  ["Annualized return", fmtPct(M.annualized_return), "CAGR since inception", cls(M.annualized_return)],
  ["Sharpe", M.sharpe.toFixed(2), "annualized, daily returns", ""],
  ["Sortino", M.sortino.toFixed(2), "downside-only risk", ""],
  ["Max drawdown", fmtPct(M.max_drawdown), "peak to trough", M.max_drawdown > 0.1 ? "neg" : ""],
  ["Volatility", fmtPct(M.volatility), "annualized", ""],
  ["Win rate", fmtPct1(M.win_rate), "closed fills with profit", ""],
  ["Avg gain / loss", fmtMoney(M.avg_gain) + " / " + fmtMoney(M.avg_loss), "per closing fill", ""],
  ["Profit factor", M.profit_factor.toFixed(2), "gross wins / gross losses", ""],
  ["Trading costs", fmtMoney(M.total_costs), fmtMoney(M.explicit_costs) + " explicit + " + fmtMoney(M.slippage_costs) + " slippage", ""],
  ["Turnover", M.turnover.toFixed(2) + "x", "annualized", ""],
];
document.getElementById("tiles").innerHTML = tiles.map(([l, v, n, c]) =>
  `<div class="tile"><div class="label">${l}</div><div class="value ${c}">${v}</div><div class="note">${n}</div></div>`).join("");

/* ---------- chart engine ---------- */
function lineChart(mount, seriesList, { fmt = fmtMoney, logScale = false, zeroLine = false } = {}) {
  const W = 1000, H = 300, Mg = { t: 16, r: 110, b: 28, l: 64 };
  const xs = D.dates;
  const n = xs.length;
  if (!n || !seriesList.length) { mount.innerHTML = "<div class='hint'>no data yet</div>"; return; }
  const all = seriesList.flatMap(s => s.data).filter(v => isFinite(v));
  let lo = Math.min(...all), hi = Math.max(...all);
  if (zeroLine) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  const pad = (hi - lo) * 0.07 || 1; hi += pad; lo -= pad;
  const tx = i => Mg.l + (i / Math.max(n - 1, 1)) * (W - Mg.l - Mg.r);
  const ty = v => Mg.t + (1 - (v - lo) / (hi - lo)) * (H - Mg.t - Mg.b);
  const niceStep = raw => { const m = Math.pow(10, Math.floor(Math.log10(Math.abs(raw) || 1)));
    const x = raw / m; return (x <= 1 ? 1 : x <= 2 ? 2 : x <= 5 ? 5 : 10) * m; };
  let grid = "", ylab = "";
  const step = niceStep((hi - lo) / 5);
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    const y = ty(v);
    grid += `<line x1="${Mg.l}" y1="${y}" x2="${W - Mg.r}" y2="${y}" stroke="var(--grid)"/>`;
    ylab += `<text x="${Mg.l - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="var(--text-muted)">${fmt(v)}</text>`;
  }
  let xlab = "";
  for (let i = 0; i < 5; i++) {
    const idx = Math.round((i / 4) * (n - 1));
    xlab += `<text x="${tx(idx)}" y="${H - 8}" text-anchor="middle" font-size="11" fill="var(--text-muted)">${String(xs[idx]).slice(0, 10)}</text>`;
  }
  let paths = "";
  seriesList.forEach(s => {
    const d = s.data.map((v, i) => `${i ? "L" : "M"}${tx(i).toFixed(1)},${ty(v).toFixed(1)}`).join("");
    paths += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round"/>`;
  });
  const GAP = 13;
  const ends = seriesList.map(s => ({ s, y: ty(s.data[s.data.length - 1]) })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i-1].y < GAP) ends[i].y = ends[i-1].y + GAP;
  const labels = ends.map(({ s, y }) =>
    `<text x="${W - Mg.r + 8}" y="${y + 4}" font-size="11" font-weight="600" fill="${s.color}">${s.name}</text>`).join("");
  const zero = zeroLine ? `<line x1="${Mg.l}" y1="${ty(0)}" x2="${W - Mg.r}" y2="${ty(0)}" stroke="var(--text-muted)" stroke-dasharray="2 3"/>` : "";
  mount.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img">${grid}${ylab}${xlab}${zero}
      <line x1="${Mg.l}" y1="${Mg.t}" x2="${Mg.l}" y2="${H - Mg.b}" stroke="var(--border)"/>
      ${paths}${labels}
      <line class="cross" y1="${Mg.t}" y2="${H - Mg.b}" stroke="var(--text-muted)" stroke-dasharray="3 3" opacity="0"/>
      <rect class="hit" x="${Mg.l}" y="${Mg.t}" width="${W - Mg.l - Mg.r}" height="${H - Mg.t - Mg.b}" fill="transparent"/>
    </svg><div class="tooltip"></div>`;
  const svg = mount.querySelector("svg"), tip = mount.querySelector(".tooltip"),
        cross = mount.querySelector(".cross"), hit = mount.querySelector(".hit");
  hit.addEventListener("pointermove", e => {
    const r = svg.getBoundingClientRect();
    let i = Math.round(((e.clientX - r.left) / r.width * W - Mg.l) / (W - Mg.l - Mg.r) * (n - 1));
    i = Math.max(0, Math.min(n - 1, i));
    cross.setAttribute("x1", tx(i)); cross.setAttribute("x2", tx(i)); cross.setAttribute("opacity", 1);
    tip.style.opacity = 1;
    tip.innerHTML = `<div class="tt-date">${xs[i]}</div>` + seriesList.map(s =>
      `<div class="tt-row"><span style="color:${s.color}">${s.name}</span><b>${fmt(s.data[i])}</b></div>`).join("");
    tip.style.left = Math.min(Math.max(e.clientX - r.left + 14, 0), r.width - 170) + "px";
    tip.style.top = Math.max(e.clientY - r.top - 10, 0) + "px";
  });
  hit.addEventListener("pointerleave", () => { tip.style.opacity = 0; cross.setAttribute("opacity", 0); });
}

function barChart(mount, values, dates) {
  const W = 1000, H = 260, Mg = { t: 14, r: 20, b: 28, l: 64 };
  if (!values.length) { mount.innerHTML = "<div class='hint'>no data yet</div>"; return; }
  let lo = Math.min(...values, 0), hi = Math.max(...values, 0);
  const pad = (hi - lo) * 0.08 || 1; hi += pad; lo -= pad;
  const n = values.length;
  const tx = i => Mg.l + (i / Math.max(n - 1, 1)) * (W - Mg.l - Mg.r);
  const ty = v => Mg.t + (1 - (v - lo) / (hi - lo)) * (H - Mg.t - Mg.b);
  const bw = Math.max((W - Mg.l - Mg.r) / n * 0.7, 1);
  let bars = "";
  values.forEach((v, i) => {
    const y0 = ty(0), y1 = ty(v);
    bars += `<rect x="${(tx(i) - bw / 2).toFixed(1)}" y="${Math.min(y0, y1).toFixed(1)}" width="${bw.toFixed(1)}"
      height="${Math.max(Math.abs(y0 - y1), 0.5).toFixed(1)}" fill="${v >= 0 ? "var(--good)" : "var(--critical)"}" opacity="0.85"/>`;
  });
  let xlab = "";
  for (let i = 0; i < 5; i++) {
    const idx = Math.round((i / 4) * (n - 1));
    xlab += `<text x="${tx(idx)}" y="${H - 8}" text-anchor="middle" font-size="11" fill="var(--text-muted)">${String(dates[idx]).slice(0, 10)}</text>`;
  }
  mount.innerHTML = `<svg viewBox="0 0 ${W} ${H}">
    <line x1="${Mg.l}" y1="${ty(0)}" x2="${W - Mg.r}" y2="${ty(0)}" stroke="var(--grid)"/>
    <text x="${Mg.l - 8}" y="${ty(0) + 4}" text-anchor="end" font-size="11" fill="var(--text-muted)">$0</text>
    ${bars}${xlab}</svg>`;
}

/* ---------- tables ---------- */
const table = (headers, rows) =>
  `<table><thead><tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>` +
  rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("") + "</tbody></table>";
const pill = a => `<span class="pill ${a}">${a}</span>`;

function renderTables() {
  document.getElementById("benchTable").innerHTML = table(
    ["Series", "Total return", "CAGR", "Sharpe", "Max drawdown", "Final equity", "vs portfolio"],
    [["<b>Paper portfolio</b>", `<span class="${cls(M.total_return)}">${fmtPct(M.total_return)}</span>`,
      fmtPct(M.annualized_return), M.sharpe.toFixed(2), fmtPct(M.max_drawdown), fmtMoney(M.equity), "—"],
     ...Object.entries(D.benchmarks).map(([k, b]) => {
       const rel = M.total_return - b.metrics.total_return;
       return [b.label, fmtPct(b.metrics.total_return), fmtPct(b.metrics.cagr), b.metrics.sharpe.toFixed(2),
         fmtPct(b.metrics.max_drawdown), fmtMoney(b.series[b.series.length - 1]),
         `<span class="${cls(rel)}">${rel >= 0 ? "+" : ""}${fmtPct(rel)}</span>`];
     })]);

  document.getElementById("monthlyTable").innerHTML = table(
    ["Month", "Return"],
    D.monthly.map(m => [m.month, `<span class="${cls(m.ret)}">${fmtPct(m.ret)}</span>`]).reverse());

  document.getElementById("holdingsTable").innerHTML = D.holdings.length ? table(
    ["Symbol", "Shares", "Cost basis", "Current price", "Market value", "Gain/Loss", "Weight", "Entered"],
    D.holdings.map(h => [
      `<b>${h.symbol}</b>`, h.quantity.toLocaleString(), "$" + h.avg_price.toFixed(2),
      "$" + h.current_price.toFixed(2), fmtMoney(h.market_value),
      `<span class="${cls(h.unrealized_pnl)}">${fmtMoney(h.unrealized_pnl)}</span>`,
      fmtPct1(h.weight), h.entry_ts || "—"]))
    : "<div class='hint'>No open positions — the agent is allowed to hold cash.</div>";

  document.getElementById("decisionsTable").innerHTML = D.decisions.length ? table(
    ["Date", "Symbol", "Action", "Conviction", "Reward prediction", "Allocation", "Result (next day)"],
    D.decisions.map(x => [x.ts, `<b>${x.symbol}</b>`, pill(x.action),
      x.conviction.toFixed(3), fmtPct(x.expected_reward), fmtPct1(x.allocation),
      x.result == null ? "pending" : `<span class="${cls(x.result)}">${fmtPct(x.result)}</span>`]))
    : "<div class='hint'>no decisions yet</div>";

  document.getElementById("tradesTable").innerHTML = D.trades.length ? table(
    ["Date", "Symbol", "Action", "Qty", "Fill", "Ref (open)", "Notional", "Fees+Comm", "Slippage", "Realized P&L", "Reason"],
    D.trades.map(t => [t.ts, `<b>${t.symbol}</b>`, pill(t.action), Math.abs(t.quantity).toLocaleString(),
      "$" + t.fill_price.toFixed(2), "$" + t.reference_price.toFixed(2), fmtMoney(t.notional),
      "$" + (t.commission + t.fees).toFixed(2), "$" + t.slippage.toFixed(2),
      `<span class="${cls(t.realized_pnl)}">${fmtMoney(t.realized_pnl)}</span>`, t.reason]))
    : "<div class='hint'>no fills yet</div>";

  const L = D.learning;
  const last = L.dates.length - 1;
  document.getElementById("learnStats").innerHTML = table(
    ["Training iterations", "Cumulative reward", "Mean reward (last day)", "Policy loss (last day)", "EW Sharpe", "Weight norm", "Exploration"],
    last >= 0 ? [[L.nUpdates.toLocaleString(), L.cumReward[last].toFixed(4), L.dayReward[last].toFixed(6),
      L.policyLoss[last].toFixed(6), L.ewSharpe[last].toFixed(4), L.weightNorm[last].toFixed(4),
      "n/a — deterministic policy"]] : []);
}

/* ---------- render ---------- */
function legend(el, items) {
  document.getElementById(el).innerHTML = items.map(([name, key]) =>
    `<span><span class="swatch" style="background:${colorOf(key)}"></span>${name}</span>`).join("");
}
function renderCharts() {
  const eqSeries = [{ name: "portfolio", key: "portfolio", color: colorOf("portfolio"), data: D.equity },
    ...Object.entries(D.benchmarks).map(([k, b]) => ({ name: k, key: k, color: colorOf(k), data: b.series }))];
  legend("eqLegend", [["Paper portfolio", "portfolio"], ...Object.entries(D.benchmarks).map(([k, b]) => [b.label, k])]);
  lineChart(document.getElementById("equityChart"), eqSeries);
  lineChart(document.getElementById("ddChart"),
    [{ name: "drawdown", color: colorOf("QQQ"), data: D.drawdown.map(v => -v) }],
    { fmt: v => v.toFixed(1) + "%", zeroLine: true });
  barChart(document.getElementById("pnlChart"), D.dailyPnl, D.dates.slice(1));
  const L = D.learning;
  if (L.dates.length) {
    const mk = (mount, data, name, fmt) =>
      lineChartOn(mount, L.dates, [{ name, color: colorOf("portfolio"), data }], fmt);
    mk(document.getElementById("rewardChart"), L.cumReward, "cumulative reward", v => v.toFixed(2));
    mk(document.getElementById("sharpeChart"), L.ewSharpe, "EW Sharpe", v => v.toFixed(3));
  }
  legend("learnLegend", [["Cumulative DSR reward (left) · policy EW Sharpe (right)", "portfolio"]]);
  renderTables();
}
/* small variant with its own x-axis (learning series) */
function lineChartOn(mount, xs, seriesList, fmt) {
  const saveDates = D.dates; D.dates = xs;
  lineChart(mount, seriesList, { fmt });
  D.dates = saveDates;
}

document.getElementById("themeToggle").onclick = () => {
  const cur = document.documentElement.dataset.theme
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.dataset.theme = cur === "dark" ? "light" : "dark";
  renderCharts();
};
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderCharts);
renderCharts();
</script>
</body>
</html>
"""
