"""Self-contained HTML analytics dashboard.

Renders equity curves, underwater (drawdown) plots, cost attribution and the
full metric table into one file with no external dependencies -- no CDN, no
network, opens straight from disk.

Design notes (dataviz method):
* Categorical hues are assigned in fixed slot order and never cycled.
* Palette validated for both light and dark surfaces. Light mode trips the
  sub-3:1 contrast warning on three slots, which obligates *relief*: every
  series is direct-labeled at its line end AND a full table view is present.
* One y-axis per chart. Equity and drawdown are separate charts rather than a
  dual-axis hybrid.
* Dark mode is a selected set of steps for the dark surface, not an auto-flip.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from swingbot.env.trading_env import EpisodeResult
from swingbot.metrics import PerformanceReport, drawdown_series

# Categorical slots in fixed order (light, dark). Never cycled: a 6th strategy
# folds into "Other" rather than inventing a hue.
_SERIES_COLORS: list[tuple[str, str]] = [
    ("#2a78d6", "#3987e5"),  # blue
    ("#008300", "#008300"),  # green
    ("#e87ba4", "#d55181"),  # magenta
    ("#eda100", "#c98500"),  # yellow
    ("#1baf7a", "#199e70"),  # aqua
]
_MAX_SERIES = len(_SERIES_COLORS)


@dataclass
class StrategyResult:
    name: str
    result: EpisodeResult
    report: PerformanceReport


def _downsample(xs: list, ys: np.ndarray, max_points: int = 900) -> tuple[list, list]:
    """Thin a series for rendering. Keeps the last point so the label is exact."""
    n = len(ys)
    if n <= max_points:
        return xs, [round(float(v), 2) for v in ys]
    idx = np.unique(np.concatenate([np.linspace(0, n - 1, max_points).astype(int), [n - 1]]))
    return [xs[i] for i in idx], [round(float(ys[i]), 2) for i in idx]


def _series_payload(strategies: list[StrategyResult]) -> list[dict]:
    payload = []
    for i, s in enumerate(strategies[:_MAX_SERIES]):
        ts = [str(t) for t in s.result.timestamps[: len(s.result.equity)]]
        x, equity = _downsample(ts, np.asarray(s.result.equity))
        _, dd = _downsample(ts, drawdown_series(np.asarray(s.result.equity)) * 100.0)
        payload.append(
            {
                "name": s.name,
                "light": _SERIES_COLORS[i][0],
                "dark": _SERIES_COLORS[i][1],
                "x": x,
                "equity": equity,
                "drawdown": [round(v, 2) for v in dd],
                "metrics": s.report.to_dict(),
            }
        )
    return payload


def _fmt(v: float, kind: str) -> str:
    if v == float("inf"):
        return "∞"
    match kind:
        case "pct":
            return f"{v:.1%}"
        case "money":
            return f"${v:,.0f}"
        case "num":
            return f"{v:.2f}"
        case "int":
            return f"{v:,.0f}"
    return str(v)


_METRIC_ROWS = [
    ("Total return", "total_return", "pct"),
    ("CAGR", "cagr", "pct"),
    ("Sharpe", "sharpe", "num"),
    ("Sortino", "sortino", "num"),
    ("Calmar", "calmar", "num"),
    ("Annual vol", "annual_volatility", "pct"),
    ("Max drawdown", "max_drawdown", "pct"),
    ("Max DD (bars)", "max_drawdown_days", "int"),
    ("VaR 95%", "var_95", "pct"),
    ("CVaR 95%", "cvar_95", "pct"),
    ("Trades", "n_trades", "int"),
    ("Win rate", "win_rate", "pct"),
    ("Profit factor", "profit_factor", "num"),
    ("Turnover (ann.)", "turnover", "num"),
    ("Total costs", "total_costs", "money"),
    ("Cost drag", "cost_drag", "pct"),
    ("PSR", "psr", "pct"),
    ("Deflated Sharpe", "dsr", "pct"),
]


def _metrics_table(strategies: list[StrategyResult]) -> str:
    names = [s.name for s in strategies[:_MAX_SERIES]]
    head = "".join(f"<th>{n}</th>" for n in names)
    rows = []
    for label, key, kind in _METRIC_ROWS:
        cells = "".join(
            f"<td>{_fmt(float(s.report.to_dict()[key]), kind)}</td>"
            for s in strategies[:_MAX_SERIES]
        )
        rows.append(f"<tr><th scope='row'>{label}</th>{cells}</tr>")
    return (
        f"<table class='metrics'><thead><tr><th scope='col'>Metric</th>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def build_dashboard(
    strategies: list[StrategyResult],
    outpath: str | Path,
    *,
    symbol: str = "",
    period: str = "",
    starting_capital: float = 100_000.0,
) -> Path:
    """Write a self-contained HTML dashboard. Returns the path."""
    if not strategies:
        raise ValueError("no strategies to render")
    if len(strategies) > _MAX_SERIES:
        raise ValueError(
            f"{len(strategies)} series exceeds the {_MAX_SERIES}-slot categorical palette; "
            "fold the extras into 'Other' or facet into small multiples"
        )

    data = _series_payload(strategies)
    best = max(strategies, key=lambda s: s.report.total_return)
    bench = next((s for s in strategies if s.name == "buy_and_hold"), None)

    # The honest headline: did anything actually beat buy-and-hold?
    if bench and best.name != "buy_and_hold":
        verdict = f"{best.name} beat buy_and_hold"
        verdict_tone = "good"
    elif bench:
        losers = [s.name for s in strategies if s is not bench]
        verdict = f"buy_and_hold beat {', '.join(losers)}"
        verdict_tone = "warn"
    else:
        verdict = f"best: {best.name}"
        verdict_tone = "neutral"

    payload = json.dumps(
        {
            "series": data,
            "startingCapital": starting_capital,
            "symbol": symbol,
            "period": period,
        }
    )
    table = _metrics_table(strategies)

    html = _TEMPLATE.replace("__DATA__", payload)
    html = html.replace("__TABLE__", table)
    html = html.replace("__SYMBOL__", symbol or "—")
    html = html.replace("__PERIOD__", period or "—")
    html = html.replace("__VERDICT__", verdict)
    html = html.replace("__TONE__", verdict_tone)
    html = html.replace("__CAPITAL__", f"${starting_capital:,.0f}")

    path = Path(outpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>swingbot — __SYMBOL__ performance</title>
<style>
  :root {
    color-scheme: light dark;
    --surface-0: #f6f6f4;
    --surface-1: #fcfcfb;
    --border: #e2e1dc;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #85837c;
    --grid: #ebeae6;
    --good: #0ca30c;
    --warning: #fab219;
    --critical: #d03b3b;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      --surface-0: #111110;
      --surface-1: #1a1a19;
      --border: #35342f;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #8d8b80;
      --grid: #2a2a27;
      --good: #0ca30c;
      --warning: #fab219;
      --critical: #d03b3b;
    }
  }
  :root[data-theme="dark"] {
    --surface-0: #111110;
    --surface-1: #1a1a19;
    --border: #35342f;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #8d8b80;
    --grid: #2a2a27;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 32px 24px 64px;
    background: var(--surface-0); color: var(--text-primary);
    font: 14px/1.5 ui-sans-serif, -apple-system, "Segoe UI", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1080px; margin: 0 auto; }
  header { margin-bottom: 28px; }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }
  .sub { color: var(--text-secondary); font-size: 13px; }
  .sim-badge {
    display: inline-block; margin-left: 8px; padding: 2px 8px; border-radius: 999px;
    background: color-mix(in srgb, var(--warning) 18%, transparent);
    color: var(--text-primary); font-size: 11px; font-weight: 600;
    border: 1px solid color-mix(in srgb, var(--warning) 45%, transparent);
  }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 28px; }
  .tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
  .tile .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); margin-bottom: 6px; }
  .tile .value { font-size: 24px; font-weight: 600; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
  .tile .note { font-size: 12px; color: var(--text-secondary); margin-top: 4px; }
  .tone-good .value { color: var(--good); }
  .tone-warn .value { color: var(--warning); }
  .tone-critical .value { color: var(--critical); }
  .card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 18px; margin-bottom: 20px; }
  .card h2 { font-size: 14px; font-weight: 600; margin: 0 0 2px; }
  .card .hint { font-size: 12px; color: var(--text-secondary); margin-bottom: 14px; }
  .legend { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 10px; }
  .legend button {
    display: flex; align-items: center; gap: 6px; background: none; border: none; padding: 2px 4px;
    cursor: pointer; color: var(--text-secondary); font: inherit; font-size: 12px; border-radius: 6px;
  }
  .legend button[aria-pressed="false"] { opacity: 0.4; }
  .legend .swatch { width: 10px; height: 10px; border-radius: 3px; flex: none; }
  .chart { position: relative; overflow-x: auto; }
  svg { display: block; width: 100%; height: auto; }
  .tooltip {
    position: absolute; pointer-events: none; opacity: 0; transition: opacity .1s;
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
    padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgb(0 0 0 / 0.14);
    min-width: 150px; z-index: 5;
  }
  .tooltip .tt-date { color: var(--text-muted); margin-bottom: 5px; font-size: 11px; }
  .tooltip .tt-row { display: flex; align-items: center; gap: 6px; justify-content: space-between; }
  .tooltip .tt-name { display: flex; align-items: center; gap: 5px; color: var(--text-secondary); }
  .tooltip .tt-val { font-variant-numeric: tabular-nums; color: var(--text-primary); font-weight: 600; }
  table.metrics { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
  table.metrics th, table.metrics td { padding: 7px 10px; text-align: right; border-bottom: 1px solid var(--border); }
  table.metrics thead th { color: var(--text-muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
  table.metrics th[scope="row"] { text-align: left; font-weight: 400; color: var(--text-secondary); }
  table.metrics tbody tr:hover { background: var(--surface-0); }
  .note-block {
    background: color-mix(in srgb, var(--warning) 10%, var(--surface-1));
    border: 1px solid color-mix(in srgb, var(--warning) 35%, transparent);
    border-radius: 10px; padding: 14px 16px; font-size: 13px; color: var(--text-secondary);
  }
  .note-block strong { color: var(--text-primary); }
  .toggle { position: fixed; top: 14px; right: 16px; background: var(--surface-1);
    border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px; cursor: pointer;
    font: inherit; font-size: 12px; color: var(--text-secondary); }
  @media (max-width: 640px) { .tile .value { font-size: 20px; } body { padding: 20px 14px 48px; } }
</style>
</head>
<body>
<button class="toggle" id="themeToggle" aria-label="Toggle theme">◐ theme</button>
<div class="wrap">
  <header>
    <h1>__SYMBOL__ — strategy comparison <span class="sim-badge">SIMULATED CAPITAL</span></h1>
    <div class="sub">Out-of-sample __PERIOD__ · starting capital __CAPITAL__ · identical market conditions, costs and execution delay for every strategy</div>
  </header>

  <div class="tiles" id="tiles"></div>

  <div class="card">
    <h2>Equity curves</h2>
    <div class="hint">Log scale. Every strategy pays spread, slippage, impact and fees, and fills one bar after the decision.</div>
    <div class="legend" id="legend"></div>
    <div class="chart" id="equityChart"></div>
  </div>

  <div class="card">
    <h2>Underwater plot</h2>
    <div class="hint">Drawdown from each strategy's own running peak. This is the experience of holding it, which the return number hides.</div>
    <div class="chart" id="ddChart"></div>
  </div>

  <div class="card">
    <h2>All metrics</h2>
    <div class="hint">Table view — the accessible complement to the charts above.</div>
    <div style="overflow-x:auto">__TABLE__</div>
  </div>

  <div class="note-block" id="verdictNote"></div>
</div>

<script>
const DATA = __DATA__;
const isDark = () => (document.documentElement.dataset.theme || "")
  ? document.documentElement.dataset.theme === "dark"
  : matchMedia("(prefers-color-scheme: dark)").matches;
const colorOf = s => isDark() ? s.dark : s.light;
const hidden = new Set();

const fmtMoney = v => "$" + Math.round(v).toLocaleString();
const fmtPct = v => (v * 100).toFixed(1) + "%";

/* ---------- tiles ---------- */
function renderTiles() {
  const byRet = [...DATA.series].sort((a, b) => b.metrics.total_return - a.metrics.total_return);
  const best = byRet[0];
  const worst = byRet[byRet.length - 1];
  const totalCosts = DATA.series.reduce((s, x) => s + x.metrics.total_costs, 0);
  const bestDsr = Math.max(...DATA.series.map(s => s.metrics.dsr));
  const tiles = [
    { label: "Best out-of-sample", value: best.name, note: fmtPct(best.metrics.total_return) + " · Sharpe " + best.metrics.sharpe.toFixed(2), tone: "" },
    { label: "Worst", value: worst.name, note: fmtPct(worst.metrics.total_return) + " · " + fmtMoney(worst.metrics.total_costs) + " of costs", tone: "tone-critical" },
    { label: "Costs paid across all", value: fmtMoney(totalCosts), note: "frictions are not a rounding error", tone: "" },
    { label: "Best deflated Sharpe", value: (bestDsr * 100).toFixed(0) + "%", note: bestDsr < 0.95 ? "below 95% — consistent with luck" : "clears the 95% bar", tone: bestDsr < 0.95 ? "tone-warn" : "tone-good" },
  ];
  document.getElementById("tiles").innerHTML = tiles.map(t =>
    `<div class="tile ${t.tone}"><div class="label">${t.label}</div><div class="value">${t.value}</div><div class="note">${t.note}</div></div>`
  ).join("");
}

/* ---------- legend ---------- */
function renderLegend() {
  const el = document.getElementById("legend");
  el.innerHTML = DATA.series.map(s =>
    `<button data-name="${s.name}" aria-pressed="${!hidden.has(s.name)}">
       <span class="swatch" style="background:${colorOf(s)}"></span>${s.name}</button>`
  ).join("");
  el.querySelectorAll("button").forEach(b => b.onclick = () => {
    const n = b.dataset.name;
    hidden.has(n) ? hidden.delete(n) : hidden.add(n);
    renderAll();
  });
}

/* ---------- chart engine ---------- */
function lineChart(mount, key, { logScale = false, fmt = fmtMoney, zeroTop = false }) {
  const W = 1000, H = 300, M = { t: 16, r: 96, b: 28, l: 60 };
  const visible = DATA.series.filter(s => !hidden.has(s.name));
  if (!visible.length) { mount.innerHTML = ""; return; }

  const xs = visible[0].x;
  const n = xs.length;
  const all = visible.flatMap(s => s[key]);
  let lo = Math.min(...all), hi = Math.max(...all);
  if (zeroTop) { lo = 0; hi = Math.max(hi, 1); }
  if (logScale) { lo = Math.max(lo, 1); }
  const pad = (hi - lo) * 0.06 || 1;
  hi += pad; if (!zeroTop) lo -= pad;

  const tx = i => M.l + (i / (n - 1)) * (W - M.l - M.r);
  const ty = v => {
    if (logScale) {
      const [a, b, c] = [Math.log(Math.max(lo, 1)), Math.log(hi), Math.log(Math.max(v, 1))];
      return M.t + (1 - (c - a) / (b - a)) * (H - M.t - M.b);
    }
    return M.t + (1 - (v - lo) / (hi - lo)) * (H - M.t - M.b);
  };

  // Snap ticks to round numbers -- "$265,666" is noise; "$250,000" is a value
  // a reader can hold in their head.
  const niceStep = raw => {
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    return (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  };
  const ticks = 5;
  const tickVals = [];
  if (logScale) {
    for (let i = 0; i <= ticks; i++) {
      const v = Math.exp(Math.log(Math.max(lo, 1)) + (i / ticks) * (Math.log(hi) - Math.log(Math.max(lo, 1))));
      const r = niceStep(v);
      if (r >= lo && r <= hi && !tickVals.includes(r)) tickVals.push(r);
    }
  } else {
    const step = niceStep((hi - lo) / ticks);
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) tickVals.push(v);
  }
  let grid = "", ylab = "";
  tickVals.forEach(v => {
    const y = ty(v);
    grid += `<line x1="${M.l}" y1="${y}" x2="${W - M.r}" y2="${y}" stroke="var(--grid)" stroke-width="1"/>`;
    ylab += `<text x="${M.l - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="var(--text-muted)">${fmt(v)}</text>`;
  });
  let xlab = "";
  for (let i = 0; i < 5; i++) {
    const idx = Math.round((i / 4) * (n - 1));
    xlab += `<text x="${tx(idx)}" y="${H - 8}" text-anchor="middle" font-size="11" fill="var(--text-muted)">${xs[idx].slice(0, 7)}</text>`;
  }

  // 2px lines; direct labels at the line end (relief for the light-mode
  // contrast warning, and it removes the legend round-trip entirely).
  let paths = "";
  visible.forEach(s => {
    const d = s[key].map((v, i) => `${i ? "L" : "M"}${tx(i).toFixed(1)},${ty(v).toFixed(1)}`).join("");
    paths += `<path d="${d}" fill="none" stroke="${colorOf(s)}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
  });

  // De-collide the end labels: series that finish close together would render
  // on top of each other and become unreadable. Push them apart, preserving
  // vertical order so each label still points at the right line.
  const LABEL_GAP = 13;
  const ends = visible
    .map(s => ({ s, y: ty(s[key][s[key].length - 1]) }))
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) {
    if (ends[i].y - ends[i - 1].y < LABEL_GAP) ends[i].y = ends[i - 1].y + LABEL_GAP;
  }
  // If the pile-up pushed past the bottom, walk back upward from the floor.
  const floor = H - M.b;
  if (ends.length && ends[ends.length - 1].y > floor) {
    ends[ends.length - 1].y = floor;
    for (let i = ends.length - 2; i >= 0; i--) {
      if (ends[i + 1].y - ends[i].y < LABEL_GAP) ends[i].y = ends[i + 1].y - LABEL_GAP;
    }
  }
  const labels = ends.map(({ s, y }) =>
    `<text x="${W - M.r + 8}" y="${y + 4}" font-size="11" font-weight="600" fill="${colorOf(s)}">${s.name}</text>`
  ).join("");

  mount.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${key} by strategy">
      ${grid}${ylab}${xlab}
      <line x1="${M.l}" y1="${M.t}" x2="${M.l}" y2="${H - M.b}" stroke="var(--border)"/>
      ${paths}${labels}
      <line id="ch-${key}" y1="${M.t}" y2="${H - M.b}" stroke="var(--text-muted)" stroke-width="1" stroke-dasharray="3 3" opacity="0"/>
      <rect x="${M.l}" y="${M.t}" width="${W - M.l - M.r}" height="${H - M.t - M.b}" fill="transparent" id="hit-${key}"/>
    </svg>
    <div class="tooltip" id="tip-${key}"></div>`;

  // Crosshair + tooltip: an HTML chart is interactive by default.
  const svg = mount.querySelector("svg");
  const hit = mount.querySelector(`#hit-${key}`);
  const tip = mount.querySelector(`#tip-${key}`);
  const cross = mount.querySelector(`#ch-${key}`);
  hit.addEventListener("pointermove", e => {
    const r = svg.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width * W;
    let i = Math.round((px - M.l) / (W - M.l - M.r) * (n - 1));
    i = Math.max(0, Math.min(n - 1, i));
    cross.setAttribute("x1", tx(i)); cross.setAttribute("x2", tx(i));
    cross.setAttribute("opacity", "1");
    tip.style.opacity = "1";
    tip.innerHTML = `<div class="tt-date">${xs[i]}</div>` + visible.map(s =>
      `<div class="tt-row"><span class="tt-name"><span class="swatch" style="width:8px;height:8px;border-radius:2px;background:${colorOf(s)}"></span>${s.name}</span><span class="tt-val">${fmt(s[key][i])}</span></div>`
    ).join("");
    const left = Math.min(Math.max(e.clientX - r.left + 14, 0), r.width - 170);
    tip.style.left = left + "px";
    tip.style.top = Math.max(e.clientY - r.top - 10, 0) + "px";
  });
  hit.addEventListener("pointerleave", () => { tip.style.opacity = "0"; cross.setAttribute("opacity", "0"); });
}

function renderAll() {
  renderLegend();
  lineChart(document.getElementById("equityChart"), "equity", { logScale: true, fmt: fmtMoney });
  lineChart(document.getElementById("ddChart"), "drawdown", { fmt: v => v.toFixed(0) + "%", zeroTop: true });
}

document.getElementById("verdictNote").innerHTML =
  `<strong>Verdict: __VERDICT__.</strong> A deflated Sharpe below 95% means the result is
   still consistent with having searched over configurations rather than having found an edge.
   Costs shown are all-in: commission and fees plus the slippage embedded in fill prices.
   Every dollar here is simulated — no order was ever placed.`;

document.getElementById("themeToggle").onclick = () => {
  const cur = document.documentElement.dataset.theme
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.dataset.theme = cur === "dark" ? "light" : "dark";
  renderAll();
};
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);

renderTiles();
renderAll();
</script>
</body>
</html>
"""
