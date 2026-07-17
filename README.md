# swingbot

A simulated-capital swing-trading research system: deep-RL-ready environment,
broker-grade accounting, realistic frictions, and overfitting-aware validation.

**Every dollar is simulated. The system has no brokerage credentials, no order
endpoint, and no code path that could place a real trade.** The agent is never
told the money is fake — observations carry position and P&L state only — so a
policy behaves exactly as it would with real capital.

## Status

Phases 1–5 and 7 of the blueprint are built and tested (**142 tests**). Data,
environment, frictions, features, baselines/RRL, metrics, validation and the
analytics dashboard are complete — and the **forward paper-trading loop is now
live**: an autonomous daily agent that explores a whole stock universe, sizes
positions by conviction, executes simulated fills through the full friction
model, and updates its RRL policy from every realized return
([docs/PAPER_TRADING.md](docs/PAPER_TRADING.md),
[docs/CONTINUAL_RRL.md](docs/CONTINUAL_RRL.md)). Deep-RL agents (PPO/SAC via
SB3, offline CQL/IQL via d3rlpy) are the next phase — the interfaces they need
(`Agent`, `SwingTradingEnv`, `BarSource`) are already in place.

## Quick start

```bash
make test                 # 142 tests
make fetch                # download bars into data/
make compare              # baselines vs RRL, out-of-sample
```

```bash
./.venv/bin/python -m swingbot.cli backtest --symbol AAPL --capital 50000 --strategy rrl
./.venv/bin/python -m swingbot.cli dashboard --symbol AAPL --open
```

`dashboard` writes one self-contained HTML file (no CDN, no network): equity
curves, underwater plot and the full metric table, in light and dark.

### Autonomous paper investing

```bash
./.venv/bin/python -m swingbot.cli invest --strategy rrl --capital 100000 --universe nasdaq100
```

Run it once per day (or whenever — it is idempotent). Each run refreshes market
data, scans every stock in the universe, ranks opportunities by conviction,
queues orders that fill at the **next day's open**, learns from every realized
return, checkpoints the model to `artifacts/models/`, and maintains one
persistent simulated portfolio under `artifacts/paper/` with a dashboard at
`artifacts/paper/dashboard.html`. Full details in
[docs/PAPER_TRADING.md](docs/PAPER_TRADING.md).

**Location matters.** This project must live **outside iCloud**. It was moved
from `~/Desktop` to `~/dev/swing-trading-bot` because iCloud silently breaks
editable installs — it re-hides `.pth` files and CPython skips hidden ones. See
[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md), which also covers the
Rosetta-emulated default Python that disables MPS.

### Cross-sectional ranker research

```bash
./.venv/bin/python -m swingbot.cli rank --universe nasdaq100 --horizon 20
```

Walk-forward evaluation of a LightGBM ranker on **20-day excess total return
vs QQQ** — a mean-zero target a model cannot beat by saying "yes" to
everything. Training windows are purged and embargoed so no label overlaps a
prediction date; the metric is per-date Spearman RankIC. The command enforces
its own go/no-go: mean IC ≥ 0.02 with stability > 0.15, or nothing gets built
on top of the signal.

Measured on this repo's cached NDX data (2019–2026, seven-feature baseline):
**mean IC +0.0044, stability 0.033 — the gate is NOT met.** The signal was
real in 2020 (+0.040) and decayed to negative by 2024–2026. A realized-efficacy
regime gate (`paper/gate.py`) was evaluated on the same series and came out
*inverted* — trailing IC anti-predicts future IC on a signal this weak. Both
results are the honest baseline the feature work in the research roadmap has
to beat; neither component is wired into the paper loop.

## What it actually found

Out-of-sample AAPL, 2016–2025 (trained pre-2016), $100k simulated:

| strategy | total ret | CAGR | Sharpe | max DD | trades | costs | DSR |
|---|---:|---:|---:|---:|---:|---:|---:|
| buy_and_hold | 150.4% | 9.6% | 1.14 | 10.8% | 206 | $413 | 0.99 |
| flat | 0.0% | 0.0% | 0.00 | 0.0% | 0 | $0 | 0.00 |
| random | -25.5% | -4.7% | -0.65 | 25.7% | 1,037 | $5,704 | 0.00 |
| ma (crossover) | -15.8% | -3.6% | -0.40 | 25.4% | 190 | $961 | 0.02 |
| rrl (trained) | 92.7% | 6.8% | 0.83 | 17.9% | 330 | $1,632 | 0.92 |

**RRL loses to buy-and-hold.** It clears random and MA-crossover comfortably, and
its Deflated Sharpe (0.92) is below the 0.95 bar, meaning the result is still
consistent with having searched over configurations. This matches the literature
exactly (Millea 2021: *"no decent profitability level was obtained"*). It is the
expected result, not a bug — and per the blueprint's Stage 0, it is the signal to
keep the bar high rather than to reach for a bigger network.

Note the cost column: `random` burned **$5,704** (5.7% of capital) purely on
frictions. Costs are not a rounding error at swing-trading turnover.

## Architecture

```
src/swingbot/
  config.py          typed config; a run is reproducible from YAML + seed
  portfolio.py       cash, positions, cost basis, realized/unrealized P&L
  execution.py       spread, slippage, sqrt impact, commission, borrow
  rewards.py         differential Sharpe (Moody & Saffell), drawdown-penalized
  metrics.py         Sharpe/Sortino/Calmar + PSR, Deflated Sharpe, MinTRL
  reporting.py       run artifacts: config, metrics, trade log, equity curve
  dashboard.py       self-contained HTML analytics (validated palette)
  cli.py             fetch / backtest / compare / dashboard / invest
  data/              schema + point-in-time adjustment, DuckDB/Parquet store
  features/          fracdiff (AFML Ch.5), causal technicals
  env/               the Gymnasium POMDP
  agents/            buy&hold, flat, random, MA crossover, RRL
  backtest/          runner + purged/combinatorial CV, PBO
  paper/             autonomous daily loop: universe scan, conviction-ranked
                     allocation, next-open fills, continual RRL, dashboard
```

### The three guarantees worth knowing

1. **Execution delay is structural.** A decision on bar *t*'s close fills at bar
   *t+1*'s **open**. `step()` advances the clock *before* it executes, so filling
   on the decision bar is impossible rather than merely discouraged.
   (`test_fills_at_next_open_not_current_close`)

2. **No feature can see the future.** `test_no_lookahead_bias` corrupts all bars
   after a cutoff and asserts every feature before it is bit-identical. That
   catches centred windows, full-sample normalisation, and forward fills at once.

3. **No free money on noise.** `test_no_free_money_on_pure_noise` churns a
   driftless random walk with costs and asserts a loss. A profit there would mean
   the simulator leaks the future.

### Realism modelled

Spread, slippage, square-root (Almgren-Chriss) impact, commission, SEC Section
31 + FINRA TAF fees on sells, overnight short borrow, execution delay,
**overnight gap risk** (a stop that gaps fills at the open, not the stop price),
stop-loss/take-profit, vol targeting, fractional Kelly, a **no-trade band**, and
portfolio kill switches (max drawdown, daily loss).

The paper loop adds **stop-loss discipline**: a stopped-out symbol sits in a
re-entry cooldown (`stop_cooldown_days`), and each stop inside the window lowers
the gross-exposure cap (`stop_degross_per_stop`, floored at
`min_gross_exposure`) — so a stop converts risk into cash instead of rotating
into the next correlated name.

Costs are reported all-in: `explicit_costs` (cash debits) **plus**
`slippage_costs` (embedded in fill prices). Counting only the former understates
true cost — an early bug here reported `$0` on a trade that cost real money, and
a later one shrank sell fees 10,000× by double-applying a bps conversion, which
is why `test_sell_fees_have_realistic_magnitude` asserts a *magnitude*, not just
`> 0`.

## Data

Default source is **Yahoo via yfinance** (no key, ~31 years, gives both raw and
adjusted close, which point-in-time adjustment needs). Unofficial and
rate-limited — fetch once, then read from the Parquet store.

**Stooq — which the research blueprint recommended as the best free bulk source —
is no longer scriptable.** As of 2026-07 every endpoint returns a JavaScript
proof-of-work anti-bot challenge instead of CSV. That challenge exists to block
automated clients, so this project does not defeat it; `get_source("stooq")`
raises with an explanation. Download the bulk ZIP manually and use
`source="csv"` if you want Stooq data.

`SyntheticSource` generates deterministic GBM/regime data for tests and for the
null hypothesis. Its seeding uses blake2b, **not** builtin `hash()` — the latter
is salted per process and made "reproducible" data differ on every run.

### Known limitation: survivorship bias

The universe is today's listed tickers, so delisted and bankrupt names are
missing and returns are biased optimistically. No free delisting-inclusive
dataset exists. This is a real ceiling on what any backtest here can prove.

## Validation

Use `PurgedKFold` and `CombinatorialPurgedCV` (purge + embargo) — never plain
k-fold, which leaks through overlapping labels and serial correlation. CPCV
yields a *distribution* of Sharpes across many paths rather than one number.

`deflated_sharpe_ratio` and `probability_of_backtest_overfitting` correct for
selection bias. **Report DSR with an honest `n_trials`.** Passing `n_trials=1`
after testing hundreds of configurations is how people fool themselves.

## Code intelligence (graphify)

The codebase is indexed as a knowledge graph in `graphify-out/`, which answers
structural questions without reading whole files:

```bash
graphify query "how does the environment prevent look-ahead bias" --budget 800
graphify explain "ExecutionModel"
graphify affected "Portfolio.execute()" --depth 2   # who breaks if I change this
graphify update .                                   # re-index after edits
```

The CLI is not on PATH: it lives at `~/.local/bin/graphify`.

## Non-goals

No real-money trading, ever. No brokerage integration beyond paper endpoints. If
this system is ever pointed at live capital, that is a decision made outside this
repo — and per the blueprint, only after a quarter-plus of paper trading across
multiple regimes, with hard kill switches and tiny size.
