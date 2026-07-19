# swingbot

Autonomous paper-trading bot, live at **https://aqiao-814.github.io/swingbot-live/**

$100k simulated capital · NASDAQ-100 · hourly bars · inception 2026-07-17.
**Every dollar is simulated** — there are no brokerage credentials and no code
path that can place a real order. Research history and measured results:
[docs/FINDINGS.md](docs/FINDINGS.md).

## How it works — plain English

- On every completed hourly bar of the trading day, the bot looks at every
  NASDAQ-100 stock and scores how strongly its model wants to own it
  ("conviction").
- It sells anything it has lost conviction in, or that has fallen too far below
  what it paid (a stop-loss sized to each stock's own volatility).
- It buys the highest-conviction stocks — at most 10 positions, at most 20% of
  the portfolio in one name, and at least 10% held back as cash.
- Orders queue and execute at the **next bar's open** (an hour later; overnight
  for the day's last bar), with realistic trading costs (spread, slippage,
  price impact, regulatory fees).
- After every bar it also **learns**: each stock's realized return nudges the
  model's weights, so the policy adapts continuously.
- Safety: if it loses too much (in a day, from the peak, or over a month) or its
  scores degenerate, a kill switch sells everything and halts until a human
  intervenes.
- The dashboard shows it all live during market hours: portfolio value, cash,
  every position's P&L, and the full buy/sell log.

## How it works — technical

**Policy.** Single linear RRL unit (Moody–Saffell direct reinforcement):
`f_t = tanh(w·x_t + u·f_{t-1} + b)` over 19 trailing-only features (z-scored
multi-horizon returns, realized + Garman-Klass vol, RSI, MACD, Bollinger
position, MA distances, volume z/ratio, ATR%, fractional differencing), all
computed per bar. Weights are shared across the universe; each symbol keeps its
own recurrent state `(F_{t-1}, ∂F/∂θ)`. Reward is the differential Sharpe ratio
of net return `F_{t-1}·r_t − cost·|F_t − F_{t-1}|` — costs live inside the
gradient. L2 plus a hard `‖w‖ ≤ 1` cap resist tanh saturation. Pretrained on
~1y of pre-inception hourly history; one online update per (symbol, bar).

**Bar loop** (`paper/engine.py`, interval-agnostic — `1d` or `60m` via
`paper.interval` — idempotent via a `last_processed` watermark; only bars whose
completion time has passed ever enter it):

1. **Fill** pending orders at the bar's open through `ExecutionModel` —
   half-spread 1 bp, slippage 0.5 bp, square-root impact, SEC §31 + FINRA TAF
   on sells. Sells first, buys in conviction order, capped so cash stays ≥ 0.
2. **Mark** at the bar close: ledger row with equity, P&L, turnover, costs, and
   buy-and-hold SPY / QQQ / equal-weight benchmarks.
3. **Learn**: one RRL update per symbol from the realized bar return.
4. **Decide**: score the universe on the bar. Exit on conviction < 0.05 or a
   2σ·√(20-bar) vol-scaled stop below basis; enter needs conviction ≥ 0.15,
   top-10 slots. Target weight = f × 20%, gross scaled to ≤ 0.90 — each stop
   inside a 10-day re-entry cooldown lowers the cap by 0.10 (floor 0.30).
   5% no-trade band. Orders fill at the *next* bar's open.

**Kill switches** (flatten + halt until `invest --clear-halt`): single-bar loss
≥ 4%, drawdown ≥ 15%, rolling 20-bar loss ≥ 10%, or conviction σ < 0.05 (model
health — scores pinned at ±1 mean ranking has degenerated).

**No lookahead by construction**: decisions at close *t* can only fill at open
*t+1*; features are trailing-only (tests corrupt future bars and assert earlier
features are bit-identical); a pure-noise churn test must lose money.

## Deployment (all free)

- **`.github/workflows/trade.yml`** — hourly during market hours (13:40–21:40
  UTC weekdays): restores state, processes every newly completed hourly bar,
  exports `data.json`, publishes state + site to
  [swingbot-live](https://github.com/aqiao-814/swingbot-live) (GitHub Pages).
- **`.github/workflows/live.yml`** — every 20 min during market hours: live
  quotes → `live.json` (live P&L between trading runs).
- Portfolio state persists in the public repo under `state/`; bar data lives in
  an Actions cache. Manual run: `gh workflow run trade.yml` (`clear_halt=true`
  to clear a fired kill switch).

## Local use

```bash
make test                                  # 168 tests
make invest                                # run the daily loop locally
python -m swingbot.cli invest --clear-halt # resume after a kill switch
```

## Layout

```
src/swingbot/paper/    the live loop: engine, continual RRL, state, dashboard
src/swingbot/          portfolio accounting, execution costs, features, data store
src/swingbot/{env,backtest,agents}/  research harness (see docs/FINDINGS.md)
scripts/               site data + live quote exporters
site/                  the hosted dashboard (single static page)
```

## Non-goals

No real-money trading, ever. Pointing this at live capital is a decision made
outside this repo.
