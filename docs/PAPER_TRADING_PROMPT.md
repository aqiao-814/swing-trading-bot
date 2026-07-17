# Build prompt: daily paper-trading loop + portfolio dashboard

Copy everything below the line and give it to Claude (in this repo) to implement.

---

## Task

Add a **forward paper-trading loop** to swingbot. It should run once per trading
day, pull the latest real market data, have a chosen strategy decide, book the
resulting trade into a **persistent simulated portfolio**, and regenerate a
**portfolio dashboard** showing performance to date. No real money, no broker
order path — this is paper trading only. Reuse the existing accounting and
friction engine; do not reinvent it.

## Why (context)

The backtest showed the trained strategy loses to buy-and-hold, so the point of
this loop is honest forward tracking: prove out (or disprove) a strategy on
unseen, real, going-forward data before anyone would ever consider real capital.

## Accuracy requirements

- Use **real daily bars from `YahooSource`** (the existing `yahoo` source). This
  is free end-of-day data, not live intraday — that is the accuracy ceiling and
  it is acceptable.
- **Decision timing:** decide on the most recent completed daily bar (yesterday's
  close), **fill at the next available open**, so there is no look-ahead. If the
  loop runs before the next open exists yet, mark the decision pending and fill
  on the next run.
- **Apply the existing frictions** in `execution.py` (spread, slippage, sqrt
  market impact, commission, borrow). Costs must be booked exactly as in
  backtests — no free trades.
- Keep the money **simulated and clearly labelled** everywhere in the UI, exactly
  like the existing dashboards ("simulated capital").

## What to build

1. **Persistent paper-portfolio state.** A new artifact dir, e.g.
   `artifacts/paper/<name>/`, holding an append-only ledger of daily marks and
   trades (Parquet + a small `state.json`). Each daily run appends one row; state
   survives across runs. Include: date, price used, action, position, cash,
   equity, daily P&L, cumulative P&L, costs. Reuse `portfolio.py` for the
   accounting and `reporting.py` conventions for artifact layout.

2. **A new CLI command** `paper` in `src/swingbot/cli.py`:
   ```
   python -m swingbot.cli paper --symbol AAPL --strategy rrl --capital 100000 [--open]
   ```
   Behaviour:
   - Fetch/refresh the latest bars for `--symbol` via the yahoo source into the
     BarStore (incremental — only new bars).
   - Load or initialize the paper portfolio for `(symbol, strategy)`.
   - Advance the simulation to the newest completed bar: for each not-yet-processed
     trading day, build the same feature observation the backtest uses
     (`build_dataset` / `feature_columns`), call `agent.act(obs)`, and apply the
     fill with frictions at next open.
   - Persist the updated ledger/state.
   - Regenerate a **portfolio dashboard** HTML (see #3).
   - Idempotent: running twice the same day must not double-book.

3. **Portfolio dashboard** (`dashboard.py`, extend or add `build_portfolio_dashboard`):
   a self-contained HTML (no CDN, matching existing style, light/dark) showing:
   equity curve since inception, cumulative & daily P&L, current position and
   cash, drawdown, cumulative costs, and a table of recent daily marks and trades.
   Header must state it is simulated/paper. Write to
   `artifacts/paper/<symbol>_<strategy>/dashboard.html`.

4. **Tests** in `tests/` matching the repo's style (pytest): idempotency (double
   run same day is a no-op), no look-ahead (fill price is next open, never same
   bar's close), costs are booked, ledger append is monotonic in date, and state
   round-trips through save/load. Keep the whole suite green.

## Constraints / house style

- Match existing interfaces: `Agent.act(obs)`, `BarSource`, `BarStore`,
  `Config`. Do not change their contracts.
- No new heavy deps. Polars, not pandas. Keep dashboards CDN-free.
- Everything reproducible from config + seed, consistent with the rest of the repo.
- Update `README.md` (this is the "forward paper-trading loop" the README lists as
  the next phase) and add a short `docs/PAPER_TRADING.md`.

## Then

Show me the `paper` command output for AAPL with the `rrl` strategy and open the
resulting portfolio dashboard. Explain what the first day's numbers mean.
