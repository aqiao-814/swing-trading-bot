# Forward paper trading

The `invest` command runs an autonomous, RL-driven paper-investment loop over a
stock universe. It maintains **one persistent simulated portfolio** that lives
across runs, learns from every completed trading day, and is benchmarked
against buy-and-hold.

**Every dollar is simulated.** The state file carries a `simulated_capital`
flag that is asserted on load, the dashboard is badged `SIMULATED CAPITAL`, and
there is no code path that could place a real order.

## Running it

```bash
python -m swingbot.cli invest \
    --strategy rrl \
    --capital 100000 \
    --universe nasdaq100        # or sp500 | sp100 | config | a watchlist file
```

Each run:

1. **Refreshes market data** for the whole universe plus benchmarks (SPY, QQQ)
   through the Yahoo source. Universe-scale refreshes go through
   `YahooBulkSource`, which batches ~50 tickers per request instead of making
   hundreds of serial calls.
2. **Finds every completed trading day not yet processed** and replays them in
   order. Running twice on the same day is a no-op — `state.json` carries a
   `last_processed` watermark, so trades and training can never be duplicated.
3. For each day: fills pending orders at the **open**, marks to market at the
   **close**, feeds every realized return to the learner
   ([CONTINUAL_RRL.md](CONTINUAL_RRL.md)), scores the whole universe, ranks by
   conviction, and queues orders for the next open.
4. Checkpoints the model, saves state, and regenerates the dashboard.

The first run establishes *inception*. With `paper.start` set to a past date
(or `--start`), the engine replays forward from there day by day — decisions at
each replayed day use only bars up to that day, so the resulting track record
is genuinely forward, not fitted.

## Timing contract (no look-ahead, ever)

- Decisions use **only the latest completed daily bar**. Before ~16:15 ET the
  engine treats today's bar as still forming and decides off yesterday's.
- A decision made on bar *t* becomes a **pending order** persisted in
  `state.json`. It fills at bar *t+1*'s open — in a later run, or the next
  replayed day. Filling on the decision bar is structurally impossible.
- All features are trailing-only (see `features/technical.py`), and the tests
  assert that appending future bars changes no past decision bit-for-bit.

## Execution and accounting

Fills go through the **same** `ExecutionModel` and `Portfolio` as the
backtester: half-spread, slippage, square-root market impact, commissions,
SEC fees on sells, and overnight borrow on shorts. Buys are shrunk until cash
covers the full cost, so simulated cash can never go negative. Sells execute
before buys each morning.

## Capital allocation

Position size is proportional to conviction: a name with policy output `f`
targets `f × max_position_weight` of equity (default cap 20%), the whole book
is scaled to stay within `max_gross_exposure` (default 90%), and at most
`max_positions` names are held (default 10). Nothing forces the system to be
invested — a day where no stock clears `min_conviction` is a valid decision,
and the remainder is cash. Exits happen on conviction decay
(`exit_conviction`), a close-to-close stop-loss on cost basis, or a target
rebalance beyond the no-trade band.

## Artifacts

```
artifacts/paper/
    dashboard.html          self-contained dashboard (no CDN, light+dark)
    portfolio/
        state.json          cash, positions, pending orders, watermark
        ledger.parquet      daily equity, P&L, costs, turnover, benchmarks
        trades.parquet      every fill with slippage/fees/realized P&L
        decisions.parquet   every decision with conviction, allocation,
                            reward prediction, and realized next-day result
        positions.parquet   current holdings snapshot
        learning.parquet    per-day learning metrics
artifacts/models/
    rrl_latest.bin          the live policy (npz payload)
    checkpoints/rrl_<date>.bin
    manifest.json
```

## Benchmarks

The ledger tracks buy-and-hold SPY, buy-and-hold QQQ, and an equal-weight
basket of the universe, all "bought" at inception with the same simulated
capital. The dashboard's benchmark table shows relative performance — that is
the honest headline, not the raw return.

## Reproducibility

Universe resolution is sorted/deduplicated, ranking ties break by symbol,
the learner is seeded from `config.seed`, and every iteration order is
deterministic — so the same config, seed, and bar data reproduce the same
portfolio to the last cent (tested).

## Known limitations

- **Adjusted-price boundary drift.** Incremental refreshes fetch a 30-day
  overlap, so a split/dividend inside that window re-adjusts cleanly; a split
  older than the cache boundary would need a full refetch (delete
  `data/bars/symbol=X`).
- **Survivorship bias.** The universes are static snapshots of current
  constituents; delisted names are absent, which flatters the equal-weight
  benchmark and the opportunity set alike.
- Universe index membership is a 2025 snapshot; the odd stale ticker degrades
  coverage gracefully (fetch failures are tolerated) rather than breaking runs.
