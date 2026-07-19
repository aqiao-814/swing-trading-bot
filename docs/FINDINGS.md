# Findings & experiments

The research record behind the live bot. Numbers here are measured, not
re-derived; treat them as the honest baseline any future work must beat.

## 1. Backtest baseline (AAPL 2016–2025, trained pre-2016, $100k)

| strategy | total ret | CAGR | Sharpe | max DD | trades | costs | DSR |
|---|---:|---:|---:|---:|---:|---:|---:|
| buy_and_hold | 150.4% | 9.6% | 1.14 | 10.8% | 206 | $413 | 0.99 |
| flat | 0.0% | 0.0% | 0.00 | 0.0% | 0 | $0 | 0.00 |
| random | −25.5% | −4.7% | −0.65 | 25.7% | 1,037 | $5,704 | 0.00 |
| ma crossover | −15.8% | −3.6% | −0.40 | 25.4% | 190 | $961 | 0.02 |
| rrl (trained) | 92.7% | 6.8% | 0.83 | 17.9% | 330 | $1,632 | 0.92 |

**RRL loses to buy-and-hold**; DSR 0.92 is below the 0.95 bar, consistent with
selection over configurations. Matches the literature (Millea 2021). `random`
burned 5.7% of capital on frictions alone — costs are not a rounding error.
`excess_sharpe` (vs buy-and-hold) was added because deflating a long-only book
against zero asks the wrong question in a bull market; rrl xSharpe +0.34
(single symbol, single seed — a caveat, not a result).

## 2. Diagnosis of the first live book (2026-07, real parquet history)

- **Zero rank signal**: per-date RankIC of the RRL scorer, mean −0.005
  (t = −0.15).
- **Saturated convictions**: 80% of scores > 0.96 — "conviction-ranked" sizing
  had degenerated into the sort's alphabetical tiebreak.
- **Stop churn**: the fixed 10% stop caused 26 of 33 position closes (~1.2σ on
  high-vol names — a coin flip that loses 10%).
- **Fee bug**: sell fees 10,000× too small (double bps conversion).

## 3. Fixes shipped

- Correct SEC (0.278 bp) + FINRA TAF fees, with a *magnitude*-asserting test.
- Vol-scaled stops (2σ of each name's 20-day horizon vol) replacing fixed 10%.
- Stop discipline: 10-day re-entry cooldown; each stop de-grosses the book by
  0.10 (floor 0.30) so freed cash stays cash.
- Saturation guards: L2 + hard ‖w‖ ≤ 1 cap; saturation metrics logged daily.
- Kill switches incl. conviction-σ **model-health** halt (fires on the live
  book by design; resume via `invest --clear-halt`).
- `artifacts/trials.jsonl`: one line per evaluated configuration, so DSR's
  `n_trials` cannot be undercounted.

## 4. Cross-sectional ranker experiment (LightGBM) — gate NOT met

Target: 20-day excess total return vs QQQ (mean-zero — can't win by always
saying yes). Walk-forward, purged + embargoed, per-date Spearman RankIC.
Pre-registered sanity gate: mean IC ≥ 0.02 **and** stability > 0.15.

- NDX 2019–2026, 7-feature panel: **mean IC +0.0044, stability 0.033 — FAIL.**
  Real in 2020 (+0.040), negative by 2024–2026.
- Realized-efficacy regime gate came out **inverted**: gate-on days IC −0.014
  vs gate-off +0.025 — trailing IC anti-predicts on a near-zero signal.
- Shuffle null (targets permuted within date): IC −0.006 → pipeline leak-free.

Consequence: **neither ranker nor gate is wired into the live loop.**

## 5. Deliberately not built

Per the roadmap's own sequencing after the failed gate: portfolio construction
(skfolio/HRP), jump models, sentiment, RL sizing. Building sizing on a signal
that fails its sanity gate is optimizing noise.

**Next, if resumed**: improve the base signal first (regime × momentum
interactions, EDGAR fundamentals); the 0.02/0.15 gate stands, and the null
result is an acceptable destination.

## 6. Validation methodology

Purged/combinatorial CV with embargo (plain k-fold leaks through overlapping
labels), Deflated Sharpe + PBO with honest `n_trials`, structural no-lookahead
tests (future-corruption bit-identity; no-free-money-on-noise), and paired
nulls for the cross-sectional pipeline.

## 7. Known limitations

- **Survivorship bias**: the universe is today's constituents; delisted names
  are missing, so every backtest here is optimistically biased. No free
  delisting-inclusive dataset exists.
- **Data**: Yahoo via yfinance (unofficial, rate-limited; ANSS is delisted and
  404s harmlessly). Stooq is behind a JS proof-of-work wall as of 2026-07 and
  is deliberately not circumvented.
- Single-symbol backtest results (§1) do not transfer to the portfolio loop;
  the live track record is the only forward evidence.
