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

## 8. Five-year portfolio backtest & 30-minute migration (2026-07)

Ran the *actual* `PaperEngine` (continual-learning RRL, the live policy) over the
last five years of daily nasdaq100 bars — `scripts/backtest_5y.py`, kill switches
off so a safety halt doesn't freeze the measurement. It is both evaluation and
training: the checkpoint it leaves becomes the seed for the live loop.

- **2021-07 → 2026-07: +142.2% total (CAGR 19.3%, Sharpe 0.92, maxDD −35.1%),
  vs QQQ +102.1% / SPY +85.8% / equal-weight +100.7%.** Beats the benchmarks —
  but read it as **beta + survivorship**, not alpha: today's-constituents
  universe, long-only in a bull market, a drawdown deeper than the index, mixed
  years (2025 lagged QQQ by −22.6%; 2022 down −11.9% but beat QQQ's ~−33%), and
  Sharpe below 1. Consistent with the near-zero RankIC of §2/§4. Charts +
  full write-up: `scripts/build_findings_page.py` → self-contained HTML tearsheet.
- **Recurrence saturation, diagnosed and fixed.** The backtest exposed *why*
  convictions saturate (§2): the recurrent weight drifts to **u > 1**, making
  `F_t = tanh(w·x + u·F_{t-1} + b)` explosive — every conviction pins to ±1 in a
  few bars. The `‖w‖` cap never watched `u`. Added `RRLAgent.max_recurrence`
  (config `paper.learn_max_recurrence`), a hard `|u| ≤ cap`. Live 30m config uses
  0.7, which restores a healthy cross-sectional spread (σ ≈ 0.25 on the full
  universe) so the model-health kill switch is satisfied.
- **30-minute live loop.** The engine is now interval-agnostic over `1d | 60m |
  30m` (13 bars/session). Yahoo serves only ~60 days of 30m bars — too little to
  pretrain — so inception **seeds** from the offline 5-year model
  (`_seed_or_new_learner`): load weights, clear the daily recurrence, temper `u`
  into the cap, then refine on the ~60 days of 30m history. `configs/cloud.yaml`
  shortens the long-memory feature windows (z-score 252→100, fracdiff 1e-4→1e-2,
  warmup 300→60) so features survive ~520 bars *without renaming any column*, so
  the seed still matches. `trade.yml` fires every 30 min from the open. **Caveat:
  daily→30m transfer is a hypothesis** — features live on a different time scale;
  the forward 30m record is the only real test.

## 9. 30m deployment verification and pre-launch replays (2026-07-21 night)

Deployed the 30m loop to the cloud (merged to `main`; site re-incepted Tue
2026-07-21 15:30 ET, day one Wed 2026-07-22). Verified end to end: seeded cloud
inception ("tempering seed recurrence u 1.038 -> ±0.7", 152,652 prior updates,
10 queued ~9% entries, conviction σ 0.289 / frac_saturated 0) and the
state-restore path (idempotent no-op, nothing republished). Two walk-forward
replays of the *live config* on real 30m bars, kill switches ON:

- **5-week replay (incept 2026-06-16):** only ~3 weeks of pretrain data fit
  inside Yahoo's 60-day window, the under-refined model's conviction spread
  came up σ 0.043 < 0.05 and the **model-health kill switch flattened the book
  two hours in**; it sat in cash five weeks (+0.12% vs QQQ −3.41% — survival
  by abstention, not signal). Lesson: the seed *needs* the full ~40-session
  refinement window; a mid-history 30m inception is structurally handicapped,
  and the halt-on-degenerate-spread guard works live.
- **2-week replay (incept 2026-07-06, full pretrain window — the fair
  rehearsal):** traded all 11 sessions, **no halt**, −1.18% absolute in a
  falling tape vs QQQ −2.19% / EW −1.85% / SPY −0.53%. Beat its own universe's
  beta, lost less than the index it draws from, lost to cash. Read: the loop
  functions and de-risks; returns remain beta-dominated; 11 sessions is
  evidence of *function*, not alpha. No pre-launch tuning done off this sample
  (§4's overfitting discipline applies). The forward paper record remains the
  pre-registered test.
- **Learning-direction audit (does it actually learn from mistakes?).** On the
  deployed model's real state (183k updates, pooled moments a≈−6.6e−4,
  var≈1.4e−5), 30 consecutive losing bars on a setup cut its conviction
  **+0.80 → +0.29** while 30 winning bars raised it to +0.97 — losses are
  corrective in the live regime. Caveat found on the way: the differential
  Sharpe has the known negative-mean pathology — `dD/dR = (B − A·R)/var^1.5`
  flips sign when a bar's reward drops below `B/A` with the pooled mean
  negative (measured threshold: a per-bar position-weighted loss worse than
  ~−2.2%), where gradient ascent would *amplify* rather than correct. A cold
  model fed only losses reproduces it (conviction rose +0.05 → +0.90 in a
  synthetic all-loss regime). Live exposure is bounded: moments pool across
  ~100 symbols so the mean sits near zero, per-bar position-weighted rewards
  are typically |r| < 0.005, and the 4%-bar / 10%-per-20-bars kill switches
  flatten the book in exactly the sustained-loss regime where the pathology
  lives.

## 10. Day-one live result + signal-refinement attempt (2026-07-22)

**Day one traded and closed red.** The 30m loop filled all 10 seeded orders at
Wed 2026-07-22's 09:30 ET open and held them to the 15:30 close: **−1.37%**
(equity $98,628.66, entirely unrealized, 0 intraday closes) vs QQQ −0.44% / SPY
−0.12% / equal-weight −0.85%. It lagged every benchmark, including the EW of its
own universe. Cron fired green all session, never halted; learning stayed
healthy (n_updates 184,652, conviction σ 0.262, grad_norm 0.836,
frac_saturated 0.15). So *starts-sharp / cron-reliable / learns-from-mistakes*
are confirmed with forward evidence; **profitable is not** — one down day is not
a track record, and the loss is consistent with §2/§4's near-zero measured
signal.

**Refinement attempt (the "run backtests to refine and learn" ask).** Rebuilt
the cross-sectional panel with theory-motivated additions — short-term reversal
(`rev_5d`), 52-week-high proximity (`close/rolling_max_252`), vol-scaled
momentum (`mom_3m/vol_20d`), log-ADV, and cross-sectional z-scoring within date
(the correct transform for a mean-zero relative target). Same purged/embargoed
walk-forward harness, same NDX 2019–2026 panel, seed 7:

| panel | mean IC | stability | t | gate (≥0.02 / >0.15) |
|---|---:|---:|---:|:--|
| baseline 7-feature (§4) | +0.0043 | 0.032 | +1.26 | FAIL |
| improved, raw | +0.0074 | 0.050 | +2.01 | FAIL |
| improved, z-scored | +0.0052 | 0.033 | +1.33 | FAIL |

Leak-free (shuffle null −0.0002). The improvement is real but small and **still
fails the go/no-go bar by ~3×.** The by-year decomposition is decisive: the edge
is a **2020–2021 phenomenon** (IC +0.050 / +0.036) that is flat-to-negative
every year since (2022 −0.038, 2023 −0.007, 2024 −0.001, 2025 +0.007, 2026
−0.018). There is no cross-sectional edge in the current regime on this
liquid-megacap universe with price/volume features. Hunting a feature set that
happens to cross 0.02 on this exact sample would be the overfitting §4/§5
explicitly forbid, and it would not produce forward profit anyway. **Honest
conclusion:** reliable profitability is not deliverable from more
price-feature engineering here; it needs a genuinely different information
source (fundamentals/EDGAR, alt-data) — the §5 "improve the base signal first"
directive stands, now with a second validated refutation behind it.
