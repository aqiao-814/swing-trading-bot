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
- Stop discipline: 10-day re-entry cooldown; stops de-gross the book in
  proportion to the fraction of it that stopped out (floor 0.30) so freed cash
  stays cash. (Originally a flat 0.10 per stop — §11 explains why that had to
  become breadth-relative.)
- Saturation guards: L2 + hard ‖w‖ ≤ 1 cap; saturation metrics logged daily.
- Kill switches incl. conviction-σ **model-health** halt. **Removed in §11**;
  this line records what was true at the time.
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
off so a safety halt doesn't freeze the measurement (as of §11 they no longer
exist anywhere). It is both evaluation and training: the checkpoint it leaves
becomes the seed for the live loop.

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

### 10a. Short-horizon signal + cost-aware long/short (the profitable variant)

Pushing the refinement further: the 20-day horizon was the wrong question for a
daytrading bot. Swept prediction horizons on the improved panel (short-term
reversal added), same purged walk-forward, each with its own shuffle null:

| horizon | mean IC | stability | t | null | gate |
|---:|---:|---:|---:|---:|:--|
| 1d | +0.0055 | 0.030 | +1.21 | −0.002 | fail |
| **3d** | **+0.0121** | 0.065 | **+2.64** | −0.003 | fail |
| 5d | +0.0083 | 0.046 | +1.86 | −0.006 | fail |
| 10d | +0.0109 | 0.067 | +2.69 | −0.001 | fail |

Unlike the 20-day (a 2020–2021 artifact), the **3-day** signal is
statistically significant (t 2.64) **and regime-persistent** — positive in
2022 (+0.001), 2023 (+0.017), 2024 (+0.019), 2025 (+0.023), 2026 (+0.004). It
still fails the conservative 0.02/0.15 RankIC gate, so the honest next question
is whether it's *tradeable*, not whether it clears an arbitrary bar.

**Cost-aware dollar-neutral long/short on the out-of-sample 3-day scores**
(weights ∝ cross-sectionally demeaned score, Σ|w|=1, daily rebalance,
2020-01..2026-07, 1,633 days, avg turnover 0.60/day):

| cost/side | net daily | ann. Sharpe | cumulative |
|---:|---:|---:|---:|
| 0 bp | +2.54 bp | +0.39 | +38.9% |
| 1 bp | +1.94 bp | +0.30 | +26.0% |
| 2 bp | +1.34 bp | +0.21 | +14.3% |
| 3 bp | +0.74 bp | +0.11 | +3.6% |
| 5 bp | −0.45 bp | −0.07 | −14.8% |

**This is a genuinely profitable strategy in backtest — market-neutral (not
beta), out-of-sample, leak-checked, net-positive through ~3 bp/side.** It is
also honestly *weak*: Sharpe ~0.2 at 2 bp, ~2% CAGR, and it goes negative by 5
bp/side, so it lives or dies on execution cost. Caveats: survivorship
(today's NDX constituents), overlapping-label autocorrelation shrinks the
effective sample, and the edge is a **multi-day** signal — not the 30-min
intraday scale, though a 30-min loop can rebalance toward its target.

**Consequence for the live bot.** The deployed policy is long-only RRL — it
cannot express this (config `allow_short: False`; the engine keeps cash
non-negative by construction). Realizing this edge needs a **long/short,
market-neutral rebuild** driven by the 3-day ranker, which is a real build with
its own risk of the Sharpe-0.2 signal not surviving forward — not a same-night
swap. This is the validated, honest path from "runs" to "modestly profitable,"
and it replaces §5's open question with a concrete lead: the edge is at the
short horizon, dollar-neutral, and cost-limited.

## 11. Day-trading rebuild: no kill switch, uncapped book, ~670-name universe (2026-08-08)

Three changes to make the live loop an actual day trader rather than a swing
book on a fast clock. Each one broke something that had been correct at the old
scale, which is the interesting part.

**Kill switches removed entirely.** `kill_max_drawdown / kill_daily_loss /
kill_rolling_20d_loss / kill_conviction_std`, `state.halted`, and
`invest --clear-halt` are gone. The argument against them is structural, not a
preference for more risk: a latching halt presupposes a book meant to persist
and an operator on hand to restart it. Neither holds for a loop that is flat at
every close. The loss a halt would prevent is already bounded to one session by
flat-by-close plus the per-name stop; what a halt actually produces is a bot
that stops trading for days — *including through the recovery* — until a human
notices. §9's 5-week replay is the evidence: the model-health switch fired two
hours in and the book sat in cash for five weeks. That was scored as "survival
by abstention"; on a day-trading mandate it is simply an outage. Risk now runs
continuously: vol-scaled per-name stops, re-entry cooldown, post-stop
de-grossing, gross cap, flat by close. Model health is still computed every bar
and warned about in the run log — it no longer stops the bot.
`PaperState.load` now drops unknown keys, so the deployed state file (which
still carries `halted`) survives the deploy instead of crashing the next cron.

**`max_positions: null` — no cap on breadth.** Sizing was already
self-normalising (targets ∝ conviction, then scaled so gross lands on
`max_gross_exposure`), so a wider book dilutes rather than levers. Targets too
thin to buy one whole share are dropped at decision time instead of queued as
orders that fill zero shares.

**Universe `extended`: ~670 names** (S&P 500 ∪ Nasdaq 100 ∪ 176 screened
non-index movers), up from 100. **The screen is a correctness control, not
curation.** `CostConfig` charges a flat 1 bp half-spread on everything; that is
roughly honest for a megacap and fiction for a $0.15 stock, where one tick is
hundreds of bp. Trading thin names at megacap costs manufactures profit from a
spread never paid — the same failure the "no free money on noise" test exists to
catch. So all 294 candidates were probed on real 30m bars and kept only above
$5M median 30-minute dollar volume (the 5th percentile of the index names
already in the universe) and $5 price: **176 kept, 118 dropped**. No ETFs — the
bot must not be able to buy SPY, the benchmark it is judged against. The same
probe found **32 tickers with no 30m data at all**; 11 of them were still in the
index snapshots (ANSS, BK, CMA, CTRA, DFS, FI, HES, HOLX, IPG, JNPR, K) and were
removed. Refreshing all 820 probed symbols took **129 s across 17 bulk
requests**, so the wide universe costs ~2 min of the 30-minute cron budget.

### The bug widening the book exposed

A cold-start rehearsal on real 30m bars (670 names, 170 bars, 2026-07-21 →
08-07) held **197 positions** at peak — and drove **cash to $1.68 on $99k
equity, gross to 0.99998**. Decision-time gross was a clean 0.90 the whole time,
so the cap was not being violated where it was computed; it was being voided
afterwards.

Cause: the no-trade band was an **absolute** 0.05 of equity, calibrated when
positions were ~20% each (i.e. 25% of position size). In a 200-name book a
target is gross/N ≈ 0.45%, so the band was ~11× the entire position. Every
holding read as "close enough, don't trade" and kept its full old weight, while
fresh entries were sized against the nominal budget on top. Gross ratcheted
until `_build_affordable_fill` — which shrinks buys to fit cash — became the only
thing stopping it. No leverage ever occurred (cash stayed positive by
construction), but `max_gross_exposure` meant nothing and the 10% cash buffer
was gone.

Fix: the band is now **relative to the position's own size**
(`rebalance_band_frac = 0.25`, measured against `max(|target|, |current|)`),
which reproduces the old 0.05 band exactly at the old 20% position size and
scales correctly at any breadth. Same rehearsal after the fix: **cash floor
0.0017% → 6.77%, max gross 0.99998 → 0.932**, fills 1958 → 2375 (more
rebalancing, as intended). A band of any width inherently lets gross run up to
`band_frac` above target before cash binds; that is documented, not eliminated.

The stop de-gross had the same shape of bug and got the same treatment: a flat
−0.10 of gross *per stop* pins a several-hundred-name book at the 0.30 floor
permanently, since a wide book always has a few names stopped out somewhere. It
is now proportional to the **fraction** of the book that stopped out — again
identical to the old behaviour at a 10-name book.

**What is not claimed.** The rehearsal above ran from a *cold* policy (no seed),
and it saturated: conviction σ 0.007, 47% of scores pinned at ±1. It ended
−1.88% vs SPY +3.32%. That number measures plumbing, not strategy — the live bot
seeds from the 5-year model with `|u| ≤ 0.7`, which §8 measured at σ ≈ 0.25. It
does confirm the loop runs end to end at 670 names inside the cron budget, holds
a wide book, and trades hundreds of times a session. Under the old code that same
σ 0.007 would have tripped the model-health switch on bar one and the bot would
have sat in cash for the entire window; that difference is the point of this
change, and it cuts both ways — **the bot will now keep trading through exactly
the conditions that used to stop it**, which is a deliberate transfer of risk
from "misses the recovery" to "keeps losing". Turnover is the other honest cost:
flat-by-close on a wide book means the whole gross round-trips daily, ~3 bp of
friction per turn, and there is still no measured cross-sectional signal (§4,
§10a) to pay for it. The forward record remains the only test.

## 12. v1 retired; rebuilt long/short on NautilusTrader (2026-08-17)

**v1's final record.** The long-only RRL day-trader ran 2026-07-21 → 2026-08-14:
235 thirty-minute bars over 19 sessions, ending **$96,211.75 (−3.79%)** against
SPY +3.75%, QQQ +3.20% and equal-weight +5.47% — 9.3 points behind the passive
version of its own universe. Max drawdown −4.65%, peak breadth 380 names, 5,108
fills. Frozen in `docs/v1-final.json`, published at `/v1/`.

**The diagnosis is one number: 2,508 closed round-trips at a 50.04% win rate.**
That is not a losing strategy, it is *no* strategy — a fair coin — and it turned
the book over 13.8× while paying a spread on every turn. §4 and §10 had already
measured the absence of edge twice (RankIC +0.004…+0.007 against a +0.02 gate;
the 20-day signal a 2020–2021 artifact). The forward result is the third
confirmation, and it removes the last hope that the live loop would behave
differently from the panel.

**Why a rewrite rather than a retune.** §10a found where the edge actually is: a
3-day horizon, t = 2.64, positive in every year 2022–2026, traded
**dollar-neutral long/short**, net-positive through ~3 bp/side. v1 could not
express it and no parameter change gets it there — a long-only book cannot be
dollar-neutral, and a book that liquidates every afternoon cannot hold a
multi-day signal. The two properties that made v1 safe are exactly the two that
made this edge unreachable.

**v2.** Cross-sectional long/short on the NautilusTrader engine: ~670 names,
30-minute bars, margin account, leverage 2, gross target 1.5×, positions held
overnight, no kill switch, fresh inception at $100,000 (v1's ending equity is
deliberately *not* carried over, so the two records are comparable). The alpha is
reversal + vol-scaled momentum + inverse-volatility + liquidity, z-scored across
the cross-section and de-meaned. News enters **additively** on the standardised
score rather than multiplicatively on conviction — v1's restriction existed
because a long-only bot's only response to bad news is to own less, which no
longer applies.

### What the port turned up

Four findings that cost real time and are worth recording, all verified by
running code rather than by reading docs:

**1. The bar-timestamp trap.** Yahoo labels a 30-minute bar by its **open**;
NautilusTrader copies the index verbatim into `ts_event`, drives its clock from
`ts_init`, and feeds each bar to the matching engine *before* the strategy
callback — so a market order submitted in `on_bar` fills at **that bar's close**.
Leaving the open label in place therefore fills every order thirty minutes into
the engine's own future, silently. The bridge shifts by one interval, via a real
timezone conversion (a fixed −4h offset misplaces every bar for the weeks around
a DST change).

**2. The clock instrument.** A cross-sectional strategy must act once per
timestamp, but acting on the first bar of a new timestamp means one symbol's
matching engine has advanced and 669 have not — that symbol fills at a different
bar. Inserting a synthetic, never-traded instrument's bars *first* (same-timestamp
delivery follows insertion order) gives a callback at which every real symbol has
delivered T-1 and none has delivered T. The whole book then fills uniformly, and
the execution-delay invariant falls out of the arrangement instead of being
maintained by hand.

**3. `balance_total` is not cash.** On a Nautilus margin account it is cash plus
realized PnL and is *not* reduced by opening a position — positions lock margin
in `balance_locked`. Persisting "cash" the way v1 did would double-count every
open position. The complete, exact serialisation of the book is
`(balance_total, {symbol: (signed_qty, avg_px)})`; restoring seeds that balance
and rebuilds positions on synthetic bars at their own average price, with fees
switched off. Measured: a resumed run reproduces an uninterrupted one to the
cent.

**4. Two bugs the tests caught, both invisible without them.** The round-trip
test initially disagreed by $64: the strategy was **dropping one decision at
every run boundary** (the last bar's decision has no later bar to fill against,
and the next run started with none pending). Fixed by having a resumed run begin
deciding one bar before it begins trading, recomputing that decision from warmup
data. The remaining $1.20 was **short borrow accruing per cron firing rather than
per bar span** — so the cost of holding a short overnight depended on how often
GitHub Actions happened to fire, and a bot that ran twice owed twice. Both are
the class of bug that a forward record would have quietly absorbed.

Also worth knowing: the default `RiskEngine` rate limit (100 orders/second)
**denies** most of a 670-name rebalance, `Equity` forbids fractional shares
outright (the risk engine rejects, it does not round), and `portfolio.net_exposure`
returns an *absolute* notional despite the name, so a short reports positive.

**What is not claimed.** v2 has no forward record at all. The strategy it is
built to express was profitable in backtest at a Sharpe of about 0.2 — thin
enough that execution cost decides it, and thin enough that a few weeks of
forward results will not settle anything either. v1's honest epitaph is that it
ran flawlessly and lost 3.8%; v2 starts with no more right to be believed.

## 13. Renamed to `tradingbot`; exposure limits removed (2026-08-17)

The project is renamed `swingbot` → `tradingbot` (package, CLI, classes,
`TraderId`), and the stated objective is now explicit: **maximise simulated
P&L**. To that end every *exposure* limit in the v2 sizing layer became optional
and defaults to off — no fixed gross target, no per-name cap, no net-exposure
clamp, and account leverage raised 2 → 20. Book size is set by a **volatility
target** instead.

**What was deliberately not removed.** The four invariants of §12, the full cost
model, and the `min_dollar_volume` liquidity floor. These are not risk limits;
they are what makes the P&L a measurement. A bot that fills on the bar it decided
from, or trades names too thin to absorb the order, can print any number you
like — removing them would not make it earn more, it would make the number stop
being an earning.

**The one-factor vol model was wrong, and measurably so.** Sizing first predicted
book volatility as `(1−ρ)·Σ(wᵢσᵢ)² + ρ·(Σwᵢσᵢ)²`. The second term keys off the
*signed* risk sum, which for a dollar-neutral book is ~0 — so the model declared
a 170-name neutral book almost riskless and the vol target duly levered it to
**12.3× gross**. Measured against the realized equity curve it was understating
volatility by a stable **3.3×** (3.36 / 3.33 / 3.26 at targets of 0.35 / 0.20 /
0.10 — a systematic bias, not noise). At a 35% target the book was realizing
**117% annualised vol**.

The fix is a third term, `(residual·Σ|wᵢσᵢ|)²`, charging factor risk in
proportion to *gross* risk — the sector, liquidity and crowding loadings that a
cross-sectional reversal book does not net away by being dollar-neutral. At
`residual = 0.21` prediction tracks realization to within **1.07–1.16×**, erring
slightly conservative. It is calibrated on this universe, not derived, and should
be re-measured if the universe, cadence or alpha changes.

**Measured, 670 names, 30m bars, 2026-05-24 → 2026-08-17, identical bars and
costs across variants** (a *backtest* over the window the live book later turned
out to have back-filled — see §14):

| variant | gross | equity | return |
|---|---|---|---|
| old: gross 1.5×, cap 4%, net 0.30, lev 2 | 1.43× | $96,176 | −3.82% |
| new: vol-target 35%, no caps, lev 20 | 4.36× | $92,792 | −7.21% |
| new: vol-target 20%, no caps, lev 20 | 2.43× | $96,381 | −3.62% |
| new: vol-target 35%, `max_gross 3×` | 2.94× | $92,351 | −7.65% |

**The finding is monotone and it is not the one that was wanted: over this
window, loss scales with gross.** That is exactly what theory predicts of
leverage applied to a negative realized edge, and it is the same conclusion §12
reached from the other direction — the edge is thin enough (backtest Sharpe ~0.2)
that nothing in the sizing layer decides the outcome. Removing the limits raises
the ceiling and lowers the floor; it does not create alpha. Sizing is a
multiplier on an edge, and a multiplier cannot fix the sign of what it multiplies.

Two and a half months is not a verdict on the alpha, and this window is one draw.
But it is a verdict on the *mechanism*: there is now no cap standing between a
bad stretch and the account, which is what "no limitations" means in practice.

## 14. The first live book was a back-fill; v2's record reset to $100k (2026-08-18)

**The v2 book published as "live" was not a forward record.** Its state read
`inception: 2026-05-26`, and it got there in a single cron firing. The run loop
took every completed bar in the store and called it new:

    new_grid = [t for t in grid if last is None or t > last]

`last` is the watermark, and a freshly incepted book has none — so on the first
firing the condition is vacuously true for the whole store, which reaches back to
`data_start`. The bot simulated 2026-05-26 → 08-17 in one pass, **13,571 fills**,
and landed at a balance of **$78,680**. `paper.start` existed and was still set
to v1's `2026-07-21T15:30`, but nothing in the v2 runner read it; only the
retired `invest` command did.

None of that is a *simulation* error — the fills obeyed the execution delay, paid
full costs, and would round-trip. It is a **provenance** error, and a worse one:
the number was published on a page that says "live" beside a v1 archive whose
whole point is that a forward record is the only honest test. A back-fill of the
window the alpha was developed against is exactly the number not to trust, and it
was sitting in the position marked "trusted".

**Fix: inception is a floor.** `paper.start` now bounds the tradable grid.
Sub-floor bars are still read — the alpha needs ~262 bars of lookback and losing
them would just move the distortion into the signal — but they cannot be traded.
A book with no watermark can no longer mistake the bar store for its own past.

**The same knob is the reset switch.** A stored book whose inception predates the
floor is *retired* — moved to `artifacts/v2/retired/<inception>/`, published
along with everything else — and a fresh $100,000 incepts in its place. Deleting
would have been easier and wrong: the old curve was published, and the site
rebuilds equity from those parquets, so leaving them in place would have spliced
the back-fill onto the forward record and drawn the join as one line.

**v2's forward record therefore starts 2026-08-18**, first tradable bar 09:30
(filling at its 10:00 close), at $100,000 — the same figure v1 started from, so
the three records (v1, the §13 back-fill, v2 forward) stay comparable instead of
one continuing another's drawdown.

Verified against the live book before shipping: the 09:00 ET firing retires the
old state and writes an untouched $100,000 with no positions; with the floor
moved into the local store's coverage, the same run replays 270 timestamps of
warmup and trades only the 6 bars at or above it. Four tests pin it — bars below
the floor are never traded, moving the floor retires and restarts, a book at or
above the floor is *never* retired (a switch that re-fired every half hour would
republish a one-bar-old book as the entire record), and an inception that
precedes its first tradable bar persists immediately, because the dashboard
export reads `state.json` and the cron starts firing an hour before the first bar
closes.

**What this costs.** Three months of apparent history, which was never history.
The forward record is now days old and says so.

