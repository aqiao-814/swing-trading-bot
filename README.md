# trading-bot

Autonomous **long/short** paper-trading bot on the
[NautilusTrader](https://github.com/nautechsystems/nautilus_trader) engine, live at
**https://aqiao-814.github.io/swingbot-live/**

$100k simulated capital · **~670 liquid US stocks** · 30-minute bars · **long and
short, held overnight, on margin** · no kill switch. The forward record runs
from **2026-08-18**. **Every dollar is simulated** — there are no brokerage credentials and no code path that can place
a real order. Research history and measured results:
[docs/FINDINGS.md](docs/FINDINGS.md).

> **v1 is retired.** The long-only reinforcement-learning day-trader that ran
> 2026-07-21 → 2026-08-14 finished at **−3.79%** ($96,211.75) against SPY +3.75%
> and its own universe's +5.47%. Its frozen record and a full breakdown of how it
> worked live at [/v1/](https://aqiao-814.github.io/swingbot-live/v1/); the
> snapshot is [docs/v1-final.json](docs/v1-final.json). Its code remains in
> `src/tradingbot/paper/` as the research harness it grew out of.

## Why v2 exists

v1's plumbing was never the problem. Over 19 sessions the cron never missed a
bar, the book stayed inside its exposure cap, and it went flat before every
close exactly as instructed. It simply had no edge: **2,508 closed round-trips
at a 50.04% win rate**, turning the book over 13.8× and paying a spread on every
turn. A fair coin that pays to flip.

The research log had already measured why, twice — cross-sectional price and
volume features on liquid US megacaps produce a rank IC of +0.004 to +0.007
against a +0.02 go/no-go bar, and the 20-day version of the signal turned out to
be a 2020–2021 artifact (FINDINGS §4, §10).

But the same work found where the edge **is**: a **3-day** horizon, significant
(t = 2.64) and regime-persistent through 2022–2026, traded **dollar-neutral
long/short**, net-positive out-of-sample through roughly 3 bp per side
(FINDINGS §10a). v1 structurally could not trade it — a long-only book cannot be
dollar-neutral, and a book that liquidates every afternoon cannot hold a 3-day
signal. **v2 exists to be able to hold a short overnight.**

It is not a claim of profitability. The edge is thin (Sharpe ≈ 0.2 at realistic
cost, negative by 5 bp/side) and unproven forward. The forward record is the
only test.

## How it works — plain English

- Every completed 30-minute bar — 13 a session — the bot scores **~670 liquid US
  stocks** on how attractive each looks *relative to the others*: cheap against
  its own last few days (short-term reversal), strong over the last month once
  you divide out its volatility, calm rather than wild, and liquid enough to
  trade honestly.
- Those scores are **de-meaned across the whole universe**, which is what makes
  this a bet on *which names beat which* rather than on the market going up. It
  buys the strongest quarter and **shorts the weakest quarter**.
- Positions are sized **inverse to each name's own volatility**, so two names it
  likes equally get equal *risk*, not equal dollars.
- **There are no exposure limits.** No per-name cap, no net-exposure clamp, and
  no fixed gross target. Book size is set by a **volatility target** instead —
  gross floats to whatever makes the book's own predicted volatility 35%
  annualised (~4.4× equity on the live universe, against the old fixed 1.5×), so
  it levers into a calm cross-section and out of a violent one. Every limit is
  still *settable* (`--max-gross`, `--max-weight`, `--max-net`); none is set by
  default. Measured consequence, honestly: **backtested** over 2026-05-24→08-17
  the bigger book lost **more** (−7.2% vs −3.8%), because leverage multiplies a
  negative realized edge just as faithfully as a positive one. See FINDINGS §13.
- **It holds overnight.** No flat-by-close, no liquidation, no gap avoidance —
  which is the whole point, because the signal it trades takes about three days
  to pay.
- It **reads the news**. Every weekend a separate job digs through free economy
  and company news, scores each story with a finance-specific lexicon, and
  publishes a per-stock score. During the week that score is **added** to the
  ranking, so bad press is a reason to be *short* a name — a change from v1,
  where news could only ever make it own less.
- Orders decided on one bar fill on the **next** one, at realistic cost: spread,
  slippage, square-root market impact, SEC and FINRA fees, and borrow on every
  short held overnight.
- There is **no kill switch** and no drawdown halt, deliberately (see below).

## How it works — technical

**Engine.** NautilusTrader 1.230, `AccountType.MARGIN`, `OmsType.NETTING`, one
`Equity` per ticker at Reg-T margins, `default_leverage=20` — a ceiling high
enough that the vol target, not the broker, decides book size. The same engine and
the same execution semantics run a backtest and the live loop, which is the
research-to-live parity v1 had to hand-maintain.

**Alpha** (`nautilus/signals.py`, no Nautilus imports so it is testable alone):
per-symbol reversal (3d, negated), vol-scaled momentum (20d), realized
volatility (negated — betting-against-beta), and log dollar volume; each
z-scored across the cross-section, winsorised at 3σ, combined, then de-meaned
and rescaled to unit σ. Sizing is `score / σ`, normalised to unit gross and then
scaled by whatever gross the **vol target** implies. Predicted book variance is

    (1−ρ)·Σ(wᵢσᵢ)²      idiosyncratic
  + ρ·(Σwᵢσᵢ)²          one common factor, keyed off the SIGNED risk sum
  + (residual·Σ|wᵢσᵢ|)²  factor risk that dollar-neutrality does not hedge

The middle term is ~0 for a neutral book and dominant for a directional one, so
the same target permits a large neutral book and a small directional one. The
third term is not optional decoration: without it the first two call a 170-name
neutral book almost riskless and the target levers to 12.3×, understating
realized vol by 3.3×. At `residual = 0.21` prediction tracks realization to
1.07–1.16× (FINDINGS §13). Gross and net are then landed **exactly, in closed
form**, by scaling the long and short sides separately (`aL = (G+N)/2`,
`bS = (G−N)/2`); the old fixed-point iteration now runs only when a per-name cap
is explicitly set and actually binding.

**The bar loop** (`nautilus/strategy.py`). A cross-sectional strategy needs every
symbol's bar for one instant before it can rank anything, and acting on the first
bar of a new timestamp would fill one symbol at a different bar than the other
669. So the engine is given a synthetic, never-traded **clock instrument** whose
bars are inserted first. On the clock bar for timestamp T, every real symbol has
delivered T-1 and none has delivered T — so the whole book fills uniformly at
T-1's close. The strategy submits the decision it computed one clock tick
earlier, which gives **exactly one bar of execution delay, measured, not
assumed**.

**State** (`nautilus/state.py`). Each cron firing is a fresh `BacktestEngine`, so
the book is serialised to JSON between runs. On a Nautilus margin account
`balance_total` is cash *plus realized PnL* and is **not** reduced by opening a
position — a position locks margin instead — so the complete state is
`(balance_total, {symbol: (signed_qty, avg_px)})`. Restoring seeds that balance,
replays two synthetic bars at each position's own average price, and rebuilds the
book with the fee model switched **free** (cleared on a *later* bar, because
fills settle after the callback that submits them returns). A resumed run
reproduces an uninterrupted one **to the cent**, and there is a test for it.

**Inception** (`paper.start` in `configs/cloud.yaml`) is a *floor*, not a label.
Bars before it are fetched and feed the alpha's ~20-session lookbacks, but none
of them can be traded — so a book with no watermark cannot mistake the contents
of the bar store for its own history. Moving the date forward is also the reset
switch: the next run retires the earlier book under `artifacts/v2/retired/`
(kept, not deleted — it was published) and incepts a fresh $100,000. Both halves
were learned the expensive way; see FINDINGS §14.

**Costs** (`nautilus/costs.py`). Every friction — half-spread, slippage,
square-root impact, SEC §31, FINRA TAF — is charged as an explicit Nautilus
commission rather than baked into the fill price. The P&L is identical either
way, but v1's split made its own cost reporting wrong: money lost to a spread
never debits cash, so v1 reported $227 of costs against $3,768 of losses on 13.8×
turnover. Now every dollar of friction is attributable to a fill. Short borrow
accrues per **bar span** (not per cron firing, which would make the cost depend
on how often GitHub Actions fired).

**No kill switch — deliberately.** A latching halt assumes an operator on hand to
clear it. What it actually produces is a bot that stops trading for days —
*including through the recovery* — until someone notices; an earlier replay had
exactly that, sitting in cash for five weeks after a model-health switch fired.
Risk is *priced*, not *forbidden*: a de-meaned, roughly market-neutral book,
volatility-scaled sizing, a volatility-targeted gross, and a liquidity floor below
which a name is not traded at all. This is a real transfer of risk from "misses
the recovery" to "keeps losing", made with open eyes — and with no exposure caps
the downside of that trade is now larger, not smaller.

**What is *not* relaxed.** The objective is to maximise simulated P&L, and that
number is only worth maximising if it is real. So the four invariants below are
not negotiable, transaction costs are charged in full, and the liquidity floor
stays: a fill in a name too thin to absorb the order is fiction, not profit.

**Invariants** (`tests/test_nautilus_invariants.py`), all re-proved against the
new engine:

1. **Execution delay** — every symbol fills on a bar strictly later than the one
   its decision was computed from, and all on the *same* bar.
2. **No look-ahead** — Yahoo labels a 30-minute bar by its OPEN and Nautilus
   fills at the bar's CLOSE, so the bar bridge shifts every timestamp forward by
   one interval. Tested across a daylight-saving transition, because a fixed
   offset would misplace every bar for weeks a year.
3. **No free money on noise** — a costed churn is compared against a *free* run
   of the identical trades, so the difference is friction exactly and luck cannot
   pass the test.
4. **State round-trip** — stop, persist, resume, and land on the same book.
5. **Inception is a floor** — no bar before `paper.start` is ever tradable, so a
   fresh book starts a forward record instead of back-filling one out of
   whatever history the bar store happens to hold.

## Deployment (all free)

- **`.github/workflows/trade.yml`** — every 30 minutes during market hours:
  restores the book, replays every newly completed bar, exports `data.json`, and
  publishes state + site to
  [swingbot-live](https://github.com/aqiao-814/swingbot-live) (GitHub Pages).
- **`.github/workflows/news.yml`** — Saturday and Sunday 13:00 UTC: collects and
  scores free economy + company news, then opens and merges a PR putting
  `news/signal.json` on `main`. The ~670-name sweep takes ~20 minutes, which is
  why it runs on a closed market. v2 holds through weekends, so a weekend signal
  now lands on a book that is actually open.
- **`.github/workflows/live.yml`** — live quotes between trading runs.
- Replay cost is bounded and measured: 670 symbols × 200 bars ≈ 3 s and ~360 MB;
  the full 60-day window is ~9 s and ~500 MB, comfortably inside the half-hour
  cron budget.

## Local use

```bash
make test                       # 293 tests
tradingbot trade                  # run the v2 loop locally
tradingbot news --out news        # collect the weekend news signal
python scripts/export_site_data.py
```

`nautilus_trader` is pinned to **1.230.0**: from 1.231 the macOS wheels target
macOS 26, so an unpinned install on macOS 15 falls back to a source build and a
Rust toolchain.

## Layout

```
src/tradingbot/nautilus/  v2: the live loop — instruments, costs, bars, signals,
                        strategy, state, runner
src/tradingbot/news/      free news collection, financial-lexicon sentiment, signal
src/tradingbot/paper/     v1, retired: the RRL day-trading loop and its learner
src/tradingbot/           portfolio accounting, execution costs, features, data store
src/tradingbot/{env,backtest,agents}/  research harness (see docs/FINDINGS.md)
site/                   the hosted dashboard; site/v1/ is the frozen v1 archive
```

## Non-goals

No real-money trading, ever. Pointing this at live capital is a decision made
outside this repo.
