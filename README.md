# swingbot

Autonomous **day-trading** paper bot, live at
**https://aqiao-814.github.io/swingbot-live/**

$100k simulated capital · **~670 liquid US stocks** · 30-minute bars · **flat by
every close, never a position held overnight** · 30m inception 2026-07-21.
**Every dollar is simulated** — there are no brokerage credentials and no code
path that can place a real order. Research history and measured results:
[docs/FINDINGS.md](docs/FINDINGS.md).

> The repository is still named `swing-trading-bot` for historical reasons (it
> began as a multi-day swing strategy). The live loop is now a **day-trading**
> loop: it opens and closes every position within the same session.

## How it works — plain English

- On every completed 30-minute bar of the trading day — 13 decision points per
  session — the bot scores **~670 liquid US stocks** on how strongly its model
  wants to own each one ("conviction").
- It sells anything it has lost conviction in, or that has fallen too far below
  what it paid (a stop-loss sized to each stock's own volatility).
- It buys the highest-conviction names. **There is no cap on how many stocks it
  holds**: position sizes are proportional to conviction and then scaled so
  total exposure lands at 90% of the portfolio, so a wider book means smaller
  positions, never borrowed money. At most 20% in any one name.
- **Day trading, not overnight.** Near the close (the 15:00 ET bar) it sells the
  whole book to zero, and it opens no new position in the last hour that it
  couldn't close again the same day. The portfolio ends every session flat —
  100% cash — so it carries no overnight or weekend gap risk.
- Orders queue and execute at the **next bar's open** (30 minutes later), with
  realistic trading costs (spread, slippage, price impact, regulatory fees).
- It also **reads the news**. Every weekend a separate job digs through free
  economy and company news (CNBC, MarketWatch, the Federal Reserve press wire,
  plus per-company headlines from Yahoo), scores the tone of each story, and
  publishes a sentiment score per stock. During the week that score **nudges**
  how badly the bot wants each name — good news makes it hold more, bad news
  less. It is a nudge and not a trigger: news can never make the bot buy
  something its price model had no opinion on.
- After every bar it also **learns**: each stock's realized return nudges the
  model's weights, so the policy adapts continuously.
- Risk is continuous, not a latching halt: per-name volatility-scaled stops, a
  re-entry cooldown, a gross-exposure cap that tightens after stop-outs, and
  above all a book that is flat at every close. There is **no kill switch** —
  see below.
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

**Bar loop** (`paper/engine.py`, interval-agnostic — `1d`, `60m` or `30m` via
`paper.interval` — idempotent via a `last_processed` watermark; only bars whose
completion time has passed ever enter it):

1. **Fill** pending orders at the bar's open through `ExecutionModel` —
   half-spread 1 bp, slippage 0.5 bp, square-root impact, SEC §31 + FINRA TAF
   on sells. Sells first, buys in conviction order, capped so cash stays ≥ 0.
2. **Mark** at the bar close: ledger row with equity, P&L, turnover, costs, and
   buy-and-hold SPY / QQQ / equal-weight benchmarks.
3. **Learn**: one RRL update per symbol from the realized bar return.
4. **Decide**: score the universe on the bar. Exit on conviction < 0.05 or a
   2σ·√(20-bar) vol-scaled stop below basis; enter needs conviction ≥ 0.15.
   **No position-count cap** (`paper.max_positions: null`): every candidate that
   clears the bar gets a target weight of f × 20%, and the whole book is then
   scaled so gross lands at ≤ 0.90. Breadth therefore dilutes rather than
   levers. Stops inside the re-entry cooldown cut the gross cap in proportion to
   the *fraction* of the book that stopped out (floor 0.30) — a flat per-stop
   slab would pin a several-hundred-name book at the floor permanently.
   Targets too thin to buy one whole share are dropped rather than queued.
   5% no-trade band. Orders fill at the *next* bar's open.
5. **Flat by close** (day-trading, when `paper.day_trading`): on the flatten bar
   — the last bar whose next-open fill still lands in the session (15:00 ET on
   the 30m loop, since it fills at the 15:30 open) — every holding is sold to
   zero, overriding the normal decide step. No new position is opened on the
   15:00 or 15:30 bars, because it could not be flattened again the same day.
   Derived from the session's own bar-time grid, so a partial final session
   (a mid-day live run) is never mistaken for a short day. `1d` loops ignore it.

**No kill switch — deliberately.** A latching halt (fire on a drawdown, flatten,
stay dead until an operator clears it) is a swing-trading control: it assumes a
book meant to persist and a human on hand to restart it. Neither holds here. The
book is already flat at every close, so the loss a halt would prevent is bounded
by one session; what a halt actually produces is a bot that silently stops
trading for days — including through the recovery — until someone notices. Risk
is carried continuously instead: per-name vol-scaled stops, post-stop
de-grossing, the gross cap, and flat-by-close. Model health (conviction σ,
saturated fraction) is still computed every bar, logged to the learning table,
and warned about in the run output — it just no longer stops the bot.

**News tilt** (`news/`, `paper.news`). Two collection tiers, both free and
keyless. *Macro*: eleven bulk RSS feeds (CNBC topics, MarketWatch/Dow Jones, the
Fed press wire), ~270 articles a pass, unthrottled. *Company*: per-ticker news
through yfinance. The obvious route — Yahoo's per-ticker RSS endpoint — is
**not** usable: probed 2026-08-08 it serves a handful of requests then hard-429s
at the IP level, still refusing after a 300-second cooldown. yfinance reads a
different (cookie+crumb) endpoint and answers normally.

Tone comes from a Loughran-McDonald-style **financial** lexicon with negation
and intensity weighting, not a general-purpose one: LM's finding is that most
"negative" words in general lexicons (*liability*, *tax*, *depreciation*) are
neutral accounting vocabulary, so a general lexicon reads every 10-K as a
disaster. Scores decay on a 2-day half-life and are shrunk toward zero by
`n/(n+3)`, so one headline scoring −1.0 is treated as under-observed rather than
bearish.

**The company score is de-meaned across the cross-section**, and that step is
what makes it a signal at all. Measured on the first full-universe run
(2026-08-08, 5,755 articles, 635 of 670 names covered): mean symbol tone
**+0.346**, median +0.400, and only **77 of 635** symbols negative. Financial
copy — earnings-call coverage above all — is overwhelmingly bullish, so the raw
score answers "is the press positive about this company?", to which the answer
is nearly always yes, and it orders almost nothing. Subtracting the
cross-sectional mean asks "does the press like this name *more than average*?"
(mean 0.000, 263 of 635 negative, σ 0.256) and surfaces downgrades, lawsuits and
disappointing prints instead of generic optimism. This is the same correction
`agents/ranker.py` makes by predicting excess rather than raw return: a signal
that can win by saying yes to everything is not a signal. The uniform component
is not discarded — it survives in the macro term, where a market-wide mood
belongs.

The engine applies it as `f' = f · (1 + 0.30 · news · sign(f))` — **multiplicative
on the policy's own conviction**, so news reorders the ranking, resizes
positions, and can push a borderline name across the entry or exit threshold,
but a name the model is neutral on stays out of the book no matter what the
headlines say. The tilt fades on the signal's own age (a Sunday score is worth
~1/4 by Tuesday) and a missing or corrupt `signal.json` degrades to no-news
rather than to an error. Model-health metrics are deliberately computed on the
*untilted* score, so a quiet news week cannot mask a saturated policy. Every
decision row records `model_conviction` and `news_score` alongside the final
`conviction`, which is the only way to ever answer whether news helped.

Symbol resolution is precision-first: mentions count only from cashtags
(`$NVDA`), exchange-qualified tickers (`NASDAQ: NVDA`), or a curated
case-sensitive company-name map. Bare uppercase matching is refused because a
large minority of real tickers are English words — `IT`, `ON`, `ALL`, `KEY`,
`NOW`, `A`, `T` — and a false mention injects a *wrong* tilt into a real
position, which is worse than missing the story. Yahoo's own per-ticker labels
are verified rather than trusted: `yf.Ticker("ADI").news` was observed serving a
different company's earnings beat.

**No news backtest yet, deliberately stated.** None of the free feeds serve
history, so the signal cannot be tested retroactively — the article archive
(`news/articles.parquet`, append-only, raw text kept so it can be rescored when
the lexicon changes) exists to accumulate the evidence going forward. Until it
has, the 0.30 tilt weight is a prior, not a measured edge.

**Universe.** `paper.universe: extended` — S&P 500 ∪ Nasdaq 100 ∪ 176 screened
high-volume non-index movers, ~670 names. The extras were probed against real
30m bars and kept only above $5M median 30-minute dollar volume and $5 price:
the cost model charges a flat 1 bp half-spread, which is roughly honest for a
liquid name and pure fiction for a $0.40 one, and a backtest that trades thin
names at megacap costs manufactures profit from a spread it never paid. No ETFs
— the bot must not be able to buy SPY, the benchmark it is judged against.

**No lookahead by construction**: decisions at close *t* can only fill at open
*t+1*; features are trailing-only (tests corrupt future bars and assert earlier
features are bit-identical); a pure-noise churn test must lose money.

## Deployment (all free)

- **`.github/workflows/trade.yml`** — every 30 minutes during market hours
  (13:00–21:30 UTC weekdays): restores state, processes every newly completed 30m bar,
  exports `data.json`, publishes state + site to
  [swingbot-live](https://github.com/aqiao-814/swingbot-live) (GitHub Pages).
- **`.github/workflows/live.yml`** — every 20 min during market hours: live
  quotes → `live.json` (live P&L between trading runs).
- **weekend news** — Saturday and Sunday 13:00 UTC: collect and score free
  economy + company news, then open and merge a PR putting `news/signal.json`
  on `main`, where the next weekday trading run reads it. The ~670-name
  per-company sweep takes ~20 minutes, which is why it runs on a closed market
  rather than inside the half-hour weekday budget. Staged at
  [docs/news-workflow.yml](docs/news-workflow.yml); install with
  `git mv docs/news-workflow.yml .github/workflows/news.yml`.
- Portfolio state persists in the public repo under `state/`; bar data lives in
  an Actions cache. Manual run: `gh workflow run trade.yml`.
- The ~670-name 30m refresh measured ~130 s across 17 bulk requests, so the wide
  universe uses a couple of minutes of the half-hour budget. Runs are serialised
  by a concurrency group rather than cancelled, so a slow run delays the next
  bar's decision instead of corrupting state.

## Local use

```bash
make test                   # 236 tests
make invest                 # run the loop locally
python -m swingbot.cli news --out news --max-symbols 50   # collect news
```

## Layout

```
src/swingbot/paper/    the live loop: engine, continual RRL, state, dashboard
src/swingbot/news/     free news collection, financial-lexicon sentiment, signal
src/swingbot/          portfolio accounting, execution costs, features, data store
src/swingbot/{env,backtest,agents}/  research harness (see docs/FINDINGS.md)
scripts/               site data + live quote exporters
site/                  the hosted dashboard (single static page)
```

## Non-goals

No real-money trading, ever. Pointing this at live capital is a decision made
outside this repo.
