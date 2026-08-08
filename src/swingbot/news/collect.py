"""One-call news collection: fetch, label, archive, aggregate, publish.

This is what the weekend workflow and the ``swingbot news`` CLI both run.

Ordering matters and is not arbitrary:

1. **Macro first.** It is the tier that always works, so if the run is going to
   produce anything it produces it here. Company news is layered on top.
2. **Archive before aggregating.** The Parquet corpus is the durable artifact;
   the signal is derived and can always be rebuilt from it. If aggregation
   throws, the articles are already banked.
3. **Aggregate from the archive, not from this run's fetch.** A single
   collection sees only what the feeds currently hold -- typically a few days.
   Reading back the archive means the signal spans the full decay window even
   when one run's fetch came back thin, and it makes re-runs idempotent.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from swingbot.news import feeds, store, tickers
from swingbot.news.signal import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_PRIOR_COUNT,
    NewsSignal,
    build_signal,
)


def collect(
    *,
    out_dir: Path,
    universe: list[str] | None = None,
    company_news: bool = True,
    max_symbols: int | None = None,
    pause: float = 1.0,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    prior_count: float = DEFAULT_PRIOR_COUNT,
    lookback_days: float = 14.0,
    as_of: datetime | None = None,
    log_fn=print,
) -> NewsSignal:
    """Run the full pipeline and write ``signal.json`` + ``articles.parquet``.

    ``universe`` restricts symbol resolution and drives the per-ticker sweep.
    ``max_symbols`` caps that sweep -- useful for a quick local run, and for a
    weekend job that would rather cover the most-traded names reliably than the
    whole universe unreliably.
    """
    out_dir = Path(out_dir)
    now = as_of or datetime.now(UTC)
    uni_set = set(universe) if universe else None

    articles = feeds.fetch_macro(log_fn=log_fn)
    log_fn(f"news: macro tier collected {len(articles)} articles")

    if company_news and universe:
        syms = universe[:max_symbols] if max_symbols else universe
        articles.extend(feeds.fetch_ticker_news(syms, pause=pause, log_fn=log_fn))

    articles = tickers.label_articles(feeds.dedupe(articles), uni_set)
    log_fn(f"news: {len(articles)} unique articles after dedupe")

    archive = out_dir / "articles.parquet"
    added = store.append(archive, articles, collected_at=now)
    log_fn(f"news: archived {added} new articles -> {archive}")

    # Rebuild the signal from the archive so it spans the whole decay window,
    # not just what this fetch happened to return.
    corpus = store.load(archive, since=now - timedelta(days=lookback_days + 1))
    sig = build_signal(
        corpus,
        as_of=now,
        half_life_days=half_life_days,
        prior_count=prior_count,
        max_age_days=lookback_days,
    )
    sig.write(out_dir / "signal.json")
    log_fn(f"news: wrote signal -> {out_dir / 'signal.json'}")
    return sig
