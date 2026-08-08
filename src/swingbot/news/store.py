"""Append-only archive of every article the bot has ever seen.

The live signal only needs the last two weeks, so this archive is not for the
trading loop -- it is for *evidence*. The one honest way to find out whether
news adds anything is to accumulate a timestamped corpus going forward and
measure the signal against realised returns later. That is impossible to do
retroactively: none of the free feeds serve history, so the corpus can only
ever start accumulating from the first collection run.

Parquet, matching the ledger/trades/decisions tables in ``paper/state.py``, and
append-only with dedupe on ``uid`` so re-running a collection is idempotent --
the weekend job may fire twice, be re-run by hand, or overlap a previous run's
window, and none of those should double-count a headline.

Storing the article text (not just the score) is on purpose: the sentiment
lexicon will change, and when it does the whole archive can be rescored.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from swingbot.news.feeds import Article

SCHEMA = {
    "uid": pl.Utf8,
    "ts": pl.Datetime(time_unit="us", time_zone="UTC"),
    "source": pl.Utf8,
    "kind": pl.Utf8,
    "title": pl.Utf8,
    "summary": pl.Utf8,
    "url": pl.Utf8,
    "symbols": pl.Utf8,  # comma-joined; Parquet lists complicate append+dedupe
    "collected_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}


def to_frame(articles: list[Article], *, collected_at: datetime | None = None) -> pl.DataFrame:
    when = collected_at or datetime.now(UTC)
    rows = [
        {
            "uid": a.uid,
            "ts": a.ts.astimezone(UTC),
            "source": a.source,
            "kind": a.kind,
            "title": a.title,
            "summary": a.summary,
            "url": a.url,
            "symbols": ",".join(a.symbols),
            "collected_at": when,
        }
        for a in articles
    ]
    if not rows:
        return pl.DataFrame(schema=SCHEMA)
    return pl.DataFrame(rows, schema=SCHEMA)


def append(path: Path, articles: list[Article], *, collected_at: datetime | None = None) -> int:
    """Merge articles into the archive at ``path``. Returns the count of new rows.

    On a uid collision the *existing* row wins for text but symbol sets union:
    a story first seen in the macro tier with no symbols may later be served by
    the per-ticker tier already labelled, and dropping that label would lose
    real information.
    """
    fresh = to_frame(articles, collected_at=collected_at)
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh.write_parquet(path)
        return fresh.height

    prior = pl.read_parquet(path)
    combined = pl.concat([prior, fresh], how="vertical_relaxed")

    # Union the symbol strings per uid, then keep the first (oldest collected)
    # row for every other column.
    unioned = (
        combined.group_by("uid")
        .agg(
            pl.col("symbols")
            .str.split(",")
            .list.explode(empty_as_null=True)
            .unique()
            .sort()
            .alias("syms")
        )
        .with_columns(
            pl.col("syms")
            .list.eval(pl.element().filter(pl.element() != ""))
            .list.join(",")
            .alias("symbols")
        )
        .drop("syms")
    )
    merged = (
        combined.drop("symbols")
        .unique(subset=["uid"], keep="first", maintain_order=True)
        .join(unioned, on="uid", how="left")
        .sort("ts", "uid")
    )
    merged.select(list(SCHEMA)).write_parquet(path)
    return max(merged.height - prior.height, 0)


def load(path: Path, *, since: datetime | None = None) -> list[Article]:
    """Read the archive back as articles, optionally only those after ``since``."""
    path = Path(path)
    if not path.exists():
        return []
    df = pl.read_parquet(path)
    if since is not None:
        cutoff = since.astimezone(UTC) if since.tzinfo else since.replace(tzinfo=UTC)
        df = df.filter(pl.col("ts") >= cutoff)
    return [
        Article(
            uid=r["uid"],
            ts=r["ts"],
            source=r["source"],
            kind=r["kind"],
            title=r["title"],
            summary=r["summary"] or "",
            url=r["url"] or "",
            symbols=tuple(s for s in (r["symbols"] or "").split(",") if s),
        )
        for r in df.iter_rows(named=True)
    ]
