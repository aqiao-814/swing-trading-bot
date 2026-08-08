"""Aggregate scored articles into a per-symbol news signal.

The output is a small JSON-serialisable object: one score in [-1, +1] per
symbol, plus a market-wide macro score, plus the evidence counts behind each.
That is deliberately the *whole* interface -- the engine consumes a number per
symbol and nothing else, so news can never reach into position sizing directly.

Four properties this module is responsible for:

**1. No lookahead, enforced not assumed.** ``build_signal`` takes an ``as_of``
instant and silently drops every article published at or after it. The paper
loop's central invariant is that a decision on bar *t* may only use information
available at bar *t*; a news pipeline is the easiest place in the whole system
to break that, because articles arrive with a timestamp that is trivial to
ignore. The filter is unconditional and there is no flag to turn it off.

**2. Recency beats volume.** Sentiment decays exponentially with a half-life in
days. A week-old earnings headline should not carry the same weight as this
morning's; without decay, a stock with one big old story outranks a stock with
three fresh ones.

**3. Thin evidence is shrunk toward zero.** A symbol with one matched article
scoring -1.0 is not bearish, it is under-observed. Scores are shrunk by
``n / (n + prior_count)``, so conviction in the news score grows with the
evidence rather than jumping to full scale on a single headline. This is the
same instinct as the RankIC calibration note in ``agents/ranker.py``: a signal
that looks too strong on too little data is a bug, not an edge.

**4. The cross-section, not the level.** Company scores are de-meaned across
symbols before the engine sees them, because financial copy is overwhelmingly
bullish (measured: mean tone +0.346, only 77 of 635 names negative) and the raw
level therefore orders almost nothing. See ``NewsSignal.score_for``.

The macro score exists because most of what moves a wide book on any given day
is not company news -- it is the Fed, CPI, jobs, tariffs. It is computed over
articles that resolved to no particular symbol and applied as a uniform tilt,
and it is deliberately the one place a market-wide *level* is allowed to live.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from swingbot.news.feeds import Article
from swingbot.news.sentiment import score_article

# Half-life of a headline's relevance. Two days is a compromise: shorter and a
# Saturday collection is already stale by Tuesday, longer and stale earnings
# copy dominates the score for a week.
DEFAULT_HALF_LIFE_DAYS = 2.0

# Pseudo-count for shrinkage. At 3, a symbol needs ~3 articles to reach half
# the raw score and ~9 to reach 75% of it.
DEFAULT_PRIOR_COUNT = 3.0


@dataclass
class SymbolNews:
    """Per-symbol aggregate and the evidence behind it."""

    symbol: str
    score: float  # shrunk, decayed polarity in [-1, +1]
    raw_score: float  # before shrinkage
    articles: int
    weight: float  # summed decay*confidence weight
    top_headline: str = ""
    # ``score`` minus the cross-sectional mean of all symbols in this signal.
    # This is what the engine actually tilts on -- see NewsSignal.score_for.
    relative: float = 0.0


@dataclass
class NewsSignal:
    """Everything the engine is allowed to know about the news."""

    as_of: str  # ISO-8601 UTC instant the signal is valid at
    macro: float = 0.0
    macro_articles: int = 0
    symbols: dict[str, SymbolNews] = field(default_factory=dict)
    total_articles: int = 0
    sources: dict[str, int] = field(default_factory=dict)
    # Cross-sectional mean of the company scores, subtracted to form ``relative``.
    mean_score: float = 0.0

    def score_for(self, symbol: str, *, macro_weight: float = 0.5, demean: bool = True) -> float:
        """Combined news view of one symbol, in [-1, +1].

        Company-specific tone plus a fraction of the market-wide tone. A symbol
        with no company news of its own still inherits the macro backdrop,
        which is the honest reading: on a day the Fed surprises, every name is
        affected whether or not a journalist wrote about it.

        **The company term is cross-sectionally de-meaned by default**, and
        that is not a detail. Measured over the full 670-name universe on
        2026-08-08: mean symbol score +0.346, median +0.400, and only 77 of 635
        symbols negative. Financial copy is overwhelmingly bullish in tone --
        earnings-call coverage especially -- so the raw score answers "is the
        press positive about this company?", to which the answer is almost
        always yes, and it barely orders the cross-section at all.

        Subtracting the mean turns it into "does the press like this name more
        than the average name?", which is the question a ranking actually
        needs. It is the same correction ``agents/ranker.py`` makes by
        predicting excess rather than raw return: a signal that can win by
        saying yes to everything is not a signal.

        The uniform component is not simply discarded -- it survives, on
        purpose, in the ``macro`` term, where a market-wide mood belongs.
        """
        own = 0.0
        if symbol in self.symbols:
            v = self.symbols[symbol]
            own = v.relative if demean else v.score
        combined = own + macro_weight * self.macro
        return float(max(-1.0, min(1.0, combined)))

    def to_json(self) -> dict:
        return {
            "as_of": self.as_of,
            "macro": round(self.macro, 6),
            "macro_articles": self.macro_articles,
            "total_articles": self.total_articles,
            "mean_score": round(self.mean_score, 6),
            "sources": self.sources,
            "symbols": {
                s: {
                    "score": round(v.score, 6),
                    "relative": round(v.relative, 6),
                    "raw_score": round(v.raw_score, 6),
                    "articles": v.articles,
                    "weight": round(v.weight, 6),
                    "top_headline": v.top_headline,
                }
                for s, v in sorted(self.symbols.items())
            },
        }

    @classmethod
    def from_json(cls, data: dict) -> NewsSignal:
        return cls(
            as_of=data["as_of"],
            macro=float(data.get("macro", 0.0)),
            macro_articles=int(data.get("macro_articles", 0)),
            total_articles=int(data.get("total_articles", 0)),
            mean_score=float(data.get("mean_score", 0.0)),
            sources=dict(data.get("sources", {})),
            symbols={
                s: SymbolNews(
                    symbol=s,
                    score=float(v["score"]),
                    raw_score=float(v.get("raw_score", v["score"])),
                    articles=int(v.get("articles", 0)),
                    weight=float(v.get("weight", 0.0)),
                    top_headline=v.get("top_headline", ""),
                    # Absent in signals written before de-meaning existed; those
                    # fall back to the raw score rather than silently tilting on
                    # a zero.
                    relative=float(v.get("relative", v["score"])),
                )
                for s, v in data.get("symbols", {}).items()
            },
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # sort_keys so a weekend commit diffs cleanly against the last one.
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: Path) -> NewsSignal | None:
        """Load a published signal; None when absent or unreadable.

        Unreadable is deliberately not fatal. The engine treats a missing news
        signal as "no news", and a corrupt file must degrade to the same thing
        rather than stopping the trading loop.
        """
        try:
            return cls.from_json(json.loads(Path(path).read_text()))
        except (OSError, ValueError, KeyError):
            return None


def _decay(age_days: float, half_life: float) -> float:
    if half_life <= 0:
        return 1.0
    return math.pow(0.5, max(age_days, 0.0) / half_life)


def build_signal(
    articles: list[Article],
    *,
    as_of: datetime | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    prior_count: float = DEFAULT_PRIOR_COUNT,
    max_age_days: float = 14.0,
) -> NewsSignal:
    """Fold articles into a signal valid at ``as_of``.

    Articles published at or after ``as_of``, or older than ``max_age_days``,
    are excluded. The first of those rules is the no-lookahead guarantee; the
    second just stops a stale feed entry from lingering forever at a tiny decay
    weight.
    """
    now = as_of or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    per_symbol: dict[str, list[tuple[float, float, str]]] = {}  # sym -> (w, polarity, headline)
    macro_num = macro_den = 0.0
    macro_n = 0
    sources: dict[str, int] = {}
    used = 0

    for a in articles:
        ts = a.ts if a.ts.tzinfo else a.ts.replace(tzinfo=UTC)
        if ts >= now:
            continue  # strictly in the future relative to the decision instant
        age = (now - ts).total_seconds() / 86400.0
        if age > max_age_days:
            continue

        s = score_article(a.title, a.summary)
        if s.matched == 0:
            continue  # no directional words: carries no tone, only noise

        used += 1
        sources[a.source] = sources.get(a.source, 0) + 1

        # Confidence rises with the amount of evidence in the copy but saturates
        # quickly -- a headline with six sentiment words is not three times more
        # informative than one with two.
        confidence = min(s.matched, 6) / 6.0
        w = _decay(age, half_life_days) * confidence

        if a.symbols:
            for sym in a.symbols:
                per_symbol.setdefault(sym, []).append((w, s.polarity, a.title))
        elif a.kind == "macro":
            macro_num += w * s.polarity
            macro_den += w
            macro_n += 1

    symbols: dict[str, SymbolNews] = {}
    for sym, rows in per_symbol.items():
        den = sum(w for w, _, _ in rows)
        if den <= 0:
            continue
        raw = sum(w * p for w, p, _ in rows) / den
        n = len(rows)
        shrunk = raw * (n / (n + prior_count))
        # The headline that drove the score hardest, for the dashboard and for
        # anyone auditing why a symbol got tilted.
        top = max(rows, key=lambda r: r[0] * abs(r[1]))[2]
        symbols[sym] = SymbolNews(
            symbol=sym,
            score=float(max(-1.0, min(1.0, shrunk))),
            raw_score=float(raw),
            articles=n,
            weight=float(den),
            top_headline=top,
        )

    # Cross-sectional de-meaning. Equal weight per symbol, not per article:
    # the ranking treats every name as one slot, so the centre it should be
    # measured against is the average *name*, not the average headline.
    mean_score = sum(v.score for v in symbols.values()) / len(symbols) if symbols else 0.0
    for v in symbols.values():
        v.relative = float(max(-1.0, min(1.0, v.score - mean_score)))

    macro_raw = macro_num / macro_den if macro_den > 0 else 0.0
    macro = macro_raw * (macro_n / (macro_n + prior_count)) if macro_n else 0.0

    return NewsSignal(
        as_of=now.isoformat(),
        macro=float(max(-1.0, min(1.0, macro))),
        macro_articles=macro_n,
        symbols=symbols,
        total_articles=used,
        sources=sources,
        mean_score=float(mean_score),
    )


def summarize(sig: NewsSignal, *, top: int = 10) -> str:
    """Human-readable digest for run logs and the weekend commit message."""
    # Ranked on the de-meaned score, because that is what the engine tilts on.
    ranked = sorted(sig.symbols.values(), key=lambda v: -abs(v.relative))
    lines = [
        f"news signal @ {sig.as_of}",
        f"  {sig.total_articles} scored articles from {len(sig.sources)} feeds",
        f"  macro tone {sig.macro:+.3f} over {sig.macro_articles} articles",
        f"  {len(sig.symbols)} symbols with company news (mean tone {sig.mean_score:+.3f})",
    ]
    if ranked:
        lines.append(f"  strongest {min(top, len(ranked))} (relative to the mean):")
        for v in ranked[:top]:
            lines.append(
                f"    {v.symbol:<6} {v.relative:+.3f}  ({v.articles} art)  {v.top_headline[:64]}"
            )
    return "\n".join(lines)
