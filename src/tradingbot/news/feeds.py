"""Free news collection: bulk macro feeds plus per-ticker company news.

**Why two tiers.** The obvious approach -- hit Yahoo's per-ticker RSS endpoint
(``feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL``) once per name -- does not
survive contact with a ~670-symbol universe. Probed 2026-08-08: the endpoint
serves the first few requests, then returns bare ``429 Too Many Requests`` for
every subsequent call, and the block is **IP-level and long-lived** -- still 429
after a 300-second cooldown, so it is a quota ban rather than a per-second rate
limit you can sleep through. Scraping it harder is both futile and rude.

Two paths that *do* work, measured the same day:

1. **Bulk feeds** (``MACRO_FEEDS``) -- CNBC's topic feeds, MarketWatch/Dow Jones,
   and the Federal Reserve press wire. Roughly 270 articles per pass, no
   throttling of any kind, no key. This is the economy-wide tier and it is the
   backbone: it is the only tier guaranteed to return something.
2. **yfinance's news API** for per-company news. yfinance negotiates a Yahoo
   cookie + crumb and reads a different endpoint than the public RSS, so it
   answers normally while raw RSS scraping is banned. It is already a hard
   dependency of this project, so the per-ticker tier costs no new install.

Company news is therefore *best-effort* and macro news is *reliable*. Every
caller here must degrade rather than fail: a dead feed returns no articles, and
a run that collects only macro news is a valid run.

Articles are deduplicated on a content hash of the URL (falling back to the
title), because the same wire story reaches several of these feeds.

Timestamps are timezone-aware UTC. Nothing downstream is allowed to use an
article before its publication instant -- see ``signal.py``.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import requests

log = logging.getLogger(__name__)

# A browser UA. CNBC and Dow Jones serve their feeds to anything, but a couple
# of the government wires 403 a bare python-requests agent.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Bulk feeds: (name, url, kind). ``kind`` marks what the feed is *about*, which
# signal.py uses to route the article -- "macro" articles move the whole-market
# tilt, "company" articles are matched to individual symbols.
#
# All eleven were probed 2026-08-08 and returned 200 with the item count noted.
# They are all free, keyless, and unthrottled.
MACRO_FEEDS: tuple[tuple[str, str, str], ...] = (
    ("cnbc-economy", "https://www.cnbc.com/id/20910258/device/rss/rss.html", "macro"),  # 30
    ("cnbc-finance", "https://www.cnbc.com/id/15839135/device/rss/rss.html", "macro"),  # 30
    ("cnbc-markets", "https://www.cnbc.com/id/10001147/device/rss/rss.html", "macro"),  # 30
    ("cnbc-business", "https://www.cnbc.com/id/100003114/device/rss/rss.html", "company"),  # 30
    ("cnbc-earnings", "https://www.cnbc.com/id/10000664/device/rss/rss.html", "company"),  # 30
    ("cnbc-tech", "https://www.cnbc.com/id/19854910/device/rss/rss.html", "company"),  # 30
    ("mw-markets", "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain", "macro"),  # 61
    ("mw-top", "https://feeds.content.dowjones.io/public/rss/mw_topstories", "macro"),  # 10
    ("mw-realtime", "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines", "company"),
    ("mw-bulletins", "https://feeds.content.dowjones.io/public/rss/mw_bulletins", "macro"),  # 10
    ("fed-press", "https://www.federalreserve.gov/feeds/press_all.xml", "macro"),  # 20
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Article:
    """One news item, normalised across feeds.

    ``symbols`` is what the article is *about*, resolved later by
    ``tickers.match_symbols`` for bulk feeds and known up-front for per-ticker
    fetches. ``kind`` is "macro" or "company".
    """

    uid: str
    ts: datetime  # tz-aware UTC publication time
    source: str
    kind: str
    title: str
    summary: str = ""
    url: str = ""
    symbols: tuple[str, ...] = field(default_factory=tuple)

    @property
    def text(self) -> str:
        """Title plus summary -- what the sentiment scorer reads."""
        return f"{self.title}. {self.summary}".strip()


def _clean(raw: str | None) -> str:
    """Strip markup and collapse whitespace; feeds mix CDATA, HTML and entities."""
    if not raw:
        return ""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", raw)).strip()


def _uid(url: str, title: str) -> str:
    """Stable id for dedupe. URL when present -- the same story syndicates
    across CNBC and MarketWatch under slightly different headlines."""
    basis = (url or title).strip().lower()
    return hashlib.sha1(basis.encode("utf-8", "replace")).hexdigest()[:16]


def _parse_ts(raw: str | None) -> datetime | None:
    """RFC-822 (RSS ``pubDate``) or ISO-8601 (Atom ``updated``), always to UTC.

    A missing or unparseable timestamp returns None and the article is dropped:
    an article whose publication time is unknown cannot be proven to predate a
    trading decision, and this system does not guess on that question.
    """
    if not raw:
        return None
    raw = raw.strip()
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_feed(xml: str | bytes, *, source: str, kind: str) -> list[Article]:
    """Parse an RSS 2.0 or Atom document into articles.

    Written against ``xml.etree`` rather than feedparser so news collection adds
    no dependency. Malformed XML yields an empty list -- feeds do occasionally
    serve a truncated body, and that must not take down a collection run.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        log.warning("news: %s served unparseable XML (%s)", source, exc)
        return []

    out: list[Article] = []
    for node in root.iter():
        if _strip_ns(node.tag) not in ("item", "entry"):
            continue
        fields: dict[str, str] = {}
        link = ""
        for child in node:
            name = _strip_ns(child.tag)
            if name == "link":
                # RSS puts the URL in the text, Atom in an href attribute.
                link = link or (child.text or "").strip() or child.get("href", "")
            elif name not in fields:
                fields[name] = child.text or ""

        ts = _parse_ts(fields.get("pubDate") or fields.get("published") or fields.get("updated"))
        title = _clean(fields.get("title"))
        if ts is None or not title:
            continue
        out.append(
            Article(
                uid=_uid(link, title),
                ts=ts,
                source=source,
                kind=kind,
                title=title,
                summary=_clean(fields.get("description") or fields.get("summary")),
                url=link,
            )
        )
    return out


def fetch_macro(
    *,
    feeds: tuple[tuple[str, str, str], ...] = MACRO_FEEDS,
    timeout: float = 20.0,
    session: requests.Session | None = None,
    log_fn=log.info,
) -> list[Article]:
    """Collect every bulk feed. Individual failures are logged and skipped.

    This tier is unthrottled in practice, but the feeds are a courtesy so the
    requests are still sequential rather than parallel.
    """
    sess = session or requests.Session()
    # Assigned, not setdefault: requests pre-populates User-Agent with
    # "python-requests/x.y", so setdefault is silently a no-op -- and CNBC
    # serves 403 to that agent while answering a browser UA with 200. This
    # exact bug cost all six CNBC feeds on the first live run.
    sess.headers["User-Agent"] = USER_AGENT
    out: list[Article] = []
    for name, url, kind in feeds:
        try:
            resp = sess.get(url, timeout=timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log_fn(f"news: feed {name} unavailable ({type(exc).__name__}) -- skipped")
            continue
        got = parse_feed(resp.content, source=name, kind=kind)
        log_fn(f"news: {name} -> {len(got)} articles")
        out.extend(got)
    return out


def fetch_ticker_news(
    symbols: list[str],
    *,
    per_symbol: int = 10,
    pause: float = 1.0,
    max_failures: int = 25,
    log_fn=log.info,
) -> list[Article]:
    """Per-company news via yfinance, best-effort.

    yfinance reads Yahoo's cookie+crumb news endpoint, which answered normally
    on 2026-08-08 while the public per-ticker RSS was IP-banned. It can still
    fail or start refusing partway through a long universe sweep, so:

    * every symbol is wrapped individually -- one bad ticker never aborts a run;
    * after ``max_failures`` consecutive failures the sweep gives up, on the
      assumption that Yahoo has started refusing us and the remaining requests
      would be pure noise against their servers;
    * ``pause`` seconds between symbols keeps the sweep polite. At the default
      1s a 670-name universe takes ~11 minutes, which is nothing on the weekend
      schedule this is designed for.

    Returns whatever it managed to collect. An empty list is a valid outcome.
    """
    try:
        import yfinance as yf
    except ImportError:  # pragma: no cover - yfinance is a hard dependency
        log_fn("news: yfinance not installed -- skipping per-company news")
        return []

    out: list[Article] = []
    consecutive = 0
    failed = 0
    for i, sym in enumerate(symbols):
        try:
            ticker = yf.Ticker(sym)
            # get_news(count=) is the current API; older yfinance only exposes
            # the .news property, which is fixed at ~10 items.
            if hasattr(ticker, "get_news"):
                raw = ticker.get_news(count=per_symbol) or []
            else:
                raw = ticker.news or []
            consecutive = 0
        except Exception as exc:  # noqa: BLE001 - yfinance raises many types
            consecutive += 1
            failed += 1
            if consecutive >= max_failures:
                log_fn(
                    f"news: {consecutive} consecutive yfinance failures at {sym} "
                    f"({type(exc).__name__}) -- abandoning per-company sweep "
                    f"after {i} of {len(symbols)} symbols"
                )
                break
            continue
        out.extend(_articles_from_yf(raw, sym))
        if pause:
            time.sleep(pause)
    log_fn(
        f"news: per-company sweep collected {len(out)} articles "
        f"over {len(symbols)} symbols ({failed} failed)"
    )
    return out


def _articles_from_yf(raw: list[dict], symbol: str) -> list[Article]:
    """Normalise yfinance's news payload.

    Two shapes exist in the wild: the current one nests everything under
    ``content`` (title / summary / pubDate / canonicalUrl), the older one is
    flat (title / providerPublishTime as a unix int). Both are handled because
    a yfinance upgrade should not silently zero out the company tier.
    """
    out: list[Article] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        body = item.get("content") if isinstance(item.get("content"), dict) else item
        title = _clean(body.get("title"))
        if not title:
            continue

        ts = _parse_ts(body.get("pubDate") or body.get("displayTime"))
        if ts is None and item.get("providerPublishTime"):
            try:
                ts = datetime.fromtimestamp(int(item["providerPublishTime"]), tz=UTC)
            except (TypeError, ValueError, OSError):
                ts = None
        if ts is None:
            continue

        url = ""
        canonical = body.get("canonicalUrl") or body.get("clickThroughUrl")
        if isinstance(canonical, dict):
            url = canonical.get("url", "")
        elif isinstance(canonical, str):
            url = canonical
        url = url or item.get("link", "") or ""

        out.append(
            Article(
                uid=_uid(url, title),
                ts=ts,
                source="yahoo",
                kind="company",
                title=title,
                summary=_clean(body.get("summary") or body.get("description")),
                url=url,
                symbols=(symbol,),
            )
        )
    return out


def dedupe(articles: list[Article]) -> list[Article]:
    """Collapse syndicated duplicates, unioning the symbols they were found under.

    The same wire story legitimately arrives once per ticker it mentions; each
    copy should contribute its symbol but only one vote of sentiment, otherwise
    a widely-syndicated headline would swamp the cross-section.
    """
    merged: dict[str, Article] = {}
    for a in sorted(articles, key=lambda x: (x.ts, x.uid)):
        prior = merged.get(a.uid)
        if prior is None:
            merged[a.uid] = a
            continue
        syms = tuple(sorted(set(prior.symbols) | set(a.symbols)))
        # Keep the earliest timestamp: that is when the information actually
        # became public, and a later syndication must not reset its decay.
        merged[a.uid] = Article(
            uid=prior.uid,
            ts=min(prior.ts, a.ts),
            source=prior.source,
            kind="company" if "company" in (prior.kind, a.kind) else prior.kind,
            title=prior.title,
            summary=prior.summary or a.summary,
            url=prior.url or a.url,
            symbols=syms,
        )
    return sorted(merged.values(), key=lambda x: (x.ts, x.uid))
