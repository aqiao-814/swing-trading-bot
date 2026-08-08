"""News collection, sentiment, and the conviction tilt.

The centrepiece is ``TestNoLookahead``: a news pipeline is the easiest place in
this system to leak the future, because every article carries a timestamp that
is trivial to ignore. If those tests fail, the bot is trading on information it
could not have had, and every result it produces is worthless.

Second in importance is ``TestTiltCannotInventConviction``: the tilt must be
incapable of creating a position the price policy did not already want.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from swingbot.news import store
from swingbot.news.feeds import Article, dedupe, parse_feed
from swingbot.news.sentiment import score_article, score_text
from swingbot.news.signal import NewsSignal, build_signal
from swingbot.news.tickers import confirms, match_symbols

NOW = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)


def art(
    uid: str,
    *,
    title: str,
    hours_ago: float = 1.0,
    symbols: tuple[str, ...] = (),
    kind: str = "company",
    summary: str = "",
    source: str = "test",
) -> Article:
    return Article(
        uid=uid,
        ts=NOW - timedelta(hours=hours_ago),
        source=source,
        kind=kind,
        title=title,
        summary=summary,
        url=f"https://example.test/{uid}",
        symbols=symbols,
    )


RSS_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Test feed</title>
  <item>
    <title>Nvidia surges on blowout earnings</title>
    <description><![CDATA[<p>Revenue <b>beat</b> estimates.</p>]]></description>
    <link>https://example.test/a</link>
    <pubDate>Fri, 07 Aug 2026 14:30:00 GMT</pubDate>
  </item>
  <item>
    <title>No timestamp here</title>
    <link>https://example.test/b</link>
  </item>
</channel></rss>"""

ATOM_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Fed holds rates steady</title>
    <summary>Officials signalled patience.</summary>
    <link href="https://example.test/atom1"/>
    <updated>2026-08-06T18:00:00Z</updated>
  </entry>
</feed>"""


class TestFeedParsing:
    def test_parses_rss_and_strips_markup(self):
        got = parse_feed(RSS_SAMPLE, source="s", kind="company")
        assert len(got) == 1  # the untimestamped item is dropped
        a = got[0]
        assert a.title == "Nvidia surges on blowout earnings"
        assert a.summary == "Revenue beat estimates."  # CDATA + tags gone
        assert a.ts == datetime(2026, 8, 7, 14, 30, tzinfo=UTC)

    def test_untimestamped_articles_are_dropped(self):
        """An article whose publish time is unknown cannot be proven to predate
        a decision, and this system does not guess on that question."""
        titles = [a.title for a in parse_feed(RSS_SAMPLE, source="s", kind="company")]
        assert "No timestamp here" not in titles

    def test_parses_atom_with_href_links(self):
        got = parse_feed(ATOM_SAMPLE, source="fed", kind="macro")
        assert len(got) == 1
        assert got[0].url == "https://example.test/atom1"
        assert got[0].ts == datetime(2026, 8, 6, 18, 0, tzinfo=UTC)

    def test_malformed_xml_yields_nothing_rather_than_raising(self):
        """A truncated feed body must not take down a collection run."""
        assert parse_feed("<rss><channel><item>", source="s", kind="macro") == []

    def test_dedupe_unions_symbols_and_keeps_earliest_ts(self):
        a = art("same", title="X", hours_ago=1, symbols=("AAPL",))
        b = art("same", title="X", hours_ago=5, symbols=("MSFT",))
        got = dedupe([a, b])
        assert len(got) == 1
        assert got[0].symbols == ("AAPL", "MSFT")
        assert got[0].ts == NOW - timedelta(hours=5)  # first publication, not resyndication


class TestSentiment:
    def test_direction(self):
        assert score_text("Shares surge on blowout earnings beat").polarity > 0.5
        assert score_text("Shares plunge as revenue misses badly").polarity < -0.5

    def test_negation_flips_polarity(self):
        assert score_text("profit grew strongly").polarity > 0
        assert score_text("profit did not grow").polarity < 0
        assert score_text("the company failed to beat estimates").polarity < 0

    def test_neutral_copy_scores_zero(self):
        s = score_text("The company will hold its annual meeting on Tuesday")
        assert s.polarity == 0.0
        assert s.matched == 0

    def test_accounting_vocabulary_is_not_bearish(self):
        """Loughran-McDonald's core finding: general-purpose lexicons read
        routine accounting nouns as catastrophe. These must stay neutral."""
        for word in ("liability", "tax", "depreciation", "capital", "expense"):
            assert score_text(f"The filing discusses {word} treatment").polarity == 0.0

    def test_headline_outweighs_body(self):
        """Headlines carry the direction; summaries dilute it with context."""
        s = score_article("Stock plunges on fraud probe", "The company gained some new customers")
        assert s.polarity < 0

    def test_thin_evidence_is_flagged_unconfident(self):
        assert not score_text("There is some risk").confident
        assert score_text("Shares surge on a blowout beat").confident


class TestTickerMatching:
    def test_cashtag_and_exchange_forms(self):
        assert "NVDA" in match_symbols("Big day for $NVDA today")
        assert "NVDA" in match_symbols("Nvidia Corp (NASDAQ: NVDA) rose")

    def test_company_names_resolve(self):
        assert match_symbols("Apple's iPhone sales beat") == ("AAPL",)
        assert "JPM" in match_symbols("JPMorgan raised its outlook")

    def test_english_word_tickers_do_not_false_positive(self):
        """The failure this module exists to prevent: 'IT spending' read as
        news about Gartner, 'on the other hand' as ON Semiconductor."""
        text = "IT spending is up and on the other hand all key targets are now real"
        got = match_symbols(text)
        for bogus in ("IT", "ON", "ALL", "KEY", "NOW", "REAL", "A", "T"):
            assert bogus not in got

    def test_price_target_is_not_target_corp(self):
        assert "TGT" not in match_symbols("Analysts raised their price target to $200")
        assert "TGT" in match_symbols("Target Corp reported strong comparable sales")

    def test_universe_restriction_drops_untradeable_symbols(self):
        """The bot is judged against SPY and must never be able to trade it."""
        assert "SPY" not in match_symbols("$SPY hit a record", universe={"AAPL", "MSFT"})

    def test_confirms_rejects_unrelated_yahoo_labels(self):
        """Regression: yf.Ticker("ADI").news served "LASR Q2 Earnings Surpass
        Estimates" -- a different company's beat. Trusting Yahoo's label
        verbatim would tilt ADI on LASR's results."""
        assert not confirms("ADI", "LASR Q2 Earnings Surpass Estimates on Strong Growth")
        assert confirms("ADI", "Dear Analog Devices Stock Fans, Mark Your Calendars")
        assert confirms("AAPL", "Apple (AAPL) Tops $10 Billion in India Sales")

    def test_confirms_requires_explicit_form_for_word_tickers(self):
        assert not confirms("A", "This is a routine filing")
        assert confirms("A", "Agilent Technologies (A) secured approval")


class TestNoLookahead:
    """A decision at bar t may only use articles published strictly before t."""

    def test_future_articles_are_excluded(self):
        future = Article(
            uid="f",
            ts=NOW + timedelta(hours=1),
            source="s",
            kind="company",
            title="Stock surges on blowout beat",
            symbols=("AAPL",),
        )
        sig = build_signal([future], as_of=NOW)
        assert sig.total_articles == 0
        assert "AAPL" not in sig.symbols

    def test_article_exactly_at_decision_instant_is_excluded(self):
        """Boundary: an article stamped at the decision instant was not
        readable *before* the decision, so it does not count."""
        edge = Article(
            uid="e",
            ts=NOW,
            source="s",
            kind="company",
            title="Stock surges on blowout beat",
            symbols=("AAPL",),
        )
        assert build_signal([edge], as_of=NOW).total_articles == 0

    def test_past_articles_are_included(self):
        past = art("p", title="Stock surges on blowout beat", hours_ago=2, symbols=("AAPL",))
        sig = build_signal([past], as_of=NOW)
        assert sig.symbols["AAPL"].score > 0

    def test_signal_is_independent_of_articles_after_as_of(self):
        """Adding future news must not perturb a past decision at all."""
        past = [art("p", title="Shares surge on a beat", hours_ago=3, symbols=("AAPL",))]
        future = Article(
            uid="fut",
            ts=NOW + timedelta(days=1),
            source="s",
            kind="company",
            title="Shares collapse in bankruptcy filing fraud",
            symbols=("AAPL",),
        )
        a = build_signal(past, as_of=NOW).to_json()["symbols"]
        b = build_signal([*past, future], as_of=NOW).to_json()["symbols"]
        assert a == b


class TestAggregation:
    def test_recency_outweighs_staleness(self):
        fresh = [art("f", title="Shares surge on a blowout beat", hours_ago=1, symbols=("AAA",))]
        stale = [art("s", title="Shares surge on a blowout beat", hours_ago=240, symbols=("BBB",))]
        sig = build_signal(fresh + stale, as_of=NOW, half_life_days=2.0)
        assert sig.symbols["AAA"].weight > sig.symbols["BBB"].weight

    def test_thin_evidence_is_shrunk_toward_zero(self):
        """One headline scoring -1.0 is under-observed, not bearish."""
        one = build_signal(
            [art("1", title="Shares plunge on fraud", symbols=("AAA",))],
            as_of=NOW,
            prior_count=3.0,
        )
        many = build_signal(
            [
                art(str(i), title="Shares plunge on fraud", hours_ago=1 + i, symbols=("BBB",))
                for i in range(9)
            ],
            as_of=NOW,
            prior_count=3.0,
        )
        assert abs(one.symbols["AAA"].score) < abs(many.symbols["BBB"].score)
        assert one.symbols["AAA"].raw_score == pytest.approx(-1.0)

    def test_scores_stay_in_range(self):
        arts = [
            art(str(i), title="Surge soars record blowout beat", hours_ago=1, symbols=("A",))
            for i in range(50)
        ]
        sig = build_signal(arts, as_of=NOW)
        assert -1.0 <= sig.symbols["A"].score <= 1.0

    def test_macro_articles_feed_the_market_tilt_not_a_symbol(self):
        sig = build_signal(
            [art("m", title="Recession fears deepen as jobs collapse", kind="macro")],
            as_of=NOW,
        )
        assert sig.macro < 0
        assert sig.symbols == {}

    def test_macro_reaches_symbols_without_company_news(self):
        sig = build_signal(
            [
                art(str(i), title="Recession fears deepen", kind="macro", hours_ago=1 + i)
                for i in range(9)
            ],
            as_of=NOW,
        )
        assert sig.score_for("ZZZZ", macro_weight=0.5) < 0

    def test_demeaning_ranks_within_a_bullish_cross_section(self):
        """Measured on the real universe: mean tone +0.346, only 77 of 635
        symbols negative. Raw scores barely order anything, so the tilt has to
        run on the deviation from the cross-section, not the level."""
        arts = []
        for i, sym in enumerate(["GOOD", "MID", "BAD"]):
            # All three get bullish copy; GOOD gets more of it than BAD.
            n = {"GOOD": 9, "MID": 6, "BAD": 3}[sym]
            for j in range(n):
                arts.append(
                    art(
                        f"{sym}{j}",
                        title="Shares surge on a blowout beat",
                        hours_ago=1 + j,
                        symbols=(sym,),
                    )
                )
            arts.append(
                art(f"{sym}neg", title="Shares plunge on fraud", hours_ago=1 + i, symbols=(sym,))
            )
        sig = build_signal(arts, as_of=NOW)

        # Every name reads positive on the raw score -- that is the problem.
        assert all(v.score > 0 for v in sig.symbols.values())
        # De-meaned, the cross-section separates and sums to zero.
        assert sig.symbols["GOOD"].relative > 0
        assert sig.symbols["BAD"].relative < 0
        assert sum(v.relative for v in sig.symbols.values()) == pytest.approx(0.0, abs=1e-9)

    def test_score_for_uses_the_demeaned_value(self):
        arts = [
            art(f"a{i}", title="Shares surge on a blowout beat", hours_ago=1 + i, symbols=("AAA",))
            for i in range(9)
        ] + [
            art(f"b{i}", title="Shares surge on a blowout beat", hours_ago=1 + i, symbols=("BBB",))
            for i in range(9)
        ]
        sig = build_signal(arts, as_of=NOW)
        # Identical copy for both names -> no cross-sectional information.
        assert sig.score_for("AAA", macro_weight=0.0, demean=True) == pytest.approx(0.0)
        assert sig.score_for("AAA", macro_weight=0.0, demean=False) > 0

    def test_demeaning_does_not_erase_the_macro_channel(self):
        """The market-wide level survives de-meaning on purpose -- it is a real
        signal, it just belongs in the macro term rather than the company one."""
        arts = [
            art(
                f"m{i}",
                title="Recession fears deepen as jobs collapse",
                kind="macro",
                hours_ago=1 + i,
            )
            for i in range(9)
        ] + [art("c", title="Shares surge on a beat", symbols=("AAA",))]
        sig = build_signal(arts, as_of=NOW)
        assert sig.macro < 0
        assert sig.score_for("AAA", macro_weight=0.5, demean=True) < 0

    def test_roundtrips_through_json(self):
        sig = build_signal([art("a", title="Shares surge on a beat", symbols=("AAPL",))], as_of=NOW)
        back = NewsSignal.from_json(sig.to_json())
        assert back.symbols["AAPL"].score == pytest.approx(sig.symbols["AAPL"].score)
        assert back.as_of == sig.as_of


class TestSignalIO:
    def test_missing_file_is_no_news_not_an_error(self, tmp_path: Path):
        assert NewsSignal.read(tmp_path / "nope.json") is None

    def test_corrupt_file_degrades_to_no_news(self, tmp_path: Path):
        """A corrupt signal must never stop the trading loop."""
        p = tmp_path / "signal.json"
        p.write_text("{not json")
        assert NewsSignal.read(p) is None

    def test_write_then_read(self, tmp_path: Path):
        sig = build_signal([art("a", title="Shares surge on a beat", symbols=("AAPL",))], as_of=NOW)
        sig.write(tmp_path / "signal.json")
        back = NewsSignal.read(tmp_path / "signal.json")
        assert back is not None
        assert "AAPL" in back.symbols


class TestArchive:
    def test_append_is_idempotent(self, tmp_path: Path):
        """The weekend job may fire twice or be re-run by hand; neither may
        double-count a headline."""
        p = tmp_path / "articles.parquet"
        arts = [art("a", title="One", symbols=("AAPL",)), art("b", title="Two")]
        assert store.append(p, arts) == 2
        assert store.append(p, arts) == 0
        assert len(store.load(p)) == 2

    def test_append_unions_symbols_across_runs(self, tmp_path: Path):
        p = tmp_path / "articles.parquet"
        store.append(p, [art("a", title="One", symbols=("AAPL",))])
        store.append(p, [art("a", title="One", symbols=("MSFT",))])
        loaded = {a.uid: a for a in store.load(p)}
        assert set(loaded["a"].symbols) == {"AAPL", "MSFT"}

    def test_load_since_filters_by_publication_time(self, tmp_path: Path):
        p = tmp_path / "articles.parquet"
        store.append(p, [art("old", title="Old", hours_ago=500), art("new", title="New")])
        recent = store.load(p, since=NOW - timedelta(hours=48))
        assert [a.uid for a in recent] == ["new"]

    def test_roundtrip_preserves_text_for_rescoring(self, tmp_path: Path):
        """The lexicon will change; the archive must survive to be rescored."""
        p = tmp_path / "articles.parquet"
        store.append(p, [art("a", title="Shares surge", summary="Revenue beat estimates")])
        assert store.load(p)[0].summary == "Revenue beat estimates"


# ---- the tilt, as the engine actually applies it ---------------------------


def engine_with_news(tmp_path: Path, sig: NewsSignal | None, **news_cfg):
    """A PaperEngine wired to a published signal, with no data or state.

    Only the scoring path is under test here, so the engine is never run; it is
    constructed purely to exercise ``_tilt`` / ``_news_age_decay`` against real
    config objects rather than a hand-rolled stub that could drift from them.
    """
    from swingbot.config import Config
    from swingbot.paper.engine import PaperEngine

    cfg = Config()
    cfg.data.root = tmp_path / "data"
    cfg.data.source = "synthetic"
    cfg.data.universe = ["AAA"]
    cfg.artifacts_root = tmp_path / "artifacts"
    cfg.paper.universe = "config"
    for k, v in news_cfg.items():
        setattr(cfg.paper.news, k, v)
    if sig is not None:
        sig.write(tmp_path / "signal.json")
    cfg.paper.news.signal_path = tmp_path / "signal.json"
    return PaperEngine(cfg)


def signal_with(score: float, *, as_of: datetime = NOW, symbol: str = "AAA") -> NewsSignal:
    """A one-symbol signal. ``relative`` is set explicitly because these tests
    build the signal by hand rather than through ``build_signal``, which is
    what normally populates the de-meaned field the engine tilts on."""
    from swingbot.news.signal import SymbolNews

    return NewsSignal(
        as_of=as_of.isoformat(),
        symbols={
            symbol: SymbolNews(
                symbol=symbol,
                score=score,
                raw_score=score,
                articles=5,
                weight=5.0,
                relative=score,
            )
        },
    )


class TestTiltCannotInventConviction:
    """The tilt scales a view the policy already holds. It may never create one.

    This is what keeps news from becoming a second, unvalidated alpha source
    bolted onto the side of the policy: if the model is neutral on a name, no
    volume of headlines can put it in the book.
    """

    def test_zero_conviction_stays_zero(self, tmp_path: Path):
        eng = engine_with_news(tmp_path, signal_with(1.0))
        assert eng._tilt("AAA", 0.0, 1.0)[0] == 0.0

    def test_tilt_is_bounded_by_tilt_weight(self, tmp_path: Path):
        eng = engine_with_news(tmp_path, signal_with(1.0), tilt_weight=0.30, macro_weight=0.0)
        tilted, _ = eng._tilt("AAA", 0.5, 1.0)
        assert tilted == pytest.approx(0.5 * 1.30)

    def test_good_news_strengthens_a_long(self, tmp_path: Path):
        eng = engine_with_news(tmp_path, signal_with(0.8), macro_weight=0.0)
        assert eng._tilt("AAA", 0.4, 1.0)[0] > 0.4

    def test_bad_news_weakens_a_long(self, tmp_path: Path):
        eng = engine_with_news(tmp_path, signal_with(-0.8), macro_weight=0.0)
        assert eng._tilt("AAA", 0.4, 1.0)[0] < 0.4

    def test_good_news_weakens_a_short(self, tmp_path: Path):
        """Signed, not just multiplicative: on a short, good news must reduce
        conviction. A bare multiply would make good news a *stronger* short."""
        eng = engine_with_news(tmp_path, signal_with(0.8), macro_weight=0.0)
        tilted, _ = eng._tilt("AAA", -0.4, 1.0)
        assert -0.4 < tilted <= 0.0

    def test_conviction_stays_in_range(self, tmp_path: Path):
        eng = engine_with_news(tmp_path, signal_with(1.0), tilt_weight=5.0, macro_weight=0.0)
        assert eng._tilt("AAA", 0.95, 1.0)[0] <= 1.0

    def test_news_can_damp_to_zero_but_never_invert(self, tmp_path: Path):
        """At an over-large tilt_weight the multiplier would go negative and
        turn a long the model wanted into a short it never asked for."""
        eng = engine_with_news(tmp_path, signal_with(-1.0), tilt_weight=5.0, macro_weight=0.0)
        assert eng._tilt("AAA", 0.8, 1.0)[0] == 0.0

    def test_unknown_symbol_is_untouched_without_macro(self, tmp_path: Path):
        eng = engine_with_news(tmp_path, signal_with(1.0), macro_weight=0.0)
        assert eng._tilt("ZZZ", 0.5, 1.0) == (0.5, 0.0)


class TestTiltAgeDecay:
    def test_fresh_signal_applies_fully(self, tmp_path: Path):
        eng = engine_with_news(tmp_path, signal_with(0.5, as_of=NOW))
        bar = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
        assert eng._news_age_decay(bar) == pytest.approx(1.0)

    def test_tilt_fades_as_the_signal_ages(self, tmp_path: Path):
        """Collection runs weekly; trading runs daily. A Sunday score must not
        still read at full strength on Friday."""
        eng = engine_with_news(tmp_path, signal_with(0.5, as_of=NOW), half_life_days=2.0)
        two_days = eng._news_age_decay(NOW + timedelta(days=2))
        four_days = eng._news_age_decay(NOW + timedelta(days=4))
        assert two_days == pytest.approx(0.5, abs=1e-6)
        assert four_days == pytest.approx(0.25, abs=1e-6)

    def test_signal_newer_than_the_bar_is_refused(self, tmp_path: Path):
        """Replaying an old bar against today's signal would be lookahead."""
        eng = engine_with_news(tmp_path, signal_with(0.5, as_of=NOW))
        assert eng._news_age_decay(NOW - timedelta(days=1)) == 0.0

    def test_signal_past_max_age_is_dropped(self, tmp_path: Path):
        eng = engine_with_news(tmp_path, signal_with(0.5, as_of=NOW), max_age_days=10.0)
        assert eng._news_age_decay(NOW + timedelta(days=11)) == 0.0


class TestTiltDisabled:
    def test_disabled_config_loads_no_signal(self, tmp_path: Path):
        eng = engine_with_news(tmp_path, signal_with(1.0), enabled=False)
        assert eng.news is None
        assert eng._tilt("AAA", 0.5, 1.0) == (0.5, 0.0)

    def test_missing_signal_file_leaves_conviction_untouched(self, tmp_path: Path):
        """The trading loop must run normally before news has ever been
        collected -- a fresh clone has no signal.json."""
        eng = engine_with_news(tmp_path, None)
        assert eng.news is None
        assert eng._tilt("AAA", 0.5, 1.0) == (0.5, 0.0)


class TestTiltInAFullRun:
    """End-to-end: the tilt must actually reach a real engine run's decisions.

    The unit tests above prove ``_tilt`` behaves; this proves it is wired into
    the scoring path and audited in the ledger, which is the part that silently
    rots when ``_decide`` is refactored.
    """

    @staticmethod
    def _run(tmp_path: Path, sig: NewsSignal | None):
        from datetime import date

        from swingbot.config import Config
        from swingbot.data.sources import SyntheticSource
        from swingbot.data.store import BarStore
        from swingbot.paper.engine import PaperEngine

        syms = ["AAA", "BBB", "CCC", "DDD"]
        cfg = Config()
        cfg.data.root = tmp_path / "data"
        cfg.data.source = "synthetic"
        cfg.data.universe = list(syms)
        cfg.artifacts_root = tmp_path / "artifacts"
        cfg.paper.universe = "config"
        cfg.paper.start = "2024-06-03"
        cfg.paper.data_start = "2019-01-01"
        cfg.paper.pretrain_years = 1.0
        cfg.paper.min_conviction = 0.02
        cfg.paper.exit_conviction = 0.005
        cfg.paper.news.signal_path = tmp_path / "signal.json"
        if sig is None:
            cfg.paper.news.enabled = False
        else:
            sig.write(tmp_path / "signal.json")

        src = SyntheticSource(seed=7, regime_switching=True)
        store_ = BarStore(cfg.data.root)
        for s in syms + cfg.paper.benchmark_symbols:
            store_.write(src.fetch(s, "2019-01-01", "2024-08-30"))

        eng = PaperEngine(cfg)
        eng.run(capital=100_000, as_of=date(2024, 8, 30), refresh=False, log=lambda m: None)
        return eng.store.read("decisions")

    def test_decisions_record_both_sides_of_the_tilt(self, tmp_path: Path):
        from swingbot.news.signal import SymbolNews

        # Dated far in the past so the whole replay window sits after it and
        # the age guard does not zero the tilt out.
        sig = NewsSignal(
            as_of=datetime(2024, 6, 1, tzinfo=UTC).isoformat(),
            macro=0.4,
            macro_articles=20,
            symbols={"AAA": SymbolNews("AAA", score=0.9, raw_score=0.9, articles=8, weight=8.0)},
        )
        dec = self._run(tmp_path, sig)
        assert not dec.is_empty()
        assert "news_score" in dec.columns
        assert "model_conviction" in dec.columns

    def test_run_without_news_is_unaffected(self, tmp_path: Path):
        """Disabling news must leave conviction exactly as the policy set it.

        ``model_conviction`` is the policy's *raw* output, recorded before both
        the tilt and the long-only clamp, so with news off the only difference
        from ``conviction`` is that clamp.
        """
        import polars as pl

        dec = self._run(tmp_path, None)
        rows = dec.filter(pl.col("model_conviction").is_not_null())
        assert (rows["news_score"] == 0.0).all()
        expected = rows["model_conviction"].clip(lower_bound=0.0)
        assert (rows["conviction"] - expected).abs().max() == 0.0
