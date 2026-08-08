"""Lexicon sentiment scoring for financial headlines.

**Why a lexicon and not a model.** The alternative is a transformer (FinBERT and
friends), which means torch in the weekend job's dependency closure and a
~400MB download on every Actions run, to classify a few thousand short
headlines. It would also be a black box in a repo whose whole discipline is
that you can point at why a number came out the way it did. A lexicon is
deterministic, unit-testable, instant, and dependency-free.

**Why a *financial* lexicon.** General-purpose sentiment word lists are
actively wrong on this domain: Loughran & McDonald (2011) showed that roughly
three quarters of the negative words in the standard Harvard-IV list are not
negative in financial writing -- "liability", "tax", "cost", "capital",
"depreciation" are neutral accounting vocabulary, and a general lexicon reads a
routine 10-K as a catastrophe. The word lists below follow the LM financial
categories: words that actually carry directional information in market copy.

**What the score means.** ``score_text`` returns a polarity in [-1, +1]:
positive means the copy reads bullish. It is a *tone* measure, not a return
forecast, and the mapping from tone to position size is deliberately somebody
else's problem (``signal.py``, and ultimately a bounded tilt in the engine).

Three refinements that matter more than a bigger word list:

* **Negation.** "not profitable" must not score as bullish. A negator within
  three tokens flips the polarity of the hit.
* **Intensity.** "plunges" is stronger evidence than "slips". Words carry
  weights rather than a flat +/-1 vote.
* **Length normalisation.** Score is the net polarity over the number of
  *matched* words, not raw counts, so a long article does not outvote a
  headline simply by being long.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Weights are intensity, not confidence: 2.0 words describe large moves or
# existential events, 1.0 words describe ordinary directional drift.
#
# fmt: off -- these are word lists, packed several per line on purpose. One
# entry per line (which ruff format would impose) turns 250 words into 250
# screens and makes the categories impossible to read as a whole.
# fmt: off
POSITIVE: dict[str, float] = {
    # large moves / strong outcomes
    "surge": 2.0, "surges": 2.0, "surged": 2.0, "surging": 2.0,
    "soar": 2.0, "soars": 2.0, "soared": 2.0, "soaring": 2.0,
    "skyrocket": 2.0, "skyrockets": 2.0, "skyrocketed": 2.0,
    "jump": 1.5, "jumps": 1.5, "jumped": 1.5, "spike": 1.5, "spiked": 1.5,
    "rally": 1.5, "rallies": 1.5, "rallied": 1.5, "rallying": 1.5,
    "record": 1.5, "records": 1.0, "surpass": 1.5, "surpassed": 1.5,
    "breakthrough": 2.0, "blowout": 2.0, "boom": 1.5, "booming": 1.5,
    # beats and raises -- the highest-information phrases in earnings copy
    "beat": 1.8, "beats": 1.8, "beating": 1.5, "topped": 1.5, "tops": 1.5,
    "upgrade": 1.8, "upgrades": 1.8, "upgraded": 1.8,
    "raise": 1.2, "raised": 1.2, "raises": 1.2, "boost": 1.5, "boosted": 1.5,
    "outperform": 1.8, "outperformed": 1.8, "outperforming": 1.5,
    "exceed": 1.5, "exceeded": 1.5, "exceeds": 1.5, "exceeding": 1.5,
    # ordinary directional drift
    "gain": 1.0, "gains": 1.0, "gained": 1.0, "rise": 1.0, "rises": 1.0,
    "rose": 1.0, "rising": 1.0, "climb": 1.0, "climbs": 1.0, "climbed": 1.0,
    "advance": 1.0, "advanced": 1.0, "higher": 1.0, "up": 0.5, "upside": 1.2,
    "strength": 1.0, "strengthen": 1.2, "strong": 1.2, "stronger": 1.2,
    "growth": 0.6, "grew": 1.0, "grow": 1.0, "growing": 1.0, "expand": 1.0,
    "expansion": 1.0, "improve": 1.2, "improved": 1.2, "improvement": 1.2,
    # Topical nouns, deliberately near-neutral: "profit" and "earnings" name
    # the subject of a headline, they do not give its direction -- "Profit
    # falls 20%" is bearish copy full of the word "profit". The verb carries
    # the sign, so the noun is weighted low enough not to cancel it out.
    "profit": 0.3, "profitable": 1.2, "profits": 0.3, "earnings": 0.3,
    "bullish": 1.8, "optimism": 1.5, "optimistic": 1.5, "confidence": 1.0,
    "rebound": 1.5, "rebounded": 1.5, "recovery": 1.2, "recovered": 1.2,
    "momentum": 1.0, "buyback": 1.5, "dividend": 0.8, "expanded": 1.0,
    "approval": 1.5, "approved": 1.5, "win": 1.2, "wins": 1.2, "won": 1.0,
    "success": 1.2, "successful": 1.2, "favorable": 1.2, "positive": 1.0,
    "opportunity": 0.8, "efficient": 0.8, "innovation": 0.8, "partnership": 0.8,
    "demand": 0.5, "robust": 1.5, "solid": 1.0, "resilient": 1.2, "upbeat": 1.5,
}

NEGATIVE: dict[str, float] = {
    # large moves / existential events
    "plunge": 2.0, "plunges": 2.0, "plunged": 2.0, "plunging": 2.0,
    "crash": 2.0, "crashes": 2.0, "crashed": 2.0, "collapse": 2.0,
    "collapses": 2.0, "collapsed": 2.0, "plummet": 2.0, "plummets": 2.0,
    "plummeted": 2.0, "tumble": 1.8, "tumbles": 1.8, "tumbled": 1.8,
    "bankruptcy": 2.0, "bankrupt": 2.0, "insolvency": 2.0, "default": 1.8,
    "fraud": 2.0, "scandal": 2.0, "probe": 1.5, "investigation": 1.5,
    "lawsuit": 1.5, "sue": 1.2, "sued": 1.2, "litigation": 1.2,
    "recall": 1.5, "recalled": 1.5, "breach": 1.5, "hack": 1.5, "hacked": 1.5,
    # misses and cuts
    "miss": 1.8, "misses": 1.8, "missed": 1.8, "shortfall": 1.8,
    "downgrade": 1.8, "downgrades": 1.8, "downgraded": 1.8,
    "cut": 1.2, "cuts": 1.2, "slash": 1.8, "slashed": 1.8, "slashes": 1.8,
    "underperform": 1.8, "underperformed": 1.8, "disappoint": 1.8,
    "disappointing": 1.8, "disappointed": 1.8, "warn": 1.5, "warns": 1.5,
    "warning": 1.5, "warned": 1.5, "guidance": 0.3,
    # ordinary directional drift
    "fall": 1.0, "falls": 1.0, "fell": 1.0, "falling": 1.0, "drop": 1.0,
    "drops": 1.0, "dropped": 1.0, "decline": 1.0, "declines": 1.0,
    "declined": 1.0, "slip": 1.0, "slips": 1.0, "slipped": 1.0,
    "sink": 1.5, "sinks": 1.5, "sank": 1.5, "slump": 1.5, "slumped": 1.5,
    "lower": 1.0, "down": 0.5, "downside": 1.2, "weak": 1.2, "weaker": 1.2,
    "weakness": 1.2, "weakened": 1.2, "loss": 1.2, "losses": 1.2, "lost": 1.0,
    "deficit": 1.2, "shrink": 1.0, "shrank": 1.0, "contraction": 1.5,
    "recession": 2.0, "downturn": 1.8, "slowdown": 1.5, "stagnation": 1.5,
    "bearish": 1.8, "pessimism": 1.5, "fear": 1.5, "fears": 1.5, "panic": 2.0,
    "concern": 1.0, "concerns": 1.0, "worry": 1.2, "worries": 1.2,
    "uncertainty": 1.2, "volatility": 0.8, "risk": 0.5, "risks": 0.5,
    "layoff": 1.5, "layoffs": 1.5, "cutting": 1.0, "restructuring": 1.2,
    "delay": 1.2, "delayed": 1.2, "halt": 1.5, "halted": 1.5, "suspend": 1.5,
    "suspended": 1.5, "reject": 1.5, "rejected": 1.5, "resign": 1.5,
    "resigned": 1.5, "negative": 1.0, "trouble": 1.5, "struggle": 1.5,
    "struggles": 1.5, "struggling": 1.5, "pressure": 0.8, "headwind": 1.5,
    "headwinds": 1.5, "tariff": 1.0, "tariffs": 1.0, "sanction": 1.2,
    "sanctions": 1.2, "inflation": 0.8, "selloff": 1.8, "sell-off": 1.8,
}

# A negator within this many tokens before a hit flips its sign. Three is the
# usual choice for English scope ("not expected to beat", "failed to deliver").
NEGATION_WINDOW = 3
NEGATORS = frozenset(
    {"not", "no", "never", "none", "nor", "cannot", "cant", "wont", "without",
     "fails", "fail", "failed", "failing", "unlikely", "denies", "denied",
     "avoid", "avoids", "avoided", "less", "lacks", "lack", "lacked", "isnt",
     "arent", "wasnt", "werent", "doesnt", "dont", "didnt", "hardly", "barely"}
)
# fmt: on

_TOKEN_RE = re.compile(r"[a-z][a-z'-]*")


@dataclass(frozen=True)
class SentimentScore:
    """Polarity plus the evidence behind it."""

    polarity: float  # [-1, +1]
    matched: int  # sentiment-bearing words found
    positive: float  # summed positive weight (after negation)
    negative: float  # summed negative weight (after negation)

    @property
    def confident(self) -> bool:
        """Whether enough words matched to be worth acting on.

        One matched word in a headline is a coin flip; the aggregator uses this
        to avoid treating a single incidental "risk" as a bearish call.
        """
        return self.matched >= 2


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().replace("’", "'"))


def score_text(text: str) -> SentimentScore:
    """Score one piece of copy in [-1, +1].

    Polarity is ``(pos - neg) / (pos + neg)`` -- the *balance* of tone, not its
    volume. A headline with one strong negative word and nothing else scores
    -1.0, which is correct: everything directional it said was bearish. The
    caller decides how much a thinly-evidenced -1.0 is worth via ``matched``.
    """
    tokens = tokenize(text)
    pos = neg = 0.0
    matched = 0
    for i, tok in enumerate(tokens):
        weight = POSITIVE.get(tok)
        sign = 1.0
        if weight is None:
            weight = NEGATIVE.get(tok)
            if weight is None:
                continue
            sign = -1.0
        matched += 1
        window = tokens[max(0, i - NEGATION_WINDOW) : i]
        if any(w.replace("'", "") in NEGATORS for w in window):
            sign = -sign
        if sign > 0:
            pos += weight
        else:
            neg += weight
    total = pos + neg
    polarity = 0.0 if total == 0 else (pos - neg) / total
    return SentimentScore(polarity=polarity, matched=matched, positive=pos, negative=neg)


def score_article(title: str, summary: str = "", *, title_weight: float = 2.0) -> SentimentScore:
    """Score a headline and its body, weighting the headline more heavily.

    Headlines are written to carry the story's direction in a few words, while
    summaries dilute it with context and boilerplate. Weighting the title 2:1
    keeps the signal where the information density actually is.
    """
    t = score_text(title)
    if not summary:
        return t
    s = score_text(summary)
    pos = t.positive * title_weight + s.positive
    neg = t.negative * title_weight + s.negative
    total = pos + neg
    return SentimentScore(
        polarity=0.0 if total == 0 else (pos - neg) / total,
        matched=t.matched + s.matched,
        positive=pos,
        negative=neg,
    )
