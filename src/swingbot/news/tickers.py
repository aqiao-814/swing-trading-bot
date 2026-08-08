"""Resolve which symbols a bulk-feed article is about.

Per-ticker news arrives already labelled -- we asked Yahoo for AAPL, so it is
about AAPL. Bulk macro feeds do not: CNBC publishes "Nvidia beats on earnings",
not "NVDA". This module maps free text to symbols.

**The false-positive problem dominates.** Naively matching bare uppercase
tokens against the universe is a disaster, because a large minority of real
tickers are ordinary English words: ``A``, ``ALL``, ``CAT``, ``F``, ``IT``,
``KEY``, ``NOW``, ``ON``, ``SO``, ``T``, ``V``. "IT spending is up" would be
read as news about Gartner, "on the other hand" as news about ON Semiconductor.
A false mention is worse than a miss here: a miss costs coverage, while a false
mention injects a *wrong* directional tilt into a real position.

So mentions are only accepted from three unambiguous shapes:

1. **Cashtags** -- ``$NVDA``. Unambiguous by construction.
2. **Exchange-qualified tickers** -- ``(NASDAQ: NVDA)``, ``NYSE: NVDA``. The
   standard way financial copy introduces a ticker.
3. **Curated company names** -- ``Nvidia`` -> NVDA, matched case-sensitively on
   word boundaries. Case matters: it separates Target the retailer from a price
   target, and Visa the network from a travel visa.

Names whose capitalised form is *still* ambiguous in market copy (``Target``
appears constantly in "price target", ``Key`` in "key level") are listed in
``REQUIRES_QUALIFIER`` and only count when a corporate suffix or exchange tag
sits next to them.

The map covers the large, heavily-covered names -- which is where bulk
financial media actually spends its words. A mid-cap that CNBC never mentions
loses nothing: the per-ticker yfinance sweep in ``feeds.py`` covers the whole
universe by construction, and this module only adds the macro tier on top.
"""

from __future__ import annotations

import re

# Curated company-name -> ticker. Only names that are (a) heavily covered by
# financial media and (b) distinctive enough to match without context.
#
# fmt: off -- a name map packed several per line, same rationale as the ticker
# tuples in paper/universe.py: one entry per line makes it unreadable as a list.
# fmt: off
COMPANY_NAMES: dict[str, str] = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "NVIDIA": "NVDA",
    "Amazon": "AMZN", "Alphabet": "GOOGL", "Google": "GOOGL", "Meta": "META",
    "Facebook": "META", "Tesla": "TSLA", "Broadcom": "AVGO", "Netflix": "NFLX",
    "Oracle": "ORCL", "Salesforce": "CRM", "Adobe": "ADBE", "Intel": "INTC",
    "Cisco": "CSCO", "Qualcomm": "QCOM", "Palantir": "PLTR", "Uber": "UBER",
    "Airbnb": "ABNB", "PayPal": "PYPL", "Shopify": "SHOP", "Snowflake": "SNOW",
    "CrowdStrike": "CRWD", "Palo Alto Networks": "PANW", "Datadog": "DDOG",
    "ServiceNow": "NOW", "Workday": "WDAY", "Intuit": "INTU", "IBM": "IBM",
    "Dell": "DELL", "Hewlett Packard": "HPQ", "Micron": "MU", "Arm Holdings": "ARM",
    "Applied Materials": "AMAT", "Lam Research": "LRCX", "KLA": "KLAC",
    "Texas Instruments": "TXN", "Analog Devices": "ADI", "Marvell": "MRVL",
    "Synopsys": "SNPS", "Cadence": "CDNS", "MicroStrategy": "MSTR",
    "Advanced Micro Devices": "AMD", "Zscaler": "ZS", "Fortinet": "FTNT",
    "MongoDB": "MDB", "Autodesk": "ADSK", "Electronic Arts": "EA",
    "Take-Two": "TTWO", "Roblox": "RBLX", "Coinbase": "COIN", "Block": "SQ",
    "DoorDash": "DASH", "Booking Holdings": "BKNG", "Expedia": "EXPE",
    "Comcast": "CMCSA", "Charter Communications": "CHTR", "Disney": "DIS",
    "Warner Bros": "WBD", "Paramount": "PARA", "Spotify": "SPOT",
    # financials
    "JPMorgan": "JPM", "JP Morgan": "JPM", "Goldman Sachs": "GS",
    "Morgan Stanley": "MS", "Bank of America": "BAC", "Wells Fargo": "WFC",
    "Citigroup": "C", "Charles Schwab": "SCHW", "BlackRock": "BLK",
    "American Express": "AXP", "Mastercard": "MA", "Berkshire Hathaway": "BRK-B",
    "Capital One": "COF", "MetLife": "MET", "Prudential": "PRU",
    "Truist": "TFC", "PNC": "PNC", "U.S. Bancorp": "USB", "Blackstone": "BX",
    # healthcare
    "UnitedHealth": "UNH", "Johnson & Johnson": "JNJ", "Eli Lilly": "LLY",
    "Pfizer": "PFE", "Merck": "MRK", "AbbVie": "ABBV", "Amgen": "AMGN",
    "Bristol Myers": "BMY", "Gilead": "GILD", "Moderna": "MRNA",
    "Thermo Fisher": "TMO", "Danaher": "DHR", "Medtronic": "MDT",
    "Abbott": "ABT", "CVS": "CVS", "Cigna": "CI", "Regeneron": "REGN",
    "Vertex": "VRTX", "Biogen": "BIIB", "Intuitive Surgical": "ISRG",
    "Novo Nordisk": "NVO", "AstraZeneca": "AZN",
    # consumer / industrial / energy
    "Walmart": "WMT", "Costco": "COST", "Home Depot": "HD", "Lowe's": "LOW",
    "Nike": "NKE", "Starbucks": "SBUX", "McDonald's": "MCD", "Coca-Cola": "KO",
    "PepsiCo": "PEP", "Procter & Gamble": "PG", "Colgate": "CL",
    "Philip Morris": "PM", "Altria": "MO", "Mondelez": "MDLZ", "Kraft Heinz": "KHC",
    "Chipotle": "CMG", "Lululemon": "LULU", "TJX": "TJX", "Dollar General": "DG",
    "Boeing": "BA", "Caterpillar": "CAT", "Deere": "DE", "Lockheed Martin": "LMT",
    "RTX Corp": "RTX", "Raytheon": "RTX", "General Electric": "GE",
    "General Motors": "GM", "Ford Motor": "F", "Honeywell": "HON", "3M": "MMM",
    "United Parcel Service": "UPS", "FedEx": "FDX", "Union Pacific": "UNP",
    "Delta Air Lines": "DAL", "United Airlines": "UAL", "American Airlines": "AAL",
    "Exxon": "XOM", "ExxonMobil": "XOM", "Chevron": "CVX", "ConocoPhillips": "COP",
    "Schlumberger": "SLB", "Occidental": "OXY", "NextEra": "NEE", "Duke Energy": "DUK",
    "Southern Company": "SO", "Linde": "LIN", "Dow Inc": "DOW",
    "Freeport-McMoRan": "FCX", "Newmont": "NEM", "Nucor": "NUE",
    "Verizon": "VZ", "AT&T": "T", "T-Mobile": "TMUS", "American Tower": "AMT",
    "Simon Property": "SPG", "Marriott": "MAR", "Las Vegas Sands": "LVS",
    "Carnival": "CCL", "Royal Caribbean": "RCL", "Rivian": "RIVN",
    "Lucid": "LCID", "Plug Power": "PLUG", "First Solar": "FSLR", "Enphase": "ENPH",
}

# Names that stay ambiguous even capitalised, because market copy uses the same
# word constantly in a non-corporate sense ("price target", "key support",
# "visa requirements"). Accepted only next to a corporate suffix or exchange tag.
# Note what is deliberately NOT here: "Apple" and "Meta". Capitalised, in
# financial copy, both are unambiguously the company -- case-sensitive matching
# already separates them from "apple" the fruit and "meta-analysis". Requiring
# a corporate suffix for them would silently discard most Apple coverage,
# because journalists write "Apple's iPhone sales", never "Apple Inc's".
REQUIRES_QUALIFIER: dict[str, str] = {
    "Target": "TGT", "Visa": "V", "Key": "KEY", "Now": "NOW", "Block": "SQ",
}

_CORP_SUFFIX = r"(?:Inc|Corp|Corporation|Co|Company|Holdings|Group|Ltd|PLC|LLC|SA|NV|AG)"

# $NVDA
_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5}(?:[-.][A-Z])?)\b")
# (NASDAQ: NVDA), NYSE: NVDA, Nasdaq:NVDA
_EXCHANGE_RE = re.compile(
    r"\b(?:NASDAQ|NYSE|NYSEARCA|AMEX|BATS|OTC)\s*:\s*([A-Z]{1,5}(?:[-.][A-Z])?)\b",
    re.IGNORECASE,
)


def _name_pattern(name: str) -> re.Pattern[str]:
    """Word-boundary, case-sensitive matcher for a company name.

    ``re.escape`` matters: several names carry ``&``, ``.``, ``'`` or ``-``.
    The trailing boundary is a lookahead for a non-word char so "Apple's" and
    "Apple," both match while "Applebee's" does not.
    """
    return re.compile(rf"(?<![\w]){re.escape(name)}(?![\w])")


_NAME_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (_name_pattern(name), sym)
    for name, sym in sorted(COMPANY_NAMES.items(), key=lambda kv: -len(kv[0]))
    if name not in REQUIRES_QUALIFIER
)

_QUALIFIED_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (
        re.compile(
            rf"(?<![\w]){re.escape(name)}(?![\w])\s*(?:{_CORP_SUFFIX}\b|\(\s*(?:NASDAQ|NYSE))"
        ),
        sym,
    )
    for name, sym in sorted(REQUIRES_QUALIFIER.items(), key=lambda kv: -len(kv[0]))
)


def match_symbols(text: str, universe: set[str] | None = None) -> tuple[str, ...]:
    """Symbols mentioned in ``text``, restricted to ``universe`` when given.

    Restricting to the universe is not cosmetic: it drops the ETF and
    index-symbol noise that fills market copy (SPY, QQQ, VIX) and keeps the
    result to names the bot can actually act on. Since the paper universe
    deliberately excludes ETFs, a headline about SPY resolves to nothing --
    which is right, because the bot is judged against SPY and must not trade it.
    """
    found: set[str] = set()

    for m in _CASHTAG_RE.finditer(text):
        found.add(m.group(1).upper())
    for m in _EXCHANGE_RE.finditer(text):
        found.add(m.group(1).upper())
    for pat, sym in _NAME_PATTERNS:
        if pat.search(text):
            found.add(sym)
    for pat, sym in _QUALIFIED_PATTERNS:
        if pat.search(text):
            found.add(sym)

    if universe is not None:
        found &= universe
    return tuple(sorted(found))


# Tickers that are also ordinary English words or common abbreviations. For
# these, a bare uppercase token in prose proves nothing, so confirmation
# demands an explicit cashtag or a parenthesised/exchange-qualified form.
AMBIGUOUS_TICKERS = frozenset(
    {"A", "ALL", "AN", "ARE", "BE", "BEN", "BIG", "BY", "CAT", "CEO", "CFO", "CO",
     "D", "DD", "DAY", "EPS", "EU", "F", "FAST", "FIVE", "FOR", "GO", "GOOD", "HAS",
     "HE", "IT", "IN", "IS", "KEY", "LOW", "MA", "MAN", "NEW", "NOW", "ON", "ONE",
     "OR", "OUT", "PM", "POST", "PSA", "RE", "REAL", "RUN", "SEE", "SO", "T", "TWO",
     "UK", "UP", "US", "V", "VS", "WELL", "WM", "X", "Y", "YOU",
     # ADP publishes the private-payrolls report, so "ADP" appears in nearly
     # every macro data wrap as an economic release rather than as the stock.
     "ADP"}
)
# fmt: on


def _symbol_names() -> dict[str, set[str]]:
    rev: dict[str, set[str]] = {}
    for name, sym in {**COMPANY_NAMES, **REQUIRES_QUALIFIER}.items():
        rev.setdefault(sym, set()).add(name)
    return rev


_SYMBOL_NAMES = _symbol_names()


def confirms(symbol: str, text: str) -> bool:
    """Does ``text`` actually reference ``symbol``?

    Needed because Yahoo's per-ticker news endpoint returns *loosely related*
    stories, not only stories about the ticker you asked for. Observed on the
    first live run: ``yf.Ticker("ADI").news`` served "LASR Q2 Earnings Surpass
    Estimates", and ``yf.Ticker("ADP").news`` served a generic S&P 500 futures
    wrap. Trusting the endpoint's label verbatim would have tilted ADI on
    another company's earnings beat -- precisely the false-mention failure this
    module exists to prevent.

    Confirmation accepts an explicit ticker reference (``$ADI``, ``(ADI)``,
    ``NASDAQ: ADI``, or a bare ``ADI`` when the ticker is not an English word)
    or any curated company name for that symbol. Anything else is unconfirmed:
    the article is still archived, it just does not vote.
    """
    sym = symbol.upper()
    esc = re.escape(sym)
    if re.search(rf"\${esc}\b", text):
        return True
    if re.search(rf"\(\s*(?:(?:NASDAQ|NYSE|NYSEARCA|AMEX|OTC)\s*:\s*)?{esc}\s*\)", text, re.I):
        return True
    if re.search(rf"\b(?:NASDAQ|NYSE|NYSEARCA|AMEX|OTC)\s*:\s*{esc}\b", text, re.I):
        return True
    if sym not in AMBIGUOUS_TICKERS and re.search(rf"(?<![\w.$]){esc}(?![\w.])", text):
        return True
    return any(_name_pattern(n).search(text) for n in _SYMBOL_NAMES.get(sym, ()))


def label_articles(articles, universe: set[str] | None = None) -> list:
    """Resolve and *verify* the symbols each article is about.

    Bulk-feed articles get symbols from ``match_symbols``. Per-ticker articles
    arrive pre-labelled by Yahoo, and that label is verified rather than
    trusted -- see ``confirms``. An article that cannot be confirmed for any
    symbol keeps its place in the archive with an empty symbol set, so it can
    be rescored later if the matcher improves, but contributes nothing now.
    """
    from dataclasses import replace

    out = []
    for a in articles:
        if a.symbols:
            kept = tuple(s for s in a.symbols if confirms(s, a.text))
            # A pre-labelled article may also name other universe members; a
            # story about Nvidia's results found under AMD is about both.
            kept = tuple(sorted(set(kept) | set(match_symbols(a.text, universe))))
            out.append(a if kept == a.symbols else replace(a, symbols=kept))
            continue
        out.append(replace(a, symbols=match_symbols(a.text, universe)))
    return out
