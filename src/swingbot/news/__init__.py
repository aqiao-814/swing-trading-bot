"""News collection and sentiment signal.

Free-only, key-free news for the trading loop: bulk macro feeds (CNBC,
MarketWatch/Dow Jones, the Federal Reserve press wire) plus per-company news
through yfinance's Yahoo endpoint. See ``feeds.py`` for why the obvious Yahoo
per-ticker RSS route does not work.

The public surface is deliberately narrow: ``collect`` runs the pipeline,
``NewsSignal`` is the only thing the trading engine ever reads.
"""

from swingbot.news.collect import collect
from swingbot.news.feeds import Article
from swingbot.news.signal import NewsSignal, SymbolNews, build_signal, summarize

__all__ = [
    "Article",
    "NewsSignal",
    "SymbolNews",
    "build_signal",
    "collect",
    "summarize",
]
