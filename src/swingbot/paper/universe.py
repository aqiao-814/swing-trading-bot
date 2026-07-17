"""Stock universes for daily exploration.

Index membership lists are static snapshots (2025), not live constituents --
good enough for a research universe, and deliberately dependency-free. A few
tickers will drift out of date over time; ``fetch_many`` tolerates individual
failures, so a stale name degrades coverage rather than breaking the run.

Symbols use Yahoo notation (``BRK-B``, not ``BRK.B``).

Custom universes: pass a file path (one ticker per line, ``#`` comments) or
``config`` to use ``data.universe`` from the YAML config.
"""

from __future__ import annotations

from pathlib import Path

from swingbot.config import Config

# fmt: off
NASDAQ_100 = (
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
    "AMZN", "ANSS", "APP", "ARM", "ASML", "AVGO", "AXON", "AZN", "BIIB", "BKNG",
    "BKR", "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD",
    "CSCO", "CSGP", "CSX", "CTAS", "CTSH", "DASH", "DDOG", "DXCM", "EA", "EXC",
    "FANG", "FAST", "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX",
    "ILMN", "INTC", "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX", "LULU",
    "MAR", "MCHP", "MDB", "MDLZ", "META", "MNST", "MRVL", "MSFT", "MSTR", "MU",
    "NFLX", "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD",
    "PEP", "PLTR", "PYPL", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SNPS", "TEAM",
    "TMUS", "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL",
    "ZS",
)

SP_100 = (
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
    "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DE", "DHR", "DIS", "DUK", "EMR", "F", "FDX", "GD", "GE",
    "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU",
    "ISRG", "JNJ", "JPM", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD",
    "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE",
    "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PLTR", "PM", "PYPL",
    "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS",
    "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT",
    "XOM",
)

SP_500 = SP_100 + (
    "A", "ADI", "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AJG",
    "AKAM", "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AME", "AMP", "ANET",
    "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO", "AVB",
    "AVY", "AWK", "AXON", "AZO", "BALL", "BAX", "BBY", "BDX", "BEN", "BG",
    "BIIB", "BLDR", "BR", "BRO", "BSX", "BWA", "BX", "BXP", "CAG", "CAH",
    "CARR", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG",
    "CF", "CFG", "CHD", "CHRW", "CI", "CINF", "CLX", "CMA", "CME", "CMG",
    "CMI", "CMS", "CNC", "CNP", "COO", "COR", "CPB", "CPRT", "CPT", "CRWD",
    "CSGP", "CSX", "CTAS", "CTRA", "CTSH", "CTVA", "D", "DAL", "DASH", "DD",
    "DECK", "DFS", "DG", "DGX", "DHI", "DLR", "DLTR", "DOC", "DOV", "DOW",
    "DPZ", "DRI", "DTE", "DVA", "DVN", "DXCM", "EA", "EBAY", "ECL", "ED",
    "EFX", "EG", "EIX", "EL", "ELV", "ENPH", "EOG", "EPAM", "EQIX", "EQR",
    "EQT", "ES", "ESS", "ETN", "ETR", "EVRG", "EW", "EXC", "EXPD", "EXPE",
    "EXR", "FANG", "FAST", "FCX", "FDS", "FE", "FFIV", "FI", "FICO", "FIS",
    "FITB", "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GEHC", "GEN",
    "GEV", "GIS", "GL", "GLW", "GNRC", "GPC", "GPN", "GRMN", "GWW", "HAL",
    "HAS", "HBAN", "HCA", "HES", "HIG", "HII", "HLT", "HOLX", "HPE", "HPQ",
    "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "ICE", "IDXX", "IEX",
    "IFF", "INCY", "INVH", "IP", "IPG", "IQV", "IR", "IRM", "IT", "ITW",
    "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY", "JNPR", "K", "KDP", "KEY",
    "KEYS", "KHC", "KIM", "KKR", "KLAC", "KMB", "KMI", "KMX", "KR", "KVUE",
    "L", "LDOS", "LEN", "LH", "LHX", "LII", "LKQ", "LNT", "LRCX", "LULU",
    "LUV", "LVS", "LW", "LYB", "LYV", "MAA", "MAR", "MAS", "MCHP", "MCK",
    "MCO", "MGM", "MHK", "MKC", "MKTX", "MLM", "MNST", "MOH", "MOS", "MPC",
    "MPWR", "MRNA", "MSCI", "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ",
    "NDSN", "NEM", "NI", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE",
    "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORLY",
    "OTIS", "OXY", "PANW", "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PFG",
    "PGR", "PH", "PHM", "PKG", "PLD", "PNC", "PNR", "PNW", "PODD", "POOL",
    "PPG", "PPL", "PRU", "PSA", "PSX", "PTC", "PWR", "QRVO", "RCL", "REG",
    "REGN", "RF", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG",
    "RVTY", "SBAC", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNPS", "SOLV", "SPGI",
    "SRE", "STE", "STLD", "STT", "STX", "STZ", "SWK", "SWKS", "SYF", "SYK",
    "SYY", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TJX", "TPL",
    "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSN", "TT", "TTWO", "TXT",
    "TYL", "UAL", "UBER", "UDR", "UHS", "ULTA", "URI", "VICI", "VLO", "VLTO",
    "VMC", "VRSK", "VRSN", "VRTX", "VST", "VTR", "VTRS", "WAB", "WAT", "WBD",
    "WDC", "WEC", "WELL", "WM", "WMB", "WRB", "WST", "WTW", "WY", "WYNN",
    "XEL", "XYL", "YUM", "ZBH", "ZBRA", "ZTS",
)
# fmt: on

_NAMED = {
    "nasdaq100": NASDAQ_100,
    "sp100": SP_100,
    "sp500": SP_500,
}


def resolve_universe(name: str, cfg: Config | None = None) -> list[str]:
    """Turn a universe name, ``config``, or a watchlist file path into tickers.

    Always returns a sorted, deduplicated list -- ordering must be stable
    because it feeds deterministic ranking and seeding downstream.
    """
    key = name.strip().lower()
    if key in _NAMED:
        symbols = _NAMED[key]
    elif key == "config":
        if cfg is None:
            raise ValueError("universe 'config' requires a Config instance")
        symbols = cfg.data.universe
    elif Path(name).expanduser().is_file():
        lines = Path(name).expanduser().read_text().splitlines()
        symbols = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    else:
        raise ValueError(
            f"unknown universe '{name}' (have: {', '.join(sorted(_NAMED))}, "
            "'config', or a watchlist file path)"
        )
    return sorted({s.upper() for s in symbols})
