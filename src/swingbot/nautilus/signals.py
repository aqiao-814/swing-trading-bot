"""The v2 alpha: a cross-sectional, dollar-neutral, short-horizon ranker.

This module is deliberately free of any NautilusTrader import. It takes
trailing price and volume arrays and returns target weights, so the thing that
decides what to own can be tested without starting an engine -- and so the
"no look-ahead" property is checkable by construction: nothing here can see an
array element it was not handed.

**Where this design comes from.** ``docs/FINDINGS.md`` §10a swept prediction
horizons over a purged, embargoed walk-forward panel and found that the 20-day
signal v1 was built around is a 2020-2021 artifact, flat-to-negative every year
since -- but that a **3-day** signal is significant (t = 2.64) *and*
regime-persistent, positive in 2022, 2023, 2024, 2025 and 2026. Traded
dollar-neutral long/short it returned +1.34 bp/day net of 2 bp per side.

Three properties of that result dictate the whole module:

1. **The edge is cross-sectional, not directional.** Every term is z-scored
   *across symbols at one instant*, and the composite is de-meaned. A signal
   that can win by liking everything is not a signal -- it is market beta with
   extra steps, and it is exactly what v1's long-only ranking degenerated into.
2. **The edge is short-horizon.** Lookbacks are days, not quarters. That is
   also the only thing available: Yahoo serves ~60 days of 30-minute bars, so a
   3-month momentum term could not be computed here even if it helped.
3. **The edge is cost-limited**, dying somewhere past 3 bp per side. So the
   ranker is paired with a no-trade band and vol-scaled sizing rather than
   maximum aggression; turnover is the tax that decides whether any of this
   survives.

None of that makes v2 profitable. It makes v2 *able to express* the one thing
the research actually found. The forward record is still the only test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The intraday clock. A regular US session is 6.5 hours, so thirteen 30-minute
# bars; everything below is specified in days and converted here, because "3
# days" is the thing the research measured and "39 bars" is an implementation
# detail that changes if the cadence ever does.
BARS_PER_DAY = 13


@dataclass(frozen=True)
class AlphaConfig:
    """Lookbacks in *days*, and the weight of each term in the composite."""

    # Short-term reversal: the strongest single term in §10a's panel. Recent
    # losers outperform recent winners at this horizon.
    reversal_days: float = 3.0
    reversal_weight: float = 1.0

    # Vol-scaled intermediate momentum. Deliberately much longer than the
    # reversal term -- the two disagree by design, and the composite is the bet
    # that a name cheap against its own last few days but strong over the month
    # is the one to own.
    momentum_days: float = 20.0
    momentum_weight: float = 0.6

    # Realized volatility, entering negatively: betting-against-beta. Also the
    # denominator of the momentum term and of position sizing.
    vol_days: float = 10.0
    vol_weight: float = 0.3

    # Dollar volume. A mild alpha term (§10a carried log-ADV) and, more
    # importantly, the liquidity filter below.
    liquidity_weight: float = 0.15

    # Winsorisation before z-scoring, in cross-sectional standard deviations.
    # One bad print in a 670-name panel otherwise sets the scale for everybody.
    clip_sigma: float = 3.0

    def bars(self, days: float) -> int:
        return max(2, int(round(days * BARS_PER_DAY)))

    @property
    def warmup_bars(self) -> int:
        """Trailing bars needed before any score is meaningful."""
        return max(self.bars(self.momentum_days), self.bars(self.vol_days)) + 2


@dataclass(frozen=True)
class SizingConfig:
    """How a ranked score becomes a dollar weight."""

    # Sum of |weight| across the book, as a multiple of equity. Above 1.0 is
    # leverage, which the margin account permits and v1 could not use at all.
    target_gross: float = 1.5

    # Cap on any single name, as a fraction of equity. At 670 names and gross
    # 1.5 the average position is ~0.22%, so this only ever binds on the
    # extreme tail of the score distribution.
    max_position_weight: float = 0.04

    # Trade only when the target differs from the current weight by this
    # fraction of the position's own size. Relative, never absolute: v1's
    # absolute 0.05 band was calibrated for 20% positions and, at 200 names,
    # became eleven times the size of an entire position -- so every holding
    # read as "close enough" and the gross cap silently stopped meaning
    # anything (FINDINGS §11).
    rebalance_band: float = 0.25

    # Only the strongest names on each side get traded. Ranking 670 names and
    # holding all of them means paying spread on 670 near-zero convictions.
    selection_fraction: float = 0.25

    # Net exposure is clamped here after the news macro tilt. Dollar-neutral is
    # the target; this is the leash on how far a macro view may pull it.
    max_net_exposure: float = 0.30

    # Below this, a name is not traded at all -- its bar volume is too thin for
    # the 1 bp half-spread the cost model charges to be anything but fiction.
    min_dollar_volume: float = 250_000.0


@dataclass(frozen=True)
class NewsConfig2:
    """How the weekend news signal enters the ranking.

    v1 applied news *multiplicatively* on conviction, so it could resize and
    reorder the book but never open a position the price model was neutral on.
    That was the right conservatism for a long-only bot whose only way to act on
    bad news was to not buy. v2 can short, so the restriction now costs real
    information: "the press has turned on this name" is a reason to be short it,
    not merely a reason to own less of it.

    So the tilt is **additive on the standardised composite**, in units of
    cross-sectional standard deviations, and news can originate a position.
    """

    # Weight of the de-meaned per-company score, in composite sigmas. At 0.35 a
    # maximally bad news score moves a name about a third of a sigma -- enough
    # to matter in the ranking, not enough to dominate the price signal.
    company_weight: float = 0.35

    # Market-wide tone tilts *net exposure* rather than any single name, which
    # is where a market-wide view belongs. Multiplied by macro in [-1, 1] and
    # then clamped by SizingConfig.max_net_exposure.
    macro_weight: float = 0.20

    # The signal is collected at a weekend and read all week; it decays on its
    # own age so a Sunday score is worth about a quarter of itself by Tuesday.
    half_life_days: float = 2.0
    max_age_days: float = 10.0


# Shared defaults. Every config above is a frozen dataclass, so one instance
# can safely back every call site; naming them also makes the defaults
# greppable rather than hidden in signatures.
DEFAULT_ALPHA = AlphaConfig()
DEFAULT_SIZING = SizingConfig()
DEFAULT_NEWS = NewsConfig2()


@dataclass
class AlphaInputs:
    """Trailing history for the cross-section at one instant.

    ``closes`` and ``volumes`` are ``{symbol: array}`` ordered oldest-to-newest
    and ending at the bar just completed. Symbols with too little history are
    dropped by :func:`score_cross_section` rather than back-filled.
    """

    closes: dict[str, np.ndarray]
    volumes: dict[str, np.ndarray] = field(default_factory=dict)


def _zscore(values: np.ndarray, clip: float) -> np.ndarray:
    """Cross-sectional z-score, winsorised, NaN-safe.

    A constant cross-section returns all zeros rather than dividing by zero --
    that is the honest reading of "nothing distinguishes these names today".
    """
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=float)
    if finite.sum() < 2:
        return out
    v = values[finite]
    mu, sd = float(np.mean(v)), float(np.std(v))
    if sd <= 0.0:
        return out
    z = (values - mu) / sd
    z[~finite] = 0.0
    return np.clip(z, -clip, clip)


def _trailing_return(arr: np.ndarray, bars: int) -> float:
    if arr.size <= bars or arr[-bars - 1] <= 0.0:
        return np.nan
    return float(arr[-1] / arr[-bars - 1] - 1.0)


def _realized_vol(arr: np.ndarray, bars: int) -> float:
    """Standard deviation of log returns over the window, per bar."""
    if arr.size <= bars + 1:
        return np.nan
    tail = arr[-bars - 1 :]
    if np.any(tail <= 0.0):
        return np.nan
    r = np.diff(np.log(tail))
    return float(np.std(r)) if r.size else np.nan


def score_cross_section(
    inputs: AlphaInputs, cfg: AlphaConfig = DEFAULT_ALPHA
) -> tuple[dict[str, float], dict[str, float]]:
    """Rank the cross-section.

    Returns ``(scores, vols)``: the de-meaned composite score per symbol, in
    cross-sectional standard deviations, and each symbol's per-bar realized
    volatility (which sizing needs and which would otherwise be recomputed).

    Symbols without enough trailing history are simply absent from both dicts.
    """
    rev_bars = cfg.bars(cfg.reversal_days)
    mom_bars = cfg.bars(cfg.momentum_days)
    vol_bars = cfg.bars(cfg.vol_days)

    symbols: list[str] = []
    rev: list[float] = []
    mom: list[float] = []
    vol: list[float] = []
    liq: list[float] = []

    for sym, closes in inputs.closes.items():
        if closes.size < cfg.warmup_bars:
            continue
        v = _realized_vol(closes, vol_bars)
        if not np.isfinite(v) or v <= 0.0:
            continue
        m = _trailing_return(closes, mom_bars)
        r = _trailing_return(closes, rev_bars)
        if not (np.isfinite(m) and np.isfinite(r)):
            continue

        volumes = inputs.volumes.get(sym)
        dollar_vol = np.nan
        if volumes is not None and volumes.size >= vol_bars:
            recent = volumes[-vol_bars:] * closes[-vol_bars:]
            recent = recent[np.isfinite(recent)]
            if recent.size:
                dollar_vol = float(np.median(recent))

        symbols.append(sym)
        rev.append(r)
        # Vol-scaling is what makes momentum comparable across a universe whose
        # names differ in volatility by an order of magnitude; the raw return
        # ranking is otherwise just a ranking of betas.
        mom.append(m / v)
        vol.append(v)
        liq.append(np.log(dollar_vol) if np.isfinite(dollar_vol) and dollar_vol > 0 else np.nan)

    if len(symbols) < 2:
        return {}, {}

    z_rev = _zscore(np.asarray(rev), cfg.clip_sigma)
    z_mom = _zscore(np.asarray(mom), cfg.clip_sigma)
    z_vol = _zscore(np.asarray(vol), cfg.clip_sigma)
    z_liq = _zscore(np.asarray(liq), cfg.clip_sigma)

    composite = (
        # Reversal enters NEGATIVE: the finding is that recent losers outperform
        # at this horizon, so a high trailing return is a reason to be short.
        -cfg.reversal_weight * z_rev
        + cfg.momentum_weight * z_mom
        # Volatility enters negative too (betting-against-beta).
        - cfg.vol_weight * z_vol
        + cfg.liquidity_weight * z_liq
    )

    # De-mean, then rescale to unit cross-sectional sigma. De-meaning is what
    # makes the book dollar-neutral by construction: the weights sum to ~0
    # before any macro tilt, so the strategy bets on relative ordering and not
    # on the market going up. Rescaling keeps the news tilt below in stable
    # units regardless of how much the terms happened to agree today.
    composite = composite - float(np.mean(composite))
    sd = float(np.std(composite))
    if sd > 0.0:
        composite = composite / sd

    return dict(zip(symbols, composite.tolist(), strict=True)), dict(zip(symbols, vol, strict=True))


def apply_news(
    scores: dict[str, float],
    *,
    company: dict[str, float],
    macro: float,
    age_days: float,
    cfg: NewsConfig2 = DEFAULT_NEWS,
) -> tuple[dict[str, float], float]:
    """Fold the weekend news signal into the ranking.

    Returns the tilted scores and the macro net-exposure tilt (a signed
    fraction of equity, before clamping).

    The signal fades on its own age, so a stale file degrades smoothly to
    no-news rather than to a wrong trade; past ``max_age_days`` it is ignored
    entirely. A missing signal is the same as a zero one.
    """
    if not scores or age_days >= cfg.max_age_days:
        return dict(scores), 0.0

    decay = 0.5 ** (max(age_days, 0.0) / cfg.half_life_days) if cfg.half_life_days > 0 else 1.0

    tilted = {
        sym: s + cfg.company_weight * decay * float(company.get(sym, 0.0))
        for sym, s in scores.items()
    }
    # The company scores arrive already de-meaned across the news cross-section,
    # but the universe scored here is not the same set, so re-centre: otherwise
    # a week whose covered names skew positive quietly buys the whole book.
    mean = sum(tilted.values()) / len(tilted)
    tilted = {s: v - mean for s, v in tilted.items()}

    return tilted, cfg.macro_weight * decay * float(macro)


def target_weights(
    scores: dict[str, float],
    vols: dict[str, float],
    *,
    net_tilt: float = 0.0,
    dollar_volume: dict[str, float] | None = None,
    cfg: SizingConfig = DEFAULT_SIZING,
) -> dict[str, float]:
    """Turn scores into signed target weights, as fractions of equity.

    Sizing is **inverse-volatility**: a name's weight is its score divided by
    its volatility, so two names the model likes equally get equal *risk*
    rather than equal dollars. Then:

    * only the strongest ``selection_fraction`` of each side is traded at all;
    * weights are scaled so gross lands on ``target_gross``;
    * any single name is capped, and the book rescaled if a cap binds;
    * the net is nudged by ``net_tilt`` (the macro news view) and clamped.
    """
    if not scores:
        return {}

    liquid = {
        s: v
        for s, v in scores.items()
        if s in vols
        and vols[s] > 0.0
        and (
            dollar_volume is None
            or dollar_volume.get(s, 0.0) >= cfg.min_dollar_volume
        )
    }
    if not liquid:
        return {}

    longs = sorted((s for s, v in liquid.items() if v > 0), key=lambda s: -liquid[s])
    shorts = sorted((s for s, v in liquid.items() if v < 0), key=lambda s: liquid[s])
    keep_l = max(1, int(len(longs) * cfg.selection_fraction))
    keep_s = max(1, int(len(shorts) * cfg.selection_fraction))
    selected = set(longs[:keep_l]) | set(shorts[:keep_s])
    if not selected:
        return {}

    raw = {s: liquid[s] / vols[s] for s in selected}
    gross = sum(abs(v) for v in raw.values())
    if gross <= 0.0:
        return {}
    weights = {s: v / gross * cfg.target_gross for s, v in raw.items()}

    # Three constraints that fight each other: land gross on target, land net on
    # the macro tilt, and keep every single name under the cap. Enforcing them
    # in one pass does not work -- the uniform shift that fixes the net pushes
    # capped names back over the cap, and rescaling to fix gross moves the net
    # again. So iterate to a fixed point, then enforce the cap once more at the
    # end so it is the constraint that is never violated. Gross and net are
    # targets; the per-name cap is a limit.
    net_target = max(-cfg.max_net_exposure, min(cfg.max_net_exposure, net_tilt))
    cap = cfg.max_position_weight
    n = len(weights)

    def clamp(w: dict[str, float]) -> dict[str, float]:
        return {s: max(-cap, min(cap, v)) for s, v in w.items()}

    for _ in range(16):
        shift = (net_target - sum(weights.values())) / n
        weights = clamp({s: v + shift for s, v in weights.items()})
        g = sum(abs(v) for v in weights.values())
        if g <= 0.0:
            return {}
        if abs(g - cfg.target_gross) < 1e-9 and abs(sum(weights.values()) - net_target) < 1e-9:
            break
        weights = {s: v * cfg.target_gross / g for s, v in weights.items()}

    return clamp(weights)


def rebalance_needed(target: float, current: float, cfg: SizingConfig = DEFAULT_SIZING) -> bool:
    """Whether the gap between target and current weight is worth paying for.

    Measured against the larger of the two, so the band means the same thing
    for a 4% position and a 0.2% one. An absolute band does not survive a book
    that changes breadth -- see FINDINGS §11 for what it cost v1.
    """
    scale = max(abs(target), abs(current))
    if scale <= 0.0:
        return False
    return abs(target - current) / scale >= cfg.rebalance_band
