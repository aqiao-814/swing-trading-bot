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
   ranker is paired with a no-trade band, a liquidity floor and vol-scaled
   sizing; turnover is the tax that decides whether any of this survives.

**On limits.** The objective is to maximise simulated P&L, so nothing here caps
exposure by default: gross is set by a volatility target, and the per-name cap
and net clamp are ``None`` unless a caller sets them. That is a choice about how
much risk to run. It is *not* a licence to relax the things that make the P&L a
real number -- the liquidity floor stays, the cost model stays, and the
execution-delay and no-look-ahead invariants stay. Removing those would not make
the bot earn more; it would make the number stop being an earning.

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
    """How a ranked score becomes a dollar weight.

    **Every hard limit here defaults to ``None``, meaning off.** The book is
    sized by *risk*, not by a fixed exposure ceiling: gross is whatever makes
    the book's own predicted volatility equal ``vol_target_annual``, so the bot
    levers up into a calm cross-section and down into a violent one instead of
    sitting at a constant 1.5x through both. Every limit remains *settable* --
    pass a number and it binds exactly as before -- but nothing is capped
    unless the caller asks for it.

    What is deliberately NOT optional is ``min_dollar_volume``. That is not a
    risk limit; it is the line past which a fill is fiction. Trading a name
    whose bar volume cannot absorb the order at the modelled 1 bp half-spread
    does not make simulated money, it makes simulated *numbers*.
    """

    # Sum of |weight| across the book, as a multiple of equity. ``None`` (the
    # default) hands sizing to the vol target below. A number pins gross there
    # regardless of what the cross-section's volatility is doing.
    target_gross: float | None = None

    # Annualised volatility the book is sized to when ``target_gross is None``.
    # This is THE knob for more or less of everything. Measured on the live
    # 670-name universe: 0.35 puts gross at ~4.4x equity and realizes ~40%
    # annualised vol -- roughly 2.5x the volatility of SPY, and about three
    # times the book the old fixed 1.5x gross cap allowed.
    vol_target_annual: float = 0.35

    # Assumed average pairwise correlation used to predict book volatility.
    # It enters as (sum w_i sigma_i)^2 -- the *signed* risk sum -- which is ~0
    # precisely because longs and shorts cancel, so it only bites once the
    # macro tilt has pushed the book directional. Which is exactly when it
    # should. The risk a neutral book *does* carry is the term below.
    correlation_rho: float = 0.25

    # Residual factor risk that a dollar-neutral book does NOT hedge away, as a
    # fraction of gross risk. This term exists because the one-factor model
    # above is provably too optimistic: it says a market-neutral book carries
    # only idiosyncratic risk, so across ~170 names it predicts near-zero
    # volatility and the vol target duly levers to ~12x gross. Measured against
    # the realized equity curve on the live 670-name universe over
    # 2026-05-24..2026-08-17, that prediction was low by a stable 3.3x
    # (3.36 / 3.33 / 3.26 at targets of 0.35 / 0.20 / 0.10) -- a systematic
    # bias, not noise, because a cross-sectional reversal book stays loaded on
    # sector, liquidity and crowding factors that dollar-neutrality does not
    # net out.
    #
    # 0.21 is what closes that gap on this universe: with it, predicted vol
    # tracks realized to within 1.07-1.16x across the same three targets, which
    # is honest and errs slightly conservative. It is CALIBRATED, not derived --
    # re-measure it if the universe, the cadence or the alpha changes. Set it to
    # 0.0 to restore the pure one-factor model and roughly triple the leverage
    # the vol target asks for.
    residual_factor_vol: float = 0.21

    # Ceiling on gross after vol targeting, as a multiple of equity. ``None``
    # = uncapped; the vol target is then the only thing setting book size.
    max_gross: float | None = None

    # Cap on any single name, as a fraction of equity. ``None`` = uncapped, so
    # a name at the extreme tail of the score distribution gets the weight its
    # score and its volatility earn it.
    max_position_weight: float | None = None

    # Clamp on net exposure after the news macro tilt. ``None`` = unclamped:
    # a macro view may take the book as directional as it likes. Note this was
    # never a binding constraint anyway -- NewsConfig2.macro_weight tops out at
    # 0.20 and the old clamp sat at 0.30 -- so removing it changes the contract,
    # not today's trades.
    max_net_exposure: float | None = None

    # Trade only when the target differs from the current weight by this
    # fraction of the position's own size. Relative, never absolute: v1's
    # absolute 0.05 band was calibrated for 20% positions and, at 200 names,
    # became eleven times the size of an entire position -- so every holding
    # read as "close enough" and the gross cap silently stopped meaning
    # anything (FINDINGS §11).
    rebalance_band: float = 0.25

    # Only the strongest names on each side get traded. Ranking 670 names and
    # holding all of them means paying spread on 670 near-zero convictions.
    # Kept at 0.25: this is a COST control, not a risk limit -- widening it
    # spends more on spread to express weaker opinions, which is a way to make
    # less money, not more.
    selection_fraction: float = 0.25

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


# Bars per trading year, for annualising a per-bar volatility.
BARS_PER_YEAR = BARS_PER_DAY * 252


def predicted_bar_vol(
    weights: dict[str, float],
    vols: dict[str, float],
    rho: float,
    residual: float = 0.0,
) -> float:
    """Predicted per-bar volatility of a book, from per-name volatilities.

        var = (1 - rho) * sum(w_i s_i)^2      # idiosyncratic
            + rho * (sum w_i s_i)^2           # one common factor, SIGNED
            + (residual * sum|w_i s_i|)^2     # what neutrality does not hedge

    The middle term is driven by the *signed* risk sum, so a dollar-neutral
    book is correctly predicted to be much quieter than its gross suggests,
    while a book the macro tilt has pushed directional is correctly predicted
    to be loud. That asymmetry is the reason to size on predicted risk at all.

    The third term is why this is not merely a one-factor model. Left out, the
    first two terms say a 170-name neutral book has almost no risk, and the vol
    target duly levers it to ~12x gross; measured against the realized equity
    curve that prediction was low by 3.3x. ``residual`` charges a floor of
    factor risk proportional to *gross* risk, which dollar-neutrality cannot
    net away. See ``SizingConfig.residual_factor_vol`` for the calibration.
    """
    risk = [w * vols[s] for s, w in weights.items() if s in vols]
    if not risk:
        return 0.0
    sum_sq = sum(x * x for x in risk)
    signed = sum(risk)
    gross = sum(abs(x) for x in risk)
    var = (1.0 - rho) * sum_sq + rho * signed * signed + (residual * gross) ** 2
    return float(np.sqrt(var)) if var > 0.0 else 0.0


def resolve_gross(
    unit_weights: dict[str, float], vols: dict[str, float], cfg: SizingConfig
) -> float:
    """The gross exposure this book should carry, as a multiple of equity.

    ``unit_weights`` must already sum to 1.0 in absolute value, so the returned
    number is exactly the factor they get scaled by. An explicit
    ``cfg.target_gross`` short-circuits everything; otherwise gross is solved
    from the vol target and then held under ``cfg.max_gross`` if one is set.
    """
    if cfg.target_gross is not None:
        gross = cfg.target_gross
    else:
        unit_vol = predicted_bar_vol(
            unit_weights, vols, cfg.correlation_rho, cfg.residual_factor_vol
        )
        annual = unit_vol * float(np.sqrt(BARS_PER_YEAR))
        if annual <= 0.0:
            # No measurable risk means no basis for levering. Fall back to
            # fully-invested rather than to the infinity the division implies.
            return 1.0
        gross = cfg.vol_target_annual / annual
    if cfg.max_gross is not None:
        gross = min(gross, cfg.max_gross)
    return max(0.0, gross)


def _scale_sides(unit: dict[str, float], gross: float, net: float) -> dict[str, float]:
    """Scale the long and short sides so the book lands on ``gross`` and ``net``.

    Solved in closed form rather than iterated: with the long side summing to
    ``L`` and the short side to ``-S``, scaling them by ``a`` and ``b`` gives
    ``aL + bS = gross`` and ``aL - bS = net``, so ``aL = (gross + net) / 2`` and
    ``bS = (gross - net) / 2``. Exact in one step, where the old fixed-point
    loop only converged because a per-name cap kept re-clipping it.

    A one-sided book cannot express an arbitrary net, so it is simply scaled to
    gross and takes whatever net that implies.
    """
    longs = {s: w for s, w in unit.items() if w > 0.0}
    shorts = {s: w for s, w in unit.items() if w < 0.0}
    lsum = sum(longs.values())
    ssum = -sum(shorts.values())
    if lsum <= 0.0 or ssum <= 0.0:
        total = lsum + ssum
        return {} if total <= 0.0 else {s: w / total * gross for s, w in unit.items()}

    net = max(-gross, min(gross, net))
    a = (gross + net) / (2.0 * lsum)
    b = (gross - net) / (2.0 * ssum)
    return {s: (w * a if w > 0.0 else w * b) for s, w in unit.items()}


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
    * gross is set by :func:`resolve_gross` -- the vol target by default, or
      ``cfg.target_gross`` if the caller pinned one;
    * the long and short sides are scaled to land gross and net exactly;
    * if (and only if) ``cfg.max_position_weight`` is set, per-name caps are
      enforced and the book re-solved until they hold.

    With no caps set the result is exact after one pass; the iteration below
    only runs when a cap is actually binding.
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
    total = sum(abs(v) for v in raw.values())
    if total <= 0.0:
        return {}
    unit = {s: v / total for s, v in raw.items()}

    gross = resolve_gross(unit, vols, cfg)
    if gross <= 0.0:
        return {}

    net_target = net_tilt
    if cfg.max_net_exposure is not None:
        net_target = max(-cfg.max_net_exposure, min(cfg.max_net_exposure, net_tilt))

    weights = _scale_sides(unit, gross, net_target)
    cap = cfg.max_position_weight
    if cap is None:
        return weights

    # A per-name cap fights the gross and net targets: clipping a name loses
    # gross, and re-scaling to recover it pushes other names over the cap. The
    # old code iterated a uniform shift; scaling the two sides is the same idea
    # with the net solved exactly at every step, so it converges faster. The cap
    # is enforced last, so it is the constraint that is never violated: gross
    # and net are targets, the per-name cap is a limit.
    def clamp(w: dict[str, float]) -> dict[str, float]:
        return {s: max(-cap, min(cap, v)) for s, v in w.items()}

    for _ in range(16):
        weights = clamp(weights)
        g = sum(abs(v) for v in weights.values())
        if g <= 0.0:
            return {}
        if abs(g - gross) < 1e-9 and abs(sum(weights.values()) - net_target) < 1e-9:
            break
        weights = _scale_sides(
            {s: v / g for s, v in weights.items()}, gross, net_target
        )

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
