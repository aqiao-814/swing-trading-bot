"""Sizing: what a ranked cross-section becomes once limits are optional.

The v2 book has no exposure limits by default -- gross floats to whatever hits
the volatility target, and there is no per-name cap and no net clamp unless a
caller sets one. That is a deliberate choice about *risk*, and it is only
defensible if the sizing arithmetic underneath it is exact. These tests pin
that arithmetic down, because until now the whole module was covered only
indirectly, through the engine-level invariants.

Two things these tests deliberately do NOT relax, because they are what make a
simulated P&L mean anything: the liquidity floor (a fill in a name too thin to
absorb the order is fiction, not profit) and the requirement that a de-meaned
cross-section stays dollar-neutral when no macro view is expressed.
"""

from __future__ import annotations

import numpy as np
import pytest

from tradingbot.nautilus.signals import (
    BARS_PER_YEAR,
    SizingConfig,
    predicted_bar_vol,
    resolve_gross,
    target_weights,
)


def make_scores(n: int = 40) -> dict[str, float]:
    """A symmetric cross-section: half positive, half negative, mean zero."""
    syms = [f"S{i}" for i in range(n)]
    return {s: (i - (n - 1) / 2.0) / n for i, s in enumerate(syms)}


def flat_vols(scores: dict[str, float], vol: float) -> dict[str, float]:
    return dict.fromkeys(scores, vol)


def gross_of(w: dict[str, float]) -> float:
    return sum(abs(v) for v in w.values())


def annualised(w: dict[str, float], vols: dict[str, float], cfg: SizingConfig) -> float:
    """The book's predicted annualised vol, under the same model sizing used."""
    v = predicted_bar_vol(w, vols, cfg.correlation_rho, cfg.residual_factor_vol)
    return v * float(np.sqrt(BARS_PER_YEAR))


class TestDefaultsAreUnlimited:
    def test_no_cap_is_set_by_default(self):
        cfg = SizingConfig()
        assert cfg.target_gross is None
        assert cfg.max_gross is None
        assert cfg.max_position_weight is None
        assert cfg.max_net_exposure is None

    def test_no_name_is_clipped_when_uncapped(self):
        scores = make_scores()
        # A single dominant name would have been shaved by the old 4% cap.
        scores["S0"] = -8.0
        vols = flat_vols(scores, 0.01)
        w = target_weights(scores, vols, cfg=SizingConfig())
        assert abs(w["S0"]) > 0.04, "the strongest name must not be capped"

    def test_gross_is_not_pinned_to_any_constant(self):
        scores = make_scores()
        calm = target_weights(scores, flat_vols(scores, 0.004), cfg=SizingConfig())
        wild = target_weights(scores, flat_vols(scores, 0.030), cfg=SizingConfig())
        assert gross_of(calm) > gross_of(wild)


class TestVolTargeting:
    @pytest.mark.parametrize("vol", [0.002, 0.005, 0.01, 0.02, 0.05])
    def test_book_lands_on_the_volatility_target(self, vol):
        cfg = SizingConfig(vol_target_annual=0.35)
        scores = make_scores()
        vols = flat_vols(scores, vol)
        w = target_weights(scores, vols, cfg=cfg)
        assert annualised(w, vols, cfg) == pytest.approx(0.35, rel=1e-9)

    def test_target_scales_the_book_proportionally(self):
        scores = make_scores()
        vols = flat_vols(scores, 0.01)
        lo = target_weights(scores, vols, cfg=SizingConfig(vol_target_annual=0.20))
        hi = target_weights(scores, vols, cfg=SizingConfig(vol_target_annual=0.40))
        assert gross_of(hi) == pytest.approx(2.0 * gross_of(lo))

    def test_gross_is_inverse_in_volatility(self):
        scores = make_scores()
        half = target_weights(scores, flat_vols(scores, 0.005), cfg=SizingConfig())
        full = target_weights(scores, flat_vols(scores, 0.010), cfg=SizingConfig())
        assert gross_of(half) == pytest.approx(2.0 * gross_of(full))

    def test_a_riskless_cross_section_does_not_lever_to_infinity(self):
        # sigma -> 0 makes the vol target divide by zero. The book must fall
        # back to fully-invested rather than to an unbounded position.
        assert resolve_gross({"A": 0.5, "B": -0.5}, {"A": 0.0, "B": 0.0}, SizingConfig()) == 1.0


class TestPredictedVol:
    def test_a_neutral_book_is_quieter_than_a_directional_one(self):
        syms = [f"S{i}" for i in range(40)]
        vols = dict.fromkeys(syms, 0.01)
        neutral = {s: (0.05 if i < 20 else -0.05) for i, s in enumerate(syms)}
        directional = dict.fromkeys(syms, 0.05)
        assert gross_of(neutral) == pytest.approx(gross_of(directional))
        assert predicted_bar_vol(neutral, vols, 0.25) < predicted_bar_vol(directional, vols, 0.25)

    def test_correlation_hedges_a_neutral_book_and_hurts_a_directional_one(self):
        # The sign of rho's effect flips with the book's net risk, which is the
        # property that makes it worth modelling at all. In a dollar-neutral
        # book the signed risk sum is zero, so only the (1 - rho) idiosyncratic
        # term survives and MORE correlation means LESS predicted vol -- the
        # longs and shorts hedge each other. Take the book directional and the
        # rho term dominates instead.
        syms = [f"S{i}" for i in range(20)]
        vols = dict.fromkeys(syms, 0.01)
        neutral = {s: (0.1 if i < 10 else -0.1) for i, s in enumerate(syms)}
        directional = dict.fromkeys(syms, 0.1)

        assert predicted_bar_vol(neutral, vols, 0.9) < predicted_bar_vol(neutral, vols, 0.0)
        assert predicted_bar_vol(directional, vols, 0.9) > predicted_bar_vol(directional, vols, 0.0)

    def test_a_neutral_book_carries_only_idiosyncratic_risk(self):
        syms = [f"S{i}" for i in range(20)]
        vols = dict.fromkeys(syms, 0.01)
        neutral = {s: (0.1 if i < 10 else -0.1) for i, s in enumerate(syms)}
        # var = (1 - rho) * sum (w_i s_i)^2, exactly -- the rho term is zeroed
        # by the signed risk sum, so the prediction is closed-form here.
        expected = np.sqrt(0.5 * sum((w * vols[s]) ** 2 for s, w in neutral.items()))
        assert predicted_bar_vol(neutral, vols, 0.5) == pytest.approx(expected)

    def test_residual_factor_risk_survives_dollar_neutrality(self):
        # The whole point of the residual term: a perfectly neutral book must
        # NOT be predicted to be almost riskless just because it nets to zero.
        syms = [f"S{i}" for i in range(200)]
        vols = dict.fromkeys(syms, 0.01)
        neutral = {s: (0.01 if i < 100 else -0.01) for i, s in enumerate(syms)}
        one_factor = predicted_bar_vol(neutral, vols, 0.25, 0.0)
        with_residual = predicted_bar_vol(neutral, vols, 0.25, 0.21)
        assert with_residual > 3 * one_factor

    def test_residual_term_scales_with_gross_not_net(self):
        syms = [f"S{i}" for i in range(50)]
        vols = dict.fromkeys(syms, 0.01)
        neutral = {s: (0.02 if i < 25 else -0.02) for i, s in enumerate(syms)}
        doubled = {s: 2 * w for s, w in neutral.items()}
        assert sum(neutral.values()) == pytest.approx(0.0)
        assert predicted_bar_vol(doubled, vols, 0.25, 0.21) == pytest.approx(
            2 * predicted_bar_vol(neutral, vols, 0.25, 0.21)
        )

    def test_a_bigger_residual_means_less_leverage(self):
        scores = make_scores()
        vols = flat_vols(scores, 0.01)
        timid = target_weights(scores, vols, cfg=SizingConfig(residual_factor_vol=0.40))
        bold = target_weights(scores, vols, cfg=SizingConfig(residual_factor_vol=0.05))
        assert gross_of(bold) > gross_of(timid)

    def test_empty_book_has_no_volatility(self):
        assert predicted_bar_vol({}, {}, 0.25) == 0.0


class TestLimitsStillWorkWhenSet:
    def test_explicit_target_gross_pins_gross_exactly(self):
        scores = make_scores()
        w = target_weights(scores, flat_vols(scores, 0.01), cfg=SizingConfig(target_gross=1.5))
        assert gross_of(w) == pytest.approx(1.5)

    def test_explicit_gross_overrides_the_vol_target(self):
        scores = make_scores()
        cfg = SizingConfig(target_gross=1.5, vol_target_annual=99.0)
        assert gross_of(target_weights(scores, flat_vols(scores, 0.01), cfg=cfg)) == pytest.approx(
            1.5
        )

    def test_max_gross_caps_a_calm_book(self):
        scores = make_scores()
        vols = flat_vols(scores, 0.004)
        uncapped = gross_of(target_weights(scores, vols, cfg=SizingConfig()))
        capped = gross_of(target_weights(scores, vols, cfg=SizingConfig(max_gross=3.0)))
        assert uncapped > 3.0
        assert capped == pytest.approx(3.0)

    def test_max_gross_does_not_raise_a_quiet_book(self):
        scores = make_scores()
        vols = flat_vols(scores, 0.05)
        cfg = SizingConfig(max_gross=10.0)
        w = target_weights(scores, vols, cfg=cfg)
        assert gross_of(w) < 10.0
        assert annualised(w, vols, cfg) == pytest.approx(0.35)

    def test_max_position_weight_binds_every_name(self):
        scores = make_scores()
        w = target_weights(
            scores, flat_vols(scores, 0.01), cfg=SizingConfig(max_position_weight=0.02)
        )
        assert max(abs(v) for v in w.values()) <= 0.02 + 1e-12

    def test_a_binding_cap_still_uses_the_whole_book(self):
        # 40 names, symmetric, selection_fraction 0.25 -> 5 long + 5 short = 10
        # names, so a 0.02 cap makes 0.20 the largest reachable gross. The
        # solver must actually get there rather than stalling short of it.
        scores = make_scores(40)
        w = target_weights(
            scores, flat_vols(scores, 0.004), cfg=SizingConfig(max_position_weight=0.02)
        )
        assert len(w) == 10
        assert gross_of(w) == pytest.approx(0.20)

    def test_max_net_exposure_clamps_the_macro_tilt(self):
        scores = make_scores()
        cfg = SizingConfig(target_gross=2.0, max_net_exposure=0.10)
        w = target_weights(scores, flat_vols(scores, 0.01), net_tilt=0.5, cfg=cfg)
        assert sum(w.values()) == pytest.approx(0.10)

    def test_an_unclamped_tilt_is_honoured_in_full(self):
        scores = make_scores()
        cfg = SizingConfig(target_gross=2.0)
        w = target_weights(scores, flat_vols(scores, 0.01), net_tilt=0.5, cfg=cfg)
        assert sum(w.values()) == pytest.approx(0.5)
        assert gross_of(w) == pytest.approx(2.0)


class TestGrossAndNetAreExact:
    @pytest.mark.parametrize("tilt", [-1.2, -0.4, 0.0, 0.3, 1.1])
    def test_both_targets_are_hit_in_one_pass(self, tilt):
        scores = make_scores()
        cfg = SizingConfig(target_gross=2.0)
        w = target_weights(scores, flat_vols(scores, 0.01), net_tilt=tilt, cfg=cfg)
        assert gross_of(w) == pytest.approx(2.0)
        assert sum(w.values()) == pytest.approx(max(-2.0, min(2.0, tilt)))

    def test_net_cannot_exceed_gross(self):
        scores = make_scores()
        cfg = SizingConfig(target_gross=1.0)
        w = target_weights(scores, flat_vols(scores, 0.01), net_tilt=5.0, cfg=cfg)
        assert sum(w.values()) == pytest.approx(1.0)
        assert gross_of(w) == pytest.approx(1.0)

    def test_a_book_with_no_macro_view_is_dollar_neutral(self):
        scores = make_scores()
        w = target_weights(scores, flat_vols(scores, 0.01), cfg=SizingConfig())
        assert sum(w.values()) == pytest.approx(0.0, abs=1e-12)

    def test_a_one_sided_cross_section_is_scaled_to_gross(self):
        scores = {f"S{i}": float(i + 1) for i in range(8)}  # all positive
        w = target_weights(scores, flat_vols(scores, 0.01), cfg=SizingConfig(target_gross=1.5))
        assert gross_of(w) == pytest.approx(1.5)
        assert all(v > 0 for v in w.values())


class TestHonestyGuardsSurvive:
    """The limits that are NOT lifted, because lifting them fakes the P&L."""

    def test_illiquid_names_are_still_excluded(self):
        scores = make_scores(20)
        vols = flat_vols(scores, 0.01)
        adv = {s: (10.0 if i % 2 else 1e9) for i, s in enumerate(scores)}
        w = target_weights(scores, vols, dollar_volume=adv, cfg=SizingConfig())
        assert all(adv[s] >= SizingConfig().min_dollar_volume for s in w)

    def test_a_wholly_illiquid_cross_section_trades_nothing(self):
        scores = make_scores(20)
        adv = dict.fromkeys(scores, 1.0)
        assert target_weights(scores, flat_vols(scores, 0.01), dollar_volume=adv) == {}

    def test_names_without_a_volatility_are_dropped(self):
        scores = make_scores(10)
        vols = flat_vols(scores, 0.01)
        del vols["S0"]
        vols["S1"] = 0.0
        w = target_weights(scores, vols, cfg=SizingConfig())
        assert "S0" not in w and "S1" not in w

    def test_an_empty_cross_section_is_not_an_error(self):
        assert target_weights({}, {}) == {}
