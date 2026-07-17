"""Purging, embargo, and the overfitting statistics."""

from __future__ import annotations

import numpy as np
import pytest

from swingbot.backtest.validation import (
    CombinatorialPurgedCV,
    PurgedKFold,
    Split,
    probability_of_backtest_overfitting,
)
from swingbot.metrics import (
    deflated_sharpe_ratio,
    excess_sharpe,
    expected_max_sharpe,
    max_drawdown,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


class TestPurgedKFold:
    def test_train_and_test_never_overlap(self):
        for split in PurgedKFold(n_splits=5, purge=21, embargo=21).split(2000):
            assert len(np.intersect1d(split.train, split.test)) == 0

    def test_purge_gap_is_actually_enforced(self):
        purge, embargo = 21, 21
        for split in PurgedKFold(n_splits=5, purge=purge, embargo=embargo).split(2000):
            before = split.train[split.train < split.test[0]]
            if len(before):
                assert split.test[0] - before.max() > purge
            after = split.train[split.train > split.test[-1]]
            if len(after):
                assert after.min() - split.test[-1] > purge + embargo

    def test_every_sample_is_tested_exactly_once(self):
        tested = np.concatenate([s.test for s in PurgedKFold(n_splits=5).split(2000)])
        assert len(tested) == len(np.unique(tested)) == 2000

    def test_zero_purge_reduces_to_plain_contiguous_kfold(self):
        splits = list(PurgedKFold(n_splits=4, purge=0, embargo=0).split(1000))
        for s in splits:
            assert len(s.train) + len(s.test) == 1000

    def test_rejects_too_few_samples(self):
        with pytest.raises(ValueError, match="too few"):
            list(PurgedKFold(n_splits=5, purge=21, embargo=21).split(50))

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=1)
        with pytest.raises(ValueError):
            PurgedKFold(purge=-1)


class TestCombinatorialPurgedCV:
    def test_path_count_is_n_choose_k(self):
        cv = CombinatorialPurgedCV(n_groups=10, n_test_groups=2)
        assert cv.n_paths == 45
        assert len(list(cv.split(5000))) == 45

    def test_produces_a_distribution_not_a_point_estimate(self):
        """The reason CPCV exists: many paths, hence many Sharpes."""
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        assert len(list(cv.split(3000))) == 15

    def test_no_overlap_on_any_path(self):
        for split in CombinatorialPurgedCV(n_groups=6, n_test_groups=2).split(3000):
            assert len(np.intersect1d(split.train, split.test)) == 0

    def test_rejects_bad_group_config(self):
        with pytest.raises(ValueError):
            CombinatorialPurgedCV(n_groups=5, n_test_groups=5)


def test_split_rejects_overlap_at_construction():
    with pytest.raises(ValueError, match="overlap"):
        Split(train=np.array([1, 2, 3]), test=np.array([3, 4]))


class TestOverfittingStatistics:
    def test_expected_max_sharpe_grows_with_trial_count(self):
        """Try more configs, expect a better-looking fluke."""
        assert expected_max_sharpe(1000) > expected_max_sharpe(10) > 0

    def test_dsr_punishes_multiple_testing(self):
        """Same returns, more trials searched -> less credible."""
        rng = np.random.default_rng(0)
        returns = rng.normal(0.0008, 0.01, 1000)
        honest = deflated_sharpe_ratio(returns, n_trials=1)
        searched = deflated_sharpe_ratio(returns, n_trials=10_000)
        assert honest > searched

    def test_psr_rises_with_sample_size(self):
        rng = np.random.default_rng(1)
        short = rng.normal(0.0008, 0.01, 50)
        long = rng.normal(0.0008, 0.01, 5000)
        assert probabilistic_sharpe_ratio(long) > probabilistic_sharpe_ratio(short)

    def test_psr_of_pure_noise_is_unconvincing(self):
        rng = np.random.default_rng(2)
        noise = rng.normal(0.0, 0.01, 2000)
        assert probabilistic_sharpe_ratio(noise) < 0.95

    def test_pbo_detects_a_pure_selection_illusion(self):
        """In-sample ranks uncorrelated with OOS -> selection is worthless."""
        rng = np.random.default_rng(3)
        is_perf = rng.normal(0, 1, (50, 10))
        oos_perf = rng.normal(0, 1, (50, 10))
        pbo = probability_of_backtest_overfitting(is_perf, oos_perf)
        assert 0.3 < pbo < 0.7  # ~0.5 == coin flip

    def test_pbo_is_low_when_skill_is_real(self):
        rng = np.random.default_rng(4)
        skill = np.linspace(0, 2, 10)
        is_perf = skill + rng.normal(0, 0.1, (50, 10))
        oos_perf = skill + rng.normal(0, 0.1, (50, 10))
        assert probability_of_backtest_overfitting(is_perf, oos_perf) < 0.1

    def test_pbo_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(np.zeros((5, 3)), np.zeros((5, 4)))


class TestMetricsSanity:
    def test_sharpe_of_constant_returns_is_zero_not_infinite(self):
        assert sharpe_ratio(np.full(100, 0.001)) == 0.0

    def test_max_drawdown_of_monotonic_growth_is_zero(self):
        assert max_drawdown(np.array([100.0, 110.0, 120.0])) == pytest.approx(0.0)

    def test_max_drawdown_measures_peak_to_trough(self):
        equity = np.array([100.0, 150.0, 75.0, 120.0])
        assert max_drawdown(equity) == pytest.approx(0.5)

    def test_excess_sharpe_vs_itself_is_zero(self):
        equity = np.array([100.0, 101.0, 99.0, 103.0, 102.0])
        assert excess_sharpe(equity, equity) == 0.0

    def test_excess_sharpe_signs_follow_the_benchmark_gap(self):
        """A bull-market book has positive raw Sharpe by default; excess
        Sharpe must be signed by out/under-performance instead."""
        rng = np.random.default_rng(0)
        curve = lambda drift: 100.0 * np.cumprod(1.0 + drift + rng.normal(0, 0.002, 250))  # noqa: E731
        bench, faster, slower = curve(0.001), curve(0.002), curve(0.0002)
        assert excess_sharpe(faster, bench) > 0
        assert excess_sharpe(slower, bench) < 0
        assert sharpe_ratio(np.diff(slower) / slower[:-1]) > 0  # the trap
