"""Feature correctness -- above all, causality.

The centrepiece is ``test_no_lookahead_bias``: it perturbs the future and
asserts the past does not move. If that test ever fails, every backtest this
system produces is worthless, so it is worth more than all the others combined.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from swingbot.config import FeatureConfig
from swingbot.data.sources import SyntheticSource
from swingbot.features.fracdiff import ffd_weights, frac_diff_ffd, min_ffd_order
from swingbot.features.technical import build_dataset, compute_features, feature_columns


@pytest.fixture(scope="module")
def bars() -> pl.DataFrame:
    return SyntheticSource(seed=1).fetch("TEST", "2015-01-01", "2024-12-31")


class TestSyntheticDeterminism:
    """Regression: seeding via builtin hash() is salted per process, so the
    'reproducible' generator silently produced different data on every run."""

    def test_same_seed_same_data(self):
        a = SyntheticSource(seed=42).fetch("AAA", "2020-01-01", "2020-12-31")
        b = SyntheticSource(seed=42).fetch("AAA", "2020-01-01", "2020-12-31")
        assert a["close"].to_list() == b["close"].to_list()

    def test_seed_is_stable_across_processes(self):
        """The real bug: identical within a run, different between runs.

        Must be a subprocess -- PYTHONHASHSEED salt is fixed for the life of an
        interpreter, so an in-process check cannot detect this.
        """
        import os
        import subprocess
        import sys
        from pathlib import Path

        src = str(Path(__file__).resolve().parents[1] / "src")
        # Pass PYTHONPATH explicitly: the editable install is unreliable here
        # (iCloud re-hides the .pth file; see the note atop the Makefile).
        env = {**os.environ, "PYTHONPATH": src}

        code = (
            "from swingbot.data.sources import SyntheticSource;"
            "d=SyntheticSource(seed=42).fetch('AAA','2020-01-01','2020-03-01');"
            "print(round(float(d['close'][-1]), 8))"
        )
        runs = {
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
            ).stdout.strip()
            for _ in range(3)
        }
        assert len(runs) == 1, f"synthetic data differs across processes: {runs}"

    def test_different_symbols_differ(self):
        a = SyntheticSource(seed=42).fetch("AAA", "2020-01-01", "2020-12-31")
        b = SyntheticSource(seed=42).fetch("BBB", "2020-01-01", "2020-12-31")
        assert a["close"].to_list() != b["close"].to_list()


class TestFracDiffWeights:
    def test_d_zero_is_identity(self):
        w = ffd_weights(0.0)
        assert w == pytest.approx([1.0])

    def test_d_one_is_first_difference(self):
        """(1-L)^1 must reproduce x_t - x_{t-1} exactly."""
        w = ffd_weights(1.0, threshold=1e-8)
        assert w == pytest.approx([-1.0, 1.0])
        x = np.array([1.0, 3.0, 6.0, 10.0])
        out = frac_diff_ffd(x, 1.0, threshold=1e-8)
        assert out[1:] == pytest.approx([2.0, 3.0, 4.0])

    def test_weights_alternate_and_decay(self):
        w = ffd_weights(0.5, threshold=1e-5)[::-1]  # newest-first
        assert w[0] == 1.0
        assert np.all(np.abs(np.diff(np.abs(w))) <= 1e-9 + np.abs(w[:-1]))
        assert abs(w[-1]) < abs(w[1])

    def test_lower_d_keeps_longer_memory(self):
        assert len(ffd_weights(0.2, 1e-4)) > len(ffd_weights(0.8, 1e-4))

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            ffd_weights(-0.5)
        with pytest.raises(ValueError):
            ffd_weights(0.5, threshold=0.0)


class TestFracDiff:
    def test_insufficient_history_is_nan_not_guessed(self):
        # d=0.4 at 1e-4 needs a ~282-bar window, so the series must exceed it.
        x = np.random.default_rng(0).normal(size=500).cumsum() + 100
        out = frac_diff_ffd(x, 0.4, threshold=1e-4)
        width = len(ffd_weights(0.4, 1e-4))
        assert np.all(np.isnan(out[: width - 1]))
        assert not np.isnan(out[width - 1])

    def test_short_series_returns_all_nan(self):
        out = frac_diff_ffd(np.array([1.0, 2.0, 3.0]), 0.4, threshold=1e-6)
        assert np.all(np.isnan(out))

    def test_reduces_nonstationarity_of_a_random_walk(self):
        rng = np.random.default_rng(3)
        walk = 100 + np.cumsum(rng.normal(0, 1, 3000))
        diffed = frac_diff_ffd(walk, 0.4)
        clean = diffed[~np.isnan(diffed)]
        # A random walk's level drifts; the FFD series should be far more anchored.
        assert abs(np.mean(clean[:500]) - np.mean(clean[-500:])) < np.std(walk)

    def test_retains_more_memory_than_plain_returns(self):
        """The entire justification for FFD over differencing."""
        rng = np.random.default_rng(5)
        walk = 100 + np.cumsum(rng.normal(0, 1, 2000))
        ffd = frac_diff_ffd(walk, 0.3)
        mask = ~np.isnan(ffd)
        returns = np.diff(walk, prepend=np.nan)

        corr_ffd = abs(np.corrcoef(ffd[mask], walk[mask])[0, 1])
        corr_ret = abs(np.corrcoef(returns[mask], walk[mask])[0, 1])
        assert corr_ffd > corr_ret

    def test_rejects_2d_input(self):
        with pytest.raises(ValueError):
            frac_diff_ffd(np.zeros((10, 2)), 0.4)

    def test_min_order_search_returns_valid_d(self):
        rng = np.random.default_rng(7)
        walk = 100 + np.cumsum(rng.normal(0, 1, 2000))
        d, _ = min_ffd_order(walk)
        assert 0.0 <= d <= 1.0


class TestTechnicalFeatures:
    def test_produces_every_declared_column(self, bars):
        cfg = FeatureConfig()
        df = build_dataset(bars, cfg)
        missing = set(feature_columns(cfg)) - set(df.columns)
        assert not missing

    def test_output_is_finite_and_complete(self, bars):
        cfg = FeatureConfig()
        df = build_dataset(bars, cfg)
        assert df.height > 1000
        for col in feature_columns(cfg):
            values = df[col].to_numpy()
            assert np.all(np.isfinite(values)), f"{col} has non-finite values"

    def test_rsi_stays_in_range(self, bars):
        df = build_dataset(bars, FeatureConfig())
        rsi = df["rsi"].to_numpy()
        assert rsi.min() >= 0.0 and rsi.max() <= 1.0

    def test_rejects_multi_symbol_input(self, bars):
        two = pl.concat([bars, bars.with_columns(pl.lit("OTHER").alias("symbol"))])
        with pytest.raises(ValueError, match="one symbol"):
            compute_features(two, FeatureConfig())

    def test_no_lookahead_bias(self, bars):
        """Mutate the future; assert the past is unchanged.

        Any feature that peeks -- a centred window, a full-sample mean, a
        forward fill -- changes historical values when future bars change. This
        catches all of them at once.
        """
        cfg = FeatureConfig()
        cutoff = 1500

        original = compute_features(bars, cfg)

        # Violently corrupt everything after the cutoff.
        tampered_bars = bars.with_columns(
            [
                pl.when(pl.int_range(pl.len()) >= cutoff)
                .then(pl.col(c) * 3.0)
                .otherwise(pl.col(c))
                .alias(c)
                for c in ("open", "high", "low", "close", "adj_close")
            ]
        )
        tampered = compute_features(tampered_bars, cfg)

        for col in feature_columns(cfg):
            a = original[col].to_numpy()[:cutoff]
            b = tampered[col].to_numpy()[:cutoff]
            mask = np.isfinite(a) & np.isfinite(b)
            assert np.allclose(a[mask], b[mask], atol=1e-9), (
                f"LOOK-AHEAD BIAS: '{col}' changed before bar {cutoff} "
                f"when only future bars were modified"
            )

    def test_warmup_rows_are_dropped(self, bars):
        cfg = FeatureConfig(warmup=300)
        full = compute_features(bars, cfg)
        trimmed = build_dataset(bars, cfg)
        assert trimmed.height < full.height
        assert trimmed["ts"].min() > full["ts"].min()

    def test_multi_symbol_dataset_keeps_symbols_separate(self):
        src = SyntheticSource(seed=2)
        two = pl.concat(
            [
                src.fetch("AAA", "2015-01-01", "2024-12-31"),
                src.fetch("BBB", "2015-01-01", "2024-12-31"),
            ]
        )
        df = build_dataset(two, FeatureConfig())
        assert set(df["symbol"].unique()) == {"AAA", "BBB"}
        # Features must not bleed across the symbol boundary.
        a = df.filter(pl.col("symbol") == "AAA")["ret_5d_z"].to_numpy()
        b = df.filter(pl.col("symbol") == "BBB")["ret_5d_z"].to_numpy()
        assert not np.allclose(a[:100], b[:100])
