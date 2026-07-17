"""Cross-sectional ranker, panel, and regime-gate invariants.

The failure modes worth testing here are the quiet ones: a feature that peeks
forward, a training label that overlaps a prediction window, an IC that looks
great because the panel leaked, and a health index that reacts to information
it could not have had. Signal-recovery tests run on planted synthetic data
where the right answer is known by construction.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from swingbot.agents.ranker import (
    ic_summary,
    rank_ic,
    shuffle_targets_within_date,
    walk_forward_scores,
)
from swingbot.features.cross_section import FEATURES, build_panel
from swingbot.paper.gate import gate_signal, health_index, should_trade
from swingbot.trials import log_trial, n_trials

HORIZON = 5

# Small trees for small synthetic panels; the defaults assume real data volume.
FAST_PARAMS = {"num_iterations": 50, "min_data_in_leaf": 20, "num_leaves": 15}


def make_bars(
    n_symbols: int = 8, n_days: int = 420, seed: int = 3
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Deterministic random-walk bars for symbols and a benchmark."""
    rng = np.random.default_rng(seed)
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
    frames = []
    for k in range(n_symbols):
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n_days)))
        frames.append(
            pl.DataFrame(
                {
                    "symbol": [f"S{k:02d}"] * n_days,
                    "ts": days,
                    "close": close,
                    "volume": rng.uniform(1e6, 5e6, n_days),
                }
            )
        )
    bench_close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n_days)))
    bench = pl.DataFrame({"ts": days, "close": bench_close})
    return pl.concat(frames), bench


def make_panel(
    n_symbols: int = 20, n_days: int = 380, *, signal: float, seed: int = 11
) -> pl.DataFrame:
    """A ready-made panel with a planted (or absent) cross-sectional signal."""
    rng = np.random.default_rng(seed)
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
    rows = {f: rng.normal(0, 1, (n_days, n_symbols)) for f in FEATURES}
    target = signal * rows["mom_1m"] + rng.normal(0, 1, (n_days, n_symbols))
    return pl.DataFrame(
        {
            "ts": [d for d in days for _ in range(n_symbols)],
            "symbol": [f"S{k:02d}" for _ in days for k in range(n_symbols)],
            **{f: rows[f].ravel() for f in FEATURES},
            "target": target.ravel(),
        }
    )


class TestPanel:
    def test_features_are_trailing_only(self):
        """Corrupting every bar after a cutoff must not move any feature
        before it. (The *target* changes -- it is forward by design.)"""
        bars, bench = make_bars()
        cutoff = date(2020, 1, 1) + timedelta(days=350)
        clean = build_panel(bars, bench, horizon=HORIZON)
        corrupted_bars = bars.with_columns(
            pl.when(pl.col("ts") > cutoff)
            .then(pl.col("close") * 3.0)
            .otherwise(pl.col("close"))
            .alias("close")
        )
        dirty = build_panel(corrupted_bars, bench, horizon=HORIZON)
        a = clean.filter(pl.col("ts") <= cutoff).select("ts", "symbol", *FEATURES)
        b = dirty.filter(pl.col("ts") <= cutoff).select("ts", "symbol", *FEATURES)
        assert a.equals(b)

    def test_target_is_null_until_matured(self):
        bars, bench = make_bars()
        panel = build_panel(bars, bench, horizon=HORIZON)
        dates = panel["ts"].unique().sort().to_list()
        unmatured = panel.filter(pl.col("ts").is_in(dates[-HORIZON:]))
        matured = panel.filter(pl.col("ts").is_in(dates[:-HORIZON]))
        assert unmatured["target"].null_count() == unmatured.height
        assert matured["target"].null_count() == 0

    def test_target_is_excess_return_vs_benchmark(self):
        bars, bench = make_bars()
        panel = build_panel(bars, bench, horizon=HORIZON)
        row = panel.filter(pl.col("target").is_not_null()).row(37, named=True)
        sym_close = bars.filter(pl.col("symbol") == row["symbol"]).sort("ts")
        ts_list = sym_close["ts"].to_list()
        i = ts_list.index(row["ts"])
        c = sym_close["close"].to_list()
        b = bench.sort("ts")["close"].to_list()
        j = bench.sort("ts")["ts"].to_list().index(row["ts"])
        expected = (c[i + HORIZON] / c[i] - 1.0) - (b[j + HORIZON] / b[j] - 1.0)
        assert row["target"] == pytest.approx(expected, rel=1e-9)

    def test_cross_sectional_rank_is_bounded(self):
        bars, bench = make_bars()
        panel = build_panel(bars, bench, horizon=HORIZON)
        r = panel["cross_sectional_rank"]
        assert float(r.min()) >= 0.0 and float(r.max()) <= 1.0


class TestWalkForward:
    def test_no_training_label_overlaps_prediction(self):
        panel = make_panel(signal=0.3)
        res = walk_forward_scores(
            panel,
            horizon=HORIZON,
            refit_every=21,
            min_train_days=100,
            embargo_days=2,
            params=FAST_PARAMS,
        )
        dates = sorted(panel["ts"].unique().to_list())
        pos = {d: i for i, d in enumerate(dates)}
        assert res.folds  # the walk actually walked
        for f in res.folds:
            # Last training label matures at train_end + HORIZON; strictly
            # before the prediction window, with the embargo on top.
            assert pos[f["train_end"]] + HORIZON + 2 < pos[f["pred_start"]]

    def test_scores_are_out_of_sample_only(self):
        panel = make_panel(signal=0.3)
        res = walk_forward_scores(
            panel,
            horizon=HORIZON,
            refit_every=21,
            min_train_days=100,
            embargo_days=2,
            params=FAST_PARAMS,
        )
        first_scored = res.scores["ts"].min()
        dates = sorted(panel["ts"].unique().to_list())
        # Nothing before the first legal prediction date is ever scored.
        assert dates.index(first_scored) >= 100 + HORIZON + 2

    def test_recovers_planted_signal(self):
        panel = make_panel(signal=0.5)
        res = walk_forward_scores(
            panel,
            horizon=HORIZON,
            refit_every=21,
            min_train_days=100,
            embargo_days=2,
            params=FAST_PARAMS,
        )
        s = ic_summary(rank_ic(res.scores))
        assert s["mean"] > 0.15, f"planted signal not recovered: {s}"

    def test_no_free_signal_on_noise(self):
        """On an independent target the walk-forward IC must be ~0. A clearly
        positive IC here means the harness leaks -- the same class of bug the
        engine's no-free-money test guards against."""
        panel = make_panel(signal=0.0)
        res = walk_forward_scores(
            panel,
            horizon=HORIZON,
            refit_every=21,
            min_train_days=100,
            embargo_days=2,
            params=FAST_PARAMS,
        )
        s = ic_summary(rank_ic(res.scores))
        assert abs(s["mean"]) < 0.06, f"IC on pure noise: {s}"

    def test_shuffle_null_kills_planted_signal(self):
        """The cross-sectional shuffle preserves every per-date marginal but
        severs the feature->outcome link. Real signal must die; if it
        survives, the pipeline is leaking."""
        panel = make_panel(signal=0.5)
        shuffled = shuffle_targets_within_date(panel, seed=1)
        # Same numbers on every date, different assignment to symbols.
        a = panel.group_by("ts").agg(pl.col("target").sort()).sort("ts")
        b = shuffled.group_by("ts").agg(pl.col("target").sort()).sort("ts")
        assert a.equals(b)
        res = walk_forward_scores(
            shuffled,
            horizon=HORIZON,
            refit_every=21,
            min_train_days=100,
            embargo_days=2,
            params=FAST_PARAMS,
        )
        s = ic_summary(rank_ic(res.scores))
        assert abs(s["mean"]) < 0.06, f"planted signal survived the shuffle: {s}"


class TestTrials:
    def test_ledger_is_append_only_and_counted(self, tmp_path):
        path = tmp_path / "trials.jsonl"
        assert n_trials(path) == 0
        assert log_trial(path, {"mean_ic": 0.01}) == 1
        assert log_trial(path, {"mean_ic": 0.02}) == 2
        import json

        lines = [json.loads(x) for x in path.read_text().splitlines()]
        assert lines[0]["mean_ic"] == 0.01
        assert all("utc" in rec for rec in lines)


class TestGate:
    @staticmethod
    def ic_frame(vals: list[float]) -> pl.DataFrame:
        days = [date(2024, 1, 1) + timedelta(days=i) for i in range(len(vals))]
        return pl.DataFrame({"ts": days, "ic": vals})

    def test_health_index_ignores_unmatured_ic(self):
        """ICs inside the maturity window cannot move the index -- at date t
        only ICs from t - horizon and earlier are realized."""
        base = [0.05] * 80
        spiked = [0.05] * 70 + [5.0] * 10  # absurd ICs in the unmatured tail
        h_base = health_index(self.ic_frame(base), horizon=10)["h_real"]
        h_spiked = health_index(self.ic_frame(spiked), horizon=10)["h_real"]
        assert h_base.equals(h_spiked)

    def test_gate_is_bounded_and_tracks_health(self):
        vals = [0.10] * 60 + [-0.10] * 60  # healthy run, then a regime break
        g = gate_signal(health_index(self.ic_frame(vals), horizon=5))
        live = g.drop_nulls(subset=["g"])
        assert float(live["g"].min()) >= 0.0 and float(live["g"].max()) <= 1.0
        # Trust at the end of the bad regime is below trust before it broke.
        assert live["g"][-1] < live["g"][0]

    def test_no_history_means_trade(self):
        assert should_trade(None)
        assert should_trade(float("nan"))
        assert not should_trade(0.1)
        assert should_trade(0.9)
