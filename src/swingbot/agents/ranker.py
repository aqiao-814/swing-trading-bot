"""Gradient-boosted cross-sectional ranker.

One model scores the whole universe each day on a mean-zero target: the
``horizon``-day forward return in excess of the benchmark. Predicting excess
return means the model cannot win by saying "yes" to everything -- the failure
mode that killed the per-symbol RRL scorer, whose long-only DSR objective was
answering "should I be long a Nasdaq name?" (always yes) rather than "which
names beat QQQ?" (mean-zero by construction).

The research metric is the per-date Spearman rank information coefficient
(RankIC), not Sharpe: Sharpe is a portfolio number contaminated by sizing,
costs, and regime, while RankIC isolates whether the scores order the
cross-section at all. Calibration: a *real* cross-sectional signal lives
around mean RankIC 0.03-0.07. Anything much bigger is a bug, usually leakage.

Walk-forward discipline: at each refit the training set is **purged** -- only
rows whose entire label window matured strictly before the prediction window
may be seen, plus an embargo against serial correlation. This mirrors the
PurgedKFold logic in ``backtest/validation.py`` for the expanding-window,
refit-as-you-go setting the paper loop actually runs in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import polars as pl

from swingbot.features.cross_section import FEATURES

# Deliberately conservative: at a few hundred effective independent
# observations, a deep tree is a lookup table. Native-API parameter names
# (lgb.train), so scikit-learn is not a dependency.
DEFAULT_PARAMS: dict = {
    "objective": "regression",
    "num_leaves": 31,
    "max_depth": 5,
    "learning_rate": 0.03,
    "num_iterations": 300,
    "min_data_in_leaf": 100,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "lambda_l2": 1.0,
    "verbosity": -1,
}


@dataclass
class WalkForwardResult:
    """Out-of-sample scores plus the fold bookkeeping that proves purging."""

    scores: pl.DataFrame  # ts, symbol, score, target
    folds: list[dict] = field(default_factory=list)  # train_end/pred window per refit


def walk_forward_scores(
    panel: pl.DataFrame,
    *,
    horizon: int = 20,
    refit_every: int = 21,
    min_train_days: int = 252,
    embargo_days: int = 5,
    params: dict | None = None,
    seed: int = 7,
) -> WalkForwardResult:
    """Expanding-window walk-forward scoring of the whole panel.

    At each refit date the model trains on every row whose label window
    (``horizon`` trading days) matured strictly before the prediction window,
    minus an ``embargo_days`` buffer, then scores the next ``refit_every``
    trading days out of sample. Every score in the result is therefore a
    genuine forecast: no training label overlaps any prediction date.
    """
    p = dict(DEFAULT_PARAMS, **(params or {}), seed=seed)
    dates = panel["ts"].unique().sort().to_list()
    pos = {d: i for i, d in enumerate(dates)}
    first_pred = min_train_days + horizon + embargo_days
    if first_pred >= len(dates):
        raise ValueError(
            f"panel has {len(dates)} dates; need > {first_pred} for one walk-forward fold"
        )

    out: list[pl.DataFrame] = []
    folds: list[dict] = []
    for start in range(first_pred, len(dates), refit_every):
        pred_dates = dates[start : start + refit_every]
        # Purge + embargo: training labels must be fully realized strictly
        # before the prediction window begins.
        train_cut = dates[start - horizon - embargo_days - 1]
        train = panel.filter((pl.col("ts") <= train_cut) & pl.col("target").is_not_null())
        pred = panel.filter(pl.col("ts").is_in(pred_dates))
        if train.is_empty() or pred.is_empty():
            continue

        booster = lgb.train(
            p, lgb.Dataset(train.select(FEATURES).to_numpy(), label=train["target"].to_numpy())
        )
        score = booster.predict(pred.select(FEATURES).to_numpy())
        out.append(pred.select("ts", "symbol", "target").with_columns(score=pl.Series(score)))
        folds.append(
            {
                "train_end": train_cut,
                "pred_start": pred_dates[0],
                "pred_end": pred_dates[-1],
                "n_train": len(train),
                "n_pred": len(pred),
            }
        )

    scores = (
        pl.concat(out).select("ts", "symbol", "score", "target")
        if out
        else pl.DataFrame(
            schema={"ts": pl.Date, "symbol": pl.Utf8, "score": pl.Float64, "target": pl.Float64}
        )
    )
    # Sanity: purging really held, per fold.
    for f in folds:
        assert pos[f["train_end"]] + horizon + embargo_days < pos[f["pred_start"]]
    return WalkForwardResult(scores=scores, folds=folds)


def rank_ic(scores: pl.DataFrame) -> pl.DataFrame:
    """Per-date Spearman rho between score and matured target. THE metric."""
    return (
        scores.drop_nulls(subset=["target"])
        .with_columns(
            rs=pl.col("score").rank("average").over("ts"),
            rt=pl.col("target").rank("average").over("ts"),
        )
        .group_by("ts")
        .agg(ic=pl.corr("rs", "rt"), n=pl.len())
        .filter(pl.col("n") >= 5)  # a rank correlation over 4 names is noise
        .drop("n")
        .sort("ts")
        .drop_nulls()
    )


def ic_summary(ic: pl.DataFrame) -> dict:
    """Mean IC, stability, and t-stat -- the numbers that decide go/no-go.

    The sanity gate before building anything on top: mean >= 0.02 with
    stability (mean/std) > 0.15 on a strict holdout. Below that, nothing
    downstream -- sizing, gating, uncertainty -- can rescue the signal.
    """
    vals = ic["ic"].to_numpy()
    n = len(vals)
    if n == 0:
        return {
            "n_days": 0,
            "mean": 0.0,
            "std": 0.0,
            "stability": 0.0,
            "t_stat": 0.0,
            "frac_positive": 0.0,
        }
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if n > 1 else 0.0
    return {
        "n_days": n,
        "mean": mean,
        "std": std,
        "stability": mean / std if std > 0 else 0.0,
        "t_stat": mean / (std / np.sqrt(n)) if std > 0 else 0.0,
        "frac_positive": float((vals > 0).mean()),
    }
