"""Purged cross-validation (Lopez de Prado, AFML Ch. 7).

Standard k-fold leaks on financial time series. Two reasons, both fatal:

1. **Overlapping labels.** A feature at time t built from a 63-day window shares
   information with samples up to 63 days away. Put one in train and its
   neighbour in test and you have trained on the test set.
2. **Serial correlation.** Adjacent samples are near-duplicates, so a random
   split gives the model something very close to the answer.

The fixes are *purging* (drop training samples whose information window overlaps
the test set) and *embargo* (additionally drop a buffer after the test fold,
since features are built from trailing windows that reach back into it).

``CombinatorialPurgedCV`` goes further: rather than one train/test path it
generates many, yielding a *distribution* of Sharpe ratios. One backtest number
tells you almost nothing; a distribution tells you whether the edge is real.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    test: np.ndarray
    path_id: int = 0

    def __post_init__(self) -> None:
        overlap = np.intersect1d(self.train, self.test)
        if len(overlap):
            raise ValueError(f"train/test overlap of {len(overlap)} samples -- purging failed")


class PurgedKFold:
    """K-fold over time with purging and an embargo.

    Parameters
    ----------
    n_splits: number of contiguous test folds.
    purge: bars dropped from train on *both* sides of the test fold.
    embargo: extra bars dropped from train *after* the test fold.
    """

    def __init__(self, n_splits: int = 5, purge: int = 21, embargo: int = 21) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if purge < 0 or embargo < 0:
            raise ValueError("purge and embargo must be non-negative")
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo

    def split(self, n_samples: int) -> Iterator[Split]:
        if n_samples < self.n_splits * (self.purge + self.embargo + 2):
            raise ValueError(
                f"{n_samples} samples too few for {self.n_splits} folds with "
                f"purge={self.purge}, embargo={self.embargo}"
            )
        indices = np.arange(n_samples)
        fold_bounds = np.array_split(indices, self.n_splits)

        for i, test_idx in enumerate(fold_bounds):
            start, end = test_idx[0], test_idx[-1]
            # Purge each side; embargo only forward (features look backward).
            lo = start - self.purge
            hi = end + self.purge + self.embargo
            train_idx = indices[(indices < lo) | (indices > hi)]
            if len(train_idx) == 0:
                continue
            yield Split(train=train_idx, test=test_idx, path_id=i)


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (CPCV).

    Partitions the series into ``n_groups`` blocks and tests on every
    combination of ``n_test_groups`` of them, purging and embargoing around each.
    With the blueprint's 10/8 configuration this yields many backtest paths, and
    therefore a *distribution* of outcomes instead of a single lucky number.

    Number of paths = C(n_groups, n_test_groups).
    """

    def __init__(
        self, n_groups: int = 10, n_test_groups: int = 2, purge: int = 21, embargo: int = 21
    ) -> None:
        if n_test_groups >= n_groups:
            raise ValueError("n_test_groups must be smaller than n_groups")
        if n_groups < 2:
            raise ValueError("n_groups must be at least 2")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.purge = purge
        self.embargo = embargo

    @property
    def n_paths(self) -> int:
        from math import comb

        return comb(self.n_groups, self.n_test_groups)

    def split(self, n_samples: int) -> Iterator[Split]:
        indices = np.arange(n_samples)
        groups = np.array_split(indices, self.n_groups)

        for path_id, combo in enumerate(combinations(range(self.n_groups), self.n_test_groups)):
            test_idx = np.concatenate([groups[g] for g in combo])
            test_idx.sort()

            # Purge/embargo around every contiguous test block.
            mask = np.ones(n_samples, dtype=bool)
            for g in combo:
                start, end = groups[g][0], groups[g][-1]
                lo = max(0, start - self.purge)
                hi = min(n_samples - 1, end + self.purge + self.embargo)
                mask[lo : hi + 1] = False

            train_idx = indices[mask]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield Split(train=train_idx, test=test_idx, path_id=path_id)


def probability_of_backtest_overfitting(in_sample: np.ndarray, out_of_sample: np.ndarray) -> float:
    """PBO: P(the config that ranked best in-sample underperforms the median OOS).

    Bailey & Lopez de Prado. Feed it per-configuration performance across CV
    paths. A PBO above ~0.5 means your selection procedure is worse than random
    -- picking the in-sample winner actively hurts you.
    """
    in_sample = np.asarray(in_sample)
    out_of_sample = np.asarray(out_of_sample)
    if in_sample.shape != out_of_sample.shape:
        raise ValueError("in_sample and out_of_sample must have the same shape")
    if in_sample.ndim != 2:
        raise ValueError("expected shape (n_paths, n_configs)")

    n_paths, n_configs = in_sample.shape
    if n_configs < 2:
        return 0.0

    failures = 0
    for path in range(n_paths):
        best = int(np.argmax(in_sample[path]))
        # Where does the in-sample champion rank out-of-sample?
        oos_rank = float(np.mean(out_of_sample[path] <= out_of_sample[path][best]))
        if oos_rank < 0.5:
            failures += 1
    return failures / n_paths
