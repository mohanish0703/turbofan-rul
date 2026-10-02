"""Leakage-safe cross-validation: folds are split by engine, never by row.

Rows from one engine are strongly correlated, so a random row split would put
near-identical windows in train and validation and overstate accuracy. Any
fitted preprocessing (scaler, condition clusters, dropped sensors) must also be
refit on each fold's training engines; ``iter_fold_frames`` returns raw frames
so callers do exactly that.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class Fold:
    index: int
    train_units: np.ndarray
    val_units: np.ndarray


def group_kfold(df: pd.DataFrame, n_folds: int, seed: int) -> list[Fold]:
    units = df["unit"].to_numpy()
    n_units = len(np.unique(units))
    if n_folds > n_units:
        raise ValueError(f"n_folds={n_folds} exceeds the number of engines ({n_units})")

    splitter = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for i, (train_idx, val_idx) in enumerate(splitter.split(units, groups=units)):
        folds.append(
            Fold(
                index=i,
                train_units=np.unique(units[train_idx]),
                val_units=np.unique(units[val_idx]),
            )
        )
    return folds


def iter_fold_frames(
    df: pd.DataFrame, n_folds: int, seed: int
) -> Iterator[tuple[Fold, pd.DataFrame, pd.DataFrame]]:
    """Yield (fold, train_rows, val_rows) with raw, unpreprocessed rows."""
    for fold in group_kfold(df, n_folds, seed):
        yield (
            fold,
            df[df["unit"].isin(fold.train_units)],
            df[df["unit"].isin(fold.val_units)],
        )
