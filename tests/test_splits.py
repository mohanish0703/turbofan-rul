import numpy as np
import pytest

from rul.data import load_dataset
from rul.features import Preprocessor
from rul.splits import group_kfold, iter_fold_frames


@pytest.fixture
def train(synthetic_dir):
    return load_dataset(synthetic_dir, "FD001").train


def test_no_engine_in_both_train_and_validation(train):
    for fold, train_rows, val_rows in iter_fold_frames(train, n_folds=5, seed=0):
        assert set(train_rows["unit"]).isdisjoint(val_rows["unit"]), f"leak in fold {fold.index}"
        assert len(train_rows) + len(val_rows) == len(train)


def test_every_engine_validated_exactly_once(train):
    folds = group_kfold(train, n_folds=5, seed=0)
    val_units = np.concatenate([f.val_units for f in folds])
    assert sorted(val_units) == sorted(train["unit"].unique())


def test_folds_are_deterministic(train):
    a = group_kfold(train, n_folds=4, seed=7)
    b = group_kfold(train, n_folds=4, seed=7)
    assert all(np.array_equal(x.val_units, y.val_units) for x, y in zip(a, b, strict=True))


def test_too_many_folds(train):
    with pytest.raises(ValueError, match="exceeds"):
        group_kfold(train, n_folds=100, seed=0)


def test_fold_preprocessor_ignores_validation_rows(train):
    fold, train_rows, val_rows = next(iter_fold_frames(train, n_folds=3, seed=0))
    pre = Preprocessor().fit(train_rows)

    corrupted = train.copy()
    is_val = corrupted["unit"].isin(fold.val_units)
    corrupted.loc[is_val, pre.feature_cols_] = 1e6
    pre_again = Preprocessor().fit(corrupted[~is_val])

    np.testing.assert_array_equal(pre.means_, pre_again.means_)
    np.testing.assert_array_equal(pre.stds_, pre_again.stds_)
