import pytest

from rul.data import COLUMNS, is_synthetic, load_dataset, missing_files, read_cmapss_file


def test_columns_named_despite_trailing_whitespace(synthetic_dir):
    df = read_cmapss_file(synthetic_dir / "train_FD001.txt")
    assert list(df.columns) == COLUMNS
    assert df["unit"].dtype == "int64"
    assert df.notna().all().all()


def test_train_rul_ends_at_zero(synthetic_dir):
    data = load_dataset(synthetic_dir, "FD001")
    last = data.train.sort_values("cycle").groupby("unit").tail(1)
    assert (last["rul"] == 0).all()
    first = data.train[data.train["cycle"] == 1].set_index("unit")["rul"]
    max_cycle = data.train.groupby("unit")["cycle"].max()
    assert (first == max_cycle - 1).all()


def test_test_rul_matches_rul_file_at_last_cycle(synthetic_dir):
    data = load_dataset(synthetic_dir, "FD002")
    last = data.test.sort_values("cycle").groupby("unit").tail(1).set_index("unit")["rul"]
    assert (last.sort_index() == data.test_rul.sort_index()).all()
    # RUL decreases by exactly one per observed cycle.
    diffs = data.test.sort_values(["unit", "cycle"]).groupby("unit")["rul"].diff().dropna()
    assert (diffs == -1).all()


def test_synthetic_flag(synthetic_dir):
    assert is_synthetic(synthetic_dir)
    assert load_dataset(synthetic_dir, "FD003").synthetic


def test_missing_files_give_clear_error(tmp_path):
    assert len(missing_files(tmp_path)) == 12
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_dataset(tmp_path, "FD001")


def test_unknown_dataset_rejected(synthetic_dir):
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset(synthetic_dir, "FD005")
