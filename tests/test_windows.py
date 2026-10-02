import numpy as np
import pandas as pd
import pytest

from rul.windows import last_windows, make_windows

FEATURES = ["a", "b"]


def _engines(lengths):
    frames = []
    for unit, n in enumerate(lengths, start=1):
        cycles = np.arange(1, n + 1)
        frames.append(
            pd.DataFrame(
                {
                    "unit": unit,
                    "cycle": cycles,
                    "a": unit * 1000 + cycles,
                    "b": -cycles,
                    "rul": n - cycles,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_valid_window_shapes_and_targets():
    df = _engines([10, 7])
    w = make_windows(df, FEATURES, window=5, target_col="rul")
    assert w.X.shape == (6 + 3, 5, 2)
    assert w.X.dtype == np.float32
    # First window of engine 1 covers cycles 1..5 and is labelled with RUL at cycle 5.
    np.testing.assert_array_equal(w.X[0, :, 0], 1000 + np.arange(1, 6))
    assert w.y[0] == 10 - 5
    assert w.cycles[0] == 5
    assert set(w.units) == {1, 2}


def test_short_engine_is_left_padded_with_first_row():
    df = _engines([3])
    w = make_windows(df, FEATURES, window=5, target_col="rul")
    assert w.X.shape == (1, 5, 2)
    np.testing.assert_array_equal(w.X[0, :, 0], [1001, 1001, 1001, 1002, 1003])
    assert w.y[0] == 0


def test_last_window_per_engine():
    df = _engines([40, 12, 31])
    w = last_windows(df, FEATURES, window=30, target_col="rul")
    assert w.X.shape == (3, 30, 2)
    assert w.units.tolist() == [1, 2, 3]
    assert w.cycles.tolist() == [40, 12, 31]
    np.testing.assert_array_equal(w.X[0, -1], [1040, -40])  # last row = last cycle
    assert (w.y == 0).all()


def test_all_ends_gives_one_window_per_cycle():
    df = _engines([8, 3])
    w = make_windows(df, FEATURES, window=5, ends="all")
    assert len(w) == 11
    assert w.y is None


def test_rows_out_of_order_are_sorted():
    df = _engines([6]).sample(frac=1, random_state=0)
    w = last_windows(df, FEATURES, window=3)
    np.testing.assert_array_equal(w.X[0, :, 0], [1004, 1005, 1006])


def test_invalid_window():
    with pytest.raises(ValueError, match="window must be"):
        make_windows(_engines([5]), FEATURES, window=0)
