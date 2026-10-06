"""Sliding windows over each engine's cycle history.

A window ending at cycle t contains the ``window`` most recent rows up to and
including t. Engines (or prefixes) shorter than the window are left-padded by
repeating their first row, so every window has the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

Ends = Literal["valid", "all", "last"]


@dataclass(frozen=True)
class Windows:
    X: np.ndarray  # (n_windows, window, n_features), float32
    y: np.ndarray | None  # (n_windows,) target at the window's last row
    units: np.ndarray  # (n_windows,) engine id of each window
    cycles: np.ndarray  # (n_windows,) cycle of each window's last row
    features: tuple[str, ...] = ()  # names of the last axis of X

    def __len__(self) -> int:
        return len(self.X)


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window: int,
    target_col: str | None = None,
    ends: Ends = "valid",
) -> Windows:
    """Build windows per engine.

    ends="valid": one window per cycle from the window-th cycle onwards (training).
                  Engines shorter than the window yield one padded window.
    ends="all":   one window per cycle, early ones padded (RUL trajectories).
    ends="last":  only the window ending at the last observed cycle (official test protocol).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if ends not in ("valid", "all", "last"):
        raise ValueError(f"Unknown ends={ends!r}")

    xs, ys, units, cycles = [], [], [], []
    for unit, g in df.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
        arr = g[feature_cols].to_numpy(dtype=np.float32)
        n = len(arr)
        padded = np.concatenate([np.repeat(arr[:1], window - 1, axis=0), arr])
        # (n, n_features, window): view k is the window ending at real row k.
        views = sliding_window_view(padded, window, axis=0)
        if ends == "all":
            idx = np.arange(n)
        elif ends == "last" or n < window:
            idx = np.array([n - 1])
        else:
            idx = np.arange(window - 1, n)

        xs.append(views[idx].transpose(0, 2, 1))
        units.append(np.full(len(idx), unit))
        cycles.append(g["cycle"].to_numpy()[idx])
        if target_col is not None:
            ys.append(g[target_col].to_numpy(dtype=np.float32)[idx])

    return Windows(
        X=np.ascontiguousarray(np.concatenate(xs), dtype=np.float32),
        y=np.concatenate(ys) if target_col is not None else None,
        units=np.concatenate(units),
        cycles=np.concatenate(cycles),
        features=tuple(feature_cols),
    )


def last_windows(
    df: pd.DataFrame, feature_cols: list[str], window: int, target_col: str | None = None
) -> Windows:
    """One window per engine ending at its last observed cycle."""
    return make_windows(df, feature_cols, window, target_col=target_col, ends="last")
