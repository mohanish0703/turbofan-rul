"""Regression metrics for RUL, including the asymmetric NASA scoring function.

NASA score (Saxena et al., 2008) with d = predicted - true:
    d <  0 (early prediction):  exp(-d / 13) - 1
    d >= 0 (late prediction):   exp( d / 10) - 1
summed over engines. Late predictions are penalised more because predicting more
life than remains means the engine may fail before maintenance happens.
"""

from __future__ import annotations

import numpy as np


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    if t.shape != p.shape:
        raise ValueError(f"Shape mismatch: y_true {t.shape} vs y_pred {p.shape}")
    if t.size == 0:
        raise ValueError("Empty inputs")
    return t, p


def rmse(y_true, y_pred) -> float:
    t, p = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(y_true, y_pred) -> float:
    t, p = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(p - t)))


def nasa_score(y_true, y_pred) -> float:
    t, p = _as_arrays(y_true, y_pred)
    d = p - t
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def all_metrics(y_true, y_pred) -> dict[str, float]:
    """``nasa_score`` is the sum reported for the official test sets; ``nasa_score_mean`` divides
    it by the number of predictions so sets of different sizes (e.g. CV folds) are comparable."""
    t, p = _as_arrays(y_true, y_pred)
    score = nasa_score(t, p)
    return {
        "rmse": rmse(t, p),
        "mae": mae(t, p),
        "nasa_score": score,
        "nasa_score_mean": score / t.size,
    }
