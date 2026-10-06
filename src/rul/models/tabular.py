"""Tabular baselines: summarise each window into per-sensor statistics, then regress.

Every window (window_length cycles x sensors) becomes one row with, per sensor, its last
value, mean, standard deviation and least-squares slope over the window, plus the current
cycle number (the engine's age, which is known at prediction time).

Models:
- naive:         the mean training target for every window (a floor any model must beat)
- linear:        ordinary least squares
- random_forest: scikit-learn RandomForestRegressor
- xgboost:       gradient-boosted trees (XGBRegressor, histogram method)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from rul.config import Config
from rul.windows import Windows

TABULAR_MODELS = ("naive", "linear", "random_forest", "xgboost")
STATS = ("last", "mean", "std", "slope")


def summarise_windows(windows: Windows) -> tuple[np.ndarray, list[str]]:
    """Return a (n_windows, n_features * len(STATS) + 1) table and its column names."""
    x = windows.X.astype(np.float64)
    n_steps, n_features = x.shape[1], x.shape[2]
    if len(windows.features) != n_features:
        raise ValueError(
            f"windows.features names {len(windows.features)} features but X has {n_features}"
        )
    # Least-squares slope against a centred time index (sum of t is 0, so no mean term needed).
    t = np.arange(n_steps) - (n_steps - 1) / 2
    denom = float(t @ t)
    slope = np.einsum("t,ntf->nf", t, x) / denom if denom > 0 else np.zeros(x.shape[::2])

    table = np.concatenate(
        [x[:, -1], x.mean(axis=1), x.std(axis=1), slope, windows.cycles[:, None]], axis=1
    )
    names = [f"{f}_{stat}" for stat in STATS for f in windows.features] + ["cycle"]
    return table, names


class TabularModel:
    """A regressor on window summary features.

    The target is RUL capped at ``max_rul``, so predictions are clipped to [0, max_rul]:
    anything outside that range is an extrapolation artefact (mostly of the linear model).
    """

    def __init__(self, name: str, estimator: Any, max_rul: float) -> None:
        self.name = name
        self.estimator = estimator
        self.max_rul = max_rul

    def fit(self, windows: Windows) -> TabularModel:
        if windows.y is None:
            raise ValueError("Training windows need targets (windows.y)")
        table, self.feature_names_ = summarise_windows(windows)
        self.estimator.fit(table, windows.y)
        return self

    def predict(self, windows: Windows) -> np.ndarray:
        if not hasattr(self, "feature_names_"):
            raise RuntimeError(f"Model {self.name!r} is not fitted")
        table, names = summarise_windows(windows)
        if names != self.feature_names_:
            raise ValueError("Windows have different features than the model was trained on")
        return np.clip(self.estimator.predict(table), 0.0, self.max_rul)

    def feature_importances(self) -> dict[str, float]:
        """Tree models: importance per summary feature, largest first. Empty for the others."""
        importances = getattr(self.estimator, "feature_importances_", None)
        if importances is None:
            return {}
        order = np.argsort(importances)[::-1]
        return {self.feature_names_[i]: float(importances[i]) for i in order}


def build_tabular_model(name: str, cfg: Config) -> TabularModel:
    if name == "naive":
        estimator: Any = DummyRegressor(strategy="mean")
    elif name == "linear":
        estimator = LinearRegression()
    elif name == "random_forest":
        estimator = RandomForestRegressor(
            **cfg.models.random_forest.model_dump(), n_jobs=-1, random_state=cfg.seed
        )
    elif name == "xgboost":
        estimator = XGBRegressor(
            **cfg.models.xgboost.model_dump(), tree_method="hist", random_state=cfg.seed
        )
    else:
        raise ValueError(f"Unknown model {name!r}; expected one of {TABULAR_MODELS}")
    return TabularModel(name, estimator, max_rul=cfg.preprocess.rul_cap)
