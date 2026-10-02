"""Target capping and the leakage-safe feature preprocessor.

The Preprocessor is fit on TRAINING engines only and then applied unchanged to
validation, test and serving data. It:

1. assigns each row to an operating-condition cluster (KMeans on the 3 settings,
   only when n_conditions > 1),
2. drops near-constant sensors, judged by pooled WITHIN-cluster variance in raw
   units (in multi-condition data, raw variance is dominated by the operating
   regime, so a sensor that never changes within a regime would look informative),
3. standardises the remaining sensors per cluster.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from rul.data import SENSOR_COLS, SETTING_COLS

logger = logging.getLogger(__name__)

_MIN_STD = 1e-8
_ID_COLS = ("unit", "cycle", "rul")


def cap_rul(rul: pd.Series | np.ndarray, cap: int) -> pd.Series | np.ndarray:
    """Piece-wise linear target: constant at ``cap`` early in life, linear decline after."""
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")
    return np.minimum(rul, cap)


class Preprocessor:
    def __init__(
        self,
        n_conditions: int = 1,
        variance_threshold: float = 1e-4,
        seed: int = 42,
        sensor_cols: list[str] | None = None,
    ) -> None:
        if n_conditions < 1:
            raise ValueError(f"n_conditions must be >= 1, got {n_conditions}")
        self.n_conditions = n_conditions
        self.variance_threshold = variance_threshold
        self.seed = seed
        self.sensor_cols = list(sensor_cols or SENSOR_COLS)

    # -- fitting ---------------------------------------------------------------------------

    def fit(self, train: pd.DataFrame) -> Preprocessor:
        self._check_columns(train, [*SETTING_COLS, *self.sensor_cols])
        if self.n_conditions > 1:
            settings = train[SETTING_COLS].to_numpy(dtype=float)
            self.settings_scaler_: StandardScaler | None = StandardScaler().fit(settings)
            self.kmeans_: KMeans | None = KMeans(
                n_clusters=self.n_conditions, n_init=10, random_state=self.seed
            ).fit(self.settings_scaler_.transform(settings))
            conditions = self.kmeans_.labels_
        else:
            self.settings_scaler_ = None
            self.kmeans_ = None
            conditions = np.zeros(len(train), dtype=int)

        sensors = train[self.sensor_cols].to_numpy(dtype=float)
        variance = self._pooled_within_variance(sensors, conditions)
        keep = variance >= self.variance_threshold
        self.sensor_variance_ = dict(zip(self.sensor_cols, variance.tolist(), strict=True))
        self.dropped_sensors_ = [c for c, k in zip(self.sensor_cols, keep, strict=True) if not k]
        self.feature_cols_ = [c for c, k in zip(self.sensor_cols, keep, strict=True) if k]
        if not self.feature_cols_:
            raise ValueError("All sensors are below the variance threshold; nothing to model")
        logger.info(
            "Dropped %d near-constant sensors (pooled within-condition variance < %g): %s",
            len(self.dropped_sensors_),
            self.variance_threshold,
            self.dropped_sensors_,
        )

        kept = sensors[:, keep]
        self.means_ = np.zeros((self.n_conditions, kept.shape[1]))
        self.stds_ = np.ones((self.n_conditions, kept.shape[1]))
        for c in range(self.n_conditions):
            rows = kept[conditions == c]
            if len(rows) == 0:
                continue
            self.means_[c] = rows.mean(axis=0)
            std = rows.std(axis=0)
            self.stds_[c] = np.where(std < _MIN_STD, 1.0, std)
        self.n_train_rows_ = len(train)
        return self

    def _pooled_within_variance(self, x: np.ndarray, conditions: np.ndarray) -> np.ndarray:
        sum_sq = np.zeros(x.shape[1])
        for c in np.unique(conditions):
            rows = x[conditions == c]
            sum_sq += ((rows - rows.mean(axis=0)) ** 2).sum(axis=0)
        return sum_sq / len(x)

    # -- applying --------------------------------------------------------------------------

    def assign_conditions(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        if self.kmeans_ is None:
            return np.zeros(len(df), dtype=int)
        settings = self.settings_scaler_.transform(df[SETTING_COLS].to_numpy(dtype=float))
        return self.kmeans_.predict(settings)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return id columns (unit, cycle, rul if present), ``condition`` and scaled features."""
        self._check_fitted()
        self._check_columns(df, [*SETTING_COLS, *self.feature_cols_])
        conditions = self.assign_conditions(df)
        x = df[self.feature_cols_].to_numpy(dtype=float)
        scaled = (x - self.means_[conditions]) / self.stds_[conditions]
        ids = df[[c for c in _ID_COLS if c in df.columns]]
        features = pd.DataFrame(
            scaled.astype(np.float32), columns=self.feature_cols_, index=df.index
        )
        return pd.concat(
            [ids, pd.Series(conditions, index=df.index, name="condition"), features], axis=1
        )

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)

    # -- persistence -----------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> Preprocessor:
        obj = joblib.load(path)
        if not isinstance(obj, Preprocessor):
            raise TypeError(f"{path} does not contain a Preprocessor")
        return obj

    # -- helpers ---------------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not hasattr(self, "feature_cols_"):
            raise RuntimeError("Preprocessor is not fitted; call fit() on training data first")

    @staticmethod
    def _check_columns(df: pd.DataFrame, required: list[str]) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
