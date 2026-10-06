"""RUL models. Each one fits on sliding windows (``rul.windows.Windows``) with targets and
predicts one RUL value per window."""

from __future__ import annotations

from rul.config import Config
from rul.models.tabular import TABULAR_MODELS, TabularModel, build_tabular_model

MODEL_NAMES = TABULAR_MODELS


def build_model(name: str, cfg: Config) -> TabularModel:
    return build_tabular_model(name, cfg)


__all__ = ["MODEL_NAMES", "TabularModel", "build_model"]
