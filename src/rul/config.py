"""Typed project configuration loaded from layered YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG_DIR = Path("configs")


class _Strict(BaseModel):
    # Unknown keys are errors, so a typo in a YAML file cannot silently fall back to a default.
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    raw_dir: Path = Path("data/raw/CMAPSS")


class PreprocessConfig(_Strict):
    rul_cap: int = Field(125, gt=0)
    variance_threshold: float = Field(1e-4, ge=0)
    n_conditions: int = Field(1, ge=1)
    window_length: int = Field(30, ge=1)


class CVConfig(_Strict):
    n_folds: int = Field(5, ge=2)


class Config(_Strict):
    seed: int = 42
    data: DataConfig = DataConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    cv: CVConfig = CVConfig()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        content = yaml.safe_load(f) or {}
    if not isinstance(content, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(content).__name__}")
    return content


def load_config(*paths: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Merge YAML files left to right (later wins), then apply ``overrides``."""
    merged: dict[str, Any] = {}
    for path in paths:
        merged = _deep_merge(merged, _read_yaml(Path(path)))
    if overrides:
        merged = _deep_merge(merged, overrides)
    return Config.model_validate(merged)


def config_for(
    dataset: str,
    extra: str | Path | None = None,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
) -> Config:
    """default.yaml, then <dataset>.yaml if present, then ``extra`` (e.g. smoke.yaml)."""
    config_dir = Path(config_dir)
    paths: list[Path] = [config_dir / "default.yaml"]
    dataset_file = config_dir / f"{dataset.lower()}.yaml"
    if dataset_file.exists():
        paths.append(dataset_file)
    if extra is not None:
        paths.append(Path(extra))
    return load_config(*paths)
