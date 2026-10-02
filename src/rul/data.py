"""Load NASA C-MAPSS turbofan files and attach ground-truth RUL.

Each train/test file is space separated with no header and 26 columns:
unit, cycle, 3 operating settings, 21 sensors. Training engines run until failure;
test engines are cut off at an unknown point and RUL_FDxxx.txt gives the true RUL
after the last observed cycle of each test engine.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATASETS = ("FD001", "FD002", "FD003", "FD004")
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTING_COLS, *SENSOR_COLS]
SYNTHETIC_MARKER = "SYNTHETIC"


@dataclass(frozen=True)
class CMAPSSData:
    dataset: str
    train: pd.DataFrame  # "rul" = cycles left until failure (uncapped)
    test: pd.DataFrame  # "rul" = true cycles left at every observed row (uncapped)
    test_rul: pd.Series  # true RUL after the last observed cycle, indexed by unit
    synthetic: bool


def _check_dataset(dataset: str) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}")


def dataset_files(raw_dir: str | Path, dataset: str) -> dict[str, Path]:
    _check_dataset(dataset)
    raw_dir = Path(raw_dir)
    return {
        "train": raw_dir / f"train_{dataset}.txt",
        "test": raw_dir / f"test_{dataset}.txt",
        "rul": raw_dir / f"RUL_{dataset}.txt",
    }


def missing_files(raw_dir: str | Path, datasets: Iterable[str] = DATASETS) -> list[Path]:
    return [p for ds in datasets for p in dataset_files(raw_dir, ds).values() if not p.exists()]


def is_synthetic(raw_dir: str | Path) -> bool:
    return (Path(raw_dir) / SYNTHETIC_MARKER).exists()


def read_cmapss_file(path: str | Path) -> pd.DataFrame:
    # sep=r"\s+" also absorbs the trailing spaces present in the original files.
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path}: expected {len(COLUMNS)} columns, found {df.shape[1]}")
    df.columns = COLUMNS
    df = df.astype({"unit": "int64", "cycle": "int64"})
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def read_rul_file(path: str | Path) -> pd.Series:
    values = pd.read_csv(path, sep=r"\s+", header=None)[0].astype("int64")
    index = pd.RangeIndex(1, len(values) + 1, name="unit")
    return pd.Series(values.to_numpy(), index=index, name="rul")


def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Training engines run to failure, so RUL at a cycle is (last cycle - cycle)."""
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    return df.assign(rul=last_cycle - df["cycle"])


def add_test_rul(df: pd.DataFrame, test_rul: pd.Series) -> pd.DataFrame:
    """RUL at each observed test row = RUL after the last cycle + cycles still observed."""
    units = set(df["unit"].unique())
    if units != set(test_rul.index):
        raise ValueError(
            f"Test units ({len(units)}) and RUL file entries ({len(test_rul)}) do not match"
        )
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    return df.assign(rul=df["unit"].map(test_rul) + last_cycle - df["cycle"])


def load_dataset(raw_dir: str | Path, dataset: str) -> CMAPSSData:
    files = dataset_files(raw_dir, dataset)
    missing = [p for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing C-MAPSS files: {', '.join(str(p) for p in missing)}. "
            "Run `python scripts/download_data.py` for download instructions, "
            "or `make synthetic` to generate SYNTHETIC stand-in data."
        )
    synthetic = is_synthetic(raw_dir)
    if synthetic:
        logger.warning("Loading SYNTHETIC data from %s - not real C-MAPSS data", raw_dir)

    test_rul = read_rul_file(files["rul"])
    return CMAPSSData(
        dataset=dataset,
        train=add_train_rul(read_cmapss_file(files["train"])),
        test=add_test_rul(read_cmapss_file(files["test"]), test_rul),
        test_rul=test_rul,
        synthetic=synthetic,
    )
