"""Train and evaluate RUL models: engine-wise cross-validation, then the official test protocol.

python -m rul.train --dataset FD001 --model xgboost
python -m rul.train --dataset all --model all
python -m rul.train --smoke --dataset all     # SYNTHETIC data, tiny models -> results/smoke/

For each (dataset, model):
1. GroupKFold by engine on the training set. The preprocessor and the model are refit on each
   fold's training engines; metrics use every window of the held-out engines against the
   capped target the model is trained to predict.
2. Refit on all training engines and predict from the last window of each test engine.
   Headline metrics compare with the raw (uncapped) RUL in RUL_FDxxx.txt; metrics against the
   capped truth are kept as a labelled extra.
3. Write one JSON record to <out>/runs/<DATASET>_<model>.json (summarise with rul.report).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rul import __version__
from rul.config import Config, config_for
from rul.data import DATASETS, CMAPSSData, load_dataset
from rul.features import Preprocessor, cap_rul
from rul.metrics import all_metrics
from rul.models import MODEL_NAMES, TabularModel, build_model
from rul.splits import iter_fold_frames
from rul.windows import last_windows, make_windows

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
SMOKE_DIR = RESULTS_DIR / "smoke"
SMOKE_CONFIG = Path("configs/smoke.yaml")
CV_SUMMARY_METRICS = ("rmse", "mae", "nasa_score_mean")
TOP_FEATURES = 15


def fit_pipeline(
    train: pd.DataFrame, cfg: Config, model_name: str
) -> tuple[Preprocessor, TabularModel]:
    """Fit the preprocessor and the model on the given (training) engines only."""
    pp = cfg.preprocess
    pre = Preprocessor(
        n_conditions=pp.n_conditions, variance_threshold=pp.variance_threshold, seed=cfg.seed
    ).fit(train)
    frame = pre.transform(train)
    frame["rul"] = cap_rul(frame["rul"], pp.rul_cap)
    windows = make_windows(frame, pre.feature_cols_, pp.window_length, target_col="rul")
    return pre, build_model(model_name, cfg).fit(windows)


def cross_validate(train: pd.DataFrame, cfg: Config, model_name: str) -> list[dict[str, Any]]:
    pp = cfg.preprocess
    folds = []
    for fold, train_rows, val_rows in iter_fold_frames(train, cfg.cv.n_folds, cfg.seed):
        pre, model = fit_pipeline(train_rows, cfg, model_name)
        val = make_windows(
            pre.transform(val_rows), pre.feature_cols_, pp.window_length, target_col="rul"
        )
        metrics = all_metrics(cap_rul(val.y, pp.rul_cap), model.predict(val))
        folds.append(
            {
                "fold": fold.index,
                "val_engines": len(fold.val_units),
                "val_windows": len(val),
                **metrics,
            }
        )
        logger.info(
            "%s fold %d/%d: RMSE %.2f", model_name, fold.index + 1, cfg.cv.n_folds, metrics["rmse"]
        )
    return folds


def summarise_folds(folds: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([f[key] for f in folds])),
            "std": float(np.std([f[key] for f in folds])),
        }
        for key in CV_SUMMARY_METRICS
    }


def evaluate_test(
    data: CMAPSSData, cfg: Config, model_name: str
) -> tuple[dict[str, Any], Preprocessor, TabularModel]:
    """Official protocol: fit on all training engines, predict each test engine's last window."""
    pre, model = fit_pipeline(data.train, cfg, model_name)
    windows = last_windows(
        pre.transform(data.test), pre.feature_cols_, cfg.preprocess.window_length
    )
    pred = model.predict(windows)
    true = data.test_rul.loc[windows.units].to_numpy()
    test = {
        "engines": len(windows),
        "metrics": all_metrics(true, pred),
        "metrics_capped_truth": all_metrics(cap_rul(true, cfg.preprocess.rul_cap), pred),
        "predictions": [
            {"unit": int(u), "true_rul": int(t), "pred_rul": round(float(p), 3)}
            for u, t, p in zip(windows.units, true, pred, strict=True)
        ],
    }
    return test, pre, model


def run_experiment(
    data: CMAPSSData, model_name: str, cfg: Config, git: dict[str, Any] | None = None
) -> dict[str, Any]:
    start = time.perf_counter()
    folds = cross_validate(data.train, cfg, model_name)
    test, pre, model = evaluate_test(data, cfg, model_name)
    config = cfg.model_dump(mode="json")
    config["data"]["raw_dir"] = Path(cfg.data.raw_dir).as_posix()
    return {
        "dataset": data.dataset,
        "model": model_name,
        "synthetic": data.synthetic,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": {"package": __version__, **(git or git_state())},
        "config": config,
        "train_engines": int(data.train["unit"].nunique()),
        "dropped_sensors": pre.dropped_sensors_,
        "sensors_used": pre.feature_cols_,
        "cv": {"folds": folds, "summary": summarise_folds(folds)},
        "test": test,
        "top_features": dict(list(model.feature_importances().items())[:TOP_FEATURES]),
        "wall_seconds": round(time.perf_counter() - start, 1),
    }


def git_state() -> dict[str, Any]:
    """Current commit, and whether tracked code differs from it (results/ is ignored)."""
    try:
        commit = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain", "--", ".", ":(exclude)results"))
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def write_run(record: dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    if record["synthetic"] and out_dir.resolve() == RESULTS_DIR.resolve():
        raise ValueError(
            "Refusing to write a SYNTHETIC run into results/, which holds real C-MAPSS results. "
            "Use --smoke (writes to results/smoke/) or --out."
        )
    path = out_dir / "runs" / f"{record['dataset']}_{record['model']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", nargs="+", default=["FD001"], choices=[*DATASETS, "all"])
    parser.add_argument("--model", nargs="+", default=["all"], choices=[*MODEL_NAMES, "all"])
    parser.add_argument("--config", default=None, help="extra YAML merged last")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"SYNTHETIC data and tiny models ({SMOKE_CONFIG}), output to {SMOKE_DIR}/",
    )
    parser.add_argument("--out", default=None, help=f"output dir (default {RESULTS_DIR}/)")
    args = parser.parse_args(argv)
    if args.smoke and args.config:
        parser.error("--smoke already selects its config; drop --config")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("rul.features").setLevel(logging.WARNING)  # dropped sensors are recorded

    datasets = DATASETS if "all" in args.dataset else tuple(dict.fromkeys(args.dataset))
    models = MODEL_NAMES if "all" in args.model else tuple(dict.fromkeys(args.model))
    extra = SMOKE_CONFIG if args.smoke else args.config
    out_dir = Path(args.out) if args.out else (SMOKE_DIR if args.smoke else RESULTS_DIR)
    git = git_state()

    for ds in datasets:
        cfg = config_for(ds, extra=extra)
        data = load_dataset(cfg.data.raw_dir, ds)
        tag = " [SYNTHETIC DATA]" if data.synthetic else ""
        for name in models:
            logger.info("%s %s: training%s", ds, name, tag)
            record = run_experiment(data, name, cfg, git=git)
            path = write_run(record, out_dir)
            cv = record["cv"]["summary"]["rmse"]
            test = record["test"]["metrics"]
            print(
                f"{ds} {name:14s} CV RMSE {cv['mean']:6.2f} +/- {cv['std']:5.2f} | "
                f"test RMSE {test['rmse']:6.2f}  NASA {test['nasa_score']:10.1f}{tag} -> {path}"
            )


if __name__ == "__main__":
    main()
