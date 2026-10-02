"""Run the data pipeline for one or more datasets and print a summary.

python -m rul.prepare --dataset FD001 FD002
python -m rul.prepare --dataset FD001 --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import logging

from rul.config import config_for
from rul.data import DATASETS, load_dataset
from rul.features import Preprocessor, cap_rul
from rul.splits import group_kfold
from rul.windows import last_windows, make_windows

logger = logging.getLogger(__name__)


def prepare(dataset: str, extra_config: str | None = None) -> dict:
    cfg = config_for(dataset, extra=extra_config)
    pp_cfg = cfg.preprocess
    data = load_dataset(cfg.data.raw_dir, dataset)

    pre = Preprocessor(
        n_conditions=pp_cfg.n_conditions,
        variance_threshold=pp_cfg.variance_threshold,
        seed=cfg.seed,
    )
    train = pre.fit_transform(data.train)
    train["rul"] = cap_rul(train["rul"], pp_cfg.rul_cap)
    test = pre.transform(data.test)

    train_w = make_windows(train, pre.feature_cols_, pp_cfg.window_length, target_col="rul")
    test_w = last_windows(test, pre.feature_cols_, pp_cfg.window_length)
    folds = group_kfold(data.train, cfg.cv.n_folds, cfg.seed)

    return {
        "dataset": dataset,
        "synthetic": data.synthetic,
        "train_engines": data.train["unit"].nunique(),
        "test_engines": data.test["unit"].nunique(),
        "n_conditions": pp_cfg.n_conditions,
        "dropped_sensors": pre.dropped_sensors_,
        "n_features": len(pre.feature_cols_),
        "train_windows": train_w.X.shape,
        "test_windows": test_w.X.shape,
        "val_engines_per_fold": [len(f.val_units) for f in folds],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", nargs="+", default=["FD001"], choices=DATASETS)
    parser.add_argument("--config", default=None, help="extra YAML merged last, e.g. smoke.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    for ds in args.dataset:
        summary = prepare(ds, args.config)
        tag = " [SYNTHETIC DATA]" if summary["synthetic"] else ""
        print(f"\n== {ds}{tag} ==")
        for key, value in summary.items():
            if key not in ("dataset", "synthetic"):
                print(f"  {key:22s} {value}")


if __name__ == "__main__":
    main()
