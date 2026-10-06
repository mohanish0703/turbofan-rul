import json
from pathlib import Path

import pandas as pd
import pytest

from rul.data import load_dataset
from rul.features import Preprocessor
from rul.metrics import all_metrics
from rul.models import TabularModel
from rul.report import main as report_main
from rul.splits import group_kfold
from rul.synthetic import FLAT_SENSORS
from rul.train import cross_validate, evaluate_test, main, run_experiment, write_run

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data(synthetic_dir):
    return load_dataset(synthetic_dir, "FD001")


def test_every_fit_sees_training_engines_only(monkeypatch, data, tiny_cfg):
    """Leakage guard: in CV each fold refits preprocessing and model on its training engines;
    the final model is fit on the training set and never on test rows."""
    cfg = tiny_cfg()
    pre_fits, model_fits = [], []
    pre_fit, model_fit = Preprocessor.fit, TabularModel.fit

    def spy_pre(self, train):
        pre_fits.append(train.copy())
        return pre_fit(self, train)

    def spy_model(self, windows):
        model_fits.append(set(windows.units.tolist()))
        return model_fit(self, windows)

    monkeypatch.setattr(Preprocessor, "fit", spy_pre)
    monkeypatch.setattr(TabularModel, "fit", spy_model)

    cross_validate(data.train, cfg, "linear")
    folds = group_kfold(data.train, cfg.cv.n_folds, cfg.seed)
    assert len(pre_fits) == len(model_fits) == len(folds)
    for frame, model_units, fold in zip(pre_fits, model_fits, folds, strict=True):
        assert set(frame["unit"]) == set(fold.train_units.tolist())
        assert model_units == set(fold.train_units.tolist())

    pre_fits.clear()
    evaluate_test(data, cfg, "linear")
    assert len(pre_fits) == 1
    pd.testing.assert_frame_equal(pre_fits[0], data.train)


def test_cv_validates_every_engine_once(data, tiny_cfg):
    folds = cross_validate(data.train, tiny_cfg(), "naive")
    assert len(folds) == 3
    assert sum(f["val_engines"] for f in folds) == data.train["unit"].nunique()
    assert {"rmse", "mae", "nasa_score", "nasa_score_mean"} <= folds[0].keys()


def test_test_protocol_last_window_and_uncapped_truth(data, tiny_cfg):
    # The synthetic test RULs are small, so lower the cap until it actually cuts some of them.
    cfg = tiny_cfg()
    cfg = cfg.model_copy(update={"preprocess": cfg.preprocess.model_copy(update={"rul_cap": 40})})
    test, _, _ = evaluate_test(data, cfg, "linear")
    preds = test["predictions"]
    assert test["engines"] == len(preds) == len(data.test_rul)
    true = [p["true_rul"] for p in preds]
    pred = [p["pred_rul"] for p in preds]
    assert true == data.test_rul.loc[[p["unit"] for p in preds]].tolist()
    assert max(true) > cfg.preprocess.rul_cap  # so capped and uncapped truth really differ
    assert test["metrics"]["rmse"] == pytest.approx(all_metrics(true, pred)["rmse"], abs=1e-2)
    capped = [min(t, cfg.preprocess.rul_cap) for t in true]
    assert test["metrics_capped_truth"]["rmse"] == pytest.approx(
        all_metrics(capped, pred)["rmse"], abs=1e-2
    )


def test_run_record_roundtrip(tmp_path, data, tiny_cfg):
    git = {"git_commit": "abc1234", "git_dirty": False}
    record = run_experiment(data, "xgboost", tiny_cfg(), git=git)
    path = write_run(record, tmp_path)
    assert path == tmp_path / "runs" / "FD001_xgboost.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["synthetic"] is True
    assert loaded["code_version"]["git_commit"] == "abc1234"
    assert len(loaded["cv"]["folds"]) == 3
    assert loaded["cv"]["summary"]["rmse"]["mean"] > 0
    assert loaded["dropped_sensors"] == [f"s_{i}" for i in FLAT_SENSORS]
    assert loaded["top_features"]
    assert loaded["config"]["models"]["xgboost"]["n_estimators"] == 20


def test_synthetic_runs_never_land_in_real_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="SYNTHETIC"):
        write_run({"synthetic": True, "dataset": "FD001", "model": "naive"}, "results")
    assert not (tmp_path / "results").exists()


def test_cli_end_to_end(tmp_path, monkeypatch, synthetic_dir):
    extra = tmp_path / "extra.yaml"
    extra.write_text(
        f"data:\n  raw_dir: '{synthetic_dir.as_posix()}'\ncv:\n  n_folds: 2\n"
        "models:\n  random_forest:\n    n_estimators: 5\n  xgboost:\n    n_estimators: 5\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(REPO_ROOT)  # configs/default.yaml is resolved from the working directory
    out = tmp_path / "out"
    args = ["--dataset", "FD001", "FD003", "--model", "naive", "xgboost", "--config", str(extra)]
    main([*args, "--out", str(out)])

    names = sorted(p.stem for p in (out / "runs").glob("*.json"))
    assert names == ["FD001_naive", "FD001_xgboost", "FD003_naive", "FD003_xgboost"]
    report_main(["--results", str(out)])
    assert "SYNTHETIC DATA" in (out / "metrics.md").read_text(encoding="utf-8")
