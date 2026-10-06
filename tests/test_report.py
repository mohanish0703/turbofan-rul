"""Report tests on minimal hand-built run records (test fixtures, not results)."""

import json

import pytest

from rul.report import load_runs, render_markdown, summary_rows, write_report


def _metrics(rmse: float) -> dict[str, float]:
    return {"rmse": rmse, "mae": rmse * 0.8, "nasa_score": rmse * 10, "nasa_score_mean": rmse / 10}


def _run(dataset, model, cv_rmse, test_rmse, synthetic=False, rul_cap=125):
    return {
        "dataset": dataset,
        "model": model,
        "synthetic": synthetic,
        "code_version": {"package": "0.1.0", "git_commit": "abcdef123", "git_dirty": False},
        "config": {"preprocess": {"rul_cap": rul_cap, "window_length": 30}, "cv": {"n_folds": 5}},
        "train_engines": 100,
        "cv": {"summary": {k: {"mean": v, "std": 1.0} for k, v in _metrics(cv_rmse).items()}},
        "test": {
            "engines": 100,
            "metrics": _metrics(test_rmse),
            "metrics_capped_truth": _metrics(test_rmse - 1),
        },
    }


def _write(runs_dir, runs):
    runs_dir.mkdir(parents=True)
    for r in runs:
        (runs_dir / f"{r['dataset']}_{r['model']}.json").write_text(json.dumps(r), encoding="utf-8")


def test_selection_uses_cv_not_test():
    # xgboost has the better test RMSE, random_forest the better CV RMSE: CV decides.
    runs = [
        _run("FD001", "random_forest", cv_rmse=15.0, test_rmse=19.0),
        _run("FD001", "xgboost", cv_rmse=16.0, test_rmse=17.0),
    ]
    assert [r["model"] for r in summary_rows(runs) if r["selected"]] == ["random_forest"]


def test_rows_ordered_with_one_selection_per_dataset():
    runs = [_run("FD003", "xgboost", 14, 15), _run("FD001", "xgboost", 13, 14)]
    runs.append(_run("FD001", "naive", 40, 42))
    rows = summary_rows(runs)
    assert [(r["dataset"], r["model"]) for r in rows] == [
        ("FD001", "naive"), ("FD001", "xgboost"), ("FD003", "xgboost")
    ]  # fmt: skip
    assert [r["model"] for r in rows if r["selected"]] == ["xgboost", "xgboost"]


def test_markdown_tables():
    runs = [_run("FD001", "naive", 40, 42), _run("FD001", "xgboost", 13, 14)]
    runs.append(_run("FD003", "xgboost", 14, 15))
    md = render_markdown(runs, summary_rows(runs), "results/runs/")
    assert "| Model | FD001 | FD003 |" in md
    assert "| naive | 42.00 | - |" in md  # FD003 has no naive run
    assert (
        "| **xgboost** (selected) | 13.00 +/- 1.00 | 10.40 | 14.00 | 11.20 | 140.0 | 13.00 |" in md
    )
    assert "100 training / 100 test engines, 5 CV folds, RUL cap 125" in md
    assert "SYNTHETIC" not in md


def test_synthetic_report_is_labelled_and_kept_out_of_results(tmp_path, monkeypatch):
    runs = [_run("FD001", "naive", 40, 42, synthetic=True)]
    _write(tmp_path / "smoke" / "runs", runs)
    md_path, json_path = write_report(tmp_path / "smoke")
    assert "SYNTHETIC DATA" in md_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))["synthetic"] is True

    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "results" / "runs", runs)
    with pytest.raises(ValueError, match="SYNTHETIC"):
        write_report("results")
    assert not (tmp_path / "results" / "metrics.md").exists()


def test_mixed_or_inconsistent_runs_rejected(tmp_path):
    mixed = [_run("FD001", "naive", 40, 42), _run("FD001", "xgboost", 14, 15, synthetic=True)]
    _write(tmp_path / "mixed", mixed)
    with pytest.raises(ValueError, match="mixes"):
        load_runs(tmp_path / "mixed")

    runs = [_run("FD001", "naive", 40, 42), _run("FD001", "xgboost", 14, 15, rul_cap=130)]
    with pytest.raises(ValueError, match="different settings"):
        render_markdown(runs, summary_rows(runs), "x")


def test_no_runs(tmp_path):
    with pytest.raises(FileNotFoundError, match="rul.train"):
        load_runs(tmp_path)
