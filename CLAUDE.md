# CLAUDE.md

Portfolio project: Remaining Useful Life (RUL) prediction on NASA C-MAPSS, shipped like a
real system (validated models, MLflow, FastAPI, Streamlit, Docker, drift monitoring, CI).

## Non-negotiable rules
1. **No fabricated results.** Every number in README/docs must come from an actual run
   (MLflow or a saved metrics file). Unknown = "TBD". If a model loses, say so plainly.
2. **Never commit** data, model binaries, `mlruns/`, `.env*`, keys or other secrets.
3. **No remote actions**: never create a remote repo, push or deploy. Give the user the commands.
4. **Dependencies**: exact pins in `pyproject.toml`; CPU-only PyTorch; justify anything heavy.
5. Independent build from public data only; no proprietary code.
6. Work in phases; each ends with lint + tests green, a conventional commit, a summary, then STOP.
7. SYNTHETIC data (`data/synthetic/`, `rul.synthetic`) is for tests/CI/smoke only and must be
   labelled synthetic wherever its outputs appear. Smoke outputs go to `results/smoke/` (ignored).

## Status and plan
Backs the "Predictive Maintenance ML Service" entry on the owner's Data Scientist resume. When a
phase produces results, report the real numbers (and the month finished) so the resume can be
updated to match the repo - never tune the repo to match the resume.

- Phase 2 (baselines) done Oct 2026: naive/linear/RF/XGBoost; XGBoost has the lowest CV RMSE on
  all four datasets. Numbers live in `results/metrics.md` only (generated; never retype them).
- Phase 1 (data pipeline) done. Real C-MAPSS files are in `data/raw/CMAPSS/` (git-ignored, from
  NASA's PCoE archive); `python -m rul.prepare --dataset FD001 FD002 FD003 FD004` runs on them.
- `.venv` already has torch 2.14.1+cpu installed but not pinned: pin it in Phase 3. XGBoost is
  pinned as `xgboost-cpu` (plain `xgboost` pulls ~480 MB of CUDA/NCCL on Linux); macOS uses `xgboost`.
- Docker is not installed on this machine; the owner installs Docker Desktop before Phase 5.

Remaining phases, one at a time:
2. ~~Baselines: Linear Regression, Random Forest, XGBoost, CV with engine-wise GroupKFold~~ (done)
3. LSTM and Transformer encoder (PyTorch)
4. MLflow tracking; conformal prediction intervals at a 90% target, reporting the measured coverage
5. FastAPI service + Dockerfile
6. Streamlit dashboard + Evidently drift monitoring
7. GitHub Actions CI + Azure deployment guide (the owner runs any login/deploy commands)

## Commands (GNU make; on Windows `winget install ezwinports.make`)
- `make venv && make install` - Python 3.11 venv + editable install with dev tools
- `make lint` / `make format` - ruff
- `make test` - pytest (uses a generated synthetic dataset; no real data needed)
- `make smoke` - full pipeline on synthetic data (outputs in `results/smoke/`, git-ignored)
- `make train DATASET=all` - engine-wise CV + official test protocol for every model; writes
  `results/runs/<DS>_<model>.json` and `results/metrics.{md,json}` (`MODEL=xgboost` for one model)
- `make results` - rebuild `results/metrics.{md,json}` from the run records
- `make data-check` - verify real files in `data/raw/CMAPSS/`
- `make prepare DATASET=FD002` - data pipeline summary on real data

## Conventions
- `src/` layout, package `rul`. Python 3.11, ruff (line length 100), type hints, `from __future__ import annotations`.
- Config: YAML in `configs/` (default -> per-dataset -> extra), validated by pydantic (`extra="forbid"`).
- Always pass `encoding="utf-8"` when opening text files (Windows defaults to cp1252).
- Leakage rules: split by engine (`unit`), never by row. Fit every preprocessing step
  (`rul.features.Preprocessor`) on training engines only, and refit it inside each CV fold.
- Test evaluation follows the official protocol: last window of each test engine vs `RUL_FDxxx.txt`.
  Headline test metrics use the raw (uncapped) true RUL; capped-truth metrics are a labelled extra.
- Model selection uses CV RMSE only; test metrics are reported, never used to choose.
- Models implement `fit(Windows)` / `predict(Windows)` and are registered in `rul.models`.
- Real runs are made from a clean commit (each run record stores the commit hash), so code and
  results land in separate commits.
- Commits: conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`).
