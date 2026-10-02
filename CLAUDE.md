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

## Commands (GNU make; on Windows `winget install ezwinports.make`)
- `make venv && make install` - Python 3.11 venv + editable install with dev tools
- `make lint` / `make format` - ruff
- `make test` - pytest (uses a generated synthetic dataset; no real data needed)
- `make smoke` - full pipeline on synthetic data
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
- Commits: conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`).
