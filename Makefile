# Task runner (GNU make). Windows: `winget install ezwinports.make`.
# Recipes are plain commands so they run under both cmd.exe (Windows) and sh (Linux/CI).

.DEFAULT_GOAL := help
export PYTHONUTF8 := 1

ifeq ($(OS),Windows_NT)
  BOOTSTRAP := py -3.11
  VENV_PY := .venv/Scripts/python.exe
else
  BOOTSTRAP := python3.11
  VENV_PY := .venv/bin/python
endif

# The project venv when it exists (local dev), otherwise `python` (CI, Docker).
PY ?= $(if $(wildcard $(VENV_PY)),$(VENV_PY),python)
DATASET ?= FD001

.PHONY: help venv install lint format test synthetic data-check prepare smoke train api dashboard

help:
	@echo Targets:
	@echo   venv        create .venv with Python 3.11
	@echo   install     install the package and dev tools into the active Python
	@echo   lint        ruff check + format check
	@echo   format      auto-format and fix lint
	@echo   test        run pytest
	@echo   synthetic   write SYNTHETIC C-MAPSS-format data to data/synthetic/CMAPSS
	@echo   data-check  check for real C-MAPSS files, print instructions if missing
	@echo   prepare     run the data pipeline on real data, DATASET=FD001 by default
	@echo   smoke       full pipeline on SYNTHETIC data

venv:
	$(BOOTSTRAP) -m venv .venv

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

test:
	$(PY) -m pytest

synthetic:
	$(PY) scripts/make_synthetic_data.py

data-check:
	$(PY) scripts/download_data.py

prepare:
	$(PY) -m rul.prepare --dataset $(DATASET)

smoke: synthetic
	$(PY) -m rul.prepare --config configs/smoke.yaml --dataset FD001 FD002 FD003 FD004

train:
	$(error `make train` arrives in Phase 2)

api:
	$(error `make api` arrives in Phase 4)

dashboard:
	$(error `make dashboard` arrives in Phase 5)
