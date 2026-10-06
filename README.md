# Turbofan RUL: predictive maintenance on NASA C-MAPSS

Predicts the Remaining Useful Life (in cycles) of turbofan engines from multivariate
sensor histories, and serves the result like a production system.

> **Status: work in progress (Phase 2 of 7: baseline models).** The results below come from
> real runs. Deep models, MLflow tracking, prediction intervals, the API, the dashboard and CI
> arrive in later phases.

## Quickstart

```bash
make venv              # Python 3.11 virtual environment in .venv
make install           # package + dev tools
make test              # unit tests on generated SYNTHETIC data
make smoke             # whole pipeline on SYNTHETIC data, outputs in results/smoke/
make train DATASET=all # real data: CV + official test evaluation -> results/
```

On Windows, install GNU make first: `winget install ezwinports.make`.

## Data

The NASA C-MAPSS files are not included. Run `make data-check` for instructions; files go in
`data/raw/CMAPSS/`. `make synthetic` writes **synthetic** files in the same format so tests and
CI run without the real data. Synthetic results say nothing about real model quality.

## Method

- **Target**: RUL capped at 125 cycles (piece-wise linear). Early in life there is no visible
  degradation, so the model is not asked to tell "300 cycles left" from "200 cycles left".
- **Preprocessing**: near-constant sensors are dropped; FD002/FD004 rows are clustered into
  their 6 operating conditions and each sensor is standardised within its condition. Everything
  is fit on training engines only.
- **Features (baselines)**: each 30-cycle window becomes, per sensor, its last value, mean,
  standard deviation and slope, plus the engine's current cycle.
- **Validation**: 5-fold GroupKFold by engine, never by row, with preprocessing refit inside
  every fold. The model per dataset is chosen by CV RMSE alone.
- **Test**: the official protocol. The last window of each test engine is compared with
  `RUL_FDxxx.txt`, using the true (uncapped) RUL.
- **Metrics**: RMSE, MAE and the NASA asymmetric score, which penalises late predictions
  (predicting more life than is left) more than early ones.

## Results

Phase 2 baselines on all four C-MAPSS sub-datasets. Every number here is copied from
[results/metrics.md](results/metrics.md), which `make train` generates from the run records in
`results/runs/` (code version `8cdf8e9`). That file also has CV scores, MAE, the NASA score and
RMSE against capped truth.

Test RMSE in cycles (official protocol, true RUL not capped; lower is better):

| Model | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| naive | 41.87 | 53.78 | 43.57 | 54.57 |
| linear | 16.37 | 28.62 | 17.63 | 30.57 |
| random_forest | 13.12 | 25.90 | 14.67 | 26.72 |
| xgboost | 13.35 | 25.66 | 13.91 | 26.29 |

- XGBoost has the lowest CV RMSE on every dataset, so it is the selected baseline. On the FD001
  test set Random Forest is slightly better (13.12 vs 13.35); selection uses CV only, never test.
- On FD002/FD004 the XGBoost test RMSE is far above its CV RMSE (25.66 vs 14.84 and 26.29 vs
  14.33). In those test sets 57 of 259 and 67 of 248 engines have a true RUL above the 125-cycle
  cap the models are trained with, and a capped model cannot predict above it. Against capped
  truth the same predictions score 12.78 and 13.77 RMSE.
- Hyperparameters are hand-set defaults, not tuned. Deep models (LSTM, Transformer) come next.

## License

MIT (code only). The C-MAPSS dataset is provided by NASA and is not redistributed here.
