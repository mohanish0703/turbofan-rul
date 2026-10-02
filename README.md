# Turbofan RUL: predictive maintenance on NASA C-MAPSS

Predicts the Remaining Useful Life (in cycles) of turbofan engines from multivariate
sensor histories, and serves the result like a production system.

> **Status: work in progress (Phase 1 of 7: data pipeline).** No model results yet.
> All result tables will be filled from real runs only; until then they read **TBD**.

## Quickstart

```bash
make venv          # Python 3.11 virtual environment in .venv
make install       # package + dev tools
make test          # unit tests on generated SYNTHETIC data
make smoke         # pipeline end to end on SYNTHETIC data
```

On Windows, install GNU make first: `winget install ezwinports.make`.

## Data

The NASA C-MAPSS files are not included. Run `make data-check` for instructions; files go in
`data/raw/CMAPSS/`. `make synthetic` writes **synthetic** files in the same format so tests and
CI run without the real data. Synthetic results say nothing about real model quality.

## Results

| Model | Dataset | Test RMSE | Test NASA score |
|---|---|---|---|
| TBD | TBD | TBD | TBD |

## License

MIT (code only). The C-MAPSS dataset is provided by NASA and is not redistributed here.
