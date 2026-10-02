"""Write SYNTHETIC C-MAPSS-format files (NOT real engine data) for tests and smoke runs."""

from __future__ import annotations

import argparse

from rul.data import DATASETS
from rul.synthetic import generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/synthetic/CMAPSS")
    parser.add_argument("--train-units", type=int, default=20)
    parser.add_argument("--test-units", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", nargs="+", default=list(DATASETS), choices=DATASETS)
    args = parser.parse_args()

    out = generate(
        args.out,
        n_train_units=args.train_units,
        n_test_units=args.test_units,
        seed=args.seed,
        datasets=args.dataset,
    )
    print(f"Wrote SYNTHETIC data for {', '.join(args.dataset)} to {out}")
    print("These are NOT real C-MAPSS results; use them only for tests and smoke runs.")


if __name__ == "__main__":
    main()
