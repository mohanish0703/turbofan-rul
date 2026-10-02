"""Check for the NASA C-MAPSS files and explain how to get them if any are missing.

No download URL is hard-coded: NASA has moved this dataset several times, so the
script points you to where to look and checks that the files ended up in the right place.
"""

from __future__ import annotations

import argparse
import sys

from rul.data import DATASETS, dataset_files, is_synthetic

INSTRUCTIONS = """
How to get the data
-------------------
1. Find "CMAPSS Jet Engine Simulated Data" (Turbofan Engine Degradation Simulation,
   Saxena et al., 2008) in the NASA Prognostics Center of Excellence data repository.
   Search for "NASA PCoE data repository CMAPSS" if the page has moved. Mirrors also
   exist on Kaggle (search "NASA CMAPSS").
2. Download and unzip the archive (usually CMAPSSData.zip).
3. Copy these files into {raw_dir}/ :
     train_FD001.txt ... train_FD004.txt
     test_FD001.txt  ... test_FD004.txt
     RUL_FD001.txt   ... RUL_FD004.txt
4. Re-run: python scripts/download_data.py

No data yet? `make smoke` runs the whole pipeline on SYNTHETIC stand-in data.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default="data/raw/CMAPSS")
    parser.add_argument("--dataset", nargs="+", default=list(DATASETS), choices=DATASETS)
    args = parser.parse_args()

    missing = 0
    for ds in args.dataset:
        for path in dataset_files(args.raw_dir, ds).values():
            ok = path.exists()
            missing += not ok
            print(f"  [{'ok' if ok else 'MISSING'}] {path}")

    if is_synthetic(args.raw_dir):
        print(f"\nWARNING: {args.raw_dir} contains SYNTHETIC data, not real C-MAPSS files.")
    if missing:
        print(f"\n{missing} file(s) missing.")
        print(INSTRUCTIONS.format(raw_dir=args.raw_dir))
        return 1
    print("\nAll requested C-MAPSS files are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
