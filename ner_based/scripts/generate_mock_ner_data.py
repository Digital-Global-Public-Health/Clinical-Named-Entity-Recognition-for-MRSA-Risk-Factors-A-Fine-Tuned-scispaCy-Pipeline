#!/usr/bin/env python3
"""Generate synthetic Track A DocBins and sidecar CSVs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.ner.mock_data import MockNERDataConfig, generate_mock_ner_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("annotations/mock"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    paths = generate_mock_ner_data(MockNERDataConfig(out_dir=args.out_dir, seed=args.seed))
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
