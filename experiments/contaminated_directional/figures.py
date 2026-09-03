"""Regenerate controlled-contamination figures from the compact aggregate."""

from __future__ import annotations

import csv
from pathlib import Path

from .aggregate import _plot


FINAL_ROOT = Path(__file__).resolve().parent / "final"


def _rows() -> list[dict]:
    path = FINAL_ROOT / "tables" / "aggregate.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric_prefixes = {
        "kappa": int,
        "epsilon": float,
        "seeds": int,
    }
    for row in rows:
        for key, value in tuple(row.items()):
            if key in numeric_prefixes:
                row[key] = numeric_prefixes[key](value)
            elif key.endswith(("_mean", "_sd")):
                row[key] = float(value)
    return rows


def main() -> None:
    _plot(_rows(), FINAL_ROOT)
    print("regenerated three controlled-contamination figure families")


if __name__ == "__main__":
    main()
