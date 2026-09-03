"""Validate that retained latent-layer results cover the paper method set."""

from __future__ import annotations

import csv
from pathlib import Path

from .runtime import summarize_runtime_records
from .utils import write_csv

FINAL = Path(__file__).resolve().parent / "final" / "results"
EXPECTED = {
    "spcauchy_direct",
    "spcauchy_direct_fixed",
    "spcauchy_neighbor",
    "spcauchy_laplace",
    "spcauchy_direct_autograd",
    "vmf_official",
    "vmf_robust",
    "power_spherical",
}


def main() -> None:
    runtime_path = FINAL / "latent_step_runtime.csv"
    with runtime_path.open(newline="", encoding="utf-8") as handle:
        runtime_rows = list(csv.DictReader(handle))
    summary = summarize_runtime_records(runtime_rows)
    write_csv(summary, FINAL / "latent_step_runtime_summary.csv")
    methods = {row["method"] for row in summary}
    if methods != EXPECTED:
        raise RuntimeError(
            f"latent-layer summary method mismatch: {sorted(methods)}"
        )
    print(f"validated {len(methods)} latent-layer benchmark routes")


if __name__ == "__main__":
    main()
