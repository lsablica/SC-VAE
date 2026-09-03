"""Continuous-domain validation of the finite even-neighbor KL rule."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import optimize, special


def direct_kl(
    dimension: int, rho: np.ndarray | float, tolerance: float = 1e-14
) -> np.ndarray:
    """Independent NumPy recurrence used only by this validation sweep."""

    concentration = np.asarray(rho, dtype=np.float64)
    squared = concentration * concentration
    leading = -np.log1p(-squared)
    if dimension == 2:
        return leading

    half = dimension / 2.0
    term = ((1.0 - half) / half) * squared
    correction = term.copy()
    index = 1
    if dimension % 2 == 0:
        stop = dimension // 2 - 1
        while index < stop:
            term = (
                term
                * squared
                * (index + 1.0 - half)
                / (index + half)
                * index
                / (index + 1.0)
            )
            correction = correction + term
            index += 1
    else:
        q = (dimension - 1) // 2
        while index < 100_000:
            term = (
                term
                * squared
                * (index + 1.0 - half)
                / (index + half)
                * index
                / (index + 1.0)
            )
            correction = correction + term
            index += 1
            if index >= q:
                one_minus = np.maximum(
                    1.0 - squared, np.finfo(np.float64).tiny
                )
                factor = np.minimum(
                    1.0 / one_minus, 1.0 + index / (2.0 * q)
                )
                bound = (
                    (dimension - 1) * np.abs(term) * factor
                )
                if float(np.max(bound)) < tolerance:
                    break
        else:
            raise RuntimeError(
                f"Odd recurrence failed to converge for D={dimension}"
            )
    return (dimension - 1) * (leading - correction)


def even_neighbor_kl(
    dimension: int, rho: np.ndarray | float
) -> np.ndarray:
    return 0.5 * (
        direct_kl(dimension - 1, rho)
        + direct_kl(dimension + 1, rho)
    )


def laplace_kl(
    dimension: int, rho: np.ndarray | float
) -> np.ndarray:
    concentration = np.asarray(rho, dtype=np.float64)
    squared = concentration * concentration
    m = float(dimension - 1)
    width = (
        special.digamma(m)
        - special.digamma(m / 2.0)
        - math.log(2.0)
    )
    weight = (2.0 * concentration / (1.0 + squared)) ** 2
    return m * (
        np.log((1.0 + squared) / (1.0 - squared))
        - width * weight
    )


def _high_concentration_constant(dimension: int) -> float:
    m = float(dimension - 1)
    return m * (
        2.0 * math.log(2.0)
        + special.digamma(m / 2.0)
        - special.digamma(m)
    )


def _boundary_neighbor_error(dimension: int) -> float:
    return abs(
        0.5
        * (
            _high_concentration_constant(dimension - 1)
            + _high_concentration_constant(dimension + 1)
        )
        - _high_concentration_constant(dimension)
    )


def _maximize_error(
    dimension: int,
    approximation,
    grid_size: int,
) -> tuple[float, float]:
    grid = np.linspace(0.0, 1.0 - 1e-9, grid_size)
    reference = direct_kl(dimension, grid)
    errors = np.abs(approximation(dimension, grid) - reference)
    candidates = np.argsort(errors)[-4:]
    best_error = 0.0
    best_rho = 0.0
    for candidate in candidates:
        lo = max(0, int(candidate) - 2)
        hi = min(len(grid) - 1, int(candidate) + 2)
        result = optimize.minimize_scalar(
            lambda value: -abs(
                float(
                    approximation(dimension, value)
                    - direct_kl(dimension, value)
                )
            ),
            bounds=(grid[lo], grid[hi]),
            method="bounded",
            options={"xatol": 1e-14},
        )
        error = -float(result.fun)
        if error > best_error:
            best_error = error
            best_rho = float(result.x)
    return best_error, best_rho


def run_neighbor_validation(
    output_dir: str | Path,
    *,
    grid_size: int = 12_001,
) -> tuple[list[dict], dict]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for dimension in range(7, 200, 2):
        neighbor_error, neighbor_rho = _maximize_error(
            dimension, even_neighbor_kl, grid_size
        )
        boundary_error = _boundary_neighbor_error(dimension)
        neighbor_rho_value: float | str = neighbor_rho
        if boundary_error >= neighbor_error:
            neighbor_error = boundary_error
            neighbor_rho_value = "1-"
        laplace_error, laplace_rho = _maximize_error(
            dimension, laplace_kl, grid_size
        )
        rows.append(
            {
                "dimension": dimension,
                "neighbor_max_abs_error": neighbor_error,
                "neighbor_rho_at_max": neighbor_rho_value,
                "laplace_max_abs_error": laplace_error,
                "laplace_rho_at_max": laplace_rho,
                "laplace_to_neighbor_error_ratio": (
                    laplace_error / neighbor_error
                ),
            }
        )

    csv_path = output / "neighbor_approximation_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    worst_neighbor = max(
        rows, key=lambda row: row["neighbor_max_abs_error"]
    )
    worst_laplace = max(
        rows, key=lambda row: row["laplace_max_abs_error"]
    )
    summary = {
        "dimensions": {"start": 7, "stop": 199, "step": 2},
        "grid_size": grid_size,
        "interior_refinement": "bounded_scalar",
        "worst_neighbor": worst_neighbor,
        "worst_laplace": worst_laplace,
    }
    (output / "neighbor_approximation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--grid-size", type=int, default=12_001)
    args = parser.parse_args()
    _, summary = run_neighbor_validation(
        args.output_dir, grid_size=args.grid_size
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
