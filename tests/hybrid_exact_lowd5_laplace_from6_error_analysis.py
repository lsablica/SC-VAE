import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import hyp2f1

matplotlib.use("Agg")
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kl import (
    kl_divergence_spcauchy_approx,
)


def build_z_grid():
    return np.r_[
        np.linspace(1e-6, 0.10, 300),
        np.linspace(0.10, 0.97, 1200),
        np.linspace(0.97, 0.995, 200),
        np.linspace(0.995, 0.9995, 120),
    ]


def z_to_rho(z_values):
    z_values = np.asarray(z_values, dtype=float)
    sqrt_one_minus_z = np.sqrt(np.clip(1.0 - z_values, 0.0, 1.0))
    return (1.0 - sqrt_one_minus_z) / (1.0 + sqrt_one_minus_z)


def stable_log_hyp2f1_for_h_derivative(z_value, dimension, fd_step):
    c = dimension - 1.0
    delta = (dimension - 1) / 2.0
    log_one_minus_z = np.log1p(-z_value)

    # Use Euler's transform so the hypergeometric factor stays near 1 when z -> 1.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lp_core = hyp2f1(-fd_step, delta, c, z_value)
        lm_core = hyp2f1(fd_step, delta, c, z_value)
        lp = (-delta - fd_step) * log_one_minus_z + np.log(lp_core)
        lm = (-delta + fd_step) * log_one_minus_z + np.log(lm_core)

    return lp, lm


def h_true_scalar(z_value, dimension, h=1e-5):
    z_value = float(np.clip(z_value, 1e-12, 1 - 1e-10))

    if dimension in {2, 3, 4, 5}:
        rho_tensor = torch.tensor([[z_to_rho([z_value])[0]]], dtype=torch.float64)
        h_value = kl_divergence_spcauchy_approx(rho_tensor, dimension, approximation="hybrid").item()
        return h_value / (dimension - 1.0) + 0.5 * np.log1p(-z_value)

    lp, lm = stable_log_hyp2f1_for_h_derivative(z_value, dimension, h)
    if not (np.isfinite(lp) and np.isfinite(lm)):
        return np.nan
    return (lp - lm) / (2.0 * h) + np.log1p(-z_value)


def kl_true_scalar(z_value, dimension):
    h_value = h_true_scalar(z_value, dimension)
    return (dimension - 1.0) * (h_value - 0.5 * np.log1p(-z_value))


def kl_hybrid_scalar(z_value, dimension, device="cpu", dtype=torch.float64):
    rho_tensor = torch.tensor([[z_to_rho([z_value])[0]]], dtype=dtype, device=device)
    return kl_divergence_spcauchy_approx(rho_tensor, dimension, approximation="hybrid").item()


def kl_error_scalar(z_value, dimension, device="cpu"):
    true_value = kl_true_scalar(z_value, dimension)
    if not np.isfinite(true_value):
        return np.nan
    return true_value - kl_hybrid_scalar(z_value, dimension, device=device)


def maximize_abs_kl_error(dimension, device="cpu"):
    if dimension in {2, 3, 4, 5}:
        return {
            "d": dimension,
            "z_star": 0.0,
            "signed_kl_error": 0.0,
            "max_abs_kl_error": 0.0,
        }

    z_grid = build_z_grid()
    signed_errors = np.asarray([kl_error_scalar(z_value, dimension, device=device) for z_value in z_grid])
    abs_errors = np.abs(signed_errors)
    peak_index = int(np.nanargmax(abs_errors))

    lo = z_grid[max(peak_index - 2, 0)]
    hi = z_grid[min(peak_index + 2, len(z_grid) - 1)]
    result = minimize_scalar(
        lambda value: (
            np.inf
            if not np.isfinite(kl_error_scalar(value, dimension, device=device))
            else -abs(kl_error_scalar(value, dimension, device=device))
        ),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 400},
    )
    z_star = float(result.x)
    signed_error = float(kl_error_scalar(z_star, dimension, device=device))

    return {
        "d": dimension,
        "z_star": z_star,
        "signed_kl_error": signed_error,
        "max_abs_kl_error": abs(signed_error),
    }


def run_hybrid_exact_lowd5_laplace_from6_error_analysis(
    dimensions=None,
    output_dir=None,
    device="cpu",
):
    if dimensions is None:
        dimensions = list(range(2, 201))

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = output_dir or os.path.join(repo_root, "figures")
    os.makedirs(output_dir, exist_ok=True)

    rows = [maximize_abs_kl_error(dimension, device=device) for dimension in dimensions]
    dataframe = pd.DataFrame(rows)

    csv_path = os.path.join(output_dir, "hybrid_exact_lowd5_laplace_from6_max_error_vs_d.csv")
    figure_path = os.path.join(output_dir, "hybrid_exact_lowd5_laplace_from6_max_kl_error_vs_d.png")

    dataframe.to_csv(csv_path, index=False)

    exact_rows = dataframe[dataframe["d"] < 6]
    laplace_rows = dataframe[dataframe["d"] >= 6]

    plt.figure(figsize=(8, 5))
    plt.scatter(exact_rows["d"], exact_rows["max_abs_kl_error"], label="exact (d=2,...,5)")
    plt.plot(laplace_rows["d"], laplace_rows["max_abs_kl_error"], label="Laplace from d=6")
    plt.xlabel("d")
    plt.ylabel("max |KL error|")
    plt.title("Hybrid rule: exact for d=2,3,4,5; Laplace for d>=6")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    return dataframe, csv_path, figure_path


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataframe, csv_path, figure_path = run_hybrid_exact_lowd5_laplace_from6_error_analysis(
        device=device,
    )
    selected = dataframe[dataframe["d"].isin([2, 3, 4, 5, 6, 7, 10, 20, 50, 100, 200])]
    print(selected.to_string(index=False))
    print(f"\nSaved CSV to {csv_path}")
    print(f"Saved figure to {figure_path}")
