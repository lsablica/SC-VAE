import argparse
import os
import sys
import time

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kl import (
    kl_divergence_spcauchy,
    kl_divergence_spcauchy2,
    kl_divergence_spcauchy_approx,
    kl_divergence_spcauchy_asympt,
    kl_divergence_spcauchy_combined,
    kl_divergence_spcauchy_reference,
)


def build_rho_grid(num_points):
    low_grid = np.linspace(0.01, 0.9, max(num_points // 2, 2))
    high_grid = np.linspace(0.9, 0.999, max(num_points - low_grid.size + 1, 2))
    return np.unique(np.concatenate([low_grid, high_grid]))


def timed_evaluation(fn, rho_tensor, device):
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    values = fn(rho_tensor)

    if device == "cuda":
        torch.cuda.synchronize()

    return values.detach().cpu().numpy(), (time.perf_counter() - start) * 1000


def summarize_dimension(dimension, rho_values, results, timings, output_dir):
    reference = results["reference"]

    print(f"\nDimension d={dimension}")
    print(f"{'Method':<16} {'Time (ms)':>12} {'Mean Abs Err':>14} {'Max Abs Err':>14} {'Mean Rel Err':>14} {'Max Rel Err':>14}")
    print("-" * 88)

    for method_name, values in results.items():
        abs_error = np.abs(values - reference)
        rel_error = abs_error / np.maximum(np.abs(reference), 1e-10)
        print(
            f"{method_name:<16} {timings[method_name]:>12.3f} "
            f"{abs_error.mean():>14.6e} {abs_error.max():>14.6e} "
            f"{rel_error.mean():>14.6e} {rel_error.max():>14.6e}"
        )

    figure, (ax_values, ax_errors) = plt.subplots(1, 2, figsize=(14, 5))
    ax_values.plot(rho_values, reference, label="reference", linewidth=2.5, color="black")

    for method_name, values in results.items():
        if method_name == "reference":
            continue
        ax_values.plot(rho_values, values, label=method_name)

    ax_values.set_title(f"KL vs rho (d={dimension})")
    ax_values.set_xlabel("rho")
    ax_values.set_ylabel("KL")
    ax_values.grid(True)
    ax_values.legend()

    for method_name, values in results.items():
        if method_name == "reference":
            continue
        rel_error = np.abs(values - reference) / np.maximum(np.abs(reference), 1e-10)
        ax_errors.semilogy(rho_values, rel_error, label=method_name)

    ax_errors.set_title(f"Relative error vs reference (d={dimension})")
    ax_errors.set_xlabel("rho")
    ax_errors.set_ylabel("relative error")
    ax_errors.grid(True, which="both")
    ax_errors.legend()

    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, f"kl_approximation_comparison_d{dimension}.png"))
    plt.close(figure)


def run_kl_approximation_comparison(
    dimensions=None,
    num_points=40,
    k_terms=2000,
    n_nodes=400,
    device="cpu",
    output_dir=None,
):
    if dimensions is None:
        dimensions = [3, 5, 10, 20, 100]

    rho_values = build_rho_grid(num_points)
    rho_tensor = torch.tensor(rho_values, dtype=torch.float64, device=device).view(-1, 1)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = output_dir or os.path.join(repo_root, "figures")
    os.makedirs(output_dir, exist_ok=True)

    for dimension in dimensions:
        method_specs = {
            "reference": lambda tensor: kl_divergence_spcauchy_reference(
                tensor,
                dimension,
                k_terms=max(k_terms, 4000),
                n_nodes=max(n_nodes, 2000),
            ),
            "combined_exact": lambda tensor: kl_divergence_spcauchy_combined(
                tensor,
                dimension,
                n_nodes=n_nodes,
            ),
            "series": lambda tensor: kl_divergence_spcauchy(
                tensor,
                dimension,
                k_terms=k_terms,
            ),
            "quadrature": lambda tensor: kl_divergence_spcauchy2(
                tensor,
                dimension,
                n_nodes=n_nodes,
            ),
            "asymptotic": lambda tensor: kl_divergence_spcauchy_asympt(tensor, dimension),
            "midpoint": lambda tensor: kl_divergence_spcauchy_approx(
                tensor,
                dimension,
                approximation="midpoint",
            ),
            "laplace": lambda tensor: kl_divergence_spcauchy_approx(
                tensor,
                dimension,
                approximation="laplace",
            ),
            "hybrid": lambda tensor: kl_divergence_spcauchy_approx(
                tensor,
                dimension,
                approximation="hybrid",
            ),
        }

        results = {}
        timings = {}

        for method_name, method_fn in method_specs.items():
            values, elapsed_ms = timed_evaluation(method_fn, rho_tensor, device)
            results[method_name] = values
            timings[method_name] = elapsed_ms

        summarize_dimension(dimension, rho_values, results, timings, output_dir)

    print(f"\nSaved plots to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare exact and surrogate spherical Cauchy KL evaluators.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--k_terms", type=int, default=2000)
    parser.add_argument("--n_nodes", type=int, default=400)
    parser.add_argument("--num_points", type=int, default=40)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[3, 5, 10, 20, 100])
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    run_kl_approximation_comparison(
        dimensions=args.dimensions,
        num_points=args.num_points,
        k_terms=args.k_terms,
        n_nodes=args.n_nodes,
        device=device,
        output_dir=args.output_dir,
    )
