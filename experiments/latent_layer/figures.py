"""Build latent-layer figures from final CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
from scipy import integrate, special

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

FINAL_ROOT = Path(__file__).resolve().parent / "final"
LATENT_RESULTS = FINAL_ROOT / "results"
DEFAULT_FIGURE_DIR = FINAL_ROOT / "figures"

CURRENT_RUNTIME_METHODS = {
    "spcauchy_direct": "Spherical Cauchy direct",
    "spcauchy_neighbor": "Spherical Cauchy neighbor",
    "spcauchy_laplace": "Spherical Cauchy Laplace",
    "vmf_robust": "vMF robust",
    "vmf_official": "vMF original",
    "power_spherical": "Power Spherical",
}


def _matched_rho(dimension: int, kappa: float) -> float:
    m = float(dimension - 1)
    if kappa == 0:
        return 0.0
    return kappa / (m + kappa + math.sqrt(m * m + 2.0 * m * kappa))


def _direct_kl(dimension: int, rho: np.ndarray | float, tol: float = 1e-14) -> np.ndarray:
    r = np.asarray(rho, dtype=float)
    x = r * r
    result = -np.log1p(-x)
    if dimension == 2:
        return result

    half = dimension / 2.0
    term = ((1.0 - half) / half) * x
    correction = term.copy()
    index = 1
    if dimension % 2 == 0:
        stop = dimension // 2 - 1
        while index < stop:
            ratio = (index + 1.0 - half) / (index + half)
            term = term * x * ratio * (index / (index + 1.0))
            correction = correction + term
            index += 1
    else:
        q = (dimension - 1) // 2
        while index < 100000:
            ratio = (index + 1.0 - half) / (index + half)
            next_term = term * x * ratio * (index / (index + 1.0))
            next_index = index + 1
            correction = correction + next_term
            term = next_term
            index = next_index
            if next_index >= q:
                one_minus_x = np.maximum(1.0 - x, np.finfo(float).tiny)
                factor = np.minimum(
                    1.0 / one_minus_x, 1.0 + next_index / (2.0 * q)
                )
                bound = (dimension - 1) * np.abs(next_term) * factor
                if np.max(bound) < tol:
                    break
        else:
            raise RuntimeError(f"odd recurrence failed for D={dimension}")
    return (dimension - 1) * (result - correction)


def _ps_kl(dimension: int, kappa: np.ndarray) -> np.ndarray:
    a = (dimension - 1) / 2.0
    lam = 2.0 * kappa
    return (
        special.betaln(a, a)
        - special.betaln(a + lam, a)
        + lam * (special.digamma(a + lam) - special.digamma(2.0 * a + lam))
    )


def _vmf_kl(dimension: int, kappa: np.ndarray) -> np.ndarray:
    kappa = np.asarray(kappa, dtype=float)
    nu = dimension / 2.0 - 1.0
    out = np.empty_like(kappa)
    small = kappa < 1e-4
    if np.any(small):
        out[small] = (kappa[small] ** 2) / (2.0 * dimension)
    if np.any(~small):
        kk = kappa[~small]
        ive_nu = special.ive(nu, kk)
        ive_next = special.ive(nu + 1.0, kk)
        log_bessel = np.log(ive_nu) + kk
        log_constant_relative_uniform = (
            (1.0 - dimension / 2.0) * np.log(2.0)
            - special.gammaln(dimension / 2.0)
            + nu * np.log(kk)
            - log_bessel
        )
        out[~small] = log_constant_relative_uniform + kk * ive_next / ive_nu
    return out


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _successful(row: dict[str, str]) -> bool:
    return row.get("success", "True").lower() == "true"


def _save_matched_profiles(figure_dir: Path) -> Path:
    dimension = 3
    kappa = 8.0
    m = dimension - 1
    theta = np.linspace(0.0, np.pi, 1001)
    s = 1.0 - np.cos(theta)
    spcauchy = (1.0 + kappa * s / m) ** (-m)
    vmf = np.exp(-kappa * s)
    power_spherical = np.maximum(1.0 - s / 2.0, 0.0) ** (2.0 * kappa)

    kappa_grid = np.logspace(-2, np.log10(50.0), 350)
    rho_grid = np.array([_matched_rho(dimension, value) for value in kappa_grid])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    axes[0].plot(theta / np.pi, spcauchy, label="spherical Cauchy")
    axes[0].plot(theta / np.pi, vmf, linestyle="--", label="von Mises-Fisher")
    axes[0].plot(theta / np.pi, power_spherical, linestyle=":", label="Power Spherical")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-15, 1.5)
    axes[0].set_xlabel(r"geodesic angle $\theta/\pi$")
    axes[0].set_ylabel("density relative to the mode")
    axes[0].legend(loc="lower left")
    axes[0].grid(alpha=0.25)

    axes[1].plot(kappa_grid, _direct_kl(dimension, rho_grid), label="spherical Cauchy")
    axes[1].plot(
        kappa_grid,
        _vmf_kl(dimension, kappa_grid),
        linestyle="--",
        label="von Mises-Fisher",
    )
    axes[1].plot(
        kappa_grid,
        _ps_kl(dimension, kappa_grid),
        linestyle=":",
        label="Power Spherical",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"matched local curvature $\kappa$")
    axes[1].set_ylabel("KL to the uniform prior")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    path = figure_dir / "matched_curvature_profiles.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_matched_survival(figure_dir: Path) -> Path:
    dimension = 3
    kappa = 8.0
    m = dimension - 1
    theta = np.linspace(0.0, np.pi, 5001)
    s = 1.0 - np.cos(theta)
    base = np.sin(theta) ** (dimension - 2)
    profiles = {
        "spherical Cauchy": (1.0 + kappa * s / m) ** (-m),
        "von Mises-Fisher": np.exp(-kappa * s),
        "Power Spherical": np.maximum(1.0 - s / 2.0, 0.0) ** (2.0 * kappa),
    }

    fig, axis = plt.subplots(figsize=(7.4, 4.6))
    for label, profile in profiles.items():
        density = base * profile
        density = density / integrate.trapezoid(density, theta)
        cumulative = integrate.cumulative_trapezoid(density, theta, initial=0.0)
        survival = np.maximum(1.0 - cumulative, 0.0)
        axis.plot(theta / np.pi, survival, label=label)
    axis.set_yscale("log")
    axis.set_ylim(1e-8, 1.2)
    axis.set_xlabel(r"geodesic angle $\theta/\pi$")
    axis.set_ylabel(r"survival probability $\Pr(\Theta>\theta)$")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = figure_dir / "matched_curvature_survival.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_runtime_total(figure_dir: Path) -> Path:
    rows = _rows(LATENT_RESULTS / "latent_step_runtime.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
    for axis, device in zip(axes, ("cpu", "cuda")):
        for method, label in CURRENT_RUNTIME_METHODS.items():
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["device"] == device
                    and row["method"] == method
                    and _successful(row)
                    and row["total_median_s"]
                ),
                key=lambda row: int(row["dim"]),
            )
            if selected:
                axis.plot(
                    [int(row["dim"]) for row in selected],
                    [1e3 * float(row["total_median_s"]) for row in selected],
                    marker="o",
                    markersize=3,
                    linewidth=1.3,
                    label=label,
                )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(device.upper())
        axis.set_xlabel("Ambient dimension")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Forward + backward median (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    path = figure_dir / "benchmark_spcauchy_vs_vmf_stress.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_runtime_decomposition(figure_dir: Path) -> Path:
    rows = _rows(LATENT_RESULTS / "latent_step_runtime.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    methods = list(CURRENT_RUNTIME_METHODS)
    for axis, device in zip(axes, ("cpu", "cuda")):
        chosen = []
        for method in methods:
            candidates = [
                row
                for row in rows
                if row["device"] == device
                and row["method"] == method
                and _successful(row)
                and row["forward_median_s"]
                and row["backward_median_s"]
            ]
            if candidates:
                chosen.append(
                    min(candidates, key=lambda row: abs(int(row["dim"]) - 128))
                )
        x = np.arange(len(chosen))
        forward = np.asarray([1e3 * float(row["forward_median_s"]) for row in chosen])
        backward = np.asarray([1e3 * float(row["backward_median_s"]) for row in chosen])
        width = 0.4
        axis.bar(x - width / 2, forward, width=width, label="Forward")
        axis.bar(x + width / 2, backward, width=width, label="Backward")
        axis.set_xticks(
            x,
            [CURRENT_RUNTIME_METHODS[row["method"]] for row in chosen],
            rotation=35,
            ha="right",
        )
        axis.set_yscale("log")
        axis.set_title(f"{device.upper()} (nearest retained dimension to 128)")
        axis.grid(axis="y", which="major", alpha=0.22)
    axes[0].yaxis.set_major_locator(
        mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12)
    )
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: f"{value:g}")
    )
    axes[0].yaxis.set_minor_formatter(mticker.NullFormatter())
    axes[0].set_ylabel("Median time (ms)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "latent_step_runtime_forward_backward.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_kl_summary(figure_dir: Path) -> Path:
    neighbor = _rows(LATENT_RESULTS / "neighbor_approximation_errors.csv")
    accuracy = _rows(LATENT_RESULTS / "direct_kl_accuracy_summary.csv")
    dimensions = np.asarray([int(row["dimension"]) for row in neighbor])
    neighbor_error = np.asarray(
        [float(row["neighbor_max_abs_error"]) for row in neighbor]
    )
    laplace_error = np.asarray(
        [float(row["laplace_max_abs_error"]) for row in neighbor]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    axes[0].plot(dimensions, neighbor_error, label="Even-neighbor", linewidth=1.6)
    axes[0].plot(dimensions, laplace_error, label="Laplace", linewidth=1.6)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Odd ambient dimension")
    axes[0].set_ylabel("Maximum absolute KL error")
    axes[0].set_title("Approximation error vs. exact direct reference")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)

    allowed = [
        row for row in accuracy if row["method"] in {"direct", "neighbor", "laplace"}
    ]
    labels = {
        "direct": "Direct",
        "neighbor": "Even-neighbor",
        "laplace": "Laplace",
    }
    axes[1].bar(
        [labels[row["method"]] for row in allowed],
        [max(float(row["max_abs_error"]), 1e-16) for row in allowed],
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Maximum absolute KL error")
    axes[1].set_title("Retained evaluation grid")
    axes[1].grid(axis="y", alpha=0.22)
    fig.tight_layout()
    path = figure_dir / "kl_evaluation_summary.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_odd_terms(figure_dir: Path) -> Path:
    rows = _rows(LATENT_RESULTS / "direct_kl_odd_term_counts.csv")
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["dim"])].append(row)
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for dimension in (7, 9, 17, 33):
        selected = sorted(grouped[dimension], key=lambda row: float(row["rho"]))
        axis.plot(
            [float(row["rho"]) for row in selected],
            [int(row["retained_terms"]) for row in selected],
            marker="o",
            label=f"D={dimension}",
        )
    axis.set_xlabel(r"Concentration $\rho$")
    axis.set_ylabel("Retained correction terms")
    axis.set_title(r"Certified direct recurrence at tolerance $10^{-10}$")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "direct_kl_odd_terms.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _matrix(
    rows: Iterable[dict[str, str]],
    *,
    method: str,
    dimensions: list[int],
    concentrations: list[float],
    concentration_key: str,
    value_key: str,
) -> np.ndarray:
    lookup = {
        (int(row["dim"]), float(row[concentration_key])): row
        for row in rows
        if row["method"] == method and row[concentration_key]
    }
    result = np.full((len(concentrations), len(dimensions)), np.nan)
    for i, concentration in enumerate(concentrations):
        for j, dimension in enumerate(dimensions):
            row = lookup.get((dimension, concentration))
            if row is not None and row.get(value_key, ""):
                result[i, j] = float(row[value_key])
    return result


def _save_spcauchy_robustness(figure_dir: Path) -> Path:
    rows = [
        row
        for row in _rows(LATENT_RESULTS / "robustness_grid.csv")
        if row["family"] == "spcauchy"
        and row["method"] in {"direct", "neighbor", "laplace"}
    ]
    dimensions = sorted({int(row["dim"]) for row in rows})
    rhos = sorted({float(row["rho"]) for row in rows if row["rho"]})
    fig = plt.figure(figsize=(13.5, 4.4))
    grid = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, 0.06), wspace=0.3)
    axes = [fig.add_subplot(grid[0, 0])]
    axes.extend(
        fig.add_subplot(grid[0, column], sharex=axes[0], sharey=axes[0])
        for column in (1, 2)
    )
    colorbar_axis = fig.add_subplot(grid[0, 3])
    image_handle = None
    for axis, method in zip(axes, ("direct", "neighbor", "laplace")):
        values = _matrix(
            rows,
            method=method,
            dimensions=dimensions,
            concentrations=rhos,
            concentration_key="rho",
            value_key="abs_error",
        )
        values = np.log10(np.maximum(values, 1e-16))
        image_handle = axis.imshow(
            values, aspect="auto", origin="lower", vmin=-16, vmax=0
        )
        axis.set_title(
            {"direct": "Direct", "neighbor": "Even-neighbor", "laplace": "Laplace"}[
                method
            ]
        )
        axis.set_xticks(range(len(dimensions)), dimensions, rotation=45)
        axis.set_yticks(range(len(rhos)), [f"{rho:g}" for rho in rhos])
        axis.set_xlabel("Ambient dimension")
    axes[0].set_ylabel(r"Concentration $\rho$")
    assert image_handle is not None
    fig.colorbar(
        image_handle,
        cax=colorbar_axis,
        label=r"$\log_{10}$ absolute KL error",
    )
    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.2, top=0.9)
    path = figure_dir / "robustness_spcauchy_heatmap.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_vmf_robustness(figure_dir: Path) -> Path:
    rows = [
        row
        for row in _rows(LATENT_RESULTS / "robustness_grid.csv")
        if row["method"] in {"vmf_official", "vmf_robust", "power_spherical"}
    ]
    dimensions = sorted({int(row["dim"]) for row in rows})
    kappas = sorted({float(row["kappa"]) for row in rows if row["kappa"]})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    titles = {
        "vmf_official": "vMF original",
        "vmf_robust": "vMF robust",
        "power_spherical": "Power Spherical",
    }
    for axis, method in zip(axes, titles):
        lookup = {
            (int(row["dim"]), float(row["kappa"])): 1.0 if _successful(row) else 0.0
            for row in rows
            if row["method"] == method and row["kappa"]
        }
        values = np.asarray(
            [
                [lookup.get((dimension, kappa), np.nan) for dimension in dimensions]
                for kappa in kappas
            ]
        )
        axis.imshow(
            values, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="RdYlGn"
        )
        axis.set_title(titles[method])
        axis.set_xticks(range(len(dimensions)), dimensions, rotation=45)
        axis.set_yticks(range(len(kappas)), [f"{kappa:g}" for kappa in kappas])
        axis.set_xlabel("Ambient dimension")
    axes[0].set_ylabel(r"Concentration $\kappa$")
    fig.suptitle("Finite forward/backward latent-step evaluation (red = failure)")
    fig.tight_layout()
    path = figure_dir / "robustness_vmf_heatmap.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def build_all_figures(figure_dir: Path = DEFAULT_FIGURE_DIR) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    return [
        _save_matched_profiles(figure_dir),
        _save_matched_survival(figure_dir),
        _save_runtime_total(figure_dir),
        _save_runtime_decomposition(figure_dir),
        _save_kl_summary(figure_dir),
        _save_odd_terms(figure_dir),
        _save_spcauchy_robustness(figure_dir),
        _save_vmf_robustness(figure_dir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()
    paths = build_all_figures(args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
