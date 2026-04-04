"""Accuracy benchmark for spCauchy KL evaluators."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .defaults import DEFAULT_ACCURACY_DIMS, DEFAULT_SPCAUCHY_ACCURACY_METHODS, DEFAULT_SPCAUCHY_RHO_GRID
from .methods import get_spcauchy_kl_method
from .utils import (
    OutputLayout,
    compute_abs_rel_error,
    dtype_name,
    finite_or_none,
    format_float,
    maybe_sync,
    prepare_output_layout,
    resolve_device,
    safe_scalar,
    seed_all,
    write_csv,
)
from src.kl import (
    _get_spcauchy_bracket_terms,
    _prepare_spcauchy_inputs,
    kl_divergence_spcauchy_approx,
)

try:
    import mpmath as mp
except ImportError:  # pragma: no cover - optional dependency
    mp = None


@dataclass
class ReferenceResult:
    value: float | None
    source: str
    success: bool
    failure_type: str | None = None
    error_message: str | None = None


@dataclass
class AccuracyConfig:
    dims: list[int] = field(default_factory=lambda: list(DEFAULT_ACCURACY_DIMS))
    rho_grid: list[float] = field(default_factory=lambda: list(DEFAULT_SPCAUCHY_RHO_GRID))
    methods: list[str] = field(default_factory=lambda: list(DEFAULT_SPCAUCHY_ACCURACY_METHODS))
    device_name: str = "cpu"
    dtype: torch.dtype = torch.float64
    seed: int = 0
    quadrature_nodes: int | None = 1000
    reference_nodes: int = 4096
    series_k_terms: int | None = None
    reference_k_terms: int | None = None
    reference_disagreement_tol: float = 1e-8
    mpmath_max_rho: float = 0.995
    out_dir: str | None = None


def _evaluate_kl_method(
    evaluator: Callable[..., torch.Tensor],
    rho_value: float,
    latent_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    quadrature_nodes: int | None = None,
    series_k_terms: int | None = None,
) -> tuple[float | None, float, bool, str | None, str | None]:
    rho = torch.tensor([[rho_value]], dtype=dtype, device=device)
    maybe_sync(device)
    t0 = time.perf_counter()
    try:
        value_tensor = evaluator(
            rho,
            latent_dim,
            quadrature_nodes=quadrature_nodes,
            series_k_terms=series_k_terms,
        )
        maybe_sync(device)
        elapsed = time.perf_counter() - t0
        value = safe_scalar(value_tensor)
    except RuntimeError as exc:
        maybe_sync(device)
        elapsed = time.perf_counter() - t0
        return None, elapsed, False, "runtime_error", str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        maybe_sync(device)
        elapsed = time.perf_counter() - t0
        return None, elapsed, False, "exception", str(exc)

    if value is None or not math.isfinite(value):
        return None, elapsed, False, "non_finite", "KL evaluator returned NaN/Inf."
    return value, elapsed, True, None, None


def _mpmath_reference(rho_value: float, latent_dim: int, *, mp_dps: int = 80) -> ReferenceResult:
    if mp is None:
        return ReferenceResult(
            value=None,
            source="mpmath_unavailable",
            success=False,
            failure_type="missing_dependency",
            error_message="mpmath is not installed.",
        )

    mp.mp.dps = mp_dps
    rho = mp.mpf(max(min(rho_value, 1.0 - 1e-8), 1e-8))
    c = mp.mpf(latent_dim - 1)
    z = 4 * rho / (1 + rho) ** 2

    def integrand(t):
        if t == 1:
            t = mp.mpf("1.0") - mp.mpf("1e-30")
        numerator = t ** (c - 1)
        denominator = 1 - t
        factor = (1 - z) / (1 - t * z)
        return (numerator / denominator) * (1 - factor ** (c / 2))

    try:
        integral = mp.quad(integrand, [0, 0.5, 0.9, 0.99, 0.999, 1])
        kl = c * mp.log((1 - rho) / (1 + rho)) + c * integral
        return ReferenceResult(value=float(kl), source="mpmath", success=True)
    except Exception as exc:  # pragma: no cover - optional path
        return ReferenceResult(
            value=None,
            source="mpmath_failed",
            success=False,
            failure_type="mpmath_failure",
            error_message=str(exc),
        )


def compute_reference_value(
    rho_value: float,
    latent_dim: int,
    config: AccuracyConfig,
) -> ReferenceResult:
    if latent_dim in {2, 3, 4, 5}:
        method = get_spcauchy_kl_method("hybrid")
        value, _, success, failure_type, error_message = _evaluate_kl_method(
            method.evaluator,
            rho_value,
            latent_dim,
            torch.device("cpu"),
            torch.float64,
        )
        return ReferenceResult(
            value=value,
            source="exact_lowd_hybrid",
            success=success,
            failure_type=failure_type,
            error_message=error_message,
        )

    series_k_terms = config.reference_k_terms or max(8000, 20 * latent_dim)
    quadrature_method = get_spcauchy_kl_method("combined")
    series_method = get_spcauchy_kl_method("series")

    q_value, _, q_success, q_failure, q_message = _evaluate_kl_method(
        quadrature_method.evaluator,
        rho_value,
        latent_dim,
        torch.device("cpu"),
        torch.float64,
        quadrature_nodes=config.reference_nodes,
    )
    s_value, _, s_success, s_failure, s_message = _evaluate_kl_method(
        series_method.evaluator,
        rho_value,
        latent_dim,
        torch.device("cpu"),
        torch.float64,
        series_k_terms=series_k_terms,
    )

    if q_success and s_success:
        difference = abs(q_value - s_value)
        threshold = config.reference_disagreement_tol * max(abs(q_value), abs(s_value), 1.0)
        if difference <= threshold:
            return ReferenceResult(value=q_value, source="strict_combined", success=True)

        if rho_value <= config.mpmath_max_rho:
            mp_reference = _mpmath_reference(rho_value, latent_dim)
            if mp_reference.success:
                return mp_reference
        return ReferenceResult(
            value=q_value,
            source="strict_combined_disagreement_no_mpmath",
            success=True,
            failure_type="reference_disagreement",
            error_message=f"Combined/series disagreement {difference:.3e} exceeded threshold {threshold:.3e}.",
        )

    if q_success:
        return ReferenceResult(value=q_value, source="strict_combined", success=True)

    if s_success:
        return ReferenceResult(
            value=s_value,
            source="long_series",
            success=True,
            failure_type=q_failure,
            error_message=q_message,
        )

    if rho_value <= config.mpmath_max_rho:
        mp_reference = _mpmath_reference(rho_value, latent_dim)
        if mp_reference.success:
            return mp_reference
    else:
        mp_reference = ReferenceResult(
            value=None,
            source="mpmath_skipped_high_rho",
            success=False,
            failure_type="skipped_high_rho",
            error_message=f"Skipped mpmath fallback for rho={rho_value:.3f} > {config.mpmath_max_rho:.3f}.",
        )

    return ReferenceResult(
        value=None,
        source="reference_failed",
        success=False,
        failure_type=q_failure or s_failure or mp_reference.failure_type,
        error_message="; ".join(
            message
            for message in [q_message, s_message, mp_reference.error_message]
            if message
        )
        or "No reference method succeeded.",
    )


def run_accuracy_benchmark(config: AccuracyConfig) -> list[dict]:
    seed_all(config.seed)
    device = resolve_device(config.device_name)
    records: list[dict] = []

    for latent_dim in config.dims:
        for rho_value in config.rho_grid:
            reference = compute_reference_value(rho_value, latent_dim, config)
            for method_name in config.methods:
                method = get_spcauchy_kl_method(method_name)
                value, elapsed, success, failure_type, error_message = _evaluate_kl_method(
                    method.evaluator,
                    rho_value,
                    latent_dim,
                    device,
                    config.dtype,
                    quadrature_nodes=config.quadrature_nodes,
                    series_k_terms=config.series_k_terms,
                )
                abs_error, rel_error = compute_abs_rel_error(value, reference.value)
                records.append(
                    {
                        "benchmark": "accuracy",
                        "method": method.name,
                        "family": method.family,
                        "device": device.type,
                        "dtype": dtype_name(config.dtype),
                        "seed": config.seed,
                        "dim": latent_dim,
                        "rho": rho_value,
                        "kappa": None,
                        "batch_size": None,
                        "success": success,
                        "failure_type": failure_type,
                        "error_message": error_message,
                        "kl_value": finite_or_none(value),
                        "eval_time_s": elapsed,
                        "reference_value": finite_or_none(reference.value),
                        "reference_source": reference.source,
                        "reference_success": reference.success,
                        "reference_failure_type": reference.failure_type,
                        "reference_error_message": reference.error_message,
                        "abs_error": abs_error,
                        "rel_error": rel_error,
                    }
                )

    return records


def save_accuracy_outputs(
    records: list[dict],
    *,
    out_dir: str | None = None,
    csv_name: str = "spcauchy_kl_accuracy.csv",
    generate_plots: bool = True,
) -> tuple[OutputLayout, str, list[str]]:
    layout = prepare_output_layout(out_dir)
    csv_path = write_csv(records, layout.results_dir / csv_name)
    figures = plot_accuracy_results(records, layout) if generate_plots else []
    return layout, str(csv_path), figures


def _compute_bound_kl_curves(rho_values: list[float], latent_dim: int) -> tuple[list[float], list[float], list[float], list[float], float]:
    rho = torch.tensor(rho_values, dtype=torch.float64).view(-1, 1)
    _, z = _prepare_spcauchy_inputs(rho)
    lower_h, upper_h, _ = _get_spcauchy_bracket_terms(z, latent_dim)
    correction = 0.5 * torch.log1p(-z)
    scale = latent_dim - 1

    lower_kl = (scale * (lower_h - correction)).squeeze(-1).cpu().tolist()
    upper_kl = (scale * (upper_h - correction)).squeeze(-1).cpu().tolist()
    midpoint_kl = kl_divergence_spcauchy_approx(rho, latent_dim, approximation="midpoint").cpu().tolist()
    laplace_kl = kl_divergence_spcauchy_approx(rho, latent_dim, approximation="laplace").cpu().tolist()
    kl_width = float(upper_kl[0] - lower_kl[0]) if upper_kl and lower_kl else float("nan")
    return lower_kl, upper_kl, midpoint_kl, laplace_kl, kl_width


def _compute_combined_curve(
    rho_values: list[float],
    latent_dim: int,
    *,
    n_nodes: int = 2000,
) -> list[float]:
    rho = torch.tensor(rho_values, dtype=torch.float64).view(-1, 1)
    method = get_spcauchy_kl_method("combined")
    values = method.evaluator(rho, latent_dim, quadrature_nodes=n_nodes)
    return values.cpu().tolist()


def _mask_outside_band(values: list[float], lower: list[float], upper: list[float]) -> list[float]:
    masked: list[float] = []
    for value, lo, hi in zip(values, lower, upper):
        if lo <= value <= hi:
            masked.append(value)
        else:
            masked.append(float("nan"))
    return masked


def _normalize_within_bracket(values: list[float], lower: list[float], upper: list[float]) -> list[float]:
    normalized: list[float] = []
    for value, lo, hi in zip(values, lower, upper):
        width = hi - lo
        if abs(width) < 1e-12:
            normalized.append(float("nan"))
        else:
            normalized.append((value - lo) / width)
    return normalized


def plot_accuracy_results(records: list[dict], layout: OutputLayout) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    figure_paths: list[str] = []
    if not records:
        return figure_paths

    representative_dims = [2, 8, 64, 512, 2048]
    representative_rhos = [0.5, 0.9, 0.99, 0.995]
    methods = sorted({record["method"] for record in records})

    fig, axes = plt.subplots(len(representative_dims), 2, figsize=(12, 3.2 * len(representative_dims)), squeeze=False)
    for row_idx, latent_dim in enumerate(representative_dims):
        subset = [record for record in records if record["dim"] == latent_dim and record["success"]]
        if not subset:
            continue
        for method_name in methods:
            method_rows = sorted(
                [record for record in subset if record["method"] == method_name],
                key=lambda item: item["rho"],
            )
            if not method_rows:
                continue
            rho_values = [row["rho"] for row in method_rows]
            abs_errors = [max(row["abs_error"] or 0.0, 1e-18) for row in method_rows]
            rel_errors = [max(row["rel_error"] or 0.0, 1e-18) for row in method_rows]
            axes[row_idx, 0].semilogy(rho_values, abs_errors, marker="o", label=method_name)
            axes[row_idx, 1].semilogy(rho_values, rel_errors, marker="o", label=method_name)

        axes[row_idx, 0].set_title(f"Absolute error vs rho (d={latent_dim})")
        axes[row_idx, 1].set_title(f"Relative error vs rho (d={latent_dim})")
        axes[row_idx, 0].set_xlabel("rho")
        axes[row_idx, 1].set_xlabel("rho")
        axes[row_idx, 0].set_ylabel("absolute error")
        axes[row_idx, 1].set_ylabel("relative error")
        axes[row_idx, 0].grid(True, which="both", alpha=0.3)
        axes[row_idx, 1].grid(True, which="both", alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.subplots_adjust(top=0.93, hspace=0.5, wspace=0.3)
    error_path = layout.figures_dir / "spcauchy_kl_error_vs_rho.png"
    fig.savefig(error_path, dpi=200)
    plt.close(fig)
    figure_paths.append(str(error_path))

    fig, axes = plt.subplots(len(representative_dims), 2, figsize=(13, 3.4 * len(representative_dims)), squeeze=False)
    for row_idx, latent_dim in enumerate(representative_dims):
        full_ax = axes[row_idx, 0]
        zoom_ax = axes[row_idx, 1]
        subset = [record for record in records if record["dim"] == latent_dim and record["success"]]
        if not subset:
            continue

        rho_values = sorted({record["rho"] for record in subset})
        lower_kl, upper_kl, midpoint_kl, laplace_kl, kl_width = _compute_bound_kl_curves(rho_values, latent_dim)
        combined_kl = _compute_combined_curve(rho_values, latent_dim)
        full_ax.fill_between(rho_values, lower_kl, upper_kl, color="#cfe8ff", alpha=0.5, label="bracket bounds")
        zoom_ax.fill_between(rho_values, lower_kl, upper_kl, color="#cfe8ff", alpha=0.5, label="bracket bounds")
        full_ax.plot(rho_values, midpoint_kl, linestyle="--", color="#1d3557", label="midpoint")
        zoom_ax.plot(rho_values, midpoint_kl, linestyle="--", color="#1d3557", label="midpoint")
        plotted_methods = ["combined", "series", "asymptotic_high_rho"]
        for method_name in plotted_methods:
            if method_name == "combined":
                method_rho = rho_values
                method_kl = combined_kl
            else:
                method_rows = sorted(
                    [record for record in subset if record["method"] == method_name and record["kl_value"] is not None],
                    key=lambda item: item["rho"],
                )
                if not method_rows:
                    continue
                method_rho = [row["rho"] for row in method_rows]
                method_kl = [row["kl_value"] for row in method_rows]
            full_ax.plot(
                method_rho,
                method_kl,
                marker="o",
                linewidth=1.2,
                label=method_name,
            )
            lower_lookup = {rho: value for rho, value in zip(rho_values, lower_kl)}
            upper_lookup = {rho: value for rho, value in zip(rho_values, upper_kl)}
            masked_kl = _mask_outside_band(
                method_kl,
                [lower_lookup[rho] for rho in method_rho],
                [upper_lookup[rho] for rho in method_rho],
            )
            zoom_ax.plot(
                method_rho,
                masked_kl,
                marker="o",
                linewidth=1.2,
                label=method_name,
            )

        full_ax.plot(rho_values, laplace_kl, linestyle="-.", color="#d62828", label="laplace")
        zoom_ax.plot(rho_values, laplace_kl, linestyle="-.", color="#d62828", label="laplace")

        band_min = min(lower_kl)
        band_max = max(upper_kl)
        band_pad = max((band_max - band_min) * 0.2, 1e-6)

        full_ax.set_title(f"KL vs rho with bracket bounds (d={latent_dim}, width={format_float(kl_width, precision=3)})")
        full_ax.set_xlabel("rho")
        full_ax.set_ylabel("KL")
        full_ax.grid(True, which="both", alpha=0.3)

        zoom_ax.set_title(f"Zoomed bracket view (d={latent_dim}, width={format_float(kl_width, precision=3)})")
        zoom_ax.set_xlabel("rho")
        zoom_ax.set_ylabel("KL")
        zoom_ax.set_ylim(band_min - band_pad, band_max + band_pad)
        zoom_ax.grid(True, which="both", alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5))
    fig.subplots_adjust(top=0.94, hspace=0.45, wspace=0.28)
    bounds_path = layout.figures_dir / "spcauchy_kl_vs_rho_with_bounds.png"
    fig.savefig(bounds_path, dpi=200)
    plt.close(fig)
    figure_paths.append(str(bounds_path))

    fig, axes = plt.subplots(len(representative_dims), 1, figsize=(8.5, 3.0 * len(representative_dims)), squeeze=False)
    for row_idx, latent_dim in enumerate(representative_dims):
        ax = axes[row_idx, 0]
        subset = [record for record in records if record["dim"] == latent_dim and record["success"]]
        if not subset:
            continue

        rho_values = sorted({record["rho"] for record in subset})
        lower_kl, upper_kl, midpoint_kl, laplace_kl, kl_width = _compute_bound_kl_curves(rho_values, latent_dim)
        combined_kl = _compute_combined_curve(rho_values, latent_dim)

        ax.axhspan(0.0, 1.0, color="#cfe8ff", alpha=0.45, label="bracket")
        ax.axhline(0.0, color="#6c757d", linewidth=1.0)
        ax.axhline(1.0, color="#6c757d", linewidth=1.0)
        ax.axhline(0.5, color="#adb5bd", linewidth=1.0, linestyle=":")

        midpoint_pos = _normalize_within_bracket(midpoint_kl, lower_kl, upper_kl)
        laplace_pos = _normalize_within_bracket(laplace_kl, lower_kl, upper_kl)
        ax.plot(rho_values, midpoint_pos, linestyle="--", color="#1d3557", label="midpoint")

        plotted_methods = ["combined", "series", "asymptotic_high_rho"]
        for method_name in plotted_methods:
            if method_name == "combined":
                method_rho = rho_values
                method_kl = combined_kl
            else:
                method_rows = sorted(
                    [record for record in subset if record["method"] == method_name and record["kl_value"] is not None],
                    key=lambda item: item["rho"],
                )
                if not method_rows:
                    continue
                method_rho = [row["rho"] for row in method_rows]
                method_kl = [row["kl_value"] for row in method_rows]
            lower_lookup = {rho: value for rho, value in zip(rho_values, lower_kl)}
            upper_lookup = {rho: value for rho, value in zip(rho_values, upper_kl)}
            normalized_position = _normalize_within_bracket(
                method_kl,
                [lower_lookup[rho] for rho in method_rho],
                [upper_lookup[rho] for rho in method_rho],
            )
            ax.plot(
                method_rho,
                normalized_position,
                marker="o",
                linewidth=1.2,
                label=method_name,
            )

        ax.plot(rho_values, laplace_pos, linestyle="-.", color="#d62828", label="laplace")

        ax.set_title(f"Position within KL bracket (d={latent_dim}, width={format_float(kl_width, precision=3)})")
        ax.set_xlabel("rho")
        ax.set_ylabel("(f - L) / (U - L)")
        ax.set_ylim(-0.25, 1.25)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(["lower", "mid", "upper"])
        ax.grid(True, which="both", alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5))
    fig.subplots_adjust(top=0.94, hspace=0.45)
    position_path = layout.figures_dir / "spcauchy_kl_position_in_bracket.png"
    fig.savefig(position_path, dpi=200)
    plt.close(fig)
    figure_paths.append(str(position_path))

    fig, axes = plt.subplots(1, len(representative_rhos), figsize=(4.5 * len(representative_rhos), 4), squeeze=False)
    for col_idx, rho_value in enumerate(representative_rhos):
        ax = axes[0, col_idx]
        subset = [record for record in records if abs(record["rho"] - rho_value) < 1e-12 and record["success"]]
        for method_name in methods:
            method_rows = sorted(
                [record for record in subset if record["method"] == method_name],
                key=lambda item: item["dim"],
            )
            if not method_rows:
                continue
            dims = [row["dim"] for row in method_rows]
            times = [row["eval_time_s"] for row in method_rows]
            ax.loglog(dims, times, marker="o", label=method_name)
        ax.set_title(f"Evaluation time vs d (rho={rho_value})")
        ax.set_xlabel("latent dimension d")
        ax.set_ylabel("time [s]")
        ax.grid(True, which="both", alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.subplots_adjust(top=0.88, wspace=0.3)
    time_path = layout.figures_dir / "spcauchy_kl_eval_time_vs_dimension.png"
    fig.savefig(time_path, dpi=200)
    plt.close(fig)
    figure_paths.append(str(time_path))

    dims = sorted({record["dim"] for record in records})
    rhos = sorted({record["rho"] for record in records})
    preferred = np.full((len(rhos), len(dims)), np.nan)
    method_to_index = {name: idx for idx, name in enumerate(methods)}

    for row_idx, rho_value in enumerate(rhos):
        for col_idx, latent_dim in enumerate(dims):
            candidates = [
                record
                for record in records
                if record["dim"] == latent_dim and record["rho"] == rho_value and record["success"]
            ]
            if not candidates:
                continue
            if latent_dim in {2, 3, 4, 5}:
                hybrid_candidates = [record for record in candidates if record["method"] == "hybrid"]
                if hybrid_candidates:
                    preferred[row_idx, col_idx] = method_to_index["hybrid"]
                    continue
            candidates.sort(key=lambda item: (item["rel_error"] if item["rel_error"] is not None else float("inf"), item["eval_time_s"]))
            preferred[row_idx, col_idx] = method_to_index[candidates[0]["method"]]

    palette = ["#6c757d", "#1d3557", "#2a9d8f", "#d62828", "#f4a261"]
    cmap = ListedColormap(palette[: len(methods)])
    norm = BoundaryNorm(np.arange(-0.5, len(methods) + 0.5, 1.0), cmap.N)

    fig, ax = plt.subplots(figsize=(12, 4))
    image = ax.imshow(preferred, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims, rotation=45, ha="right")
    ax.set_yticks(range(len(rhos)))
    ax.set_yticklabels([format_float(rho, precision=3) for rho in rhos])
    ax.set_xlabel("latent dimension d")
    ax.set_ylabel("rho")
    ax.set_title("Preferred successful KL evaluator by regime")
    colorbar = fig.colorbar(image, ax=ax, ticks=list(method_to_index.values()), boundaries=np.arange(-0.5, len(methods) + 0.5, 1.0))
    colorbar.ax.set_yticklabels(list(method_to_index.keys()))
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.22, top=0.9)
    heatmap_path = layout.figures_dir / "spcauchy_kl_preferred_method_heatmap.png"
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)
    figure_paths.append(str(heatmap_path))

    return figure_paths
