"""Robustness sweep for latent-layer methods across concentration regimes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .accuracy import AccuracyConfig, compute_reference_value
from .defaults import (
    DEFAULT_ACCURACY_DIMS,
    DEFAULT_SPCAUCHY_RHO_GRID,
    DEFAULT_SPCAUCHY_ROBUSTNESS_METHODS,
    DEFAULT_VMF_KAPPA_GRID,
    DEFAULT_VMF_ROBUSTNESS_METHODS,
)
from .methods import get_latent_step_method, get_spcauchy_kl_method
from .runtime import RuntimeConfig, _benchmark_one_runtime_config
from .utils import (
    OutputLayout,
    compute_abs_rel_error,
    dtype_name,
    format_float,
    prepare_output_layout,
    resolve_device,
    seed_all,
    write_csv,
)


@dataclass
class RobustnessConfig:
    spcauchy_dims: list[int] = field(default_factory=lambda: list(DEFAULT_ACCURACY_DIMS))
    vmf_dims: list[int] = field(default_factory=lambda: [dim for dim in DEFAULT_ACCURACY_DIMS if dim >= 8])
    rho_grid: list[float] = field(default_factory=lambda: list(DEFAULT_SPCAUCHY_RHO_GRID))
    kappa_grid: list[float] = field(default_factory=lambda: list(DEFAULT_VMF_KAPPA_GRID))
    spcauchy_methods: list[str] = field(default_factory=lambda: list(DEFAULT_SPCAUCHY_ROBUSTNESS_METHODS))
    vmf_methods: list[str] = field(default_factory=lambda: list(DEFAULT_VMF_ROBUSTNESS_METHODS))
    power_methods: list[str] = field(
        default_factory=lambda: ["power_spherical"]
    )
    device_name: str = "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 0
    batch_size: int = 32
    timeout_s: float = 5.0
    reference_nodes: int = 4096
    reference_disagreement_tol: float = 1e-8
    kl_error_rel_threshold: float = 1e-4
    out_dir: str | None = None


def _spcauchy_runtime_config(config: RobustnessConfig) -> RuntimeConfig:
    return RuntimeConfig(
        dims=[],
        spcauchy_methods=[],
        vmf_methods=[],
        power_methods=[],
        device_name=config.device_name,
        dtype=config.dtype,
        seed=config.seed,
        batch_size=config.batch_size,
        # One untimed setup iteration excludes one-time torch.compile or
        # Triton compilation from the stability classification.
        warmup_iters=1,
        measure_iters=1,
        repeats=1,
        timeout_s=config.timeout_s,
    )


def run_robustness_benchmark(config: RobustnessConfig) -> list[dict]:
    seed_all(config.seed)
    device = resolve_device(config.device_name)
    records: list[dict] = []
    runtime_config = _spcauchy_runtime_config(config)
    reference_config = AccuracyConfig(
        dims=[],
        rho_grid=[],
        methods=[],
        device_name="cpu",
        dtype=torch.float64,
        seed=config.seed,
        reference_nodes=config.reference_nodes,
        reference_disagreement_tol=config.reference_disagreement_tol,
    )

    for latent_dim in config.spcauchy_dims:
        for rho_value in config.rho_grid:
            reference = compute_reference_value(rho_value, latent_dim, reference_config)
            for method_name in config.spcauchy_methods:
                method = get_spcauchy_kl_method(method_name)
                record = _benchmark_one_runtime_config(
                    f"spcauchy_{method_name}",
                    latent_dim=latent_dim,
                    rho_value=rho_value,
                    kappa_value=None,
                    config=runtime_config,
                    device=device,
                )

                evaluator_record = {
                    "benchmark": "robustness",
                    "method": method.name,
                    "family": method.family,
                    "device": device.type,
                    "dtype": dtype_name(config.dtype),
                    "seed": config.seed,
                    "dim": latent_dim,
                    "rho": rho_value,
                    "kappa": None,
                    "batch_size": config.batch_size,
                    "direct_backend_requested": record[
                        "direct_backend_requested"
                    ],
                    "direct_backend_resolved": record[
                        "direct_backend_resolved"
                    ],
                    "retained_correction_terms": record[
                        "retained_correction_terms"
                    ],
                    "warmup_iters": record["warmup_iters"],
                    "measure_iters": record["measure_iters"],
                    "repeats": record["repeats"],
                    "timeout_s": record["timeout_s"],
                    "success": record["success"],
                    "failure_type": record["failure_type"],
                    "error_message": record["error_message"],
                    "nan_or_inf_loss": record["nan_or_inf_loss"],
                    "nan_or_inf_grad": record["nan_or_inf_grad"],
                    "runtime_error": record["runtime_error"],
                    "timeout_hit": record["timeout_hit"],
                    "kl_error_threshold_exceeded": None,
                    "reference_source": reference.source,
                }

                if record["success"] and reference.success:
                    rho_tensor = torch.tensor([[rho_value]], dtype=config.dtype, device=device)
                    kl_value = method.evaluator(
                        rho_tensor,
                        latent_dim,
                        # The production backend was already exercised by the
                        # full forward/backward step above. Use the independent
                        # vectorized path for the untimed error measurement so
                        # mixed requires-grad signatures do not consume
                        # compiler specializations.
                        direct_backend="vectorized",
                    )
                    abs_error, rel_error = compute_abs_rel_error(float(kl_value.item()), reference.value)
                    evaluator_record["abs_error"] = abs_error
                    evaluator_record["rel_error"] = rel_error
                    evaluator_record["kl_error_threshold_exceeded"] = bool(
                        rel_error is not None and rel_error > config.kl_error_rel_threshold
                    )
                records.append(evaluator_record)

    for latent_dim in config.vmf_dims:
        for kappa_value in config.kappa_grid:
            for method_name in config.vmf_methods + config.power_methods:
                record = _benchmark_one_runtime_config(
                    method_name,
                    latent_dim=latent_dim,
                    rho_value=None,
                    kappa_value=kappa_value,
                    config=runtime_config,
                    device=device,
                )
                record.update(
                    {
                        "benchmark": "robustness",
                        "method": method_name,
                        "family": get_latent_step_method(method_name).family,
                        "device": device.type,
                        "dtype": dtype_name(config.dtype),
                        "seed": config.seed,
                        "dim": latent_dim,
                        "rho": None,
                        "kappa": kappa_value,
                        "batch_size": config.batch_size,
                        "kl_error_threshold_exceeded": None,
                        "reference_source": None,
                    }
                )
                records.append(record)

    return records


def save_robustness_outputs(
    records: list[dict],
    *,
    out_dir: str | None = None,
    csv_name: str = "robustness_grid.csv",
    generate_plots: bool = True,
) -> tuple[OutputLayout, str, list[str]]:
    layout = prepare_output_layout(out_dir)
    csv_path = write_csv(records, layout.results_dir / csv_name)
    figures = plot_robustness_results(records, layout) if generate_plots else []
    return layout, str(csv_path), figures


def _heatmap_matrix(records: list[dict], *, methods: list[str], dims: list[int], values: list[float], value_key: str) -> tuple[np.ndarray, list[str]]:
    matrix = np.full((len(methods), len(values), len(dims)), np.nan)
    labels = []
    for method_idx, method_name in enumerate(methods):
        labels.append(method_name)
        for value_idx, value in enumerate(values):
            for dim_idx, latent_dim in enumerate(dims):
                matches = [
                    record
                    for record in records
                    if record["method"] == method_name and record["dim"] == latent_dim and record[value_key] == value
                ]
                if not matches:
                    continue
                matrix[method_idx, value_idx, dim_idx] = 1.0 if matches[0]["success"] else 0.0
    return matrix, labels


def plot_robustness_results(records: list[dict], layout: OutputLayout) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    figure_paths: list[str] = []
    if not records:
        return figure_paths

    stability_cmap = ListedColormap(["#d1495b", "#2b9348"])

    sp_records = [record for record in records if record["family"] == "spcauchy"]
    if sp_records:
        available = {record["method"] for record in sp_records}
        methods = [
            method
            for method in ["direct", "neighbor", "laplace"]
            if method in available
        ]
        dims = sorted({record["dim"] for record in sp_records})
        rho_values = sorted({record["rho"] for record in sp_records if record["rho"] is not None})
        matrix, labels = _heatmap_matrix(sp_records, methods=methods, dims=dims, values=rho_values, value_key="rho")

        fig, axes = plt.subplots(
            1,
            len(methods),
            figsize=(3.7 * len(methods), 4),
            squeeze=False,
        )
        for ax, method_idx, method_name in zip(axes[0], range(len(methods)), labels):
            ax.imshow(matrix[method_idx], aspect="auto", vmin=0.0, vmax=1.0, cmap=stability_cmap)
            ax.set_title(method_name)
            ax.set_xticks(range(len(dims)))
            ax.set_xticklabels(dims, rotation=45, ha="right")
            ax.set_yticks(range(len(rho_values)))
            ax.set_yticklabels([format_float(rho, precision=3) for rho in rho_values])
            ax.set_xlabel("latent dimension d")
            ax.set_ylabel("rho")
        fig.suptitle("Robustness by regime: green = pass, red = fail", fontsize=12)
        fig.tight_layout(rect=[0, 0.0, 1, 0.93])
        path = layout.figures_dir / "robustness_spcauchy_heatmap.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        figure_paths.append(str(path))

    vmf_records = [
        record
        for record in records
        if record["family"] in {"vmf", "powerspherical"}
    ]
    if vmf_records:
        available = {record["method"] for record in vmf_records}
        methods = [
            method
            for method in [
                "vmf_official",
                "vmf_robust",
                "power_spherical",
            ]
            if method in available
        ]
        dims = sorted({record["dim"] for record in vmf_records})
        kappa_values = sorted({record["kappa"] for record in vmf_records if record["kappa"] is not None})
        matrix, labels = _heatmap_matrix(vmf_records, methods=methods, dims=dims, values=kappa_values, value_key="kappa")

        fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4), squeeze=False)
        for ax, method_idx, method_name in zip(axes[0], range(len(methods)), labels):
            ax.imshow(matrix[method_idx], aspect="auto", vmin=0.0, vmax=1.0, cmap=stability_cmap)
            ax.set_title(method_name)
            ax.set_xticks(range(len(dims)))
            ax.set_xticklabels(dims, rotation=45, ha="right")
            ax.set_yticks(range(len(kappa_values)))
            ax.set_yticklabels([format_float(kappa, precision=3) for kappa in kappa_values])
            ax.set_xlabel("latent dimension d")
            ax.set_ylabel("kappa")
        fig.suptitle("Robustness by regime: green = pass, red = fail", fontsize=12)
        fig.tight_layout(rect=[0, 0.0, 1, 0.93])
        path = layout.figures_dir / "robustness_vmf_heatmap.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        figure_paths.append(str(path))

    return figure_paths
