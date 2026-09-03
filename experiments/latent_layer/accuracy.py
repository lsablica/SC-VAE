"""Accuracy sweep for the spherical Cauchy KL routes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .defaults import (
    DEFAULT_ACCURACY_DIMS,
    DEFAULT_SPCAUCHY_ACCURACY_METHODS,
    DEFAULT_SPCAUCHY_RHO_GRID,
)
from .methods import get_spcauchy_kl_method
from .utils import (
    OutputLayout,
    compute_abs_rel_error,
    dtype_name,
    prepare_output_layout,
    resolve_device,
    seed_all,
    write_csv,
)


@dataclass
class AccuracyConfig:
    dims: list[int] = field(default_factory=lambda: list(DEFAULT_ACCURACY_DIMS))
    rho_grid: list[float] = field(
        default_factory=lambda: list(DEFAULT_SPCAUCHY_RHO_GRID)
    )
    methods: list[str] = field(
        default_factory=lambda: list(DEFAULT_SPCAUCHY_ACCURACY_METHODS)
    )
    device_name: str = "cpu"
    dtype: torch.dtype = torch.float64
    seed: int = 0
    reference_nodes: int = 4096
    reference_k_terms: int | None = None
    reference_disagreement_tol: float = 1e-8
    out_dir: str | None = None


@dataclass(frozen=True)
class ReferenceResult:
    value: float | None
    source: str
    success: bool
    failure_type: str | None = None
    error_message: str | None = None


def compute_reference_value(
    rho_value: float,
    ambient_dim: int,
    _config: AccuracyConfig,
) -> ReferenceResult:
    """Evaluate the certified float64 vectorized direct recurrence."""

    try:
        rho = torch.tensor([[rho_value]], dtype=torch.float64)
        value = get_spcauchy_kl_method("direct").evaluator(
            rho, ambient_dim, direct_backend="vectorized"
        )
        return ReferenceResult(
            value=float(value.item()),
            source="direct_certified",
            success=True,
        )
    except Exception as exc:  # pragma: no cover - diagnostic record path
        return ReferenceResult(
            value=None,
            source="direct_certified",
            success=False,
            failure_type=type(exc).__name__,
            error_message=str(exc),
        )


def _evaluate(
    method_name: str,
    rho_value: float,
    ambient_dim: int,
    config: AccuracyConfig,
) -> tuple[float | None, float, str | None, str | None]:
    device = resolve_device(config.device_name)
    rho = torch.tensor(
        [[rho_value]], device=device, dtype=config.dtype
    )
    try:
        started = time.perf_counter()
        value = get_spcauchy_kl_method(method_name).evaluator(
            rho,
            ambient_dim,
            direct_backend="vectorized",
            fixed_maximum_concentration=max(config.rho_grid),
        )
        elapsed = time.perf_counter() - started
        return float(value.item()), elapsed, None, None
    except Exception as exc:  # pragma: no cover - diagnostic record path
        return None, 0.0, type(exc).__name__, str(exc)


def run_accuracy_benchmark(config: AccuracyConfig) -> list[dict]:
    seed_all(config.seed)
    records: list[dict] = []
    for ambient_dim in config.dims:
        for rho_value in config.rho_grid:
            reference = compute_reference_value(rho_value, ambient_dim, config)
            for method_name in config.methods:
                method = get_spcauchy_kl_method(method_name)
                value, elapsed, failure_type, error_message = _evaluate(
                    method_name, rho_value, ambient_dim, config
                )
                success = value is not None and reference.success
                absolute_error = relative_error = None
                if success:
                    absolute_error, relative_error = compute_abs_rel_error(
                        value, reference.value
                    )
                records.append(
                    {
                        "benchmark": "accuracy",
                        "method": method.name,
                        "family": method.family,
                        "device": resolve_device(config.device_name).type,
                        "dtype": dtype_name(config.dtype),
                        "seed": config.seed,
                        "dim": ambient_dim,
                        "rho": rho_value,
                        "success": success,
                        "failure_type": failure_type,
                        "error_message": error_message,
                        "kl_value": value,
                        "eval_time_s": elapsed,
                        "reference_value": reference.value,
                        "reference_source": reference.source,
                        "reference_success": reference.success,
                        "reference_failure_type": reference.failure_type,
                        "reference_error_message": reference.error_message,
                        "abs_error": absolute_error,
                        "rel_error": relative_error,
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
    figures = _plot_accuracy(records, layout) if generate_plots else []
    return layout, str(csv_path), figures


def _plot_accuracy(records: list[dict], layout: OutputLayout) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    successful = [row for row in records if row.get("success")]
    if not successful:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for method_name in sorted({row["method"] for row in successful}):
        rows = [row for row in successful if row["method"] == method_name]
        rows.sort(key=lambda row: (row["dim"], row["rho"]))
        axes[0].plot(
            range(len(rows)),
            [max(row["abs_error"], 1e-18) for row in rows],
            label=method_name,
        )
        axes[1].plot(
            range(len(rows)),
            [row["eval_time_s"] for row in rows],
            label=method_name,
        )
    axes[0].set_yscale("log")
    axes[0].set_title("Absolute error")
    axes[1].set_yscale("log")
    axes[1].set_title("Evaluation time")
    for axis in axes:
        axis.set_xlabel("Grid point")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    path = layout.figures_dir / "direct_kl_accuracy.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return [str(path)]
