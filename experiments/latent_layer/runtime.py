"""Full latent-step runtime benchmark."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from spherical_cauchy import SphericalCauchy
from spherical_cauchy.direct import direct_kl_diagnostics
from spherical_cauchy.laplace import resolve_laplace_backend
from spherical_cauchy.triton_backend import triton_is_available

from .defaults import (
    DEFAULT_NEIGHBOR_RUNTIME_DIMS,
    DEFAULT_POWER_RUNTIME_METHODS,
    DEFAULT_RUNTIME_DIMS,
    DEFAULT_SPCAUCHY_RUNTIME_METHODS,
    DEFAULT_VMF_RUNTIME_METHODS,
)
from .methods import (
    build_power_spherical_distribution,
    build_vmf_distribution,
    get_latent_step_method,
    kl_for_spcauchy_runtime,
)
from .utils import (
    OutputLayout,
    dtype_name,
    maybe_sync,
    prepare_output_layout,
    resolve_device,
    rho_from_kappa_dim,
    seed_all,
    summarize_timings,
    write_csv,
)

RUNTIME_LABELS = {
    "power_spherical": "Power Spherical",
    "spcauchy_direct": "SC direct",
    "spcauchy_direct_autograd": "SC termwise autograd reference",
    "spcauchy_direct_fixed": "SC fixed certified",
    "spcauchy_laplace": "SC Laplace",
    "spcauchy_neighbor": "SC even-neighbor",
    "vmf_official": "vMF original",
    "vmf_robust": "vMF robust",
}


@dataclass
class RuntimeConfig:
    dims: list[int] = field(default_factory=lambda: list(DEFAULT_RUNTIME_DIMS))
    neighbor_dims: list[int] = field(
        default_factory=lambda: list(DEFAULT_NEIGHBOR_RUNTIME_DIMS)
    )
    spcauchy_methods: list[str] = field(default_factory=lambda: list(DEFAULT_SPCAUCHY_RUNTIME_METHODS))
    vmf_methods: list[str] = field(default_factory=lambda: list(DEFAULT_VMF_RUNTIME_METHODS))
    power_methods: list[str] = field(
        default_factory=lambda: list(DEFAULT_POWER_RUNTIME_METHODS)
    )
    device_name: str = "auto"
    dtype: torch.dtype = torch.float32
    seed: int = 0
    batch_size: int | None = None
    warmup_iters: int = 10
    measure_iters: int = 50
    repeats: int = 5
    timeout_s: float = 5.0
    direct_backend: str = "auto"
    laplace_backend: str = "auto"
    fixed_maximum_concentration: float | None = None
    fixed_value_tolerance: float = 2e-6
    fixed_gradient_tolerance: float = 2e-6
    kappa: float = 10.0
    concentration_mode: str = "matched"
    rho_values: list[float] = field(default_factory=lambda: [0.9])
    out_dir: str | None = None


def _default_batch_size(device: torch.device) -> int:
    return 1024 if device.type == "cuda" else 128


def _gradient_has_nonfinite(*tensors: torch.Tensor | None) -> bool:
    for tensor in tensors:
        if tensor is None:
            continue
        grad = tensor.grad
        if grad is not None and (torch.isnan(grad).any() or torch.isinf(grad).any()):
            return True
    return False


def _run_spcauchy_iteration(
    method_name: str,
    *,
    batch_size: int,
    latent_dim: int,
    rho_value: float,
    device: torch.device,
    dtype: torch.dtype,
    direct_backend: str,
    laplace_backend: str,
    fixed_maximum_concentration: float | None,
    fixed_value_tolerance: float,
    fixed_gradient_tolerance: float,
) -> dict:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    maybe_sync(device)
    t0 = time.perf_counter()
    raw_mu = torch.randn(batch_size, latent_dim, device=device, dtype=dtype, requires_grad=True)
    mu = F.normalize(raw_mu, dim=1)
    rho = torch.full((batch_size, 1), rho_value, device=device, dtype=dtype, requires_grad=True)
    maybe_sync(device)
    t_parameters = time.perf_counter()
    z = SphericalCauchy(mu, rho).rsample()
    maybe_sync(device)
    t_sample = time.perf_counter()
    kl = kl_for_spcauchy_runtime(
        method_name,
        rho,
        latent_dim,
        direct_backend=direct_backend,
        laplace_backend=laplace_backend,
        fixed_maximum_concentration=fixed_maximum_concentration,
        fixed_value_tolerance=fixed_value_tolerance,
        fixed_gradient_tolerance=fixed_gradient_tolerance,
    )
    maybe_sync(device)
    t_kl = time.perf_counter()
    loss = z.sum() + kl.sum()
    maybe_sync(device)
    t1 = time.perf_counter()

    loss.backward()
    maybe_sync(device)
    t2 = time.perf_counter()
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )

    nan_or_inf_loss = bool(torch.isnan(loss) or torch.isinf(loss))
    nan_or_inf_grad = _gradient_has_nonfinite(raw_mu, rho)
    return {
        "forward_time_s": t1 - t0,
        "backward_time_s": t2 - t1,
        "total_time_s": t2 - t0,
        "parameter_time_s": t_parameters - t0,
        "sampling_time_s": t_sample - t_parameters,
        "kl_time_s": t_kl - t_sample,
        "loss_construction_time_s": t1 - t_kl,
        "peak_memory_bytes": peak_memory,
        "nan_or_inf_loss": nan_or_inf_loss,
        "nan_or_inf_grad": nan_or_inf_grad,
    }


def _run_vmf_iteration(
    method_name: str,
    *,
    batch_size: int,
    latent_dim: int,
    kappa_value: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    maybe_sync(device)
    t0 = time.perf_counter()
    raw_loc = torch.randn(batch_size, latent_dim, device=device, dtype=dtype, requires_grad=True)
    loc = F.normalize(raw_loc, dim=1)
    kappa = torch.full((batch_size, 1), kappa_value, device=device, dtype=dtype, requires_grad=True)

    maybe_sync(device)
    t_parameters = time.perf_counter()
    if abs(kappa_value) <= 1e-12:
        z = loc
    else:
        vmf, hyu, kl_fn = build_vmf_distribution(method_name, loc, kappa)
        z = vmf.rsample()
    maybe_sync(device)
    t_sample = time.perf_counter()
    if abs(kappa_value) <= 1e-12:
        kl = torch.zeros(batch_size, device=device, dtype=dtype)
    else:
        kl = kl_fn(vmf, hyu)
    maybe_sync(device)
    t_kl = time.perf_counter()
    loss = z.sum() + kl.sum()
    maybe_sync(device)
    t1 = time.perf_counter()

    loss.backward()
    maybe_sync(device)
    t2 = time.perf_counter()
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )

    nan_or_inf_loss = bool(torch.isnan(loss) or torch.isinf(loss))
    nan_or_inf_grad = _gradient_has_nonfinite(raw_loc, kappa)
    return {
        "forward_time_s": t1 - t0,
        "backward_time_s": t2 - t1,
        "total_time_s": t2 - t0,
        "parameter_time_s": t_parameters - t0,
        "sampling_time_s": t_sample - t_parameters,
        "kl_time_s": t_kl - t_sample,
        "loss_construction_time_s": t1 - t_kl,
        "peak_memory_bytes": peak_memory,
        "nan_or_inf_loss": nan_or_inf_loss,
        "nan_or_inf_grad": nan_or_inf_grad,
    }


def _run_power_spherical_iteration(
    *,
    batch_size: int,
    latent_dim: int,
    exponent_value: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    maybe_sync(device)
    t0 = time.perf_counter()
    raw_loc = torch.randn(
        batch_size,
        latent_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    loc = F.normalize(raw_loc, dim=1)
    exponent = torch.full(
        (batch_size, 1),
        exponent_value,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    maybe_sync(device)
    t_parameters = time.perf_counter()
    distribution, prior = build_power_spherical_distribution(loc, exponent)
    z = distribution.rsample()
    maybe_sync(device)
    t_sample = time.perf_counter()
    kl = torch.distributions.kl_divergence(distribution, prior)
    maybe_sync(device)
    t_kl = time.perf_counter()
    loss = z.sum() + kl.sum()
    maybe_sync(device)
    t1 = time.perf_counter()
    loss.backward()
    maybe_sync(device)
    t2 = time.perf_counter()
    return {
        "forward_time_s": t1 - t0,
        "backward_time_s": t2 - t1,
        "total_time_s": t2 - t0,
        "parameter_time_s": t_parameters - t0,
        "sampling_time_s": t_sample - t_parameters,
        "kl_time_s": t_kl - t_sample,
        "loss_construction_time_s": t1 - t_kl,
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "nan_or_inf_loss": bool(torch.isnan(loss) or torch.isinf(loss)),
        "nan_or_inf_grad": _gradient_has_nonfinite(raw_loc, exponent),
    }


def _benchmark_one_runtime_config(
    method_name: str,
    *,
    latent_dim: int,
    rho_value: float | None,
    kappa_value: float | None,
    config: RuntimeConfig,
    device: torch.device,
) -> dict:
    parameter_times: list[float] = []
    sampling_times: list[float] = []
    kl_times: list[float] = []
    loss_construction_times: list[float] = []
    forward_times: list[float] = []
    backward_times: list[float] = []
    total_times: list[float] = []
    peak_memories: list[int] = []
    repeat_total_medians: list[float] = []
    nan_or_inf_loss = False
    nan_or_inf_grad = False
    success = True
    failure_type = None
    error_message = None
    timeout_hit = False

    batch_size = config.batch_size or _default_batch_size(device)
    total_iters = config.warmup_iters + config.measure_iters
    family = get_latent_step_method(method_name).family

    for repeat in range(config.repeats):
        seed_all(config.seed + repeat)
        current_repeat_totals: list[float] = []
        for step_idx in range(total_iters):
            try:
                if family == "spcauchy":
                    result = _run_spcauchy_iteration(
                        method_name,
                        batch_size=batch_size,
                        latent_dim=latent_dim,
                        rho_value=float(rho_value),
                        device=device,
                        dtype=config.dtype,
                        direct_backend=config.direct_backend,
                        laplace_backend=config.laplace_backend,
                        fixed_maximum_concentration=(
                            config.fixed_maximum_concentration
                            if config.fixed_maximum_concentration is not None
                            else float(rho_value)
                        ),
                        fixed_value_tolerance=config.fixed_value_tolerance,
                        fixed_gradient_tolerance=config.fixed_gradient_tolerance,
                    )
                elif family == "powerspherical":
                    result = _run_power_spherical_iteration(
                        batch_size=batch_size,
                        latent_dim=latent_dim,
                        exponent_value=2.0 * float(kappa_value),
                        device=device,
                        dtype=config.dtype,
                    )
                else:
                    result = _run_vmf_iteration(
                        method_name,
                        batch_size=batch_size,
                        latent_dim=latent_dim,
                        kappa_value=float(kappa_value),
                        device=device,
                        dtype=config.dtype,
                    )
            except RuntimeError as exc:
                success = False
                failure_type = "runtime_error"
                error_message = str(exc)
                break
            except Exception as exc:  # pragma: no cover - defensive
                success = False
                failure_type = "exception"
                error_message = str(exc)
                break

            nan_or_inf_loss = nan_or_inf_loss or result["nan_or_inf_loss"]
            nan_or_inf_grad = nan_or_inf_grad or result["nan_or_inf_grad"]
            if result["nan_or_inf_loss"] or result["nan_or_inf_grad"]:
                success = False
                failure_type = "non_finite"
                error_message = "Encountered NaN/Inf in loss or gradients."
                break

            if (
                step_idx >= config.warmup_iters
                and result["total_time_s"] > config.timeout_s
            ):
                success = False
                timeout_hit = True
                failure_type = "timeout"
                error_message = (
                    f"Iteration exceeded timeout {config.timeout_s:.3f}s."
                )
                break

            if step_idx >= config.warmup_iters:
                parameter_times.append(result["parameter_time_s"])
                sampling_times.append(result["sampling_time_s"])
                kl_times.append(result["kl_time_s"])
                loss_construction_times.append(
                    result["loss_construction_time_s"]
                )
                forward_times.append(result["forward_time_s"])
                backward_times.append(result["backward_time_s"])
                total_times.append(result["total_time_s"])
                current_repeat_totals.append(result["total_time_s"])
                if result["peak_memory_bytes"] is not None:
                    peak_memories.append(result["peak_memory_bytes"])
        if current_repeat_totals:
            repeat_total_medians.append(
                float(torch.tensor(current_repeat_totals).median())
            )
        if not success:
            break

    parameter_summary = summarize_timings(parameter_times)
    sampling_summary = summarize_timings(sampling_times)
    kl_summary = summarize_timings(kl_times)
    loss_construction_summary = summarize_timings(loss_construction_times)
    forward_summary = summarize_timings(forward_times)
    backward_summary = summarize_timings(backward_times)
    total_summary = summarize_timings(total_times)
    method = get_latent_step_method(method_name)
    retained_terms = None
    if method_name in {
        "spcauchy_direct",
        "spcauchy_direct_autograd",
    }:
        retained_terms = direct_kl_diagnostics(
            torch.zeros((), device=device, dtype=config.dtype),
            latent_dim,
            backend=config.direct_backend,
        ).retained_terms
    elif method_name == "spcauchy_direct_fixed":
        retained_terms = direct_kl_diagnostics(
            torch.zeros((), device=device, dtype=config.dtype),
            latent_dim,
            maximum_concentration=float(rho_value),
            value_tolerance=config.fixed_value_tolerance,
            gradient_tolerance=config.fixed_gradient_tolerance,
            backend=config.direct_backend,
        ).retained_terms
    elif method_name == "spcauchy_neighbor":
        retained_terms = (latent_dim - 1) // 2

    resolved_direct_backend = None
    if method_name == "spcauchy_direct_autograd":
        resolved_direct_backend = "horner_reference"
    elif method_name in {
        "spcauchy_direct",
        "spcauchy_direct_fixed",
        "spcauchy_neighbor",
    }:
        if config.direct_backend != "auto":
            resolved_direct_backend = config.direct_backend
        elif device.type == "cuda" and triton_is_available():
            resolved_direct_backend = "triton"
        elif (
            device.type == "cpu"
            and retained_terms is not None
            and retained_terms >= 32
        ):
            resolved_direct_backend = (
                "compiled_horner_neighbor"
                if method_name == "spcauchy_neighbor"
                else "compiled_horner_direct"
            )
        else:
            resolved_direct_backend = "vectorized"

    resolved_laplace_backend = None
    if method_name == "spcauchy_laplace":
        resolved_laplace_backend = resolve_laplace_backend(
            torch.zeros((), device=device, dtype=config.dtype),
            config.laplace_backend,
        )

    return {
        "benchmark": "runtime",
        "method": method.name,
        "family": method.family,
        "device": device.type,
        "dtype": dtype_name(config.dtype),
        "seed": config.seed,
        "dim": latent_dim,
        "rho": rho_value,
        "kappa": kappa_value,
        "power_exponent": (
            2.0 * float(kappa_value)
            if method.family == "powerspherical"
            and kappa_value is not None
            else None
        ),
        "batch_size": batch_size,
        "direct_backend_requested": config.direct_backend,
        "direct_backend_resolved": resolved_direct_backend,
        "laplace_backend_requested": config.laplace_backend,
        "laplace_backend_resolved": resolved_laplace_backend,
        "retained_correction_terms": retained_terms,
        "success": success,
        "failure_type": failure_type,
        "error_message": error_message,
        "warmup_iters": config.warmup_iters,
        "measure_iters": config.measure_iters,
        "repeats": config.repeats,
        "timeout_s": config.timeout_s,
        "nan_or_inf_loss": nan_or_inf_loss,
        "nan_or_inf_grad": nan_or_inf_grad,
        "runtime_error": failure_type in {"runtime_error", "exception"},
        "timeout_hit": timeout_hit,
        "parameter_mean_s": parameter_summary["mean"],
        "parameter_std_s": parameter_summary["std"],
        "parameter_median_s": parameter_summary["median"],
        "parameter_iqr_s": parameter_summary["iqr"],
        "sampling_mean_s": sampling_summary["mean"],
        "sampling_std_s": sampling_summary["std"],
        "sampling_median_s": sampling_summary["median"],
        "sampling_iqr_s": sampling_summary["iqr"],
        "kl_mean_s": kl_summary["mean"],
        "kl_std_s": kl_summary["std"],
        "kl_median_s": kl_summary["median"],
        "kl_iqr_s": kl_summary["iqr"],
        "loss_construction_mean_s": loss_construction_summary["mean"],
        "loss_construction_std_s": loss_construction_summary["std"],
        "loss_construction_median_s": loss_construction_summary["median"],
        "loss_construction_iqr_s": loss_construction_summary["iqr"],
        "peak_cuda_memory_bytes": max(peak_memories) if peak_memories else None,
        "kl_autograd_route": (
            "reference_term_graph"
            if method_name == "spcauchy_direct_autograd"
            else "custom_single_node"
            if method_name
            in {
                "spcauchy_direct",
                "spcauchy_direct_fixed",
                "spcauchy_neighbor",
                "spcauchy_laplace",
            }
            else "native"
        ),
        "repeat_total_medians_s": ";".join(
            f"{value:.12g}" for value in repeat_total_medians
        ),
        "forward_mean_s": forward_summary["mean"],
        "forward_std_s": forward_summary["std"],
        "forward_median_s": forward_summary["median"],
        "forward_iqr_s": forward_summary["iqr"],
        "forward_min_s": forward_summary["min"],
        "forward_max_s": forward_summary["max"],
        "backward_mean_s": backward_summary["mean"],
        "backward_std_s": backward_summary["std"],
        "backward_median_s": backward_summary["median"],
        "backward_iqr_s": backward_summary["iqr"],
        "backward_min_s": backward_summary["min"],
        "backward_max_s": backward_summary["max"],
        "total_mean_s": total_summary["mean"],
        "total_std_s": total_summary["std"],
        "total_median_s": total_summary["median"],
        "total_iqr_s": total_summary["iqr"],
        "total_min_s": total_summary["min"],
        "total_max_s": total_summary["max"],
    }


def run_runtime_benchmark(config: RuntimeConfig) -> list[dict]:
    seed_all(config.seed)
    device = resolve_device(config.device_name)
    records: list[dict] = []

    common_spcauchy_methods = [
        method
        for method in config.spcauchy_methods
        if method != "spcauchy_neighbor"
    ]
    run_neighbor = "spcauchy_neighbor" in config.spcauchy_methods

    for latent_dim in config.dims:
        if config.concentration_mode == "matched":
            rho_values = [rho_from_kappa_dim(config.kappa, latent_dim)]
            kappa_values = [config.kappa]
        else:
            rho_values = list(config.rho_values)
            kappa_values = [None]

        for rho_value in rho_values:
            for method_name in common_spcauchy_methods:
                records.append(
                    _benchmark_one_runtime_config(
                        method_name,
                        latent_dim=latent_dim,
                        rho_value=rho_value,
                        kappa_value=config.kappa if config.concentration_mode == "matched" else None,
                        config=config,
                        device=device,
                    )
                )

        if config.concentration_mode == "matched":
            for kappa_value in kappa_values:
                for method_name in config.vmf_methods + config.power_methods:
                    records.append(
                        _benchmark_one_runtime_config(
                            method_name,
                            latent_dim=latent_dim,
                            rho_value=None,
                            kappa_value=kappa_value,
                            config=config,
                            device=device,
                        )
                    )

    if run_neighbor:
        for latent_dim in config.neighbor_dims:
            rho_values = (
                [rho_from_kappa_dim(config.kappa, latent_dim)]
                if config.concentration_mode == "matched"
                else list(config.rho_values)
            )
            for rho_value in rho_values:
                records.append(
                    _benchmark_one_runtime_config(
                        "spcauchy_neighbor",
                        latent_dim=latent_dim,
                        rho_value=rho_value,
                        kappa_value=(
                            config.kappa
                            if config.concentration_mode == "matched"
                            else None
                        ),
                        config=config,
                        device=device,
                    )
                )
    return records


def save_runtime_outputs(
    records: list[dict],
    *,
    out_dir: str | None = None,
    csv_name: str = "latent_step_runtime.csv",
    generate_plots: bool = True,
) -> tuple[OutputLayout, str, list[str]]:
    layout = prepare_output_layout(out_dir)
    csv_path = write_csv(records, layout.results_dir / csv_name)
    write_csv(
        summarize_runtime_records(records),
        layout.results_dir / "latent_step_runtime_summary.csv",
    )
    figures = plot_runtime_results(records, layout) if generate_plots else []
    return layout, str(csv_path), figures


def summarize_runtime_records(records: list[dict]) -> list[dict]:
    """Reduce the full timing grid to the compact paper verification table."""

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["device"], record["method"])].append(record)

    summary = []
    for (device, method), rows in sorted(grouped.items()):
        successful = [
            row
            for row in rows
            if (
                row["success"]
                if isinstance(row["success"], bool)
                else str(row["success"]).lower() == "true"
            )
            and row["total_median_s"] not in {None, ""}
        ]
        nearest = (
            min(
                successful,
                key=lambda row: (abs(int(row["dim"]) - 128), int(row["dim"])),
            )
            if successful
            else None
        )
        total_times = [float(row["total_median_s"]) for row in successful]
        peak_memories = [
            int(row["peak_cuda_memory_bytes"])
            for row in rows
            if row["peak_cuda_memory_bytes"] not in {None, ""}
        ]
        failure_types = sorted(
            {str(row["failure_type"]) for row in rows if row["failure_type"]}
        )
        summary.append(
            {
                "device": device,
                "evaluations": len(rows),
                "failure_types": ";".join(failure_types),
                "failures": len(rows) - len(successful),
                "geomean_total_median_s": (
                    math.exp(
                        sum(math.log(value) for value in total_times) / len(total_times)
                    )
                    if total_times
                    else None
                ),
                "label": RUNTIME_LABELS[method],
                "maximum_peak_cuda_memory_bytes": (
                    max(peak_memories) if peak_memories else None
                ),
                "method": method,
                "near_128_dimension": nearest["dim"] if nearest else None,
                "near_128_total_median_s": (
                    nearest["total_median_s"] if nearest else None
                ),
                "successes": len(successful),
            }
        )
    return summary


def plot_runtime_results(records: list[dict], layout: OutputLayout) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    figure_paths: list[str] = []
    if not records:
        return figure_paths

    devices = sorted({record["device"] for record in records})
    method_names = sorted({record["method"] for record in records})
    ncols = max(1, len(devices))
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4), squeeze=False)

    for ax, device_name in zip(axes[0], devices):
        device_records = [record for record in records if record["device"] == device_name and record["success"]]
        for method_name in method_names:
            method_records = sorted(
                [record for record in device_records if record["method"] == method_name],
                key=lambda item: (item["dim"], item["rho"] if item["rho"] is not None else -1.0),
            )
            if not method_records:
                continue
            dims = [record["dim"] for record in method_records]
            total_median = [
                record["total_median_s"] for record in method_records
            ]
            ax.loglog(dims, total_median, marker="o", label=method_name)
        ax.set_title(f"Latent-step total time ({device_name})")
        ax.set_xlabel("ambient dimension D")
        ax.set_ylabel("time [s]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    total_path = layout.figures_dir / "latent_step_runtime_total.png"
    fig.savefig(total_path, dpi=220)
    plt.close(fig)
    figure_paths.append(str(total_path))

    fig, axes = plt.subplots(
        2, ncols, figsize=(6 * ncols, 7.5), squeeze=False
    )
    for column, device_name in enumerate(devices):
        forward_ax = axes[0, column]
        backward_ax = axes[1, column]
        device_records = [record for record in records if record["device"] == device_name and record["success"]]
        for method_name in method_names:
            if method_name == "spcauchy_direct_autograd":
                continue
            method_records = sorted(
                [record for record in device_records if record["method"] == method_name],
                key=lambda item: (item["dim"], item["rho"] if item["rho"] is not None else -1.0),
            )
            if not method_records:
                continue
            dims = [record["dim"] for record in method_records]
            forward_median = [
                record["forward_median_s"] for record in method_records
            ]
            backward_median = [
                record["backward_median_s"] for record in method_records
            ]
            forward_ax.loglog(
                dims,
                forward_median,
                marker="o",
                label=method_name,
            )
            backward_ax.loglog(
                dims,
                backward_median,
                marker="s",
                label=method_name,
            )
        forward_ax.set_title(f"Forward ({device_name.upper()})")
        backward_ax.set_title(f"Backward ({device_name.upper()})")
        for ax in [forward_ax, backward_ax]:
            ax.set_xlabel("ambient dimension D")
            ax.set_ylabel("median time [s]")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=6, ncol=2)

    fig.tight_layout()
    split_path = layout.figures_dir / "direct_kl_forward_backward.png"
    fig.savefig(split_path, dpi=220)
    fig.savefig(
        layout.figures_dir / "latent_step_runtime_forward_backward.png",
        dpi=220,
    )
    plt.close(fig)
    figure_paths.append(str(split_path))

    for device_name in devices:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        _plot_primary_runtime_device(ax, records, device_name)
        fig.tight_layout()
        device_path = (
            layout.figures_dir / f"direct_kl_runtime_{device_name}.png"
        )
        fig.savefig(device_path, dpi=220)
        plt.close(fig)
        figure_paths.append(str(device_path))

    classic_paths = _plot_classic_runtime_figure(records, layout, plt, mticker)
    figure_paths.extend(classic_paths)

    return figure_paths


def _plot_classic_runtime_figure(records: list[dict], layout: OutputLayout, plt, mticker) -> list[str]:
    devices = sorted({record["device"] for record in records})
    if not devices:
        return []

    figure_paths: list[str] = []
    fig, axes = plt.subplots(1, len(devices), figsize=(6 * len(devices), 4), squeeze=False)

    for ax, device_name in zip(axes[0], devices):
        _plot_classic_runtime_device(ax, records, device_name, mticker)

    fig.suptitle("Hyperspherical latent-step runtime", fontsize=14)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    classic_path = layout.figures_dir / "benchmark_spcauchy_vs_vmf_stress.png"
    fig.savefig(classic_path, dpi=200)
    plt.close(fig)
    figure_paths.append(str(classic_path))
    return figure_paths


def _plot_classic_runtime_device(ax, records: list[dict], device_name: str, mticker) -> None:
    method_specs = [
        ("spcauchy_direct", "SC direct", "o"),
        ("spcauchy_neighbor", "SC neighbor (odd D)", "D"),
        ("spcauchy_laplace", "SC Laplace", "v"),
        ("vmf_official", "vMF original", "s"),
        ("vmf_robust", "vMF robust", "^"),
        ("power_spherical", "Power Spherical", "X"),
    ]
    device_records = [record for record in records if record["device"] == device_name]

    for method_name, label, marker in method_specs:
        success_records = sorted(
            [
                record
                for record in device_records
                if record["method"] == method_name and record["success"] and record["total_median_s"] is not None
            ],
            key=lambda item: item["dim"],
        )
        if not success_records:
            continue
        ax.plot(
            [record["dim"] for record in success_records],
            [record["total_median_s"] for record in success_records],
            f"-{marker}",
            label=label,
        )

    _annotate_first_failure(ax, device_records, "vmf_official", "red", "vMF official fails")
    _annotate_first_failure(ax, device_records, "vmf_robust", "purple", "robust fails")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ambient dimension D")
    ax.set_ylabel("median time per iteration [s]")
    ax.set_title(f"{device_name.upper()} benchmark")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.yaxis.set_major_locator(
        mticker.LogLocator(base=10.0, subs=[1.0, 2.0, 4.0, 6.0, 8.0], numticks=12)
    )
    ax.yaxis.set_major_formatter(
        mticker.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(float("inf"), float("inf")))
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    if device_name == "cpu":
        ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.01), fontsize=8, framealpha=0.8)
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=8, framealpha=0.8)


def _plot_primary_runtime_device(
    ax, records: list[dict], device_name: str
) -> None:
    """Paper-facing median total-time panel with failures left visible."""

    method_specs = [
        ("spcauchy_direct", "SC direct", "o"),
        ("spcauchy_direct_fixed", "SC fixed certified", "h"),
        ("spcauchy_neighbor", "SC neighbor (odd D)", "D"),
        ("spcauchy_laplace", "SC Laplace", "v"),
        ("vmf_official", "vMF original", "s"),
        ("vmf_robust", "vMF robust", "^"),
        ("power_spherical", "Power Spherical", "X"),
    ]
    device_records = [
        record for record in records if record["device"] == device_name
    ]
    for method_name, label, marker in method_specs:
        rows = sorted(
            [
                record
                for record in device_records
                if record["method"] == method_name
                and record["success"]
                and record["total_median_s"] is not None
            ],
            key=lambda item: item["dim"],
        )
        if rows:
            ax.loglog(
                [row["dim"] for row in rows],
                [row["total_median_s"] for row in rows],
                marker=marker,
                linewidth=1.4,
                label=label,
            )
    ax.set_title(f"Full latent step on {device_name.upper()}")
    ax.set_xlabel("ambient dimension D")
    ax.set_ylabel("median time [s]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)


def _annotate_first_failure(ax, device_records: list[dict], method_name: str, color: str, label_prefix: str) -> None:
    failures = sorted(
        [record for record in device_records if record["method"] == method_name and not record["success"]],
        key=lambda item: item["dim"],
    )
    if not failures:
        return

    first_fail_dim = failures[0]["dim"]
    ax.axvline(first_fail_dim, color=color, linestyle="--", alpha=0.5)
    if method_name == "vmf_official":
        ax.text(
            first_fail_dim,
            0.9,
            f"{label_prefix} ≥ d={first_fail_dim}",
            color=color,
            rotation=0,
            va="top",
            ha="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.7),
            transform=ax.get_xaxis_transform(),
        )
    else:
        ymin, ymax = ax.get_ylim()
        ax.text(
            first_fail_dim,
            ymax * 0.45,
            f"{label_prefix} ≥ d={first_fail_dim}",
            color=color,
            rotation=0,
            va="center",
            ha="right",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.7),
        )
