"""Full latent-step runtime benchmark."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .defaults import DEFAULT_RUNTIME_DIMS, DEFAULT_SPCAUCHY_RUNTIME_METHODS, DEFAULT_VMF_RUNTIME_METHODS
from .methods import build_vmf_distribution, get_latent_step_method, kl_for_spcauchy_runtime
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
from src.spcauchy import sample_spcauchy


@dataclass
class RuntimeConfig:
    dims: list[int] = field(default_factory=lambda: list(DEFAULT_RUNTIME_DIMS))
    spcauchy_methods: list[str] = field(default_factory=lambda: list(DEFAULT_SPCAUCHY_RUNTIME_METHODS))
    vmf_methods: list[str] = field(default_factory=lambda: list(DEFAULT_VMF_RUNTIME_METHODS))
    device_name: str = "auto"
    dtype: torch.dtype = torch.float32
    seed: int = 0
    batch_size: int | None = None
    warmup_iters: int = 10
    measure_iters: int = 50
    timeout_s: float = 5.0
    quadrature_nodes: int | None = 1000
    series_k_terms: int | None = None
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
    quadrature_nodes: int | None,
    series_k_terms: int | None,
) -> dict:
    raw_mu = torch.randn(batch_size, latent_dim, device=device, dtype=dtype, requires_grad=True)
    mu = F.normalize(raw_mu, dim=1)
    rho = torch.full((batch_size, 1), rho_value, device=device, dtype=dtype, requires_grad=True)

    maybe_sync(device)
    t0 = time.perf_counter()
    z = sample_spcauchy(mu, rho)
    kl = kl_for_spcauchy_runtime(
        method_name,
        rho,
        latent_dim,
        quadrature_nodes=quadrature_nodes,
        series_k_terms=series_k_terms,
    )
    loss = z.sum() + kl.sum()
    maybe_sync(device)
    t1 = time.perf_counter()

    loss.backward()
    maybe_sync(device)
    t2 = time.perf_counter()

    nan_or_inf_loss = bool(torch.isnan(loss) or torch.isinf(loss))
    nan_or_inf_grad = _gradient_has_nonfinite(raw_mu, rho)
    return {
        "forward_time_s": t1 - t0,
        "backward_time_s": t2 - t1,
        "total_time_s": t2 - t0,
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
    raw_loc = torch.randn(batch_size, latent_dim, device=device, dtype=dtype, requires_grad=True)
    loc = F.normalize(raw_loc, dim=1)
    kappa = torch.full((batch_size, 1), kappa_value, device=device, dtype=dtype, requires_grad=True)

    maybe_sync(device)
    t0 = time.perf_counter()
    if abs(kappa_value) <= 1e-12:
        z = loc
        kl = torch.zeros(batch_size, device=device, dtype=dtype)
    else:
        vmf, hyu, kl_fn = build_vmf_distribution(method_name, loc, kappa)
        z = vmf.rsample()
        kl = kl_fn(vmf, hyu)
    loss = z.sum() + kl.sum()
    maybe_sync(device)
    t1 = time.perf_counter()

    loss.backward()
    maybe_sync(device)
    t2 = time.perf_counter()

    nan_or_inf_loss = bool(torch.isnan(loss) or torch.isinf(loss))
    nan_or_inf_grad = _gradient_has_nonfinite(raw_loc, kappa)
    return {
        "forward_time_s": t1 - t0,
        "backward_time_s": t2 - t1,
        "total_time_s": t2 - t0,
        "nan_or_inf_loss": nan_or_inf_loss,
        "nan_or_inf_grad": nan_or_inf_grad,
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
    forward_times: list[float] = []
    backward_times: list[float] = []
    total_times: list[float] = []
    nan_or_inf_loss = False
    nan_or_inf_grad = False
    success = True
    failure_type = None
    error_message = None
    timeout_hit = False

    batch_size = config.batch_size or _default_batch_size(device)
    total_iters = config.warmup_iters + config.measure_iters

    for step_idx in range(total_iters):
        try:
            if method_name.startswith("spcauchy_"):
                result = _run_spcauchy_iteration(
                    method_name,
                    batch_size=batch_size,
                    latent_dim=latent_dim,
                    rho_value=float(rho_value),
                    device=device,
                    dtype=config.dtype,
                    quadrature_nodes=config.quadrature_nodes,
                    series_k_terms=config.series_k_terms,
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

        if result["total_time_s"] > config.timeout_s:
            success = False
            timeout_hit = True
            failure_type = "timeout"
            error_message = f"Iteration exceeded timeout {config.timeout_s:.3f}s."
            break

        if step_idx >= config.warmup_iters:
            forward_times.append(result["forward_time_s"])
            backward_times.append(result["backward_time_s"])
            total_times.append(result["total_time_s"])

    forward_summary = summarize_timings(forward_times)
    backward_summary = summarize_timings(backward_times)
    total_summary = summarize_timings(total_times)
    method = get_latent_step_method(method_name)

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
        "batch_size": batch_size,
        "success": success,
        "failure_type": failure_type,
        "error_message": error_message,
        "warmup_iters": config.warmup_iters,
        "measure_iters": config.measure_iters,
        "timeout_s": config.timeout_s,
        "nan_or_inf_loss": nan_or_inf_loss,
        "nan_or_inf_grad": nan_or_inf_grad,
        "runtime_error": failure_type in {"runtime_error", "exception"},
        "timeout_hit": timeout_hit,
        "forward_mean_s": forward_summary["mean"],
        "forward_std_s": forward_summary["std"],
        "forward_median_s": forward_summary["median"],
        "forward_min_s": forward_summary["min"],
        "forward_max_s": forward_summary["max"],
        "backward_mean_s": backward_summary["mean"],
        "backward_std_s": backward_summary["std"],
        "backward_median_s": backward_summary["median"],
        "backward_min_s": backward_summary["min"],
        "backward_max_s": backward_summary["max"],
        "total_mean_s": total_summary["mean"],
        "total_std_s": total_summary["std"],
        "total_median_s": total_summary["median"],
        "total_min_s": total_summary["min"],
        "total_max_s": total_summary["max"],
    }


def run_runtime_benchmark(config: RuntimeConfig) -> list[dict]:
    seed_all(config.seed)
    device = resolve_device(config.device_name)
    records: list[dict] = []

    for latent_dim in config.dims:
        if config.concentration_mode == "matched":
            rho_values = [rho_from_kappa_dim(config.kappa, latent_dim)]
            kappa_values = [config.kappa]
        else:
            rho_values = list(config.rho_values)
            kappa_values = [None]

        for rho_value in rho_values:
            for method_name in config.spcauchy_methods:
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
                for method_name in config.vmf_methods:
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
    figures = plot_runtime_results(records, layout) if generate_plots else []
    return layout, str(csv_path), figures


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
            total_mean = [record["total_mean_s"] for record in method_records]
            ax.loglog(dims, total_mean, marker="o", label=method_name)
        ax.set_title(f"Latent-step total time ({device_name})")
        ax.set_xlabel("latent dimension d")
        ax.set_ylabel("time [s]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    total_path = layout.figures_dir / "latent_step_runtime_total.png"
    fig.savefig(total_path, dpi=200)
    plt.close(fig)
    figure_paths.append(str(total_path))

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
            forward_mean = [record["forward_mean_s"] for record in method_records]
            backward_mean = [record["backward_mean_s"] for record in method_records]
            ax.loglog(dims, forward_mean, marker="o", linestyle="-", label=f"{method_name} forward")
            ax.loglog(dims, backward_mean, marker="s", linestyle="--", label=f"{method_name} backward")
        ax.set_title(f"Forward/backward split ({device_name})")
        ax.set_xlabel("latent dimension d")
        ax.set_ylabel("time [s]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    split_path = layout.figures_dir / "latent_step_runtime_forward_backward.png"
    fig.savefig(split_path, dpi=200)
    plt.close(fig)
    figure_paths.append(str(split_path))

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

    fig.suptitle("spCauchy vs. vMF: Time per iteration vs. latent dimension", fontsize=14)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    classic_path = layout.figures_dir / "benchmark_spcauchy_vs_vmf_stress.png"
    fig.savefig(classic_path, dpi=200)
    plt.close(fig)
    figure_paths.append(str(classic_path))
    return figure_paths


def _plot_classic_runtime_device(ax, records: list[dict], device_name: str, mticker) -> None:
    method_specs = [
        ("spcauchy_combined", "spCauchy (quadrature)", "o"),
        ("spcauchy_hybrid", "spCauchy (hybrid)", "D"),
        ("vmf_official", "vMF (official)", "s"),
        ("vmf_robust", "vMF (robust)", "^"),
    ]
    device_records = [record for record in records if record["device"] == device_name]
    dims = sorted({record["dim"] for record in device_records})

    for method_name, label, marker in method_specs:
        success_records = sorted(
            [
                record
                for record in device_records
                if record["method"] == method_name and record["success"] and record["total_mean_s"] is not None
            ],
            key=lambda item: item["dim"],
        )
        if not success_records:
            continue
        ax.plot(
            [record["dim"] for record in success_records],
            [record["total_mean_s"] for record in success_records],
            f"-{marker}",
            label=label,
        )

    _annotate_first_failure(ax, device_records, "vmf_official", "red", "vMF official fails")
    _annotate_first_failure(ax, device_records, "vmf_robust", "purple", "robust fails")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Latent dimension d")
    ax.set_ylabel("Time per iteration [s]")
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
