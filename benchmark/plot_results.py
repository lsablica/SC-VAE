# benchmark/plot_results.py
import numpy as np
import json
import os
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt


def load_device_results(path):
    """
    Load a benchmark_results_XXX_stress.json file and extract:
    - dims (sorted list of ints)
    - spcauchy_times (list of floats)
    - vmf_times (list of floats or None)
    - vmf_status (list of strings)
    - first_fail_dim_vmf (int or None)
    - vmf_robust_times (list of floats or None)
    - vmf_robust_status (list of strings)
    - first_fail_dim_vmf_robust (int or None)
    """
    with open(path, "r") as f:
        data = json.load(f)

    sp = data["spcauchy"]
    vm = data["vmf"]
    vr = data["vmf_robust"]

    dims = sorted(int(d) for d in sp.keys())
    dims_str = [str(d) for d in dims]

    spcauchy_times = []

    vmf_times = []
    vmf_status = []
    first_fail_dim_vmf = None

    vmf_robust_times = []
    vmf_robust_status = []
    first_fail_dim_vmf_robust = None

    for d_str, d in zip(dims_str, dims):
        r_sp = sp[d_str]
        r_vmf = vm[d_str]
        r_vr = vr[d_str]

        spcauchy_times.append(r_sp["time_per_iter"])

        vmf_times.append(r_vmf["time_per_iter"])
        vmf_status.append(r_vmf["status"])
        if first_fail_dim_vmf is None and r_vmf["status"] != "OK":
            first_fail_dim_vmf = d

        vmf_robust_times.append(r_vr["time_per_iter"])
        vmf_robust_status.append(r_vr["status"])
        if first_fail_dim_vmf_robust is None and r_vr["status"] != "OK":
            first_fail_dim_vmf_robust = d

    return (
        dims,
        spcauchy_times,
        vmf_times,
        vmf_status,
        first_fail_dim_vmf,
        vmf_robust_times,
        vmf_robust_status,
        first_fail_dim_vmf_robust,
    )


def plot_single_device(
    ax,
    dims,
    sp_times,
    vm_times,
    vm_status,
    first_fail_dim_vmf,
    vr_times,
    vr_status,
    first_fail_dim_vmf_robust,
    title,
):
    """
    Make a log–log plot for one device on the given Axes.
    """
    dims = list(dims)

    ax.plot(dims, sp_times, "-o", label="spCauchy (ours)")

    vm_ok_dims = [d for d, t, s in zip(dims, vm_times, vm_status) if s == "OK" and t is not None]
    vm_ok_times = [t for t, s in zip(vm_times, vm_status) if s == "OK" and t is not None]
    if vm_ok_dims:
        ax.plot(vm_ok_dims, vm_ok_times, "-s", label="vMF (official)")

    vr_ok_dims = [d for d, t, s in zip(dims, vr_times, vr_status) if s == "OK" and t is not None]
    vr_ok_times = [t for t, s in zip(vr_times, vr_status) if s == "OK" and t is not None]
    if vr_ok_dims:
        ax.plot(vr_ok_dims, vr_ok_times, "-^", label="vMF (robust)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Latent dimension d")
    ax.set_ylabel("Time per iteration [s]")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    if first_fail_dim_vmf is not None:
        ax.axvline(first_fail_dim_vmf, color="red", linestyle="--", alpha=0.5)
        ymin, ymax = ax.get_ylim()
        ax.text(
            first_fail_dim_vmf,
            0.9,
            f"vMF official fails ≥ d={first_fail_dim_vmf}",
            color="red",
            rotation=0,
            va="top",
            ha="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.7),
            transform=ax.get_xaxis_transform(),
        )

    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=np.concatenate(([1], np.arange(2, 9, 2))), numticks=12))

    def force_sci_format(x, pos):
        return f"{x:.1e}".replace("e-0", " × 10^{-").replace("e", " × 10^{") + "}"

    formatter = mticker.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(np.inf, np.inf))
    ax.yaxis.set_major_formatter(formatter)
    
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


    if first_fail_dim_vmf_robust is not None:
        ax.axvline(first_fail_dim_vmf_robust, color="purple", linestyle="--", alpha=0.5)
        ymin, ymax = ax.get_ylim()
        ax.text(
            first_fail_dim_vmf_robust,
            ymax * 0.45,
            f"robust fails ≥ d={first_fail_dim_vmf_robust}",
            color="purple",
            rotation=0,
            va="center",
            ha="right",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="purple", alpha=0.7),
        )

    if "CPU" in title:
        # Snap to bottom right
        ax.legend(loc='lower right', bbox_to_anchor=(0.99, 0.01), fontsize=8, framealpha=0.8)
    else:
        # Snap to top left (CUDA)
        ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=8, framealpha=0.8)


def main():
    cpu_path = "benchmark/benchmark_results_cpu_stress.json"
    gpu_path = "benchmark/benchmark_results_cuda_stress.json"

    device_files = []
    if os.path.exists(cpu_path):
        device_files.append(("CPU", cpu_path))
    if os.path.exists(gpu_path):
        device_files.append(("CUDA", gpu_path))

    if not device_files:
        raise FileNotFoundError(
            "No benchmark_results_*_stress.json files found in benchmark/.\n"
            "Run `python -m benchmark.run_stress_test` first."
        )

    n = len(device_files)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    axes = axes[0]  # flatten row

    for ax, (label, path) in zip(axes, device_files):
        (
            dims,
            sp_times,
            vm_times,
            vm_status,
            first_fail_dim_vmf,
            vr_times,
            vr_status,
            first_fail_dim_vmf_robust,
        ) = load_device_results(path)

        plot_single_device(
            ax,
            dims,
            sp_times,
            vm_times,
            vm_status,
            first_fail_dim_vmf,
            vr_times,
            vr_status,
            first_fail_dim_vmf_robust,
            title=f"{label} benchmark",
        )

    fig.suptitle("spCauchy vs. vMF: Time per iteration vs. latent dimension", fontsize=14)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])

    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", "benchmark_spcauchy_vs_vmf_stress.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
