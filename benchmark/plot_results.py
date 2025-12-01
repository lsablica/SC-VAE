# benchmark/plot_results.py

import json
import os

import matplotlib.pyplot as plt


def load_device_results(path):
    """
    Load a benchmark_results_XXX_stress.json file and extract:
    - dims (sorted list of ints)
    - spcauchy_times (list of floats)
    - vmf_times (list of floats or None)
    - vmf_status (list of strings)
    - first_fail_dim (int or None)
    """
    with open(path, "r") as f:
        data = json.load(f)

    sp = data["spcauchy"]
    vm = data["vmf"]

    # Sort dimensions numerically
    dims = sorted(int(d) for d in sp.keys())
    dims_str = [str(d) for d in dims]

    spcauchy_times = []
    vmf_times = []
    vmf_status = []
    first_fail_dim = None

    for d_str, d in zip(dims_str, dims):
        r_sp = sp[d_str]
        r_vmf = vm[d_str]

        spcauchy_times.append(r_sp["time_per_iter"])
        vmf_times.append(r_vmf["time_per_iter"])
        vmf_status.append(r_vmf["status"])

        if first_fail_dim is None and r_vmf["status"] != "OK":
            first_fail_dim = d

    return dims, spcauchy_times, vmf_times, vmf_status, first_fail_dim


def plot_single_device(ax, dims, sp_times, vm_times, vm_status, first_fail_dim, title):
    """
    Make a log–log plot for one device on the given Axes.
    """
    # Convert to floats and mask NaNs
    dims = list(dims)

    # spCauchy: all OK, so we can just plot
    ax.plot(dims, sp_times, "-o", label="spCauchy (ours)")

    # vMF: only plot dimensions where status == 'OK' and time is not None
    vm_ok_dims = [d for d, t, s in zip(dims, vm_times, vm_status) if s == "OK" and t is not None]
    vm_ok_times = [t for t, s in zip(vm_times, vm_status) if s == "OK" and t is not None]

    if vm_ok_dims:
        ax.plot(vm_ok_dims, vm_ok_times, "-s", label="vMF (official)")

    # Axis scales and labels
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Latent dimension d")
    ax.set_ylabel("Time per iteration [s]")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    if first_fail_dim is not None:
        ax.axvline(first_fail_dim, color="red", linestyle="--", alpha=0.6)

        ymin, ymax = ax.get_ylim()
        ax.text(
            first_fail_dim,
            ymax * 0.6,
            f"vMF fails ≥ d={first_fail_dim}",
            color="red",
            rotation=90,
            va="center",
            ha="right",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.7),
        )

    ax.legend()


def main():
    # Paths to the JSON results
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

    # Prepare figure: one subplot per available device
    n = len(device_files)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    axes = axes[0]  # flatten row

    for ax, (label, path) in zip(axes, device_files):
        dims, sp_times, vm_times, vm_status, first_fail_dim = load_device_results(path)
        plot_single_device(
            ax,
            dims,
            sp_times,
            vm_times,
            vm_status,
            first_fail_dim,
            title=f"{label} benchmark",
        )

    fig.suptitle("spCauchy vs. vMF: Time per iteration vs. latent dimension", fontsize=14)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])

    # Ensure figures directory exists
    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", "benchmark_spcauchy_vs_vmf_stress.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()

