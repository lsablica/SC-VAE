import json
import sys
import time

import torch
import torch.nn.functional as F

from benchmark import vendor_vmf
from benchmark.vendor_vmf import VonMisesFisher, HypersphericalUniform, kl_vmf_official
from src.spcauchy import sample_spcauchy
from src.kl import kl_divergence_spcauchy_combined


def maybe_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_spcauchy_step(batch_size, dim, device, n_warmup, n_iter):
    """
    Time one spCauchy forward+backward pass for given (batch_size, dim).
    """
    results = []

    for step in range(n_warmup + n_iter):
        mu = torch.randn(batch_size, dim, device=device, requires_grad=True)
        mu = F.normalize(mu, dim=1)
        rho = torch.full(
            (batch_size, 1),
            0.7,
            device=device,
            requires_grad=True,
        )

        maybe_sync(device)
        t0 = time.perf_counter()

        z = sample_spcauchy(mu, rho)
        kl = kl_divergence_spcauchy_combined(rho, dim)
        loss = z.sum() + kl.sum()
        loss.backward()

        maybe_sync(device)
        t1 = time.perf_counter()

        if torch.isnan(loss) or torch.isinf(loss):
            return {"status": "FAIL", "time_per_iter": None}

        if step >= n_warmup:
            results.append(t1 - t0)

    return {
        "status": "OK",
        "time_per_iter": sum(results) / len(results),
    }


def time_vmf_step(batch_size, dim, device, n_warmup, n_iter, timeout):
    """
    Time one vMF forward+backward pass for given (batch_size, dim).
    Uses the official-style VonMisesFisher + HypersphericalUniform from vendor_vmf.
    """
    results = []

    for step in range(n_warmup + n_iter):
        loc = torch.randn(batch_size, dim, device=device, requires_grad=True)
        loc = F.normalize(loc, dim=1)
        kappa = torch.full(
            (batch_size, 1),
            10.0,
            device=device,
            requires_grad=True,
        )

        vmf = VonMisesFisher(loc, kappa)
        hyu = HypersphericalUniform(dim - 1, device=device)

        maybe_sync(device)
        t0 = time.perf_counter()

        try:
            z = vmf.rsample()
            kl = kl_vmf_official(vmf, hyu)
            loss = z.sum() + kl.sum()
            loss.backward()
        except RuntimeError as e:
            print(f"vMF RuntimeError at d={dim}: {e}")
            return {"status": 'FAIL', "time_per_iter": None}

        maybe_sync(device)
        t1 = time.perf_counter()
        dt = t1 - t0

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"vMF NaN/Inf at d={dim}")
            return {"status": "FAIL", "time_per_iter": None}

        if dt > timeout:
            print(f"vMF TIMEOUT at d={dim}: {dt:.3f}s per iter")
            return {"status": "TIMEOUT", "time_per_iter": dt}

        if step >= n_warmup:
            results.append(dt)

    return {
        "status": "OK",
        "time_per_iter": sum(results) / len(results),
    }


def run_stress_test(
    dims,
    batch_size=128,
    device=None,
    out_path="benchmark_results_cpu_smoke.json",
    n_warmup=10,
    n_iter=50,
    timeout=5.0,
    debug=False,
):
    """
    Run the spCauchy vs vMF throughput benchmark over a list of dimensions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode_str = "DEBUG" if debug else "NORMAL"
    print(f">>> Running stress test in {mode_str} mode")
    print(f"Using device: {device}")
    print(f"Dims: {dims}")
    print(f"Batch size: {batch_size}")
    print(f"Warmup iters: {n_warmup}, measured iters: {n_iter}\n")

    all_results = {"spcauchy": {}, "vmf": {}}

    for d in dims:
        print(f"\n=== Dimension d={d} ===")

        # spCauchy
        print("  [spCauchy] benchmarking...")
        res_spc = time_spcauchy_step(batch_size, d, device, n_warmup, n_iter)
        all_results["spcauchy"][str(d)] = res_spc
        print(f"    status={res_spc['status']}, time/iter={res_spc['time_per_iter']}")

        # vMF
        print("  [vMF] benchmarking...")
        res_vmf = time_vmf_step(batch_size, d, device, n_warmup, n_iter, timeout)
        all_results["vmf"][str(d)] = res_vmf
        print(f"    status={res_vmf['status']}, time/iter={res_vmf['time_per_iter']}")

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved benchmark results to {out_path}")
    return all_results


def main():
    # Parse command line: python -m benchmark.run_stress_test [debug]
    args = sys.argv[1:]
    debug = any(a.lower() in ("debug", "--debug", "-d") for a in args)
    cpu = any(a.lower() in ("cpu", "--cpu") for a in args)

    if cpu:
        device = torch.device("cpu")
    else:    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # these you defined in vendor_vmf.py
    vendor_vmf.DEBUG_VMF_IVE = debug
    vendor_vmf.DEBUG_VMF_ENTROPY = debug

    if debug:
        # Smaller, quicker, noisy run
        dims = (32, 64, 128, 256)
        n_warmup = 2
        n_iter = 5
        suffix = "debug"
        timeout = 10.0  # generous
    else:
        # Full stress test
        dims = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
        n_warmup = 10
        n_iter = 50
        suffix = "stress"
        timeout = 5.0

    if device.type == "cpu":
        # On CPU, reduce the max dim to avoid excessive runtimes
        batch_size=128
    else:
        batch_size=1024        
        
    out_path = f"benchmark/benchmark_results_{device.type}_{suffix}.json"
    
    run_stress_test(
        dims=dims,
        batch_size=batch_size,
        device=device,
        out_path=out_path,
        n_warmup=n_warmup,
        n_iter=n_iter,
        timeout=timeout,
        debug=debug,
    )


if __name__ == "__main__":
    main()

