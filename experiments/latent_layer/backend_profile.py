"""Profile direct-KL polynomial backends after one-time compilation."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from spherical_cauchy import (
    spherical_cauchy_kl,
    spherical_cauchy_neighbor_kl,
)

from .utils import maybe_sync, rho_from_kappa_dim, summarize_timings, write_csv


@dataclass
class BackendProfileConfig:
    device: str
    output_dir: Path
    backends: tuple[str, ...] = (
        "vectorized",
        "compiled",
        "triton",
    )
    direct_dims: tuple[int, ...] = (8, 128, 2048)
    neighbor_dims: tuple[int, ...] = (9, 129, 2049)
    warmups: int = 10
    measurements: int = 50
    repeats: int = 5
    kappa: float = 10.0


def _run_one(
    evaluator,
    dimension: int,
    backend: str,
    *,
    device: torch.device,
    batch_size: int,
    warmups: int,
    measurements: int,
    repeats: int,
    kappa: float,
) -> dict:
    rho_value = rho_from_kappa_dim(kappa, dimension)
    forward: list[float] = []
    backward: list[float] = []
    total: list[float] = []
    memories: list[int] = []
    success = True
    error_message = None
    try:
        for repeat in range(repeats):
            torch.manual_seed(repeat)
            for index in range(warmups + measurements):
                rho = torch.full(
                    (batch_size, 1),
                    rho_value,
                    device=device,
                    dtype=torch.float32,
                    requires_grad=True,
                )
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                maybe_sync(device)
                start = time.perf_counter()
                value = evaluator(rho, dimension, backend=backend)
                loss = value.sum()
                maybe_sync(device)
                forward_end = time.perf_counter()
                loss.backward()
                maybe_sync(device)
                backward_end = time.perf_counter()
                if index >= warmups:
                    forward.append(forward_end - start)
                    backward.append(backward_end - forward_end)
                    total.append(backward_end - start)
                    if device.type == "cuda":
                        memories.append(
                            int(torch.cuda.max_memory_allocated(device))
                        )
    except Exception as exc:
        success = False
        error_message = str(exc)

    forward_summary = summarize_timings(forward)
    backward_summary = summarize_timings(backward)
    total_summary = summarize_timings(total)
    return {
        "backend": backend,
        "device": device.type,
        "dimension": dimension,
        "batch_size": batch_size,
        "rho": rho_value,
        "warmups": warmups,
        "measurements": measurements,
        "repeats": repeats,
        "success": success,
        "error_message": error_message,
        "forward_mean_s": forward_summary["mean"],
        "forward_std_s": forward_summary["std"],
        "forward_median_s": forward_summary["median"],
        "forward_iqr_s": forward_summary["iqr"],
        "backward_mean_s": backward_summary["mean"],
        "backward_std_s": backward_summary["std"],
        "backward_median_s": backward_summary["median"],
        "backward_iqr_s": backward_summary["iqr"],
        "total_mean_s": total_summary["mean"],
        "total_std_s": total_summary["std"],
        "total_median_s": total_summary["median"],
        "total_iqr_s": total_summary["iqr"],
        "peak_cuda_memory_bytes": max(memories) if memories else None,
    }


def run_backend_profile(config: BackendProfileConfig) -> list[dict]:
    device = torch.device(config.device)
    batch_size = 1024 if device.type == "cuda" else 128
    backends = [
        backend
        for backend in config.backends
        if not (backend == "triton" and device.type != "cuda")
    ]
    records: list[dict] = []
    for evaluator_name, evaluator, dimensions in [
        (
            "direct",
            spherical_cauchy_kl,
            config.direct_dims,
        ),
        (
            "neighbor",
            spherical_cauchy_neighbor_kl,
            config.neighbor_dims,
        ),
    ]:
        for dimension in dimensions:
            for backend in backends:
                record = _run_one(
                    evaluator,
                    dimension,
                    backend,
                    device=device,
                    batch_size=batch_size,
                    warmups=config.warmups,
                    measurements=config.measurements,
                    repeats=config.repeats,
                    kappa=config.kappa,
                )
                record["evaluator"] = evaluator_name
                records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--measurements", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    config = BackendProfileConfig(
        device=args.device,
        output_dir=Path(args.output_dir),
        warmups=args.warmups,
        measurements=args.measurements,
        repeats=args.repeats,
    )
    records = run_backend_profile(config)
    print(
        write_csv(
            records,
            config.output_dir
            / "results"
            / f"direct_kl_backend_profile_{args.device}.csv",
        )
    )


if __name__ == "__main__":
    main()
