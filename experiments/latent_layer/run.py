"""Command-line entry point for the paper's latent-layer benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .defaults import (
    DEFAULT_ACCURACY_DIMS,
    DEFAULT_NEIGHBOR_RUNTIME_DIMS,
    DEFAULT_POWER_RUNTIME_METHODS,
    DEFAULT_RUNTIME_DIMS,
    DEFAULT_SPCAUCHY_ACCURACY_METHODS,
    DEFAULT_SPCAUCHY_RHO_GRID,
    DEFAULT_SPCAUCHY_ROBUSTNESS_METHODS,
    DEFAULT_SPCAUCHY_RUNTIME_METHODS,
    DEFAULT_VMF_KAPPA_GRID,
    DEFAULT_VMF_ROBUSTNESS_METHODS,
    DEFAULT_VMF_RUNTIME_METHODS,
)
from .utils import prepare_output_layout


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-run spherical Cauchy latent-layer evaluations."
    )
    parser.add_argument(
        "command", choices=("accuracy", "runtime", "robustness", "all")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--dims", nargs="+", type=int)
    parser.add_argument("--rho-grid", nargs="+", type=float)
    parser.add_argument("--kappa-grid", nargs="+", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--measure-iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--kappa", type=float, default=10.0)
    parser.add_argument(
        "--direct-backend",
        choices=("auto", "vectorized", "compiled", "triton"),
        default="auto",
    )
    parser.add_argument(
        "--laplace-backend",
        choices=("auto", "eager", "compiled", "triton"),
        default="auto",
    )
    parser.add_argument("--out-dir")
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def _devices(requested: str) -> list[str]:
    if requested == "auto":
        return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    return [requested]


def _dtype(name: str | None, default: torch.dtype) -> torch.dtype:
    if name is None:
        return default
    return torch.float32 if name == "float32" else torch.float64


def _accuracy(args: argparse.Namespace) -> tuple[str, list[str]]:
    from .accuracy import AccuracyConfig, run_accuracy_benchmark, save_accuracy_outputs

    config = AccuracyConfig(
        dims=args.dims or list(DEFAULT_ACCURACY_DIMS),
        rho_grid=args.rho_grid or list(DEFAULT_SPCAUCHY_RHO_GRID),
        methods=list(DEFAULT_SPCAUCHY_ACCURACY_METHODS),
        device_name="cpu" if args.device == "auto" else args.device,
        dtype=_dtype(args.dtype, torch.float64),
        seed=args.seed,
        out_dir=args.out_dir,
    )
    rows = run_accuracy_benchmark(config)
    _, csv_path, figures = save_accuracy_outputs(
        rows, out_dir=args.out_dir, generate_plots=not args.skip_plots
    )
    return csv_path, figures


def _runtime(args: argparse.Namespace) -> tuple[str, list[str]]:
    from .runtime import RuntimeConfig, run_runtime_benchmark, save_runtime_outputs

    rows: list[dict] = []
    for device in _devices(args.device):
        config = RuntimeConfig(
            dims=args.dims or list(DEFAULT_RUNTIME_DIMS),
            neighbor_dims=list(DEFAULT_NEIGHBOR_RUNTIME_DIMS),
            spcauchy_methods=list(DEFAULT_SPCAUCHY_RUNTIME_METHODS),
            vmf_methods=list(DEFAULT_VMF_RUNTIME_METHODS),
            power_methods=list(DEFAULT_POWER_RUNTIME_METHODS),
            device_name=device,
            dtype=_dtype(args.dtype, torch.float32),
            seed=args.seed,
            batch_size=args.batch_size,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
            repeats=args.repeats,
            timeout_s=args.timeout_s,
            direct_backend=args.direct_backend,
            laplace_backend=args.laplace_backend,
            kappa=args.kappa,
            out_dir=args.out_dir,
        )
        rows.extend(run_runtime_benchmark(config))
    _, csv_path, figures = save_runtime_outputs(
        rows, out_dir=args.out_dir, generate_plots=not args.skip_plots
    )
    return csv_path, figures


def _robustness(args: argparse.Namespace) -> tuple[str, list[str]]:
    from .robustness import (
        RobustnessConfig,
        run_robustness_benchmark,
        save_robustness_outputs,
    )

    config = RobustnessConfig(
        spcauchy_dims=args.dims or list(DEFAULT_ACCURACY_DIMS),
        vmf_dims=args.dims or list(DEFAULT_RUNTIME_DIMS),
        rho_grid=args.rho_grid or list(DEFAULT_SPCAUCHY_RHO_GRID),
        kappa_grid=args.kappa_grid or list(DEFAULT_VMF_KAPPA_GRID),
        spcauchy_methods=list(DEFAULT_SPCAUCHY_ROBUSTNESS_METHODS),
        vmf_methods=list(DEFAULT_VMF_ROBUSTNESS_METHODS),
        power_methods=list(DEFAULT_POWER_RUNTIME_METHODS),
        device_name="cpu" if args.device == "auto" else args.device,
        dtype=_dtype(args.dtype, torch.float32),
        seed=args.seed,
        batch_size=args.batch_size or 32,
        timeout_s=args.timeout_s,
        out_dir=args.out_dir,
    )
    rows = run_robustness_benchmark(config)
    _, csv_path, figures = save_robustness_outputs(
        rows, out_dir=args.out_dir, generate_plots=not args.skip_plots
    )
    return csv_path, figures


def _write_summary(out_dir: str | None, sections: list[tuple[str, str, list[str]]]) -> None:
    layout = prepare_output_layout(out_dir)
    lines = ["# Latent-layer benchmark", ""]
    for title, csv_path, figures in sections:
        lines.extend([f"## {title}", "", f"- Results: `{Path(csv_path).as_posix()}`"])
        lines.extend(f"- Figure: `{Path(path).as_posix()}`" for path in figures)
        lines.append("")
    layout.summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parser().parse_args()
    runners = {
        "accuracy": _accuracy,
        "runtime": _runtime,
        "robustness": _robustness,
    }
    requested = tuple(runners) if args.command == "all" else (args.command,)
    sections = []
    for name in requested:
        csv_path, figures = runners[name](args)
        sections.append((name.title(), csv_path, figures))
    _write_summary(args.out_dir, sections)


if __name__ == "__main__":
    main()
