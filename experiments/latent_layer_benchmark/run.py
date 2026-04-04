"""CLI entrypoint for the latent-layer benchmark refresh."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .defaults import (
    DEFAULT_ACCURACY_DIMS,
    DEFAULT_RUNTIME_DIMS,
    DEFAULT_SPCAUCHY_ACCURACY_METHODS,
    DEFAULT_SPCAUCHY_RHO_GRID,
    DEFAULT_VMF_KAPPA_GRID,
)
from .utils import dtype_name, parse_dtype, prepare_output_layout


def _dtype_from_args(args: argparse.Namespace) -> torch.dtype:
    return parse_dtype(args.dtype)


def _accuracy_dtype_from_args(args: argparse.Namespace) -> torch.dtype:
    return parse_dtype(args.dtype or "float64")


def _runtime_or_robustness_dtype_from_args(args: argparse.Namespace) -> torch.dtype:
    return parse_dtype(args.dtype or "float32")


def _write_summary(layout_path: str | None, sections: list[dict]) -> Path:
    layout = prepare_output_layout(layout_path)
    lines = ["# Latent-Layer Benchmark Summary", ""]
    for section in sections:
        lines.append(f"## {section['title']}")
        lines.append("")
        lines.append(section["body"])
        lines.append("")
        if section.get("csv"):
            lines.append(f"- CSV: `{_display_path(section['csv'])}`")
        for figure in section.get("figures", []):
            lines.append(f"- Figure: `{_display_path(figure)}`")
        lines.append("")

    layout.summary_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return layout.summary_path


def _accuracy_body(config, csv_path: str) -> str:
    methods = ", ".join(config.methods)
    return (
        f"Evaluated spCauchy KL routes `{methods}` over {len(config.dims)} dimensions and "
        f"{len(config.rho_grid)} rho values using `{dtype_name(config.dtype)}` on `{config.device_name}`. "
        f"Results were written to `{_display_path(csv_path)}`."
    )


def _runtime_body_for_devices(config, csv_path: str, devices: list[str]) -> str:
    methods = ", ".join(config.spcauchy_methods + config.vmf_methods)
    device_label = " + ".join(device.upper() for device in devices)
    return (
        f"Benchmarked latent-step methods `{methods}` over {len(config.dims)} dimensions with "
        f"`{config.measure_iters}` measured iterations on `{device_label}` using `{dtype_name(config.dtype)}`. "
        f"Results were written to `{_display_path(csv_path)}`."
    )


def _robustness_body(config, csv_path: str) -> str:
    return (
        f"Swept spCauchy robustness over {len(config.spcauchy_dims)} dimensions and {len(config.rho_grid)} rho values, "
        f"and vMF robustness over {len(config.vmf_dims)} dimensions and {len(config.kappa_grid)} kappa values. "
        f"Results were written to `{_display_path(csv_path)}`."
    )


def _display_path(path: str | Path) -> str:
    return Path(path).as_posix()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Latent-layer benchmark refresh for the spCauchy-VAE paper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--seed", type=int, default=0)
        subparser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
        subparser.add_argument("--dtype", choices=["float32", "float64"], default=None)
        subparser.add_argument("--dims", nargs="+", type=int, default=None)
        subparser.add_argument("--rho-grid", nargs="+", type=float, default=None)
        subparser.add_argument("--kappa-grid", nargs="+", type=float, default=None)
        subparser.add_argument("--batch-size", type=int, default=None)
        subparser.add_argument("--warmup-iters", type=int, default=10)
        subparser.add_argument("--measure-iters", type=int, default=50)
        subparser.add_argument("--timeout-s", type=float, default=5.0)
        subparser.add_argument("--out-dir", type=str, default=None)
        subparser.add_argument("--skip-plots", action="store_true")

    accuracy_parser = subparsers.add_parser("accuracy", help="Run the spCauchy KL accuracy sweep.")
    add_shared_arguments(accuracy_parser)
    accuracy_parser.add_argument("--reference-nodes", type=int, default=4096)
    accuracy_parser.add_argument("--reference-k-terms", type=int, default=None)
    accuracy_parser.add_argument("--reference-disagreement-tol", type=float, default=1e-8)
    accuracy_parser.add_argument("--quadrature-nodes", type=int, default=1000)
    accuracy_parser.add_argument("--series-k-terms", type=int, default=None)
    accuracy_parser.add_argument("--include-auto", action="store_true")

    runtime_parser = subparsers.add_parser("runtime", help="Run the latent-step runtime benchmark.")
    add_shared_arguments(runtime_parser)
    runtime_parser.add_argument("--kappa", type=float, default=10.0)
    runtime_parser.add_argument("--concentration-mode", choices=["matched", "direct-rho"], default="matched")
    runtime_parser.add_argument("--rho", nargs="+", type=float, default=[0.9])
    runtime_parser.add_argument("--quadrature-nodes", type=int, default=1000)
    runtime_parser.add_argument("--series-k-terms", type=int, default=None)
    runtime_parser.add_argument("--include-series", action="store_true")
    runtime_parser.add_argument("--include-auto", action="store_true")

    robustness_parser = subparsers.add_parser("robustness", help="Run the stability regime sweep.")
    add_shared_arguments(robustness_parser)
    robustness_parser.add_argument("--reference-nodes", type=int, default=4096)
    robustness_parser.add_argument("--reference-disagreement-tol", type=float, default=1e-8)
    robustness_parser.add_argument("--quadrature-nodes", type=int, default=1000)
    robustness_parser.add_argument("--series-k-terms", type=int, default=None)
    robustness_parser.add_argument("--include-auto", action="store_true")

    all_parser = subparsers.add_parser("all", help="Run accuracy, runtime, and robustness with defaults.")
    add_shared_arguments(all_parser)
    all_parser.add_argument("--reference-nodes", type=int, default=4096)
    all_parser.add_argument("--reference-k-terms", type=int, default=None)
    all_parser.add_argument("--reference-disagreement-tol", type=float, default=1e-8)
    all_parser.add_argument("--quadrature-nodes", type=int, default=1000)
    all_parser.add_argument("--series-k-terms", type=int, default=None)
    all_parser.add_argument("--include-auto", action="store_true")
    all_parser.add_argument("--include-series", action="store_true")
    all_parser.add_argument("--kappa", type=float, default=10.0)
    return parser


def _runtime_devices_from_arg(device_arg: str) -> list[str]:
    normalized = device_arg.strip().lower()
    if normalized == "auto":
        return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    return [normalized]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "accuracy":
        from .accuracy import AccuracyConfig, run_accuracy_benchmark, save_accuracy_outputs

        methods = list(DEFAULT_SPCAUCHY_ACCURACY_METHODS)
        if args.include_auto:
            methods.append("auto")
        config = AccuracyConfig(
            dims=args.dims or list(DEFAULT_ACCURACY_DIMS),
            rho_grid=args.rho_grid or list(DEFAULT_SPCAUCHY_RHO_GRID),
            methods=methods,
            device_name="cpu" if args.device == "auto" else args.device,
            dtype=_accuracy_dtype_from_args(args),
            seed=args.seed,
            quadrature_nodes=args.quadrature_nodes,
            reference_nodes=args.reference_nodes,
            series_k_terms=args.series_k_terms,
            reference_k_terms=args.reference_k_terms,
            reference_disagreement_tol=args.reference_disagreement_tol,
            out_dir=args.out_dir,
        )
        records = run_accuracy_benchmark(config)
        _, csv_path, figures = save_accuracy_outputs(records, out_dir=config.out_dir, generate_plots=not args.skip_plots)
        _write_summary(
            config.out_dir,
            [{"title": "KL Accuracy", "body": _accuracy_body(config, csv_path), "csv": csv_path, "figures": figures}],
        )
        return

    if args.command == "runtime":
        from .runtime import RuntimeConfig, run_runtime_benchmark, save_runtime_outputs

        spcauchy_methods = ["spcauchy_combined", "spcauchy_hybrid"]
        if args.include_series:
            spcauchy_methods.append("spcauchy_series")
        if args.include_auto:
            spcauchy_methods.append("spcauchy_auto")
        vmf_methods = ["vmf_official", "vmf_robust"] if args.concentration_mode == "matched" else []
        runtime_devices = _runtime_devices_from_arg(args.device)
        records = []
        config = None
        for runtime_device in runtime_devices:
            config = RuntimeConfig(
                dims=args.dims or list(DEFAULT_RUNTIME_DIMS),
                spcauchy_methods=spcauchy_methods,
                vmf_methods=vmf_methods,
                device_name=runtime_device,
                dtype=_runtime_or_robustness_dtype_from_args(args),
                seed=args.seed,
                batch_size=args.batch_size,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                timeout_s=args.timeout_s,
                quadrature_nodes=args.quadrature_nodes,
                series_k_terms=args.series_k_terms,
                kappa=args.kappa,
                concentration_mode=args.concentration_mode,
                rho_values=args.rho,
                out_dir=args.out_dir,
            )
            records.extend(run_runtime_benchmark(config))
        _, csv_path, figures = save_runtime_outputs(records, out_dir=config.out_dir, generate_plots=not args.skip_plots)
        _write_summary(
            config.out_dir,
            [{"title": "Runtime", "body": _runtime_body_for_devices(config, csv_path, runtime_devices), "csv": csv_path, "figures": figures}],
        )
        return

    if args.command == "robustness":
        from .robustness import RobustnessConfig, run_robustness_benchmark, save_robustness_outputs

        sp_methods = ["series", "combined", "asymptotic_high_rho", "hybrid"]
        if args.include_auto:
            sp_methods.append("auto")
        config = RobustnessConfig(
            spcauchy_dims=args.dims or list(DEFAULT_ACCURACY_DIMS),
            vmf_dims=args.dims or list(DEFAULT_RUNTIME_DIMS),
            rho_grid=args.rho_grid or list(DEFAULT_SPCAUCHY_RHO_GRID),
            kappa_grid=args.kappa_grid or list(DEFAULT_VMF_KAPPA_GRID),
            spcauchy_methods=sp_methods,
            vmf_methods=["vmf_official", "vmf_robust"],
            device_name="cpu" if args.device == "auto" else args.device,
            dtype=_runtime_or_robustness_dtype_from_args(args),
            seed=args.seed,
            batch_size=args.batch_size or 32,
            timeout_s=args.timeout_s,
            quadrature_nodes=args.quadrature_nodes,
            series_k_terms=args.series_k_terms,
            reference_nodes=args.reference_nodes,
            reference_disagreement_tol=args.reference_disagreement_tol,
            out_dir=args.out_dir,
        )
        records = run_robustness_benchmark(config)
        _, csv_path, figures = save_robustness_outputs(records, out_dir=config.out_dir, generate_plots=not args.skip_plots)
        _write_summary(
            config.out_dir,
            [{"title": "Robustness", "body": _robustness_body(config, csv_path), "csv": csv_path, "figures": figures}],
        )
        return

    if args.command == "all":
        from .accuracy import AccuracyConfig, run_accuracy_benchmark, save_accuracy_outputs
        from .robustness import RobustnessConfig, run_robustness_benchmark, save_robustness_outputs
        from .runtime import RuntimeConfig, run_runtime_benchmark, save_runtime_outputs

        sections = []
        shared_out_dir = args.out_dir
        accuracy_methods = list(DEFAULT_SPCAUCHY_ACCURACY_METHODS)
        if args.include_auto:
            accuracy_methods.append("auto")
        accuracy_config = AccuracyConfig(
            dims=args.dims or list(DEFAULT_ACCURACY_DIMS),
            rho_grid=args.rho_grid or list(DEFAULT_SPCAUCHY_RHO_GRID),
            methods=accuracy_methods,
            device_name="cpu",
            dtype=torch.float64,
            seed=args.seed,
            quadrature_nodes=args.quadrature_nodes,
            reference_nodes=args.reference_nodes,
            series_k_terms=args.series_k_terms,
            reference_k_terms=args.reference_k_terms,
            reference_disagreement_tol=args.reference_disagreement_tol,
            out_dir=shared_out_dir,
        )
        accuracy_records = run_accuracy_benchmark(accuracy_config)
        _, accuracy_csv, accuracy_figures = save_accuracy_outputs(
            accuracy_records,
            out_dir=shared_out_dir,
            generate_plots=not args.skip_plots,
        )
        sections.append({"title": "KL Accuracy", "body": _accuracy_body(accuracy_config, accuracy_csv), "csv": accuracy_csv, "figures": accuracy_figures})

        runtime_methods = ["spcauchy_combined", "spcauchy_hybrid"]
        if args.include_series:
            runtime_methods.append("spcauchy_series")
        if args.include_auto:
            runtime_methods.append("spcauchy_auto")
        runtime_records = []
        runtime_devices = ["cpu"]
        if torch.cuda.is_available():
            runtime_devices.append("cuda")
        for runtime_device in runtime_devices:
            runtime_config = RuntimeConfig(
                dims=args.dims or list(DEFAULT_RUNTIME_DIMS),
                spcauchy_methods=runtime_methods,
                vmf_methods=["vmf_official", "vmf_robust"],
                device_name=runtime_device,
                dtype=_runtime_or_robustness_dtype_from_args(args),
                seed=args.seed,
                batch_size=args.batch_size,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                timeout_s=args.timeout_s,
                quadrature_nodes=args.quadrature_nodes,
                series_k_terms=args.series_k_terms,
                kappa=args.kappa,
                concentration_mode="matched",
                out_dir=shared_out_dir,
            )
            runtime_records.extend(run_runtime_benchmark(runtime_config))
        _, runtime_csv, runtime_figures = save_runtime_outputs(
            runtime_records,
            out_dir=shared_out_dir,
            generate_plots=not args.skip_plots,
        )
        sections.append({"title": "Runtime", "body": f"Ran CPU runtime sweeps and {'CUDA + CPU' if torch.cuda.is_available() else 'CPU-only'} latent-step sweeps. Results were written to `{_display_path(runtime_csv)}`.", "csv": runtime_csv, "figures": runtime_figures})

        robustness_methods = ["series", "combined", "asymptotic_high_rho", "hybrid"]
        if args.include_auto:
            robustness_methods.append("auto")
        robustness_config = RobustnessConfig(
            spcauchy_dims=args.dims or list(DEFAULT_ACCURACY_DIMS),
            vmf_dims=args.dims or list(DEFAULT_RUNTIME_DIMS),
            rho_grid=args.rho_grid or list(DEFAULT_SPCAUCHY_RHO_GRID),
            kappa_grid=args.kappa_grid or list(DEFAULT_VMF_KAPPA_GRID),
            spcauchy_methods=robustness_methods,
            vmf_methods=["vmf_official", "vmf_robust"],
            device_name="cpu",
            dtype=_runtime_or_robustness_dtype_from_args(args),
            seed=args.seed,
            batch_size=args.batch_size or 32,
            timeout_s=args.timeout_s,
            quadrature_nodes=args.quadrature_nodes,
            series_k_terms=args.series_k_terms,
            reference_nodes=args.reference_nodes,
            reference_disagreement_tol=args.reference_disagreement_tol,
            out_dir=shared_out_dir,
        )
        robustness_records = run_robustness_benchmark(robustness_config)
        _, robustness_csv, robustness_figures = save_robustness_outputs(
            robustness_records,
            out_dir=shared_out_dir,
            generate_plots=not args.skip_plots,
        )
        body = _robustness_body(robustness_config, robustness_csv)
        if not torch.cuda.is_available():
            body += " CUDA was not available, so GPU-specific plots were skipped."
        sections.append({"title": "Robustness", "body": body, "csv": robustness_csv, "figures": robustness_figures})

        _write_summary(shared_out_dir, sections)
        return


if __name__ == "__main__":
    main()
