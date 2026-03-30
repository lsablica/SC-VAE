from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mnist_experiment.aggregate import aggregate_benchmark_outputs  # noqa: E402
from experiments.mnist_experiment.config import (  # noqa: E402
    BENCHMARK_PRESET,
    OUTPUT_ROOT,
    QUALITATIVE_PRESET,
    DEFAULT_DATA_DIR,
)
from experiments.mnist_experiment.evaluate import run_evaluation_jobs  # noqa: E402
from experiments.mnist_experiment.plotting import run_plot_jobs  # noqa: E402
from experiments.mnist_experiment.train import run_training_jobs  # noqa: E402
from experiments.mnist_experiment.utils import resolve_device  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full MNIST paper reproduction pipeline.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--benchmark-models", nargs="+", choices=["gaussian", "vmf", "spcauchy"])
    parser.add_argument("--benchmark-reported-dims", nargs="+", type=int)
    parser.add_argument("--benchmark-seeds", nargs="+", type=int)
    parser.add_argument("--benchmark-epochs", type=int)
    parser.add_argument("--qualitative-seeds", nargs="+", type=int)
    parser.add_argument("--qualitative-epochs", type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)
    output_root = Path(args.output_root)

    print("Training benchmark runs...")
    print(
        run_training_jobs(
            preset=BENCHMARK_PRESET,
            data_dir=args.data_dir,
            output_root=output_root,
            device=device,
            models=args.benchmark_models,
            reported_dims=args.benchmark_reported_dims,
            seeds=args.benchmark_seeds,
            epochs=args.benchmark_epochs,
            force=args.force,
        )
    )

    print("Training qualitative run...")
    print(
        run_training_jobs(
            preset=QUALITATIVE_PRESET,
            data_dir=args.data_dir,
            output_root=output_root,
            device=device,
            seeds=args.qualitative_seeds,
            epochs=args.qualitative_epochs,
            force=args.force,
        )
    )

    print("Evaluating benchmark runs...")
    print(
        run_evaluation_jobs(
            preset=BENCHMARK_PRESET,
            data_dir=args.data_dir,
            output_root=output_root,
            device=device,
            models=args.benchmark_models,
            reported_dims=args.benchmark_reported_dims,
            seeds=args.benchmark_seeds,
            epochs=args.benchmark_epochs,
        )
    )

    print("Evaluating qualitative run...")
    print(
        run_evaluation_jobs(
            preset=QUALITATIVE_PRESET,
            data_dir=args.data_dir,
            output_root=output_root,
            device=device,
            seeds=args.qualitative_seeds,
            epochs=args.qualitative_epochs,
        )
    )

    print("Generating benchmark plots...")
    print(
        run_plot_jobs(
            preset=BENCHMARK_PRESET,
            data_dir=args.data_dir,
            output_root=output_root,
            device=device,
            models=args.benchmark_models,
            reported_dims=args.benchmark_reported_dims,
            seeds=args.benchmark_seeds,
            epochs=args.benchmark_epochs,
        )
    )

    print("Generating qualitative plots...")
    print(
        run_plot_jobs(
            preset=QUALITATIVE_PRESET,
            data_dir=args.data_dir,
            output_root=output_root,
            device=device,
            seeds=args.qualitative_seeds,
            epochs=args.qualitative_epochs,
        )
    )

    print("Aggregating benchmark outputs...")
    print(aggregate_benchmark_outputs(output_root))


if __name__ == "__main__":
    main()
