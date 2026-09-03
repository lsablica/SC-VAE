"""Run the final four-family MNIST benchmark and aggregate its results."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.mnist.aggregate import aggregate_benchmark_outputs
from experiments.mnist.config import BENCHMARK_PRESET, DEFAULT_DATA_DIR, OUTPUT_ROOT
from experiments.mnist.evaluate import run_evaluation_jobs
from experiments.mnist.train import run_training_jobs
from experiments.mnist.utils import resolve_device


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("gaussian", "vmf", "spcauchy", "powerspherical"),
    )
    parser.add_argument("--reported-dims", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)
    output_root = Path(args.output_root)
    shared = {
        "preset": BENCHMARK_PRESET,
        "data_dir": args.data_dir,
        "output_root": output_root,
        "device": device,
        "models": args.models,
        "reported_dims": args.reported_dims,
        "seeds": args.seeds,
        "epochs": args.epochs,
    }
    print("Training benchmark runs...")
    print(run_training_jobs(force=args.force, **shared))
    print("Evaluating benchmark runs...")
    print(run_evaluation_jobs(**shared))
    print("Aggregating benchmark outputs...")
    print(aggregate_benchmark_outputs(output_root))


if __name__ == "__main__":
    main()
