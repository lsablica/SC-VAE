"""Run the full repeated Plan B grid and aggregate it."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from experiments.smallnorb.utils import (
    ensure_dir,
    write_commands,
    write_environment,
)

from .aggregate import aggregate
from .fit import FitConfig, fit_one


FAMILIES = ("spcauchy", "vmf", "powerspherical")
OBJECTIVES = ("forward_kl", "reverse_kl")
KAPPAS = (20, 100, 500)
EPSILONS = (0.0, 0.01, 0.05, 0.10, 0.20)


def run(args: argparse.Namespace) -> None:
    root = ensure_dir(args.output)
    exact_arguments = getattr(sys, "orig_argv", ["python", *sys.argv])
    command = " ".join(
        shlex.quote(value) for value in exact_arguments
    )
    canonical = (
        "python -m experiments.contaminated_directional.run_all "
        f"--output {shlex.quote(str(args.output))} "
        f"--device {args.device} --seeds {args.seeds} "
        f"--steps {args.steps} --batch-size {args.batch_size} "
        f"--evaluation-samples {args.evaluation_samples}"
    )
    write_commands(root / "commands.txt", [command, canonical])
    write_environment(root / "environment.txt")
    for kappa in KAPPAS:
        for epsilon in EPSILONS:
            for objective in OBJECTIVES:
                for family in FAMILIES:
                    for seed in range(args.seeds):
                        output = (
                            root
                            / f"kappa_{kappa}"
                            / f"epsilon_{epsilon:.2f}"
                            / objective
                            / family
                            / f"seed_{seed}"
                        )
                        summary = output / "evaluation_summary.json"
                        if summary.exists() and not args.rerun:
                            continue
                        config = FitConfig(
                            family=family,
                            objective=objective,
                            kappa=kappa,
                            epsilon=epsilon,
                            seed=seed,
                            steps=args.steps,
                            batch_size=args.batch_size,
                            evaluation_samples=args.evaluation_samples,
                            initial_curvature=float(kappa),
                            device=args.device,
                        )
                        result = fit_one(config, output)
                        print(
                            json.dumps(
                                {
                                    "family": family,
                                    "objective": objective,
                                    "kappa": kappa,
                                    "epsilon": epsilon,
                                    "seed": seed,
                                    "forward_kl": result["forward_kl"],
                                    "reverse_kl": result["reverse_kl"],
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
    aggregate(root)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/contaminated_directional/final"),
    )
    result.add_argument("--device", default="cuda")
    result.add_argument("--seeds", type=int, default=3)
    result.add_argument("--steps", type=int, default=600)
    result.add_argument("--batch-size", type=int, default=8192)
    result.add_argument("--evaluation-samples", type=int, default=1_000_000)
    result.add_argument("--rerun", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
