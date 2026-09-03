"""Sequential five-seed runner guarded by the validation-only smoke gate."""

from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Any

from .config import (
    MAIN_FAMILIES,
    RUNS_ROOT,
    SEARCH_ROOT,
    SEEDS,
    RunConfig,
)
from .evaluate import evaluate_run
from .interpolate import interpolation_run
from .probe import probe_run
from .train import train_run
from .utils import read_json, resolve_device, write_json


def final_config(
    family: str,
    seed: int,
    frozen: dict[str, Any],
) -> RunConfig:
    shared = RunConfig.from_dict(frozen["shared_config"])
    return dataclasses.replace(
        shared,
        family=family,
        seed=seed,
        stage="final",
        run_name=frozen["selected_candidate"],
        output_root=None,
        notes=(
            "Five-seed locked smallNORB comparison. Test evaluation occurs "
            "only after the validation-only smoke gate."
        ),
        tags=("smallnorb", "stage3", "locked", "five-seed"),
    )


def run_all(
    device_name: str,
    *,
    families: tuple[str, ...],
    seeds: tuple[int, ...],
    force: bool,
    resume: bool,
) -> list[dict[str, Any]]:
    gate = read_json(RUNS_ROOT / "smoke" / "smoke_gate.json")
    if not gate["proceed_full_five_seed"]:
        raise RuntimeError(
            "The validation-only smoke gate did not authorize test access"
        )
    frozen = read_json(SEARCH_ROOT / "frozen_setup.json")
    device = resolve_device(device_name)
    completed = []
    for family in families:
        for seed in seeds:
            config = final_config(family, seed, frozen)
            command = (
                "python -m experiments.smallnorb.run_all "
                f"--families {' '.join(families)} "
                f"--seeds {' '.join(str(value) for value in seeds)} "
                f"--device {device_name}"
            )
            train_run(
                config,
                device,
                command=command,
                force=force,
                resume=resume,
            )
            evaluation = evaluate_run(
                config.run_dir, device, include_test=True
            )
            probe = probe_run(
                config.run_dir, device, include_test=True
            )
            interpolation = interpolation_run(
                config.run_dir, device, split="test"
            )
            completed.append(
                {
                    "family": family,
                    "seed": seed,
                    "run_dir": str(config.run_dir),
                    "test_gap_reconstruction_nll": evaluation["test"][
                        "test_gap"
                    ]["reconstruction_nll"]["mean"],
                    "test_gap_pose_error_degrees": probe["pose_probe"][
                        "partitions"
                    ]["test_gap"]["mean_absolute_error_degrees"],
                    "test_interpolation_interior_mse": interpolation[
                        "summaries"
                    ]["interior_gap"]["pixel_mse"]["mean"],
                }
            )
            write_json(
                RUNS_ROOT / "final_completed_runs.json", completed
            )
    return completed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families",
        nargs="+",
        choices=MAIN_FAMILIES,
        default=list(MAIN_FAMILIES),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(SEEDS)
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    completed = run_all(
        args.device,
        families=tuple(args.families),
        seeds=tuple(args.seeds),
        force=args.force,
        resume=args.resume,
    )
    print(json.dumps(completed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
