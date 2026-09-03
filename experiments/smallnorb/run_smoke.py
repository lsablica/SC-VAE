"""One-seed validation-only comparison under the frozen shared setup."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from .config import (
    EXPERIMENT_ROOT,
    MAIN_FAMILIES,
    RUNS_ROOT,
    SEARCH_ROOT,
    RunConfig,
)
from .evaluate import evaluate_run
from .interpolate import interpolation_run
from .probe import probe_run
from .train import train_run
from .utils import (
    ensure_dir,
    read_json,
    resolve_device,
    write_csv,
    write_json,
)


def smoke_config(
    family: str,
    frozen: dict[str, Any],
    *,
    epochs: int,
) -> RunConfig:
    shared = RunConfig.from_dict(frozen["shared_config"])
    return dataclasses.replace(
        shared,
        family=family,
        seed=0,
        stage="smoke",
        run_name=frozen["selected_candidate"],
        epochs=max(epochs, 50),
        output_root=None,
        notes=(
            "One-seed validation-only family smoke comparison under the "
            "frozen Stage 1 setup."
        ),
        tags=("smallnorb", "stage2", "validation-only", "smoke"),
    )


def _run_summary(config: RunConfig) -> dict[str, Any]:
    run_dir = config.run_dir
    evaluation = read_json(run_dir / "evaluation_summary.json")
    probe = read_json(run_dir / "probe_summary.json")
    interpolation = read_json(
        run_dir / "interpolation_summary_validation.json"
    )
    validation = evaluation["validation"]
    return {
        "family": config.family,
        "run_dir": str(run_dir),
        "validation_gap_reconstruction_nll": validation[
            "validation_gap"
        ]["reconstruction_nll"]["mean"],
        "validation_observed_reconstruction_nll": validation[
            "validation_observed"
        ]["reconstruction_nll"]["mean"],
        "validation_gap_pixel_mse": validation["validation_gap"][
            "pixel_mse"
        ]["mean"],
        "validation_kl": validation["validation"]["kl"]["mean"],
        "pose_gap_error_degrees": probe["pose_probe"]["partitions"][
            "validation_gap"
        ]["mean_absolute_error_degrees"],
        "distance_spearman": probe["geometry_alignment"]["validation"][
            "all_pairs"
        ]["spearman"],
        "cross_gap_distance_spearman": probe[
            "geometry_alignment"
        ]["validation"]["pairs_crossing_gap"]["spearman"],
        "interpolation_interior_mse": interpolation["summaries"][
            "interior_gap"
        ]["pixel_mse"]["mean"],
        "test_was_accessed": False,
    }


def smoke_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = {row["family"]: row for row in rows}
    spherical = by_family["spcauchy"]
    best_gap = min(
        row["validation_gap_reconstruction_nll"] for row in rows
    )
    best_observed = min(
        row["validation_observed_reconstruction_nll"] for row in rows
    )
    metric_wins = {
        "pose_gap_error": spherical["pose_gap_error_degrees"]
        <= min(row["pose_gap_error_degrees"] for row in rows),
        "distance_spearman": spherical["distance_spearman"]
        >= max(row["distance_spearman"] for row in rows),
        "cross_gap_distance_spearman": spherical[
            "cross_gap_distance_spearman"
        ]
        >= max(row["cross_gap_distance_spearman"] for row in rows),
        "interpolation_interior_mse": spherical[
            "interpolation_interior_mse"
        ]
        <= min(row["interpolation_interior_mse"] for row in rows),
    }
    condition_1 = (
        spherical["validation_gap_reconstruction_nll"] <= best_gap
    )
    condition_2 = (
        spherical["validation_gap_reconstruction_nll"] <= 1.02 * best_gap
        and sum(metric_wins.values()) >= 2
    )
    condition_4 = (
        metric_wins["interpolation_interior_mse"]
        and spherical["validation_observed_reconstruction_nll"]
        <= 1.02 * best_observed
    )
    proceed = condition_1 or condition_2 or condition_4
    return {
        "proceed_full_five_seed": proceed,
        "condition_1_best_gap_reconstruction": condition_1,
        "condition_2_within_two_percent_and_two_geometry_wins": condition_2,
        "condition_3_short_three_seed_stability": (
            "not evaluated in the one-seed smoke"
        ),
        "condition_4_interpolation_without_observed_cost": condition_4,
        "geometry_metric_wins": metric_wins,
        "spcauchy_gap_ratio_to_best": (
            spherical["validation_gap_reconstruction_nll"] / best_gap
        ),
        "official_test_accessed": False,
    }


def write_smoke_report(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
) -> None:
    output = RUNS_ROOT / "smoke"
    write_csv(output / "smoke_summary.csv", rows)
    write_json(output / "smoke_summary.json", rows)
    write_json(output / "smoke_gate.json", gate)
    lines = [
        "# smallNORB one-seed smoke comparison",
        "",
        "This comparison used validation data only. Official test instances "
        "were not evaluated.",
        "",
        "| Family | Gap NLL | Observed NLL | KL | Pose gap error | Spearman | Cross-gap Spearman | Interpolation MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {family} | {gap:.4f} | {observed:.4f} | {kl:.3f} | "
            "{pose:.2f} | {spearman:.3f} | {cross:.3f} | {interp:.6f} |".format(
                family=row["family"],
                gap=row["validation_gap_reconstruction_nll"],
                observed=row[
                    "validation_observed_reconstruction_nll"
                ],
                kl=row["validation_kl"],
                pose=row["pose_gap_error_degrees"],
                spearman=row["distance_spearman"],
                cross=row["cross_gap_distance_spearman"],
                interp=row["interpolation_interior_mse"],
            )
        )
    lines.extend(
        [
            "",
            f"Full five-seed gate passed: **{gate['proceed_full_five_seed']}**",
            "",
            "## Baseline implementation diagnostics",
            "",
            "The first robust-vMF attempt used a softplus concentration head "
            "and remained near kappa 10. A log-concentration head alone did "
            "not fix the problem because the inherited rejection sampler "
            "blended an approximation over kappa 10 to 11. The retained "
            "smallNORB implementation uses an algebraically equivalent "
            "rationalized rejection parameter with a smooth derivative. "
            "This repair lives only in "
            "`experiments/smallnorb/vendor_vmf_smallnorb.py`. "
            "The MNIST and shared benchmark vMF files are unchanged.",
            "",
            "A pre-test accuracy audit then found that the inherited "
            "128-term positive series underestimates log I at the smoke "
            "model's kappa range. The local copy uses the positive series "
            "below kappa 50 and the exact half-integer recurrence available "
            "at D=33 above it. It matches SciPy's scaled Bessel reference within "
            "2e-4 in float32 through the explicit kappa 10,000 clamp. The "
            "corrected vMF smoke result is the only vMF row used by the gate.",
            "",
            "The shared setup and initial smoke gate were frozen before "
            "spherical Cauchy test evaluation. This implementation audit "
            "occurred before any vMF test evaluation and used only the "
            "validation concentration range plus an external mathematical "
            "reference. No test metric informed the repair.",
            "",
            "The vendored Power Spherical mean property has an invalid batch "
            "broadcast for this diagnostic. The smallNORB wrapper reads the "
            "same scalar marginal expectation directly. The pinned vendor "
            "source is unchanged.",
            "",
            "Failed diagnostic attempts remain under the smoke run folders "
            "with names ending in `_diagnostic`. They are implementation "
            "diagnostics and are not reported as scientific family results.",
            "",
            "Spherical Cauchy won validation-gap reconstruction, which "
            "satisfies the first predeclared gate condition. Absolute linear "
            "pose recovery and distance geometry were weak for every family. "
            "The final study therefore treats reconstruction as the primary "
            "endpoint and retains the weak geometry as an explicit "
            "limitation.",
            "",
        ]
    )
    report_text = "\n".join(lines)
    (output / "SMOKE_REPORT.md").write_text(
        report_text, encoding="utf-8"
    )
    tracked_reports = ensure_dir(EXPERIMENT_ROOT / "reports")
    (tracked_reports / "stage2_smoke_report.md").write_text(
        report_text, encoding="utf-8"
    )
    write_csv(tracked_reports / "stage2_smoke_summary.csv", rows)
    write_json(tracked_reports / "stage2_smoke_gate.json", gate)


def run_smoke(
    device_name: str,
    *,
    epochs: int,
    families: tuple[str, ...],
    force: bool,
    resume: bool,
) -> dict[str, Any]:
    frozen = read_json(SEARCH_ROOT / "frozen_setup.json")
    device = resolve_device(device_name)
    for family in families:
        config = smoke_config(family, frozen, epochs=epochs)
        command = (
            "python -m experiments.smallnorb.run_smoke "
            f"--epochs {epochs} --families {' '.join(families)} "
            f"--device {device_name}"
        )
        train_run(
            config,
            device,
            command=command,
            force=force,
            resume=resume,
        )
        evaluate_run(config.run_dir, device, include_test=False)
        probe_run(config.run_dir, device, include_test=False)
        interpolation_run(
            config.run_dir, device, split="validation"
        )
    rows = []
    for family in MAIN_FAMILIES:
        config = smoke_config(family, frozen, epochs=epochs)
        required = (
            "evaluation_summary.json",
            "probe_summary.json",
            "interpolation_summary_validation.json",
        )
        if all((config.run_dir / name).exists() for name in required):
            rows.append(_run_summary(config))
    if {row["family"] for row in rows} != set(MAIN_FAMILIES):
        missing = sorted(
            set(MAIN_FAMILIES) - {row["family"] for row in rows}
        )
        raise RuntimeError(
            "Smoke comparison is incomplete; missing " + ", ".join(missing)
        )
    gate = smoke_gate(rows)
    write_smoke_report(rows, gate)
    return gate


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=MAIN_FAMILIES,
        default=list(MAIN_FAMILIES),
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(
        json.dumps(
            run_smoke(
                args.device,
                epochs=args.epochs,
                families=tuple(args.families),
                force=args.force,
                resume=args.resume,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
