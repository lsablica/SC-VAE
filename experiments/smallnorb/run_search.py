"""Sequential validation-only spherical Cauchy setup search."""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
from pathlib import Path
from typing import Any

from .config import SEARCH_ROOT, RunConfig
from .evaluate import evaluate_run
from .interpolate import interpolation_run
from .probe import probe_run
from .train import train_run
from .utils import (
    read_json,
    repo_relative,
    resolve_device,
    write_csv,
    write_json,
)


CANDIDATES: dict[str, dict[str, Any]] = {
    "baseline_b050_w20_s020": {
        "beta_target": 0.5,
        "beta_warmup_epochs": 20,
        "learning_rate": 2e-4,
        "sigma_x": 0.20,
        "concentration_learning_rate_multiplier": 1.0,
        "diagnostic_reason": "Baseline configuration.",
    },
    "lower_beta_b025_w30_s020": {
        "beta_target": 0.25,
        "beta_warmup_epochs": 30,
        "learning_rate": 2e-4,
        "sigma_x": 0.20,
        "concentration_learning_rate_multiplier": 1.0,
        "diagnostic_reason": (
            "Use only if the default collapses or loses pose information."
        ),
    },
    "lower_beta_slow_lr_b025_w30_s020": {
        "beta_target": 0.25,
        "beta_warmup_epochs": 30,
        "learning_rate": 1e-4,
        "sigma_x": 0.20,
        "concentration_learning_rate_multiplier": 1.0,
        "diagnostic_reason": (
            "Follow-up after the slower schedule improved gap "
            "reconstruction but retained weak pose geometry; test whether "
            "more retained information improves both."
        ),
    },
    "deep_cnn_b025_w30_s020": {
        "beta_target": 0.25,
        "beta_warmup_epochs": 30,
        "learning_rate": 1e-4,
        "sigma_x": 0.20,
        "concentration_learning_rate_multiplier": 1.0,
        "architecture": "deep_residual_cnn",
        "diagnostic_reason": (
            "Shared deeper residual encoder and decoder pilot after the "
            "baseline CNN produced strong reconstruction but weak absolute "
            "viewpoint geometry."
        ),
    },
    "slower_lr_b050_w30_s020": {
        "beta_target": 0.5,
        "beta_warmup_epochs": 30,
        "learning_rate": 1e-4,
        "sigma_x": 0.20,
        "concentration_learning_rate_multiplier": 1.0,
        "diagnostic_reason": (
            "Test a gentler optimizer and longer information warmup."
        ),
    },
    "sharper_likelihood_b050_w20_s015": {
        "beta_target": 0.5,
        "beta_warmup_epochs": 20,
        "learning_rate": 2e-4,
        "sigma_x": 0.15,
        "concentration_learning_rate_multiplier": 1.0,
        "diagnostic_reason": (
            "Use only if KL is healthy but reconstructions remain blurry."
        ),
    },
    "stronger_beta_b100_w20_s020": {
        "beta_target": 1.0,
        "beta_warmup_epochs": 20,
        "learning_rate": 2e-4,
        "sigma_x": 0.20,
        "concentration_learning_rate_multiplier": 0.5,
        "diagnostic_reason": (
            "Use only if concentration remains pinned near one."
        ),
    },
}


def candidate_config(
    name: str,
    *,
    epochs: int = 40,
    batch_size: int = 128,
    num_workers: int = 4,
) -> RunConfig:
    if name not in CANDIDATES:
        raise KeyError(name)
    values = dict(CANDIDATES[name])
    diagnostic_reason = values.pop("diagnostic_reason")
    return RunConfig(
        family="spcauchy",
        seed=0,
        stage="search",
        run_name=name,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        notes=diagnostic_reason,
        tags=("smallnorb", "stage1", "validation-only"),
        **values,
    )


def summarize_candidate(config: RunConfig) -> dict[str, Any]:
    run_dir = config.run_dir
    selection = read_json(run_dir / "selection_summary.json")
    evaluation = read_json(run_dir / "evaluation_summary.json")
    probe = read_json(run_dir / "probe_summary.json")
    interpolation = read_json(
        run_dir / "interpolation_summary_validation.json"
    )
    history = read_json(run_dir / "history.json")
    validation = evaluation["validation"]
    all_metrics = validation["validation"]
    observed = validation["validation_observed"]
    gap = validation["validation_gap"]
    last = history[-1]["validation"]["validation"]
    warmup_epoch = min(
        config.beta_zero_epochs + config.beta_warmup_epochs,
        len(history),
    )
    warmup_start_epoch = min(config.beta_zero_epochs + 1, len(history))
    warmup_start_gap = history[warmup_start_epoch - 1]["validation"][
        "validation_gap"
    ]["reconstruction_nll"]["mean"]
    post_warmup_gaps = [
        row["validation"]["validation_gap"]["reconstruction_nll"][
            "mean"
        ]
        for row in history[warmup_epoch - 1 :]
    ]
    best_post_warmup_gap = min(post_warmup_gaps)
    final_window_gap = statistics.median(post_warmup_gaps[-5:])
    kl_mean = float(last["kl"]["mean"])
    scale_median = float(last["posterior_scale"]["median"])
    gap_nll = float(gap["reconstruction_nll"]["mean"])
    observed_nll = float(observed["reconstruction_nll"]["mean"])
    pose_gap = float(
        probe["pose_probe"]["partitions"]["validation_gap"][
            "mean_absolute_error_degrees"
        ]
    )
    spearman = float(
        probe["geometry_alignment"]["validation"]["all_pairs"][
            "spearman"
        ]
    )
    checks = {
        "finite_completed": True,
        "kl_above_collapse_threshold": kl_mean >= 2.0,
        "median_rho_not_pinned": scale_median <= 0.9995,
        "reconstruction_improves_after_warmup": (
            best_post_warmup_gap < warmup_start_gap
            and final_window_gap <= 1.05 * best_post_warmup_gap
        ),
        "gap_not_catastrophic": gap_nll <= 1.5 * observed_nll,
        "pose_recoverable": pose_gap < 60.0,
        "positive_distance_alignment": spearman > 0.0,
    }
    healthy = all(
        checks[key]
        for key in (
            "finite_completed",
            "kl_above_collapse_threshold",
            "median_rho_not_pinned",
            "reconstruction_improves_after_warmup",
            "gap_not_catastrophic",
        )
    )
    return {
        "candidate": config.run_name,
        "run_dir": repo_relative(run_dir),
        "config": config.to_dict(),
        "selected_epoch": selection["selected_epoch"],
        "validation_reconstruction_nll": float(
            all_metrics["reconstruction_nll"]["mean"]
        ),
        "validation_observed_reconstruction_nll": observed_nll,
        "validation_gap_reconstruction_nll": gap_nll,
        "validation_gap_pixel_mse": float(gap["pixel_mse"]["mean"]),
        "validation_kl_mean_last_epoch": kl_mean,
        "validation_rho_median_last_epoch": scale_median,
        "validation_pose_gap_mean_error_degrees": pose_gap,
        "validation_distance_spearman": spearman,
        "validation_cross_gap_distance_spearman": float(
            probe["geometry_alignment"]["validation"][
                "pairs_crossing_gap"
            ]["spearman"]
        ),
        "validation_interpolation_interior_mse": float(
            interpolation["summaries"]["interior_gap"]["pixel_mse"][
                "mean"
            ]
        ),
        "warmup_start_gap_reconstruction_nll": float(
            warmup_start_gap
        ),
        "best_post_warmup_gap_reconstruction_nll": float(
            best_post_warmup_gap
        ),
        "final_five_epoch_median_gap_reconstruction_nll": float(
            final_window_gap
        ),
        "post_warmup_stability_tolerance": 0.05,
        "checks": checks,
        "healthy": healthy,
        "official_test_accessed": False,
    }


def _candidate_summary_paths() -> list[Path]:
    return sorted(SEARCH_ROOT.glob("*/candidate_summary.json"))


def resummarize_completed_candidates() -> list[dict[str, Any]]:
    """Rebuild summaries from retained logs without rerunning training."""

    rows = []
    for config_path in sorted(
        SEARCH_ROOT.glob("*/spcauchy/seed_0/config.json")
    ):
        run_dir = config_path.parent
        required = (
            "selection_summary.json",
            "evaluation_summary.json",
            "probe_summary.json",
            "interpolation_summary_validation.json",
            "history.json",
        )
        if not all((run_dir / name).exists() for name in required):
            continue
        config = RunConfig.from_dict(read_json(config_path))
        summary = summarize_candidate(config)
        write_json(
            SEARCH_ROOT
            / config.run_name
            / "candidate_summary.json",
            summary,
        )
        rows.append(summary)
    refresh_search_report()
    return rows


def refresh_search_report() -> list[dict[str, Any]]:
    rows = [read_json(path) for path in _candidate_summary_paths()]
    selected_name = None
    if rows:
        try:
            selected_name = select_candidate(rows)["candidate"]
        except RuntimeError:
            selected_name = None
    flat_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"config", "checks"}
        }
        | {
            f"check_{key}": value
            for key, value in row["checks"].items()
        }
        for row in rows
    ]
    write_json(SEARCH_ROOT / "search_candidates.json", rows)
    write_csv(SEARCH_ROOT / "search_candidates.csv", flat_rows)
    lines = [
        "# smallNORB validation-only setup search",
        "",
        "No official test image or test metric was accessed during setup selection.",
        "",
        "| Candidate | Numeric health | Decision | Gap NLL | Observed NLL | KL | Median rho | Pose gap error | Distance Spearman | Cross-gap Spearman | Interpolation MSE |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    best_gap = min(
        (
            row["validation_gap_reconstruction_nll"]
            for row in rows
            if row["healthy"]
        ),
        default=float("inf"),
    )
    for row in rows:
        if row["candidate"] == selected_name:
            decision = "accepted"
        elif not row["healthy"]:
            failed = [
                key
                for key, value in row["checks"].items()
                if not value
            ]
            decision = "rejected: " + ", ".join(failed)
        else:
            degradation = (
                row["validation_gap_reconstruction_nll"] / best_gap - 1.0
            )
            decision = f"rejected: gap +{100 * degradation:.1f}%"
        lines.append(
            "| {candidate} | {healthy} | {decision} | {gap:.4f} | "
            "{observed:.4f} | {kl:.3f} | {rho:.6f} | {pose:.2f} | "
            "{spearman:.3f} | {cross:.3f} | {interp:.6f} |".format(
                candidate=row["candidate"],
                healthy=row["healthy"],
                decision=decision,
                gap=row["validation_gap_reconstruction_nll"],
                observed=row[
                    "validation_observed_reconstruction_nll"
                ],
                kl=row["validation_kl_mean_last_epoch"],
                rho=row["validation_rho_median_last_epoch"],
                pose=row[
                    "validation_pose_gap_mean_error_degrees"
                ],
                spearman=row["validation_distance_spearman"],
                cross=row[
                    "validation_cross_gap_distance_spearman"
                ],
                interp=row[
                    "validation_interpolation_interior_mse"
                ],
            )
        )
    lines.extend(
        [
            "",
            "Numeric health requires finite completion, KL above 2 nats, "
            "median rho below 0.9995, non-catastrophic gap reconstruction, "
            "improvement from the start of KL warmup, and no more than five "
            "percent regression from the best post-warmup gap NLL in the final "
            "five-epoch median.",
            "",
            "The setup is chosen lexicographically from numerically healthy "
            "runs using gap reconstruction, observed reconstruction among "
            "candidates within two percent of the best gap result, pose "
            "geometry, then schedule simplicity.",
            "",
            "No candidate made absolute held-out-gap pose linearly recoverable. "
            "This limitation is retained explicitly; the Stage 2 comparative "
            "geometry and interpolation gate determines whether a full sweep "
            "is justified.",
            "",
        ]
    )
    (SEARCH_ROOT / "SEARCH_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return rows


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    healthy = [row for row in rows if row["healthy"]]
    if not healthy:
        raise RuntimeError("No healthy spherical Cauchy candidate is available")
    best_gap = min(
        row["validation_gap_reconstruction_nll"] for row in healthy
    )
    close = [
        row
        for row in healthy
        if row["validation_gap_reconstruction_nll"] <= 1.02 * best_gap
    ]
    return min(
        close,
        key=lambda row: (
            row["validation_observed_reconstruction_nll"],
            row["validation_pose_gap_mean_error_degrees"],
            -row["validation_distance_spearman"],
            row["config"]["beta_warmup_epochs"],
            abs(row["config"]["learning_rate"] - 2e-4),
        ),
    )


def freeze_best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = select_candidate(rows)
    frozen_config = dict(selected["config"])
    frozen_config.update(
        {
            "stage": "final",
            "run_name": "frozen_shared_setup",
            "epochs": 100,
            "train_limit": None,
            "validation_limit": None,
            "test_limit": None,
            "notes": (
                "Shared setup frozen using validation data only after Stage 1."
            ),
            "tags": ["smallnorb", "frozen", "five-seed"],
        }
    )
    payload = {
        "schema_version": 1,
        "selected_candidate": selected["candidate"],
        "selection_rule": (
            "healthy first; lowest validation-gap NLL; among candidates within "
            "two percent prefer observed reconstruction, pose error, distance "
            "alignment, then the simpler schedule"
        ),
        "evidence": selected,
        "shared_config": frozen_config,
        "official_test_accessed": False,
    }
    write_json(SEARCH_ROOT / "frozen_setup.json", payload)
    return payload


def run_candidate(
    name: str,
    device_name: str,
    *,
    epochs: int,
    force: bool,
    resume: bool,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    config = candidate_config(name, epochs=epochs)
    command = (
        "python -m experiments.smallnorb.run_search "
        f"--candidate {name} --epochs {epochs} --device {device_name}"
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
    interpolation_run(config.run_dir, device, split="validation")
    summary = summarize_candidate(config)
    write_json(
        SEARCH_ROOT / name / "candidate_summary.json", summary
    )
    refresh_search_report()
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=tuple(CANDIDATES))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--freeze-best", action="store_true")
    parser.add_argument("--resummarize-completed", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.resummarize_completed:
        resummarize_completed_candidates()
    if args.candidate:
        summary = run_candidate(
            args.candidate,
            args.device,
            epochs=args.epochs,
            force=args.force,
            resume=args.resume,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    rows = refresh_search_report()
    if args.freeze_best:
        print(
            json.dumps(
                freeze_best_candidate(rows),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
