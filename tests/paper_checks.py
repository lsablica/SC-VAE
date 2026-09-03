"""Checks for final artifacts and paper values."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "paper_results.json"

FINAL_DIRS = (
    Path("experiments/latent_layer/final"),
    Path("experiments/mnist/final"),
    Path("experiments/contaminated_directional/final"),
    Path("experiments/smallnorb/final"),
)
GENERATED_MANIFEST_NAMES = {"artifact_checksums.sha256", "manifest.json"}

RESULT_PATHS = {
    "direct": (
        "experiments/latent_layer/final/results/"
        "direct_kl_accuracy_summary.csv"
    ),
    "neighbor": (
        "experiments/latent_layer/final/results/"
        "neighbor_approximation_errors.csv"
    ),
    "runtime": (
        "experiments/latent_layer/final/results/"
        "latent_step_runtime_summary.csv"
    ),
    "mnist": "experiments/mnist/final/summary.csv",
    "contamination": (
        "experiments/contaminated_directional/final/tables/aggregate.csv"
    ),
    "smallnorb": "experiments/smallnorb/final/tables/aggregate.csv",
    "smallnorb_paired": (
        "experiments/smallnorb/final/tables/"
        "paired_statistics.csv"
    ),
    "pairwise": "tests/fixtures/pairwise_kl_validation.txt",
}

MNIST_FIELDS = [
    "model_family",
    "reported_dim",
    "ambient_latent_dim",
    "num_completed_seeds",
    "num_failed_seeds",
    "nonfinite_run_count",
    "best_eval_recon_loss_mean",
    "best_eval_recon_loss_std",
    "best_eval_recon_loss_min",
    "best_eval_recon_loss_max",
    "best_eval_total_loss_mean",
    "best_eval_total_loss_std",
    "best_eval_kl_mean",
    "best_eval_kl_std",
    "selected_epoch_mean",
    "selected_epoch_min",
    "selected_epoch_max",
    "wall_clock_training_s_mean",
    "bold_best_mean",
    "dagger_paired_bh",
    "paired_runner_up",
    "paired_num_seeds",
    "paired_improvement_mean",
    "paired_improvement_ci_low",
    "paired_improvement_ci_high",
    "paired_p_value",
    "paired_q_value",
]

SMALLNORB_FIELDS = [
    "family",
    "num_seeds",
    "seeds",
    "test_gap_reconstruction_nll_mean",
    "test_gap_reconstruction_nll_std",
    "test_gap_pixel_mse_mean",
    "test_gap_pixel_mse_std",
    "test_observed_reconstruction_nll_mean",
    "test_observed_reconstruction_nll_std",
    "pose_test_gap_mean_error_degrees_mean",
    "pose_test_gap_mean_error_degrees_std",
    "geometry_test_spearman_mean",
    "geometry_test_spearman_std",
    "interpolation_interior_pixel_mse_mean",
    "interpolation_interior_pixel_mse_std",
]

SMALLNORB_PAIR_FIELDS = [
    "competitor",
    "metric",
    "num_paired_seeds",
    "spcauchy_minus_competitor_mean",
    "paired_difference_std",
    "paired_t_ci_low",
    "paired_t_ci_high",
    "paired_t_p_value",
    "paired_wilcoxon_p_value",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_files(final_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in final_dir.rglob("*")
        if path.is_file() and path.name not in GENERATED_MANIFEST_NAMES
    )


def verify_manifest(relative_dir: Path) -> int:
    final_dir = REPO_ROOT / relative_dir
    manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_paths = {
        path.relative_to(final_dir).as_posix() for path in artifact_files(final_dir)
    }
    recorded_paths = {record["path"] for record in manifest["artifacts"]}
    if recorded_paths != expected_paths:
        missing = sorted(expected_paths - recorded_paths)
        stale = sorted(recorded_paths - expected_paths)
        raise AssertionError(f"{relative_dir}: missing={missing}, stale={stale}")
    for record in manifest["artifacts"]:
        path = final_dir / record["path"]
        assert path.stat().st_size == record["bytes"], path
        assert sha256(path) == record["sha256"], path
    checksum_records = {}
    for line in (final_dir / "artifact_checksums.sha256").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, path = line.split("  ", maxsplit=1)
        checksum_records[path] = digest
    assert checksum_records == {
        record["path"]: record["sha256"] for record in manifest["artifacts"]
    }
    return len(recorded_paths)


def verify_final_artifacts() -> dict[str, int]:
    return {path.parent.name: verify_manifest(path) for path in FINAL_DIRS}


def _rows(relative_path: str) -> list[dict[str, str]]:
    with (REPO_ROOT / relative_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _project(
    rows: Iterable[dict[str, str]], fields: Iterable[str]
) -> list[dict[str, str]]:
    return [{field: row[field] for field in fields} for row in rows]


def _contamination_cells() -> list[dict[str, str]]:
    cells = []
    for row in _rows(RESULT_PATHS["contamination"]):
        metric = "forward_kl" if row["objective"] == "forward_kl" else "reverse_kl"
        mean = row[f"{metric}_mean"]
        sd = row[f"{metric}_sd"]
        cells.append(
            {
                "kappa": row["kappa"],
                "epsilon": row["epsilon"],
                "objective": row["objective"],
                "family": row["family"],
                "seeds": row["seeds"],
                "mean": mean,
                "sd": sd,
                "paper_mean_4dp": f"{float(mean):.4f}",
                "paper_sd_4dp": f"{float(sd):.4f}",
            }
        )
    return cells


def _pairwise_rows() -> list[dict[str, str]]:
    text = (REPO_ROOT / RESULT_PATHS["pairwise"]).read_text(encoding="utf-8")
    rows = []
    pattern = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$"
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            dimension, case, delta, target, forward, reverse = match.groups()
            rows.append(
                {
                    "dimension": dimension,
                    "case": case,
                    "delta": delta,
                    "target": target,
                    "forward": forward,
                    "reverse": reverse,
                }
            )
    return rows


def verify_paper_results() -> dict[str, int]:
    """Raise ``AssertionError`` if reported-value source artifacts drift."""

    fixture: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    expected = fixture["reported_values"]

    direct = [
        row for row in _rows(RESULT_PATHS["direct"]) if row["method"] == "direct"
    ]
    assert direct == expected["direct_kl_high_precision"]["direct_summary"]

    neighbor = [
        row
        for row in _rows(RESULT_PATHS["neighbor"])
        if row["dimension"] in {"7", "9", "11", "21", "51", "101"}
    ]
    assert neighbor == expected["neighbor_and_laplace"]["representative_rows"]

    runtime = _rows(RESULT_PATHS["runtime"])
    assert runtime == expected["latent_runtime_summary"]

    mnist = _project(_rows(RESULT_PATHS["mnist"]), MNIST_FIELDS)
    assert mnist == expected["mnist_summary"]

    assert _contamination_cells() == expected["contamination_cells"]

    smallnorb = _project(_rows(RESULT_PATHS["smallnorb"]), SMALLNORB_FIELDS)
    assert smallnorb == expected["smallnorb_summary"]

    pairs = _project(
        (
            row
            for row in _rows(RESULT_PATHS["smallnorb_paired"])
            if row["is_primary_metric"] == "True"
        ),
        SMALLNORB_PAIR_FIELDS,
    )
    assert pairs == expected["smallnorb_primary_pairs"]

    pairwise = _pairwise_rows()
    assert pairwise == expected["pairwise_kl_validation"]["rows"]

    return {
        "direct_rows": len(direct),
        "neighbor_rows": len(neighbor),
        "runtime_rows": len(runtime),
        "mnist_rows": len(mnist),
        "contamination_cells": len(expected["contamination_cells"]),
        "smallnorb_rows": len(smallnorb),
        "smallnorb_primary_pairs": len(pairs),
        "pairwise_rows": len(pairwise),
    }
