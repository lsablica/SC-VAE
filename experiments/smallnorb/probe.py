"""Circular pose probes and latent-distance alignment diagnostics."""

from __future__ import annotations

import argparse
import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from .data import build_dataloaders
from .evaluate import load_checkpoint
from .metrics import (
    circular_absolute_error_degrees,
    circular_distance_radians,
    summarize_circular_errors,
)
from .utils import resolve_device, set_global_seed, write_csv, write_json


RIDGE_STRENGTHS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
GEOMETRY_PAIR_COUNT = 20_000


@torch.no_grad()
def extract_locations(model, loader, device: torch.device) -> dict[str, Any]:
    model.eval()
    locations = []
    metadata_rows: dict[str, list[np.ndarray]] = defaultdict(list)
    for images, metadata in loader:
        parameters = model.encode(
            images.to(device, non_blocking=True)
        )
        locations.append(parameters.location.detach().cpu().numpy())
        for key, values in metadata.items():
            metadata_rows[key].append(values.detach().cpu().numpy())
    return {
        "location": np.concatenate(locations, axis=0),
        **{
            key: np.concatenate(chunks, axis=0)
            for key, chunks in metadata_rows.items()
        },
    }


def _pose_targets(azimuth_degrees: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(azimuth_degrees)
    return np.stack((np.cos(radians), np.sin(radians)), axis=1)


def _predicted_angles(predictions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(predictions, axis=1, keepdims=True)
    normalized = predictions / np.maximum(norms, 1e-12)
    return np.rad2deg(
        np.arctan2(normalized[:, 1], normalized[:, 0])
    ) % 360.0


def _pose_partition(
    model: Ridge,
    features: np.ndarray,
    azimuth_degrees: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    predicted = _predicted_angles(model.predict(features))
    errors = circular_absolute_error_degrees(
        predicted, azimuth_degrees
    )
    return summarize_circular_errors(errors), predicted, errors


def fit_pose_probe(
    train: dict[str, Any],
    validation: dict[str, Any],
    *,
    include_test: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    targets = _pose_targets(train["azimuth_degrees"])
    observed_mask = validation["is_gap"] == 0
    validation_target = validation["azimuth_degrees"]
    candidates = []
    for alpha in RIDGE_STRENGTHS:
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(train["location"], targets)
        metrics, _, _ = _pose_partition(
            ridge,
            validation["location"][observed_mask],
            validation_target[observed_mask],
        )
        candidates.append((metrics["mean_absolute_error_degrees"], alpha))
    _, selected_alpha = min(candidates)
    ridge = Ridge(alpha=selected_alpha, fit_intercept=True)
    ridge.fit(train["location"], targets)

    datasets = {"validation": validation}
    if include_test is not None:
        datasets["test"] = include_test
    summary: dict[str, Any] = {
        "ridge_strength_grid": list(RIDGE_STRENGTHS),
        "selected_ridge_strength": selected_alpha,
        "selection_partition": "validation_observed",
        "partitions": {},
    }
    prediction_tables: dict[str, list[dict[str, Any]]] = {}
    for name, dataset in datasets.items():
        predictions, errors = None, None
        all_metrics, predictions, errors = _pose_partition(
            ridge, dataset["location"], dataset["azimuth_degrees"]
        )
        summary["partitions"][name] = all_metrics
        for suffix, mask in (
            ("observed", dataset["is_gap"] == 0),
            ("gap", dataset["is_gap"] == 1),
        ):
            metrics, _, _ = _pose_partition(
                ridge,
                dataset["location"][mask],
                dataset["azimuth_degrees"][mask],
            )
            summary["partitions"][f"{name}_{suffix}"] = metrics
        prediction_tables[name] = [
            {
                "category": int(dataset["category"][index]),
                "instance": int(dataset["instance"][index]),
                "elevation": int(dataset["elevation"][index]),
                "lighting": int(dataset["lighting"][index]),
                "azimuth_index": int(dataset["azimuth_index"][index]),
                "true_degrees": int(dataset["azimuth_degrees"][index]),
                "predicted_degrees": float(predictions[index]),
                "absolute_error_degrees": float(errors[index]),
                "is_gap": int(dataset["is_gap"][index]),
            }
            for index in range(len(predictions))
        ]
    return summary, prediction_tables


def _crosses_gap(
    first: int,
    second: int,
    gap: set[int],
) -> bool:
    forward = (second - first) % 18
    if forward <= 9:
        path = [(first + step) % 18 for step in range(forward + 1)]
    else:
        path = [
            (first - step) % 18 for step in range(18 - forward + 1)
        ]
    return any(index in gap for index in path)


def _candidate_pairs(
    dataset: dict[str, Any],
    gap_azimuth_indices: tuple[int, ...],
) -> list[tuple[int, int, bool]]:
    groups: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for index in range(len(dataset["location"])):
        key = (
            int(dataset["category"][index]),
            int(dataset["instance"][index]),
            int(dataset["elevation"][index]),
            int(dataset["lighting"][index]),
        )
        groups[key].append(index)
    gap = set(gap_azimuth_indices)
    pairs = []
    for indices in groups.values():
        for first, second in itertools.combinations(indices, 2):
            pairs.append(
                (
                    first,
                    second,
                    _crosses_gap(
                        int(dataset["azimuth_index"][first]),
                        int(dataset["azimuth_index"][second]),
                        gap,
                    ),
                )
            )
    return pairs


def geometry_alignment(
    dataset: dict[str, Any],
    *,
    spherical: bool,
    gaussian_mean: np.ndarray | None,
    gaussian_std: np.ndarray | None,
    seed: int,
    pair_count: int = GEOMETRY_PAIR_COUNT,
    gap_azimuth_indices: tuple[int, ...] = (16, 17, 0, 1),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = _candidate_pairs(dataset, gap_azimuth_indices)
    if not candidates:
        raise ValueError("No within-object viewpoint pairs are available")
    generator = np.random.default_rng(seed)
    selected_indices = generator.choice(
        len(candidates),
        size=min(pair_count, len(candidates)),
        replace=False,
    )
    pairs = [candidates[index] for index in selected_indices]
    first = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    second = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    crossing = np.asarray([pair[2] for pair in pairs], dtype=bool)
    location = dataset["location"]
    if spherical:
        dots = np.sum(location[first] * location[second], axis=1)
        latent_distance = np.arccos(np.clip(dots, -1.0, 1.0))
        distance_name = "spherical_geodesic_radians"
    else:
        if gaussian_mean is None or gaussian_std is None:
            raise ValueError("Gaussian training standardization is required")
        standardized = (location - gaussian_mean) / gaussian_std
        latent_distance = np.linalg.norm(
            standardized[first] - standardized[second], axis=1
        )
        distance_name = "standardized_euclidean"
    true_distance = circular_distance_radians(
        dataset["azimuth_degrees"][first],
        dataset["azimuth_degrees"][second],
    )

    def correlation(mask: np.ndarray) -> dict[str, Any]:
        statistic, p_value = spearmanr(
            true_distance[mask], latent_distance[mask]
        )
        return {
            "count": int(mask.sum()),
            "spearman": float(statistic),
            "p_value": float(p_value),
        }

    summary = {
        "pair_sampling_seed": seed,
        "requested_pair_count": pair_count,
        "candidate_pair_count": len(candidates),
        "latent_distance": distance_name,
        "all_pairs": correlation(np.ones(len(pairs), dtype=bool)),
        "pairs_crossing_gap": correlation(crossing),
        "crossing_definition": (
            "the shortest discrete circular path, including endpoints, "
            "contains at least one held-out azimuth index"
        ),
    }
    rows = [
        {
            "first_source_index": int(dataset["source_index"][a]),
            "second_source_index": int(dataset["source_index"][b]),
            "first_azimuth_degrees": int(dataset["azimuth_degrees"][a]),
            "second_azimuth_degrees": int(dataset["azimuth_degrees"][b]),
            "true_circular_distance_radians": float(true_distance[index]),
            "latent_distance": float(latent_distance[index]),
            "crosses_gap": int(crossing[index]),
        }
        for index, (a, b, _) in enumerate(pairs)
    ]
    return summary, rows


def probe_run(
    run_dir: str | Path,
    device: torch.device,
    *,
    include_test: bool,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    model, _, config = load_checkpoint(
        run_path / "checkpoint_best.pt", device
    )
    loaders = build_dataloaders(config, include_test=include_test)
    set_global_seed(config.seed)
    train = extract_locations(model, loaders["train"], device)
    validation = extract_locations(
        model, loaders["validation"], device
    )
    test = (
        extract_locations(model, loaders["test"], device)
        if include_test
        else None
    )
    pose_summary, predictions = fit_pose_probe(
        train, validation, include_test=test
    )
    for split, rows in predictions.items():
        write_csv(run_path / f"pose_predictions_{split}.csv", rows)

    gaussian_mean = (
        train["location"].mean(axis=0)
        if not model.posterior.is_spherical
        else None
    )
    gaussian_std = (
        np.maximum(train["location"].std(axis=0), 1e-8)
        if not model.posterior.is_spherical
        else None
    )
    geometry = {}
    validation_geometry, validation_pairs = geometry_alignment(
        validation,
        spherical=model.posterior.is_spherical,
        gaussian_mean=gaussian_mean,
        gaussian_std=gaussian_std,
        seed=config.seed + 20_000,
        gap_azimuth_indices=config.gap_azimuth_indices,
    )
    geometry["validation"] = validation_geometry
    write_csv(
        run_path / "geometry_pairs_validation.csv",
        validation_pairs,
    )
    if test is not None:
        test_geometry, test_pairs = geometry_alignment(
            test,
            spherical=model.posterior.is_spherical,
            gaussian_mean=gaussian_mean,
            gaussian_std=gaussian_std,
            seed=config.seed + 30_000,
            gap_azimuth_indices=config.gap_azimuth_indices,
        )
        geometry["test"] = test_geometry
        write_csv(run_path / "geometry_pairs_test.csv", test_pairs)
    summary = {
        "config": config.to_dict(),
        "test_was_accessed": include_test,
        "pose_probe": pose_summary,
        "geometry_alignment": geometry,
    }
    write_json(run_path / "probe_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--include-test", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = probe_run(
        args.run_dir,
        resolve_device(args.device),
        include_test=args.include_test,
    )
    print(summary["pose_probe"]["selected_ridge_strength"])


if __name__ == "__main__":
    main()
