"""Ground-truth interpolation through the held-out angular sector."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .data import SmallNORBViewDataset
from .evaluate import load_checkpoint
from .metrics import (
    reconstruction_metric_vectors,
    slerp,
    summarize_tensor,
)
from .utils import resolve_device, set_global_seed, write_csv, write_json


PATH_AZIMUTH_INDICES = (15, 16, 17, 0, 1, 2)
PATH_AZIMUTH_DEGREES = (300, 320, 340, 0, 20, 40)


def _path_lookup(dataset: SmallNORBViewDataset):
    groups: dict[
        tuple[int, int, int, int], dict[int, int]
    ] = defaultdict(dict)
    for local_index, source_index in enumerate(dataset.indices):
        key = (
            int(dataset.metadata["category"][source_index]),
            int(dataset.metadata["instance"][source_index]),
            int(dataset.metadata["elevation"][source_index]),
            int(dataset.metadata["lighting"][source_index]),
        )
        azimuth = int(dataset.metadata["azimuth_index"][source_index])
        groups[key][azimuth] = local_index
    complete = []
    for key, by_azimuth in sorted(groups.items()):
        if all(index in by_azimuth for index in PATH_AZIMUTH_INDICES):
            complete.append(
                (key, [by_azimuth[index] for index in PATH_AZIMUTH_INDICES])
            )
    return complete


@torch.no_grad()
def interpolation_run(
    run_dir: str | Path,
    device: torch.device,
    *,
    split: str,
    batch_paths: int = 64,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    run_path = Path(run_dir)
    model, _, config = load_checkpoint(
        run_path / "checkpoint_best.pt", device
    )
    dataset = SmallNORBViewDataset(
        config.data_root,
        split,
        gap_azimuth_indices=config.gap_azimuth_indices,
        limit=config.test_limit if split == "test" else None,
        subset_seed=config.seed,
    )
    paths = _path_lookup(dataset)
    set_global_seed(config.seed)
    records: list[dict[str, Any]] = []
    gallery_candidates: list[dict[str, Any]] = []
    fractions = torch.linspace(0.0, 1.0, 6, device=device)
    for start in range(0, len(paths), batch_paths):
        current = paths[start : start + batch_paths]
        endpoint_start = torch.stack(
            [dataset[indices[0]][0] for _, indices in current]
        ).to(device)
        endpoint_end = torch.stack(
            [dataset[indices[-1]][0] for _, indices in current]
        ).to(device)
        true_images = torch.stack(
            [
                torch.stack([dataset[index][0] for index in indices])
                for _, indices in current
            ]
        ).to(device)
        start_parameters = model.encode(endpoint_start)
        end_parameters = model.encode(endpoint_end)
        start_location = start_parameters.location
        end_location = end_parameters.location
        interpolated = []
        for fraction in fractions:
            if model.posterior.is_spherical:
                latent = slerp(
                    start_location, end_location, fraction
                )
            else:
                gaussian = (
                    (1.0 - fraction) * start_location
                    + fraction * end_location
                )
                latent = F.pad(gaussian, (0, 1), value=0.0)
            interpolated.append(latent)
        latent = torch.stack(interpolated, dim=1)
        decoded = model.decode(latent.flatten(0, 1)).view(
            len(current), 6, 1, 64, 64
        )
        vectors = reconstruction_metric_vectors(
            true_images.flatten(0, 1),
            decoded.flatten(0, 1),
            config.sigma_x,
            include_ssim=True,
        )
        vector_cpu = {
            key: value.view(len(current), 6).cpu().numpy()
            for key, value in vectors.items()
        }
        decoded_uint8 = (
            decoded.detach().cpu().mul(255).round().clamp(0, 255)
            .to(torch.uint8)
            .numpy()
        )
        truth_uint8 = (
            true_images.detach().cpu().mul(255).round().clamp(0, 255)
            .to(torch.uint8)
            .numpy()
        )
        for path_index, (key, indices) in enumerate(current):
            category, instance, elevation, lighting = key
            for position, (
                azimuth_index,
                azimuth_degrees,
                local_index,
            ) in enumerate(
                zip(
                    PATH_AZIMUTH_INDICES,
                    PATH_AZIMUTH_DEGREES,
                    indices,
                )
            ):
                _, metadata = dataset[local_index]
                records.append(
                    {
                        "split": split,
                        "category": category,
                        "instance": instance,
                        "elevation": elevation,
                        "lighting": lighting,
                        "path_position": position,
                        "azimuth_index": azimuth_index,
                        "azimuth_degrees": azimuth_degrees,
                        "is_endpoint": int(position in {0, 5}),
                        "is_gap": int(position in {1, 2, 3, 4}),
                        "source_index": metadata["source_index"],
                        **{
                            metric: float(values[path_index, position])
                            for metric, values in vector_cpu.items()
                        },
                    }
                )
            gallery_candidates.append(
                {
                    "key": key,
                    "interior_mse": float(
                        vector_cpu["pixel_mse"][path_index, 1:5].mean()
                    ),
                    "ground_truth": truth_uint8[path_index],
                    "decoded": decoded_uint8[path_index],
                }
            )
    summaries = {}
    for name, selector in (
        ("all", lambda row: True),
        ("endpoints", lambda row: row["is_endpoint"] == 1),
        ("interior_gap", lambda row: row["is_gap"] == 1),
    ):
        selected = [row for row in records if selector(row)]
        summaries[name] = {
            metric: summarize_tensor(
                torch.tensor([row[metric] for row in selected])
            )
            for metric in (
                "reconstruction_nll",
                "pixel_mse",
                "psnr_db",
                "ssim",
            )
        }
    summary = {
        "config": config.to_dict(),
        "split": split,
        "test_was_accessed": split == "test",
        "path_azimuth_indices": list(PATH_AZIMUTH_INDICES),
        "path_azimuth_degrees": list(PATH_AZIMUTH_DEGREES),
        "interpolation": (
            "SLERP between posterior locations"
            if model.posterior.is_spherical
            else "linear interpolation between Gaussian posterior means"
        ),
        "path_count": len(paths),
        "summaries": summaries,
    }
    if gallery_candidates:
        median_path_mse = float(
            np.median(
                [
                    candidate["interior_mse"]
                    for candidate in gallery_candidates
                ]
            )
        )
        representative = min(
            gallery_candidates,
            key=lambda candidate: (
                abs(candidate["interior_mse"] - median_path_mse),
                candidate["key"],
            ),
        )
        gallery_path = (
            run_path / f"interpolation_representative_{split}.npz"
        )
        np.savez_compressed(
            gallery_path,
            ground_truth=representative["ground_truth"],
            decoded=representative["decoded"],
            key=np.asarray(representative["key"], dtype=np.int16),
            azimuth_indices=np.asarray(
                PATH_AZIMUTH_INDICES, dtype=np.int16
            ),
            azimuth_degrees=np.asarray(
                PATH_AZIMUTH_DEGREES, dtype=np.int16
            ),
            interior_mse=np.asarray(
                representative["interior_mse"], dtype=np.float64
            ),
            seed_level_median_interior_mse=np.asarray(
                median_path_mse, dtype=np.float64
            ),
        )
        summary["representative_path"] = {
            "selection_rule": (
                "path whose interior interpolation MSE is closest to the "
                "seed-level median, with lexicographic metadata tie break"
            ),
            "category": representative["key"][0],
            "instance": representative["key"][1],
            "elevation": representative["key"][2],
            "lighting": representative["key"][3],
            "interior_mse": representative["interior_mse"],
            "seed_level_median_interior_mse": median_path_mse,
            "gallery_path": str(gallery_path),
        }
    write_csv(
        run_path / f"interpolation_records_{split}.csv", records
    )
    write_json(
        run_path / f"interpolation_summary_{split}.json", summary
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = interpolation_run(
        args.run_dir,
        resolve_device(args.device),
        split=args.split,
    )
    print(
        {
            "path_count": summary["path_count"],
            "interior_mse": summary["summaries"]["interior_gap"][
                "pixel_mse"
            ]["mean"],
        }
    )


if __name__ == "__main__":
    main()
