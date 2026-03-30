from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    from common import (
        DEFAULT_CAMERA,
        DEFAULT_PAYLOAD_FILENAME,
        DEFAULT_RUN_DIR,
        PALETTE,
        ROOT,
        ensure_site_data_dir,
        mnist_image_data_uri,
        read_json,
        write_json,
    )
else:
    from .common import (
        DEFAULT_CAMERA,
        DEFAULT_PAYLOAD_FILENAME,
        DEFAULT_RUN_DIR,
        PALETTE,
        ROOT,
        ensure_site_data_dir,
        mnist_image_data_uri,
        read_json,
        write_json,
    )

from experiments.mnist_experiment.data import build_mnist_dataloaders
from experiments.mnist_experiment.utils import repo_relative_path
from experiments.mnist_experiment.workflow import load_model_from_checkpoint


def _collect_eval_embeddings(model, eval_loader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    image_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch, labels in eval_loader:
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            point_chunks.append(mu.cpu().numpy())
            label_chunks.append(labels.numpy())
            image_chunks.append(batch.cpu().numpy())
    points = np.concatenate(point_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    images = np.concatenate(image_chunks, axis=0)
    return points, labels, images


def _balanced_subset_indices(labels: np.ndarray, per_label: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected_chunks: list[np.ndarray] = []
    for label in range(10):
        label_indices = np.flatnonzero(labels == label)
        if len(label_indices) < per_label:
            raise ValueError(f"Not enough examples for label {label}: need {per_label}, found {len(label_indices)}")
        chosen = rng.choice(label_indices, size=per_label, replace=False)
        selected_chunks.append(np.sort(chosen))
    selected = np.concatenate(selected_chunks, axis=0)
    rng.shuffle(selected)
    return selected


def build_payload(
    run_dir: Path,
    data_dir: Path,
    output_path: Path,
    num_points: int,
    seed: int,
    device: str,
) -> dict:
    if num_points % 10 != 0:
        raise ValueError("num_points must be divisible by 10 for balanced digit sampling.")

    checkpoint_path = run_dir / "best_recon_checkpoint.pt"
    selection_summary = read_json(run_dir / "selection_summary.json")
    if selection_summary.get("checkpoint_path"):
        selection_summary["checkpoint_path"] = repo_relative_path(selection_summary["checkpoint_path"])
    model, _, run_spec = load_model_from_checkpoint(checkpoint_path, device=device)
    _, eval_loader = build_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=run_spec.batch_size,
        seed=run_spec.seed,
    )

    points, labels, images = _collect_eval_embeddings(model, eval_loader, device)
    chosen_indices = _balanced_subset_indices(labels, per_label=num_points // 10, seed=seed)
    label_names = [str(label) for label in range(10)]

    exported_points = []
    for subset_index, dataset_index in enumerate(chosen_indices):
        point = points[dataset_index]
        image = images[dataset_index].squeeze()
        label = int(labels[dataset_index])
        exported_points.append(
            {
                "subset_index": subset_index,
                "dataset_index": int(dataset_index),
                "label": label,
                "label_name": label_names[label],
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
                "image_uri": mnist_image_data_uri(image),
            }
        )

    payload = {
        "title": "MNIST Posterior Means on S^2",
        "description": (
            "Interactive posterior means from the qualitative spCauchy-VAE MNIST run. "
            "Hover over a point to inspect the original handwritten digit."
        ),
        "source_run": repo_relative_path(run_dir),
        "source_checkpoint": repo_relative_path(checkpoint_path),
        "selection_summary": selection_summary,
        "run_config": run_spec.to_dict(),
        "default_camera": DEFAULT_CAMERA,
        "palette": PALETTE,
        "label_names": label_names,
        "num_points": len(exported_points),
        "points_per_label": num_points // 10,
        "points": exported_points,
    }
    write_json(output_path, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the qualitative MNIST S^2 Plotly payload.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-points", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser


def main() -> None:
    from experiments.mnist_experiment.config import DEFAULT_DATA_DIR
    from experiments.mnist_experiment.utils import resolve_device

    args = build_arg_parser().parse_args()
    output_dir = ensure_site_data_dir()
    output_path = Path(args.output) if args.output else output_dir / DEFAULT_PAYLOAD_FILENAME
    build_payload(
        run_dir=Path(args.run_dir),
        data_dir=Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR,
        output_path=output_path,
        num_points=args.num_points,
        seed=args.seed,
        device=resolve_device(args.device),
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
