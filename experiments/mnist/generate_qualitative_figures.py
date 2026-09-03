"""Generate all five deterministic MNIST qualitative paper figures.

The checkpoint is selected mechanically: among the five final spherical
Cauchy runs with reported intrinsic dimension two, choose the seed whose
held-out reconstruction loss is closest to their median (then the lower seed
on an exact tie). Compact decoded arrays are retained so ordinary paper
reproduction does not require the 14 MiB checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402

from experiments.mnist.workflow import (  # noqa: E402
    load_model_from_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_RESULTS = (
    REPO_ROOT
    / "experiments"
    / "mnist"
    / "final"
    / "seed_level.csv"
)
DEFAULT_FINAL_ROOT = Path(__file__).resolve().parent / "final"
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
OUTPUT_NAMES = (
    "posterior_sphere_trimmed.png",
    "decoded_uniform_sphere_trimmed.png",
    "reconstructions.png",
    "interpolation_1_to_7.png",
    "convergence.png",
)

POSTERIOR_INDICES = np.arange(500, dtype=np.int64)
RECONSTRUCTION_INDICES = np.arange(10, dtype=np.int64)
INTERPOLATION_STEPS = 10
UNIFORM_POINT_COUNT = 50
POSTERIOR_CAMERA = {"elevation": 24.0, "azimuth": 42.0}
DECODED_CAMERA = {"elevation": 25.0, "azimuth": 120.0}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_checkpoint(seed_results: Path = SEED_RESULTS) -> dict[str, Any]:
    candidates = [
        row
        for row in _read_csv(seed_results)
        if row["model_family"] == "spcauchy" and row["reported_dim"] == "2"
    ]
    if len(candidates) != 5:
        raise RuntimeError(
            f"expected five final p=2 spherical Cauchy seeds, found {len(candidates)}"
        )
    losses = np.asarray(
        [float(row["best_eval_recon_loss"]) for row in candidates],
        dtype=np.float64,
    )
    median = float(np.median(losses))
    selected = min(
        candidates,
        key=lambda row: (
            abs(float(row["best_eval_recon_loss"]) - median),
            int(row["seed"]),
        ),
    )
    run_dir = DEFAULT_FINAL_ROOT / "selected_run"
    checkpoint = (
        Path(__file__).resolve().parent
        / "checkpoints"
        / "selected_seed_0_epoch_40.pt"
    )
    config = run_dir / "run_config.json"
    evaluation = run_dir / "evaluation_summary.json"
    history = run_dir / "history.csv"
    for path in (checkpoint, config, evaluation, history):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "selected_row": selected,
        "median_reconstruction_loss": median,
        "all_candidates": sorted(candidates, key=lambda row: int(row["seed"])),
        "checkpoint": checkpoint,
        "config": config,
        "evaluation": evaluation,
        "history": history,
    }


def _validate_checkpoint(selection: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path: Path = selection["checkpoint"]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = json.loads(selection["config"].read_text(encoding="utf-8"))
    checkpoint_config = payload.get("run_config", {})
    compared_keys = (checkpoint_config.keys() | config.keys()) - {"notes"}
    config_mismatches = {
        key: {
            "checkpoint": checkpoint_config.get(key),
            "run_config": config.get(key),
        }
        for key in compared_keys
        if checkpoint_config.get(key) != config.get(key)
    }
    if config_mismatches:
        raise RuntimeError(
            "checkpoint run_config does not match run_config.json: "
            f"{config_mismatches}"
        )
    required = {
        "model_family": "spcauchy",
        "reported_dim": 2,
        "ambient_latent_dim": 3,
        "spcauchy_kl_method": "direct",
        "config_schema_version": 2,
        "epochs": 40,
        "seed": int(selection["selected_row"]["seed"]),
        "hidden_dims": [32, 64, 128],
        "encoder_type": "cnn",
        "decoder_type": "cnn",
    }
    mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"checkpoint contract mismatch: {mismatches}")
    if int(payload.get("epoch", -1)) != 40:
        raise RuntimeError("selected checkpoint is not the epoch-40 checkpoint")
    return {"payload": payload, "config": config}


def _fibonacci_sphere(count: int) -> np.ndarray:
    indices = np.arange(count, dtype=np.float64)
    y = 1.0 - 2.0 * indices / max(count - 1, 1)
    radius = np.sqrt(np.maximum(1.0 - y * y, 0.0))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * indices
    return np.column_stack((np.cos(theta) * radius, y, np.sin(theta) * radius))


def _slerp(first: torch.Tensor, second: torch.Tensor, steps: int) -> torch.Tensor:
    dot = torch.clamp(torch.sum(first * second), -1.0, 1.0)
    omega = torch.acos(dot)
    fractions = torch.linspace(0.0, 1.0, steps, device=first.device)
    if float(omega) < 1e-8:
        return first.expand(steps, -1).clone()
    points = (
        torch.sin((1.0 - fractions) * omega)[:, None] * first
        + torch.sin(fractions * omega)[:, None] * second
    ) / torch.sin(omega)
    return torch.nn.functional.normalize(points, dim=-1)


def _dataset(data_root: Path):
    return datasets.MNIST(
        root=str(data_root),
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )


def _stack_examples(dataset, indices: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    examples = [dataset[int(index)] for index in indices]
    images = torch.stack([image for image, _ in examples])
    labels = np.asarray([int(label) for _, label in examples], dtype=np.int64)
    return images, labels


def _encode_batches(model, images: torch.Tensor, device: str) -> torch.Tensor:
    locations = []
    with torch.inference_mode():
        for batch in images.split(128):
            loc, _ = model.encode(batch.to(device))
            locations.append(loc.cpu())
    return torch.cat(locations)


def _extract_inputs(
    output_root: Path, data_root: Path, device: str
) -> dict[str, Any]:
    selection = select_checkpoint()
    validated = _validate_checkpoint(selection)
    model, _, run_spec = load_model_from_checkpoint(
        selection["checkpoint"], device=device
    )
    model.eval()
    dataset = _dataset(data_root)

    posterior_images, posterior_labels = _stack_examples(
        dataset, POSTERIOR_INDICES
    )
    posterior_locations = _encode_batches(model, posterior_images, device)

    reconstruction_images, reconstruction_labels = _stack_examples(
        dataset, RECONSTRUCTION_INDICES
    )
    reconstruction_locations = _encode_batches(
        model, reconstruction_images, device
    )
    with torch.inference_mode():
        reconstructions = model.decode(
            reconstruction_locations.to(device)
        ).cpu()

    labels = np.asarray(dataset.targets, dtype=np.int64)
    first_one = int(np.flatnonzero(labels == 1)[0])
    first_seven = int(np.flatnonzero(labels == 7)[0])
    endpoint_indices = np.asarray([first_one, first_seven], dtype=np.int64)
    endpoint_images, endpoint_labels = _stack_examples(dataset, endpoint_indices)
    endpoint_locations = _encode_batches(model, endpoint_images, device)
    interpolation_locations = _slerp(
        endpoint_locations[0].to(device),
        endpoint_locations[1].to(device),
        INTERPOLATION_STEPS,
    )
    with torch.inference_mode():
        interpolation_decoded = model.decode(interpolation_locations).cpu()

    uniform_points = _fibonacci_sphere(UNIFORM_POINT_COUNT)
    with torch.inference_mode():
        decoded_uniform = model.decode(
            torch.from_numpy(uniform_points).to(
                device=device, dtype=next(model.parameters()).dtype
            )
        ).cpu()

    history = _read_csv(selection["history"])
    history_fields = list(history[0])
    history_values = np.asarray(
        [[float(row[field]) for field in history_fields] for row in history],
        dtype=np.float64,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    inputs_path = output_root / "qualitative_inputs.npz"
    np.savez_compressed(
        inputs_path,
        posterior_indices=POSTERIOR_INDICES,
        posterior_images=posterior_images.numpy(),
        posterior_labels=posterior_labels,
        posterior_locations=posterior_locations.numpy(),
        reconstruction_indices=RECONSTRUCTION_INDICES,
        reconstruction_labels=reconstruction_labels,
        reconstruction_inputs=reconstruction_images.numpy(),
        reconstruction_locations=reconstruction_locations.numpy(),
        reconstructions=reconstructions.numpy(),
        uniform_points=uniform_points,
        decoded_uniform=decoded_uniform.numpy(),
        interpolation_endpoint_indices=endpoint_indices,
        interpolation_endpoint_labels=endpoint_labels,
        interpolation_endpoint_locations=endpoint_locations.numpy(),
        interpolation_fractions=np.linspace(
            0.0, 1.0, INTERPOLATION_STEPS, dtype=np.float64
        ),
        interpolation_locations=interpolation_locations.cpu().numpy(),
        interpolation_decoded=interpolation_decoded.numpy(),
        history_fields=np.asarray(history_fields),
        history_values=history_values,
    )

    evaluation = json.loads(
        selection["evaluation"].read_text(encoding="utf-8")
    )["best_recon_checkpoint"]
    provenance = {
        "schema_version": 1,
        "selection_rule": (
            "Among final spcauchy p=2 seeds, select the held-out "
            "reconstruction loss closest to the five-seed median; break an "
            "exact tie by lower seed."
        ),
        "all_candidate_reconstruction_losses": {
            row["seed"]: float(row["best_eval_recon_loss"])
            for row in selection["all_candidates"]
        },
        "median_reconstruction_loss": selection["median_reconstruction_loss"],
        "selected_seed": run_spec.seed,
        "selected_epoch": int(validated["payload"]["epoch"]),
        "selected_evaluation_metrics": {
            "reconstruction_loss": evaluation["eval_recon_loss"],
            "total_loss": evaluation["eval_total_loss"],
            "kl": evaluation["eval_kl"],
        },
        "checkpoint": {
            "discovery_path": selection["checkpoint"].relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(selection["checkpoint"]),
            "committed": False,
            "reproduction_command": (
                "make train-mnist MODEL=spcauchy DIM=2 SEED=0"
            ),
        },
        "run_config": {
            "path": selection["config"].relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(selection["config"]),
            "config": validated["config"],
        },
        "history": {
            "path": selection["history"].relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(selection["history"]),
            "epochs": len(history),
        },
        "inputs": {
            "path": inputs_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(inputs_path),
        },
        "posterior_indices": POSTERIOR_INDICES.tolist(),
        "reconstruction_indices": RECONSTRUCTION_INDICES.tolist(),
        "interpolation_endpoint_indices": endpoint_indices.tolist(),
        "interpolation_endpoint_labels": endpoint_labels.tolist(),
        "interpolation_fractions": np.linspace(
            0.0, 1.0, INTERPOLATION_STEPS
        ).tolist(),
        "uniform_grid": {
            "kind": "deterministic Fibonacci sphere",
            "count": UNIFORM_POINT_COUNT,
        },
        "plotting": {
            "posterior_camera": POSTERIOR_CAMERA,
            "decoded_camera": DECODED_CAMERA,
            "savefig_dpi": 200,
            "crop_policy": "matplotlib tight bounding box; no postprocessing",
            "posterior_representation": "deterministic posterior location",
            "reconstruction_representation": "decoder at posterior location",
        },
        "generator": {
            "path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    return provenance


def _sphere(ax, *, color: str = "lightgray", alpha: float = 0.12) -> None:
    longitude = np.linspace(0.0, 2.0 * np.pi, 100)
    colatitude = np.linspace(0.0, np.pi, 100)
    x = np.outer(np.cos(longitude), np.sin(colatitude))
    y = np.outer(np.sin(longitude), np.sin(colatitude))
    z = np.outer(np.ones_like(longitude), np.cos(colatitude))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)
    ax.plot_wireframe(
        x[::8, ::8],
        y[::8, ::8],
        z[::8, ::8],
        color="navy",
        alpha=0.24,
        linewidth=0.35,
    )


def _clean_sphere_axes(ax) -> None:
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_zlabel("$z_3$")


def _plot_posterior(data: Any, path: Path) -> None:
    points = data["posterior_locations"]
    labels = data["posterior_labels"]
    fig = plt.figure(figsize=(8.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    _sphere(ax)
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=labels, cmap="tab10", s=18, alpha=0.9,
    )
    ax.view_init(
        elev=POSTERIOR_CAMERA["elevation"],
        azim=POSTERIOR_CAMERA["azimuth"],
    )
    _clean_sphere_axes(ax)
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.08)
    colorbar.set_ticks(range(10))
    colorbar.set_label("Digit label (visualization only)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_uniform_decodes(data: Any, path: Path) -> None:
    points = data["uniform_points"]
    decoded = data["decoded_uniform"]
    fig = plt.figure(figsize=(8.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    _sphere(ax)
    ax.view_init(
        elev=DECODED_CAMERA["elevation"],
        azim=DECODED_CAMERA["azimuth"],
    )
    camera_elevation = np.deg2rad(DECODED_CAMERA["elevation"])
    camera_azimuth = np.deg2rad(DECODED_CAMERA["azimuth"])
    camera = np.asarray(
        [
            np.cos(camera_elevation) * np.cos(camera_azimuth),
            np.cos(camera_elevation) * np.sin(camera_azimuth),
            np.sin(camera_elevation),
        ]
    )
    fig.canvas.draw()
    for point, image in zip(points, decoded):
        if float(np.dot(point, camera)) <= 0.0:
            continue
        x_2d, y_2d, _ = proj3d.proj_transform(*point, ax.get_proj())
        annotation = AnnotationBbox(
            OffsetImage(image.squeeze(), zoom=0.58, cmap="gray"),
            (x_2d, y_2d),
            frameon=False,
            pad=0.0,
        )
        ax.add_artist(annotation)
    _clean_sphere_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_reconstructions(data: Any, path: Path) -> None:
    originals = data["reconstruction_inputs"]
    reconstructions = data["reconstructions"]
    labels = data["reconstruction_labels"]
    count = originals.shape[0]
    fig, axes = plt.subplots(2, count, figsize=(1.35 * count, 3.1))
    for index in range(count):
        axes[0, index].imshow(originals[index].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0, index].set_title(str(int(labels[index])), fontsize=9)
        axes[1, index].imshow(
            reconstructions[index].squeeze(), cmap="gray", vmin=0, vmax=1
        )
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    axes[0, 0].set_ylabel("Input")
    axes[1, 0].set_ylabel("Reconstruction")
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_interpolation(data: Any, path: Path) -> None:
    decoded = data["interpolation_decoded"]
    fractions = data["interpolation_fractions"]
    labels = data["interpolation_endpoint_labels"]
    fig, axes = plt.subplots(1, decoded.shape[0], figsize=(13.5, 1.8))
    for index, (axis, image, fraction) in enumerate(zip(axes, decoded, fractions)):
        axis.imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"{fraction:.2f}", fontsize=8)
        axis.axis("off")
        if index == 0:
            axis.set_ylabel(str(int(labels[0])))
        elif index == decoded.shape[0] - 1:
            axis.set_ylabel(str(int(labels[1])))
    fig.tight_layout(pad=0.25)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_convergence(data: Any, path: Path) -> None:
    fields = [str(value) for value in data["history_fields"]]
    values = data["history_values"]
    columns = {field: values[:, index] for index, field in enumerate(fields)}
    panels = [
        ("total_loss", "Objective"),
        ("recon_loss", "Reconstruction"),
        ("kl", "KL"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.6))
    for axis, (suffix, title) in zip(axes, panels):
        axis.plot(columns["epoch"], columns[f"train_{suffix}"], label="Train")
        axis.plot(columns["epoch"], columns[f"eval_{suffix}"], label="Test")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_figures(output_root: Path) -> dict[str, str]:
    inputs_path = output_root / "qualitative_inputs.npz"
    if not inputs_path.is_file():
        raise FileNotFoundError(
            f"missing {inputs_path}; rerun with --refresh-from-checkpoint"
        )
    with np.load(inputs_path, allow_pickle=False) as data:
        _plot_posterior(data, output_root / OUTPUT_NAMES[0])
        _plot_uniform_decodes(data, output_root / OUTPUT_NAMES[1])
        _plot_reconstructions(data, output_root / OUTPUT_NAMES[2])
        _plot_interpolation(data, output_root / OUTPUT_NAMES[3])
        _plot_convergence(data, output_root / OUTPUT_NAMES[4])
    return {name: _sha256(output_root / name) for name in OUTPUT_NAMES}


def generate_all(
    output_root: Path,
    data_root: Path,
    device: str,
    *,
    refresh_from_checkpoint: bool,
) -> dict[str, Any]:
    provenance_path = output_root / "qualitative_provenance.json"
    if refresh_from_checkpoint:
        provenance = _extract_inputs(output_root, data_root, device)
    else:
        if not provenance_path.is_file():
            raise FileNotFoundError(
                f"missing {provenance_path}; rerun with --refresh-from-checkpoint"
            )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["figure_sha256"] = render_figures(output_root)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FINAL_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--refresh-from-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    provenance = generate_all(
        args.output_root,
        args.data_root,
        args.device,
        refresh_from_checkpoint=args.refresh_from_checkpoint,
    )
    print(json.dumps(provenance["figure_sha256"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
