from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.mplot3d import proj3d

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mnist_experiment.config import (  # noqa: E402
    BENCHMARK_PRESET,
    OUTPUT_ROOT,
    QUALITATIVE_PRESET,
    DEFAULT_DATA_DIR,
    build_specs_for_preset,
)
from experiments.mnist_experiment.data import build_mnist_dataloaders  # noqa: E402
from experiments.mnist_experiment.utils import checkpoint_exists, ensure_dir, read_csv, resolve_device  # noqa: E402
from experiments.mnist_experiment.workflow import load_model_from_checkpoint  # noqa: E402


def generate_convergence_plot(history_rows: list[dict], output_path: Path, title: str) -> None:
    epochs = [int(row["epoch"]) for row in history_rows]
    train_total = [float(row["train_total_loss"]) for row in history_rows]
    eval_total = [float(row["eval_total_loss"]) for row in history_rows]
    train_recon = [float(row["train_recon_loss"]) for row in history_rows]
    eval_recon = [float(row["eval_recon_loss"]) for row in history_rows]
    train_kl = [float(row["train_kl"]) for row in history_rows]
    eval_kl = [float(row["eval_kl"]) for row in history_rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(epochs, train_total, label="Train")
    axes[0].plot(epochs, eval_total, label="Eval")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_recon, label="Train")
    axes[1].plot(epochs, eval_recon, label="Eval")
    axes[1].set_title("Reconstruction Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    axes[2].plot(epochs, train_kl, label="Train")
    axes[2].plot(epochs, eval_kl, label="Eval")
    axes[2].set_title("KL Divergence")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_reconstruction_panel(model, eval_loader, device: str, output_path: Path, n: int = 10) -> None:
    batch, _ = next(iter(eval_loader))
    batch = batch[:n].to(device)
    with torch.no_grad():
        reconstructions, _, _ = model(batch)

    originals = batch.cpu().numpy()
    recons = reconstructions.cpu().numpy()
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for index in range(n):
        axes[0, index].imshow(originals[index].squeeze(), cmap="gray")
        axes[0, index].axis("off")
        if index == 0:
            axes[0, index].set_title("Original")
        axes[1, index].imshow(recons[index].squeeze(), cmap="gray")
        axes[1, index].axis("off")
        if index == 0:
            axes[1, index].set_title("Reconstruction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def spherical_interpolation(p1: torch.Tensor, p2: torch.Tensor, steps: int = 10) -> torch.Tensor:
    dot = torch.clamp(torch.sum(p1 * p2), -1.0, 1.0)
    omega = torch.acos(dot)
    if float(omega.item()) < 1e-6:
        return torch.stack([p1] * steps)
    t_values = torch.linspace(0.0, 1.0, steps, device=p1.device)
    sin_omega = torch.sin(omega)
    points = []
    for t_value in t_values:
        point = (
            torch.sin((1.0 - t_value) * omega) / sin_omega * p1
            + torch.sin(t_value * omega) / sin_omega * p2
        )
        points.append(torch.nn.functional.normalize(point, dim=0))
    return torch.stack(points)


def generate_interpolation_panel(model, eval_loader, device: str, output_path: Path, steps: int = 10) -> None:
    batch, labels = next(iter(eval_loader))
    digit_indices: dict[int, int] = {}
    for index, label in enumerate(labels):
        label_value = int(label.item())
        if label_value not in digit_indices:
            digit_indices[label_value] = index

    idx1 = digit_indices.get(1, 0)
    idx7 = digit_indices.get(7, 1)

    x1 = batch[idx1 : idx1 + 1].to(device)
    x2 = batch[idx7 : idx7 + 1].to(device)
    with torch.no_grad():
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)
        interpolation_points = spherical_interpolation(mu1[0], mu2[0], steps=steps)
        decoded = model.decode(interpolation_points).cpu().numpy()

    fig, axes = plt.subplots(1, steps, figsize=(1.6 * steps, 2.5))
    for index in range(steps):
        axes[index].imshow(decoded[index].squeeze(), cmap="gray")
        axes[index].axis("off")
    fig.suptitle("Spherical interpolation: 1 to 7")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_posterior_sphere_plot(model, eval_loader, device: str, output_path: Path, n: int = 500) -> None:
    latent_vectors = []
    labels = []
    with torch.no_grad():
        for batch, batch_labels in eval_loader:
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            latent_vectors.append(mu.cpu().numpy())
            labels.append(batch_labels.numpy())
            if sum(chunk.shape[0] for chunk in latent_vectors) >= n:
                break

    points = np.concatenate(latent_vectors, axis=0)[:n]
    point_labels = np.concatenate(labels, axis=0)[:n]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="lightgray", alpha=0.12)
    ax.plot_wireframe(xs[::8, ::8], ys[::8, ::8], zs[::8, ::8], color="navy", alpha=0.25, linewidth=0.4)
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=point_labels, cmap="tab10", s=16)
    plt.colorbar(scatter, ax=ax, shrink=0.7)
    ax.set_title("Posterior means on $S^2$")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_posterior_sphere_side_by_side_plot(
    model,
    eval_loader,
    device: str,
    output_path: Path,
    n: int = 500,
    figsize: tuple[float, float] = (12, 10),
    alpha: float = 0.1,
) -> None:
    latent_vectors = []
    labels = []
    with torch.no_grad():
        for batch, batch_labels in eval_loader:
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            latent_vectors.append(mu.cpu().numpy())
            labels.append(batch_labels.numpy())
            if sum(chunk.shape[0] for chunk in latent_vectors) >= n:
                break

    points = np.concatenate(latent_vectors, axis=0)[:n]
    point_labels = np.concatenate(labels, axis=0)[:n]
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.clip(norms, a_min=1e-12, a_max=None)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="b", alpha=alpha)

    u_grid = np.linspace(0, 2 * np.pi, 20)
    v_grid = np.linspace(0, np.pi, 10)
    x_grid = np.outer(np.cos(u_grid), np.sin(v_grid))
    y_grid = np.outer(np.sin(u_grid), np.sin(v_grid))
    z_grid = np.outer(np.ones(np.size(u_grid)), np.cos(v_grid))
    ax.plot_wireframe(x_grid, y_grid, z_grid, color="navy", alpha=0.3, linewidth=0.5)

    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=point_labels, cmap="tab10", s=20)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")
    ax.grid(False)
    ax.set_box_aspect([1, 1, 1])
    # Use a less edge-on perspective than the notebook default for a clearer class layout.
    ax.view_init(elev=24, azim=42)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Latent Space Visualization on Unit Sphere")

    cbar_ax = fig.add_axes([0.05, 0.15, 0.03, 0.7])
    colorbar = plt.colorbar(scatter, cax=cbar_ax)
    colorbar.set_label("Label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def sample_sphere_points(n_points: int) -> np.ndarray:
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for index in range(n_points):
        y = 1.0 - (index / max(n_points - 1, 1)) * 2.0
        radius = np.sqrt(max(1.0 - y * y, 0.0))
        theta = phi * index
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append((x, y, z))
    return np.asarray(points)


def generate_uniform_sphere_decodes(model, device: str, output_path: Path, n_points: int = 24) -> None:
    points = sample_sphere_points(n_points)
    z_points = torch.tensor(points, dtype=torch.float32, device=device)
    with torch.no_grad():
        decoded = model.decode(z_points).cpu().numpy()

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=30, azim=120)
    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(0, np.pi, 80)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="lightgray", alpha=0.1)
    ax.plot_wireframe(xs[::8, ::8], ys[::8, ::8], zs[::8, ::8], color="navy", alpha=0.25, linewidth=0.4)

    azim_rad = np.deg2rad(ax.azim)
    elev_rad = np.deg2rad(ax.elev)
    camera = np.array(
        [
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad),
        ]
    )

    for point, image in zip(points, decoded):
        if np.dot(point, camera) <= 0:
            continue
        x_coord, y_coord, z_coord = point
        x_2d, y_2d, _ = proj3d.proj_transform(x_coord, y_coord, z_coord, ax.get_proj())
        imagebox = OffsetImage(image.squeeze(), zoom=0.95, cmap="gray")
        annotation = AnnotationBbox(imagebox, (x_2d, y_2d), frameon=False, pad=0.0)
        ax.add_artist(annotation)

    ax.set_title("Decoded samples from uniformly chosen points on $S^2$")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_visible_only_sphere_with_images(
    model,
    device: str,
    output_path: Path,
    n_points: int = 50,
    figsize: tuple[float, float] = (12, 10),
    alpha: float = 0.1,
) -> None:
    points = sample_sphere_points(n_points)
    z_points = torch.tensor(points, dtype=torch.float32, device=device)
    with torch.no_grad():
        decoded = model.decode(z_points).cpu().numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")
    ax.grid(False)

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="b", alpha=alpha)

    u_grid = np.linspace(0, 2 * np.pi, 20)
    v_grid = np.linspace(0, np.pi, 10)
    x_grid = np.outer(np.cos(u_grid), np.sin(v_grid))
    y_grid = np.outer(np.sin(u_grid), np.sin(v_grid))
    z_grid = np.outer(np.ones(np.size(u_grid)), np.cos(v_grid))
    ax.plot_wireframe(x_grid, y_grid, z_grid, color="navy", alpha=0.3, linewidth=0.5)

    azim_rad = np.deg2rad(ax.azim)
    elev_rad = np.deg2rad(ax.elev)
    camera = np.array(
        [
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad),
        ]
    )

    for point, image in zip(points, decoded):
        if np.dot(point, camera) <= 0:
            continue
        x_coord, y_coord, z_coord = point
        x_2d, y_2d, _ = proj3d.proj_transform(x_coord, y_coord, z_coord, ax.get_proj())
        imagebox = OffsetImage(image.squeeze(), zoom=0.55, cmap="gray")
        annotation = AnnotationBbox(imagebox, (x_2d, y_2d), frameon=False, pad=0.0)
        ax.add_artist(annotation)

    ax.set_box_aspect([1, 1, 1])
    ax.set_title("3D Latent Space Visualization with Visible MNIST Digits")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_run(run_dir: Path, data_dir: Path, device: str) -> dict:
    history_path = run_dir / "history.csv"
    best_checkpoint = run_dir / "best_recon_checkpoint.pt"
    if not history_path.exists() or not best_checkpoint.exists():
        return {"status": "missing_artifacts", "run_dir": str(run_dir)}

    history_rows = read_csv(history_path)
    model, _, run_spec = load_model_from_checkpoint(best_checkpoint, device=device)
    _, eval_loader = build_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=run_spec.batch_size,
        seed=run_spec.seed,
    )

    generate_convergence_plot(
        history_rows=history_rows,
        output_path=run_dir / "convergence.png",
        title=f"{run_spec.model_family} | reported d={run_spec.reported_dim} | seed={run_spec.seed}",
    )
    generate_reconstruction_panel(model, eval_loader, device, run_dir / "reconstructions.png")

    if run_spec.model_family == "spcauchy" and run_spec.ambient_latent_dim == 3:
        generate_interpolation_panel(model, eval_loader, device, run_dir / "interpolation_1_to_7.png")
        generate_posterior_sphere_plot(model, eval_loader, device, run_dir / "posterior_sphere.png")
        generate_posterior_sphere_side_by_side_plot(
            model,
            eval_loader,
            device,
            run_dir / "posterior_sphere_side_by_side.png",
        )
        generate_uniform_sphere_decodes(model, device, run_dir / "decoded_uniform_sphere.png")
        generate_visible_only_sphere_with_images(
            model,
            device,
            run_dir / "decoded_uniform_sphere_visible_only.png",
        )

    return {"status": "plotted", "run_dir": str(run_dir)}


def run_plot_jobs(
    preset: str,
    data_dir: str | Path,
    output_root: str | Path,
    device: str,
    models: list[str] | None = None,
    reported_dims: list[int] | None = None,
    seeds: list[int] | None = None,
    epochs: int | None = None,
) -> list[dict]:
    results = []
    for run_spec in build_specs_for_preset(preset, model_families=models, reported_dims=reported_dims, seeds=seeds, epochs=epochs):
        run_dir = run_spec.output_dir(Path(output_root))
        ensure_dir(run_dir)
        if not checkpoint_exists(run_dir):
            results.append({"status": "skipped", "run_dir": str(run_dir), "reason": "training_artifacts_missing"})
            continue
        results.append(plot_run(run_dir, Path(data_dir), device))
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MNIST experiment plots from saved runs.")
    parser.add_argument("--preset", choices=[BENCHMARK_PRESET, QUALITATIVE_PRESET], required=True)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--models", nargs="+", choices=["gaussian", "vmf", "spcauchy"])
    parser.add_argument("--reported-dims", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)
    results = run_plot_jobs(
        preset=args.preset,
        data_dir=args.data_dir,
        output_root=args.output_root,
        device=device,
        models=args.models,
        reported_dims=args.reported_dims,
        seeds=args.seeds,
        epochs=args.epochs,
    )
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
