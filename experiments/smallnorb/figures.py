"""Publication figures for the locked smallNORB comparison."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import t
from sklearn.decomposition import PCA

from .config import FINAL_ROOT
from .data import SmallNORBViewDataset
from .evaluate import load_checkpoint
from .interpolate import (
    PATH_AZIMUTH_DEGREES,
    PATH_AZIMUTH_INDICES,
)
from .metrics import slerp
from .utils import ensure_dir, read_csv, read_json, resolve_device

FAMILY_LABELS = {
    "spcauchy": "Spherical Cauchy",
    "vmf_robust": "Robust vMF",
    "powerspherical": "Power Spherical",
    "gaussian_isotropic": "Isotropic Gaussian",
    "gaussian_diagonal": "Diagonal Gaussian",
}
FAMILY_COLORS = {
    "spcauchy": "#0072B2",
    "vmf_robust": "#D55E00",
    "powerspherical": "#009E73",
    "gaussian_isotropic": "#CC79A7",
    "gaussian_diagonal": "#7F7F7F",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 350,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_both(figure: plt.Figure, stem: Path) -> None:
    ensure_dir(stem.parent)
    figure.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _normalize_trajectory_by_max(values: list[float]) -> np.ndarray:
    """Scale one seed trajectory by its finite maximum over training."""
    trajectory = np.asarray(values, dtype=float)
    finite = trajectory[np.isfinite(trajectory)]
    if finite.size == 0:
        return trajectory
    maximum = float(finite.max())
    if maximum <= 0.0:
        raise ValueError("posterior concentration maxima must be positive")
    return trajectory / maximum


def _seed_rows() -> list[dict[str, Any]]:
    rows = read_json(FINAL_ROOT / "tables" / "seed_level.json")
    return rows


def _representative_sc_row(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    spherical = [row for row in rows if row["family"] == "spcauchy"]
    median_value = float(
        np.median(
            [row["test_gap_reconstruction_nll"] for row in spherical]
        )
    )
    return min(
        spherical,
        key=lambda row: (
            abs(row["test_gap_reconstruction_nll"] - median_value),
            row["seed"],
        ),
    )


def _row_for(
    rows: list[dict[str, Any]], family: str, seed: int
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["family"] == family and int(row["seed"]) == seed
    )


def _path_indices_for_key(dataset, key):
    lookup = {}
    target = tuple(int(value) for value in key)
    for local_index, source_index in enumerate(dataset.indices):
        current = (
            int(dataset.metadata["category"][source_index]),
            int(dataset.metadata["instance"][source_index]),
            int(dataset.metadata["elevation"][source_index]),
            int(dataset.metadata["lighting"][source_index]),
        )
        if current == target:
            lookup[
                int(dataset.metadata["azimuth_index"][source_index])
            ] = local_index
    return [lookup[index] for index in PATH_AZIMUTH_INDICES]


@torch.no_grad()
def _decode_path(
    run_dir: Path,
    key: tuple[int, int, int, int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model, _, config = load_checkpoint(
        run_dir / "checkpoint_best.pt", device
    )
    dataset = SmallNORBViewDataset(
        config.data_root,
        "test",
        gap_azimuth_indices=config.gap_azimuth_indices,
    )
    indices = _path_indices_for_key(dataset, key)
    truth = torch.stack([dataset[index][0] for index in indices]).to(
        device
    )
    start = model.encode(truth[:1]).location
    end = model.encode(truth[-1:]).location
    latent = []
    for fraction in torch.linspace(0.0, 1.0, 6, device=device):
        if model.posterior.is_spherical:
            latent.append(slerp(start, end, fraction))
        else:
            value = (1.0 - fraction) * start + fraction * end
            latent.append(F.pad(value, (0, 1), value=0.0))
    decoded = model.decode(torch.cat(latent, dim=0))
    return (
        truth[:, 0].cpu().numpy(),
        decoded[:, 0].cpu().numpy(),
    )


def angular_gap_figure(
    rows: list[dict[str, Any]],
    device: torch.device,
) -> tuple[int, tuple[int, int, int, int]]:
    representative = _representative_sc_row(rows)
    seed = int(representative["seed"])
    gallery = np.load(
        Path(representative["run_dir"])
        / "interpolation_representative_test.npz"
    )
    key = tuple(int(value) for value in gallery["key"])
    families = (
        "spcauchy",
        "vmf_robust",
        "powerspherical",
        "gaussian_isotropic",
    )
    decoded_rows = {}
    ground_truth = None
    for family in families:
        row = _row_for(rows, family, seed)
        truth, decoded = _decode_path(
            Path(row["run_dir"]), key, device
        )
        ground_truth = truth
        decoded_rows[family] = decoded
    figure, axes = plt.subplots(
        5, 6, figsize=(8.2, 6.8), constrained_layout=True
    )
    display_rows = [("ground_truth", ground_truth)] + [
        (family, decoded_rows[family]) for family in families
    ]
    for row_index, (family, images) in enumerate(display_rows):
        for column in range(6):
            axes[row_index, column].imshow(
                images[column], cmap="gray", vmin=0, vmax=1
            )
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            if row_index == 0:
                axes[row_index, column].set_title(
                    f"{PATH_AZIMUTH_DEGREES[column]}°"
                )
            if column == 0:
                label = (
                    "Ground truth"
                    if family == "ground_truth"
                    else FAMILY_LABELS[family]
                )
                axes[row_index, column].set_ylabel(label)
            if column in {1, 2, 3, 4}:
                for spine in axes[row_index, column].spines.values():
                    spine.set_edgecolor("#D55E00")
                    spine.set_linewidth(1.2)
    figure.suptitle(
        "Interpolation through the held-out wraparound sector"
    )
    _save_both(
        figure, FINAL_ROOT / "figures" / "angular_gap_reconstruction"
    )
    return seed, key


@torch.no_grad()
def _object_locations(
    run_dir: Path,
    key: tuple[int, int, int, int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model, _, config = load_checkpoint(
        run_dir / "checkpoint_best.pt", device
    )
    dataset = SmallNORBViewDataset(config.data_root, "test")
    indices_by_azimuth = {}
    target = tuple(key)
    for local_index, source_index in enumerate(dataset.indices):
        current = (
            int(dataset.metadata["category"][source_index]),
            int(dataset.metadata["instance"][source_index]),
            int(dataset.metadata["elevation"][source_index]),
            int(dataset.metadata["lighting"][source_index]),
        )
        if current == target:
            indices_by_azimuth[
                int(dataset.metadata["azimuth_index"][source_index])
            ] = local_index
    indices = [indices_by_azimuth[index] for index in range(18)]
    images = torch.stack([dataset[index][0] for index in indices]).to(
        device
    )
    locations = model.encode(images).location.cpu().numpy()
    return locations, np.arange(18) * 20


def latent_geometry_figure(
    rows: list[dict[str, Any]],
    seed: int,
    key: tuple[int, int, int, int],
    device: torch.device,
) -> None:
    spherical = _row_for(rows, "spcauchy", seed)
    locations, degrees = _object_locations(
        Path(spherical["run_dir"]), key, device
    )
    projected = PCA(n_components=2).fit_transform(locations)
    pairs = read_csv(
        Path(spherical["run_dir"]) / "geometry_pairs_test.csv"
    )
    true_distance = np.asarray(
        [float(row["true_circular_distance_radians"]) for row in pairs]
    )
    latent_distance = np.asarray(
        [float(row["latent_distance"]) for row in pairs]
    )
    if len(pairs) > 5000:
        selection = np.random.default_rng(0).choice(
            len(pairs), size=5000, replace=False
        )
        true_plot = true_distance[selection]
        latent_plot = latent_distance[selection]
    else:
        true_plot, latent_plot = true_distance, latent_distance
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    colors = plt.cm.hsv(degrees / 360.0)
    axes[0].plot(projected[:, 0], projected[:, 1], color="0.7", zorder=0)
    axes[0].scatter(
        projected[:, 0],
        projected[:, 1],
        c=colors,
        edgecolor="black",
        linewidth=0.3,
    )
    for index, degree in enumerate(degrees):
        axes[0].annotate(
            str(degree), projected[index], fontsize=6, xytext=(2, 2),
            textcoords="offset points"
        )
    axes[0].set_title("Posterior location PCA")
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")

    axes[1].scatter(
        true_plot,
        latent_plot,
        s=4,
        alpha=0.12,
        color=FAMILY_COLORS["spcauchy"],
        rasterized=True,
    )
    distance_levels = np.unique(np.round(true_distance, decimals=8))
    trends = [
        latent_distance[
            np.isclose(true_distance, level, rtol=0.0, atol=1e-7)
        ].mean()
        for level in distance_levels
    ]
    axes[1].plot(
        distance_levels, trends, color="black", linewidth=1.5
    )
    axes[1].set_title(
        f"Distance alignment, ρ={spherical['geometry_test_spearman']:.2f}"
    )
    axes[1].set_xlabel("True circular distance")
    axes[1].set_ylabel("Latent geodesic distance")

    families = [family for family in FAMILY_LABELS if any(
        row["family"] == family for row in rows
    )]
    for position, family in enumerate(families):
        values = np.asarray(
            [
                row["test_gap_reconstruction_nll"]
                for row in rows
                if row["family"] == family
            ]
        )
        axes[2].scatter(
            np.full_like(values, position, dtype=float)
            + np.linspace(-0.08, 0.08, len(values)),
            values,
            color=FAMILY_COLORS[family],
            s=18,
        )
        half = (
            t.ppf(0.975, len(values) - 1)
            * values.std(ddof=1)
            / np.sqrt(len(values))
        )
        axes[2].errorbar(
            position,
            values.mean(),
            yerr=half,
            fmt="o",
            color="black",
            capsize=3,
        )
    axes[2].set_xticks(
        range(len(families)),
        [FAMILY_LABELS[family] for family in families],
        rotation=25,
        ha="right",
    )
    axes[2].set_ylabel("Test-gap reconstruction NLL")
    axes[2].set_title("Seed-level gap performance")
    figure.tight_layout()
    _save_both(
        figure, FINAL_ROOT / "figures" / "latent_viewpoint_geometry"
    )


def training_diagnostic_figures(rows: list[dict[str, Any]]) -> None:
    metrics = (
        (
            "validation_gap_reconstruction_nll_mean",
            "Gap reconstruction NLL",
            False,
        ),
        ("validation_kl_mean", "Validation KL", False),
        (
            "validation_posterior_scale_median",
            "Normalized concentration / scale",
            True,
        ),
    )
    stems = ("training_curves", "kl_curves", "concentration_curves")
    for (metric, ylabel, normalize_by_max), stem in zip(metrics, stems):
        figure, axis = plt.subplots(figsize=(5.6, 3.6))
        for family in sorted({row["family"] for row in rows}):
            family_histories = []
            for row in rows:
                if row["family"] != family:
                    continue
                details = read_json(Path(row["run_dir"]) / "history.json")
                values = [
                    detail["flat"].get(metric, np.nan)
                    for detail in details
                ]
                if normalize_by_max:
                    values = _normalize_trajectory_by_max(values)
                family_histories.append(values)
                axis.plot(
                    np.arange(1, len(values) + 1),
                    values,
                    color=FAMILY_COLORS[family],
                    alpha=0.18,
                    linewidth=0.7,
                )
            common = min(map(len, family_histories))
            matrix = np.asarray(
                [values[:common] for values in family_histories],
                dtype=float,
            )
            axis.plot(
                np.arange(1, common + 1),
                np.nanmean(matrix, axis=0),
                color=FAMILY_COLORS[family],
                label=FAMILY_LABELS[family],
                linewidth=1.8,
            )
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        if normalize_by_max:
            axis.set_ylim(0.0, 1.05)
            axis.set_yticks(np.linspace(0.0, 1.0, 5))
        axis.legend(frameon=False)
        figure.tight_layout()
        _save_both(figure, FINAL_ROOT / "figures" / stem)


def pose_polar_figure(
    rows: list[dict[str, Any]], seed: int
) -> None:
    families = sorted({row["family"] for row in rows})
    figure, axes = plt.subplots(
        1,
        len(families),
        figsize=(2.7 * len(families), 2.8),
        subplot_kw={"projection": "polar"},
    )
    for axis, family in zip(np.atleast_1d(axes), families):
        row = _row_for(rows, family, seed)
        predictions = read_csv(
            Path(row["run_dir"]) / "pose_predictions_test.csv"
        )
        truth = np.deg2rad(
            [float(value["true_degrees"]) for value in predictions]
        )
        predicted = np.deg2rad(
            [float(value["predicted_degrees"]) for value in predictions]
        )
        selection = np.linspace(
            0, len(predictions) - 1, min(1500, len(predictions))
        ).astype(int)
        axis.scatter(
            truth[selection],
            np.ones(len(selection)),
            c=predicted[selection],
            cmap="hsv",
            s=3,
            alpha=0.35,
        )
        axis.set_yticks([])
        axis.set_title(FAMILY_LABELS[family])
    figure.suptitle("True azimuth with predicted angle color")
    figure.tight_layout()
    _save_both(figure, FINAL_ROOT / "figures" / "pose_probe_polar")


def distance_matrix_figure(
    rows: list[dict[str, Any]],
    seed: int,
    key: tuple[int, int, int, int],
    device: torch.device,
) -> None:
    families = sorted({row["family"] for row in rows})
    figure, axes = plt.subplots(
        1, len(families), figsize=(2.8 * len(families), 2.8)
    )
    for axis, family in zip(np.atleast_1d(axes), families):
        row = _row_for(rows, family, seed)
        locations, _ = _object_locations(
            Path(row["run_dir"]), key, device
        )
        if family in {"spcauchy", "vmf_robust", "powerspherical"}:
            matrix = np.arccos(
                np.clip(locations @ locations.T, -1.0, 1.0)
            )
        else:
            standardized = (
                locations - locations.mean(axis=0)
            ) / np.maximum(locations.std(axis=0), 1e-8)
            matrix = np.linalg.norm(
                standardized[:, None, :] - standardized[None, :, :],
                axis=-1,
            )
        image = axis.imshow(matrix, cmap="viridis")
        axis.set_title(FAMILY_LABELS[family])
        axis.set_xlabel("Azimuth index")
        if axis is axes[0]:
            axis.set_ylabel("Azimuth index")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    _save_both(figure, FINAL_ROOT / "figures" / "distance_matrices")


def factor_breakdown_figure(rows: list[dict[str, Any]]) -> None:
    factors = ("category", "elevation", "lighting")
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for axis, factor in zip(axes, factors):
        for family in sorted({row["family"] for row in rows}):
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in rows:
                if row["family"] != family:
                    continue
                for record in read_csv(
                    Path(row["run_dir"]) / "evaluation_records_test.csv"
                ):
                    if record["partition"] != "test_gap":
                        continue
                    grouped[int(record[factor])].append(
                        float(record["pixel_mse"])
                    )
            x = sorted(grouped)
            y = [np.mean(grouped[value]) for value in x]
            axis.plot(
                x,
                y,
                marker="o",
                markersize=3,
                color=FAMILY_COLORS[family],
                label=FAMILY_LABELS[family],
            )
        axis.set_xlabel(factor.capitalize())
        axis.set_ylabel("Test-gap pixel MSE")
    axes[0].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    _save_both(figure, FINAL_ROOT / "figures" / "factor_breakdowns")


def generate_all_figures(device: torch.device) -> None:
    _style()
    rows = _seed_rows()
    seed, key = angular_gap_figure(rows, device)
    latent_geometry_figure(rows, seed, key, device)
    training_diagnostic_figures(rows)
    pose_polar_figure(rows, seed)
    distance_matrix_figure(rows, seed, key, device)
    factor_breakdown_figure(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    generate_all_figures(resolve_device(args.device))
    print(FINAL_ROOT / "figures")


if __name__ == "__main__":
    main()
