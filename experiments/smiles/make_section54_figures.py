from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the final Section 5.4 figures.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("experiments/smiles/results/SC-VAE-runs-zinc/zinc250k"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/smiles/main_figures"),
    )
    parser.add_argument("--run-id", default="section54_zinc250k_main")
    parser.add_argument("--model-name", default="spcauchy-128")
    return parser


def render_molecule(smiles: str, width: int = 280, height: int = 210) -> np.ndarray | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    options.bondLineWidth = 1.6
    options.fixedBondLength = 28
    options.additionalAtomLabelPadding = 0.15
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return mpimg.imread(io.BytesIO(drawer.GetDrawingText()), format="png")


def set_panel_border(ax: plt.Axes, *, endpoint: bool) -> None:
    edgecolor = "#1f1f1f" if endpoint else "#4d4d4d"
    linewidth = 1.6 if endpoint else 0.9
    ax.set_facecolor("white")
    ax.patch.set_alpha(1.0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
        spine.set_edgecolor(edgecolor)


def create_training_figure(output_dir: Path, runs_root: Path, run_id: str, model_name: str) -> None:
    latent_dir = "latent_128" if model_name == "spcauchy-128" else "latent_64"
    histories = []
    for seed_dir in sorted((runs_root / model_name / latent_dir).glob("seed_*")):
        history_path = seed_dir / run_id / "metrics" / "train_history.csv"
        if not history_path.exists():
            continue
        frame = pd.read_csv(history_path)
        frame["seed"] = int(seed_dir.name.split("_")[-1])
        histories.append(frame)
    if not histories:
        return

    history = pd.concat(histories, ignore_index=True)
    summary = (
        history.groupby("epoch")
        .agg(
            train_recon_mean=("train_recon_loss", "mean"),
            train_recon_std=("train_recon_loss", "std"),
            val_recon_mean=("val_recon_loss", "mean"),
            val_recon_std=("val_recon_loss", "std"),
            train_kl_mean=("train_kl_loss", "mean"),
            train_kl_std=("train_kl_loss", "std"),
            val_kl_mean=("val_kl_loss", "mean"),
            val_kl_std=("val_kl_loss", "std"),
            train_loss_mean=("train_loss", "mean"),
            train_loss_std=("train_loss", "std"),
            val_loss_mean=("val_loss", "mean"),
            val_loss_std=("val_loss", "std"),
        )
        .reset_index()
        .fillna(0.0)
    )

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), facecolor="white")
    specs = [
        ("Reconstruction Loss", "train_recon", "val_recon"),
        ("KL Divergence", "train_kl", "val_kl"),
        ("Validation Objective", "train_loss", "val_loss"),
    ]
    colors = {"train": "#1f77b4", "val": "#d55e00"}

    for ax, (title, train_prefix, val_prefix) in zip(axes, specs):
        epochs = summary["epoch"].to_numpy()
        for prefix, label in ((train_prefix, "train"), (val_prefix, "validation")):
            mean = summary[f"{prefix}_mean"].to_numpy()
            std = summary[f"{prefix}_std"].to_numpy()
            color = colors["train" if label == "train" else "val"]
            ax.plot(epochs, mean, color=color, linewidth=2.0, label=label)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Mean ± s.d. across seeds")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("spCauchy-128 Training Dynamics on ZINC-250k", fontsize=16, y=1.02)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "section54_spcauchy_training_curves.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "section54_spcauchy_training_curves.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def create_interpolation_showcase(output_dir: Path, runs_root: Path, run_id: str, model_name: str) -> None:
    chosen = {
        "seed": 1,
        "pair_id": 32,
        "reference_cosine": 0.315021,
        "valid_fraction": 0.5454545454545454,
        "path_novelty": 1.0,
        "smoothness": 0.875,
        "display_steps": [0, 1, 7, 8, 10],
    }
    latent_dir = "latent_128" if model_name == "spcauchy-128" else "latent_64"
    interpolation_dir = runs_root / model_name / latent_dir / f"seed_{chosen['seed']}" / run_id / "interpolation"
    steps = pd.read_csv(interpolation_dir / "interpolation_steps.csv")
    row_steps = (
        steps.loc[(steps["pair_id"] == chosen["pair_id"]) & (steps["step"].isin(chosen["display_steps"]))]
        .sort_values("step")
        .reset_index(drop=True)
    )
    endpoint_a = str(row_steps.iloc[0]["endpoint_a"])
    endpoint_b = str(row_steps.iloc[-1]["endpoint_b"])

    fig = plt.figure(figsize=(19.2, 3.8), facecolor="white")
    grid = fig.add_gridspec(1, len(chosen["display_steps"]) + 1, width_ratios=[2.4] + [1] * len(chosen["display_steps"]), wspace=0.08)
    label_ax = fig.add_subplot(grid[0, 0])
    label_ax.axis("off")
    label_ax.text(
        0.0,
        0.55,
        (
            f"Representative geodesic interpolation,\ncosine = {chosen['reference_cosine']:.3f}\n"
            f"valid path = {int(round(chosen['valid_fraction'] * 11))}/11\n"
            f"novelty = {chosen['path_novelty']:.2f}\n"
            f"smoothness = {chosen['smoothness']:.2f}"
        ),
        ha="left",
        va="center",
        fontsize=13,
        color="#111111",
        linespacing=1.5,
    )

    for col_idx, step_value in enumerate(chosen["display_steps"], start=1):
        ax = fig.add_subplot(grid[0, col_idx])
        step_row = row_steps.loc[row_steps["step"] == step_value].iloc[0]
        if step_value == chosen["display_steps"][0]:
            smiles_to_draw = endpoint_a
        elif step_value == chosen["display_steps"][-1]:
            smiles_to_draw = endpoint_b
        else:
            smiles_to_draw = step_row["canonical_smiles"] if isinstance(step_row["canonical_smiles"], str) and step_row["canonical_smiles"] else step_row["decoded_smiles"]
        image = render_molecule(smiles_to_draw)
        if image is not None:
            ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        set_panel_border(ax, endpoint=step_value in (chosen["display_steps"][0], chosen["display_steps"][-1]))
        ax.set_title(f"t={step_value}", fontsize=10, pad=5)

    fig.suptitle("Representative Geodesic Interpolation in spCauchy-128", fontsize=17, y=1.02)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "section54_interpolation_showcase.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "section54_interpolation_showcase.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    create_training_figure(args.output_dir, args.runs_root, args.run_id, args.model_name)
    create_interpolation_showcase(args.output_dir, args.runs_root, args.run_id, args.model_name)
    print(f"Saved figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
