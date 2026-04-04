from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw
    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover - fallback for environments without drawing support
    Chem = None
    Draw = None


def plot_training_curves(history: pd.DataFrame, output_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history["epoch"], history["train_recon_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_recon_loss"], label="val")
    axes[0].set_title("Reconstruction Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_kl_loss"], label="train")
    axes[1].plot(history["epoch"], history["val_kl_loss"], label="val")
    axes[1].set_title("KL Divergence")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history["epoch"], history["train_loss"], label="train")
    axes[2].plot(history["epoch"], history["val_loss"], label="val")
    axes[2].set_title("ELBO")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_property_histograms(hist_frame: pd.DataFrame, output_path: str | Path) -> None:
    if hist_frame.empty:
        return
    properties = list(hist_frame["property"].unique())
    fig, axes = plt.subplots(len(properties), 1, figsize=(8, 3 * len(properties)))
    if len(properties) == 1:
        axes = [axes]
    for ax, property_name in zip(axes, properties):
        frame = hist_frame.loc[hist_frame["property"] == property_name]
        centers = 0.5 * (frame["bin_left"] + frame["bin_right"])
        ax.plot(centers, frame["generated_density"], label="generated")
        ax.plot(centers, frame["reference_density"], label="reference")
        ax.set_title(property_name)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_interpolation_bin_summary(summary_frame: pd.DataFrame, output_path: str | Path) -> None:
    metrics = [column for column in summary_frame.columns if column.endswith("_mean") and column != "reference_cosine_mean"]
    if summary_frame.empty or not metrics:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(summary_frame))
    width = max(0.08, 0.8 / max(len(metrics), 1))
    for idx, metric in enumerate(metrics):
        ax.bar(
            [position + width * idx for position in x],
            summary_frame[metric],
            width=width,
            label=metric.replace("_mean", ""),
        )
    ax.set_xticks([position + width * (len(metrics) - 1) / 2 for position in x])
    ax.set_xticklabels(summary_frame["bin"], rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_representative_interpolations(step_frame: pd.DataFrame, summary_frame: pd.DataFrame, output_path: str | Path) -> None:
    if step_frame.empty or summary_frame.empty:
        return
    representatives = []
    for bin_name in summary_frame["bin"].unique():
        bin_summary = summary_frame.loc[summary_frame["bin"] == bin_name].sort_values("valid_fraction", ascending=False)
        if not bin_summary.empty:
            representatives.append((bin_name, int(bin_summary.iloc[0]["pair_id"])))
    if not representatives:
        return

    max_steps = int(step_frame["step"].max()) + 1
    fig, axes = plt.subplots(len(representatives), max_steps, figsize=(2.2 * max_steps, 2.6 * len(representatives)))
    if len(representatives) == 1:
        axes = [axes]

    for row_idx, (bin_name, pair_id) in enumerate(representatives):
        pair_steps = step_frame.loc[step_frame["pair_id"] == pair_id].sort_values("step")
        row_axes = axes[row_idx]
        for col_idx, (_, step_row) in enumerate(pair_steps.iterrows()):
            ax = row_axes[col_idx]
            smiles = step_row["canonical_smiles"] if isinstance(step_row["canonical_smiles"], str) else step_row["decoded_smiles"]
            drawn = False
            if Chem is not None and Draw is not None and isinstance(smiles, str):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    image = Draw.MolToImage(mol, size=(220, 180))
                    ax.imshow(image)
                    drawn = True
            if not drawn:
                ax.text(0.5, 0.5, str(smiles), ha="center", va="center", wrap=True, fontsize=8)
            ax.set_axis_off()
            title = f"t={int(step_row['step'])}"
            if row_idx == 0:
                ax.set_title(title)
        for ax in row_axes[len(pair_steps):]:
            ax.set_axis_off()
        row_axes[0].set_ylabel(bin_name, rotation=90, fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
