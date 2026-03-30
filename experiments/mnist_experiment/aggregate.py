from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path

import matplotlib.pyplot as plt

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mnist_experiment.config import OUTPUT_ROOT  # noqa: E402
from experiments.mnist_experiment.utils import ensure_dir, read_csv, read_json, repo_relative_path, sample_std, write_csv  # noqa: E402


def _collect_run_summaries(output_root: Path) -> list[dict]:
    rows: list[dict] = []
    benchmark_root = output_root / "benchmark"
    if not benchmark_root.exists():
        return rows

    for summary_path in benchmark_root.glob("*/dim_*/seed_*/evaluation_summary.json"):
        run_dir = summary_path.parent
        evaluation = read_json(summary_path)
        selection = read_json(run_dir / "selection_summary.json")
        config = evaluation["run_config"]
        rows.append(
            {
                "model_family": config["model_family"],
                "reported_dim": int(config["reported_dim"]),
                "ambient_latent_dim": int(config["ambient_latent_dim"]),
                "seed": int(config["seed"]),
                "selected_epoch": int(selection["selected_epoch"]),
                "best_eval_recon_loss": float(evaluation["best_recon_checkpoint"]["eval_recon_loss"]),
                "best_eval_total_loss": float(evaluation["best_recon_checkpoint"]["eval_total_loss"]),
                "best_eval_kl": float(evaluation["best_recon_checkpoint"]["eval_kl"]),
                "final_eval_recon_loss": float(evaluation["final_checkpoint"]["eval_recon_loss"]),
                "final_eval_total_loss": float(evaluation["final_checkpoint"]["eval_total_loss"]),
                "final_eval_kl": float(evaluation["final_checkpoint"]["eval_kl"]),
                "run_dir": repo_relative_path(run_dir),
            }
        )
    return rows


def _aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_family"], int(row["reported_dim"]))].append(row)

    summary_rows = []
    for (model_family, reported_dim), group_rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        summary_rows.append(
            {
                "model_family": model_family,
                "reported_dim": reported_dim,
                "ambient_latent_dim": group_rows[0]["ambient_latent_dim"],
                "num_completed_seeds": len(group_rows),
                "best_eval_recon_loss_mean": sum(row["best_eval_recon_loss"] for row in group_rows) / len(group_rows),
                "best_eval_recon_loss_std": sample_std(row["best_eval_recon_loss"] for row in group_rows),
                "best_eval_total_loss_mean": sum(row["best_eval_total_loss"] for row in group_rows) / len(group_rows),
                "best_eval_total_loss_std": sample_std(row["best_eval_total_loss"] for row in group_rows),
                "best_eval_kl_mean": sum(row["best_eval_kl"] for row in group_rows) / len(group_rows),
                "best_eval_kl_std": sample_std(row["best_eval_kl"] for row in group_rows),
            }
        )
    return summary_rows


def _write_latex_table(summary_rows: list[dict], output_path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & Reported $d$ & Ambient dim & Seeds & Recon mean & Recon std & Total mean & KL mean \\",
        r"\midrule",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['model_family']} & {row['reported_dim']} & {row['ambient_latent_dim']} & "
            f"{row['num_completed_seeds']} & {row['best_eval_recon_loss_mean']:.4f} & "
            f"{row['best_eval_recon_loss_std']:.4f} & {row['best_eval_total_loss_mean']:.4f} & "
            f"{row['best_eval_kl_mean']:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _mean_and_ci(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) <= 1:
        return mean, 0.0
    std = sample_std(values)
    ci = 1.96 * std / math.sqrt(len(values))
    return mean, ci


def _plot_aggregate_convergence(output_root: Path, aggregate_root: Path) -> None:
    benchmark_root = output_root / "benchmark"
    if not benchmark_root.exists():
        return

    grouped_histories: dict[tuple[str, int], list[list[dict]]] = defaultdict(list)
    for history_path in benchmark_root.glob("*/dim_*/seed_*/history.csv"):
        parts = history_path.parts
        model_family = parts[-4]
        reported_dim = int(parts[-3].split("_", maxsplit=1)[1])
        grouped_histories[(model_family, reported_dim)].append(read_csv(history_path))

    for (model_family, reported_dim), histories in grouped_histories.items():
        min_epochs = min(len(history) for history in histories)
        epochs = list(range(1, min_epochs + 1))
        metrics = {
            "train_total_loss": [],
            "eval_total_loss": [],
            "train_recon_loss": [],
            "eval_recon_loss": [],
            "train_kl": [],
            "eval_kl": [],
        }
        for epoch_index in range(min_epochs):
            for metric_name in metrics:
                metrics[metric_name].append(
                    [float(history[epoch_index][metric_name]) for history in histories]
                )

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for axis, metric_names, title in zip(
            axes,
            (
                ("train_total_loss", "eval_total_loss"),
                ("train_recon_loss", "eval_recon_loss"),
                ("train_kl", "eval_kl"),
            ),
            ("Total Loss", "Reconstruction Loss", "KL"),
        ):
            for metric_name, color, label in zip(metric_names, ("tab:blue", "tab:orange"), ("Train", "Eval")):
                stats = [_mean_and_ci(values) for values in metrics[metric_name]]
                means = [mean for mean, _ in stats]
                cis = [ci for _, ci in stats]
                axis.plot(epochs, means, color=color, label=label)
                axis.fill_between(
                    epochs,
                    [mean_value - ci_value for mean_value, ci_value in zip(means, cis)],
                    [mean_value + ci_value for mean_value, ci_value in zip(means, cis)],
                    color=color,
                    alpha=0.18,
                )
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Loss")
            axis.legend()
        fig.suptitle(f"{model_family} | reported d={reported_dim}")
        fig.tight_layout()
        fig.savefig(aggregate_root / f"convergence_{model_family}_dim_{reported_dim}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def aggregate_benchmark_outputs(output_root: str | Path) -> dict:
    output_root = Path(output_root)
    aggregate_root = ensure_dir(output_root / "aggregate")
    long_rows = _collect_run_summaries(output_root)
    summary_rows = _aggregate_rows(long_rows)

    write_csv(aggregate_root / "benchmark_seed_level.csv", long_rows)
    write_csv(aggregate_root / "benchmark_summary.csv", summary_rows)
    _write_latex_table(summary_rows, aggregate_root / "benchmark_summary.tex")
    _plot_aggregate_convergence(output_root, aggregate_root)
    return {
        "aggregate_root": repo_relative_path(aggregate_root),
        "num_seed_rows": len(long_rows),
        "num_summary_rows": len(summary_rows),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate completed MNIST benchmark runs.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(aggregate_benchmark_outputs(args.output_root))


if __name__ == "__main__":
    main()
