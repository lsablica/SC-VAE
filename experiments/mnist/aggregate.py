from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import t as student_t

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mnist.config import OUTPUT_ROOT  # noqa: E402
from experiments.mnist.utils import ensure_dir, read_csv, read_json, repo_relative_path, sample_std, write_csv  # noqa: E402


COMPARISON_ALPHA = 0.05


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return Benjamini-Hochberg adjusted p-values in input order."""
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running_minimum = 1.0
    num_tests = len(p_values)
    for reverse_rank in range(num_tests - 1, -1, -1):
        original_index = order[reverse_rank]
        rank = reverse_rank + 1
        candidate = p_values[original_index] * num_tests / rank
        running_minimum = min(running_minimum, candidate)
        adjusted[original_index] = min(1.0, running_minimum)
    return adjusted


def _paired_reconstruction_test(
    winner_rows: list[dict], runner_up_rows: list[dict]
) -> dict | None:
    """Compare reconstruction by matching identical seed indices."""
    winner_by_seed = {int(row["seed"]): row for row in winner_rows}
    runner_by_seed = {int(row["seed"]): row for row in runner_up_rows}
    common_seeds = sorted(set(winner_by_seed) & set(runner_by_seed))
    if len(common_seeds) < 2:
        return None

    # Positive differences favor the lower-loss winner.
    differences = [
        float(runner_by_seed[seed]["best_eval_recon_loss"])
        - float(winner_by_seed[seed]["best_eval_recon_loss"])
        for seed in common_seeds
    ]
    mean_difference = sum(differences) / len(differences)
    difference_std = sample_std(differences)
    if difference_std == 0.0:
        p_value = 1.0 if mean_difference == 0.0 else 0.0
        ci_low = mean_difference
        ci_high = mean_difference
    else:
        standard_error = difference_std / math.sqrt(len(differences))
        t_statistic = mean_difference / standard_error
        degrees_of_freedom = len(differences) - 1
        p_value = float(
            2.0 * student_t.sf(abs(t_statistic), degrees_of_freedom)
        )
        half_width = float(
            student_t.ppf(
                1.0 - COMPARISON_ALPHA / 2.0, degrees_of_freedom
            )
            * standard_error
        )
        ci_low = mean_difference - half_width
        ci_high = mean_difference + half_width
    return {
        "paired_num_seeds": len(common_seeds),
        "paired_improvement_mean": mean_difference,
        "paired_improvement_ci_low": ci_low,
        "paired_improvement_ci_high": ci_high,
        "paired_p_value": p_value,
    }


def _comparison_defaults() -> dict:
    return {
        "bold_best_mean": False,
        "dagger_paired_bh": False,
        "paired_runner_up": None,
        "paired_num_seeds": None,
        "paired_improvement_mean": None,
        "paired_improvement_ci_low": None,
        "paired_improvement_ci_high": None,
        "paired_p_value": None,
        "paired_q_value": None,
    }


def _annotate_reconstruction_comparisons(
    summary_rows: list[dict],
    grouped_seed_rows: dict[tuple[str, int], list[dict]],
) -> None:
    tests: list[tuple[dict, float]] = []
    for reported_dim in sorted(
        {row["reported_dim"] for row in summary_rows}
    ):
        candidates = [
            row
            for row in summary_rows
            if row["reported_dim"] == reported_dim
            and row["best_eval_recon_loss_mean"] is not None
        ]
        if len(candidates) < 2:
            continue
        ordered = sorted(
            candidates,
            key=lambda row: row["best_eval_recon_loss_mean"],
        )
        winner, runner_up = ordered[:2]
        winner["bold_best_mean"] = True
        comparison = _paired_reconstruction_test(
            grouped_seed_rows[
                (winner["model_family"], int(winner["reported_dim"]))
            ],
            grouped_seed_rows[
                (runner_up["model_family"], int(runner_up["reported_dim"]))
            ],
        )
        if comparison is None:
            continue
        winner.update(comparison)
        winner["paired_runner_up"] = runner_up["model_family"]
        tests.append((winner, comparison["paired_p_value"]))

    adjusted = _benjamini_hochberg(
        [p_value for _, p_value in tests]
    )
    for (winner, _), q_value in zip(tests, adjusted):
        winner["paired_q_value"] = q_value
        winner["dagger_paired_bh"] = bool(
            q_value < COMPARISON_ALPHA
            and winner["paired_improvement_ci_low"] > 0.0
        )


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
                "concentration_mean": (
                    float(evaluation["best_recon_checkpoint"]["concentration"]["mean"])
                    if evaluation["best_recon_checkpoint"].get("concentration")
                    else None
                ),
                "concentration_q95": (
                    float(evaluation["best_recon_checkpoint"]["concentration"]["q95"])
                    if evaluation["best_recon_checkpoint"].get("concentration")
                    else None
                ),
                "concentration_max": (
                    float(evaluation["best_recon_checkpoint"]["concentration"]["max"])
                    if evaluation["best_recon_checkpoint"].get("concentration")
                    else None
                ),
                "wall_clock_training_s": float(
                    selection.get("wall_clock_training_s", 0.0)
                ),
                "final_eval_recon_loss": float(evaluation["final_checkpoint"]["eval_recon_loss"]),
                "final_eval_total_loss": float(evaluation["final_checkpoint"]["eval_total_loss"]),
                "final_eval_kl": float(evaluation["final_checkpoint"]["eval_kl"]),
                "run_dir": repo_relative_path(run_dir),
            }
        )
    return rows


def _collect_failures(output_root: Path) -> list[dict]:
    failures: list[dict] = []
    benchmark_root = output_root / "benchmark"
    if not benchmark_root.exists():
        return failures
    for failure_path in benchmark_root.glob(
        "*/dim_*/seed_*/failure_summary.json"
    ):
        if (failure_path.parent / "evaluation_summary.json").exists():
            continue
        payload = read_json(failure_path)
        config = payload["run_config"]
        failures.append(
            {
                "model_family": config["model_family"],
                "reported_dim": int(config["reported_dim"]),
                "ambient_latent_dim": int(config["ambient_latent_dim"]),
                "seed": int(config["seed"]),
                "failure_type": payload["failure_type"],
                "error_message": payload["error_message"],
                "completed_epochs": int(payload["completed_epochs"]),
                "wall_clock_training_s": float(
                    payload["wall_clock_training_s"]
                ),
                "run_dir": repo_relative_path(failure_path.parent),
            }
        )
    for failure_path in benchmark_root.glob(
        "*/dim_*/seed_*/evaluation_failure.json"
    ):
        if (failure_path.parent / "evaluation_summary.json").exists():
            continue
        payload = read_json(failure_path)
        config = read_json(failure_path.parent / "run_config.json")
        failures.append(
            {
                "model_family": config["model_family"],
                "reported_dim": int(config["reported_dim"]),
                "ambient_latent_dim": int(config["ambient_latent_dim"]),
                "seed": int(config["seed"]),
                "failure_type": "evaluation_"
                + payload["failure_type"],
                "error_message": payload["error_message"],
                "completed_epochs": int(config["epochs"]),
                "wall_clock_training_s": None,
                "run_dir": repo_relative_path(failure_path.parent),
            }
        )
    return failures


def _aggregate_rows(rows: list[dict], failures: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    grouped_failures: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_family"], int(row["reported_dim"]))].append(row)
    for row in failures:
        grouped_failures[
            (row["model_family"], int(row["reported_dim"]))
        ].append(row)

    summary_rows = []
    keys = sorted(
        set(grouped) | set(grouped_failures),
        key=lambda item: (item[1], item[0]),
    )
    for model_family, reported_dim in keys:
        group_rows = grouped[(model_family, reported_dim)]
        failure_rows = grouped_failures[(model_family, reported_dim)]
        if not group_rows:
            summary_rows.append(
                {
                    "model_family": model_family,
                    "reported_dim": reported_dim,
                    "ambient_latent_dim": failure_rows[0][
                        "ambient_latent_dim"
                    ],
                    "num_completed_seeds": 0,
                    "num_failed_seeds": len(failure_rows),
                    "nonfinite_run_count": sum(
                        row["failure_type"] == "NonFiniteTrainingError"
                        for row in failure_rows
                    ),
                    "best_eval_recon_loss_mean": None,
                    "best_eval_recon_loss_std": None,
                    "best_eval_recon_loss_min": None,
                    "best_eval_recon_loss_max": None,
                    "best_eval_total_loss_mean": None,
                    "best_eval_total_loss_std": None,
                    "best_eval_kl_mean": None,
                    "best_eval_kl_std": None,
                    "selected_epoch_mean": None,
                    "selected_epoch_min": None,
                    "selected_epoch_max": None,
                    "concentration_mean": None,
                    "concentration_q95_mean": None,
                    "concentration_max": None,
                    "wall_clock_training_s_mean": None,
                    **_comparison_defaults(),
                }
            )
            continue
        concentration_values = [
            row["concentration_mean"]
            for row in group_rows
            if row["concentration_mean"] is not None
        ]
        concentration_q95_values = [
            row["concentration_q95"]
            for row in group_rows
            if row["concentration_q95"] is not None
        ]
        concentration_max_values = [
            row["concentration_max"]
            for row in group_rows
            if row["concentration_max"] is not None
        ]
        selected_epochs = [row["selected_epoch"] for row in group_rows]
        summary_rows.append(
            {
                "model_family": model_family,
                "reported_dim": reported_dim,
                "ambient_latent_dim": group_rows[0]["ambient_latent_dim"],
                "num_completed_seeds": len(group_rows),
                "num_failed_seeds": len(failure_rows),
                "nonfinite_run_count": sum(
                    row["failure_type"] == "NonFiniteTrainingError"
                    for row in failure_rows
                ),
                "best_eval_recon_loss_mean": sum(row["best_eval_recon_loss"] for row in group_rows) / len(group_rows),
                "best_eval_recon_loss_std": sample_std(row["best_eval_recon_loss"] for row in group_rows),
                "best_eval_recon_loss_min": min(
                    row["best_eval_recon_loss"] for row in group_rows
                ),
                "best_eval_recon_loss_max": max(
                    row["best_eval_recon_loss"] for row in group_rows
                ),
                "best_eval_total_loss_mean": sum(row["best_eval_total_loss"] for row in group_rows) / len(group_rows),
                "best_eval_total_loss_std": sample_std(row["best_eval_total_loss"] for row in group_rows),
                "best_eval_kl_mean": sum(row["best_eval_kl"] for row in group_rows) / len(group_rows),
                "best_eval_kl_std": sample_std(row["best_eval_kl"] for row in group_rows),
                "selected_epoch_mean": sum(selected_epochs)
                / len(selected_epochs),
                "selected_epoch_min": min(selected_epochs),
                "selected_epoch_max": max(selected_epochs),
                "concentration_mean": (
                    sum(concentration_values) / len(concentration_values)
                    if concentration_values
                    else None
                ),
                "concentration_q95_mean": (
                    sum(concentration_q95_values)
                    / len(concentration_q95_values)
                    if concentration_q95_values
                    else None
                ),
                "concentration_max": (
                    max(concentration_max_values)
                    if concentration_max_values
                    else None
                ),
                "wall_clock_training_s_mean": sum(
                    row["wall_clock_training_s"] for row in group_rows
                )
                / len(group_rows),
                **_comparison_defaults(),
            }
        )

    _annotate_reconstruction_comparisons(summary_rows, grouped)
    return summary_rows


def _write_latex_table(summary_rows: list[dict], output_path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Reported $d$ & Ambient dim & Seeds & Reconstruction & Total mean & KL mean \\",
        r"\midrule",
    ]
    for row in summary_rows:
        reconstruction = (
            f"{row['best_eval_recon_loss_mean']:.4f} "
            f"$\\pm$ {row['best_eval_recon_loss_std']:.4f}"
            if row["best_eval_recon_loss_mean"] is not None
            else "failed"
        )
        if row["bold_best_mean"]:
            reconstruction = f"\\textbf{{{reconstruction}}}"
        if row["dagger_paired_bh"]:
            reconstruction += r"$^{\dagger}$"
        lines.append(
            f"{row['model_family']} & {row['reported_dim']} & {row['ambient_latent_dim']} & "
            f"{row['num_completed_seeds']} & {reconstruction} & "
            f"{row['best_eval_total_loss_mean']:.4f} & "
            f"{row['best_eval_kl_mean']:.4f} \\\\"
            if row["best_eval_total_loss_mean"] is not None
            else (
                f"{row['model_family']} & {row['reported_dim']} & "
                f"{row['ambient_latent_dim']} & 0 & failed & n/a & n/a \\\\"
            )
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


def _plot_reconstruction_summary(
    summary_rows: list[dict], aggregate_root: Path
) -> None:
    valid = [
        row
        for row in summary_rows
        if row["best_eval_recon_loss_mean"] is not None
    ]
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    labels = {
        "gaussian": "Gaussian",
        "spcauchy": "Spherical Cauchy",
        "vmf": "vMF",
        "powerspherical": "Power Spherical",
    }
    for model_family in labels:
        rows = sorted(
            [
                row
                for row in valid
                if row["model_family"] == model_family
            ],
            key=lambda row: row["reported_dim"],
        )
        if not rows:
            continue
        ax.errorbar(
            [row["reported_dim"] for row in rows],
            [row["best_eval_recon_loss_mean"] for row in rows],
            yerr=[row["best_eval_recon_loss_std"] for row in rows],
            marker="o",
            capsize=3,
            label=labels[model_family],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 3, 5, 10, 20])
    ax.set_xticklabels(["2", "3", "5", "10", "20"])
    ax.set_xlabel("reported latent dimension p")
    ax.set_ylabel("held-out reconstruction loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    for filename in [
        "power_spherical_mnist_reconstruction.png",
        "mnist_reconstruction_comparison.png",
    ]:
        fig.savefig(aggregate_root / filename, dpi=220)
    plt.close(fig)


def aggregate_benchmark_outputs(output_root: str | Path) -> dict:
    output_root = Path(output_root)
    aggregate_root = ensure_dir(output_root / "aggregate")
    long_rows = _collect_run_summaries(output_root)
    failure_rows = _collect_failures(output_root)
    summary_rows = _aggregate_rows(long_rows, failure_rows)

    write_csv(aggregate_root / "benchmark_seed_level.csv", long_rows)
    write_csv(aggregate_root / "benchmark_failures.csv", failure_rows)
    write_csv(aggregate_root / "benchmark_summary.csv", summary_rows)
    _write_latex_table(summary_rows, aggregate_root / "benchmark_summary.tex")
    _plot_aggregate_convergence(output_root, aggregate_root)
    _plot_reconstruction_summary(summary_rows, aggregate_root)
    return {
        "aggregate_root": repo_relative_path(aggregate_root),
        "num_seed_rows": len(long_rows),
        "num_summary_rows": len(summary_rows),
        "num_failure_rows": len(failure_rows),
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
