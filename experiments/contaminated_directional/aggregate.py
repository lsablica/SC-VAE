"""Aggregate seed-level Plan B results and render its diagnostic figure."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.smallnorb.utils import (
    ensure_dir,
    sha256_file,
    write_csv,
    write_json,
)


LABELS = {
    "spcauchy": "Spherical Cauchy",
    "vmf": "vMF",
    "powerspherical": "Power Spherical",
}
COLORS = {
    "spcauchy": "#0072B2",
    "vmf": "#D55E00",
    "powerspherical": "#009E73",
}


def _load_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("kappa_*/epsilon_*/*/*/seed_*/evaluation_summary.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _validate_complete_grid(rows: list[dict[str, Any]]) -> int:
    """Require the same contiguous seed set in every predeclared grid cell."""

    observed: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    shared_budgets = set()
    for row in rows:
        key = (
            int(row["kappa"]),
            float(row["epsilon"]),
            row["objective"],
            row["family"],
        )
        seed = int(row["seed"])
        if seed in observed[key]:
            raise RuntimeError(f"Duplicate result for grid cell {key}, seed {seed}")
        observed[key].add(seed)
        if not np.isclose(
            float(row["initial_curvature"]),
            float(row["kappa"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Final controlled-target fits must use target-matched "
                f"initial curvature: {key}, seed {seed}"
            )
        if not np.isclose(
            float(row["initial_location_angle_degrees"]),
            10.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Final controlled-target fits must use the shared fixed "
                f"10-degree location offset: {key}, seed {seed}"
            )
        if int(row.get("nonfinite_count", 0)) != 0:
            raise RuntimeError(
                f"Nonfinite diagnostics in grid cell {key}, seed {seed}"
            )
        shared_budgets.add(
            (
                int(row["steps"]),
                int(row["batch_size"]),
                int(row["evaluation_samples"]),
                float(row["learning_rate"]),
            )
        )
    expected_cells = {
        (kappa, epsilon, objective, family)
        for kappa in (20, 100, 500)
        for epsilon in (0.0, 0.01, 0.05, 0.10, 0.20)
        for objective in ("forward_kl", "reverse_kl")
        for family in ("spcauchy", "vmf", "powerspherical")
    }
    missing_cells = expected_cells - set(observed)
    extra_cells = set(observed) - expected_cells
    if missing_cells or extra_cells:
        raise RuntimeError(
            "Incomplete or unexpected controlled-target grid: "
            f"missing={sorted(missing_cells)}, extra={sorted(extra_cells)}"
        )
    seed_sets = {tuple(sorted(seeds)) for seeds in observed.values()}
    if len(seed_sets) != 1:
        raise RuntimeError(
            f"Every controlled-target cell must share one seed set: {seed_sets}"
        )
    seeds = next(iter(seed_sets))
    if seeds != tuple(range(len(seeds))):
        raise RuntimeError(
            f"Seeds must be contiguous indices starting at zero, got {seeds}"
        )
    if len(shared_budgets) != 1:
        raise RuntimeError(
            "Every controlled-target cell must use one shared optimization "
            f"and evaluation budget, got {shared_budgets}"
        )
    return len(seeds)


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["kappa"],
            row["epsilon"],
            row["objective"],
            row["family"],
        )
        grouped[key].append(row)
    output = []
    metrics = (
        "forward_kl",
        "reverse_kl",
        "heldout_nll",
        "tail_probability",
        "tail_calibration_absolute_error",
        "central_calibration_absolute_error",
        "joint_mass_calibration_absolute_error",
        "theta_q50_rad",
        "theta_q90_rad",
        "theta_q99_rad",
        "fitted_local_curvature",
        "location_cosine",
    )
    for key, group in sorted(grouped.items()):
        record: dict[str, Any] = {
            "kappa": key[0],
            "epsilon": key[1],
            "objective": key[2],
            "family": key[3],
            "seeds": len(group),
        }
        for metric in metrics:
            values = np.asarray([row[metric] for row in group], dtype=float)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        output.append(record)
    return output


def _plot(rows: list[dict[str, Any]], root: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "savefig.dpi": 350,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(
        2, 3, figsize=(9.2, 5.7), sharex=True, constrained_layout=True
    )
    for column, kappa in enumerate((20, 100, 500)):
        for row_index, objective in enumerate(
            ("forward_kl", "reverse_kl")
        ):
            axis = axes[row_index, column]
            metric = (
                "forward_kl_mean"
                if objective == "forward_kl"
                else "reverse_kl_mean"
            )
            for family in ("spcauchy", "vmf", "powerspherical"):
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["kappa"] == kappa
                        and row["objective"] == objective
                        and row["family"] == family
                    ),
                    key=lambda row: row["epsilon"],
                )
                axis.errorbar(
                    [row["epsilon"] for row in selected],
                    [row[metric] for row in selected],
                    yerr=[
                        row[metric.replace("_mean", "_sd")]
                        for row in selected
                    ],
                    marker="o",
                    capsize=2,
                    label=LABELS[family],
                    color=COLORS[family],
                )
            axis.set_title(f"$\\kappa={kappa}$")
            axis.set_xlabel("Uniform contamination $\\epsilon$")
            if column == 0:
                axis.set_ylabel(
                    "Forward KL" if row_index == 0 else "Reverse KL"
                )
            axis.grid(alpha=0.2)
    axes[0, 2].legend(frameon=False, fontsize=8)
    for suffix in ("png", "pdf"):
        figure.savefig(
            root / "figures" / f"contamination_transition.{suffix}",
            dpi=350,
            bbox_inches="tight",
        )
    plt.close(figure)

    tail_figure, tail_axes = plt.subplots(
        1, 3, figsize=(9.2, 2.9), sharex=True, constrained_layout=True
    )
    for axis, kappa in zip(tail_axes, (20, 100, 500)):
        for family in ("spcauchy", "vmf", "powerspherical"):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["kappa"] == kappa
                    and row["objective"] == "forward_kl"
                    and row["family"] == family
                ),
                key=lambda row: row["epsilon"],
            )
            axis.errorbar(
                [row["epsilon"] for row in selected],
                [
                    row["tail_calibration_absolute_error_mean"]
                    for row in selected
                ],
                yerr=[
                    row["tail_calibration_absolute_error_sd"]
                    for row in selected
                ],
                marker="o",
                capsize=2,
                label=LABELS[family],
                color=COLORS[family],
            )
        axis.set_title(f"$\\kappa={kappa}$")
        axis.set_xlabel("Uniform contamination $\\epsilon$")
        axis.grid(alpha=0.2)
    tail_axes[0].set_ylabel("Remote-mass absolute error")
    tail_axes[-1].legend(frameon=False, fontsize=8)
    for suffix in ("png", "pdf"):
        tail_figure.savefig(
            root / "figures" / f"tail_calibration_transition.{suffix}",
            dpi=350,
            bbox_inches="tight",
        )
    plt.close(tail_figure)

    mass_figure, mass_axes = plt.subplots(
        1, 3, figsize=(9.2, 2.9), sharex=True, constrained_layout=True
    )
    for axis, kappa in zip(mass_axes, (20, 100, 500)):
        for family in ("spcauchy", "vmf", "powerspherical"):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["kappa"] == kappa
                    and row["objective"] == "forward_kl"
                    and row["family"] == family
                ),
                key=lambda row: row["epsilon"],
            )
            axis.errorbar(
                [row["epsilon"] for row in selected],
                [
                    row["joint_mass_calibration_absolute_error_mean"]
                    for row in selected
                ],
                yerr=[
                    row["joint_mass_calibration_absolute_error_sd"]
                    for row in selected
                ],
                marker="o",
                capsize=2,
                label=LABELS[family],
                color=COLORS[family],
            )
        axis.set_title(f"$\\kappa={kappa}$")
        axis.set_xlabel("Uniform contamination $\\epsilon$")
        axis.grid(alpha=0.2)
    mass_axes[0].set_ylabel("Central + remote mass error")
    mass_axes[-1].legend(frameon=False, fontsize=8)
    for suffix in ("png", "pdf"):
        mass_figure.savefig(
            root / "figures" / f"mass_calibration_transition.{suffix}",
            dpi=350,
            bbox_inches="tight",
        )
    plt.close(mass_figure)


def _decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def best(
        kappa: int, epsilon: float, objective: str, metric: str
    ) -> str:
        candidates = [
            row
            for row in rows
            if row["kappa"] == kappa
            and row["epsilon"] == epsilon
            and row["objective"] == objective
        ]
        return min(candidates, key=lambda row: row[metric])["family"]

    zero_cells = [
        best(kappa, 0.0, objective, f"{objective}_mean")
        for kappa in (20, 100, 500)
        for objective in ("forward_kl", "reverse_kl")
    ]
    contaminated_cells = [
        best(kappa, epsilon, "forward_kl", "forward_kl_mean")
        for kappa in (20, 100, 500)
        for epsilon in (0.10, 0.20)
    ]
    calibration_cells = [
        best(
            kappa,
            epsilon,
            "forward_kl",
            "joint_mass_calibration_absolute_error_mean",
        )
        for kappa in (20, 100, 500)
        for epsilon in (0.10, 0.20)
    ]
    sanity = all(family == "vmf" for family in zero_cells)
    sc_forward_wins = sum(
        family == "spcauchy" for family in contaminated_cells
    )
    sc_calibration_wins = sum(
        family == "spcauchy" for family in calibration_cells
    )
    clean = (
        sanity
        and sc_forward_wins >= 4
        and sc_calibration_wins >= 4
    )
    return {
        "epsilon_zero_vmf_sanity_passed": sanity,
        "epsilon_zero_best_families": zero_cells,
        "high_contamination_forward_kl_spcauchy_wins": sc_forward_wins,
        "high_contamination_joint_mass_calibration_spcauchy_wins": (
            sc_calibration_wins
        ),
        "high_contamination_cell_count": len(contaminated_cells),
        "clean_interpretable_transition": clean,
        "recommend_include_in_paper": clean,
        "rule": (
            "Require vMF to win all epsilon-zero forward and reverse cells, "
            "then spherical Cauchy to win at least four of six epsilon 0.10 "
            "or 0.20 forward-KL cells and four of six corresponding joint "
            "central-plus-remote mass calibration cells. Hemisphere mass "
            "is also reported separately and is not used to break ties."
        ),
    }


def _kl_latex(
    rows: list[dict[str, Any]], objective: str
) -> str:
    if objective not in {"forward_kl", "reverse_kl"}:
        raise ValueError(objective)
    labels = {
        "spcauchy": "Spherical Cauchy",
        "vmf": "vMF",
        "powerspherical": "Power Spherical",
    }
    lookup = {
        (
            int(row["kappa"]),
            float(row["epsilon"]),
            row["family"],
        ): row
        for row in rows
        if row["objective"] == objective
    }
    epsilons = (0.0, 0.01, 0.05, 0.10, 0.20)
    best = {
        (kappa, epsilon): min(
            ("spcauchy", "vmf", "powerspherical"),
            key=lambda family: lookup[
                (kappa, epsilon, family)
            ][f"{objective}_mean"],
        )
        for kappa in (20, 100, 500)
        for epsilon in epsilons
    }

    def entry(kappa: int, epsilon: float, family: str) -> str:
        row = lookup[(kappa, epsilon, family)]
        value = (
            f"{row[f'{objective}_mean']:.4f} "
            f"$\\pm$ {row[f'{objective}_sd']:.4f}"
        )
        return (
            f"\\textbf{{{value}}}"
            if best[(kappa, epsilon)] == family
            else value
        )

    lines = [
        r"\begin{tabular}{rlrrrrr}",
        r"\toprule",
        r"$\kappa$ & Model & $\epsilon=0$ & $0.01$ & $0.05$ & $0.10$ & $0.20$ \\",
        r"\midrule",
    ]
    for kappa in (20, 100, 500):
        for family_index, family in enumerate(
            ("spcauchy", "vmf", "powerspherical")
        ):
            kappa_entry = str(kappa) if family_index == 0 else ""
            lines.append(
                "{} & {} & {} \\\\".format(
                    kappa_entry,
                    labels[family],
                    " & ".join(
                        entry(kappa, epsilon, family)
                        for epsilon in epsilons
                    ),
                )
            )
        if kappa != 500:
            lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_paper_fragments(
    rows: list[dict[str, Any]],
    decision: dict[str, Any],
    root: Path,
) -> None:
    paper = ensure_dir(root / "paper")
    lookup = {
        (
            int(row["kappa"]),
            float(row["epsilon"]),
            row["objective"],
            row["family"],
        ): row
        for row in rows
    }
    selected = {
        family: lookup[(100, 0.10, "forward_kl", family)]
        for family in ("spcauchy", "vmf", "powerspherical")
    }
    maximum_sc_tail = max(
        row["tail_probability_mean"]
        for row in rows
        if row["family"] == "spcauchy"
        and row["objective"] == "forward_kl"
        and row["epsilon"] > 0.0
    )
    recommendation = (
        "The frozen rule supports including this control in the paper."
        if decision["recommend_include_in_paper"]
        else "The frozen rule does not support paper inclusion."
    )
    section = f"""# Proposed controlled-target paper section

We fit spherical Cauchy, vMF, and Power Spherical approximations to a
vMF-plus-uniform target on the 32-dimensional sphere. The target concentration
is 20, 100, or 500, and the uniform contamination weight ranges from zero to
0.20. Every fit learns location and concentration from the same initialization
with the same optimization budget. We repeat forward-KL and reverse-KL fitting
over three seeds.

At zero contamination, vMF is best in every concentration and objective cell.
At kappa 100 and epsilon 0.10, forward KL is
{selected['spcauchy']['forward_kl_mean']:.3f} for spherical Cauchy,
{selected['vmf']['forward_kl_mean']:.3f} for vMF, and
{selected['powerspherical']['forward_kl_mean']:.3f} for Power Spherical.
Spherical Cauchy wins
{decision['high_contamination_forward_kl_spcauchy_wins']} of 6
high-contamination forward-KL cells and
{decision['high_contamination_joint_mass_calibration_spcauchy_wins']} of 6
joint mass-calibration cells.

The hemisphere diagnostic is an important limit. The largest fitted spherical
Cauchy probability estimate beyond pi over two among contaminated forward-KL
cells is {maximum_sc_tail:.6g}. The KL advantage therefore shows higher remote
log density and a better central-to-remote compromise. It does not show that a
single spherical Cauchy reproduces the target's uniform hemisphere mass.

{recommendation}
"""
    (paper / "CONTAMINATED_DIRECTIONAL_SECTION_DRAFT.md").write_text(
        section, encoding="utf-8"
    )
    (paper / "contaminated_directional_results.tex").write_text(
        _kl_latex(rows, "forward_kl"), encoding="utf-8"
    )
    (paper / "contaminated_directional_reverse_results.tex").write_text(
        _kl_latex(rows, "reverse_kl"), encoding="utf-8"
    )
    (paper / "contaminated_directional_supplement.tex").write_text(
        "The controlled target is "
        "$p(z)=(1-\\epsilon)\\operatorname{vMF}(\\mu,\\kappa)"
        "+\\epsilon\\operatorname{Unif}(\\mathbb S^{32})$. "
        "We use three seeds, 600 optimization steps, batches of 8192, and "
        "one million held-out Monte Carlo draws per fit. Separate forward "
        "and reverse KL objectives are reported. The zero-contamination vMF "
        "sanity check and the central, hemisphere, curvature, and angular "
        "quantile diagnostics are retained in the repository tables.\n",
        encoding="utf-8",
    )


def _report(
    rows: list[dict[str, Any]], decision: dict[str, Any]
) -> str:
    sanity_pass = decision["epsilon_zero_vmf_sanity_passed"]
    maximum_sc_tail = max(
        row["tail_probability_mean"]
        for row in rows
        if row["family"] == "spcauchy"
        and row["objective"] == "forward_kl"
        and row["epsilon"] > 0.0
    )
    lines = [
        "# Controlled contaminated directional target",
        "",
        "This validation-independent control fits three spherical families to a",
        "vMF plus uniform mixture on S^32. It uses matched initial local",
        "curvature and identical optimization budgets.",
        "",
        (
            "The target-matched initialization and the interrupted "
            "low-curvature reverse-KL basin are documented in "
            "`DESIGN_PILOT.md`."
        ),
        "",
        f"The epsilon=0 vMF sanity check {'passed' if sanity_pass else 'failed'}.",
        "",
        "Paper inclusion recommendation: "
        + (
            "**include as a controlled companion**."
            if decision["recommend_include_in_paper"]
            else "**do not include**."
        ),
        "",
        (
            "Spherical Cauchy wins "
            f"{decision['high_contamination_forward_kl_spcauchy_wins']} "
            "of 6 high-contamination forward-KL cells and "
            f"{decision['high_contamination_joint_mass_calibration_spcauchy_wins']} "
            "of 6 joint mass-calibration cells."
        ),
        "",
        (
            "The largest fitted spherical Cauchy hemisphere probability "
            "among contaminated forward-KL cells is "
            f"{maximum_sc_tail:.6g}. This diagnostic is reported separately "
            "because a KL advantage does not imply that one unimodal family "
            "recovers the target's uniform probability floor."
        ),
        "",
        "Best family by objective and grid point:",
        "",
        "| kappa | epsilon | objective | best family | fitted KL |",
        "|---:|---:|:---|:---|---:|",
    ]
    for kappa in (20, 100, 500):
        for epsilon in (0.0, 0.01, 0.05, 0.10, 0.20):
            for objective in ("forward_kl", "reverse_kl"):
                metric = (
                    "forward_kl_mean"
                    if objective == "forward_kl"
                    else "reverse_kl_mean"
                )
                selected = [
                    row
                    for row in rows
                    if row["kappa"] == kappa
                    and row["epsilon"] == epsilon
                    and row["objective"] == objective
                ]
                best = min(selected, key=lambda row: row[metric])
                lines.append(
                    f"| {kappa} | {epsilon:.2f} | {objective} | "
                    f"{LABELS[best['family']]} | {best[metric]:.5f} |"
                )
    lines.extend(
        [
            "",
            "The experiment should enter the paper only if the family ordering",
            "changes cleanly as contamination increases and the tail calibration",
            "metrics agree with the KL transition.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate(root: Path) -> None:
    rows = _load_rows(root)
    if not rows:
        raise RuntimeError(f"No completed fits under {root}")
    seed_count = _validate_complete_grid(rows)
    aggregate_rows = _aggregate_rows(rows)
    decision = _decision(aggregate_rows)
    ensure_dir(root / "tables")
    ensure_dir(root / "figures")
    write_json(root / "tables" / "seed_level.json", rows)
    write_csv(root / "tables" / "seed_level.csv", rows)
    write_json(root / "tables" / "aggregate.json", aggregate_rows)
    write_csv(root / "tables" / "aggregate.csv", aggregate_rows)
    write_json(root / "decision.json", decision)
    (root / "tables" / "forward_kl.tex").write_text(
        _kl_latex(aggregate_rows, "forward_kl"), encoding="utf-8"
    )
    (root / "tables" / "reverse_kl.tex").write_text(
        _kl_latex(aggregate_rows, "reverse_kl"), encoding="utf-8"
    )
    _plot(aggregate_rows, root)
    _write_paper_fragments(aggregate_rows, decision, root)
    (root / "REPORT.md").write_text(
        _report(aggregate_rows, decision), encoding="utf-8"
    )
    artifacts = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {
            "artifact_checksums.sha256",
            "manifest.json",
        }
        and path.suffix != ".pt"
    )
    checksum_lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in artifacts
    ]
    (root / "artifact_checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256(
        "\n".join(checksum_lines).encode("utf-8")
    ).hexdigest()
    repository_root = Path(__file__).resolve().parents[2]
    source_paths = {
        "spherical_cauchy_sampler": (
            repository_root / "src" / "spherical_cauchy" / "functional.py"
        ),
        "smallnorb_local_vmf": (
            repository_root
            / "experiments"
            / "smallnorb"
            / "vendor_vmf_smallnorb.py"
        ),
        "pinned_power_spherical": (
            repository_root
            / "benchmark"
            / "vendor_power_spherical.py"
        ),
    }
    write_json(
        root / "manifest.json",
        {
            "fit_count": len(rows),
            "aggregate_row_count": len(aggregate_rows),
            "seed_count": seed_count,
            "complete_predeclared_grid": True,
            "protocol": {
                "ambient_dimension": 33,
                "steps": int(rows[0]["steps"]),
                "batch_size": int(rows[0]["batch_size"]),
                "evaluation_samples": int(
                    rows[0]["evaluation_samples"]
                ),
                "learning_rate": float(rows[0]["learning_rate"]),
                "initialization": (
                    "target-matched local curvature and a fixed 10-degree "
                    "location offset, identical across families within each "
                    "grid cell"
                ),
                "evaluation_randomness": (
                    "target draws are common across families and objectives "
                    "within each target and seed"
                ),
            },
            "sources": {
                name: {
                    "path": path.relative_to(repository_root).as_posix(),
                    "sha256": sha256_file(path),
                }
                for name, path in source_paths.items()
            },
            "power_spherical_upstream_commit": (
                "3d4619a9d6c01bc9b427533d386271a233e304cd"
            ),
            "decision": decision,
            "artifact_checksum_manifest_sha256": digest,
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/contaminated_directional/final"),
    )
    aggregate(parser.parse_args().root)
