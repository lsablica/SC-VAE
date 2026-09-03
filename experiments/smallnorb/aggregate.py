"""Five-seed aggregation, paired statistics, and paper-ready fragments."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import numpy as np
from scipy.stats import t, ttest_rel, wilcoxon

from .config import (
    FINAL_ROOT,
    MAIN_FAMILIES,
    RUNS_ROOT,
    SEEDS,
)
from .utils import (
    ensure_dir,
    read_json,
    repo_relative,
    write_csv,
    write_json,
)


METRIC_DIRECTIONS = {
    "test_gap_reconstruction_nll": "lower",
    "test_gap_pixel_mse": "lower",
    "test_gap_psnr_db": "higher",
    "test_gap_ssim": "higher",
    "test_observed_reconstruction_nll": "lower",
    "test_observed_pixel_mse": "lower",
    "test_observed_psnr_db": "higher",
    "test_observed_ssim": "higher",
    "pose_test_observed_mean_error_degrees": "lower",
    "pose_test_gap_mean_error_degrees": "lower",
    "pose_test_gap_median_error_degrees": "lower",
    "pose_test_gap_q90_error_degrees": "lower",
    "geometry_test_spearman": "higher",
    "geometry_test_cross_gap_spearman": "higher",
    "interpolation_interior_pixel_mse": "lower",
    "interpolation_interior_psnr_db": "higher",
    "interpolation_interior_ssim": "higher",
}
PRIMARY_METRIC = "test_gap_reconstruction_nll"


def _mean(summary: dict[str, Any], key: str) -> float:
    return float(summary[key]["mean"])


def _seed_row(run_dir: Path) -> dict[str, Any]:
    evaluation = read_json(run_dir / "evaluation_summary.json")
    probe = read_json(run_dir / "probe_summary.json")
    interpolation = read_json(
        run_dir / "interpolation_summary_test.json"
    )
    selection = read_json(run_dir / "selection_summary.json")
    seed_manifest = read_json(run_dir / "seed_manifest.json")
    history = read_json(run_dir / "history.json")
    config = evaluation["config"]
    test = evaluation["test"]
    gap = test["test_gap"]
    observed = test["test_observed"]
    pose = probe["pose_probe"]["partitions"]
    geometry = probe["geometry_alignment"]["test"]
    interior = interpolation["summaries"]["interior_gap"]
    scale = test["test"]["posterior_scale"]
    sample_diagnostics = evaluation["posterior_sample_diagnostics"]
    gradient_means = [
        float(epoch["train"]["gradient_norm"]["mean"])
        for epoch in history
        if epoch["train"]["gradient_norm"].get("count", 0)
    ]
    gradient_q90 = [
        float(epoch["train"]["gradient_norm"]["q90"])
        for epoch in history
        if epoch["train"]["gradient_norm"].get("count", 0)
    ]
    gradient_maxima = [
        float(epoch["train"]["gradient_norm"]["max"])
        for epoch in history
        if epoch["train"]["gradient_norm"].get("count", 0)
    ]
    return {
        "family": config["family"],
        "seed": int(config["seed"]),
        "run_dir": repo_relative(run_dir),
        "selected_epoch": int(selection["selected_epoch"]),
        "wall_clock_training_s": float(
            selection["wall_clock_training_s"]
        ),
        "peak_cuda_memory_bytes": int(
            selection["peak_cuda_memory_bytes"]
        ),
        "trainable_parameters": int(
            seed_manifest["parameter_counts"]["total"]
        ),
        "initial_kl_mean": float(
            seed_manifest["initial_diagnostics"]["kl"]["mean"]
        ),
        "initial_kl_median": float(
            seed_manifest["initial_diagnostics"]["kl"]["median"]
        ),
        "initial_kl_q95": float(
            seed_manifest["initial_diagnostics"]["kl"]["q95"]
        ),
        "spcauchy_direct_retained_terms": (
            int(
                seed_manifest["initial_diagnostics"]["direct_kl"][
                    "retained_terms"
                ]
            )
            if config["family"] == "spcauchy"
            else None
        ),
        "test_gap_reconstruction_nll": _mean(
            gap, "reconstruction_nll"
        ),
        "test_gap_pixel_mse": _mean(gap, "pixel_mse"),
        "test_gap_psnr_db": _mean(gap, "psnr_db"),
        "test_gap_ssim": _mean(gap, "ssim"),
        "test_observed_reconstruction_nll": _mean(
            observed, "reconstruction_nll"
        ),
        "test_observed_pixel_mse": _mean(observed, "pixel_mse"),
        "test_observed_psnr_db": _mean(observed, "psnr_db"),
        "test_observed_ssim": _mean(observed, "ssim"),
        "test_kl_mean": _mean(test["test"], "kl"),
        "test_kl_median": float(test["test"]["kl"]["median"]),
        "test_kl_q95": float(test["test"]["kl"]["q95"]),
        "posterior_scale_mean": float(scale["mean"]),
        "posterior_scale_median": float(scale["median"]),
        "posterior_scale_q95": float(scale["q95"]),
        "expected_cosine_to_mode": (
            _mean(test["test"], "expected_cosine_to_mode")
            if "expected_cosine_to_mode" in test["test"]
            else None
        ),
        "opposite_hemisphere_fraction": (
            sample_diagnostics.get(
                "fraction_in_opposite_hemisphere"
            )
        ),
        "gradient_norm_epoch_mean": float(np.mean(gradient_means)),
        "gradient_norm_epoch_q90_mean": float(np.mean(gradient_q90)),
        "gradient_norm_max": float(max(gradient_maxima)),
        "amp_skipped_steps_total": int(
            sum(
                int(epoch["train"]["amp_skipped_steps"])
                for epoch in history
            )
        ),
        "nonfinite_count": 0,
        "pose_test_observed_mean_error_degrees": float(
            pose["test_observed"]["mean_absolute_error_degrees"]
        ),
        "pose_test_gap_mean_error_degrees": float(
            pose["test_gap"]["mean_absolute_error_degrees"]
        ),
        "pose_test_gap_median_error_degrees": float(
            pose["test_gap"]["median_absolute_error_degrees"]
        ),
        "pose_test_gap_q90_error_degrees": float(
            pose["test_gap"]["q90_absolute_error_degrees"]
        ),
        "geometry_test_spearman": float(
            geometry["all_pairs"]["spearman"]
        ),
        "geometry_test_cross_gap_spearman": float(
            geometry["pairs_crossing_gap"]["spearman"]
        ),
        "interpolation_interior_pixel_mse": _mean(
            interior, "pixel_mse"
        ),
        "interpolation_interior_psnr_db": _mean(
            interior, "psnr_db"
        ),
        "interpolation_interior_ssim": _mean(interior, "ssim"),
    }


def discover_seed_rows() -> list[dict[str, Any]]:
    rows = []
    for summary in sorted(
        RUNS_ROOT.glob("final/*/*/seed_*/evaluation_summary.json")
    ):
        run_dir = summary.parent
        required = (
            run_dir / "probe_summary.json",
            run_dir / "interpolation_summary_test.json",
        )
        if all(path.exists() for path in required):
            rows.append(_seed_row(run_dir))
    if not rows:
        raise FileNotFoundError("No completed full-study runs found")
    completed = {
        family: {
            int(row["seed"])
            for row in rows
            if row["family"] == family
        }
        for family in MAIN_FAMILIES
    }
    expected = set(SEEDS)
    incomplete = {
        family: sorted(expected - seeds)
        for family, seeds in completed.items()
        if seeds != expected
    }
    if incomplete:
        raise RuntimeError(
            "Locked aggregation requires all five seeds for every main "
            f"family; missing seeds: {incomplete}"
        )
    return sorted(rows, key=lambda row: (row["family"], row["seed"]))


def aggregate_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    summaries = []
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and key not in {"seed"}
        }
    )
    for family, family_rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "family": family,
            "num_seeds": len(family_rows),
            "seeds": " ".join(
                str(row["seed"]) for row in family_rows
            ),
        }
        for key in numeric_keys:
            values = [
                float(row[key])
                for row in family_rows
                if row.get(key) is not None
            ]
            if not values:
                continue
            summary[f"{key}_mean"] = mean(values)
            summary[f"{key}_std"] = (
                stdev(values) if len(values) > 1 else 0.0
            )
            summary[f"{key}_median"] = median(values)
        summaries.append(summary)
    return summaries


def _bh_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 1.0
    count = len(p_values)
    for reverse_rank in range(count - 1, -1, -1):
        original = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, p_values[original] * count / rank)
        adjusted[original] = min(running, 1.0)
    return adjusted


def paired_statistics(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped = {
        family: {int(row["seed"]): row for row in rows if row["family"] == family}
        for family in {row["family"] for row in rows}
    }
    spherical = grouped["spcauchy"]
    results = []
    for competitor in sorted(set(grouped) - {"spcauchy"}):
        common = sorted(set(spherical) & set(grouped[competitor]))
        for metric, direction in METRIC_DIRECTIONS.items():
            differences = np.asarray(
                [
                    float(spherical[seed][metric])
                    - float(grouped[competitor][seed][metric])
                    for seed in common
                ],
                dtype=np.float64,
            )
            count = len(differences)
            difference_mean = float(differences.mean())
            difference_std = (
                float(differences.std(ddof=1)) if count > 1 else 0.0
            )
            if count > 1:
                half_width = float(
                    t.ppf(0.975, count - 1)
                    * difference_std
                    / math.sqrt(count)
                )
                t_result = ttest_rel(
                    [spherical[seed][metric] for seed in common],
                    [grouped[competitor][seed][metric] for seed in common],
                )
                t_p = float(t_result.pvalue)
                try:
                    wilcoxon_p = float(
                        wilcoxon(differences).pvalue
                    )
                except ValueError:
                    wilcoxon_p = 1.0
            else:
                half_width = math.nan
                t_p = math.nan
                wilcoxon_p = math.nan
            results.append(
                {
                    "competitor": competitor,
                    "metric": metric,
                    "direction": direction,
                    "num_paired_seeds": count,
                    "spcauchy_minus_competitor_mean": difference_mean,
                    "paired_difference_std": difference_std,
                    "paired_t_ci_low": difference_mean - half_width,
                    "paired_t_ci_high": difference_mean + half_width,
                    "paired_t_p_value": t_p,
                    "paired_wilcoxon_p_value": wilcoxon_p,
                    "is_primary_metric": metric == PRIMARY_METRIC,
                }
            )
    secondary_indices = [
        index
        for index, row in enumerate(results)
        if not row["is_primary_metric"]
        and math.isfinite(row["paired_t_p_value"])
    ]
    adjusted = _bh_adjust(
        [results[index]["paired_t_p_value"] for index in secondary_indices]
    )
    for row in results:
        row["secondary_bh_q_value"] = None
    for index, q_value in zip(secondary_indices, adjusted):
        results[index]["secondary_bh_q_value"] = q_value
    return results


def _family_means(
    summaries: list[dict[str, Any]], metric: str
) -> dict[str, float]:
    return {
        row["family"]: float(row[f"{metric}_mean"])
        for row in summaries
    }


def replacement_decision(
    summaries: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> dict[str, Any]:
    gap = _family_means(summaries, PRIMARY_METRIC)
    observed = _family_means(
        summaries, "test_observed_reconstruction_nll"
    )
    geometry_metrics = (
        ("pose_test_gap_mean_error_degrees", "lower"),
        ("geometry_test_spearman", "higher"),
        ("geometry_test_cross_gap_spearman", "higher"),
        ("interpolation_interior_pixel_mse", "lower"),
    )
    best_competitor = min(
        (family for family in gap if family != "spcauchy"),
        key=gap.__getitem__,
    )
    primary_comparison = next(
        row
        for row in paired
        if row["competitor"] == best_competitor
        and row["metric"] == PRIMARY_METRIC
    )
    best_or_tied_gap = (
        gap["spcauchy"] <= min(gap.values())
        or (
            primary_comparison["paired_t_ci_low"] <= 0.0
            <= primary_comparison["paired_t_ci_high"]
        )
    )
    geometry_wins = []
    for metric, direction in geometry_metrics:
        values = _family_means(summaries, metric)
        spherical = values["spcauchy"]
        best = min(values.values()) if direction == "lower" else max(values.values())
        if spherical == best:
            geometry_wins.append(metric)
    observed_not_worse = observed["spcauchy"] <= 1.02 * min(
        observed.values()
    )
    interpolation = _family_means(
        summaries, "interpolation_interior_pixel_mse"
    )
    figure_agrees = interpolation["spcauchy"] <= 1.02 * min(
        interpolation.values()
    )
    replace = (
        best_or_tied_gap
        and bool(geometry_wins)
        and observed_not_worse
        and figure_agrees
    )
    return {
        "recommend_main_paper": replace,
        "best_non_spcauchy_primary_competitor": best_competitor,
        "spcauchy_best_or_statistically_tied_on_gap": best_or_tied_gap,
        "spcauchy_geometry_wins": geometry_wins,
        "observed_reconstruction_not_materially_worse": observed_not_worse,
        "interpolation_quantitative_result_agrees": figure_agrees,
        "rule": (
            "replace only if spherical Cauchy is best or tied on gap "
            "reconstruction, wins at least one geometry metric, has no "
            "material observed-view cost, and interpolation agrees"
        ),
    }


def _display_test(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    metric: str,
    direction: str,
) -> tuple[str, str, float]:
    """Return best, runner-up, and their seed-paired t-test p value."""

    means = _family_means(summaries, metric)
    ranking = sorted(
        means,
        key=means.__getitem__,
        reverse=direction == "higher",
    )
    best, runner_up = ranking[:2]
    by_family = {
        family: {
            int(row["seed"]): float(row[metric])
            for row in rows
            if row["family"] == family
        }
        for family in means
    }
    common = sorted(set(by_family[best]) & set(by_family[runner_up]))
    if len(common) < 2:
        return best, runner_up, math.nan
    result = ttest_rel(
        [by_family[best][seed] for seed in common],
        [by_family[runner_up][seed] for seed in common],
    )
    return best, runner_up, float(result.pvalue)


def _latex_table(
    summaries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> str:
    labels = {
        "spcauchy": "Spherical Cauchy",
        "vmf_robust": "Robust vMF",
        "powerspherical": "Power Spherical",
        "gaussian_isotropic": "Isotropic Gaussian",
        "gaussian_diagonal": "Diagonal Gaussian",
    }
    columns = (
        ("test_gap_reconstruction_nll", "lower", 2),
        ("test_gap_pixel_mse", "lower", 5),
        ("test_observed_reconstruction_nll", "lower", 2),
        ("pose_test_gap_mean_error_degrees", "lower", 2),
        ("geometry_test_spearman", "higher", 3),
        ("interpolation_interior_pixel_mse", "lower", 5),
    )
    display_tests = {
        metric: _display_test(rows, summaries, metric, direction)
        for metric, direction, _ in columns
    }
    secondary_metrics = [
        metric
        for metric, _, _ in columns
        if metric != PRIMARY_METRIC
        and math.isfinite(display_tests[metric][2])
    ]
    secondary_q = dict(
        zip(
            secondary_metrics,
            _bh_adjust(
                [display_tests[metric][2] for metric in secondary_metrics]
            ),
        )
    )
    significant = {
        metric: (
            test[2] < 0.05
            if metric == PRIMARY_METRIC
            else secondary_q.get(metric, 1.0) < 0.05
        )
        for metric, test in display_tests.items()
    }

    def entry(
        row: dict[str, Any], metric: str, digits: int
    ) -> str:
        value = (
            f"{row[f'{metric}_mean']:.{digits}f} "
            f"$\\pm$ {row[f'{metric}_std']:.{digits}f}"
        )
        best = display_tests[metric][0]
        if row["family"] == best:
            marker = "$^\\dagger$" if significant[metric] else ""
            return f"\\textbf{{{value}}}{marker}"
        return value

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & Gap NLL $\downarrow$ & Gap MSE $\downarrow$ & Observed NLL $\downarrow$ & Pose error $\downarrow$ & Spearman $\uparrow$ & Interp. MSE $\downarrow$ \\",
        r"\midrule",
    ]
    for row in summaries:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                labels[row["family"]],
                *[
                    entry(row, metric, digits)
                    for metric, _, digits in columns
                ],
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\footnotesize Bold marks the best seed mean. "
            r"$^\dagger$ marks a significant paired difference from the "
            r"next-best mean by a two-sided paired \(t\)-test at \(0.05\). "
            r"Secondary endpoints use Benjamini--Hochberg correction across "
            r"the displayed secondary metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def write_paper_fragments(
    summaries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    decision: dict[str, Any],
) -> None:
    paper = ensure_dir(FINAL_ROOT / "paper")
    by_family = {row["family"]: row for row in summaries}
    spherical = by_family["spcauchy"]
    competitor_name = decision[
        "best_non_spcauchy_primary_competitor"
    ]
    competitor = by_family[competitor_name]
    primary = next(
        row
        for row in paired
        if row["competitor"] == competitor_name
        and row["metric"] == PRIMARY_METRIC
    )
    relative_gain = 100.0 * (
        competitor["test_gap_reconstruction_nll_mean"]
        - spherical["test_gap_reconstruction_nll_mean"]
    ) / competitor["test_gap_reconstruction_nll_mean"]
    recommendation = (
        "The locked decision rule supports inclusion in the main paper."
        if decision["recommend_main_paper"]
        else (
            "The locked decision rule supports reporting smallNORB only as a "
            "supplementary mixed result."
        )
    )
    interpretation = (
        "The primary result is consistent with useful decoder behavior "
        "across the unobserved angular sector. It does not establish a "
        "globally ordered viewpoint geometry. The pooled linear pose probe "
        "does not extrapolate accurately into the gap, and the global "
        "distance correlations are near zero."
        if decision["recommend_main_paper"]
        else (
            "The result is mixed. It tests the matched-curvature hypothesis "
            "under a controlled gap, but it does not establish a coherent "
            "advantage in both reconstruction and latent geometry."
        )
    )
    section = f"""# Proposed smallNORB main-paper section

## Motivation

smallNORB exposes a known periodic viewpoint factor. It therefore separates local posterior precision from behavior across remote angular directions.

## Protocol

We use the left camera at 64 by 64 pixels and the official instance split. Instance 9 in each category is reserved for validation. Training excludes azimuths 320, 340, 0, and 20 degrees. The intrinsic latent dimension is 32. Spherical models use ambient dimension 33. The isotropic Gaussian has 32 means and one scale. Every family uses the same convolutional architecture, optimizer schedule, checkpoint rule, and five seed indices.

## Results

Spherical Cauchy obtains test-gap reconstruction NLL {spherical['test_gap_reconstruction_nll_mean']:.2f} plus or minus {spherical['test_gap_reconstruction_nll_std']:.2f}. The best competing family is {competitor_name}, with {competitor['test_gap_reconstruction_nll_mean']:.2f} plus or minus {competitor['test_gap_reconstruction_nll_std']:.2f}. The relative spherical Cauchy gain is {relative_gain:.1f} percent. The paired spherical Cauchy minus competitor difference is {primary['spcauchy_minus_competitor_mean']:.2f}, with a 95 percent interval from {primary['paired_t_ci_low']:.2f} to {primary['paired_t_ci_high']:.2f} and paired t test p value {primary['paired_t_p_value']:.4f}.

Spherical Cauchy test-gap pixel MSE is {spherical['test_gap_pixel_mse_mean']:.5f}. Mean circular pose error in the gap is {spherical['pose_test_gap_mean_error_degrees_mean']:.2f} degrees. Latent-distance Spearman correlation is {spherical['geometry_test_spearman_mean']:.3f}. Interior ground-truth interpolation MSE is {spherical['interpolation_interior_pixel_mse_mean']:.5f}. Observed-view reconstruction NLL is {spherical['test_observed_reconstruction_nll_mean']:.2f}.

## Interpretation

{interpretation} The experiment does not imply that heavy tails universally improve variational autoencoders. Architecture, optimization, and dataset structure also matter.

## Recommendation

{recommendation}
"""
    (paper / "SMALLNORB_SECTION_DRAFT.md").write_text(
        section, encoding="utf-8"
    )
    (paper / "smallnorb_results.tex").write_text(
        _latex_table(summaries, rows), encoding="utf-8"
    )
    (paper / "smallnorb_figure_captions.tex").write_text(
        "\\paragraph{Angular gap reconstruction.} Ground truth and "
        "deterministic latent interpolations through the four held-out "
        "azimuths. The object is selected by the registered median-error "
        "rule.\n\n"
        "\\paragraph{Latent viewpoint geometry.} Posterior locations, "
        "latent distance alignment, and seed-level gap performance under "
        "the locked protocol.\n",
        encoding="utf-8",
    )
    (paper / "smallnorb_supplement.tex").write_text(
        "\\subsection{smallNORB protocol and diagnostics}\n"
        "We report training curves, posterior KL and concentration, pose "
        "probe predictions, distance matrices, interpolation metrics, and "
        "per-factor sensitivity for all seeds. Hyperparameters were chosen "
        "using validation instances only. Official test instances were "
        "accessed after the shared setup and smoke gate were frozen.\n\n"
        f"Spherical Cauchy test-gap reconstruction NLL is "
        f"{spherical['test_gap_reconstruction_nll_mean']:.2f} "
        f"$\\pm$ {spherical['test_gap_reconstruction_nll_std']:.2f}. "
        f"Its observed-view NLL is "
        f"{spherical['test_observed_reconstruction_nll_mean']:.2f} "
        f"$\\pm$ {spherical['test_observed_reconstruction_nll_std']:.2f}. "
        f"The gap pose error is "
        f"{spherical['pose_test_gap_mean_error_degrees_mean']:.2f} degrees "
        f"and the distance Spearman value is "
        f"{spherical['geometry_test_spearman_mean']:.3f}.\n",
        encoding="utf-8",
    )


def aggregate() -> dict[str, Any]:
    ensure_dir(FINAL_ROOT / "tables")
    ensure_dir(FINAL_ROOT / "figures")
    rows = discover_seed_rows()
    summaries = aggregate_rows(rows)
    paired = paired_statistics(rows)
    decision = replacement_decision(summaries, paired)
    spherical = next(
        row for row in summaries if row["family"] == "spcauchy"
    )
    write_csv(FINAL_ROOT / "tables" / "seed_level.csv", rows)
    write_csv(FINAL_ROOT / "tables" / "aggregate.csv", summaries)
    write_csv(FINAL_ROOT / "tables" / "paired_statistics.csv", paired)
    write_json(FINAL_ROOT / "tables" / "seed_level.json", rows)
    write_json(FINAL_ROOT / "tables" / "aggregate.json", summaries)
    write_json(FINAL_ROOT / "replacement_decision.json", decision)
    (FINAL_ROOT / "tables" / "main_results.tex").write_text(
        _latex_table(summaries, rows), encoding="utf-8"
    )
    write_paper_fragments(summaries, rows, paired, decision)
    limitations = [
        "smallNORB contains rendered toy objects and may not transfer to natural video.",
        "Five seeds provide limited power for distributional significance claims.",
        "The deterministic reconstruction endpoint evaluates posterior locations rather than a full posterior predictive average.",
        "The architecture and optimization schedule can interact with posterior family.",
        "Only one intrinsic latent dimension and one final gap width are confirmatory.",
        "The pooled linear pose probe does not extrapolate into the held-out sector, and global latent-distance Spearman correlations are near zero.",
    ]
    lines = [
        "# smallNORB viewpoint generalization",
        "",
        f"Completed runs: {len(rows)}",
        "",
        f"Recommendation: **{'main paper' if decision['recommend_main_paper'] else 'supplement only'}**",
        "",
        "## Primary result",
        "",
    ]
    for row in summaries:
        lines.append(
            "- {}: test-gap NLL {:.2f} ± {:.2f}, pixel MSE {:.5f} ± {:.5f}".format(
                row["family"],
                row["test_gap_reconstruction_nll_mean"],
                row["test_gap_reconstruction_nll_std"],
                row["test_gap_pixel_mse_mean"],
                row["test_gap_pixel_mse_std"],
            )
        )
    lines.extend(
        [
            "",
            "## Geometry sanity check",
            "",
            (
                "Spherical Cauchy has the lowest mean gap pose error at "
                f"{spherical['pose_test_gap_mean_error_degrees_mean']:.2f} "
                "degrees, but this absolute error is worse than the "
                "90-degree expectation of an independent uniform angle. "
                "Its global latent-distance Spearman correlation is "
                f"{spherical['geometry_test_spearman_mean']:.3f}. The "
                "relative geometry win satisfies the mechanical replacement "
                "rule, but these absolute diagnostics do not support a claim "
                "of globally ordered viewpoint coordinates."
            ),
        ]
    )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in limitations)
    lines.extend(
        [
            "",
            "## Posterior and training diagnostics",
            "",
            "| Family | Selected epoch | Test KL | Posterior scale | "
            "Skipped AMP steps | Training seconds | Peak CUDA MiB |",
            "|:---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            "| {family} | {epoch:.1f} | {kl:.2f} | {scale:.4f} | "
            "{skips:.1f} | {seconds:.1f} | {memory:.1f} |".format(
                family=row["family"],
                epoch=row["selected_epoch_mean"],
                kl=row["test_kl_mean_mean"],
                scale=row["posterior_scale_median_mean"],
                skips=row["amp_skipped_steps_total_mean"],
                seconds=row["wall_clock_training_s_mean"],
                memory=(
                    row["peak_cuda_memory_bytes_mean"] / (1024.0**2)
                ),
            )
        )
    lines.extend(
        [
            "",
            "The validation-only search record is in "
            "`search/SEARCH_REPORT.md`. It includes the rejected deeper CNN "
            "pilot and every numerical setup candidate. The one-seed family "
            "gate is in `reports/stage2_smoke_report.md`.",
            "",
        ]
    )
    (FINAL_ROOT / "REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return {
        "seed_rows": rows,
        "summaries": summaries,
        "paired": paired,
        "decision": decision,
    }


def main() -> None:
    result = aggregate()
    print(
        {
            "runs": len(result["seed_rows"]),
            "recommend_main_paper": result["decision"][
                "recommend_main_paper"
            ],
        }
    )


if __name__ == "__main__":
    main()
