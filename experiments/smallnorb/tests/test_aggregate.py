from experiments.smallnorb.aggregate import _latex_table


METRICS = (
    "test_gap_reconstruction_nll",
    "test_gap_pixel_mse",
    "test_observed_reconstruction_nll",
    "pose_test_gap_mean_error_degrees",
    "geometry_test_spearman",
    "interpolation_interior_pixel_mse",
)


def test_latex_table_bolds_best_and_daggers_significant_winner():
    families = (
        "spcauchy",
        "vmf_robust",
        "powerspherical",
        "gaussian_isotropic",
    )
    rows = []
    for family_index, family in enumerate(families):
        for seed in range(5):
            row = {"family": family, "seed": seed}
            for metric in METRICS:
                separation = family_index * 0.2 * (1.0 + 0.05 * seed)
                if metric == "geometry_test_spearman":
                    row[metric] = 1.0 - separation + seed * 0.001
                else:
                    row[metric] = 1.0 + separation + seed * 0.001
            rows.append(row)
    summaries = []
    for family_index, family in enumerate(families):
        summary = {"family": family}
        for metric in METRICS:
            values = [
                row[metric] for row in rows if row["family"] == family
            ]
            summary[f"{metric}_mean"] = sum(values) / len(values)
            summary[f"{metric}_std"] = 0.001
        summaries.append(summary)
    latex = _latex_table(summaries, rows)
    assert latex.count(r"\textbf{") == len(METRICS)
    bold_lines = [line for line in latex.splitlines() if r"\textbf{" in line]
    assert sum(line.count(r"$^\dagger$") for line in bold_lines) == len(
        METRICS
    )
    assert "Benjamini--Hochberg" in latex
