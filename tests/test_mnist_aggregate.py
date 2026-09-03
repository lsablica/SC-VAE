import pytest

from experiments.mnist.aggregate import (
    _aggregate_rows,
    _benjamini_hochberg,
)


def _seed_row(model_family: str, seed: int, reconstruction: float) -> dict:
    return {
        "model_family": model_family,
        "reported_dim": 2,
        "ambient_latent_dim": 3,
        "seed": seed,
        "selected_epoch": 40,
        "best_eval_recon_loss": reconstruction,
        "best_eval_total_loss": reconstruction + 1.0,
        "best_eval_kl": 1.0,
        "concentration_mean": None,
        "concentration_q95": None,
        "concentration_max": None,
        "wall_clock_training_s": 1.0,
    }


def test_benjamini_hochberg_preserves_input_order():
    adjusted = _benjamini_hochberg([0.02325, 0.001758, 0.019554])
    assert adjusted == pytest.approx([0.02325, 0.005274, 0.02325])


def test_aggregate_marks_best_mean_and_significant_paired_win():
    rows = []
    winner = [10.0, 10.2, 9.9, 10.1, 10.0]
    runner_up = [11.0, 11.4, 10.8, 11.3, 10.9]
    for seed, value in enumerate(winner):
        rows.append(_seed_row("winner", seed, value))
    for seed, value in enumerate(runner_up):
        rows.append(_seed_row("runner", seed, value))

    summary = _aggregate_rows(rows, [])
    by_family = {row["model_family"]: row for row in summary}

    assert by_family["winner"]["bold_best_mean"] is True
    assert by_family["winner"]["dagger_paired_bh"] is True
    assert by_family["winner"]["paired_runner_up"] == "runner"
    assert by_family["winner"]["paired_num_seeds"] == 5
    assert by_family["winner"]["paired_improvement_ci_low"] > 0.0
    assert by_family["runner"]["bold_best_mean"] is False
    assert by_family["runner"]["dagger_paired_bh"] is False
