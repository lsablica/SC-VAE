import math

import pytest
import torch
from scipy.special import ive

from experiments.contaminated_directional.aggregate import (
    _decision,
    _validate_complete_grid,
)
from experiments.contaminated_directional.distributions import (
    AMBIENT_DIMENSION,
    DirectionalFamily,
    TargetMixture,
    log_bessel_i,
)
from experiments.contaminated_directional.fit import initial_location


@pytest.mark.parametrize("kappa", [20.0, 100.0, 500.0, 800.0])
def test_long_series_bessel_matches_scipy(kappa):
    order = AMBIENT_DIMENSION / 2.0 - 1.0
    value = torch.tensor(kappa, dtype=torch.float64)
    actual = float(log_bessel_i(order, value))
    expected = math.log(float(ive(order, kappa))) + kappa
    assert actual == pytest.approx(expected, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize(
    "family", ["spcauchy", "vmf", "powerspherical"]
)
def test_family_samples_log_prob_and_gradients(family):
    location = torch.zeros(AMBIENT_DIMENSION, dtype=torch.float64)
    location[0] = 1.0
    model = DirectionalFamily(family, location)
    assert float(model.local_curvature.detach()) == pytest.approx(
        10.0, rel=1e-10, abs=1e-10
    )
    sample = model.rsample(32)
    log_prob = model.log_prob(sample)
    assert sample.shape == (32, AMBIENT_DIMENSION)
    assert torch.allclose(
        sample.norm(dim=-1), torch.ones(32, dtype=torch.float64), atol=2e-5
    )
    assert torch.isfinite(log_prob).all()
    (-log_prob.mean()).backward()
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_target_mixture_density_and_samples():
    location = torch.zeros(AMBIENT_DIMENSION, dtype=torch.float64)
    location[0] = 1.0
    target = TargetMixture(location, kappa=100, epsilon=0.1)
    sample = target.sample(128)
    assert sample.shape == (128, AMBIENT_DIMENSION)
    assert torch.isfinite(target.log_prob(sample)).all()


def test_initial_locations_have_one_fixed_angular_error():
    target = torch.zeros(AMBIENT_DIMENSION, dtype=torch.float64)
    target[0] = 1.0
    expected_cosine = math.cos(math.radians(10.0))
    directions = [initial_location(target, seed) for seed in range(3)]
    for direction in directions:
        assert float(direction.norm()) == pytest.approx(1.0, abs=1e-12)
        assert float(direction @ target) == pytest.approx(
            expected_cosine, abs=1e-12
        )
    assert not torch.allclose(directions[0], directions[1])


def test_predeclared_transition_decision():
    rows = []
    for kappa in (20, 100, 500):
        for epsilon in (0.0, 0.01, 0.05, 0.10, 0.20):
            for objective in ("forward_kl", "reverse_kl"):
                for family in ("spcauchy", "vmf", "powerspherical"):
                    preferred = (
                        "vmf"
                        if epsilon == 0.0 or objective == "reverse_kl"
                        else "spcauchy"
                    )
                    value = 0.0 if family == preferred else 1.0
                    rows.append(
                        {
                            "kappa": kappa,
                            "epsilon": epsilon,
                            "objective": objective,
                            "family": family,
                            f"{objective}_mean": value,
                            "tail_calibration_absolute_error_mean": (
                                0.0
                                if family
                                == (
                                    "vmf"
                                    if epsilon == 0.0
                                    else "spcauchy"
                                )
                                else 1.0
                            ),
                            "joint_mass_calibration_absolute_error_mean": (
                                0.0
                                if family
                                == (
                                    "vmf"
                                    if epsilon == 0.0
                                    else "spcauchy"
                                )
                                else 1.0
                            ),
                        }
                    )
    decision = _decision(rows)
    assert decision["epsilon_zero_vmf_sanity_passed"]
    assert decision["clean_interpretable_transition"]
    assert decision["recommend_include_in_paper"]


def _complete_grid_rows(seed_count=3):
    return [
        {
            "kappa": kappa,
            "epsilon": epsilon,
            "objective": objective,
            "family": family,
            "seed": seed,
            "initial_curvature": float(kappa),
            "initial_location_angle_degrees": 10.0,
            "steps": 600,
            "batch_size": 8192,
            "evaluation_samples": 1_000_000,
            "learning_rate": 0.03,
            "nonfinite_count": 0,
        }
        for kappa in (20, 100, 500)
        for epsilon in (0.0, 0.01, 0.05, 0.10, 0.20)
        for objective in ("forward_kl", "reverse_kl")
        for family in ("spcauchy", "vmf", "powerspherical")
        for seed in range(seed_count)
    ]


def test_complete_grid_validation_accepts_shared_contiguous_seeds():
    assert _validate_complete_grid(_complete_grid_rows()) == 3


def test_complete_grid_validation_rejects_missing_result():
    rows = _complete_grid_rows()
    rows.pop()
    with pytest.raises(RuntimeError, match="one seed set"):
        _validate_complete_grid(rows)


def test_complete_grid_validation_rejects_wrong_initial_curvature():
    rows = _complete_grid_rows()
    rows[0]["initial_curvature"] = 10.0
    with pytest.raises(RuntimeError, match="target-matched"):
        _validate_complete_grid(rows)
