"""Focused tests for the exact pairwise spherical Cauchy KL identity."""

from __future__ import annotations

import pytest
import torch

from spherical_cauchy import (
    pseudohyperbolic_distance,
    spherical_cauchy_kl,
    spherical_cauchy_pairwise_kl,
)


def _ball_points(
    batch: int, dimension: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260726 + dimension)
    a = torch.randn(batch, dimension, generator=generator, dtype=dtype)
    b = torch.randn(batch, dimension, generator=generator, dtype=dtype)
    a = a / torch.linalg.vector_norm(a, dim=-1, keepdim=True)
    b = b / torch.linalg.vector_norm(b, dim=-1, keepdim=True)
    radii_a = torch.linspace(0.1, 0.75, batch, dtype=dtype).unsqueeze(-1)
    radii_b = torch.linspace(0.7, 0.2, batch, dtype=dtype).unsqueeze(-1)
    return radii_a * a, radii_b * b


@pytest.mark.parametrize("dimension", [2, 3, 4, 8, 17, 33])
def test_pairwise_kl_is_symmetric_and_reduces_to_direct(dimension: int) -> None:
    a, b = _ball_points(5, dimension, torch.float64)
    delta_ab = pseudohyperbolic_distance(a, b)
    delta_ba = pseudohyperbolic_distance(b, a)
    expected = spherical_cauchy_kl(delta_ab, dimension, backend="vectorized")
    forward = spherical_cauchy_pairwise_kl(a, b, backend="vectorized")
    reverse = spherical_cauchy_pairwise_kl(b, a, backend="vectorized")

    torch.testing.assert_close(delta_ab, delta_ba, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(forward, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(reverse, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dimension", [2, 5, 8, 33])
def test_pairwise_kl_with_uniform_parameter_matches_uniform_prior(
    dimension: int,
) -> None:
    a, _ = _ball_points(4, dimension, torch.float64)
    zero = torch.zeros_like(a)
    radius = torch.linalg.vector_norm(a, dim=-1)
    expected = spherical_cauchy_kl(radius, dimension, backend="vectorized")
    pairwise = spherical_cauchy_pairwise_kl(a, zero, backend="vectorized")
    torch.testing.assert_close(pairwise, expected, rtol=1e-12, atol=1e-12)


def test_pairwise_kl_has_finite_zero_gradient_at_equality() -> None:
    a = torch.tensor([[0.2, -0.1, 0.05, 0.0]], dtype=torch.float64, requires_grad=True)
    value = spherical_cauchy_pairwise_kl(a, a, backend="vectorized").sum()
    gradient = torch.autograd.grad(value, a)[0]
    assert torch.isfinite(value)
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(value, torch.zeros_like(value), atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        gradient, torch.zeros_like(gradient), atol=1e-14, rtol=0.0
    )


def test_pseudohyperbolic_distance_is_stable_near_ball_boundary() -> None:
    dtype = torch.float32
    radius = torch.tensor(0.9999, dtype=dtype)
    angle = torch.tensor(1e-3, dtype=dtype)
    a = torch.stack((radius, torch.tensor(0.0, dtype=dtype))).unsqueeze(0)
    b = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle))).unsqueeze(0)
    distance = pseudohyperbolic_distance(a, b)

    difference_squared = (a - b).square().sum(dim=-1).double()
    reference = torch.sqrt(
        difference_squared
        / (
            difference_squared
            + (1.0 - a.double().square().sum(dim=-1))
            * (1.0 - b.double().square().sum(dim=-1))
        )
    ).float()
    assert torch.isfinite(distance).all()
    assert bool((distance > 0).all())
    assert bool((distance < 1).all())
    torch.testing.assert_close(distance, reference, rtol=2e-4, atol=2e-5)


def test_pairwise_kl_gradients_match_finite_differences() -> None:
    a = torch.tensor(
        [[0.21, -0.14, 0.08, 0.03]], dtype=torch.float64, requires_grad=True
    )
    b = torch.tensor(
        [[-0.17, 0.19, 0.04, -0.06]], dtype=torch.float64, requires_grad=True
    )
    value = spherical_cauchy_pairwise_kl(a, b, backend="vectorized").sum()
    grad_a, grad_b = torch.autograd.grad(value, (a, b))

    step = 1e-6
    numerical_a = torch.zeros_like(a)
    numerical_b = torch.zeros_like(b)
    for index in range(a.shape[-1]):
        direction = torch.zeros_like(a)
        direction[0, index] = step
        plus = spherical_cauchy_pairwise_kl(
            a.detach() + direction, b.detach(), backend="vectorized"
        )
        minus = spherical_cauchy_pairwise_kl(
            a.detach() - direction, b.detach(), backend="vectorized"
        )
        numerical_a[0, index] = (plus - minus) / (2 * step)

        plus = spherical_cauchy_pairwise_kl(
            a.detach(), b.detach() + direction, backend="vectorized"
        )
        minus = spherical_cauchy_pairwise_kl(
            a.detach(), b.detach() - direction, backend="vectorized"
        )
        numerical_b[0, index] = (plus - minus) / (2 * step)

    torch.testing.assert_close(grad_a, numerical_a, rtol=2e-6, atol=2e-8)
    torch.testing.assert_close(grad_b, numerical_b, rtol=2e-6, atol=2e-8)


def test_pairwise_parameter_validation() -> None:
    a = torch.zeros(2, 3, dtype=torch.float64)
    b = torch.zeros(2, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="identical event shapes"):
        pseudohyperbolic_distance(a, b)

    outside = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="open unit ball"):
        pseudohyperbolic_distance(outside, outside)
