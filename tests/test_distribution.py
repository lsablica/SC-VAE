import math

import pytest
import torch
from torch.distributions import Distribution, kl_divergence

from spherical_cauchy import HypersphericalUniform, SphericalCauchy
from spherical_cauchy.functional import _log_surface_area, mobius_transform


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_distribution_shapes_expand_and_sample(dtype):
    loc = torch.nn.functional.normalize(torch.randn(2, 1, 5, dtype=dtype), dim=-1)
    concentration = torch.tensor([[0.2, 0.7, 0.8]], dtype=dtype)
    distribution = SphericalCauchy(loc, concentration, validate_args=True)

    assert isinstance(distribution, Distribution)
    assert distribution.batch_shape == torch.Size((2, 3))
    assert distribution.event_shape == torch.Size((5,))
    assert distribution.ambient_dim == 5
    assert distribution.rho.shape == (2, 3)
    assert distribution.ball_parameter.shape == (2, 3, 5)
    assert torch.equal(distribution.mode, distribution.loc)

    sample = distribution.rsample((7, 4))
    assert sample.shape == (7, 4, 2, 3, 5)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    assert torch.allclose(
        sample.norm(dim=-1), torch.ones_like(sample[..., 0]), atol=tolerance
    )
    assert distribution.log_prob(sample).shape == (7, 4, 2, 3)
    assert distribution.entropy().shape == (2, 3)
    assert not distribution.sample().requires_grad

    expanded = distribution.expand((6, 2, 3))
    assert expanded.batch_shape == torch.Size((6, 2, 3))
    assert expanded.rsample((2,)).shape == (2, 6, 2, 3, 5)


def test_trailing_singleton_concentration_and_scalar_broadcast():
    loc = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    column = SphericalCauchy(loc, torch.tensor([[0.2], [0.4]]))
    scalar = SphericalCauchy(loc, 0.3)
    assert column.concentration.shape == (2,)
    assert scalar.concentration.shape == (2,)


def test_mobius_map_matches_frozen_sampler_for_fixed_noise():
    generator = torch.Generator().manual_seed(981)
    base = torch.randn(12, 6, generator=generator, dtype=torch.float64)
    base = torch.nn.functional.normalize(base, dim=-1)
    loc = torch.randn(12, 6, generator=generator, dtype=torch.float64)
    loc = torch.nn.functional.normalize(loc, dim=-1)
    concentration = torch.linspace(0.0, 0.9, 12, dtype=torch.float64)
    rho = concentration[:, None]
    normalized_loc = torch.nn.functional.normalize(loc, dim=-1)
    inner = (base * normalized_loc).sum(dim=-1, keepdim=True)
    expected = (1.0 - rho.square()) * (
        (base + rho * normalized_loc) / (1.0 + 2.0 * rho * inner + rho.square())
    ) + rho * normalized_loc
    actual = mobius_transform(base, loc, concentration)
    assert torch.equal(actual, expected)


def test_validation_rejects_invalid_parameters_and_samples():
    with pytest.raises(ValueError, match="Expected parameter loc"):
        SphericalCauchy(torch.tensor([2.0, 0.0, 0.0]), 0.2, validate_args=True)
    with pytest.raises(ValueError, match="Expected parameter concentration"):
        SphericalCauchy(torch.tensor([1.0, 0.0, 0.0]), 1.0, validate_args=True)

    distribution = SphericalCauchy(
        torch.tensor([1.0, 0.0, 0.0]), 0.2, validate_args=True
    )
    with pytest.raises(ValueError, match="support"):
        distribution.log_prob(torch.tensor([0.5, 0.0, 0.0]))


def test_rsample_gradients_flow_through_both_parameters():
    raw_loc = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    loc = torch.nn.functional.normalize(raw_loc, dim=-1)
    concentration = torch.full((8,), 0.55, dtype=torch.float64, requires_grad=True)
    sample = SphericalCauchy(loc, concentration).rsample((32,))
    loss = sample[..., 0].mean() + sample[..., 1].square().mean()
    loss.backward()
    assert raw_loc.grad is not None and torch.isfinite(raw_loc.grad).all()
    assert concentration.grad is not None
    assert torch.isfinite(concentration.grad).all()


def test_uncertain_moments_are_not_invented():
    distribution = SphericalCauchy(torch.tensor([1.0, 0.0, 0.0]), 0.3)
    with pytest.raises(NotImplementedError):
        _ = distribution.mean
    with pytest.raises(NotImplementedError):
        _ = distribution.variance


@pytest.mark.parametrize("ambient_dim", [2, 3, 4, 8, 33])
def test_log_density_normalizes_numerically(ambient_dim):
    generator = torch.Generator().manual_seed(1200 + ambient_dim)
    base = torch.randn(120_000, ambient_dim, generator=generator, dtype=torch.float64)
    base = torch.nn.functional.normalize(base, dim=-1)
    loc = torch.zeros(ambient_dim, dtype=torch.float64)
    loc[0] = 1.0
    # A moderate fixed concentration keeps uniform-importance variance small
    # enough for the high-dimensional check to remain deterministic.
    distribution = SphericalCauchy(loc, 0.1)
    log_uniform = -_log_surface_area(ambient_dim, device=base.device, dtype=base.dtype)
    integral = torch.exp(distribution.log_prob(base) - log_uniform).mean()
    assert integral.item() == pytest.approx(1.0, abs=0.01)


def test_hyperspherical_uniform_semantics():
    distribution = HypersphericalUniform(
        7,
        batch_shape=torch.Size((2, 3)),
        dtype=torch.float64,
        validate_args=True,
    )
    sample = distribution.rsample((5,))
    assert isinstance(distribution, Distribution)
    assert distribution.batch_shape == torch.Size((2, 3))
    assert distribution.event_shape == torch.Size((7,))
    assert sample.shape == (5, 2, 3, 7)
    assert distribution.log_prob(sample).shape == (5, 2, 3)
    expected_entropy = math.log(2.0 * math.pi ** (7.0 / 2.0) / math.gamma(7.0 / 2.0))
    assert torch.allclose(
        distribution.entropy(),
        torch.full((2, 3), expected_entropy, dtype=torch.float64),
    )
    assert distribution.expand((4, 2, 3)).batch_shape == (4, 2, 3)


def test_registered_kl_routes_broadcast_and_differentiate():
    first_raw = torch.randn(2, 1, 5, dtype=torch.float64, requires_grad=True)
    second_raw = torch.randn(1, 3, 5, dtype=torch.float64, requires_grad=True)
    first_loc = torch.nn.functional.normalize(first_raw, dim=-1)
    second_loc = torch.nn.functional.normalize(second_raw, dim=-1)
    first_rho = torch.full((2, 1), 0.3, dtype=torch.float64, requires_grad=True)
    second_rho = torch.full((1, 3), 0.6, dtype=torch.float64, requires_grad=True)
    first = SphericalCauchy(first_loc, first_rho)
    second = SphericalCauchy(second_loc, second_rho)
    uniform_first = HypersphericalUniform(5, first.batch_shape, dtype=torch.float64)
    uniform_second = HypersphericalUniform(5, second.batch_shape, dtype=torch.float64)

    assert kl_divergence(first, uniform_second).shape == (2, 3)
    assert kl_divergence(uniform_first, second).shape == (2, 3)
    pairwise = kl_divergence(first, second)
    reverse = kl_divergence(second, first)
    assert pairwise.shape == (2, 3)
    assert torch.allclose(pairwise, reverse, atol=2e-12, rtol=2e-12)
    assert torch.equal(
        kl_divergence(uniform_first, uniform_second),
        torch.zeros((2, 3), dtype=torch.float64),
    )

    pairwise.sum().backward()
    for gradient in (first_raw.grad, second_raw.grad, first_rho.grad, second_rho.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
