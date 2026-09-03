"""Local CUDA acceptance tests for the release package and paper benchmark."""

from __future__ import annotations

import pytest
import torch
from torch.distributions import kl_divergence

from experiments.latent_layer.runtime import _run_spcauchy_iteration
from spherical_cauchy import (
    HypersphericalUniform,
    SphericalCauchy,
    spherical_cauchy_laplace_kl,
)
from spherical_cauchy.direct import (
    kl_divergence_spcauchy_direct,
    kl_divergence_spcauchy_direct_with_gradient,
)
from spherical_cauchy.triton_backend import triton_is_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)


def test_direct_triton_parity():
    assert triton_is_available(), "The local CUDA release check requires Triton"
    rho_cpu = torch.tensor([0.0, 0.01, 0.1, 0.5, 0.9, 0.999], dtype=torch.float64)
    for ambient_dim in (7, 8, 17, 128, 2048):
        expected, expected_gradient = kl_divergence_spcauchy_direct_with_gradient(
            rho_cpu, ambient_dim, backend="vectorized"
        )
        rho_cuda = rho_cpu.cuda().requires_grad_(True)
        actual = kl_divergence_spcauchy_direct(rho_cuda, ambient_dim, backend="triton")
        gradient = torch.autograd.grad(actual.sum(), rho_cuda)[0]
        torch.testing.assert_close(actual.cpu(), expected, atol=3e-11, rtol=3e-12)
        torch.testing.assert_close(
            gradient.cpu(), expected_gradient, atol=3e-10, rtol=3e-11
        )


def test_custom_backward_parity():
    rho = torch.linspace(0.01, 0.98, 257, device="cuda", dtype=torch.float64)
    expected_value, expected_gradient = kl_divergence_spcauchy_direct_with_gradient(
        rho, 33, backend="vectorized"
    )
    differentiable_rho = rho.detach().requires_grad_(True)
    value = kl_divergence_spcauchy_direct(differentiable_rho, 33, backend="vectorized")
    gradient = torch.autograd.grad(value.sum(), differentiable_rho)[0]
    torch.testing.assert_close(value, expected_value, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(gradient, expected_gradient, atol=2e-11, rtol=2e-11)


def test_laplace_triton_parity():
    assert triton_is_available(), "The local CUDA release check requires Triton"
    rho_cpu = torch.tensor(
        [0.0, 1e-4, 0.01, 0.1, 0.5, 0.9, 0.99],
        dtype=torch.float64,
        requires_grad=True,
    )
    for ambient_dim in (8, 128, 2048, 4096):
        expected = spherical_cauchy_laplace_kl(
            rho_cpu,
            ambient_dim,
            backend="eager",
        )
        expected_gradient = torch.autograd.grad(
            expected.sum(), rho_cpu, retain_graph=True
        )[0]
        rho_cuda = rho_cpu.detach().cuda().requires_grad_(True)
        actual = spherical_cauchy_laplace_kl(
            rho_cuda,
            ambient_dim,
            backend="triton",
        )
        gradient = torch.autograd.grad(actual.sum(), rho_cuda)[0]
        torch.testing.assert_close(actual.cpu(), expected, atol=3e-11, rtol=3e-12)
        torch.testing.assert_close(
            gradient.cpu(), expected_gradient, atol=4e-10, rtol=4e-11
        )


def test_distribution_rsample_gradient():
    raw_loc = torch.randn(32, 33, device="cuda", requires_grad=True)
    loc = torch.nn.functional.normalize(raw_loc, dim=-1)
    concentration = torch.full((32,), 0.7, device="cuda", requires_grad=True)
    sample = SphericalCauchy(loc, concentration).rsample((8,))
    loss = sample[..., :2].square().mean()
    loss.backward()
    assert raw_loc.grad is not None and torch.isfinite(raw_loc.grad).all()
    assert concentration.grad is not None
    assert torch.isfinite(concentration.grad).all()


def test_registered_kl_gradient():
    raw_loc = torch.randn(32, 33, device="cuda", requires_grad=True)
    loc = torch.nn.functional.normalize(raw_loc, dim=-1)
    concentration = torch.full((32,), 0.7, device="cuda", requires_grad=True)
    posterior = SphericalCauchy(loc, concentration)
    prior = HypersphericalUniform(33, batch_shape=posterior.batch_shape, device="cuda")
    kl_divergence(posterior, prior).sum().backward()
    assert concentration.grad is not None
    assert torch.isfinite(concentration.grad).all()


def test_paper_benchmark_smoke():
    result = _run_spcauchy_iteration(
        "spcauchy_direct",
        batch_size=64,
        latent_dim=128,
        rho_value=0.4,
        device=torch.device("cuda"),
        dtype=torch.float32,
        direct_backend="triton",
        laplace_backend="triton",
        fixed_maximum_concentration=None,
        fixed_value_tolerance=2e-6,
        fixed_gradient_tolerance=2e-6,
    )
    assert result["total_time_s"] > 0.0
    assert result["peak_memory_bytes"] > 0
    assert not result["nan_or_inf_loss"]
    assert not result["nan_or_inf_grad"]
