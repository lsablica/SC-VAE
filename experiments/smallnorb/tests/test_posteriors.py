from __future__ import annotations

import math

import pytest
import torch
from scipy.special import ive

from experiments.smallnorb.posteriors import build_posterior
from experiments.smallnorb.vendor_vmf_smallnorb import (
    VonMisesFisher,
    bessel_ratio_perron_Iv_Ivminus1,
    log_besselI_via_0f1,
)


@pytest.mark.parametrize(
    "family",
    (
        "spcauchy",
        "vmf_robust",
        "powerspherical",
        "gaussian_isotropic",
        "gaussian_diagonal",
    ),
)
def test_posterior_shapes_samples_kl_and_gradients(family):
    posterior = build_posterior(family)
    hidden = torch.randn(4, 512, requires_grad=True)
    parameters = posterior.encode(hidden)
    sample = posterior.sample(parameters)
    kl = posterior.kl(parameters)
    assert parameters.location.shape[0] == 4
    assert sample.shape == (4, 33)
    assert kl.shape == (4,)
    assert torch.isfinite(sample).all()
    assert torch.isfinite(kl).all()
    (sample.square().mean() + kl.mean()).backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    if posterior.is_spherical:
        assert torch.allclose(
            sample.norm(dim=-1), torch.ones(4), atol=2e-4
        )
        expected_cosine = posterior.expected_cosine(parameters)
        if expected_cosine is not None:
            assert expected_cosine.shape == (4,)
            assert torch.isfinite(expected_cosine).all()
            assert ((expected_cosine >= -1) & (expected_cosine <= 1)).all()


@pytest.mark.parametrize(
    "family", ("spcauchy", "vmf_robust", "powerspherical")
)
def test_spherical_initial_kl_is_matched_to_point_one_nat(family):
    posterior = build_posterior(family, initial_kl=0.1)
    parameters = posterior.encode(torch.zeros(16, 512))
    kl = posterior.kl(parameters)
    assert float(kl.mean().detach()) == pytest.approx(0.1, abs=2e-5)


def test_spcauchy_uses_certified_direct_custom_backward():
    posterior = build_posterior("spcauchy")
    parameters = posterior.encode(torch.randn(3, 512))
    kl = posterior.kl(parameters)
    function_names = set()
    pending = [kl.grad_fn]
    while pending:
        function = pending.pop()
        if function is None or id(function) in function_names:
            continue
        function_names.add(id(function))
        function_names.add(type(function).__name__)
        pending.extend(
            child for child, _ in function.next_functions
        )
    assert "_DirectKlAutogradBackward" in function_names
    diagnostics = posterior.term_diagnostics(torch.device("cpu"))
    assert diagnostics["retained_terms"] > 0
    assert diagnostics["backend"] in {
        "vectorized",
        "compiled",
        "triton",
    }


def test_isotropic_gaussian_exact_kl_formula():
    posterior = build_posterior("gaussian_isotropic")
    parameters = posterior.encode(torch.randn(2, 512))
    variance = parameters.scale.square().squeeze(-1)
    expected = 0.5 * (
        parameters.location.square().sum(dim=-1)
        + 32 * (variance - 1 - torch.log(variance))
    )
    assert torch.allclose(posterior.kl(parameters), expected)


@pytest.mark.parametrize(
    "kappa",
    (
        1.0,
        20.0,
        49.0,
        50.0,
        51.0,
        100.0,
        200.0,
        350.0,
        500.0,
        1000.0,
        5000.0,
        10000.0,
    ),
)
def test_smallnorb_vmf_normalizer_covers_full_concentration_clamp(kappa):
    order = 33 / 2 - 1
    value = torch.tensor(
        [kappa], dtype=torch.float32, requires_grad=True
    )
    log_bessel = log_besselI_via_0f1(order, value)
    actual = float(log_bessel.detach())
    expected = math.log(float(ive(order, kappa))) + kappa
    assert actual == pytest.approx(expected, abs=2e-4)
    derivative = float(torch.autograd.grad(log_bessel.sum(), value)[0])
    expected_derivative = (
        float(ive(order + 1.0, kappa) / ive(order, kappa))
        + order / kappa
    )
    assert derivative == pytest.approx(expected_derivative, abs=3e-6)
    ratio = float(
        bessel_ratio_perron_Iv_Ivminus1(
            order + 1.0,
            torch.tensor([kappa], dtype=torch.float32),
        )
    )
    expected_ratio = float(
        ive(order + 1.0, kappa) / ive(order, kappa)
    )
    assert ratio == pytest.approx(expected_ratio, abs=2e-6)


def test_smallnorb_vmf_high_concentration_sample_entropy_and_gradient():
    location = torch.zeros(8, 33)
    location[:, 1] = 1.0
    concentration = torch.full(
        (8, 1), 10_000.0, requires_grad=True
    )
    distribution = VonMisesFisher(location, concentration)
    sample = distribution.rsample()
    entropy = distribution.entropy()
    assert torch.isfinite(sample).all()
    assert torch.isfinite(entropy).all()
    assert torch.allclose(
        sample.norm(dim=-1), torch.ones(8), atol=2e-4
    )
    (sample[:, 0].mean() + entropy.mean()).backward()
    assert concentration.grad is not None
    assert torch.isfinite(concentration.grad).all()
