import os
import sys

import pytest
import torch
import torch.nn.functional as F


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SpCauchyVAEConfig
from src.kl import (
    canonicalize_spcauchy_kl_approximation,
    kl_divergence_spcauchy_approx,
    kl_divergence_spcauchy_combined,
    spcauchy_h_approximation,
)
from src.model import SpCauchyVAE


def _spcauchy_h_bounds(z, latent_dim):
    c = z.new_tensor(float(latent_dim - 1))
    delta = z.new_tensor((latent_dim - 1) / 2)
    lower = torch.digamma(delta) - torch.digamma(c) + torch.log(2 - z)
    upper = torch.log1p(-0.5 * z)
    return lower, upper


def test_invalid_spcauchy_kl_approximation_raises():
    with pytest.raises(ValueError):
        canonicalize_spcauchy_kl_approximation("bad-mode")


def test_dynamic_alias_matches_weighted_surrogate():
    rho = torch.tensor([[0.2], [0.7], [0.95]], dtype=torch.float32)
    weighted = kl_divergence_spcauchy_combined(rho, 16, approximation="weighted")
    dynamic = kl_divergence_spcauchy_combined(rho, 16, approximation="dynamic")

    assert torch.allclose(weighted, dynamic)


def test_h_approximations_stay_inside_proven_bracket():
    rho = torch.linspace(0.05, 0.99, 33, dtype=torch.float32).view(-1, 1)
    z = 4 * rho / ((1 + rho) ** 2)

    for latent_dim in [3, 10, 100]:
        lower, upper = _spcauchy_h_bounds(z, latent_dim)
        midpoint = spcauchy_h_approximation(z, latent_dim, approximation="midpoint")
        weighted = spcauchy_h_approximation(z, latent_dim, approximation="weighted")

        assert torch.all(midpoint >= lower - 1e-6)
        assert torch.all(midpoint <= upper + 1e-6)
        assert torch.all(weighted >= lower - 1e-6)
        assert torch.all(weighted <= upper + 1e-6)


def test_model_uses_configured_spcauchy_kl_approximation():
    config = SpCauchyVAEConfig(
        input_dim=8,
        latent_dim=5,
        hidden_dims=[12],
        distribution_type="spcauchy",
        spcauchy_kl_approximation="dynamic",
        is_image_input=False,
    )
    model = SpCauchyVAE(config)

    mu = F.normalize(torch.randn(4, config.latent_dim), p=2, dim=1)
    rho = torch.tensor([[0.2], [0.4], [0.85], [0.97]], dtype=torch.float32)

    actual = model.kl_divergence(mu, rho)
    expected = kl_divergence_spcauchy_approx(rho, config.latent_dim, approximation="weighted")

    assert model.spcauchy_kl_approximation == "weighted"
    assert torch.allclose(actual, expected)
