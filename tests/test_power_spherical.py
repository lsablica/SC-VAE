import torch
import torch.nn.functional as F

from benchmark.vendor_power_spherical import (
    HypersphericalUniform,
    PowerSpherical,
)
from experiments.mnist.config import RunSpec, build_benchmark_specs
from experiments.mnist.models import create_model


def test_power_spherical_sample_and_kl_are_finite():
    loc = F.normalize(torch.randn(8, 17), dim=-1)
    exponent = torch.full((8,), 20.0, requires_grad=True)
    distribution = PowerSpherical(loc, exponent)
    prior = HypersphericalUniform(17)
    sample = distribution.rsample()
    kl = torch.distributions.kl_divergence(distribution, prior)
    assert sample.shape == loc.shape
    assert torch.allclose(sample.norm(dim=-1), torch.ones(8), atol=1e-5)
    assert torch.all(kl >= 0)
    kl.sum().backward()
    assert torch.isfinite(exponent.grad).all()


def test_power_spherical_kl_matches_closed_form():
    for dimension in [2, 3, 17]:
        loc = F.normalize(torch.randn(4, dimension, dtype=torch.float64), dim=-1)
        exponent = torch.tensor([0.0, 0.5, 2.0, 20.0], dtype=torch.float64)
        distribution = PowerSpherical(loc, exponent)
        prior = HypersphericalUniform(dimension, dtype=torch.float64)
        actual = torch.distributions.kl_divergence(distribution, prior)
        a = torch.tensor((dimension - 1) / 2.0, dtype=torch.float64)
        expected = (
            torch.lgamma(a)
            + torch.lgamma(a)
            - torch.lgamma(2.0 * a)
            - torch.lgamma(a + exponent)
            - torch.lgamma(a)
            + torch.lgamma(2.0 * a + exponent)
            + exponent
            * (torch.digamma(a + exponent) - torch.digamma(2.0 * a + exponent))
        )
        assert torch.allclose(actual, expected, atol=2e-12, rtol=2e-12)


def test_power_spherical_mnist_model_smoke():
    spec = build_benchmark_specs(["powerspherical"], [3], [0], epochs=1)[0]
    model = create_model(spec, "cpu")
    batch = torch.rand(2, 1, 28, 28)
    reconstruction, loc, exponent = model(batch)
    loss, recon_loss, kl = model.loss_function(batch, reconstruction, loc, exponent)
    loss.backward()
    assert reconstruction.shape == batch.shape
    assert exponent.shape == (2,)
    assert torch.isfinite(loss)
    assert recon_loss > 0
    assert kl >= 0


def test_missing_mnist_kl_method_defaults_to_direct():
    current = build_benchmark_specs(["spcauchy"], [3], [0], epochs=1)[0]
    payload = current.to_dict()
    payload.pop("spcauchy_kl_method")
    payload.pop("config_schema_version")
    payload.pop("num_workers")
    restored = RunSpec.from_dict(payload)
    assert restored.spcauchy_kl_method == "direct"
    assert restored.config_schema_version == 1
    assert restored.num_workers == 0
