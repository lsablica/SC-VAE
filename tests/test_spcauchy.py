import torch

from src.spcauchy import sample_spcauchy


def test_sample_spcauchy_outputs_unit_vectors():
    mu = torch.tensor([[1.0, 0.0, 0.0]])
    rho = torch.tensor([[0.5]])

    samples = sample_spcauchy(mu, rho)
    norms = torch.norm(samples, dim=1)

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
