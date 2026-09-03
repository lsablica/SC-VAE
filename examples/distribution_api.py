"""Basic distribution, sampling, density, entropy, and KL usage."""

import torch
from torch.distributions import kl_divergence

from spherical_cauchy import HypersphericalUniform, SphericalCauchy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loc = torch.nn.functional.normalize(torch.randn(8, 5, device=device), dim=-1)
concentration = torch.linspace(0.1, 0.8, 8, device=device)
posterior = SphericalCauchy(loc, concentration)
prior = HypersphericalUniform(
    ambient_dim=5,
    batch_shape=posterior.batch_shape,
    device=device,
    dtype=loc.dtype,
)

sample = posterior.rsample((4,))
print("sample:", sample.shape, sample.device)
print("log_prob:", posterior.log_prob(sample).shape)
print("entropy:", posterior.entropy().shape)
print("KL(q || uniform):", kl_divergence(posterior, prior).shape)
