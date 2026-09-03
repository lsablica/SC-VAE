"""A compact MNIST-shaped encoder using a spherical Cauchy posterior."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence

from spherical_cauchy import HypersphericalUniform, SphericalCauchy


class MinimalVAE(nn.Module):
    def __init__(self, ambient_dim: int = 3):
        super().__init__()
        self.ambient_dim = ambient_dim
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU())
        self.loc_head = nn.Linear(128, ambient_dim)
        self.rho_head = nn.Linear(128, 1)
        self.decoder = nn.Sequential(
            nn.Linear(ambient_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
        )

    def forward(self, images: torch.Tensor):
        hidden = self.encoder(images)
        loc = F.normalize(self.loc_head(hidden), dim=-1)
        rho = torch.sigmoid(self.rho_head(hidden)).squeeze(-1)
        posterior = SphericalCauchy(loc, rho)
        prior = HypersphericalUniform(
            self.ambient_dim,
            batch_shape=posterior.batch_shape,
            device=images.device,
            dtype=images.dtype,
        )
        latent = posterior.rsample()
        logits = self.decoder(latent).view_as(images)
        return logits, kl_divergence(posterior, prior)


model = MinimalVAE()
images = torch.rand(16, 1, 28, 28)
logits, kl = model(images)
reconstruction = (
    F.binary_cross_entropy_with_logits(logits, images, reduction="none")
    .flatten(1)
    .sum(-1)
)
loss = (reconstruction + kl).mean()
loss.backward()
print("ELBO loss:", loss.detach().item())
