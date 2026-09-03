"""Shared convolutional architecture for every posterior family."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AMBIENT_DIMENSION, RunConfig
from .posteriors import (
    PosteriorParameters,
    build_posterior,
)
from .utils import parameter_count


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = max(1, channels // 8)
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs + self.layers(inputs))


class SharedEncoder(nn.Module):
    def __init__(self, architecture: str = "baseline_cnn"):
        super().__init__()
        deep = architecture == "deep_residual_cnn"
        layers: list[nn.Module] = [
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        ]
        if deep:
            layers.append(ResidualBlock(32))
        layers.extend(
            [
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            ResidualBlock(64),
            *([ResidualBlock(64)] if deep else []),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            ResidualBlock(128),
            *([ResidualBlock(128)] if deep else []),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            *([ResidualBlock(256)] if deep else []),
            ]
        )
        self.features = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(inputs))


class SharedDecoder(nn.Module):
    def __init__(self, architecture: str = "baseline_cnn"):
        super().__init__()
        deep = architecture == "deep_residual_cnn"
        self.projection = nn.Sequential(
            nn.Linear(AMBIENT_DIMENSION, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 256 * 4 * 4),
            nn.SiLU(),
        )
        layers: list[nn.Module] = []
        if deep:
            layers.append(ResidualBlock(256))
        layers.extend(
            [
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            ResidualBlock(128),
            *([ResidualBlock(128)] if deep else []),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            ResidualBlock(64),
            *([ResidualBlock(64)] if deep else []),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            *([ResidualBlock(32)] if deep else []),
            nn.ConvTranspose2d(
                32, 1, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
            ]
        )
        self.image = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] != AMBIENT_DIMENSION:
            raise ValueError(
                "Every decoder input must have dimension 33, got "
                f"{latent.shape[-1]}"
            )
        hidden = self.projection(latent)
        return self.image(hidden.view(-1, 256, 4, 4))


@dataclass
class VAEOutput:
    reconstruction: torch.Tensor
    parameters: PosteriorParameters
    latent: torch.Tensor
    kl: torch.Tensor


class SmallNORBViewVAE(nn.Module):
    """One encoder and decoder with a family-specific posterior head."""

    def __init__(self, config: RunConfig):
        super().__init__()
        self.config = config
        self.encoder = SharedEncoder(config.architecture)
        self.posterior = build_posterior(
            config.family,
            hidden_dimension=512,
            initial_kl=config.initial_spherical_kl,
            spcauchy_backend=config.spcauchy_backend,
        )
        self.decoder = SharedDecoder(config.architecture)

    @property
    def family(self) -> str:
        return self.config.family

    def encode(self, inputs: torch.Tensor) -> PosteriorParameters:
        return self.posterior.encode(self.encoder(inputs))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        sample: bool = True,
    ) -> VAEOutput:
        parameters = self.encode(inputs)
        latent = (
            self.posterior.sample(parameters)
            if sample
            else self.posterior.representative(parameters)
        )
        reconstruction = self.decode(latent)
        kl = self.posterior.kl(parameters)
        return VAEOutput(reconstruction, parameters, latent, kl)

    def deterministic_reconstruction(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, PosteriorParameters]:
        parameters = self.encode(inputs)
        latent = self.posterior.representative(parameters)
        return self.decode(latent), parameters

    def loss(
        self,
        inputs: torch.Tensor,
        *,
        beta: float,
        sigma_x: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], VAEOutput]:
        output = self(inputs, sample=True)
        squared_error = (
            output.reconstruction.float() - inputs.float()
        ).square()
        reconstruction_nll = squared_error.flatten(1).sum(dim=1) / (
            2.0 * sigma_x * sigma_x
        )
        total = reconstruction_nll + beta * output.kl.float()
        components = {
            "total": total,
            "reconstruction_nll": reconstruction_nll,
            "kl": output.kl.float(),
            "pixel_mse": squared_error.flatten(1).mean(dim=1),
        }
        return total.mean(), components, output

    def parameter_summary(self) -> dict[str, int]:
        return {
            "total": parameter_count(self),
            "shared_encoder": parameter_count(self.encoder),
            "posterior_heads": parameter_count(self.posterior),
            "shared_decoder": parameter_count(self.decoder),
        }


def build_model(config: RunConfig, device: torch.device) -> SmallNORBViewVAE:
    return SmallNORBViewVAE(config).to(device)


__all__ = [
    "ResidualBlock",
    "SharedDecoder",
    "SharedEncoder",
    "SmallNORBViewVAE",
    "VAEOutput",
    "build_model",
]
