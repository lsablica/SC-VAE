"""Image VAE backbone frozen for the paper's MNIST comparison."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from spherical_cauchy import (
    SphericalCauchy,
    spherical_cauchy_kl,
    spherical_cauchy_kl_fixed,
    spherical_cauchy_laplace_kl,
    spherical_cauchy_neighbor_kl,
)


@dataclass
class ImageVAEConfig:
    input_dim: list[int]
    latent_dim: int
    hidden_dims: list[int]
    distribution_type: str = "spcauchy"
    spcauchy_kl_method: str = "direct"
    encoder_type: str = "cnn"
    decoder_type: str = "cnn"
    is_image_input: bool = True
    kl_weight: float = 1.0
    dropout_rate: float = 0.0
    activation: str = "relu"
    spcauchy_rho_bias_init: float = 2.0


class Reshape(nn.Module):
    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.shape = shape

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.view(*self.shape)


class ImageVAE(nn.Module):
    """CNN/MLP VAE used by all four MNIST posterior families."""

    def __init__(self, config: ImageVAEConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.is_image_input = config.is_image_input
        self.distribution_type = config.distribution_type
        if self.distribution_type not in {"spcauchy", "normal"}:
            raise ValueError(
                f"Unsupported distribution type: {self.distribution_type}"
            )
        if config.spcauchy_kl_method not in {
            "direct",
            "direct_fixed",
            "neighbor",
            "laplace",
        }:
            raise ValueError(
                f"Unsupported spherical Cauchy KL route: "
                f"{config.spcauchy_kl_method}"
            )
        self.spcauchy_kl_method = config.spcauchy_kl_method
        channels, height, width = config.input_dim
        self.input_size = channels * height * width
        self.input_shape = config.input_dim
        self.encoder = (
            self._build_cnn_encoder()
            if config.encoder_type == "cnn"
            else self._build_mlp_encoder()
        )
        self.fc_mu = nn.Linear(config.hidden_dims[-1], self.latent_dim)
        second_width = 1 if self.distribution_type == "spcauchy" else self.latent_dim
        self.fc_second_param = nn.Linear(config.hidden_dims[-1], second_width)
        if self.distribution_type == "spcauchy":
            with torch.no_grad():
                self.fc_second_param.bias.fill_(config.spcauchy_rho_bias_init)
        self.decoder = (
            self._build_cnn_decoder()
            if config.decoder_type == "cnn"
            else self._build_mlp_decoder()
        )

    def _activation(self) -> nn.Module:
        if self.config.activation == "relu":
            return nn.ReLU()
        if self.config.activation == "leaky_relu":
            return nn.LeakyReLU(0.2)
        if self.config.activation == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unsupported activation: {self.config.activation}")

    def _build_mlp_encoder(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        previous = self.input_size
        for width in self.config.hidden_dims:
            layers.extend((nn.Linear(previous, width), self._activation()))
            if self.config.dropout_rate > 0:
                layers.append(nn.Dropout(self.config.dropout_rate))
            previous = width
        return nn.Sequential(*layers)

    def _build_mlp_decoder(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        previous = self.latent_dim
        for width in reversed(self.config.hidden_dims):
            layers.extend((nn.Linear(previous, width), self._activation()))
            if self.config.dropout_rate > 0:
                layers.append(nn.Dropout(self.config.dropout_rate))
            previous = width
        layers.extend((nn.Linear(previous, self.input_size), nn.Sigmoid()))
        return nn.Sequential(*layers)

    def _build_cnn_encoder(self) -> nn.Sequential:
        channels, height, width = self.input_shape
        height //= 8
        width //= 8
        return nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * height * width, self.config.hidden_dims[-1]),
            nn.ReLU(),
        )

    def _build_cnn_decoder(self) -> nn.Sequential:
        channels, height, width = self.input_shape
        height //= 4
        width //= 4
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.config.hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dims[-1], 128 * height * width),
            nn.ReLU(),
            Reshape((-1, 128, height, width)),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.encoder_type == "mlp":
            value = value.view(value.size(0), -1)
        elif value.ndim != 4:
            value = value.view(-1, *self.input_shape)
        hidden = self.encoder(value)
        location = self.fc_mu(hidden)
        second = self.fc_second_param(hidden)
        if self.distribution_type == "spcauchy":
            location = F.normalize(location, p=2, dim=1)
            concentration = torch.sigmoid(second)
            one = concentration.new_ones(())
            concentration = torch.minimum(
                concentration,
                torch.nextafter(one, torch.zeros_like(one)),
            )
            return location, concentration
        return location, second

    def reparameterize(
        self, location: torch.Tensor, second: torch.Tensor
    ) -> torch.Tensor:
        if self.distribution_type == "spcauchy":
            return SphericalCauchy(location, second).rsample()
        standard_deviation = torch.exp(0.5 * second)
        return location + torch.randn_like(standard_deviation) * standard_deviation

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        reconstruction = self.decoder(latent)
        if self.config.decoder_type == "mlp":
            reconstruction = reconstruction.view(-1, *self.input_shape)
        return reconstruction

    def sample_prior(
        self, count: int, device: torch.device | str | None = None
    ) -> torch.Tensor:
        device = device or next(self.parameters()).device
        sample = torch.randn(count, self.latent_dim, device=device)
        if self.distribution_type == "spcauchy":
            sample = F.normalize(sample, p=2, dim=1)
        return sample

    def forward(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        location, second = self.encode(value)
        return self.decode(self.reparameterize(location, second)), location, second

    def kl_divergence(
        self, location: torch.Tensor, second: torch.Tensor
    ) -> torch.Tensor:
        if self.distribution_type == "normal":
            return -0.5 * torch.sum(
                1.0 + second - location.square() - second.exp(), dim=-1
            )
        concentration = second
        if self.spcauchy_kl_method == "direct":
            return spherical_cauchy_kl(concentration, self.latent_dim)
        if self.spcauchy_kl_method == "direct_fixed":
            return spherical_cauchy_kl_fixed(
                concentration,
                self.latent_dim,
                maximum_concentration=0.999,
                value_tolerance=2e-6,
                gradient_tolerance=2e-6,
            )
        if self.spcauchy_kl_method == "neighbor":
            return spherical_cauchy_neighbor_kl(
                concentration, self.latent_dim
            )
        return spherical_cauchy_laplace_kl(concentration, self.latent_dim)

    def loss_function(
        self,
        value: torch.Tensor,
        reconstruction: torch.Tensor,
        location: torch.Tensor,
        second: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstruction_loss = F.binary_cross_entropy(
            reconstruction, value, reduction="sum"
        ) / value.size(0)
        kl_loss = self.kl_divergence(location, second).mean()
        total = reconstruction_loss + self.config.kl_weight * kl_loss
        return total, reconstruction_loss, kl_loss
