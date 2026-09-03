from __future__ import annotations

import torch
import torch.nn.functional as F

from ._bootstrap import ensure_repo_root_on_path
from .config import RunSpec

ensure_repo_root_on_path()

from benchmark.vendor_vmf import HypersphericalUniform, VonMisesFisher  # noqa: E402
from benchmark.vendor_power_spherical import (  # noqa: E402
    HypersphericalUniform as PowerHypersphericalUniform,
    PowerSpherical,
)
from .vae import ImageVAE, ImageVAEConfig


class VMFVAE(ImageVAE):
    """MNIST vMF baseline sharing the common CNN backbone."""

    def __init__(self, config: ImageVAEConfig):
        original_dist = getattr(config, "distribution_type", "spcauchy")
        config.distribution_type = "spcauchy"
        super().__init__(config)
        self.distribution_type = "vmf"
        self.fc_kappa = self.fc_second_param
        config.distribution_type = original_dist

    def encode(self, x):
        if self.is_image_input and len(x.shape) != 4:
            x = x.view(-1, *self.input_shape)
        elif not self.is_image_input:
            x = x.view(x.size(0), -1)

        h = self.encoder(x)
        loc = F.normalize(self.fc_mu(h), p=2, dim=1)
        kappa = F.softplus(self.fc_kappa(h)) + 1.0
        return loc, kappa

    def reparameterize(self, loc, kappa):
        return VonMisesFisher(loc, kappa).rsample()

    def kl_divergence(self, loc, kappa):
        q_z = VonMisesFisher(loc, kappa)
        p_z = HypersphericalUniform(self.latent_dim - 1, device=loc.device)
        return torch.distributions.kl.kl_divergence(q_z, p_z)


class PowerSphericalVAE(ImageVAE):
    """Power Spherical baseline sharing the paper's CNN backbone."""

    def __init__(self, config: ImageVAEConfig):
        original_dist = getattr(config, "distribution_type", "spcauchy")
        config.distribution_type = "spcauchy"
        super().__init__(config)
        self.distribution_type = "powerspherical"
        self.fc_scale = self.fc_second_param
        with torch.no_grad():
            self.fc_scale.bias.fill_(
                float(getattr(config, "power_scale_bias_init", 0.0))
            )
        config.distribution_type = original_dist

    def encode(self, x):
        if self.is_image_input and len(x.shape) != 4:
            x = x.view(-1, *self.input_shape)
        elif not self.is_image_input:
            x = x.view(x.size(0), -1)
        h = self.encoder(x)
        loc = F.normalize(self.fc_mu(h), p=2, dim=1)
        scale = (
            F.softplus(self.fc_scale(h)).squeeze(-1)
            + torch.finfo(h.dtype).eps
        )
        return loc, scale

    def reparameterize(self, loc, scale):
        return PowerSpherical(loc, scale).rsample()

    def kl_divergence(self, loc, scale):
        q_z = PowerSpherical(loc, scale)
        p_z = PowerHypersphericalUniform(
            self.latent_dim,
            device=loc.device,
            dtype=loc.dtype,
        )
        return torch.distributions.kl.kl_divergence(q_z, p_z)

    def sample_prior(self, num_samples, device=None):
        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return F.normalize(z, p=2, dim=1)


def build_model_config(spec: RunSpec, device: str) -> ImageVAEConfig:
    distribution_type = "normal" if spec.model_family == "gaussian" else "spcauchy"
    config = ImageVAEConfig(
        input_dim=[1, 28, 28],
        latent_dim=spec.ambient_latent_dim,
        hidden_dims=list(spec.hidden_dims),
        distribution_type=distribution_type,
        spcauchy_kl_method=spec.spcauchy_kl_method,
        encoder_type=spec.encoder_type,
        decoder_type=spec.decoder_type,
        is_image_input=True,
        kl_weight=spec.kl_weight,
        dropout_rate=spec.dropout_rate,
        activation=spec.activation,
    )
    config.learning_rate = spec.learning_rate
    config.batch_size = spec.batch_size
    config.weight_decay = 0.0
    config.seed = spec.seed
    config.power_scale_bias_init = spec.power_scale_bias_init
    return config


def create_model(spec: RunSpec, device: str) -> torch.nn.Module:
    config = build_model_config(spec, device=device)
    if spec.model_family == "vmf":
        model = VMFVAE(config)
    elif spec.model_family == "powerspherical":
        model = PowerSphericalVAE(config)
    else:
        model = ImageVAE(config)
    return model.to(device)
