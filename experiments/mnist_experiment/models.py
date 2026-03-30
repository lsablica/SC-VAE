from __future__ import annotations

import torch
import torch.nn.functional as F

from ._bootstrap import ensure_repo_root_on_path
from .config import RunSpec

ensure_repo_root_on_path()

from benchmark.vendor_vmf import HypersphericalUniform, VonMisesFisher  # noqa: E402
from src.config import SpCauchyVAEConfig  # noqa: E402
from src.model import SpCauchyVAE  # noqa: E402


class VMFVAE(SpCauchyVAE):
    """Notebook-faithful vMF baseline sharing the existing CNN backbone."""

    def __init__(self, config: SpCauchyVAEConfig):
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


def build_model_config(spec: RunSpec, device: str) -> SpCauchyVAEConfig:
    distribution_type = "normal" if spec.model_family == "gaussian" else "spcauchy"
    config = SpCauchyVAEConfig(
        input_dim=[1, 28, 28],
        latent_dim=spec.ambient_latent_dim,
        hidden_dims=list(spec.hidden_dims),
        distribution_type=distribution_type,
        encoder_type=spec.encoder_type,
        decoder_type=spec.decoder_type,
        is_image_input=True,
        kl_weight=spec.kl_weight,
        dropout_rate=spec.dropout_rate,
        activation=spec.activation,
        num_heads=4,
        num_layers=2,
    )
    config.device = device
    config.learning_rate = spec.learning_rate
    config.batch_size = spec.batch_size
    config.weight_decay = 0.0
    config.seed = spec.seed
    if hasattr(config, "__post_init__"):
        config.__post_init__()
    return config


def create_model(spec: RunSpec, device: str) -> torch.nn.Module:
    config = build_model_config(spec, device=device)
    if spec.model_family == "vmf":
        model = VMFVAE(config)
    else:
        model = SpCauchyVAE(config)
    return model.to(device)
