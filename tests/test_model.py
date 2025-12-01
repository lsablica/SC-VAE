import torch

from src.config import SpCauchyVAEConfig
from src.model import SpCauchyVAE


def test_cnn_forward_and_loss_shapes():
    config = SpCauchyVAEConfig(
        input_dim=[1, 28, 28],
        latent_dim=4,
        hidden_dims=[32, 64],
        encoder_type="cnn",
        decoder_type="cnn",
        distribution_type="spcauchy",
    )
    model = SpCauchyVAE(config)

    x = torch.rand(2, 1, 28, 28)
    x_hat, mu, second_param = model(x)

    assert x_hat.shape == x.shape
    assert mu.shape == (2, config.latent_dim)
    assert second_param.shape[0] == 2

    loss, recon, kl = model.loss_function(x, x_hat, mu, second_param)
    assert loss.shape == torch.Size([])
    assert recon.shape == torch.Size([])
    assert kl.shape == torch.Size([])


def test_generate_samples_uses_prior_shape():
    config = SpCauchyVAEConfig(
        input_dim=[1, 28, 28],
        latent_dim=3,
        hidden_dims=[16, 32],
        encoder_type="cnn",
        decoder_type="cnn",
        distribution_type="spcauchy",
    )
    model = SpCauchyVAE(config)

    samples = model.generate_samples(num_samples=5, device="cpu")
    assert samples.shape[0] == 5
    assert samples.shape[1:] == (1, 28, 28)
