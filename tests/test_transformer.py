import torch

from src.config import SpCauchyVAEConfig
from src.model import SpCauchyVAE


def test_transformer_forward_and_loss():
    # Minimal transformer config for SMILES-style sequence data
    cfg = SpCauchyVAEConfig(
        input_dim=12,  # not used by transformer, but required
        latent_dim=8,
        hidden_dims=[16, 12],
        encoder_type="transformer",
        decoder_type="transformer",
        distribution_type="spcauchy",
        vocab_size=7,
        embedding_dim=12,  # divisible by num_heads
        max_seq_len=12,
        pad_token_id=0,
        dropout_rate=0.1,
    )
    model = SpCauchyVAE(cfg)

    # Fake token batch with some padding
    batch_size = 2
    tokens = torch.randint(1, cfg.vocab_size, (batch_size, cfg.max_seq_len))
    tokens[0, -2:] = cfg.pad_token_id

    x_hat, mu, second_param = model(tokens)
    assert x_hat.shape == (batch_size, cfg.max_seq_len, cfg.vocab_size)
    assert mu.shape == (batch_size, cfg.latent_dim)
    assert second_param.shape[0] == batch_size

    loss, recon_loss, kl_loss = model.loss_function(tokens, x_hat, mu, second_param)
    assert torch.isfinite(loss)
    assert torch.isfinite(recon_loss)
    assert torch.isfinite(kl_loss)
