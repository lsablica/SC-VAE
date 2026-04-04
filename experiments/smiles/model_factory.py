from __future__ import annotations

from pathlib import Path

import torch

from experiments.smiles.config import ExperimentConfig, build_sequence_model_config
from src.model import SpCauchyVAE


def build_model(
    experiment: ExperimentConfig,
    *,
    vocab_size: int,
    max_seq_len: int,
    pad_token_id: int,
) -> tuple[SpCauchyVAE, object]:
    model_config = build_sequence_model_config(
        experiment=experiment,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        pad_token_id=pad_token_id,
    )
    model = SpCauchyVAE(model_config)
    model = model.to(model_config.device)
    return model, model_config


def load_model_from_checkpoint(checkpoint_path: str | Path, device: str | None = None):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["config"]
    if device is not None:
        model_config.device = device
    model = SpCauchyVAE(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(model_config.device)
    model.eval()
    return model, model_config, checkpoint
