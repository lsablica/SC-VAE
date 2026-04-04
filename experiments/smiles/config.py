from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


SPECIAL_TOKENS = ("<pad>", "<sos>", "<eos>")


@dataclass
class SequenceVAEConfig:
    """Model config compatible with :class:`src.model.SpCauchyVAE`."""

    input_dim: int
    latent_dim: int
    hidden_dims: list[int]
    distribution_type: str
    vocab_size: int
    max_seq_len: int
    pad_token_id: int
    embedding_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    dropout_rate: float = 0.1
    encoder_type: str = "transformer"
    decoder_type: str = "transformer"
    is_image_input: bool = False
    kl_weight: float = 0.0
    activation: str = "relu"
    spcauchy_kl_approximation: str = "hybrid"
    spcauchy_rho_bias_init: float = 0.0
    device: str = "cpu"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    distribution_type: str
    latent_dim: int
    fairness_regime: str


MODEL_SPECS = {
    "spcauchy-128": ModelSpec(
        name="spcauchy-128",
        distribution_type="spcauchy",
        latent_dim=128,
        fairness_regime="spherical_reference",
    ),
    "gaussian-64": ModelSpec(
        name="gaussian-64",
        distribution_type="normal",
        latent_dim=64,
        fairness_regime="matched_posterior_budget",
    ),
    "gaussian-128": ModelSpec(
        name="gaussian-128",
        distribution_type="normal",
        latent_dim=128,
        fairness_regime="matched_latent_size",
    ),
}


@dataclass
class ExperimentConfig:
    dataset_name: str = "zinc250k"
    model_name: str = "spcauchy-128"
    seed: int = 0
    device: str = "cuda"
    epochs: int = 300
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta_start: float = 0.0
    beta_target: float = 0.015
    beta_zero_epochs: int = 1
    beta_warmup_epochs: int = 20
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    split_seed: int = 13
    max_smiles_length: int = 68
    data_root: str = "experiments/smiles/datasets/zinc250k/raw"
    processed_root: str = "experiments/smiles/datasets/zinc250k/processed"
    runs_root: str = "experiments/smiles/runs"
    run_id: str | None = None
    output_root: str | None = None
    decode_strategy: str = "greedy"
    num_prior_samples: int = 10_000
    interpolation_steps: int = 11
    interpolation_pool_size: int = 512
    interpolation_pairs_per_bin: int = 25
    interpolation_seed: int = 0
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None
    embedding_dim: int = 256
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    spcauchy_rho_bias_init: float = 0.0
    grad_clip_norm: float | None = 1.0

    def resolved_device(self) -> str:
        if self.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def model_spec(self) -> ModelSpec:
        if self.model_name not in MODEL_SPECS:
            raise KeyError(f"Unknown model name: {self.model_name}")
        return MODEL_SPECS[self.model_name]

    def ensure_run_id(self) -> str:
        if self.run_id is None:
            self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return self.run_id

    def resolve_output_root(self) -> Path:
        if self.output_root is not None:
            return Path(self.output_root)
        run_id = self.ensure_run_id()
        spec = self.model_spec
        return (
            Path(self.runs_root)
            / self.dataset_name
            / spec.name
            / f"latent_{spec.latent_dim}"
            / f"seed_{self.seed}"
            / run_id
        )

    def to_manifest_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        spec = self.model_spec
        payload["resolved_device"] = self.resolved_device()
        payload["latent_dim"] = spec.latent_dim
        payload["distribution_type"] = spec.distribution_type
        payload["fairness_regime"] = spec.fairness_regime
        payload["run_id"] = self.ensure_run_id()
        payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        return payload


def build_sequence_model_config(
    experiment: ExperimentConfig,
    vocab_size: int,
    max_seq_len: int,
    pad_token_id: int,
) -> SequenceVAEConfig:
    spec = experiment.model_spec
    return SequenceVAEConfig(
        input_dim=max_seq_len,
        latent_dim=spec.latent_dim,
        hidden_dims=[experiment.embedding_dim, experiment.hidden_dim],
        distribution_type=spec.distribution_type,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        pad_token_id=pad_token_id,
        embedding_dim=experiment.embedding_dim,
        num_heads=experiment.num_heads,
        num_layers=experiment.num_layers,
        dropout_rate=experiment.dropout,
        kl_weight=0.0,
        spcauchy_rho_bias_init=experiment.spcauchy_rho_bias_init,
        device=experiment.resolved_device(),
    )


def beta_for_epoch(epoch: int, cfg: ExperimentConfig) -> float:
    if epoch <= cfg.beta_zero_epochs:
        return 0.0
    ramp_epoch = epoch - cfg.beta_zero_epochs
    if cfg.beta_warmup_epochs <= 0:
        return cfg.beta_target
    if ramp_epoch <= cfg.beta_warmup_epochs:
        progress = ramp_epoch / cfg.beta_warmup_epochs
        return cfg.beta_start + (cfg.beta_target - cfg.beta_start) * progress
    return cfg.beta_target


def ensure_output_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "metrics": root / "metrics",
        "samples": root / "samples",
        "interpolation": root / "interpolation",
        "plots": root / "plots",
        "tables": root / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
