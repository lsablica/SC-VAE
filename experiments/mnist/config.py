from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

BENCHMARK_PRESET = "benchmark_comparison"

DEFAULT_MODELS = ("gaussian", "vmf", "spcauchy", "powerspherical")
DEFAULT_REPORTED_DIMS = (2, 3, 5, 10, 20)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)

DEFAULT_HIDDEN_DIMS = (32, 64, 128)
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 4
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_KL_WEIGHT = 1.0
DEFAULT_ACTIVATION = "relu"
DEFAULT_ENCODER_TYPE = "cnn"
DEFAULT_DECODER_TYPE = "cnn"
DEFAULT_BENCHMARK_EPOCHS = 40
DEFAULT_SELECTION_METRIC = "eval_recon_loss"
DEFAULT_LOW_DIM_WARMUP_STEPS = 200


def learning_rate_for_dim(reported_dim: int) -> float:
    # Low-dimensional runs are more sensitive, especially the spherical models.
    # A slightly lower LR than the original default improves cross-seed stability
    # without changing the overall training recipe.
    return 3e-4 if reported_dim in (2, 3) else 1e-4


def warmup_steps_for_dim(reported_dim: int) -> int:
    # The low-dimensional benchmark runs are prone to very early optimizer-step
    # collapse, especially for spherical latents. A short warmup preserves the
    # original target LR while avoiding an aggressive first few dozen updates.
    return DEFAULT_LOW_DIM_WARMUP_STEPS if reported_dim in (2, 3) else 0


def benchmark_ambient_dim(model_family: str, reported_dim: int) -> int:
    if model_family == "gaussian":
        return reported_dim
    if model_family in {"vmf", "spcauchy", "powerspherical"}:
        return reported_dim + 1
    raise ValueError(f"Unsupported model family: {model_family}")


@dataclass(frozen=True)
class RunSpec:
    preset: str
    model_family: str
    reported_dim: int
    ambient_latent_dim: int
    seed: int
    epochs: int
    learning_rate: float
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS
    dropout_rate: float = DEFAULT_DROPOUT_RATE
    kl_weight: float = DEFAULT_KL_WEIGHT
    activation: str = DEFAULT_ACTIVATION
    encoder_type: str = DEFAULT_ENCODER_TYPE
    decoder_type: str = DEFAULT_DECODER_TYPE
    optimizer_name: str = "AdamW"
    scheduler_name: str = "ReduceLROnPlateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    warmup_steps: int = 0
    selection_metric: str = DEFAULT_SELECTION_METRIC
    spcauchy_kl_method: str = "direct"
    power_scale_bias_init: float = 0.0
    config_schema_version: int = 2
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def output_dir(self, output_root: Path) -> Path:
        if self.preset != BENCHMARK_PRESET:
            raise ValueError(f"Unsupported preset: {self.preset}")
        return (
            output_root
            / "benchmark"
            / self.model_family
            / f"dim_{self.reported_dim}"
            / f"seed_{self.seed}"
        )

    def to_dict(self) -> dict:
        return {
            "preset": self.preset,
            "model_family": self.model_family,
            "reported_dim": self.reported_dim,
            "ambient_latent_dim": self.ambient_latent_dim,
            "seed": self.seed,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "hidden_dims": list(self.hidden_dims),
            "dropout_rate": self.dropout_rate,
            "kl_weight": self.kl_weight,
            "activation": self.activation,
            "encoder_type": self.encoder_type,
            "decoder_type": self.decoder_type,
            "optimizer_name": self.optimizer_name,
            "scheduler_name": self.scheduler_name,
            "scheduler_factor": self.scheduler_factor,
            "scheduler_patience": self.scheduler_patience,
            "warmup_steps": self.warmup_steps,
            "selection_metric": self.selection_metric,
            "spcauchy_kl_method": self.spcauchy_kl_method,
            "power_scale_bias_init": self.power_scale_bias_init,
            "config_schema_version": self.config_schema_version,
            "notes": self.notes,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RunSpec":
        return cls(
            preset=payload["preset"],
            model_family=payload["model_family"],
            reported_dim=int(payload["reported_dim"]),
            ambient_latent_dim=int(payload["ambient_latent_dim"]),
            seed=int(payload["seed"]),
            epochs=int(payload["epochs"]),
            learning_rate=float(payload["learning_rate"]),
            batch_size=int(payload.get("batch_size", DEFAULT_BATCH_SIZE)),
            num_workers=int(payload.get("num_workers", 0)),
            hidden_dims=tuple(
                int(v) for v in payload.get("hidden_dims", DEFAULT_HIDDEN_DIMS)
            ),
            dropout_rate=float(payload.get("dropout_rate", DEFAULT_DROPOUT_RATE)),
            kl_weight=float(payload.get("kl_weight", DEFAULT_KL_WEIGHT)),
            activation=payload.get("activation", DEFAULT_ACTIVATION),
            encoder_type=payload.get("encoder_type", DEFAULT_ENCODER_TYPE),
            decoder_type=payload.get("decoder_type", DEFAULT_DECODER_TYPE),
            optimizer_name=payload.get("optimizer_name", "AdamW"),
            scheduler_name=payload.get("scheduler_name", "ReduceLROnPlateau"),
            scheduler_factor=float(payload.get("scheduler_factor", 0.5)),
            scheduler_patience=int(payload.get("scheduler_patience", 5)),
            warmup_steps=int(payload.get("warmup_steps", 0)),
            selection_metric=payload.get("selection_metric", DEFAULT_SELECTION_METRIC),
            spcauchy_kl_method=payload.get("spcauchy_kl_method", "direct"),
            power_scale_bias_init=float(payload.get("power_scale_bias_init", 0.0)),
            config_schema_version=int(payload.get("config_schema_version", 1)),
            notes=payload.get("notes", ""),
            tags=tuple(payload.get("tags", [])),
        )


def _normalize_strings(
    values: Iterable[str] | None, defaults: tuple[str, ...]
) -> tuple[str, ...]:
    if values is None:
        return defaults
    return tuple(values)


def _normalize_ints(
    values: Iterable[int] | None, defaults: tuple[int, ...]
) -> tuple[int, ...]:
    if values is None:
        return defaults
    return tuple(int(v) for v in values)


def build_benchmark_specs(
    model_families: Iterable[str] | None = None,
    reported_dims: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    epochs: int = DEFAULT_BENCHMARK_EPOCHS,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    models = _normalize_strings(model_families, DEFAULT_MODELS)
    dims = _normalize_ints(reported_dims, DEFAULT_REPORTED_DIMS)
    run_seeds = _normalize_ints(seeds, DEFAULT_SEEDS)

    for model_family in models:
        for reported_dim in dims:
            ambient_dim = benchmark_ambient_dim(model_family, reported_dim)
            lr = learning_rate_for_dim(reported_dim)
            warmup_steps = warmup_steps_for_dim(reported_dim)
            for seed in run_seeds:
                specs.append(
                    RunSpec(
                        preset=BENCHMARK_PRESET,
                        model_family=model_family,
                        reported_dim=reported_dim,
                        ambient_latent_dim=ambient_dim,
                        seed=seed,
                        epochs=epochs,
                        learning_rate=lr,
                        warmup_steps=warmup_steps,
                        notes="Paper benchmark comparison on MNIST.",
                        tags=("benchmark", "mnist"),
                    )
                )
    return specs


def build_specs_for_preset(
    preset: str,
    model_families: Iterable[str] | None = None,
    reported_dims: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    epochs: int | None = None,
) -> list[RunSpec]:
    if preset == BENCHMARK_PRESET:
        target_epochs = DEFAULT_BENCHMARK_EPOCHS if epochs is None else epochs
        return build_benchmark_specs(
            model_families=model_families,
            reported_dims=reported_dims,
            seeds=seeds,
            epochs=target_epochs,
        )
    raise ValueError(f"Unsupported preset: {preset}")
