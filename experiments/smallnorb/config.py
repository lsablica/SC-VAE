"""Configuration and frozen protocol constants for the smallNORB study."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "smallnorb"
RUNS_ROOT = EXPERIMENT_ROOT / "runs"
SEARCH_ROOT = EXPERIMENT_ROOT / "search"
FINAL_ROOT = EXPERIMENT_ROOT / "final"

INTRINSIC_DIMENSION = 32
AMBIENT_DIMENSION = 33
GAP_AZIMUTH_INDICES = (16, 17, 0, 1)
OBSERVED_AZIMUTH_INDICES = tuple(range(2, 16))
TRAIN_INSTANCES = (4, 6, 7, 8)
VALIDATION_INSTANCES = (9,)
TEST_INSTANCES = (0, 1, 2, 3, 5)

MAIN_FAMILIES = (
    "spcauchy",
    "vmf_robust",
    "powerspherical",
    "gaussian_isotropic",
)
SECONDARY_FAMILIES = ("gaussian_diagonal",)
ALL_FAMILIES = MAIN_FAMILIES + SECONDARY_FAMILIES
ARCHITECTURES = ("baseline_cnn", "deep_residual_cnn")
SEEDS = (0, 1, 2, 3, 4)

EXPECTED_SPLIT_COUNTS = {
    "train": 15_120,
    "validation": 4_860,
    "validation_observed": 3_780,
    "validation_gap": 1_080,
    "test": 24_300,
    "test_observed": 18_900,
    "test_gap": 5_400,
}


@dataclass(frozen=True)
class RunConfig:
    """Complete, serializable definition of one training run."""

    family: str
    seed: int
    architecture: str = "baseline_cnn"
    stage: str = "search"
    run_name: str = "default"
    epochs: int = 100
    learning_rate: float = 2e-4
    minimum_learning_rate: float = 2e-5
    learning_rate_warmup_epochs: int = 5
    weight_decay: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    mixed_precision: bool = True
    gradient_clip_norm: float = 5.0
    beta_target: float = 0.5
    beta_zero_epochs: int = 2
    beta_warmup_epochs: int = 20
    sigma_x: float = 0.20
    concentration_learning_rate_multiplier: float = 1.0
    initial_spherical_kl: float = 0.1
    spcauchy_kl_method: str = "direct"
    spcauchy_backend: str = "auto"
    gap_azimuth_indices: tuple[int, ...] = GAP_AZIMUTH_INDICES
    train_limit: int | None = None
    validation_limit: int | None = None
    test_limit: int | None = None
    select_every_epochs: int = 1
    evaluate_ssim_every_epochs: int = 5
    data_root: str = str(DEFAULT_DATA_ROOT)
    output_root: str | None = None
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.family not in ALL_FAMILIES:
            raise ValueError(f"Unsupported posterior family: {self.family}")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture: {self.architecture}"
            )
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 < self.sigma_x:
            raise ValueError("sigma_x must be positive")
        if self.beta_target < 0.0:
            raise ValueError("beta_target must be nonnegative")
        if tuple(sorted(set(self.gap_azimuth_indices))) != tuple(
            sorted(self.gap_azimuth_indices)
        ):
            raise ValueError("gap_azimuth_indices must be unique")
        if any(not 0 <= index < 18 for index in self.gap_azimuth_indices):
            raise ValueError("azimuth indices must lie in [0, 17]")
        if self.spcauchy_kl_method != "direct":
            raise ValueError(
                "The main smallNORB protocol requires direct spherical "
                "Cauchy KL evaluation"
            )

    @property
    def intrinsic_dimension(self) -> int:
        return INTRINSIC_DIMENSION

    @property
    def ambient_dimension(self) -> int:
        return AMBIENT_DIMENSION

    @property
    def root(self) -> Path:
        if self.output_root is not None:
            return Path(self.output_root)
        if self.stage == "search":
            return SEARCH_ROOT
        return RUNS_ROOT

    @property
    def run_dir(self) -> Path:
        if self.stage == "search":
            return (
                self.root
                / self.run_name
                / self.family
                / f"seed_{self.seed}"
            )
        return (
            self.root
            / self.stage
            / self.run_name
            / self.family
            / f"seed_{self.seed}"
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gap_azimuth_indices"] = list(
            self.gap_azimuth_indices
        )
        payload["tags"] = list(self.tags)
        payload["intrinsic_dimension"] = self.intrinsic_dimension
        payload["ambient_dimension"] = self.ambient_dimension
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunConfig":
        valid = {
            field_name
            for field_name in cls.__dataclass_fields__
        }
        normalized = {
            key: value for key, value in payload.items() if key in valid
        }
        normalized["gap_azimuth_indices"] = tuple(
            int(value)
            for value in normalized.get(
                "gap_azimuth_indices", GAP_AZIMUTH_INDICES
            )
        )
        normalized["tags"] = tuple(normalized.get("tags", ()))
        serialized_data_root = Path(
            normalized.get("data_root", DEFAULT_DATA_ROOT)
        )
        if (
            serialized_data_root.is_absolute()
            and not serialized_data_root.exists()
            and serialized_data_root.parts[-2:]
            == ("data", "smallnorb")
        ):
            # Archived manifests retain the exact machine path. A clone in a
            # different location transparently falls back to its own standard
            # cache path when that archived path is unavailable.
            normalized["data_root"] = str(DEFAULT_DATA_ROOT)
        return cls(**normalized)


def beta_for_epoch(config: RunConfig, epoch: int) -> float:
    """Return the shared deterministic KL schedule."""

    if epoch <= config.beta_zero_epochs:
        return 0.0
    progress = (epoch - config.beta_zero_epochs) / max(
        config.beta_warmup_epochs, 1
    )
    return config.beta_target * min(max(progress, 0.0), 1.0)


def learning_rate_for_epoch(config: RunConfig, epoch: int) -> float:
    """Five-epoch warmup followed by cosine decay."""

    import math

    warmup = config.learning_rate_warmup_epochs
    if warmup > 0 and epoch <= warmup:
        return config.learning_rate * epoch / warmup
    decay_epochs = max(config.epochs - warmup, 1)
    progress = min(max((epoch - warmup) / decay_epochs, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (
        config.minimum_learning_rate
        + (config.learning_rate - config.minimum_learning_rate) * cosine
    )
