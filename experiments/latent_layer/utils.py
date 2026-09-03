"""Shared helpers for experiment runners."""

from __future__ import annotations

import csv
import math
import os
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class OutputLayout:
    """Resolved output locations for one experiment root."""

    root: Path
    results_dir: Path
    figures_dir: Path
    summary_path: Path


def prepare_output_layout(out_dir: str | os.PathLike[str] | None = None) -> OutputLayout:
    root = Path(out_dir or Path("experiments") / "latent_layer")
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return OutputLayout(
        root=root,
        results_dir=results_dir,
        figures_dir=figures_dir,
        summary_path=root / "summary.md",
    )


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.strip().lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
    }
    if normalized not in mapping:
        supported = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported dtype {dtype_name!r}. Supported values: {supported}.")
    return mapping[normalized]


def dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float64:
        return "float64"
    if dtype == torch.float32:
        return "float32"
    return str(dtype).replace("torch.", "")


def resolve_device(device_name: str) -> torch.device:
    normalized = device_name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(normalized)


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rho_from_kappa_dim(kappa: float, latent_dim: int) -> float:
    m = float(latent_dim - 1)
    if kappa == 0:
        return 0.0
    return kappa / (
        m + kappa + math.sqrt(m * m + 2.0 * kappa * m)
    )


def safe_scalar(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    detached = value.detach().reshape(-1).cpu()
    if detached.numel() == 0:
        return None
    return float(detached[0].item())


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return value if math.isfinite(value) else None


def compute_abs_rel_error(value: float | None, reference: float | None) -> tuple[float | None, float | None]:
    if value is None or reference is None or not math.isfinite(value) or not math.isfinite(reference):
        return None, None
    abs_error = abs(value - reference)
    denom = max(abs(reference), 1e-12)
    rel_error = abs_error / denom
    return abs_error, rel_error


def summarize_timings(values: Iterable[float]) -> dict[str, float | None]:
    values = list(values)
    if not values:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "iqr": None,
            "min": None,
            "max": None,
        }

    if len(values) == 1:
        std_value = 0.0
    else:
        std_value = statistics.pstdev(values)

    return {
        "mean": float(statistics.fmean(values)),
        "std": float(std_value),
        "median": float(statistics.median(values)),
        "iqr": float(
            np.percentile(np.asarray(values), 75)
            - np.percentile(np.asarray(values), 25)
        ),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def write_csv(records: list[dict], path: str | os.PathLike[str]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        output_path.write_text("", encoding="utf-8")
        return output_path

    fieldnames = sorted({key for record in records for key in record.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    return output_path


def format_float(value: float | None, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e4 or magnitude < 1e-3:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"
