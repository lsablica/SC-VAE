from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Iterable

import torch

from ._bootstrap import REPO_ROOT, ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.utils import set_all_seeds  # noqa: E402


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def resolve_device(requested: str) -> str:
    return "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"


def set_global_seed(seed: int) -> None:
    set_all_seeds(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str] | tuple[str, ...] | None = None) -> None:
    csv_path = Path(path)
    ensure_dir(csv_path.parent)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def checkpoint_exists(run_dir: str | Path) -> bool:
    run_path = Path(run_dir)
    return (
        (run_path / "best_recon_checkpoint.pt").exists()
        and (run_path / "final_checkpoint.pt").exists()
        and (run_path / "history.csv").exists()
        and (run_path / "run_config.json").exists()
        and (run_path / "selection_summary.json").exists()
    )


def repo_relative_path(path: str | Path) -> str:
    path_obj = Path(path)
    resolved = path_obj.resolve() if not path_obj.is_absolute() else path_obj
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def sample_std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)
