"""Small, dependency-light experiment utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .config import REPO_ROOT


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    ensure_dir(output.parent)
    output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Iterable[str] | None = None,
) -> None:
    output = Path(path)
    ensure_dir(output.parent)
    materialized = list(rows)
    if fieldnames is not None:
        names = list(fieldnames)
    else:
        names = []
        seen = set()
        for row in materialized:
            for key in row:
                if key not in seen:
                    names.append(key)
                    seen.add(key)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        if names:
            writer.writeheader()
            writer.writerows(materialized)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def command_output(arguments: list[str]) -> str:
    result = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or result.stderr.strip()


def capture_environment() -> dict[str, Any]:
    gpu: dict[str, Any] = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu.update(
            {
                "name": properties.name,
                "compute_capability": (
                    f"{properties.major}.{properties.minor}"
                ),
                "total_memory_bytes": properties.total_memory,
                "nvidia_smi": command_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,driver_version,memory.total,"
                        "temperature.gpu,clocks.current.sm",
                        "--format=csv,noheader",
                    ]
                ),
            }
        )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": os.path.realpath(os.sys.executable),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]).splitlines(),
        "gpu": gpu,
    }


def write_environment(path: str | Path) -> dict[str, Any]:
    environment = capture_environment()
    lines = [
        f"{key}={json.dumps(value, sort_keys=True)}"
        for key, value in environment.items()
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return environment


def write_commands(path: str | Path, commands: Iterable[str]) -> None:
    output = Path(path)
    existing = (
        output.read_text(encoding="utf-8").splitlines()
        if output.exists()
        else []
    )
    merged = existing.copy()
    for command in commands:
        if command not in merged:
            merged.append(command)
    output.write_text("\n".join(merged) + "\n", encoding="utf-8")


def parameter_count(module: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
