"""Fit and evaluate one controlled directional approximation."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from experiments.smallnorb.utils import (
    ensure_dir,
    set_global_seed,
    write_csv,
    write_json,
)

from .distributions import (
    AMBIENT_DIMENSION,
    DirectionalFamily,
    TargetMixture,
)


@dataclass
class FitConfig:
    family: str
    objective: str
    kappa: int
    epsilon: float
    seed: int
    steps: int = 600
    batch_size: int = 8192
    evaluation_samples: int = 1_000_000
    learning_rate: float = 0.03
    initial_curvature: float = 10.0
    initial_location_angle_degrees: float = 10.0
    device: str = "cuda"


def fixed_target_location(
    device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    location = torch.zeros(
        AMBIENT_DIMENSION, device=device, dtype=dtype
    )
    location[0] = 1.0
    return location


def initial_location(
    target: torch.Tensor,
    seed: int,
    angle_degrees: float = 10.0,
) -> torch.Tensor:
    """Return a seed-specific direction at one fixed angle from the target."""

    generator = torch.Generator(device=target.device)
    generator.manual_seed(91_003 + seed)
    noise = torch.randn(
        target.shape,
        generator=generator,
        device=target.device,
        dtype=target.dtype,
    )
    noise[0] = 0.0
    tangent = torch.nn.functional.normalize(noise, dim=-1)
    angle = math.radians(angle_degrees)
    return math.cos(angle) * target + math.sin(angle) * tangent


def _quantiles(theta: torch.Tensor) -> dict[str, float]:
    values = torch.quantile(
        theta,
        torch.tensor(
            [0.5, 0.9, 0.99],
            device=theta.device,
            dtype=theta.dtype,
        ),
    )
    return {
        "theta_q50_rad": float(values[0]),
        "theta_q90_rad": float(values[1]),
        "theta_q99_rad": float(values[2]),
    }


@torch.no_grad()
def evaluate_fit(
    model: DirectionalFamily,
    target: TargetMixture,
    sample_count: int,
    seed: int,
    chunk_size: int = 10_000,
) -> dict[str, float]:
    target_log_p: list[torch.Tensor] = []
    target_log_q: list[torch.Tensor] = []
    candidate_log_p: list[torch.Tensor] = []
    candidate_log_q: list[torch.Tensor] = []
    target_theta: list[torch.Tensor] = []
    candidate_theta: list[torch.Tensor] = []
    for chunk_index, start in enumerate(
        range(0, sample_count, chunk_size)
    ):
        count = min(chunk_size, sample_count - start)
        target_seed = 70_000_000 + 10_000 * seed + chunk_index
        torch.manual_seed(target_seed)
        if target.location.is_cuda:
            torch.cuda.manual_seed_all(target_seed)
        p_sample = target.sample(count)
        candidate_seed = 80_000_000 + 10_000 * seed + chunk_index
        torch.manual_seed(candidate_seed)
        if target.location.is_cuda:
            torch.cuda.manual_seed_all(candidate_seed)
        q_sample = model.rsample(count)
        target_log_p.append(target.log_prob(p_sample).cpu())
        target_log_q.append(model.log_prob(p_sample).cpu())
        candidate_log_p.append(target.log_prob(q_sample).cpu())
        candidate_log_q.append(model.log_prob(q_sample).cpu())
        target_theta.append(
            torch.acos(
                (p_sample * target.location).sum(-1).clamp(-1.0, 1.0)
            ).cpu()
        )
        candidate_theta.append(
            torch.acos(
                (q_sample * target.location).sum(-1).clamp(-1.0, 1.0)
            ).cpu()
        )
    p_log_p = torch.cat(target_log_p)
    p_log_q = torch.cat(target_log_q)
    q_log_p = torch.cat(candidate_log_p)
    q_log_q = torch.cat(candidate_log_q)
    theta_p = torch.cat(target_theta)
    theta_q = torch.cat(candidate_theta)
    central_threshold = math.pi / 6.0
    remote_threshold = math.pi / 2.0
    target_central = float((theta_p < central_threshold).float().mean())
    fitted_central = float((theta_q < central_threshold).float().mean())
    target_remote = float((theta_p > remote_threshold).float().mean())
    fitted_remote = float((theta_q > remote_threshold).float().mean())
    angular = _quantiles(theta_q)
    angular.update(
        {
            f"target_{key}": value
            for key, value in _quantiles(theta_p).items()
        }
    )
    return {
        "forward_kl": float((p_log_p - p_log_q).mean()),
        "reverse_kl": float((q_log_q - q_log_p).mean()),
        "heldout_nll": float(-p_log_q.mean()),
        "target_entropy": float(-p_log_p.mean()),
        "tail_probability": fitted_remote,
        "target_tail_probability": target_remote,
        "tail_calibration_absolute_error": abs(
            fitted_remote - target_remote
        ),
        "central_mass": fitted_central,
        "target_central_mass": target_central,
        "central_calibration_absolute_error": abs(
            fitted_central - target_central
        ),
        "joint_mass_calibration_absolute_error": (
            abs(fitted_central - target_central)
            + abs(fitted_remote - target_remote)
        ),
        **angular,
    }


def fit_one(config: FitConfig, output_dir: Path) -> dict[str, Any]:
    if config.objective not in {"forward_kl", "reverse_kl"}:
        raise ValueError(config.objective)
    ensure_dir(output_dir)
    set_global_seed(config.seed)
    device = torch.device(config.device)
    dtype = torch.float64
    target_location = fixed_target_location(device, dtype)
    target = TargetMixture(
        target_location, config.kappa, config.epsilon
    )
    model = DirectionalFamily(
        config.family,
        initial_location(
            target_location,
            config.seed,
            config.initial_location_angle_degrees,
        ),
        config.initial_curvature,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.steps,
        eta_min=config.learning_rate * 0.01,
    )
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if config.objective == "forward_kl":
            sample = target.sample(config.batch_size)
            loss = -model.log_prob(sample).mean()
        else:
            sample = model.rsample(config.batch_size)
            loss = (
                model.log_prob(sample) - target.log_prob(sample)
            ).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Nonfinite loss at optimization step {step}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 10 == 0 or step == config.steps:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "learning_rate": float(
                        optimizer.param_groups[0]["lr"]
                    ),
                    "concentration": float(
                        model.concentration.detach()
                    ),
                    "local_curvature": float(
                        model.local_curvature.detach()
                    ),
                    "location_cosine": float(
                        (model.location * target_location).sum().detach()
                    ),
                }
            )
    elapsed = time.perf_counter() - started
    metrics = evaluate_fit(
        model, target, config.evaluation_samples, config.seed
    )
    result: dict[str, Any] = {
        **asdict(config),
        **metrics,
        "fitted_concentration": float(model.concentration.detach()),
        "fitted_local_curvature": float(
            model.local_curvature.detach()
        ),
        "target_local_curvature": float(config.kappa),
        "location_cosine": float(
            (model.location * target_location).sum().detach()
        ),
        "wall_seconds": elapsed,
        "nonfinite_count": 0,
    }
    write_json(output_dir / "config.json", asdict(config))
    write_csv(output_dir / "history.csv", history)
    write_json(output_dir / "history.json", history)
    write_json(output_dir / "evaluation_summary.json", result)
    return result


__all__ = ["FitConfig", "evaluate_fit", "fit_one"]
