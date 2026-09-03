"""Numerically stable image, angular, and distribution metrics."""

from __future__ import annotations

import functools
import math

import numpy as np
import torch
import torch.nn.functional as F


@functools.lru_cache(maxsize=32)
def _gaussian_kernel_cpu(
    window_size: int,
    sigma: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = torch.arange(window_size, dtype=dtype)
    coordinates = coordinates - (window_size - 1) / 2
    kernel = torch.exp(-(coordinates.square()) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return torch.outer(kernel, kernel).view(1, 1, window_size, window_size)


def structural_similarity(
    images: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Return standard Gaussian-window SSIM for each grayscale image."""

    if images.shape != reconstructions.shape or images.ndim != 4:
        raise ValueError("SSIM expects matching NCHW image tensors")
    if images.shape[1] != 1:
        raise ValueError("The smallNORB SSIM implementation is grayscale")
    x = images.float()
    y = reconstructions.float()
    kernel = _gaussian_kernel_cpu(
        window_size, sigma, torch.float32
    ).to(x.device)
    padding = window_size // 2
    x_pad = F.pad(x, (padding,) * 4, mode="reflect")
    y_pad = F.pad(y, (padding,) * 4, mode="reflect")
    mu_x = F.conv2d(x_pad, kernel)
    mu_y = F.conv2d(y_pad, kernel)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(x_pad.square(), kernel) - mu_x_sq
    sigma_y_sq = F.conv2d(y_pad.square(), kernel) - mu_y_sq
    sigma_xy = F.conv2d(x_pad * y_pad, kernel) - mu_xy
    c1 = 0.01**2
    c2 = 0.03**2
    score = (
        (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    ) / (
        (mu_x_sq + mu_y_sq + c1)
        * (sigma_x_sq + sigma_y_sq + c2)
    ).clamp_min(torch.finfo(torch.float32).eps)
    return score.flatten(1).mean(dim=1)


def reconstruction_metric_vectors(
    images: torch.Tensor,
    reconstructions: torch.Tensor,
    sigma_x: float,
    *,
    include_ssim: bool = True,
) -> dict[str, torch.Tensor]:
    error = (images.float() - reconstructions.float()).square()
    sum_squared_error = error.flatten(1).sum(dim=1)
    mse = error.flatten(1).mean(dim=1)
    psnr = 10.0 * torch.log10(
        mse.clamp_min(torch.finfo(torch.float32).eps).reciprocal()
    )
    result = {
        "reconstruction_nll": sum_squared_error
        / (2.0 * sigma_x * sigma_x),
        "pixel_mse": mse,
        "psnr_db": psnr,
    }
    if include_ssim:
        result["ssim"] = structural_similarity(images, reconstructions)
    return result


def summarize_tensor(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().reshape(-1).to(torch.float64).cpu()
    if values.numel() == 0:
        return {"count": 0}
    quantiles = torch.quantile(
        values,
        torch.tensor(
            [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0],
            dtype=torch.float64,
        ),
    )
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q90": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "max": float(quantiles[7]),
    }


def circular_absolute_error_degrees(
    predicted_degrees: np.ndarray | torch.Tensor,
    true_degrees: np.ndarray | torch.Tensor,
) -> np.ndarray:
    predicted = np.asarray(predicted_degrees, dtype=np.float64)
    truth = np.asarray(true_degrees, dtype=np.float64)
    return np.abs((predicted - truth + 180.0) % 360.0 - 180.0)


def summarize_circular_errors(
    errors_degrees: np.ndarray,
) -> dict[str, float | int]:
    errors = np.asarray(errors_degrees, dtype=np.float64)
    return {
        "count": int(errors.size),
        "mean_absolute_error_degrees": float(errors.mean()),
        "median_absolute_error_degrees": float(np.median(errors)),
        "q90_absolute_error_degrees": float(
            np.quantile(errors, 0.9)
        ),
    }


def circular_distance_radians(
    first_degrees: np.ndarray,
    second_degrees: np.ndarray,
) -> np.ndarray:
    difference = np.deg2rad(
        np.abs(
            (np.asarray(first_degrees) - np.asarray(second_degrees) + 180)
            % 360
            - 180
        )
    )
    return difference


def slerp(
    start: torch.Tensor,
    end: torch.Tensor,
    fraction: torch.Tensor | float,
) -> torch.Tensor:
    """Stable shortest-path spherical linear interpolation."""

    start = F.normalize(start, dim=-1)
    end = F.normalize(end, dim=-1)
    dot = (start * end).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    sine = torch.sin(angle)
    t = torch.as_tensor(
        fraction, device=start.device, dtype=start.dtype
    )
    while t.ndim < start.ndim:
        t = t.unsqueeze(-1)
    linear = F.normalize((1.0 - t) * start + t * end, dim=-1)
    safe = sine.abs() > 1e-6
    spherical = (
        torch.sin((1.0 - t) * angle) / sine.clamp_min(1e-8) * start
        + torch.sin(t * angle) / sine.clamp_min(1e-8) * end
    )
    return torch.where(safe, spherical, linear)


def opposite_hemisphere_fraction(
    samples: torch.Tensor,
    locations: torch.Tensor,
) -> float:
    """Fraction of samples whose dot product with their mode is negative."""

    if samples.ndim == 2:
        samples = samples.unsqueeze(0)
    dots = (samples * locations.unsqueeze(0)).sum(dim=-1)
    return float((dots < 0.0).float().mean())


__all__ = [
    "circular_absolute_error_degrees",
    "circular_distance_radians",
    "opposite_hemisphere_fraction",
    "reconstruction_metric_vectors",
    "slerp",
    "structural_similarity",
    "summarize_circular_errors",
    "summarize_tensor",
]
