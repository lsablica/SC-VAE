"""Fast, stable evaluation of the Laplace-weighted KL approximation."""

from __future__ import annotations

import functools
import math
from typing import Literal

import torch

from .direct import _validate_dimension, _validate_rho
from .triton_backend import evaluate_laplace_triton, triton_is_available

LaplaceBackend = Literal["auto", "eager", "compiled", "triton"]


@functools.lru_cache(maxsize=256)
def _laplace_constants_cpu(dimension: int) -> tuple[float, float]:
    """Compute dimension-only constants once, without float32 cancellation."""

    _validate_dimension(dimension)
    multiplier = float(dimension - 1)
    value = torch.tensor(multiplier, dtype=torch.float64)
    width = torch.digamma(value) - torch.digamma(value / 2.0) - math.log(2.0)
    return multiplier, float(width)


@functools.lru_cache(maxsize=512)
def _laplace_constants_device(
    dimension: int,
    dtype: torch.dtype,
    device_type: str,
    device_index: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    multiplier, width = _laplace_constants_cpu(dimension)
    device = torch.device(device_type, device_index)
    return (
        torch.tensor(multiplier, dtype=dtype, device=device),
        torch.tensor(width, dtype=dtype, device=device),
    )


def _laplace_value_gradient(
    rho: torch.Tensor,
    multiplier: torch.Tensor,
    width: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the approximation and its analytic derivative.

    The first logarithmic term is evaluated as

    ``0.5 log(1 + z^2 / (4 (1-z)))``

    instead of as the difference of two nearly equal logarithms.  Computing
    ``1-z`` from rho also avoids rounding ``z`` to one near the boundary.
    """

    one_plus_rho = 1.0 + rho
    z = 4.0 * rho / one_plus_rho.square()
    one_minus_ratio = (1.0 - rho) / one_plus_rho
    one_minus_z = one_minus_ratio.square()
    two_minus_z = 2.0 - z
    scaled_z = z / two_minus_z

    value = multiplier * (
        0.5 * torch.log1p(z.square() / (4.0 * one_minus_z)) - width * scaled_z.square()
    )
    derivative_z = (
        multiplier
        * z
        * (1.0 / (2.0 * two_minus_z * one_minus_z) - 4.0 * width / two_minus_z.pow(3))
    )
    derivative_rho = derivative_z * 4.0 * (1.0 - rho) / one_plus_rho.pow(3)
    return value, derivative_rho


@functools.lru_cache(maxsize=1)
def _compiled_laplace_evaluator():
    if not hasattr(torch, "compile"):
        raise RuntimeError("The compiled Laplace backend requires torch.compile")
    return torch.compile(
        _laplace_value_gradient,
        fullgraph=True,
        dynamic=True,
    )


def resolve_laplace_backend(rho: torch.Tensor, backend: LaplaceBackend) -> str:
    """Resolve an automatic backend selection for diagnostics and execution."""

    if backend == "auto":
        if rho.device.type == "cuda" and triton_is_available():
            return "triton"
        if rho.device.type in {"cpu", "cuda"} and hasattr(torch, "compile"):
            return "compiled"
        return "eager"
    if backend == "triton":
        if rho.device.type != "cuda":
            raise ValueError("The Triton backend requires CUDA")
        if not triton_is_available():
            raise RuntimeError(
                "The Triton backend was requested, but Triton is not installed"
            )
    elif backend == "compiled" and not hasattr(torch, "compile"):
        raise RuntimeError("The compiled Laplace backend requires torch.compile")
    elif backend not in {"eager", "compiled"}:
        raise ValueError(f"Unknown Laplace KL backend: {backend!r}")
    return backend


def _evaluate_laplace(
    rho: torch.Tensor,
    multiplier: torch.Tensor,
    width: torch.Tensor,
    backend: LaplaceBackend,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved = resolve_laplace_backend(rho, backend)
    if resolved == "triton":
        try:
            return evaluate_laplace_triton(rho, multiplier, width)
        except (ImportError, RuntimeError):
            if backend != "auto":
                raise
            resolved = "compiled" if hasattr(torch, "compile") else "eager"
    if resolved == "compiled":
        return _compiled_laplace_evaluator()(rho, multiplier, width)
    return _laplace_value_gradient(rho, multiplier, width)


class _LaplaceKlAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rho: torch.Tensor,
        multiplier: torch.Tensor,
        width: torch.Tensor,
        backend: str,
    ) -> torch.Tensor:
        value, gradient = _evaluate_laplace(
            rho,
            multiplier,
            width,
            backend,
        )
        ctx.save_for_backward(gradient)
        return value

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (gradient,) = ctx.saved_tensors
        return grad_output * gradient, None, None, None


def kl_divergence_spcauchy_laplace(
    rho: torch.Tensor,
    dimension: int,
    *,
    backend: LaplaceBackend = "auto",
) -> torch.Tensor:
    """Evaluate the Laplace-weighted KL with an analytic custom backward."""

    _validate_dimension(dimension)
    _validate_rho(rho)
    multiplier, width = _laplace_constants_device(
        dimension,
        rho.dtype,
        rho.device.type,
        rho.device.index,
    )
    return _LaplaceKlAutograd.apply(rho, multiplier, width, backend)


__all__ = [
    "LaplaceBackend",
    "kl_divergence_spcauchy_laplace",
    "resolve_laplace_backend",
]
