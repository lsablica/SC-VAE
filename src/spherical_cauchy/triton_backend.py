"""Optional fused CUDA backend, imported lazily on first CUDA use."""

from __future__ import annotations

import importlib.util
from typing import Any

import torch

_TRITON: Any | None = None
_TL: Any | None = None
_KERNEL: Any | None = None
_LAPLACE_KERNEL: Any | None = None


def triton_is_available() -> bool:
    """Return whether Triton can be imported without importing it eagerly."""

    return importlib.util.find_spec("triton") is not None


def _load_kernel() -> tuple[Any, Any]:
    global _TRITON, _TL, _KERNEL
    if _KERNEL is not None:
        return _TRITON, _KERNEL
    if not triton_is_available():
        raise RuntimeError(
            "The Triton backend was requested, but Triton is not installed"
        )

    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    _TRITON = triton
    _TL = tl

    @triton.jit
    def _spcauchy_polynomial_kernel(
        rho_ptr,
        value_coefficients_ptr,
        gradient_coefficients_ptr,
        value_ptr,
        gradient_ptr,
        n_elements: tl.constexpr,
        n_terms: tl.constexpr,
        multiplier: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        rho = tl.load(rho_ptr + offsets, mask=mask, other=0.0)
        x = rho * rho

        value_polynomial = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        gradient_polynomial = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        if rho.dtype == tl.float32:
            value_polynomial = value_polynomial.to(tl.float32)
            gradient_polynomial = gradient_polynomial.to(tl.float32)

        for reverse_index in tl.range(0, n_terms, loop_unroll_factor=1):
            coefficient_index = n_terms - 1 - reverse_index
            value_coefficient = tl.load(value_coefficients_ptr + coefficient_index)
            gradient_coefficient = tl.load(
                gradient_coefficients_ptr + coefficient_index
            )
            value_polynomial = value_polynomial * x + value_coefficient
            gradient_polynomial = gradient_polynomial * x + gradient_coefficient

        correction = x * value_polynomial
        gradient_correction = rho * gradient_polynomial
        base_value = -libdevice.log1p(-x)
        base_gradient_half = rho / (1.0 - x)
        value = multiplier * (base_value - correction)
        gradient = 2.0 * multiplier * (base_gradient_half - gradient_correction)
        tl.store(value_ptr + offsets, value, mask=mask)
        tl.store(gradient_ptr + offsets, gradient, mask=mask)

    _KERNEL = _spcauchy_polynomial_kernel
    return _TRITON, _KERNEL


def evaluate_polynomial_triton(
    rho: torch.Tensor,
    value_coefficients: torch.Tensor,
    gradient_coefficients: torch.Tensor,
    dimension: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a finite correction polynomial and analytic derivative."""

    if rho.device.type != "cuda":
        raise ValueError("The Triton backend requires a CUDA tensor")
    if rho.dtype not in {torch.float32, torch.float64}:
        raise TypeError("The Triton backend supports float32 and float64")
    if value_coefficients.numel() != gradient_coefficients.numel():
        raise ValueError("Value and gradient coefficient counts must agree")

    triton, kernel = _load_kernel()
    flat_rho = rho.contiguous().reshape(-1)
    value = torch.empty_like(flat_rho)
    gradient = torch.empty_like(flat_rho)
    n_elements = flat_rho.numel()
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    kernel[grid](
        flat_rho,
        value_coefficients.contiguous(),
        gradient_coefficients.contiguous(),
        value,
        gradient,
        n_elements=n_elements,
        n_terms=value_coefficients.numel(),
        multiplier=float(dimension - 1),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return value.reshape_as(rho), gradient.reshape_as(rho)


def _load_laplace_kernel() -> tuple[Any, Any]:
    global _TRITON, _TL, _LAPLACE_KERNEL
    if _LAPLACE_KERNEL is not None:
        return _TRITON, _LAPLACE_KERNEL
    if not triton_is_available():
        raise RuntimeError(
            "The Triton backend was requested, but Triton is not installed"
        )

    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    _TRITON = triton
    _TL = tl

    @triton.jit
    def _spcauchy_laplace_kernel(
        rho_ptr,
        multiplier_ptr,
        width_ptr,
        value_ptr,
        gradient_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        rho = tl.load(rho_ptr + offsets, mask=mask, other=0.0)
        multiplier = tl.load(multiplier_ptr)
        width = tl.load(width_ptr)

        one_plus_rho = 1.0 + rho
        z = 4.0 * rho / (one_plus_rho * one_plus_rho)
        one_minus_ratio = (1.0 - rho) / one_plus_rho
        one_minus_z = one_minus_ratio * one_minus_ratio
        two_minus_z = 2.0 - z
        scaled_z = z / two_minus_z

        value = multiplier * (
            0.5 * libdevice.log1p(z * z / (4.0 * one_minus_z))
            - width * scaled_z * scaled_z
        )
        derivative_z = (
            multiplier
            * z
            * (
                1.0 / (2.0 * two_minus_z * one_minus_z)
                - 4.0 * width / (two_minus_z * two_minus_z * two_minus_z)
            )
        )
        derivative_rho = (
            derivative_z
            * 4.0
            * (1.0 - rho)
            / (one_plus_rho * one_plus_rho * one_plus_rho)
        )
        tl.store(value_ptr + offsets, value, mask=mask)
        tl.store(gradient_ptr + offsets, derivative_rho, mask=mask)

    _LAPLACE_KERNEL = _spcauchy_laplace_kernel
    return _TRITON, _LAPLACE_KERNEL


def evaluate_laplace_triton(
    rho: torch.Tensor,
    multiplier: torch.Tensor,
    width: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the Laplace approximation and derivative in one CUDA kernel."""

    if rho.device.type != "cuda":
        raise ValueError("The Triton backend requires a CUDA tensor")
    if rho.dtype not in {torch.float32, torch.float64}:
        raise TypeError("The Triton backend supports float32 and float64")
    if multiplier.device != rho.device or width.device != rho.device:
        raise ValueError("Laplace constants and rho must share a device")
    if multiplier.dtype != rho.dtype or width.dtype != rho.dtype:
        raise TypeError("Laplace constants and rho must share a dtype")

    triton, kernel = _load_laplace_kernel()
    flat_rho = rho.contiguous().reshape(-1)
    value = torch.empty_like(flat_rho)
    gradient = torch.empty_like(flat_rho)
    block_size = 256
    kernel[(triton.cdiv(flat_rho.numel(), block_size),)](
        flat_rho,
        multiplier,
        width,
        value,
        gradient,
        flat_rho.numel(),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return value.reshape_as(rho), gradient.reshape_as(rho)


__all__ = [
    "evaluate_laplace_triton",
    "evaluate_polynomial_triton",
    "triton_is_available",
]
