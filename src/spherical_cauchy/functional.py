"""Functional API for spherical Cauchy sampling and KL evaluation."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as torch_functional

from .direct import (
    DirectBackend,
    kl_divergence_spcauchy_direct,
    kl_divergence_spcauchy_direct_fixed,
    kl_divergence_spcauchy_even_neighbor,
    kl_divergence_spcauchy_pairwise,
    spcauchy_pseudohyperbolic_distance,
)
from .laplace import LaplaceBackend, kl_divergence_spcauchy_laplace


def sample_uniform_sphere(
    sample_shape: torch.Size | Iterable[int],
    batch_shape: torch.Size | Iterable[int],
    ambient_dim: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Draw reparameterized uniform samples from ``S^(ambient_dim-1)``."""

    if (
        not isinstance(ambient_dim, int)
        or isinstance(ambient_dim, bool)
        or ambient_dim < 2
    ):
        raise ValueError("ambient_dim must be an integer at least 2")
    sample_shape = torch.Size(sample_shape)
    batch_shape = torch.Size(batch_shape)
    dtype = torch.get_default_dtype() if dtype is None else dtype
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point")
    noise = torch.randn(
        sample_shape + batch_shape + (ambient_dim,),
        device=device,
        dtype=dtype,
    )
    return torch_functional.normalize(noise, dim=-1)


def mobius_transform(
    base_sample: torch.Tensor,
    loc: torch.Tensor,
    concentration: torch.Tensor,
) -> torch.Tensor:
    """Map uniform sphere samples to spherical Cauchy samples exactly."""

    if base_sample.shape[-1] != loc.shape[-1]:
        raise ValueError("base_sample and loc must share ambient dimension")
    if base_sample.device != loc.device or base_sample.device != concentration.device:
        raise ValueError("base_sample, loc, and concentration must share a device")
    if base_sample.dtype != loc.dtype or base_sample.dtype != concentration.dtype:
        raise ValueError("base_sample, loc, and concentration must share a dtype")

    loc = torch_functional.normalize(loc, dim=-1)
    if concentration.ndim == loc.ndim and concentration.shape[-1:] == (1,):
        concentration = concentration.squeeze(-1)
    rho = concentration.unsqueeze(-1)
    inner_product = (base_sample * loc).sum(dim=-1, keepdim=True)
    numerator = base_sample + rho * loc
    denominator = 1.0 + 2.0 * rho * inner_product + rho.square()
    transformed = (1.0 - rho.square()) * (numerator / denominator) + rho * loc
    return transformed


def spherical_cauchy_kl(
    concentration: torch.Tensor,
    ambient_dim: int,
    *,
    absolute_tolerance: float | None = None,
    max_terms: int = 2_000_000,
    backend: DirectBackend = "auto",
) -> torch.Tensor:
    """Evaluate the exact direct KL to the hyperspherical uniform law."""

    return kl_divergence_spcauchy_direct(
        concentration,
        ambient_dim,
        absolute_tolerance=absolute_tolerance,
        max_terms=max_terms,
        backend=backend,
    )


def spherical_cauchy_kl_fixed(
    concentration: torch.Tensor,
    ambient_dim: int,
    *,
    maximum_concentration: float,
    value_tolerance: float,
    gradient_tolerance: float,
    max_terms: int = 2_000_000,
    backend: DirectBackend = "auto",
) -> torch.Tensor:
    """Evaluate a fixed-budget route certified on a concentration interval."""

    return kl_divergence_spcauchy_direct_fixed(
        concentration,
        ambient_dim,
        maximum_concentration=maximum_concentration,
        value_tolerance=value_tolerance,
        gradient_tolerance=gradient_tolerance,
        max_terms=max_terms,
        backend=backend,
    )


def spherical_cauchy_neighbor_kl(
    concentration: torch.Tensor,
    ambient_dim: int,
    *,
    backend: DirectBackend = "auto",
) -> torch.Tensor:
    """Use exact supported cases and the finite even-neighbor odd-D route."""

    if ambient_dim % 2 == 0 or ambient_dim in {3, 5}:
        return spherical_cauchy_kl(concentration, ambient_dim, backend=backend)
    return kl_divergence_spcauchy_even_neighbor(
        concentration, ambient_dim, backend=backend
    )


def spherical_cauchy_laplace_kl(
    concentration: torch.Tensor,
    ambient_dim: int,
    *,
    backend: LaplaceBackend = "auto",
) -> torch.Tensor:
    """Evaluate the paper's constant-cost Laplace-weighted approximation."""

    return kl_divergence_spcauchy_laplace(
        concentration,
        ambient_dim,
        backend=backend,
    )


def pseudohyperbolic_distance(
    first_ball_parameter: torch.Tensor,
    second_ball_parameter: torch.Tensor,
) -> torch.Tensor:
    """Return the stable pseudohyperbolic distance in the open unit ball."""

    _validate_pairwise_event_shapes(first_ball_parameter, second_ball_parameter)
    first, second = torch.broadcast_tensors(first_ball_parameter, second_ball_parameter)
    return spcauchy_pseudohyperbolic_distance(first, second)


def spherical_cauchy_pairwise_kl(
    first_ball_parameter: torch.Tensor,
    second_ball_parameter: torch.Tensor,
    *,
    absolute_tolerance: float | None = None,
    max_terms: int = 2_000_000,
    backend: DirectBackend = "auto",
) -> torch.Tensor:
    """Evaluate exact KL between two spherical Cauchy laws."""

    _validate_pairwise_event_shapes(first_ball_parameter, second_ball_parameter)
    first, second = torch.broadcast_tensors(first_ball_parameter, second_ball_parameter)
    return kl_divergence_spcauchy_pairwise(
        first,
        second,
        absolute_tolerance=absolute_tolerance,
        max_terms=max_terms,
        backend=backend,
    )


def _validate_pairwise_event_shapes(first: torch.Tensor, second: torch.Tensor) -> None:
    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise TypeError("ball parameters must be torch tensors")
    if first.ndim < 1 or second.ndim < 1:
        raise ValueError("ball parameters must include an event dimension")
    if first.shape[-1] != second.shape[-1]:
        raise ValueError("ball parameters must have identical event shapes")


def _log_surface_area(
    ambient_dim: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    half_dimension = torch.tensor(ambient_dim / 2.0, device=device, dtype=dtype)
    return (
        half_dimension.new_tensor(math.log(2.0))
        + half_dimension * half_dimension.new_tensor(math.log(math.pi))
        - torch.lgamma(half_dimension)
    )


__all__ = [
    "mobius_transform",
    "pseudohyperbolic_distance",
    "sample_uniform_sphere",
    "spherical_cauchy_kl",
    "spherical_cauchy_kl_fixed",
    "spherical_cauchy_laplace_kl",
    "spherical_cauchy_neighbor_kl",
    "spherical_cauchy_pairwise_kl",
]
