"""Paper-supported evaluators used by the latent-layer benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from benchmark.vendor_power_spherical import (
    HypersphericalUniform as PowerHypersphericalUniform,
    PowerSpherical,
)
from benchmark.vendor_vmf import (
    HypersphericalUniform,
    VonMisesFisher,
    kl_vmf_official,
)
from benchmark.vendor_vmf_robust import (
    HypersphericalUniform as HypersphericalUniformRobust,
    VonMisesFisher as VonMisesFisherRobust,
    kl_vmf_official as kl_vmf_official_robust,
)
from spherical_cauchy import (
    spherical_cauchy_kl,
    spherical_cauchy_kl_fixed,
    spherical_cauchy_laplace_kl,
    spherical_cauchy_neighbor_kl,
)
from spherical_cauchy.direct import (
    direct_kl_diagnostics,
    kl_divergence_spcauchy_direct_with_gradient,
)


@dataclass(frozen=True)
class SpCauchyKlMethod:
    name: str
    family: str
    evaluator: Callable[..., torch.Tensor]


@dataclass(frozen=True)
class LatentStepMethod:
    name: str
    family: str


def _direct_eval(
    rho: torch.Tensor,
    ambient_dim: int,
    *,
    direct_backend: str = "auto",
    **_: object,
) -> torch.Tensor:
    return spherical_cauchy_kl(rho, ambient_dim, backend=direct_backend)


def _direct_autograd_eval(
    rho: torch.Tensor,
    ambient_dim: int,
    *,
    direct_backend: str = "vectorized",
    **_: object,
) -> torch.Tensor:
    """Benchmark-only termwise autograd reference; not a package API route."""

    if ambient_dim in {3, 5}:
        value, _ = kl_divergence_spcauchy_direct_with_gradient(
            rho, ambient_dim, backend=direct_backend
        )
        return value
    count = direct_kl_diagnostics(
        rho.detach(), ambient_dim, backend="vectorized"
    ).retained_terms
    half = ambient_dim / 2.0
    coefficient = (1.0 - half) / half
    coefficients = []
    for index in range(1, count + 1):
        if index > 1:
            previous = index - 1
            coefficient *= (previous + 1.0 - half) / (previous + half)
        coefficients.append(coefficient / index)
    values = rho.new_tensor(coefficients)
    x = rho.square()
    polynomial = torch.zeros_like(rho)
    for index in range(count - 1, -1, -1):
        polynomial = polynomial * x + values[index]
    return (ambient_dim - 1.0) * (
        -torch.log1p(-x) - x * polynomial
    )


def _direct_fixed_eval(
    rho: torch.Tensor,
    ambient_dim: int,
    *,
    direct_backend: str = "auto",
    fixed_maximum_concentration: float | None = None,
    fixed_value_tolerance: float = 2e-6,
    fixed_gradient_tolerance: float = 2e-6,
    **_: object,
) -> torch.Tensor:
    maximum = (
        float(fixed_maximum_concentration)
        if fixed_maximum_concentration is not None
        else float(rho.detach().max())
    )
    return spherical_cauchy_kl_fixed(
        rho,
        ambient_dim,
        maximum_concentration=maximum,
        value_tolerance=fixed_value_tolerance,
        gradient_tolerance=fixed_gradient_tolerance,
        backend=direct_backend,
    )


def _neighbor_eval(
    rho: torch.Tensor,
    ambient_dim: int,
    *,
    direct_backend: str = "auto",
    **_: object,
) -> torch.Tensor:
    return spherical_cauchy_neighbor_kl(
        rho, ambient_dim, backend=direct_backend
    )


def _laplace_eval(
    rho: torch.Tensor,
    ambient_dim: int,
    *,
    laplace_backend: str = "auto",
    **_: object,
) -> torch.Tensor:
    return spherical_cauchy_laplace_kl(
        rho,
        ambient_dim,
        backend=laplace_backend,
    )


SPCAUCHY_KL_METHODS = {
    "direct": SpCauchyKlMethod("direct", "spcauchy", _direct_eval),
    "direct_fixed": SpCauchyKlMethod(
        "direct_fixed", "spcauchy", _direct_fixed_eval
    ),
    "neighbor": SpCauchyKlMethod("neighbor", "spcauchy", _neighbor_eval),
    "laplace": SpCauchyKlMethod("laplace", "spcauchy", _laplace_eval),
}


LATENT_STEP_METHODS = {
    "spcauchy_direct": LatentStepMethod("spcauchy_direct", "spcauchy"),
    "spcauchy_direct_fixed": LatentStepMethod(
        "spcauchy_direct_fixed", "spcauchy"
    ),
    "spcauchy_neighbor": LatentStepMethod("spcauchy_neighbor", "spcauchy"),
    "spcauchy_laplace": LatentStepMethod("spcauchy_laplace", "spcauchy"),
    "spcauchy_direct_autograd": LatentStepMethod(
        "spcauchy_direct_autograd", "spcauchy"
    ),
    "vmf_official": LatentStepMethod("vmf_official", "vmf"),
    "vmf_robust": LatentStepMethod("vmf_robust", "vmf"),
    "power_spherical": LatentStepMethod(
        "power_spherical", "powerspherical"
    ),
}


def get_spcauchy_kl_method(name: str) -> SpCauchyKlMethod:
    normalized = name.strip().lower()
    if normalized not in SPCAUCHY_KL_METHODS:
        supported = ", ".join(sorted(SPCAUCHY_KL_METHODS))
        raise KeyError(
            f"Unknown spherical Cauchy KL method {name!r}. "
            f"Supported values: {supported}."
        )
    return SPCAUCHY_KL_METHODS[normalized]


def get_latent_step_method(name: str) -> LatentStepMethod:
    normalized = name.strip().lower()
    if normalized not in LATENT_STEP_METHODS:
        supported = ", ".join(sorted(LATENT_STEP_METHODS))
        raise KeyError(
            f"Unknown latent-step method {name!r}. Supported values: {supported}."
        )
    return LATENT_STEP_METHODS[normalized]


def kl_for_spcauchy_runtime(
    method_name: str,
    rho: torch.Tensor,
    ambient_dim: int,
    *,
    direct_backend: str = "auto",
    laplace_backend: str = "auto",
    fixed_maximum_concentration: float | None = None,
    fixed_value_tolerance: float = 2e-6,
    fixed_gradient_tolerance: float = 2e-6,
    **_: object,
) -> torch.Tensor:
    if method_name == "spcauchy_direct_autograd":
        return _direct_autograd_eval(
            rho, ambient_dim, direct_backend="vectorized"
        )
    route = method_name.removeprefix("spcauchy_")
    method = get_spcauchy_kl_method(route)
    return method.evaluator(
        rho,
        ambient_dim,
        direct_backend=direct_backend,
        laplace_backend=laplace_backend,
        fixed_maximum_concentration=fixed_maximum_concentration,
        fixed_value_tolerance=fixed_value_tolerance,
        fixed_gradient_tolerance=fixed_gradient_tolerance,
    )


def build_power_spherical_distribution(
    loc: torch.Tensor,
    exponent: torch.Tensor,
) -> tuple[torch.distributions.Distribution, torch.distributions.Distribution]:
    distribution = PowerSpherical(loc, exponent.squeeze(-1))
    prior = PowerHypersphericalUniform(
        loc.shape[-1], device=loc.device, dtype=loc.dtype
    )
    return distribution, prior


def build_vmf_distribution(
    method_name: str,
    loc: torch.Tensor,
    kappa: torch.Tensor,
) -> tuple[
    torch.distributions.Distribution,
    torch.distributions.Distribution,
    Callable,
]:
    if method_name == "vmf_official":
        return (
            VonMisesFisher(loc, kappa),
            HypersphericalUniform(loc.shape[-1] - 1, device=loc.device),
            kl_vmf_official,
        )
    if method_name == "vmf_robust":
        return (
            VonMisesFisherRobust(loc, kappa),
            HypersphericalUniformRobust(
                loc.shape[-1] - 1, device=loc.device
            ),
            kl_vmf_official_robust,
        )
    raise KeyError(f"Unsupported vMF latent-step method {method_name!r}.")
