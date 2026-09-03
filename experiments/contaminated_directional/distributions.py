"""Differentiable directional distributions used only by Plan B.

The vMF sampler comes from the smallNORB-local repaired copy.  In particular,
this module never imports or changes the vMF implementation used by MNIST.
The vMF normalizer below uses a long log-space series because the target grid
extends to kappa=500, beyond the range where a short 0F1 truncation is safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from benchmark.vendor_power_spherical import PowerSpherical
from experiments.smallnorb.vendor_vmf_smallnorb import (
    VonMisesFisher,
)
from spherical_cauchy import SphericalCauchy


AMBIENT_DIMENSION = 33
LOG_SURFACE_AREA = (
    math.log(2.0)
    + 0.5 * AMBIENT_DIMENSION * math.log(math.pi)
    - math.lgamma(0.5 * AMBIENT_DIMENSION)
)
LOG_UNIFORM_DENSITY = -LOG_SURFACE_AREA
VMF_ORDER = 0.5 * AMBIENT_DIMENSION - 1.0


def log_bessel_i(
    order: float,
    value: torch.Tensor,
    terms: int = 1024,
) -> torch.Tensor:
    """Differentiable log I_order(value) via a positive log-space series.

    The fixed 1024 terms are conservative for every kappa in the benchmark.
    The function is evaluated on scalar concentrations during fitting, so the
    longer series has negligible memory cost.
    """

    value = value.clamp_min(torch.finfo(value.dtype).tiny)
    index = torch.arange(terms, device=value.device, dtype=value.dtype)
    x = value.square() / 4.0
    log_terms = (
        index * torch.log(x).unsqueeze(-1)
        - torch.lgamma(index + 1.0)
        - torch.lgamma(order + 1.0 + index)
        + math.lgamma(order + 1.0)
    )
    log_hypergeometric = torch.logsumexp(log_terms, dim=-1)
    return (
        order * torch.log(value / 2.0)
        - math.lgamma(order + 1.0)
        + log_hypergeometric
    )


def vmf_log_normalizer(kappa: torch.Tensor) -> torch.Tensor:
    return (
        VMF_ORDER * torch.log(kappa)
        - 0.5 * AMBIENT_DIMENSION * math.log(2.0 * math.pi)
        - log_bessel_i(VMF_ORDER, kappa)
    )


def vmf_log_prob(
    value: torch.Tensor,
    location: torch.Tensor,
    kappa: torch.Tensor,
) -> torch.Tensor:
    return vmf_log_normalizer(kappa) + kappa * (
        value * location
    ).sum(dim=-1)


def spcauchy_log_prob(
    value: torch.Tensor,
    location: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    dot = (value * location).sum(dim=-1)
    denominator = 1.0 + rho.square() - 2.0 * rho * dot
    return LOG_UNIFORM_DENSITY + (AMBIENT_DIMENSION - 1.0) * (
        torch.log1p(-rho.square()) - torch.log(denominator)
    )


def sample_uniform(
    count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return F.normalize(
        torch.randn(count, AMBIENT_DIMENSION, device=device, dtype=dtype),
        dim=-1,
    )


def _batched_vmf_sample(
    location: torch.Tensor,
    kappa: torch.Tensor,
    count: int,
) -> torch.Tensor:
    locations = location.reshape(1, -1).expand(count, -1)
    concentrations = kappa.reshape(1, 1).expand(count, 1)
    return VonMisesFisher(locations, concentrations).rsample()


@dataclass(frozen=True)
class TargetMixture:
    """vMF plus a uniform remote-mass floor."""

    location: torch.Tensor
    kappa: float
    epsilon: float

    def sample(self, count: int) -> torch.Tensor:
        concentration = torch.as_tensor(
            self.kappa,
            device=self.location.device,
            dtype=self.location.dtype,
        )
        vmf = _batched_vmf_sample(self.location, concentration, count)
        if self.epsilon == 0.0:
            return vmf
        uniform = sample_uniform(
            count, device=self.location.device, dtype=self.location.dtype
        )
        mask = (
            torch.rand(count, 1, device=self.location.device)
            < self.epsilon
        )
        return torch.where(mask, uniform, vmf)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        concentration = torch.as_tensor(
            self.kappa,
            device=value.device,
            dtype=value.dtype,
        )
        sharp = vmf_log_prob(
            value, self.location.to(value), concentration
        )
        if self.epsilon == 0.0:
            return sharp
        log_sharp_weight = math.log1p(-self.epsilon)
        log_uniform_weight = math.log(self.epsilon)
        return torch.logaddexp(
            sharp + log_sharp_weight,
            torch.full_like(
                sharp, LOG_UNIFORM_DENSITY + log_uniform_weight
            ),
        )


def rho_for_curvature(curvature: float) -> float:
    """Solve 2(D-1)rho/(1-rho)^2 = curvature."""

    a = float(curvature)
    b = 2.0 * (AMBIENT_DIMENSION - 1.0)
    # Stable smaller root of a*rho^2 - (2a+b)*rho + a = 0.
    return 2.0 * a / (
        2.0 * a + b + math.sqrt(b * b + 4.0 * a * b)
    )


class DirectionalFamily(torch.nn.Module):
    """One learned location and concentration with matched initialization."""

    def __init__(
        self,
        family: str,
        initial_location: torch.Tensor,
        initial_curvature: float = 10.0,
    ):
        super().__init__()
        if family not in {"spcauchy", "vmf", "powerspherical"}:
            raise KeyError(family)
        self.family = family
        self.raw_location = torch.nn.Parameter(initial_location.clone())
        if family == "spcauchy":
            initial_scale = rho_for_curvature(initial_curvature)
            raw_scale = math.log(initial_scale) - math.log1p(-initial_scale)
        elif family == "vmf":
            raw_scale = math.log(initial_curvature)
        else:
            raw_scale = math.log(2.0 * initial_curvature)
        self.raw_scale = torch.nn.Parameter(
            torch.tensor(
                raw_scale,
                device=initial_location.device,
                dtype=initial_location.dtype,
            )
        )

    @property
    def location(self) -> torch.Tensor:
        return F.normalize(self.raw_location, dim=-1)

    @property
    def concentration(self) -> torch.Tensor:
        if self.family == "spcauchy":
            return torch.sigmoid(self.raw_scale).clamp(max=1.0 - 1e-6)
        maximum = 800.0 if self.family == "vmf" else 4000.0
        return torch.exp(
            self.raw_scale.clamp(
                min=math.log(1e-5), max=math.log(maximum)
            )
        )

    @property
    def local_curvature(self) -> torch.Tensor:
        scale = self.concentration
        if self.family == "spcauchy":
            return (
                2.0
                * (AMBIENT_DIMENSION - 1.0)
                * scale
                / (1.0 - scale).square()
            )
        if self.family == "vmf":
            return scale
        return scale / 2.0

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self.family == "spcauchy":
            return spcauchy_log_prob(
                value, self.location, self.concentration
            )
        if self.family == "vmf":
            return vmf_log_prob(
                value, self.location, self.concentration
            )
        return PowerSpherical(
            self.location, self.concentration
        ).log_prob(value)

    def rsample(self, count: int) -> torch.Tensor:
        if self.family == "spcauchy":
            locations = self.location.reshape(1, -1).expand(count, -1)
            scales = self.concentration.reshape(1, 1).expand(count, 1)
            return SphericalCauchy(locations, scales).rsample()
        if self.family == "vmf":
            return _batched_vmf_sample(
                self.location, self.concentration, count
            )
        return PowerSpherical(
            self.location, self.concentration
        ).rsample((count,))


__all__ = [
    "AMBIENT_DIMENSION",
    "DirectionalFamily",
    "LOG_SURFACE_AREA",
    "LOG_UNIFORM_DENSITY",
    "TargetMixture",
    "log_bessel_i",
    "rho_for_curvature",
    "spcauchy_log_prob",
    "vmf_log_normalizer",
    "vmf_log_prob",
]
