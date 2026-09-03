"""Matched posterior families used by the smallNORB experiment."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.vendor_power_spherical import (
    PowerSpherical,
)
from spherical_cauchy import SphericalCauchy, spherical_cauchy_kl
from spherical_cauchy.direct import direct_kl_diagnostics

from .config import AMBIENT_DIMENSION, INTRINSIC_DIMENSION
from .vendor_vmf_smallnorb import VonMisesFisher


@dataclass
class PosteriorParameters:
    location: torch.Tensor
    scale: torch.Tensor
    raw_scale: torch.Tensor


def log_surface_area(ambient_dimension: int) -> float:
    return (
        math.log(2.0)
        + (ambient_dimension / 2.0) * math.log(math.pi)
        - math.lgamma(ambient_dimension / 2.0)
    )


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("softplus target must be positive")
    return value + math.log(-math.expm1(-value))


def _logit(value: float) -> float:
    return math.log(value) - math.log1p(-value)


def _vmf_kl(location: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    distribution = VonMisesFisher(location, kappa)
    return -distribution.entropy() + log_surface_area(location.shape[-1])


def _power_kl(
    location: torch.Tensor, concentration: torch.Tensor
) -> torch.Tensor:
    distribution = PowerSpherical(location, concentration)
    return -distribution.entropy() + log_surface_area(location.shape[-1])


@functools.lru_cache(maxsize=32)
def solve_initial_scale(
    family: str,
    target_kl: float = 0.1,
    ambient_dimension: int = AMBIENT_DIMENSION,
) -> float:
    """Numerically match each spherical family's initial information."""

    if target_kl <= 0.0:
        return 0.0
    location = torch.zeros(
        1, ambient_dimension, dtype=torch.float64
    )
    location[:, 0] = 1.0

    def objective(value: float) -> float:
        scale = torch.tensor([[value]], dtype=torch.float64)
        if family == "spcauchy":
            result = spherical_cauchy_kl(
                scale,
                ambient_dimension,
                backend="vectorized",
            )
        elif family == "vmf_robust":
            result = _vmf_kl(location, scale)
        elif family == "powerspherical":
            result = _power_kl(location, scale.squeeze(-1))
        else:
            raise KeyError(family)
        return float(result.reshape(-1)[0])

    low = 0.0
    high = 0.5 if family == "spcauchy" else 1.0
    maximum = 0.999999 if family == "spcauchy" else 256.0
    while objective(high) < target_kl:
        high = min(high * 2.0, maximum)
        if high >= maximum:
            break
    if objective(high) < target_kl:
        raise RuntimeError(
            f"Could not bracket initial KL for {family}"
        )
    for _ in range(80):
        center = 0.5 * (low + high)
        if objective(center) < target_kl:
            low = center
        else:
            high = center
    return 0.5 * (low + high)


class Posterior(nn.Module):
    family: str
    is_spherical: bool

    def encode(self, hidden: torch.Tensor) -> PosteriorParameters:
        raise NotImplementedError

    def sample(self, parameters: PosteriorParameters) -> torch.Tensor:
        raise NotImplementedError

    def representative(
        self, parameters: PosteriorParameters
    ) -> torch.Tensor:
        raise NotImplementedError

    def kl(self, parameters: PosteriorParameters) -> torch.Tensor:
        raise NotImplementedError

    def expected_cosine(
        self, parameters: PosteriorParameters
    ) -> torch.Tensor | None:
        return None

    @property
    def concentration_head(self) -> nn.Module:
        raise NotImplementedError


class _SphericalPosterior(Posterior):
    is_spherical = True

    def __init__(self, hidden_dimension: int, initial_kl: float):
        super().__init__()
        self.location_head = nn.Linear(
            hidden_dimension, AMBIENT_DIMENSION
        )
        self.scale_head = nn.Linear(hidden_dimension, 1)
        nn.init.normal_(self.location_head.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.location_head.bias, mean=0.0, std=1e-3)
        nn.init.zeros_(self.scale_head.weight)
        self.initial_scale = solve_initial_scale(
            self.family, initial_kl, AMBIENT_DIMENSION
        )

    @property
    def concentration_head(self) -> nn.Module:
        return self.scale_head

    def _location(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(
            self.location_head(hidden).float(),
            p=2,
            dim=-1,
            eps=1e-8,
        )

    def representative(
        self, parameters: PosteriorParameters
    ) -> torch.Tensor:
        return parameters.location


class SphericalCauchyPosterior(_SphericalPosterior):
    family = "spcauchy"

    def __init__(
        self,
        hidden_dimension: int,
        initial_kl: float,
        backend: str = "auto",
    ):
        self.backend = backend
        super().__init__(hidden_dimension, initial_kl)
        with torch.no_grad():
            self.scale_head.bias.fill_(_logit(self.initial_scale))

    def encode(self, hidden: torch.Tensor) -> PosteriorParameters:
        raw = self.scale_head(hidden).float()
        rho = torch.sigmoid(raw)
        one = torch.ones((), dtype=rho.dtype, device=rho.device)
        rho = torch.minimum(
            rho, torch.nextafter(one, torch.zeros_like(one))
        )
        return PosteriorParameters(self._location(hidden), rho, raw)

    def sample(self, parameters: PosteriorParameters) -> torch.Tensor:
        return SphericalCauchy(
            parameters.location, parameters.scale
        ).rsample()

    def kl(self, parameters: PosteriorParameters) -> torch.Tensor:
        return spherical_cauchy_kl(
            parameters.scale,
            AMBIENT_DIMENSION,
            backend=self.backend,
        ).squeeze(-1)

    def term_diagnostics(self, device: torch.device) -> dict:
        probe = torch.tensor([0.5], device=device, dtype=torch.float32)
        diagnostics = direct_kl_diagnostics(
            probe, AMBIENT_DIMENSION, backend=self.backend
        )
        return {
            "retained_terms": diagnostics.retained_terms,
            "terminating_terms": diagnostics.terminating_terms,
            "is_exact_terminating": diagnostics.is_exact_terminating,
            "backend": diagnostics.backend,
            "absolute_tolerance": diagnostics.absolute_tolerance,
            "maximum_concentration": diagnostics.maximum_concentration,
        }


class RobustVMFPosterior(_SphericalPosterior):
    family = "vmf_robust"
    minimum_scale = 1e-4
    maximum_scale = 10_000.0

    def __init__(self, hidden_dimension: int, initial_kl: float):
        super().__init__(hidden_dimension, initial_kl)
        with torch.no_grad():
            self.scale_head.bias.fill_(math.log(self.initial_scale))

    def encode(self, hidden: torch.Tensor) -> PosteriorParameters:
        raw = self.scale_head(hidden).float()
        # A log-concentration head is the standard stable parameterization
        # for vMF. Softplus made d(kappa)/d(raw) approach one, which left
        # kappa near 10 under the shared budget while the other families
        # reached tens of nats. Exp preserves relative scale updates.
        log_kappa = raw.clamp(
            min=math.log(self.minimum_scale),
            max=math.log(self.maximum_scale),
        )
        kappa = torch.exp(log_kappa)
        return PosteriorParameters(self._location(hidden), kappa, raw)

    def _distribution(
        self, parameters: PosteriorParameters
    ) -> VonMisesFisher:
        return VonMisesFisher(
            parameters.location, parameters.scale
        )

    def sample(self, parameters: PosteriorParameters) -> torch.Tensor:
        return self._distribution(parameters).rsample()

    def kl(self, parameters: PosteriorParameters) -> torch.Tensor:
        return _vmf_kl(
            parameters.location, parameters.scale
        ).reshape(-1)

    def expected_cosine(
        self, parameters: PosteriorParameters
    ) -> torch.Tensor:
        return self._distribution(parameters).mean.norm(dim=-1)


class PowerSphericalPosterior(_SphericalPosterior):
    family = "powerspherical"
    minimum_scale = 1e-6
    maximum_scale = 10_000.0

    def __init__(self, hidden_dimension: int, initial_kl: float):
        super().__init__(hidden_dimension, initial_kl)
        with torch.no_grad():
            self.scale_head.bias.fill_(math.log(self.initial_scale))

    def encode(self, hidden: torch.Tensor) -> PosteriorParameters:
        raw = self.scale_head(hidden).float()
        log_concentration = raw.clamp(
            min=math.log(self.minimum_scale),
            max=math.log(self.maximum_scale),
        )
        concentration = torch.exp(log_concentration)
        return PosteriorParameters(
            self._location(hidden), concentration.squeeze(-1), raw
        )

    def _distribution(
        self, parameters: PosteriorParameters
    ) -> PowerSpherical:
        return PowerSpherical(
            parameters.location, parameters.scale
        )

    def sample(self, parameters: PosteriorParameters) -> torch.Tensor:
        return self._distribution(parameters).rsample()

    def kl(self, parameters: PosteriorParameters) -> torch.Tensor:
        return _power_kl(
            parameters.location, parameters.scale
        ).reshape(-1)

    def expected_cosine(
        self, parameters: PosteriorParameters
    ) -> torch.Tensor:
        distribution = self._distribution(parameters)
        # The vendored mean property multiplies (B, D) by (B,) and therefore
        # fails for batched locations. Its scalar marginal t mean is exactly
        # E[z dot mu], which is the diagnostic required here.
        return distribution.base_dist.marginal_t.mean.reshape(-1)


class _GaussianPosterior(Posterior):
    is_spherical = False

    def __init__(
        self,
        hidden_dimension: int,
        diagonal: bool,
    ):
        super().__init__()
        self.diagonal = diagonal
        self.location_head = nn.Linear(
            hidden_dimension, INTRINSIC_DIMENSION
        )
        scale_outputs = INTRINSIC_DIMENSION if diagonal else 1
        self.scale_head = nn.Linear(hidden_dimension, scale_outputs)
        nn.init.normal_(self.location_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.location_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        target = 1.0 - 1e-5
        nn.init.constant_(
            self.scale_head.bias, _inverse_softplus(target)
        )

    @property
    def concentration_head(self) -> nn.Module:
        return self.scale_head

    def encode(self, hidden: torch.Tensor) -> PosteriorParameters:
        location = self.location_head(hidden).float()
        raw = self.scale_head(hidden).float()
        sigma = F.softplus(raw) + 1e-5
        return PosteriorParameters(location, sigma, raw)

    def sample(self, parameters: PosteriorParameters) -> torch.Tensor:
        sample = parameters.location + parameters.scale * torch.randn_like(
            parameters.location
        )
        return F.pad(sample, (0, 1), value=0.0)

    def representative(
        self, parameters: PosteriorParameters
    ) -> torch.Tensor:
        return F.pad(parameters.location, (0, 1), value=0.0)

    def kl(self, parameters: PosteriorParameters) -> torch.Tensor:
        variance = parameters.scale.square()
        scale_term = variance - 1.0 - torch.log(variance)
        if self.diagonal:
            return 0.5 * (
                parameters.location.square() + scale_term
            ).sum(dim=-1)
        return 0.5 * (
            parameters.location.square().sum(dim=-1)
            + INTRINSIC_DIMENSION * scale_term.squeeze(-1)
        )


class IsotropicGaussianPosterior(_GaussianPosterior):
    family = "gaussian_isotropic"

    def __init__(self, hidden_dimension: int, initial_kl: float = 0.0):
        del initial_kl
        super().__init__(hidden_dimension, diagonal=False)


class DiagonalGaussianPosterior(_GaussianPosterior):
    family = "gaussian_diagonal"

    def __init__(self, hidden_dimension: int, initial_kl: float = 0.0):
        del initial_kl
        super().__init__(hidden_dimension, diagonal=True)


def build_posterior(
    family: str,
    hidden_dimension: int = 512,
    initial_kl: float = 0.1,
    spcauchy_backend: str = "auto",
) -> Posterior:
    if family == "spcauchy":
        return SphericalCauchyPosterior(
            hidden_dimension,
            initial_kl,
            backend=spcauchy_backend,
        )
    if family == "vmf_robust":
        return RobustVMFPosterior(hidden_dimension, initial_kl)
    if family == "powerspherical":
        return PowerSphericalPosterior(hidden_dimension, initial_kl)
    if family == "gaussian_isotropic":
        return IsotropicGaussianPosterior(hidden_dimension)
    if family == "gaussian_diagonal":
        return DiagonalGaussianPosterior(hidden_dimension)
    raise KeyError(family)


__all__ = [
    "DiagonalGaussianPosterior",
    "IsotropicGaussianPosterior",
    "Posterior",
    "PosteriorParameters",
    "PowerSphericalPosterior",
    "RobustVMFPosterior",
    "SphericalCauchyPosterior",
    "build_posterior",
    "log_surface_area",
    "solve_initial_scale",
]
