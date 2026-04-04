"""Method registries for spCauchy KL evaluators and latent-step runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from benchmark.vendor_vmf import HypersphericalUniform, VonMisesFisher, kl_vmf_official
from benchmark.vendor_vmf_robust import (
    HypersphericalUniform as HypersphericalUniformRobust,
    VonMisesFisher as VonMisesFisherRobust,
    kl_vmf_official as kl_vmf_official_robust,
)
from src.kl import (
    kl_divergence_spcauchy,
    kl_divergence_spcauchy2,
    kl_divergence_spcauchy_approx,
    kl_divergence_spcauchy_asympt,
    kl_divergence_spcauchy_combined,
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


def _series_eval(rho: torch.Tensor, latent_dim: int, *, series_k_terms: int | None = None, **_: object) -> torch.Tensor:
    return kl_divergence_spcauchy(rho, latent_dim, k_terms=series_k_terms)


def _quadrature_eval(
    rho: torch.Tensor,
    latent_dim: int,
    *,
    quadrature_nodes: int | None = None,
    **_: object,
) -> torch.Tensor:
    return kl_divergence_spcauchy_combined(rho, latent_dim, n_nodes=quadrature_nodes)


def _asymptotic_eval(rho: torch.Tensor, latent_dim: int, **_: object) -> torch.Tensor:
    return kl_divergence_spcauchy_asympt(rho, latent_dim)


def _hybrid_eval(rho: torch.Tensor, latent_dim: int, **_: object) -> torch.Tensor:
    return kl_divergence_spcauchy_approx(rho, latent_dim, approximation="hybrid")


def _auto_eval(
    rho: torch.Tensor,
    latent_dim: int,
    *,
    quadrature_nodes: int | None = None,
    **_: object,
) -> torch.Tensor:
    return kl_divergence_spcauchy_combined(rho, latent_dim, n_nodes=quadrature_nodes)


SPCAUCHY_KL_METHODS = {
    "series": SpCauchyKlMethod("series", "spcauchy", _series_eval),
    "combined": SpCauchyKlMethod("combined", "spcauchy", _quadrature_eval),
    "quadrature": SpCauchyKlMethod("combined", "spcauchy", _quadrature_eval),
    "asymptotic_high_rho": SpCauchyKlMethod("asymptotic_high_rho", "spcauchy", _asymptotic_eval),
    "hybrid": SpCauchyKlMethod("hybrid", "spcauchy", _hybrid_eval),
    "auto": SpCauchyKlMethod("auto", "spcauchy", _auto_eval),
}


LATENT_STEP_METHODS = {
    "spcauchy_asymptotic_high_rho": LatentStepMethod("spcauchy_asymptotic_high_rho", "spcauchy"),
    "spcauchy_combined": LatentStepMethod("spcauchy_combined", "spcauchy"),
    "spcauchy_quadrature": LatentStepMethod("spcauchy_combined", "spcauchy"),
    "spcauchy_hybrid": LatentStepMethod("spcauchy_hybrid", "spcauchy"),
    "spcauchy_series": LatentStepMethod("spcauchy_series", "spcauchy"),
    "spcauchy_auto": LatentStepMethod("spcauchy_auto", "spcauchy"),
    "vmf_official": LatentStepMethod("vmf_official", "vmf"),
    "vmf_robust": LatentStepMethod("vmf_robust", "vmf"),
}


def get_spcauchy_kl_method(name: str) -> SpCauchyKlMethod:
    normalized = name.strip().lower()
    if normalized not in SPCAUCHY_KL_METHODS:
        supported = ", ".join(sorted(SPCAUCHY_KL_METHODS))
        raise KeyError(f"Unknown spCauchy KL method {name!r}. Supported values: {supported}.")
    return SPCAUCHY_KL_METHODS[normalized]


def get_latent_step_method(name: str) -> LatentStepMethod:
    normalized = name.strip().lower()
    if normalized not in LATENT_STEP_METHODS:
        supported = ", ".join(sorted(LATENT_STEP_METHODS))
        raise KeyError(f"Unknown latent-step method {name!r}. Supported values: {supported}.")
    return LATENT_STEP_METHODS[normalized]


def kl_for_spcauchy_runtime(
    method_name: str,
    rho: torch.Tensor,
    latent_dim: int,
    *,
    quadrature_nodes: int | None = None,
    series_k_terms: int | None = None,
) -> torch.Tensor:
    if method_name == "spcauchy_asymptotic_high_rho":
        return _asymptotic_eval(rho, latent_dim)
    if method_name in {"spcauchy_quadrature", "spcauchy_combined"}:
        return _quadrature_eval(rho, latent_dim, quadrature_nodes=quadrature_nodes)
    if method_name == "spcauchy_hybrid":
        return _hybrid_eval(rho, latent_dim)
    if method_name == "spcauchy_series":
        return _series_eval(rho, latent_dim, series_k_terms=series_k_terms)
    if method_name == "spcauchy_auto":
        return _auto_eval(rho, latent_dim, quadrature_nodes=quadrature_nodes)
    raise KeyError(f"Unsupported spCauchy latent-step method {method_name!r}.")


def build_vmf_distribution(
    method_name: str,
    loc: torch.Tensor,
    kappa: torch.Tensor,
) -> tuple[torch.distributions.Distribution, torch.distributions.Distribution, Callable]:
    if method_name == "vmf_official":
        vmf = VonMisesFisher(loc, kappa)
        hyu = HypersphericalUniform(loc.shape[-1] - 1, device=loc.device)
        return vmf, hyu, kl_vmf_official
    if method_name == "vmf_robust":
        vmf = VonMisesFisherRobust(loc, kappa)
        hyu = HypersphericalUniformRobust(loc.shape[-1] - 1, device=loc.device)
        return vmf, hyu, kl_vmf_official_robust
    raise KeyError(f"Unsupported vMF latent-step method {method_name!r}.")
