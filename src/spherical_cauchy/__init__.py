"""PyTorch spherical Cauchy distributions and paper-supported KL routes."""

from .distributions import HypersphericalUniform, SphericalCauchy
from .functional import (
    mobius_transform,
    pseudohyperbolic_distance,
    sample_uniform_sphere,
    spherical_cauchy_kl,
    spherical_cauchy_kl_fixed,
    spherical_cauchy_laplace_kl,
    spherical_cauchy_neighbor_kl,
    spherical_cauchy_pairwise_kl,
)

__all__ = [
    "HypersphericalUniform",
    "SphericalCauchy",
    "mobius_transform",
    "pseudohyperbolic_distance",
    "sample_uniform_sphere",
    "spherical_cauchy_kl",
    "spherical_cauchy_kl_fixed",
    "spherical_cauchy_laplace_kl",
    "spherical_cauchy_neighbor_kl",
    "spherical_cauchy_pairwise_kl",
]
