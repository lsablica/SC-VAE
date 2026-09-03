"""Vendored official Power Spherical PyTorch implementation.

Upstream: https://github.com/nicola-decao/power_spherical
Commit: 3d4619a9d6c01bc9b427533d386271a233e304cd
License: MIT, reproduced in ``benchmark/POWER_SPHERICAL_LICENSE``.

The implementation is the source supplied with the revised manuscript.
Compatibility adjustments keep constant tensors on the input device and
dtype, and evaluate the uniform-sphere normalizer from a Python integer so a
float64 distribution does not inherit PyTorch's float32 default dtype. The
scale constraint includes zero, whose Beta law and uniform-limit KL are
well-defined.
"""

from __future__ import annotations

import math

import torch
from torch.distributions.kl import register_kl


_EPS = 1e-7


class _TTransform(torch.distributions.Transform):
    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.real

    def _call(self, x):
        t = x[..., 0].unsqueeze(-1)
        v = x[..., 1:]
        return torch.cat(
            (t, v * torch.sqrt(torch.clamp(1 - t**2, _EPS))), -1
        )

    def _inverse(self, y):
        t = y[..., 0].unsqueeze(-1)
        v = y[..., 1:]
        return torch.cat(
            (t, v / torch.sqrt(torch.clamp(1 - t**2, _EPS))), -1
        )

    def log_abs_det_jacobian(self, x, y):
        t = x[..., 0]
        return ((x.shape[-1] - 3) / 2) * torch.log(
            torch.clamp(1 - t**2, _EPS)
        )


class _HouseholderRotationTransform(torch.distributions.Transform):
    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.real

    def __init__(self, loc):
        super().__init__()
        self.loc = loc
        self.e1 = torch.zeros_like(self.loc)
        self.e1[..., 0] = 1

    def _call(self, x):
        u = self.e1 - self.loc
        u = u / (u.norm(dim=-1, keepdim=True) + _EPS)
        return x - 2 * (x * u).sum(-1, keepdim=True) * u

    def _inverse(self, y):
        u = self.e1 - self.loc
        u = u / (u.norm(dim=-1, keepdim=True) + _EPS)
        return y - 2 * (y * u).sum(-1, keepdim=True) * u

    def log_abs_det_jacobian(self, x, y):
        return 0


class HypersphericalUniform(torch.distributions.Distribution):
    arg_constraints = {
        "dim": torch.distributions.constraints.positive_integer,
    }

    def __init__(
        self,
        dim,
        device="cpu",
        dtype=torch.float32,
        validate_args=None,
    ):
        self._dim_value = (
            int(dim.detach().cpu().item())
            if isinstance(dim, torch.Tensor)
            else int(dim)
        )
        self.dim = (
            dim
            if isinstance(dim, torch.Tensor)
            else torch.tensor(self._dim_value, device=device)
        )
        super().__init__(validate_args=validate_args)
        self.device, self.dtype = device, dtype

    def rsample(self, sample_shape=()):
        v = torch.empty(
            sample_shape + (self._dim_value,),
            device=self.device,
            dtype=self.dtype,
        ).normal_()
        return v / (v.norm(dim=-1, keepdim=True) + _EPS)

    def log_prob(self, value):
        half_dimension = self._dim_value / 2.0
        return torch.full_like(
            value[..., 0],
            math.lgamma(half_dimension)
            - (
                math.log(2)
                + half_dimension * math.log(math.pi)
            ),
            device=self.device,
            dtype=self.dtype,
        )

    def entropy(self):
        return -self.log_prob(
            torch.empty(1, device=self.device, dtype=self.dtype)
        )

    def __repr__(self):
        return "HypersphericalUniform(dim={}, device={}, dtype={})".format(
            self.dim, self.device, self.dtype
        )


class MarginalTDistribution(torch.distributions.TransformedDistribution):
    arg_constraints = {
        "dim": torch.distributions.constraints.positive_integer,
        "scale": torch.distributions.constraints.nonnegative,
    }
    has_rsample = True

    def __init__(self, dim, scale, validate_args=None):
        self.dim = (
            dim
            if isinstance(dim, torch.Tensor)
            else torch.tensor(dim, device=scale.device)
        )
        self.scale = scale
        super().__init__(
            torch.distributions.Beta(
                (dim - 1) / 2 + scale,
                (dim - 1) / 2,
                validate_args=validate_args,
            ),
            transforms=torch.distributions.AffineTransform(loc=-1, scale=2),
        )

    def entropy(self):
        return self.base_dist.entropy() + math.log(2)

    @property
    def mean(self):
        return 2 * self.base_dist.mean - 1

    @property
    def stddev(self):
        return self.variance.sqrt()

    @property
    def variance(self):
        return 4 * self.base_dist.variance


class _JointTSDistribution(torch.distributions.Distribution):
    def __init__(self, marginal_t, marginal_s):
        super().__init__(validate_args=False)
        self.marginal_t, self.marginal_s = marginal_t, marginal_s

    def rsample(self, sample_shape=()):
        return torch.cat(
            (
                self.marginal_t.rsample(sample_shape).unsqueeze(-1),
                self.marginal_s.rsample(
                    sample_shape + self.marginal_t.scale.shape
                ),
            ),
            -1,
        )

    def log_prob(self, value):
        return self.marginal_t.log_prob(
            value[..., 0]
        ) + self.marginal_s.log_prob(value[..., 1:])

    def entropy(self):
        return self.marginal_t.entropy() + self.marginal_s.entropy()


class PowerSpherical(torch.distributions.TransformedDistribution):
    arg_constraints = {
        "loc": torch.distributions.constraints.real,
        "scale": torch.distributions.constraints.nonnegative,
    }
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = loc, scale
        super().__init__(
            _JointTSDistribution(
                MarginalTDistribution(
                    loc.shape[-1], scale, validate_args=validate_args
                ),
                HypersphericalUniform(
                    loc.shape[-1] - 1,
                    device=loc.device,
                    dtype=loc.dtype,
                    validate_args=validate_args,
                ),
            ),
            [_TTransform(), _HouseholderRotationTransform(loc)],
        )

    def log_prob(self, value):
        return self.log_normalizer() + self.scale * torch.log1p(
            (self.loc * value).sum(-1)
        )

    def log_normalizer(self):
        alpha = self.base_dist.marginal_t.base_dist.concentration1
        beta = self.base_dist.marginal_t.base_dist.concentration0
        return -(
            (alpha + beta) * math.log(2)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + beta)
            + beta * math.log(math.pi)
        )

    def entropy(self):
        alpha = self.base_dist.marginal_t.base_dist.concentration1
        beta = self.base_dist.marginal_t.base_dist.concentration0
        return -(
            self.log_normalizer()
            + self.scale
            * (
                math.log(2)
                + torch.digamma(alpha)
                - torch.digamma(alpha + beta)
            )
        )

    @property
    def mean(self):
        return self.loc * self.base_dist.marginal_t.mean

    @property
    def stddev(self):
        return self.variance.sqrt()

    @property
    def variance(self):
        alpha = self.base_dist.marginal_t.base_dist.concentration1
        beta = self.base_dist.marginal_t.base_dist.concentration0
        ratio = (alpha + beta) / (2 * beta)
        identity = torch.eye(
            self.loc.shape[-1],
            device=self.loc.device,
            dtype=self.loc.dtype,
        )
        return self.base_dist.marginal_t.variance * (
            (1 - ratio)
            * self.loc.unsqueeze(-1)
            @ self.loc.unsqueeze(-2)
            + ratio * identity
        )


@register_kl(PowerSpherical, HypersphericalUniform)
def _kl_powerspherical_uniform(p, q):
    return -p.entropy() + q.entropy()


__all__ = [
    "HypersphericalUniform",
    "MarginalTDistribution",
    "PowerSpherical",
]
