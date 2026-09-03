"""Idiomatic PyTorch distributions on the unit hypersphere."""

from __future__ import annotations

import torch
from torch.distributions import Distribution
from torch.distributions.kl import register_kl

from .constraints import half_open_unit_interval, unit_sphere
from .functional import (
    _log_surface_area,
    mobius_transform,
    sample_uniform_sphere,
    spherical_cauchy_kl,
    spherical_cauchy_pairwise_kl,
)


class SphericalCauchy(Distribution):
    """Spherical Cauchy law parameterized by direction and concentration."""

    arg_constraints = {
        "loc": unit_sphere,
        "concentration": half_open_unit_interval,
    }
    support = unit_sphere
    has_rsample = True

    def __init__(
        self,
        loc: torch.Tensor,
        concentration: torch.Tensor | float,
        *,
        validate_args: bool | None = None,
    ) -> None:
        if not isinstance(loc, torch.Tensor):
            raise TypeError("loc must be a torch.Tensor")
        if loc.ndim < 1 or loc.shape[-1] < 2:
            raise ValueError("loc must have an ambient dimension of at least 2")
        if not torch.is_floating_point(loc):
            raise TypeError("loc must be floating point")
        concentration = torch.as_tensor(
            concentration, device=loc.device, dtype=loc.dtype
        )
        if concentration.ndim == loc.ndim and concentration.shape[-1:] == (1,):
            concentration = concentration.squeeze(-1)

        batch_shape = torch.broadcast_shapes(loc.shape[:-1], concentration.shape)
        self.loc = loc.expand(batch_shape + (loc.shape[-1],))
        self.concentration = concentration.expand(batch_shape)
        self.ambient_dim = loc.shape[-1]
        super().__init__(
            batch_shape=batch_shape,
            event_shape=torch.Size((self.ambient_dim,)),
            validate_args=validate_args,
        )

    @property
    def rho(self) -> torch.Tensor:
        return self.concentration

    @property
    def ball_parameter(self) -> torch.Tensor:
        return self.concentration.unsqueeze(-1) * self.loc

    @property
    def mode(self) -> torch.Tensor:
        return self.loc

    def expand(
        self,
        batch_shape: torch.Size,
        _instance: SphericalCauchy | None = None,
    ) -> SphericalCauchy:
        batch_shape = torch.Size(batch_shape)
        new = self._get_checked_instance(SphericalCauchy, _instance)
        SphericalCauchy.__init__(
            new,
            self.loc.expand(batch_shape + self.event_shape),
            self.concentration.expand(batch_shape),
            validate_args=False,
        )
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        sample_shape = torch.Size(sample_shape)
        base = sample_uniform_sphere(
            sample_shape,
            self.batch_shape,
            self.ambient_dim,
            self.loc.device,
            self.loc.dtype,
        )
        return mobius_transform(base, self.loc, self.concentration)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        dot = (self.loc * value).sum(dim=-1).clamp(min=-1.0, max=1.0)
        rho = self.concentration
        denominator = ((1.0 - rho).square() + 2.0 * rho * (1.0 - dot)).clamp_min(
            torch.finfo(value.dtype).tiny
        )
        multiplier = float(self.ambient_dim - 1)
        log_uniform = -_log_surface_area(
            self.ambient_dim, device=value.device, dtype=value.dtype
        )
        return (
            log_uniform
            + multiplier * torch.log1p(-rho.square())
            - multiplier * torch.log(denominator)
        )

    def entropy(self) -> torch.Tensor:
        log_area = _log_surface_area(
            self.ambient_dim,
            device=self.loc.device,
            dtype=self.loc.dtype,
        )
        return log_area - spherical_cauchy_kl(self.concentration, self.ambient_dim)


class HypersphericalUniform(Distribution):
    """Uniform surface-measure distribution on ``S^(ambient_dim-1)``."""

    arg_constraints: dict[str, object] = {}
    support = unit_sphere
    has_rsample = True

    def __init__(
        self,
        ambient_dim: int,
        batch_shape: torch.Size = torch.Size(),
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if (
            not isinstance(ambient_dim, int)
            or isinstance(ambient_dim, bool)
            or ambient_dim < 2
        ):
            raise ValueError("ambient_dim must be an integer at least 2")
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if not dtype.is_floating_point:
            raise TypeError("dtype must be floating point")
        self.ambient_dim = ambient_dim
        self._anchor = torch.empty((), device=device, dtype=dtype)
        super().__init__(
            batch_shape=torch.Size(batch_shape),
            event_shape=torch.Size((ambient_dim,)),
            validate_args=validate_args,
        )

    def expand(
        self,
        batch_shape: torch.Size,
        _instance: HypersphericalUniform | None = None,
    ) -> HypersphericalUniform:
        new = self._get_checked_instance(HypersphericalUniform, _instance)
        HypersphericalUniform.__init__(
            new,
            self.ambient_dim,
            torch.Size(batch_shape),
            device=self._anchor.device,
            dtype=self._anchor.dtype,
            validate_args=False,
        )
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return sample_uniform_sphere(
            sample_shape,
            self.batch_shape,
            self.ambient_dim,
            self._anchor.device,
            self._anchor.dtype,
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        shape = torch.broadcast_shapes(value.shape[:-1], self.batch_shape)
        return -_log_surface_area(
            self.ambient_dim, device=value.device, dtype=value.dtype
        ).expand(shape)

    def entropy(self) -> torch.Tensor:
        return _log_surface_area(
            self.ambient_dim,
            device=self._anchor.device,
            dtype=self._anchor.dtype,
        ).expand(self.batch_shape)


def _check_ambient_dimensions(
    first: SphericalCauchy | HypersphericalUniform,
    second: SphericalCauchy | HypersphericalUniform,
) -> None:
    if first.ambient_dim != second.ambient_dim:
        raise ValueError("KL operands must share the same ambient dimension")


@register_kl(SphericalCauchy, HypersphericalUniform)
def _kl_spherical_cauchy_uniform(
    first: SphericalCauchy, second: HypersphericalUniform
) -> torch.Tensor:
    _check_ambient_dimensions(first, second)
    batch_shape = torch.broadcast_shapes(first.batch_shape, second.batch_shape)
    return spherical_cauchy_kl(
        first.concentration.expand(batch_shape), first.ambient_dim
    )


@register_kl(HypersphericalUniform, SphericalCauchy)
def _kl_uniform_spherical_cauchy(
    first: HypersphericalUniform, second: SphericalCauchy
) -> torch.Tensor:
    _check_ambient_dimensions(first, second)
    batch_shape = torch.broadcast_shapes(first.batch_shape, second.batch_shape)
    zero = second.ball_parameter.new_zeros(batch_shape + second.event_shape)
    return spherical_cauchy_pairwise_kl(
        zero, second.ball_parameter.expand(batch_shape + second.event_shape)
    )


@register_kl(SphericalCauchy, SphericalCauchy)
def _kl_spherical_cauchy_spherical_cauchy(
    first: SphericalCauchy, second: SphericalCauchy
) -> torch.Tensor:
    _check_ambient_dimensions(first, second)
    batch_shape = torch.broadcast_shapes(first.batch_shape, second.batch_shape)
    event_shape = (first.ambient_dim,)
    return spherical_cauchy_pairwise_kl(
        first.ball_parameter.expand(batch_shape + event_shape),
        second.ball_parameter.expand(batch_shape + event_shape),
    )


@register_kl(HypersphericalUniform, HypersphericalUniform)
def _kl_uniform_uniform(
    first: HypersphericalUniform, second: HypersphericalUniform
) -> torch.Tensor:
    _check_ambient_dimensions(first, second)
    if first._anchor.device != second._anchor.device:
        raise ValueError("KL operands must share a device")
    if first._anchor.dtype != second._anchor.dtype:
        raise ValueError("KL operands must share a dtype")
    batch_shape = torch.broadcast_shapes(first.batch_shape, second.batch_shape)
    return first._anchor.new_zeros(batch_shape)


__all__ = ["HypersphericalUniform", "SphericalCauchy"]
