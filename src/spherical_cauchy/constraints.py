"""Constraints used by spherical probability distributions."""

from __future__ import annotations

import torch
from torch.distributions import constraints


class _UnitSphere(constraints.Constraint):
    is_discrete = False
    event_dim = 1

    def check(self, value: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(value):
            return torch.zeros(value.shape[:-1], dtype=torch.bool, device=value.device)
        finite = torch.isfinite(value).all(dim=-1)
        norm = torch.linalg.vector_norm(value, dim=-1)
        tolerance = 64.0 * torch.finfo(value.dtype).eps
        return finite & ((norm - 1.0).abs() <= tolerance)

    def __repr__(self) -> str:
        return "UnitSphere()"


class _HalfOpenUnitInterval(constraints.Constraint):
    is_discrete = False
    event_dim = 0

    def check(self, value: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(value) & (value >= 0.0) & (value < 1.0)

    def __repr__(self) -> str:
        return "HalfOpenUnitInterval(lower=0, upper=1)"


unit_sphere = _UnitSphere()
half_open_unit_interval = _HalfOpenUnitInterval()


__all__ = ["half_open_unit_interval", "unit_sphere"]
