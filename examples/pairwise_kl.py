"""Registered pairwise KL, symmetry, distance, and gradients."""

import torch
from torch.distributions import kl_divergence

from spherical_cauchy import SphericalCauchy, pseudohyperbolic_distance

first_raw = torch.randn(6, 9, requires_grad=True)
second_raw = torch.randn(6, 9, requires_grad=True)
first_loc = torch.nn.functional.normalize(first_raw, dim=-1)
second_loc = torch.nn.functional.normalize(second_raw, dim=-1)
first_rho = torch.full((6,), 0.35, requires_grad=True)
second_rho = torch.full((6,), 0.65, requires_grad=True)

first = SphericalCauchy(first_loc, first_rho)
second = SphericalCauchy(second_loc, second_rho)
forward = kl_divergence(first, second)
reverse = kl_divergence(second, first)
distance = pseudohyperbolic_distance(first.ball_parameter, second.ball_parameter)

torch.testing.assert_close(forward, reverse)
forward.mean().backward()
print("pseudohyperbolic distance:", distance)
print("finite gradients:", torch.isfinite(first_raw.grad).all().item())
