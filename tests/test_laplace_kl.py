from __future__ import annotations

import functools
import math

import mpmath as mp
import pytest
import torch

from spherical_cauchy import spherical_cauchy_laplace_kl
from spherical_cauchy.laplace import resolve_laplace_backend


@functools.lru_cache(maxsize=None)
def _mp_laplace_value_gradient(
    dimension: int,
    rho_value: float,
) -> tuple[float, float]:
    mp.mp.dps = 80
    rho = mp.mpf(rho_value)
    multiplier = mp.mpf(dimension - 1)
    width = mp.digamma(multiplier) - mp.digamma(multiplier / 2) - mp.log(2)
    z = 4 * rho / (1 + rho) ** 2
    scaled_z = z / (2 - z)
    value = multiplier * (mp.log1p(-z / 2) - width * scaled_z**2 - mp.log1p(-z) / 2)
    derivative_z = (
        multiplier * z * (1 / (2 * (2 - z) * (1 - z)) - 4 * width / (2 - z) ** 3)
    )
    gradient = derivative_z * 4 * (1 - rho) / (1 + rho) ** 3
    return float(value), float(gradient)


@pytest.mark.parametrize("dimension", [2, 3, 8, 128, 2048, 4096])
@pytest.mark.parametrize("rho_value", [0.0, 1e-4, 0.01, 0.5, 0.9, 0.99])
def test_laplace_value_and_gradient_match_high_precision(
    dimension: int,
    rho_value: float,
) -> None:
    rho = torch.tensor(rho_value, dtype=torch.float64, requires_grad=True)
    value = spherical_cauchy_laplace_kl(rho, dimension, backend="eager")
    gradient = torch.autograd.grad(value, rho)[0]
    expected_value, expected_gradient = _mp_laplace_value_gradient(
        dimension,
        rho_value,
    )
    assert math.isclose(
        float(value.detach()), expected_value, rel_tol=3e-12, abs_tol=3e-13
    )
    assert math.isclose(
        float(gradient.detach()), expected_gradient, rel_tol=3e-12, abs_tol=3e-12
    )


@pytest.mark.parametrize("backend", ["eager", "compiled"])
def test_float32_laplace_backends_are_stable(backend: str) -> None:
    rho_values = [1e-4, 0.01, 0.1, 0.5, 0.9, 0.99]
    for dimension in (8, 128, 2048, 4096):
        rho = torch.tensor(rho_values, dtype=torch.float32, requires_grad=True)
        value = spherical_cauchy_laplace_kl(rho, dimension, backend=backend)
        gradient = torch.autograd.grad(value.sum(), rho)[0]
        expected = [
            _mp_laplace_value_gradient(dimension, rho_value) for rho_value in rho_values
        ]
        expected_value = torch.tensor(
            [item[0] for item in expected], dtype=torch.float64
        )
        expected_gradient = torch.tensor(
            [item[1] for item in expected], dtype=torch.float64
        )
        torch.testing.assert_close(value.double(), expected_value, atol=2e-5, rtol=3e-5)
        torch.testing.assert_close(
            gradient.double(), expected_gradient, atol=2e-4, rtol=3e-5
        )


def test_laplace_uses_one_custom_backward_node() -> None:
    rho = torch.linspace(0.01, 0.99, 17, dtype=torch.float64, requires_grad=True)
    value = spherical_cauchy_laplace_kl(rho, 128, backend="eager")
    assert type(value.grad_fn).__name__ == "_LaplaceKlAutogradBackward"
    assert len(value.grad_fn.saved_tensors) == 1
    gradient = torch.autograd.grad(value.sum(), rho)[0]
    assert gradient.shape == rho.shape
    assert torch.isfinite(gradient).all()


def test_laplace_validates_inputs_and_backends() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        spherical_cauchy_laplace_kl(torch.tensor(0.2), 1)
    with pytest.raises(TypeError, match="floating-point"):
        spherical_cauchy_laplace_kl(torch.tensor(1), 8)
    with pytest.raises(ValueError, match="0 <= rho < 1"):
        spherical_cauchy_laplace_kl(torch.tensor(1.0), 8)
    with pytest.raises(ValueError, match="Unknown Laplace"):
        spherical_cauchy_laplace_kl(torch.tensor(0.2), 8, backend="unknown")
    with pytest.raises(ValueError, match="requires CUDA"):
        spherical_cauchy_laplace_kl(torch.tensor(0.2), 8, backend="triton")


def test_auto_backend_resolves_to_compiled_on_cpu() -> None:
    rho = torch.full((128, 1), 0.2)
    expected = "compiled" if hasattr(torch, "compile") else "eager"
    assert resolve_laplace_backend(rho, "auto") == expected
