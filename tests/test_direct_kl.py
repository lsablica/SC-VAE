from __future__ import annotations

import functools
import math

import mpmath as mp
import pytest
import torch

from spherical_cauchy import (
    spherical_cauchy_kl as kl_divergence_spcauchy_direct,
)
from spherical_cauchy import (
    spherical_cauchy_kl_fixed as kl_divergence_spcauchy_direct_fixed,
)
from spherical_cauchy import (
    spherical_cauchy_neighbor_kl as kl_divergence_spcauchy_even_neighbor,
)
from spherical_cauchy.direct import (
    direct_kl_diagnostics,
    kl_divergence_spcauchy_direct_with_gradient,
)


@functools.lru_cache(maxsize=None)
def _mp_value_gradient(dimension: int, rho_value: float) -> tuple[float, float]:
    mp.mp.dps = 80
    rho = mp.mpf(rho_value)
    if rho == 0:
        return 0.0, 0.0
    half = mp.mpf(dimension) / 2
    coefficient = (1 - half) / half
    value_sum = mp.mpf("0")
    gradient_sum = mp.mpf("0")
    index = 1
    while index < 2_000_000:
        value_term = coefficient * rho ** (2 * index) / index
        gradient_term = coefficient * rho ** (2 * index - 1)
        value_sum += value_term
        gradient_sum += gradient_term
        if (
            abs(value_term) < mp.mpf("1e-70")
            and abs(gradient_term) < mp.mpf("1e-70")
            and index >= max(2, dimension // 2)
        ):
            break
        coefficient *= (index + 1 - half) / (index + half)
        index += 1
        if coefficient == 0:
            break
    multiplier = dimension - 1
    value = multiplier * (-mp.log1p(-rho * rho) - value_sum)
    gradient = 2 * multiplier * (rho / (1 - rho * rho) - gradient_sum)
    return float(value), float(gradient)


@functools.lru_cache(maxsize=None)
def _mp_compact_integral(dimension: int, rho_value: float) -> float:
    """Independent high-precision compact-integral representation."""

    mp.mp.dps = 80
    rho = mp.mpf(rho_value)
    if rho == 0:
        return 0.0
    intrinsic_dimension = dimension - 1
    half_intrinsic = mp.mpf(intrinsic_dimension) / 2
    transformed = 4 * rho / (1 + rho) ** 2

    def integrand(value):
        if value == 1:
            return half_intrinsic * transformed / (1 - transformed)
        ratio = (1 - transformed) / (1 - transformed * value)
        return value ** (dimension - 2) / (1 - value) * (1 - ratio**half_intrinsic)

    integral = mp.quad(
        integrand,
        [0, mp.mpf("0.5"), mp.mpf("0.9"), mp.mpf("0.99"), 1],
    )
    return float(intrinsic_dimension * (mp.log((1 - rho) / (1 + rho)) + integral))


@pytest.mark.parametrize(
    ("dimension", "rho_value"),
    [(3, 0.2), (7, 0.95), (16, 0.6), (128, 0.9)],
)
def test_direct_matches_independent_high_precision_integral(
    dimension: int, rho_value: float
):
    rho = torch.tensor(rho_value, dtype=torch.float64)
    actual = kl_divergence_spcauchy_direct(rho, dimension, backend="vectorized")
    expected = _mp_compact_integral(dimension, rho_value)
    assert math.isclose(float(actual), expected, rel_tol=2e-12, abs_tol=2e-12)


@pytest.mark.parametrize(
    "dimension",
    [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        16,
        17,
        32,
        33,
        64,
        128,
        256,
        512,
        1024,
        2048,
    ],
)
@pytest.mark.parametrize(
    "rho_value", [0.0, 0.01, 0.1, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999]
)
def test_direct_value_and_gradient_match_high_precision(
    dimension: int, rho_value: float
):
    rho = torch.tensor(rho_value, dtype=torch.float64)
    value, gradient = kl_divergence_spcauchy_direct_with_gradient(
        rho, dimension, backend="vectorized"
    )
    reference_value, reference_gradient = _mp_value_gradient(dimension, rho_value)
    assert math.isclose(float(value), reference_value, rel_tol=8e-12, abs_tol=8e-12)
    assert math.isclose(
        float(gradient),
        reference_gradient,
        rel_tol=8e-11,
        abs_tol=8e-11,
    )


@pytest.mark.parametrize(
    "dimension",
    [2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 32, 33, 64, 128, 256, 512, 1024, 2048],
)
@pytest.mark.parametrize(
    "rho_value", [0.0, 0.01, 0.1, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999]
)
def test_float32_value_and_gradient_match_high_precision(
    dimension: int, rho_value: float
):
    rho = torch.tensor(rho_value, dtype=torch.float32)
    value, gradient = kl_divergence_spcauchy_direct_with_gradient(
        rho, dimension, backend="vectorized"
    )
    reference_value, reference_gradient = _mp_value_gradient(dimension, rho_value)
    assert math.isclose(float(value), reference_value, rel_tol=1e-5, abs_tol=3e-6)
    assert math.isclose(
        float(gradient),
        reference_gradient,
        rel_tol=4e-5,
        abs_tol=3e-6,
    )


def test_direct_preserves_shape_and_zero():
    for shape in [(), (4,), (2, 3), (2, 3, 1)]:
        rho = torch.zeros(shape, dtype=torch.float64, requires_grad=True)
        value = kl_divergence_spcauchy_direct(rho, 17)
        gradient = torch.autograd.grad(value.sum(), rho)[0]
        assert value.shape == rho.shape
        assert gradient.shape == rho.shape
        assert torch.equal(value, torch.zeros_like(value))
        assert torch.equal(gradient, torch.zeros_like(gradient))


def test_direct_input_validation():
    with pytest.raises(TypeError):
        kl_divergence_spcauchy_direct(torch.tensor([0], dtype=torch.int64), 8)
    with pytest.raises(ValueError):
        kl_divergence_spcauchy_direct(torch.tensor([-0.1]), 8)
    with pytest.raises(ValueError):
        kl_divergence_spcauchy_direct(torch.tensor([1.0]), 8)
    with pytest.raises(ValueError):
        kl_divergence_spcauchy_direct(torch.tensor([0.5]), 1)


def test_even_termination_and_fixed_truncation():
    rho = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    diagnostics = direct_kl_diagnostics(rho, 128)
    assert diagnostics.retained_terms == 63
    assert diagnostics.terminating_terms == 63
    exact = kl_divergence_spcauchy_direct(rho, 128)
    fixed = kl_divergence_spcauchy_direct_fixed(
        rho,
        128,
        maximum_concentration=0.8,
        value_tolerance=1e-9,
        gradient_tolerance=1e-9,
    )
    assert torch.allclose(fixed, exact, atol=1e-9, rtol=0)


def test_odd_fixed_certificate_bounds_value_and_gradient():
    for dimension in [7, 17, 33]:
        for rho_value in [0.5, 0.95, 0.999]:
            rho = torch.tensor(rho_value, dtype=torch.float64, requires_grad=True)
            tolerance = 1e-10
            diagnostics = direct_kl_diagnostics(
                rho,
                dimension,
                maximum_concentration=rho_value,
                value_tolerance=tolerance,
                gradient_tolerance=tolerance,
            )
            value = kl_divergence_spcauchy_direct_fixed(
                rho,
                dimension,
                maximum_concentration=rho_value,
                value_tolerance=tolerance,
                gradient_tolerance=tolerance,
                backend="vectorized",
            )
            gradient = torch.autograd.grad(value, rho)[0]
            reference_value, reference_gradient = _mp_value_gradient(
                dimension, rho_value
            )
            assert diagnostics.retained_terms >= (dimension - 3) // 2
            assert abs(float(value.detach()) - reference_value) <= tolerance
            assert abs(float(gradient.detach()) - reference_gradient) <= (
                tolerance + 5e-14 * abs(reference_gradient)
            )


def test_custom_backward_and_gradcheck():
    rho = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: kl_divergence_spcauchy_direct(value, 17, backend="vectorized"),
        (rho,),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-5,
    )
    value = kl_divergence_spcauchy_direct(rho, 128)
    assert value.grad_fn.__class__.__name__ == "_DirectKlAutogradBackward"
    # Only the analytic derivative is retained; the forward polynomial
    # intermediates are deliberately absent from the autograd tape.
    assert len(value.grad_fn.saved_tensors) == 1
    assert value.grad_fn.saved_tensors[0].shape == rho.shape


def test_custom_backward_collapses_termwise_autograd_graph():
    def count_nodes(tensor: torch.Tensor) -> int:
        seen = set()
        stack = [tensor.grad_fn]
        while stack:
            node = stack.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            stack.extend(parent for parent, _ in node.next_functions)
        return len(seen)

    rho = torch.tensor([0.4, 0.8], dtype=torch.float64, requires_grad=True)
    production = kl_divergence_spcauchy_direct(rho, 128)
    reference, _ = kl_divergence_spcauchy_direct_with_gradient(
        rho, 128, backend="vectorized"
    )
    assert count_nodes(production) == 2
    assert count_nodes(reference) > count_nodes(production)


def test_neighbor_identity_and_certificate():
    rho = torch.tensor([0.0, 0.1, 0.5, 0.95, 0.999], dtype=torch.float64)
    for dimension in [7, 9, 17, 33, 101]:
        neighbor = kl_divergence_spcauchy_even_neighbor(rho, dimension)
        lower = kl_divergence_spcauchy_direct(rho, dimension - 1)
        upper = kl_divergence_spcauchy_direct(rho, dimension + 1)
        exact = kl_divergence_spcauchy_direct(rho, dimension)
        assert torch.allclose(neighbor, 0.5 * (lower + upper), atol=2e-12, rtol=2e-12)
        assert torch.all((neighbor - exact).abs() <= 0.5 * (upper - lower) + 2e-12)


def test_reported_neighbor_worst_case_is_reproduced():
    from experiments.latent_layer.neighbor_validation import (
        _boundary_neighbor_error,
    )

    errors = {
        dimension: _boundary_neighbor_error(dimension) for dimension in range(7, 200, 2)
    }
    worst_dimension = max(errors, key=errors.get)
    assert worst_dimension == 7
    assert math.isclose(
        errors[worst_dimension],
        0.0010995000526756726,
        rel_tol=2e-12,
        abs_tol=2e-14,
    )


def test_finite_rule_routes_by_parity():
    rho = torch.tensor([0.2, 0.8], dtype=torch.float64)
    even = kl_divergence_spcauchy_even_neighbor(rho, 8)
    direct = kl_divergence_spcauchy_direct(rho, 8)
    assert torch.equal(even, direct)


def test_monotonicity_convexity_and_dimension_ordering():
    rho = torch.linspace(0.01, 0.99, 129, dtype=torch.float64)
    value = kl_divergence_spcauchy_direct(rho, 17)
    first = torch.diff(value)
    second = torch.diff(value, n=2)
    assert torch.all(first > 0)
    assert torch.all(second > 0)
    fixed_rho = torch.tensor(0.7, dtype=torch.float64)
    values = torch.stack(
        [
            kl_divergence_spcauchy_direct(fixed_rho, dimension)
            for dimension in range(2, 34)
        ]
    )
    assert torch.all(torch.diff(values) > 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_triton_matches_cpu():
    for dtype, atol, rtol in [
        (torch.float32, 2e-3, 5e-6),
        (torch.float64, 3e-11, 3e-12),
    ]:
        rho_cpu = torch.tensor([0.0, 0.01, 0.1, 0.5, 0.9, 0.999], dtype=dtype)
        for dimension in [7, 8, 17, 128, 2048]:
            expected, expected_gradient = kl_divergence_spcauchy_direct_with_gradient(
                rho_cpu, dimension, backend="vectorized"
            )
            rho_cuda = rho_cpu.cuda().requires_grad_(True)
            actual = kl_divergence_spcauchy_direct(
                rho_cuda, dimension, backend="triton"
            )
            gradient = torch.autograd.grad(actual.sum(), rho_cuda)[0]
            assert torch.allclose(actual.cpu(), expected, atol=atol, rtol=rtol)
            assert torch.allclose(
                gradient.cpu(),
                expected_gradient,
                atol=atol * 10,
                rtol=rtol * 10,
            )
