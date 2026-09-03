"""Exact and paper-supported spherical Cauchy KL evaluators."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Literal

import torch

from .triton_backend import evaluate_polynomial_triton, triton_is_available

DEFAULT_FLOAT32_TOLERANCE = 2e-6
DEFAULT_FLOAT64_TOLERANCE = 2e-13
AUTO_COMPILED_MIN_TERMS = 32
DirectBackend = Literal["auto", "vectorized", "compiled", "triton"]


@dataclass(frozen=True)
class DirectKlDiagnostics:
    """Static diagnostics for one direct evaluator invocation."""

    dimension: int
    retained_terms: int
    terminating_terms: int | None
    is_exact_terminating: bool
    is_fixed_truncation: bool
    backend: str
    absolute_tolerance: float
    gradient_tolerance: float
    maximum_concentration: float


def _validate_dimension(dimension: int) -> None:
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 2:
        raise ValueError(f"latent_dim must be an integer at least 2, got {dimension!r}")


def _validate_rho(rho: torch.Tensor) -> None:
    if not isinstance(rho, torch.Tensor):
        raise TypeError("rho must be a torch.Tensor")
    if not torch.is_floating_point(rho):
        raise TypeError("rho must be a floating-point tensor")
    if rho.dtype not in {torch.float32, torch.float64}:
        raise TypeError("rho must use float32 or float64")
    valid = torch.logical_and(rho >= 0, rho < 1).all()
    if rho.device.type == "cuda":
        # Avoid a host synchronization in the production CUDA path while
        # retaining a device-side range assertion.
        torch._assert_async(valid, "rho must satisfy 0 <= rho < 1")
    elif not bool(valid):
        raise ValueError("rho must satisfy 0 <= rho < 1")


def _validate_ball_pair(a: torch.Tensor, b: torch.Tensor) -> None:
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("a and b must be torch tensors")
    if a.shape != b.shape:
        raise ValueError("a and b must have identical shapes")
    if a.ndim < 1 or a.shape[-1] < 2:
        raise ValueError("the final dimension must be at least 2")
    if a.device != b.device or a.dtype != b.dtype:
        raise ValueError("a and b must share dtype and device")
    if not torch.is_floating_point(a) or a.dtype not in {torch.float32, torch.float64}:
        raise TypeError("a and b must use float32 or float64")
    valid = torch.logical_and(
        a.square().sum(dim=-1) < 1,
        b.square().sum(dim=-1) < 1,
    ).all()
    if a.device.type == "cuda":
        torch._assert_async(valid, "a and b must lie in the open unit ball")
    elif not bool(valid):
        raise ValueError("a and b must lie in the open unit ball")


def _default_tolerance(dtype: torch.dtype) -> float:
    return (
        DEFAULT_FLOAT32_TOLERANCE
        if dtype == torch.float32
        else DEFAULT_FLOAT64_TOLERANCE
    )


def _validate_max_terms(max_terms: int) -> None:
    if not isinstance(max_terms, int) or isinstance(max_terms, bool) or max_terms <= 0:
        raise ValueError("max_terms must be a positive integer")


@functools.lru_cache(maxsize=256)
def _even_coefficients_cpu(
    dimension: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_dimension(dimension)
    if dimension % 2:
        raise ValueError("Finite exact coefficients require an even dimension")
    q = dimension // 2
    if q == 1:
        empty = torch.empty(0, dtype=torch.float64)
        return empty, empty

    half = dimension / 2.0
    coefficient = (1.0 - half) / half
    value_coefficients: list[float] = []
    gradient_coefficients: list[float] = []
    for index in range(1, q):
        if index > 1:
            previous = index - 1
            coefficient *= (previous + 1.0 - half) / (previous + half)
        value_coefficients.append(coefficient / index)
        gradient_coefficients.append(coefficient)
    return (
        torch.tensor(value_coefficients, dtype=torch.float64),
        torch.tensor(gradient_coefficients, dtype=torch.float64),
    )


@functools.lru_cache(maxsize=256)
def _neighbor_coefficients_cpu(
    dimension: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_dimension(dimension)
    if dimension < 7 or dimension % 2 == 0:
        raise ValueError(
            "The even-neighbor approximation requires an odd dimension at least 7"
        )
    lower_value, lower_gradient = _even_coefficients_cpu(dimension - 1)
    upper_value, upper_gradient = _even_coefficients_cpu(dimension + 1)
    length = upper_value.numel()
    padded_lower_value = torch.zeros(length, dtype=torch.float64)
    padded_lower_gradient = torch.zeros(length, dtype=torch.float64)
    padded_lower_value[: lower_value.numel()] = lower_value
    padded_lower_gradient[: lower_gradient.numel()] = lower_gradient
    denominator = 2.0 * (dimension - 1)
    return (
        ((dimension - 2) * padded_lower_value + dimension * upper_value) / denominator,
        ((dimension - 2) * padded_lower_gradient + dimension * upper_gradient)
        / denominator,
    )


def _coefficient_at(dimension: int, index: int, previous: float | None) -> float:
    half = dimension / 2.0
    if index == 1:
        return (1.0 - half) / half
    assert previous is not None
    previous_index = index - 1
    return previous * ((previous_index + 1.0 - half) / (previous_index + half))


@functools.lru_cache(maxsize=512)
def _certified_odd_coefficients_cpu(
    dimension: int,
    value_tolerance: float,
    gradient_tolerance: float,
    maximum_concentration: float,
    max_terms: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_dimension(dimension)
    if dimension < 7 or dimension % 2 == 0:
        raise ValueError("Certified odd coefficients require odd D >= 7")
    if not 0 <= maximum_concentration < 1:
        raise ValueError("maximum_concentration must satisfy 0 <= rho < 1")

    q = (dimension - 1) // 2
    x = maximum_concentration * maximum_concentration
    one_minus_x = max(1.0 - x, float.fromhex("0x1p-1022"))
    coefficients: list[float] = []
    coefficient: float | None = None

    for index in range(1, max_terms + 1):
        coefficient = _coefficient_at(dimension, index, coefficient)
        coefficients.append(coefficient)

        next_index = index + 1
        next_coefficient = _coefficient_at(dimension, next_index, coefficient)
        if next_index >= q:
            value_factor = min(
                1.0 / one_minus_x,
                1.0 + next_index / (2.0 * q),
            )
            gradient_factor = min(
                1.0 / one_minus_x,
                1.0 + next_index / (2.0 * q - 1.0),
            )
            next_even_power = maximum_concentration ** (2 * next_index)
            next_odd_power = (
                maximum_concentration ** (2 * next_index - 1)
                if maximum_concentration > 0
                else 0.0
            )
            value_bound = (
                (dimension - 1)
                * abs(next_coefficient / next_index)
                * next_even_power
                * value_factor
            )
            gradient_bound = (
                2.0
                * (dimension - 1)
                * abs(next_coefficient)
                * next_odd_power
                * gradient_factor
            )
            if value_bound <= value_tolerance and gradient_bound <= gradient_tolerance:
                value = [
                    coefficient_value / coefficient_index
                    for coefficient_index, coefficient_value in enumerate(
                        coefficients, start=1
                    )
                ]
                return (
                    torch.tensor(value, dtype=torch.float64),
                    torch.tensor(coefficients, dtype=torch.float64),
                )

    raise RuntimeError(
        f"Certified odd recurrence exceeded {max_terms} terms for D={dimension}"
    )


def _select_even_fixed_count(
    dimension: int,
    maximum_concentration: float,
    value_tolerance: float,
    gradient_tolerance: float,
) -> int:
    value, gradient = _even_coefficients_cpu(dimension)
    full_count = value.numel()
    if full_count == 0:
        return 0
    indices = torch.arange(1, full_count + 1, dtype=torch.float64)
    rho = torch.tensor(maximum_concentration, dtype=torch.float64)
    value_contributions = (dimension - 1) * value.abs() * rho.pow(2 * indices)
    gradient_contributions = (
        2.0 * (dimension - 1) * gradient.abs() * rho.pow(2 * indices - 1)
    )
    for count in range(full_count + 1):
        if (
            float(value_contributions[count:].sum()) <= value_tolerance
            and float(gradient_contributions[count:].sum()) <= gradient_tolerance
        ):
            return count
    return full_count


@functools.lru_cache(maxsize=512)
def _device_coefficients(
    kind: str,
    dimension: int,
    count: int,
    dtype: torch.dtype,
    device_type: str,
    device_index: int | None,
    value_tolerance: float,
    gradient_tolerance: float,
    maximum_concentration: float,
    max_terms: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kind == "even":
        value, gradient = _even_coefficients_cpu(dimension)
    elif kind == "neighbor":
        value, gradient = _neighbor_coefficients_cpu(dimension)
    elif kind == "odd":
        value, gradient = _certified_odd_coefficients_cpu(
            dimension,
            value_tolerance,
            gradient_tolerance,
            maximum_concentration,
            max_terms,
        )
    else:  # pragma: no cover - internal contract
        raise KeyError(kind)
    if count >= 0:
        value = value[:count]
        gradient = gradient[:count]
    device = torch.device(device_type, device_index)
    return (
        value.to(device=device, dtype=dtype),
        gradient.to(device=device, dtype=dtype),
    )


def _coefficients_on_input(
    rho: torch.Tensor,
    *,
    kind: str,
    dimension: int,
    count: int,
    value_tolerance: float,
    gradient_tolerance: float,
    maximum_concentration: float,
    max_terms: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _device_coefficients(
        kind,
        dimension,
        count,
        rho.dtype,
        rho.device.type,
        rho.device.index,
        value_tolerance,
        gradient_tolerance,
        maximum_concentration,
        max_terms,
    )


def _polynomial_vectorized(
    rho: torch.Tensor,
    value_coefficients: torch.Tensor,
    gradient_coefficients: torch.Tensor,
    dimension: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = rho.square()
    base_value = -torch.log1p(-x)
    base_gradient_half = rho / (1.0 - x)
    count = value_coefficients.numel()
    if count == 0:
        multiplier = float(dimension - 1)
        return multiplier * base_value, 2.0 * multiplier * base_gradient_half
    flat_x = x.reshape(-1)
    flat_rho = rho.reshape(-1)
    indices = torch.arange(1, count + 1, device=rho.device, dtype=rho.dtype)
    powers = flat_x.unsqueeze(1).pow(indices)
    gradient_powers = flat_x.unsqueeze(1).pow(indices - 1)
    correction = (powers * value_coefficients).sum(dim=1).reshape_as(rho)
    gradient_correction = (
        (flat_rho.unsqueeze(1) * gradient_powers * gradient_coefficients)
        .sum(dim=1)
        .reshape_as(rho)
    )
    multiplier = float(dimension - 1)
    return (
        multiplier * (base_value - correction),
        2.0 * multiplier * (base_gradient_half - gradient_correction),
    )


def _polynomial_horner(
    rho: torch.Tensor,
    value_coefficients: torch.Tensor,
    gradient_coefficients: torch.Tensor,
    dimension: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = rho.square()
    value_polynomial = torch.zeros_like(rho)
    gradient_polynomial = torch.zeros_like(rho)
    for index in range(value_coefficients.numel() - 1, -1, -1):
        value_polynomial = value_polynomial * x + value_coefficients[index]
        gradient_polynomial = gradient_polynomial * x + gradient_coefficients[index]
    multiplier = float(dimension - 1)
    return (
        multiplier * (-torch.log1p(-x) - x * value_polynomial),
        2.0 * multiplier * (rho / (1.0 - x) - rho * gradient_polynomial),
    )


def _polynomial_horner_neighbor(
    rho: torch.Tensor,
    value_coefficients: torch.Tensor,
    gradient_coefficients: torch.Tensor,
    dimension: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent compiler lane for finite even-neighbor polynomials."""

    x = rho.square()
    value_polynomial = torch.zeros_like(rho)
    gradient_polynomial = torch.zeros_like(rho)
    for index in range(value_coefficients.numel() - 1, -1, -1):
        value_polynomial = value_polynomial * x + value_coefficients[index]
        gradient_polynomial = gradient_polynomial * x + gradient_coefficients[index]
    multiplier = float(dimension - 1)
    return (
        multiplier * (-torch.log1p(-x) - x * value_polynomial),
        2.0 * multiplier * (rho / (1.0 - x) - rho * gradient_polynomial),
    )


@functools.lru_cache(maxsize=2)
def _compiled_horner(lane: str = "direct"):
    """Return a compiled Horner evaluator with a safe eager overflow path.

    Coefficient lengths are static in Inductor, so each length consumes one
    Dynamo specialization. ``fullgraph=False`` lets PyTorch fall back to the
    eager Horner implementation instead of raising if an unusually
    heterogeneous process exceeds its specialization budget. The production
    auto policy compiles only long polynomials and stays below that budget on
    the complete benchmark grid.
    """

    function = _polynomial_horner_neighbor if lane == "neighbor" else _polynomial_horner
    if not hasattr(torch, "compile"):
        return function
    return torch.compile(
        function,
        fullgraph=False,
        dynamic=False,
    )


def _d3_value_gradient(rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = rho.square()
    small = rho < rho.new_tensor(0.1)
    safe_rho = torch.where(small, torch.ones_like(rho), rho)
    log_ratio = 2.0 * torch.atanh(rho)
    exact_value = ((1.0 + x) / safe_rho) * log_ratio - 2.0
    exact_gradient = (1.0 - 1.0 / safe_rho.square()) * log_ratio + (
        safe_rho + 1.0 / safe_rho
    ) * 2.0 / (1.0 - x)
    series_value = torch.zeros_like(rho)
    series_gradient = torch.zeros_like(rho)
    power_even = x
    power_odd = rho
    for index in range(1, 9):
        denominator = 4.0 * index * index - 1.0
        series_value = series_value + (8.0 * index / denominator) * power_even
        series_gradient = (
            series_gradient + (16.0 * index * index / denominator) * power_odd
        )
        power_even = power_even * x
        power_odd = power_odd * x
    return (
        torch.where(small, series_value, exact_value),
        torch.where(small, series_gradient, exact_gradient),
    )


def _short_direct_series(
    rho: torch.Tensor,
    dimension: int,
    terms: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = rho.square()
    coefficient: float | None = None
    value_correction = torch.zeros_like(rho)
    gradient_correction = torch.zeros_like(rho)
    even_power = x
    odd_power = rho
    for index in range(1, terms + 1):
        coefficient = _coefficient_at(dimension, index, coefficient)
        value_correction = value_correction + (coefficient / index) * even_power
        gradient_correction = gradient_correction + coefficient * odd_power
        even_power = even_power * x
        odd_power = odd_power * x
    multiplier = float(dimension - 1)
    return (
        multiplier * (-torch.log1p(-x) - value_correction),
        2.0 * multiplier * (rho / (1.0 - x) - gradient_correction),
    )


def _d5_value_gradient(rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # The elementary D=5 expression contains several inverse powers of z.
    # A slightly wider local-series branch materially improves float32 near
    # the crossover while ten terms remain far below machine precision.
    small = rho <= rho.new_tensor(0.25)
    safe_rho = torch.where(small, torch.full_like(rho, 0.5), rho)
    one_plus_rho = 1.0 + safe_rho
    z = 4.0 * safe_rho / one_plus_rho.square()
    log_ratio_negative = torch.log1p(-safe_rho) - torch.log1p(safe_rho)
    log_one_minus_z = 2.0 * log_ratio_negative
    inverse_z = z.reciprocal()
    a = (2.0 - 3.0 * z) * inverse_z.pow(3)
    exact_value = 4.0 * (
        log_ratio_negative
        + 2.0 * inverse_z.square()
        - 2.0 * inverse_z
        - 5.0 / 6.0
        + a * log_one_minus_z
    )

    one_minus_z = ((1.0 - safe_rho) / one_plus_rho).square()
    dz_drho = 4.0 * (1.0 - safe_rho) / one_plus_rho.pow(3)
    da_dz = 6.0 * (z - 1.0) * inverse_z.pow(4)
    df_dz = (
        -4.0 * inverse_z.pow(3)
        + 2.0 * inverse_z.square()
        + da_dz * log_one_minus_z
        - a / one_minus_z
    )
    exact_gradient = 4.0 * (-2.0 / (1.0 - safe_rho.square()) + df_dz * dz_drho)
    series_value, series_gradient = _short_direct_series(rho, 5, terms=10)
    return (
        torch.where(small, series_value, exact_value),
        torch.where(small, series_gradient, exact_gradient),
    )


def _resolve_backend(
    rho: torch.Tensor,
    backend: str,
    coefficient_count: int | None = None,
) -> str:
    if backend in {"auto", "auto_neighbor"}:
        if rho.device.type == "cuda" and triton_is_available():
            return "triton"
        if rho.device.type == "cpu" and hasattr(torch, "compile"):
            return (
                ("compiled_neighbor" if backend == "auto_neighbor" else "compiled")
                if coefficient_count is not None
                and coefficient_count >= AUTO_COMPILED_MIN_TERMS
                else "vectorized"
            )
        return "vectorized"
    if backend == "triton" and rho.device.type != "cuda":
        raise ValueError("The Triton backend requires CUDA")
    return backend


def _evaluate_coefficients(
    rho: torch.Tensor,
    value_coefficients: torch.Tensor,
    gradient_coefficients: torch.Tensor,
    dimension: int,
    backend: DirectBackend,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved = _resolve_backend(
        rho,
        backend,
        value_coefficients.numel(),
    )
    if resolved == "triton":
        if value_coefficients.numel() == 0:
            return _polynomial_horner(
                rho, value_coefficients, gradient_coefficients, dimension
            )
        try:
            return evaluate_polynomial_triton(
                rho, value_coefficients, gradient_coefficients, dimension
            )
        except (ImportError, RuntimeError):
            if backend not in {"auto", "auto_neighbor"}:
                raise
            return _polynomial_vectorized(
                rho, value_coefficients, gradient_coefficients, dimension
            )
    if resolved == "vectorized":
        return _polynomial_vectorized(
            rho, value_coefficients, gradient_coefficients, dimension
        )
    if resolved in {"compiled", "compiled_neighbor"}:
        lane = "neighbor" if resolved == "compiled_neighbor" else "direct"
        return _compiled_horner(lane)(
            rho,
            value_coefficients,
            gradient_coefficients,
            dimension,
        )
    raise ValueError(f"Unknown direct KL backend: {backend!r}")


def _raw_value_gradient(
    rho: torch.Tensor,
    value_coefficients: torch.Tensor,
    gradient_coefficients: torch.Tensor,
    dimension: int,
    backend: DirectBackend,
    low_dimension: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if low_dimension == 3:
        return _d3_value_gradient(rho)
    if low_dimension == 5:
        return _d5_value_gradient(rho)
    return _evaluate_coefficients(
        rho,
        value_coefficients,
        gradient_coefficients,
        dimension,
        backend,
    )


class _DirectKlAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rho: torch.Tensor,
        value_coefficients: torch.Tensor,
        gradient_coefficients: torch.Tensor,
        dimension: int,
        backend: str,
        low_dimension: int,
    ) -> torch.Tensor:
        value, gradient = _raw_value_gradient(
            rho,
            value_coefficients,
            gradient_coefficients,
            dimension,
            backend,
            low_dimension,
        )
        ctx.save_for_backward(gradient)
        return value

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (gradient,) = ctx.saved_tensors
        return grad_output * gradient, None, None, None, None, None


def _direct_setup(
    rho: torch.Tensor,
    dimension: int,
    *,
    value_tolerance: float,
    gradient_tolerance: float,
    maximum_concentration: float,
    max_terms: int,
    fixed: bool,
) -> tuple[torch.Tensor, torch.Tensor, int, int | None, int]:
    if dimension in {3, 5}:
        return (
            rho.new_empty(0),
            rho.new_empty(0),
            0,
            None,
            dimension,
        )
    if dimension % 2 == 0:
        full_count = dimension // 2 - 1
        count = (
            _select_even_fixed_count(
                dimension,
                maximum_concentration,
                value_tolerance,
                gradient_tolerance,
            )
            if fixed
            else full_count
        )
        value, gradient = _coefficients_on_input(
            rho,
            kind="even",
            dimension=dimension,
            count=count,
            value_tolerance=value_tolerance,
            gradient_tolerance=gradient_tolerance,
            maximum_concentration=maximum_concentration,
            max_terms=max_terms,
        )
        return value, gradient, count, full_count, 0

    # Use a uniform certified count, valid for every rho no larger than the
    # selected maximum. The default maximum is the largest representable rho
    # below one, so the ordinary direct API has a static training loop bound.
    value_full, gradient_full = _certified_odd_coefficients_cpu(
        dimension,
        value_tolerance,
        gradient_tolerance,
        maximum_concentration,
        max_terms,
    )
    count = value_full.numel()
    value, gradient = _coefficients_on_input(
        rho,
        kind="odd",
        dimension=dimension,
        count=count,
        value_tolerance=value_tolerance,
        gradient_tolerance=gradient_tolerance,
        maximum_concentration=maximum_concentration,
        max_terms=max_terms,
    )
    return value, gradient, count, None, 0


def _largest_rho_below_one(dtype: torch.dtype) -> float:
    one = torch.tensor(1.0, dtype=dtype)
    zero = torch.tensor(0.0, dtype=dtype)
    return float(torch.nextafter(one, zero))


def kl_divergence_spcauchy_direct_with_gradient(
    rho: torch.Tensor,
    latent_dim: int,
    *,
    absolute_tolerance: float | None = None,
    max_terms: int = 2_000_000,
    backend: DirectBackend = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the direct KL and analytic derivative with respect to ``rho``."""

    _validate_dimension(latent_dim)
    _validate_rho(rho)
    _validate_max_terms(max_terms)
    tolerance = (
        _default_tolerance(rho.dtype)
        if absolute_tolerance is None
        else float(absolute_tolerance)
    )
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("absolute_tolerance must be positive and finite")
    maximum_concentration = _largest_rho_below_one(rho.dtype)
    value_coefficients, gradient_coefficients, _, _, low_dimension = _direct_setup(
        rho,
        latent_dim,
        value_tolerance=tolerance,
        gradient_tolerance=tolerance,
        maximum_concentration=maximum_concentration,
        max_terms=max_terms,
        fixed=False,
    )
    return _raw_value_gradient(
        rho,
        value_coefficients,
        gradient_coefficients,
        latent_dim,
        backend,
        low_dimension,
    )


def kl_divergence_spcauchy_direct(
    rho: torch.Tensor,
    latent_dim: int,
    *,
    absolute_tolerance: float | None = None,
    max_terms: int = 2_000_000,
    backend: DirectBackend = "auto",
) -> torch.Tensor:
    """Return the direct spherical Cauchy KL using analytic custom backward."""

    _validate_dimension(latent_dim)
    _validate_rho(rho)
    _validate_max_terms(max_terms)
    tolerance = (
        _default_tolerance(rho.dtype)
        if absolute_tolerance is None
        else float(absolute_tolerance)
    )
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("absolute_tolerance must be positive and finite")
    maximum_concentration = _largest_rho_below_one(rho.dtype)
    value_coefficients, gradient_coefficients, _, _, low_dimension = _direct_setup(
        rho,
        latent_dim,
        value_tolerance=tolerance,
        gradient_tolerance=tolerance,
        maximum_concentration=maximum_concentration,
        max_terms=max_terms,
        fixed=False,
    )
    return _DirectKlAutograd.apply(
        rho,
        value_coefficients,
        gradient_coefficients,
        latent_dim,
        backend,
        low_dimension,
    )


def spcauchy_pseudohyperbolic_distance(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Return the pseudohyperbolic distance between open-ball parameters."""

    _validate_ball_pair(a, b)
    a2 = a.square().sum(dim=-1)
    b2 = b.square().sum(dim=-1)
    difference = a - b
    difference_squared = difference.square().sum(dim=-1)
    denominator = difference_squared + (1.0 - a2) * (1.0 - b2)
    distance = torch.linalg.vector_norm(difference, dim=-1) / torch.sqrt(denominator)
    one = torch.ones((), dtype=a.dtype, device=a.device)
    zero = torch.zeros((), dtype=a.dtype, device=a.device)
    upper = torch.nextafter(one, zero)
    return torch.clamp(distance, min=0.0, max=upper)


def kl_divergence_spcauchy_pairwise(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    absolute_tolerance: float | None = None,
    max_terms: int = 2_000_000,
    backend: DirectBackend = "auto",
) -> torch.Tensor:
    """Return ``KL(spCauchy(a) || spCauchy(b))``.

    Within the spherical Cauchy family this KL is symmetric and equals the
    uniform-prior direct KL evaluated at the pseudohyperbolic distance between
    ``a`` and ``b``.
    """

    distance = spcauchy_pseudohyperbolic_distance(a, b)
    return kl_divergence_spcauchy_direct(
        distance,
        a.shape[-1],
        absolute_tolerance=absolute_tolerance,
        max_terms=max_terms,
        backend=backend,
    )


def kl_divergence_spcauchy_direct_fixed(
    rho: torch.Tensor,
    latent_dim: int,
    *,
    maximum_concentration: float,
    value_tolerance: float,
    gradient_tolerance: float,
    max_terms: int = 2_000_000,
    backend: DirectBackend = "auto",
) -> torch.Tensor:
    """Return a fixed-count evaluator certified on ``[0, maximum_concentration]``."""

    _validate_dimension(latent_dim)
    _validate_rho(rho)
    _validate_max_terms(max_terms)
    if not math.isfinite(maximum_concentration) or not 0 <= maximum_concentration < 1:
        raise ValueError("maximum_concentration must satisfy 0 <= rho < 1")
    if (
        not math.isfinite(value_tolerance)
        or not math.isfinite(gradient_tolerance)
        or value_tolerance <= 0
        or gradient_tolerance <= 0
    ):
        raise ValueError("value and gradient tolerances must be positive and finite")
    if rho.device.type == "cuda":
        torch._assert_async(
            (rho <= maximum_concentration).all(),
            "rho exceeds the certified maximum concentration",
        )
    elif bool((rho > maximum_concentration).any()):
        raise ValueError("rho exceeds the certified maximum concentration")
    value_coefficients, gradient_coefficients, _, _, low_dimension = _direct_setup(
        rho,
        latent_dim,
        value_tolerance=float(value_tolerance),
        gradient_tolerance=float(gradient_tolerance),
        maximum_concentration=float(maximum_concentration),
        max_terms=max_terms,
        fixed=True,
    )
    return _DirectKlAutograd.apply(
        rho,
        value_coefficients,
        gradient_coefficients,
        latent_dim,
        backend,
        low_dimension,
    )


def kl_divergence_spcauchy_even_neighbor(
    rho: torch.Tensor,
    latent_dim: int,
    *,
    backend: DirectBackend = "auto",
) -> torch.Tensor:
    """Return the finite even-neighbor approximation for odd ``D >= 7``."""

    _validate_dimension(latent_dim)
    _validate_rho(rho)
    value_coefficients, gradient_coefficients = _coefficients_on_input(
        rho,
        kind="neighbor",
        dimension=latent_dim,
        count=(latent_dim - 1) // 2,
        value_tolerance=0.0,
        gradient_tolerance=0.0,
        maximum_concentration=0.0,
        max_terms=0,
    )
    internal_backend = (
        "auto_neighbor"
        if backend == "auto"
        else "compiled_neighbor"
        if backend == "compiled"
        else backend
    )
    return _DirectKlAutograd.apply(
        rho,
        value_coefficients,
        gradient_coefficients,
        latent_dim,
        internal_backend,
        0,
    )


def direct_kl_diagnostics(
    rho: torch.Tensor,
    latent_dim: int,
    *,
    absolute_tolerance: float | None = None,
    maximum_concentration: float | None = None,
    value_tolerance: float | None = None,
    gradient_tolerance: float | None = None,
    max_terms: int = 2_000_000,
    backend: DirectBackend = "auto",
) -> DirectKlDiagnostics:
    """Return retained-term and backend diagnostics without evaluating the KL."""

    _validate_dimension(latent_dim)
    _validate_rho(rho)
    _validate_max_terms(max_terms)
    default = _default_tolerance(rho.dtype)
    value_tol = float(
        value_tolerance
        if value_tolerance is not None
        else absolute_tolerance
        if absolute_tolerance is not None
        else default
    )
    gradient_tol = float(
        gradient_tolerance
        if gradient_tolerance is not None
        else absolute_tolerance
        if absolute_tolerance is not None
        else default
    )
    max_rho = (
        _largest_rho_below_one(rho.dtype)
        if maximum_concentration is None
        else float(maximum_concentration)
    )
    if not math.isfinite(value_tol) or value_tol <= 0:
        raise ValueError("value_tolerance must be positive and finite")
    if not math.isfinite(gradient_tol) or gradient_tol <= 0:
        raise ValueError("gradient_tolerance must be positive and finite")
    if not math.isfinite(max_rho) or not 0 <= max_rho < 1:
        raise ValueError("maximum_concentration must satisfy 0 <= rho < 1")
    fixed = maximum_concentration is not None
    _, _, count, terminating_count, _ = _direct_setup(
        rho,
        latent_dim,
        value_tolerance=value_tol,
        gradient_tolerance=gradient_tol,
        maximum_concentration=max_rho,
        max_terms=max_terms,
        fixed=fixed,
    )
    return DirectKlDiagnostics(
        dimension=latent_dim,
        retained_terms=count,
        terminating_terms=terminating_count,
        is_exact_terminating=latent_dim % 2 == 0 or latent_dim in {3, 5},
        is_fixed_truncation=fixed,
        backend=_resolve_backend(rho, backend, count),
        absolute_tolerance=value_tol,
        gradient_tolerance=gradient_tol,
        maximum_concentration=max_rho,
    )


__all__ = [
    "DirectBackend",
    "DirectKlDiagnostics",
    "direct_kl_diagnostics",
    "kl_divergence_spcauchy_direct",
    "kl_divergence_spcauchy_direct_fixed",
    "kl_divergence_spcauchy_direct_with_gradient",
    "kl_divergence_spcauchy_even_neighbor",
    "kl_divergence_spcauchy_pairwise",
    "spcauchy_pseudohyperbolic_distance",
]
