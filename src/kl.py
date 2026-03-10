import functools
import math

import numpy as np
import torch


SPCAUCHY_KL_APPROXIMATION_ALIASES = {
    "dynamic": "weighted",
    "dynamic_weight": "weighted",
    "dynamic_weighted": "weighted",
}
SPCAUCHY_KL_APPROXIMATION_MODES = {"midpoint", "weighted"}
SPCAUCHY_RHO_EPS = 1e-3


def canonicalize_spcauchy_kl_approximation(approximation):
    """Normalize the optional KL approximation selector."""
    if approximation is None:
        return None

    normalized = str(approximation).strip().lower().replace("-", "_")
    normalized = SPCAUCHY_KL_APPROXIMATION_ALIASES.get(normalized, normalized)

    if normalized in {"exact", "none"}:
        return None

    if normalized not in SPCAUCHY_KL_APPROXIMATION_MODES:
        supported = ", ".join(sorted(SPCAUCHY_KL_APPROXIMATION_MODES))
        raise ValueError(
            f"Unsupported spherical Cauchy KL approximation: {approximation!r}. "
            f"Supported values are None, 'exact', {supported}."
        )

    return normalized


def _prepare_spcauchy_inputs(rho):
    """Clamp rho to a numerically safe range and compute z = 4 rho / (1 + rho)^2."""
    if rho.ndim == 0:
        rho = rho.view(1, 1)
    elif rho.ndim == 1:
        rho = rho.unsqueeze(-1)

    rho = torch.clamp(rho, min=SPCAUCHY_RHO_EPS, max=1 - SPCAUCHY_RHO_EPS)
    z = 4 * rho / ((1 + rho) ** 2)
    return rho, z


def _get_spcauchy_bracket_terms(z, latent_dim):
    """Return the lower and upper H_d brackets and their constant width."""
    c = latent_dim - 1
    delta = z.new_tensor(c / 2)
    c_tensor = z.new_tensor(float(c))

    digamma_delta = torch.digamma(delta)
    digamma_c = torch.digamma(c_tensor)
    width = digamma_c - digamma_delta - math.log(2.0)

    lower = digamma_delta - digamma_c + torch.log(2 - z)
    upper = torch.log1p(-0.5 * z)
    return lower, upper, width


def spcauchy_h_approximation(z, latent_dim, approximation="weighted"):
    """
    Approximate the bracketed H_d(z) term from the paper.

    Args:
        z (torch.Tensor): z = 4 rho / (1 + rho)^2.
        latent_dim (int): Dimension d of the latent space.
        approximation (str): One of {'midpoint', 'weighted'}.

    Returns:
        torch.Tensor: Approximation of H_d(z) with the same shape as z.
    """
    approximation = canonicalize_spcauchy_kl_approximation(approximation)
    if approximation is None:
        raise ValueError("An approximation mode is required for spcauchy_h_approximation.")

    lower, upper, width = _get_spcauchy_bracket_terms(z, latent_dim)

    if approximation == "midpoint":
        return 0.5 * (lower + upper)

    z_squared = z.square()
    a_star = 1 / (16 * latent_dim * width) - 1
    alpha_star = ((1 - z_squared) / (1 + a_star * z_squared)).square()
    return lower + width * alpha_star


def kl_divergence_spcauchy_approx(rho, latent_dim, approximation="weighted"):
    """
    Approximate the spherical Cauchy KL divergence with a closed-form bracket surrogate.

    Args:
        rho (torch.Tensor): Concentration parameter of shape (batch_size, 1).
        latent_dim (int): Dimension d of the latent space.
        approximation (str): One of {'midpoint', 'weighted'}.

    Returns:
        torch.Tensor: Approximate KL divergence of shape (batch_size,).
    """
    approximation = canonicalize_spcauchy_kl_approximation(approximation)
    if approximation is None:
        raise ValueError("An approximation mode is required for kl_divergence_spcauchy_approx.")

    _, z = _prepare_spcauchy_inputs(rho)
    h_approx = spcauchy_h_approximation(z, latent_dim, approximation=approximation)
    kl = (latent_dim - 1) * (h_approx - 0.5 * torch.log1p(-z))
    return kl.squeeze(-1)


@functools.lru_cache(maxsize=32)
def get_gauss_legendre_nodes_weights(n_nodes, device, dtype):
    """
    Get Gauss-Legendre nodes and weights, with caching for improved performance.

    Args:
        n_nodes (int): Number of quadrature nodes.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        tuple: (nodes, weights) where both are torch tensors of shape [1, n_nodes].
    """
    nodes_np, weights_np = np.polynomial.legendre.leggauss(n_nodes)

    # Transform nodes from [-1, 1] to [0, 1]: t = 0.5 * (x + 1), dt/dx = 0.5.
    t = torch.tensor(0.5 * (nodes_np + 1), dtype=dtype, device=device).view(1, -1)
    weights = torch.tensor(0.5 * weights_np, dtype=dtype, device=device).view(1, -1)

    return t, weights


def kl_divergence_spcauchy(rho, latent_dim, k_terms=None):
    """
    Compute the exact spherical Cauchy KL divergence with the power-series form.

    Args:
        rho (torch.Tensor): Concentration parameter of shape (batch_size, 1).
        latent_dim (int): Dimension of the latent space.
        k_terms (int, optional): Number of terms in the series expansion.
            If None, uses max(latent_dim * 10, 1000).

    Returns:
        torch.Tensor: KL divergence of shape (batch_size,).
    """
    if k_terms is None:
        k_terms = max(latent_dim * 10, 1000)

    rho, z = _prepare_spcauchy_inputs(rho)
    device = rho.device
    batch_size = rho.shape[0]

    log_ratio = torch.log((1 - rho) / (1 + rho))
    first_term = (latent_dim - 1) * log_ratio
    log_one_minus_z_pow = (latent_dim - 1) * log_ratio

    half_d_minus_1 = (latent_dim - 1) / 2
    k_indices = torch.arange(k_terms, dtype=rho.dtype, device=device).view(1, -1)

    log_gamma_half_d_plus_k = torch.lgamma(half_d_minus_1 + k_indices)
    log_gamma_half_d = torch.lgamma(rho.new_tensor(half_d_minus_1))
    log_pochhammer = log_gamma_half_d_plus_k - log_gamma_half_d

    log_factorial = torch.lgamma(k_indices + 1)

    digamma_d_minus_1 = torch.digamma(rho.new_tensor(float(latent_dim - 1)))
    digamma_d_minus_1_plus_k = torch.digamma(latent_dim - 1 + k_indices)
    digamma_diff = digamma_d_minus_1_plus_k - digamma_d_minus_1

    log_z = torch.log(z).view(batch_size, 1)
    log_z_k = k_indices * log_z

    log_series_terms = log_pochhammer - log_factorial + log_z_k + log_one_minus_z_pow.view(batch_size, 1)

    series_terms = torch.exp(log_series_terms) * digamma_diff
    series_sum = torch.sum(series_terms, dim=1, keepdim=True)

    second_term = (latent_dim - 1) * series_sum
    kl = first_term + second_term
    return kl.squeeze(-1)


def kl_divergence_spcauchy2(rho, latent_dim, n_nodes=None):
    """
    Compute the exact spherical Cauchy KL divergence with Gauss-Legendre quadrature.

    Args:
        rho (torch.Tensor): Concentration parameter of shape (batch_size, 1).
        latent_dim (int): Dimension of the latent space.
        n_nodes (int, optional): Number of quadrature nodes. Defaults to 1000.

    Returns:
        torch.Tensor: KL divergence of shape (batch_size,).
    """
    if n_nodes is None:
        n_nodes = 1000

    rho, z = _prepare_spcauchy_inputs(rho)
    device = rho.device
    dtype = rho.dtype
    batch_size = rho.shape[0]
    c = latent_dim - 1

    log_ratio = torch.log((1 - rho) / (1 + rho))
    first_term = c * log_ratio

    t, weights = get_gauss_legendre_nodes_weights(n_nodes, device, dtype)

    z = z.view(batch_size, 1)
    numerator = t.pow(c - 1)
    denominator = 1 - t

    factor = (1 - z) / (1 - t * z)
    integrand = (numerator / denominator) * (1 - factor.pow(c / 2))

    series_sum = torch.sum(weights * integrand, dim=1, keepdim=True)
    second_term = c * series_sum
    return (first_term + second_term).squeeze(-1)


def kl_divergence_spcauchy_asympt(rho, latent_dim, n_nodes=None):
    """
    Compute the fixed-d asymptotic spherical Cauchy KL divergence near rho -> 1.

    Args:
        rho (torch.Tensor): Concentration parameter of shape (batch_size, 1).
        latent_dim (int): Dimension of the latent space.
        n_nodes (int): Unused. Kept for API compatibility.

    Returns:
        torch.Tensor: KL divergence of shape (batch_size,).
    """
    rho, _ = _prepare_spcauchy_inputs(rho)
    c = rho.new_tensor(float(latent_dim - 1))
    kldiv = torch.log((1 + rho) / (1 - rho)) + torch.digamma(c / 2) - torch.digamma(c)
    return (c * kldiv).squeeze(-1)


def kl_divergence_spcauchy_combined(rho, latent_dim, n_nodes=None, approximation=None):
    """
    Compute the spherical Cauchy KL divergence with either the exact or surrogate path.

    By default this keeps the current exact behavior: quadrature for moderate rho and
    the asymptotic form for large rho. Setting ``approximation`` to ``"midpoint"`` or
    ``"weighted"`` switches to the closed-form bracket surrogates from the paper.

    Args:
        rho (torch.Tensor): Concentration parameter of shape (batch_size, 1).
        latent_dim (int): Dimension of the latent space.
        n_nodes (int, optional): Number of quadrature nodes for the exact path.
        approximation (str, optional): Optional surrogate selector.

    Returns:
        torch.Tensor: KL divergence of shape (batch_size,).
    """
    approximation = canonicalize_spcauchy_kl_approximation(approximation)
    if approximation is not None:
        return kl_divergence_spcauchy_approx(rho, latent_dim, approximation=approximation)

    asympt_mask = rho > 0.9
    kl = torch.zeros_like(rho)

    if asympt_mask.any():
        kl[asympt_mask] = kl_divergence_spcauchy_asympt(rho[asympt_mask], latent_dim)

    if (~asympt_mask).any():
        kl[~asympt_mask] = kl_divergence_spcauchy2(rho[~asympt_mask], latent_dim, n_nodes)

    return kl.squeeze(-1)


def kl_divergence_normal(mu, logvar):
    """
    Compute KL divergence between a normal distribution and a standard normal prior.

    Args:
        mu: Mean vectors of shape (batch_size, latent_dim).
        logvar: Log-variance vectors of shape (batch_size, latent_dim).

    Returns:
        torch.Tensor: KL divergence of shape (batch_size,).
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl
