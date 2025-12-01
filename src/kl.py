import torch
import torch.nn.functional as F
import numpy as np
import functools


@functools.lru_cache(maxsize=32)
def get_gauss_legendre_nodes_weights(n_nodes, device, dtype):
    """
    Get Gauss-Legendre nodes and weights, with caching for improved performance.
    
    Args:
        n_nodes (int): Number of quadrature nodes
        device_str (str): String representation of device
        dtype_str (str): String representation of dtype
        
    Returns:
        tuple: (nodes, weights) where both are torch tensors of shape [1, n_nodes]
    """
    nodes_np, weights_np = np.polynomial.legendre.leggauss(n_nodes)
    
    # Transform nodes from [-1,1] to [0,1]: t = 0.5*(x+1), dt/dx = 0.5
    t = torch.tensor(0.5 * (nodes_np + 1), dtype=dtype, device=device).view(1, -1)
    weights = torch.tensor(0.5 * weights_np, dtype=dtype, device=device).view(1, -1)
    
    return t, weights


def kl_divergence_spcauchy(rho, latent_dim, k_terms=None):
    """
    Compute KL divergence between spherical Cauchy distribution and uniform prior
    using the formula from Section 3.2 in a fully vectorized manner:
    
    KL(p||q) = (d-1)·log((1-ρ)/(1+ρ)) + 
              (d-1)·(1-z)^((d-1)/2)·sum_k { ((d-1)/2)_k / k! · z^k · [ψ(d-1+k) - ψ(d-1)] }
    
    where z = 4ρ/(1+ρ)²
    
    Args:
        rho (torch.Tensor): Concentration parameter of shape (batch_size, 1)
        latent_dim (int): Dimension of the latent space
        k_terms (int, optional): Number of terms in series expansion. 
                               If None, uses max(latent_dim*10, 1000) for adaptive precision.
        
    Returns:
        torch.Tensor: KL divergence of shape (batch_size,)
    """
    # Set default k_terms to scale with dimension, with minimum 1000
    if k_terms is None:
        k_terms = max(latent_dim * 10, 1000)
    
    # Ensure valid rho (avoid numerical issues)
    rho = torch.clamp(rho, min=1e-3, max=1-1e-3)
    device = rho.device
    batch_size = rho.shape[0]
    
    #  first term: (d-1)·log((1-ρ)/(1+ρ))
    log_ratio = torch.log((1 - rho) / (1 + rho))
    first_term = (latent_dim - 1) * log_ratio
    
    # log of (1-z)^((d-1)/2) = log(((1-ρ)/(1+ρ))^(d-1))
    log_one_minus_z_pow = (latent_dim - 1) * log_ratio
    
    # z = 4ρ/(1+ρ)²
    z = 4 * rho / ((1 + rho) ** 2)
    
    half_d_minus_1 = (latent_dim - 1) / 2
    
    k_indices = torch.arange(k_terms, dtype=torch.float32, device=device).view(1, -1)
    
    # log(Pochhammer((d-1)/2, k)) = log(Γ((d-1)/2 + k)) - log(Γ((d-1)/2))
    log_gamma_half_d_plus_k = torch.lgamma(half_d_minus_1 + k_indices)
    log_gamma_half_d = torch.lgamma(torch.tensor(half_d_minus_1, device=device))
    log_pochhammer = log_gamma_half_d_plus_k - log_gamma_half_d
    
    #   log(k!)
    log_factorial = torch.lgamma(k_indices + 1)
    
    #   digamma(d-1+k) and digamma difference
    digamma_d_minus_1 = torch.digamma(torch.tensor(float(latent_dim - 1), device=device))
    digamma_d_minus_1_plus_k = torch.digamma(latent_dim - 1 + k_indices)
    digamma_diff = digamma_d_minus_1_plus_k - digamma_d_minus_1
    
    #  log(z^k) = k * log(z)
    log_z = torch.log(z).view(batch_size, 1)
    log_z_k = k_indices * log_z
    
    log_series_terms = log_pochhammer - log_factorial + log_z_k + log_one_minus_z_pow.view(batch_size, 1)
    
    series_terms = torch.exp(log_series_terms) * digamma_diff
    series_sum = torch.sum(series_terms, dim=1, keepdim=True)
    
    second_term = (latent_dim - 1) * series_sum
    
    kl = first_term + second_term
    
    return kl.squeeze(-1)  # Return shape [batch_size]



def kl_divergence_spcauchy2(rho, latent_dim, n_nodes=None):
    """
    Compute KL divergence between spherical Cauchy distribution and uniform prior
    using Gauss-Legendre quadrature for numerical integration.
    
    KL(p||q) = (d-1)·log((1-ρ)/(1+ρ)) + (d-1)·integral_representation
    
    Args:
        rho (torch.Tensor): Concentration parameter of shape (batch_size, 1)
        latent_dim (int): Dimension of the latent space
        n_nodes (int): Number of quadrature nodes (default: 100)
        
    Returns:
        torch.Tensor: KL divergence of shape (batch_size,)
    """
    # Ensure valid rho (avoid numerical issues)
    if n_nodes is None:
        n_nodes = 1000
    rho = torch.clamp(rho, min=1e-3, max=1-1e-3)
    device = rho.device
    dtype = rho.dtype
    batch_size = rho.shape[0]
    c = latent_dim - 1  # For notational simplicity
    
    # first term: (d-1)·log((1-ρ)/(1+ρ))
    log_ratio = torch.log((1 - rho) / (1 + rho))
    first_term = c * log_ratio
    #  z = 4ρ/(1+ρ)²
    z = 4 * rho / ((1 + rho) ** 2)
    
    t, weights = get_gauss_legendre_nodes_weights(n_nodes, device, dtype)
    
    z = z.view(batch_size, 1)
    numerator = t.pow(c - 1)  # [1, n_nodes]
    denominator = (1 - t)  # [1, n_nodes]
    
    factor = (1 - z) / (1 - t * z)  # [batch_size, n_nodes]
    
    integrand = (numerator / denominator) * (1 - factor.pow(c / 2))  # [batch_size, n_nodes]
    

    series_sum = torch.sum(weights * integrand, dim=1)  # [batch_size, 1]
    second_term = c * series_sum
    return first_term + second_term   # Return shape [batch_size]


def kl_divergence_spcauchy_asympt(rho, latent_dim, n_nodes=None):
    """
    Compute KL divergence between spherical Cauchy distribution and uniform prior
    using Gauss-Legendre quadrature for numerical integration.
    
    KL(p||q) = (d-1)(log((1+rho)/(1-rho)) + digamma((d-1)/2) - digamma(d-1) ) 
    
    Args:
        rho (torch.Tensor): Concentration parameter of shape (batch_size, 1)
        latent_dim (int): Dimension of the latent space
        n_nodes (int): not used
        
    Returns:
        torch.Tensor: KL divergence of shape (batch_size,)
    """
    # Ensure valid rho (avoid numerical issues)
    device = rho.device
    dtype = rho.dtype
    batch_size = rho.shape[0]
    c = torch.tensor(latent_dim - 1, device = device)  # For notational simplicity
    
    kldiv = torch.log((1 + rho) / (1 - rho)) + torch.digamma(c / 2) - torch.digamma(c)
    return c * kldiv

def kl_divergence_spcauchy_combined(rho, latent_dim, n_nodes=None):
    """
    Compute the KL divergence between a spherical Cauchy distribution and a uniform prior,
    using two different methods depending on the value of rho.
    
    Args:
        rho (torch.Tensor): Concentration parameter of shape (batch_size, 1)
        latent_dim (int): Dimension of the latent space.
        n_nodes (int, optional): Number of quadrature nodes for the integration method. 
                                 Default is None (which sets it internally to 1000).
        
    Returns:
        torch.Tensor: KL divergence of shape (batch_size,)
    """
    # Create a mask
    asympt_mask = rho > 0.9
    
    kl = torch.zeros_like(rho)
    
    if asympt_mask.any():
        kl[asympt_mask] = kl_divergence_spcauchy_asympt(rho[asympt_mask], latent_dim)
    
    if (~asympt_mask).any():
        kl[(~asympt_mask)] = kl_divergence_spcauchy2(rho[(~asympt_mask)], latent_dim, n_nodes)
    return kl.squeeze(-1)


def kl_divergence_normal(mu, logvar):
    """
    Compute KL divergence between a normal distribution and a standard normal prior
    
    Args:
        mu: mean vectors - shape (batch_size, latent_dim)
        logvar: log variance vectors - shape (batch_size, latent_dim)
    
    Returns:
        KL divergence - shape (batch_size,)
    """
    # KL = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl
