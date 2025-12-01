import torch
import torch.nn.functional as F

def sample_uniform_sphere(batch_size, latent_dim, device='cuda'):
    """
    Generate uniform samples on the unit sphere S^{d-1}
    
    Args:
        batch_size (int): Number of samples to generate
        latent_dim (int): Dimension of the sphere
        device (str): Device to place the samples on
        
    Returns:
        torch.Tensor: Uniform samples on the unit sphere of shape (batch_size, latent_dim)
    """
    # Sample from standard normal distribution
    x = torch.randn(batch_size, latent_dim, device=device)
    
    # Normalize to the unit sphere
    x_normalized = F.normalize(x, dim=1)
    
    return x_normalized

def moebius_transform(x, mu, rho):
    """
    Apply Möbius transformation to uniform samples on the sphere
    
    Args:
        x (torch.Tensor): Uniform samples on the sphere of shape (batch_size, latent_dim)
        mu (torch.Tensor): Direction parameter of shape (batch_size, latent_dim)
        rho (torch.Tensor): Concentration parameter of shape (batch_size, 1)
        
    Returns:
        torch.Tensor: Samples from spherical Cauchy distribution
    """
    
    mu = F.normalize(mu, dim=1)
    rho = torch.clamp(rho, min=0.001, max=0.999)
    inner_prod = torch.sum(x * mu, dim=1, keepdim=True)
    numerator = x + rho * mu
    denominator = 1 + 2 * rho * inner_prod + rho.pow(2)
    
    transformed = (1 - rho.pow(2)) * (numerator / denominator) + rho * mu
    
    return transformed

def sample_spcauchy(mu, rho):
    """
    Sample from spherical Cauchy distribution
    
    Args:
        mu (torch.Tensor): Direction parameter of shape (batch_size, latent_dim)
        rho (torch.Tensor): Concentration parameter of shape (batch_size, 1)
        
    Returns:
        torch.Tensor: Samples from spherical Cauchy distribution
    """
    batch_size, latent_dim = mu.shape
    device = mu.device
    
    uniform_samples = sample_uniform_sphere(batch_size, latent_dim, device)
    spcauchy_samples = moebius_transform(uniform_samples, mu, rho)
    
    return spcauchy_samples
