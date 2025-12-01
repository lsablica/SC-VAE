import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kl import kl_divergence_spcauchy

def test_kl_accuracy_vs_rho(k_default=10, k_accurate=100, num_points=100, latent_dim=10, device="cpu"):
    """
    Test 1: Compare KL divergence with k=10 vs k=100 for different rho values
    
    Args:
        k_default (int): Default number of terms in the series expansion
        k_accurate (int): High accuracy number of terms as reference
        num_points (int): Number of rho values to test
        latent_dim (int): Dimension of the latent space
        device (str): Device to run on ('cpu' or 'cuda')
    """
    print(f"Test 1: KL accuracy vs rho, comparing k={k_default} with k={k_accurate}, d={latent_dim}")
    
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Generate rho values from 0.01 to 0.99
    rho_range = np.linspace(0.01, 0.99, num_points)
    rho_tensor = torch.tensor(rho_range, device=device).view(-1, 1)
    
    kl_default = kl_divergence_spcauchy(rho_tensor, latent_dim, k_terms=k_default).cpu().numpy()
    
    kl_accurate = kl_divergence_spcauchy(rho_tensor, latent_dim, k_terms=k_accurate).cpu().numpy()
    
    abs_diff = np.abs(kl_default - kl_accurate)
    rel_diff = abs_diff / np.maximum(np.abs(kl_accurate), 1e-10)
    
    print(f"Max absolute difference: {np.max(abs_diff):.8f}")
    print(f"Mean absolute difference: {np.mean(abs_diff):.8f}")
    print(f"Max relative difference: {np.max(rel_diff):.8f}")
    print(f"Mean relative difference: {np.mean(rel_diff):.8f}")
    
    # Plot KL values
    plt.figure(figsize=(10, 6))
    plt.plot(rho_range, kl_default, label=f'k={k_default} (default)')
    plt.plot(rho_range, kl_accurate, label=f'k={k_accurate} (accurate)', linestyle='--')
    plt.title(f'KL Divergence vs. Concentration Parameter ρ (d={latent_dim})')
    plt.xlabel('ρ')
    plt.ylabel('KL(p||q)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(figures_dir, f'kl_values_d{latent_dim}_k{k_default}_vs_k{k_accurate}.png'))
    plt.close()
    
    # Plot absolute difference
    plt.figure(figsize=(10, 6))
    plt.plot(rho_range, abs_diff)
    plt.title(f'Absolute Difference in KL Divergence (k={k_default} vs k={k_accurate}, d={latent_dim})')
    plt.xlabel('ρ')
    plt.ylabel(f'|KL_k{k_default} - KL_k{k_accurate}|')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, f'kl_abs_diff_d{latent_dim}_k{k_default}_vs_k{k_accurate}.png'))
    plt.close()
    
    # Plot relative difference
    plt.figure(figsize=(10, 6))
    plt.semilogy(rho_range, rel_diff)
    plt.title(f'Relative Difference in KL Divergence (k={k_default} vs k={k_accurate}, d={latent_dim})')
    plt.xlabel('ρ')
    plt.ylabel(f'|KL_k{k_default} - KL_k{k_accurate}| / |KL_k{k_accurate}|')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, f'kl_rel_diff_d{latent_dim}_k{k_default}_vs_k{k_accurate}.png'))
    plt.close()


def test_kl_convergence_vs_k(rho=0.4, dimensions=[5, 10, 20, 50, 100, 200, 500, 1000], max_k=100, device="cpu"):
    """
    Test 2: Show convergence as k increases for multiple dimensions with fixed rho
    
    Args:
        rho (float): Fixed concentration parameter
        dimensions (list): Dimensions to test
        max_k (int): Maximum number of terms in series expansion
        device (str): Device to run on ('cpu' or 'cuda')
    """
    print(f"Test 2: KL convergence vs k for rho={rho}, dimensions={dimensions}")
    
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    k_values = np.arange(1, max_k + 1)
    
    rho_tensor = torch.tensor([[rho]], device=device)
    
    plt.figure(figsize=(12, 8))
    
    results = {}
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(dimensions)))
    
    for i, d in enumerate(dimensions):
        print(f"Computing for dimension d={d}")
        
        kl_values = []
        for k in k_values:
            kl = kl_divergence_spcauchy(rho_tensor, d, k_terms=int(k)).item()
            kl_values.append(kl)
        
        results[d] = kl_values
        print(kl_values)
        # Plot
        plt.plot(k_values, kl_values, label=f'd={d}', color=colors[i])
    
    plt.title(f'KL Divergence Convergence vs. Number of Terms k (ρ={rho})')
    plt.xlabel('Number of terms k')
    plt.ylabel('KL(p||q)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(figures_dir, f'kl_convergence_rho{rho}_multiple_dims.png'))
    plt.close()
    
    plt.figure(figsize=(12, 8))
    
    for i, d in enumerate(dimensions):
        plt.plot(k_values, results[d], label=f'd={d}', color=colors[i])
    
    plt.xscale('log')
    plt.title(f'KL Divergence Convergence vs. Number of Terms k (Log Scale, ρ={rho})')
    plt.xlabel('Number of terms k (log scale)')
    plt.ylabel('KL(p||q)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(figures_dir, f'kl_convergence_rho{rho}_multiple_dims_logx.png'))
    plt.close()
    
    plt.figure(figsize=(12, 8))
    
    for i, d in enumerate(dimensions):
        kl_ref = results[d][-1]
        rel_changes = [abs(kl - kl_ref) / max(abs(kl_ref), 1e-10) for kl in results[d]]
        plt.semilogy(k_values, rel_changes, label=f'd={d}', color=colors[i])
    
    plt.title(f'Relative Change in KL Divergence vs. Number of Terms k (ρ={rho})')
    plt.xlabel('Number of terms k')
    plt.ylabel('|KL_k - KL_{max}| / |KL_{max}|')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(figures_dir, f'kl_rel_change_rho{rho}_multiple_dims.png'))
    plt.close()


def test_kl_accuracy_vs_k(rho=0.6, dimensions=[5, 10, 20, 50, 100], reference_k=1000, device="cpu"):
    """
    Test KL divergence accuracy by comparing different k values against a reference k=1000
    across multiple dimensions.
    
    Args:
        rho (float): Fixed concentration parameter
        dimensions (list): Dimensions to test
        reference_k (int): Reference k value for "ground truth" KL
        device (str): Device to run on ('cpu' or 'cuda')
    """
    print(f"Testing KL accuracy vs k for rho={rho}, dimensions={dimensions}")
    
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    k_values = np.unique(np.concatenate([
        np.arange(10, 100, 10),      # 10, 20, ..., 90
        np.arange(100, 300, 25),     # 100, 125, ..., 275
        np.arange(300, 600, 50),     # 300, 350, ..., 550
        np.arange(600, 1000, 100),   # 600, 700, ..., 900
        [reference_k]                # Include reference k
    ])).astype(int)
    
    rho_tensor = torch.tensor([[rho]], device=device)
    
    results = {}
    rel_errors = {}
    
    for d in dimensions:
        print(f"Computing for dimension d={d}")
        
        reference_kl = kl_divergence_spcauchy(rho_tensor, d, k_terms=reference_k).item()
        print(f"  Reference KL (k={reference_k}): {reference_kl:.8f}")
        
        kl_values = []
        errors = []
        
        for k in k_values:
            if k != reference_k:  # Skip reference k as it's already computed
                kl = kl_divergence_spcauchy(rho_tensor, d, k_terms=int(k)).item()
                rel_error = abs(kl - reference_kl) / max(abs(reference_kl), 1e-10)
                
                kl_values.append(kl)
                errors.append(rel_error)
                
                if k in [10, 20, 50, 100, 200, 500]:
                    print(f"  k={k:4d}: KL={kl:.8f}, Rel Error={rel_error:.8f}")
            else:
                kl_values.append(reference_kl)
                errors.append(0.0)  # Zero error for reference k
        
        results[d] = kl_values
        rel_errors[d] = errors
    
    # Plot relative errors vs k for all dimensions
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(dimensions)))
    
    for i, d in enumerate(dimensions):
        plt.loglog(k_values, rel_errors[d], label=f'd={d}', color=colors[i], marker='o')
    
    plt.title(f'Relative Error in KL Divergence vs. Number of Terms k (ρ={rho})')
    plt.xlabel('Number of terms k (log scale)')
    plt.ylabel('Relative Error |KL_k - KL_{reference}| / |KL_{reference}| (log scale)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, f'kl_accuracy_vs_k_rho{rho}.png'))
    
    # Also save a linear-scale version for k
    plt.figure(figsize=(12, 8))
    for i, d in enumerate(dimensions):
        plt.semilogy(k_values, rel_errors[d], label=f'd={d}', color=colors[i], marker='o')
    
    plt.title(f'Relative Error in KL Divergence vs. Number of Terms k (ρ={rho})')
    plt.xlabel('Number of terms k')
    plt.ylabel('Relative Error |KL_k - KL_{reference}| / |KL_{reference}| (log scale)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, f'kl_accuracy_vs_k_rho{rho}_linear_x.png'))
    
    # Plot errors for each dimension at specific k values
    specific_k_values = [10, 20, 50, 100, 200, 500]
    specific_k_indices = [np.where(k_values == k)[0][0] for k in specific_k_values if k in k_values]
    
    plt.figure(figsize=(10, 7))
    markers = ['o', 's', '^', 'D', '*', 'x']
    
    for idx, k_idx in enumerate(specific_k_indices):
        k = k_values[k_idx]
        errors_at_k = [rel_errors[d][k_idx] for d in dimensions]
        plt.semilogy(dimensions, errors_at_k, label=f'k={k}', marker=markers[idx % len(markers)])
    
    plt.title(f'Relative Error in KL Divergence vs. Dimension (ρ={rho})')
    plt.xlabel('Dimension d')
    plt.ylabel('Relative Error |KL_k - KL_{reference}| / |KL_{reference}|')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, f'kl_accuracy_vs_dimension_rho{rho}.png'))
    
    print("Done. Check figures directory for plots.")
    
    return {
        'k_values': k_values,
        'dimensions': dimensions,
        'results': results,
        'rel_errors': rel_errors
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test KL divergence accuracy')
    parser.add_argument('--test', type=int, default=1, choices=[1, 2, 3], 
                       help='Test to run: 1=KL vs rho, 2=KL convergence vs k, 3=KL accuracy vs k')
    parser.add_argument('--dim', type=int, default=10, 
                       help='Latent dimension for test 1')
    parser.add_argument('--rho', type=float, default=0.4,
                       help='Concentration parameter for test 2 and 3')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], 
                       help='Device to run on')
    parser.add_argument('--reference_k', type=int, default=1000,
                       help='Reference k value for high accuracy comparison in test 3')
    
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    
    if args.test == 1:
        test_kl_accuracy_vs_rho(latent_dim=args.dim, device=device)
    elif args.test == 2:
        test_kl_convergence_vs_k(rho=args.rho, device=device)
    else:
        test_kl_accuracy_vs_k(rho=args.rho, reference_k=args.reference_k, device=device)
    
    print("Done. Check figures directory for plots.")
