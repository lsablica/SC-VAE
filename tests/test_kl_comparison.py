import torch
import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kl import kl_divergence_spcauchy, kl_divergence_spcauchy2, kl_divergence_spcauchy_asympt

def test_kl_methods_comparison(
    dimensions=[3],
    rho_values=[0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    k_terms=None,
    n_nodes=100,
    device="cpu"
):
    """
    Compare kl_divergence_spcauchy (series expansion) and kl_divergence_spcauchy2 (Gauss-Legendre quadrature)
    implementations across different dimensions and rho values.
    
    Args:
        dimensions (list): List of latent dimensions to test
        rho_values (list): List of concentration parameters to test
        k_terms (int, optional): Number of terms for series expansion. If None, uses the default
        n_nodes (int): Number of quadrature nodes for Gauss-Legendre (default: 100)
        device (str): Device to run on ('cpu' or 'cuda')
    """
    print(f"\n{'='*80}")
    print(f"Comparing KL divergence implementations")
    print(f"Series expansion (kl_divergence_spcauchy) vs Gauss-Legendre (kl_divergence_spcauchy2)")
    print(f"{'='*80}")
    
    print(f"\nParameters:")
    print(f"- Dimensions: {dimensions}")
    print(f"- Rho values: {rho_values}")
    print(f"- Series terms (k): {k_terms if k_terms else 'Default (max(dim*10, 1000))'}")
    print(f"- Quadrature nodes: {n_nodes}")
    print(f"- Device: {device}")
    
    print(f"\n{'='*80}")
    print(f"{'Dimension':^10} | {'Rho':^10} | {'Series KL':^15} | {'Quadrature KL':^15} | {'Asympt KL':^15} | {'Abs Diff':^15} | {'Rel Diff':^15} | {'Series Time (ms)':^18} | {'Quadrature Time (ms)':^20}")
    print(f"{'-'*10:^10}-+-{'-'*10:^10}-+-{'-'*15:^15}-+-{'-'*15:^15}-+-{'-'*15:^15}-+-{'-'*15:^15}-+-{'-'*15:^15}-+-{'-'*18:^18}-+-{'-'*20:^20}")
    
    # Initialize dictionaries to store timing results
    series_times = {d: [] for d in dimensions}
    quad_times = {d: [] for d in dimensions}
    
    for dim in dimensions:
        for rho_val in rho_values:
            rho_tensor = torch.tensor([[rho_val]], device=device)
            
            start_time = time.time()
            kl_series = kl_divergence_spcauchy(rho_tensor, dim, k_terms=k_terms).item()
            series_time_ms = (time.time() - start_time) * 1000
            series_times[dim].append(series_time_ms)
            
            start_time = time.time()
            kl_quad = kl_divergence_spcauchy2(rho_tensor, dim, n_nodes=n_nodes).item()
            quad_time_ms = (time.time() - start_time) * 1000
            quad_times[dim].append(quad_time_ms)
            
            abs_diff = abs(kl_series - kl_quad)
            rel_diff = abs_diff / max(abs(kl_quad), 1e-10)

            kl_quad_as = kl_divergence_spcauchy_asympt(rho_tensor, dim, n_nodes=n_nodes).item()
            
            print(f"{dim:^10} | {rho_val:^10.2f} | {kl_series:^15.6f} | {kl_quad:^15.6f} | {kl_quad_as:^15.6f} | {abs_diff:^15.6e} | {rel_diff:^15.6e} | {series_time_ms:^18.3f} | {quad_time_ms:^20.3f}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"Performance Summary:")
    print(f"{'-'*80}")
    print(f"{'Dimension':^10} | {'Avg Series Time (ms)':^25} | {'Avg Quadrature Time (ms)':^25} | {'Speedup Factor':^15}")
    print(f"{'-'*10:^10}-+-{'-'*25:^25}-+-{'-'*25:^25}-+-{'-'*15:^15}")
    
    for dim in dimensions:
        avg_series_time = sum(series_times[dim]) / len(series_times[dim])
        avg_quad_time = sum(quad_times[dim]) / len(quad_times[dim])
        speedup = avg_series_time / avg_quad_time if avg_quad_time > 0 else float('inf')
        
        print(f"{dim:^10} | {avg_series_time:^25.3f} | {avg_quad_time:^25.3f} | {speedup:^15.2f}")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare KL divergence implementations')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], 
                       help='Device to run on')
    parser.add_argument('--k_terms', type=int, default=None, 
                       help='Number of terms for series expansion')
    parser.add_argument('--n_nodes', type=int, default=300, 
                       help='Number of quadrature nodes for Gauss-Legendre')
    
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    
    # Run the comparison with different dimensions and rho values
    test_kl_methods_comparison(
        dimensions=[3, 5, 10, 20, 50, 100, 1000],
        rho_values=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999],
        k_terms=args.k_terms,
        n_nodes=args.n_nodes,
        device=device
    )
