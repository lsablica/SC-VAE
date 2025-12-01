import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision.utils import make_grid
from typing import List, Optional, Tuple
from matplotlib import cm
from sklearn.decomposition import PCA
import random
import os

def set_all_seeds(seed: int):
    """
    Set all seeds for reproducibility
    
    Args:
        seed: Seed value to use
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
        
        # Make CUDA operations deterministic
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
    
    os.environ["PYTHONHASHSEED"] = str(seed)

def plot_latent_space_3d(model, data_loader, device='cuda', num_batches=1):
    """
    Visualize the latent space in 3D (works when latent_dim=3)
    
    Args:
        model: The VAE model
        data_loader: DataLoader containing the data
        device: Device to use
        num_batches: Number of batches to process
    """
    model.eval()
    
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            if i >= num_batches:
                break
                
            data = data.to(device)
            mu, rho = model.encode(data)
            
            latent_vectors.append(mu.cpu().numpy())
            labels.append(label.numpy())
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, color='gray', alpha=0.1)
    
    scatter = ax.scatter(
        latent_vectors[:, 0],
        latent_vectors[:, 1],
        latent_vectors[:, 2],
        c=labels,
        cmap='tab10',
        s=15,
        alpha=0.8
    )
    
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    
    ax.set_title('3D Latent Space Visualization')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.tight_layout()
    
    return fig

def plot_reconstructions(model, dataloader, num_examples=8, device='cpu'):
    """
    Plot original images and their reconstructions
    """
    model.eval()
    
    for data, _ in dataloader:
        inputs = data[:num_examples].to(device)
        break
    
    with torch.no_grad():
        reconstructions, _, _ = model(inputs)
    
    inputs = inputs.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()
    
    fig, axes = plt.subplots(2, num_examples, figsize=(2*num_examples, 4))
    
    for i in range(num_examples):
        if inputs.shape[1] == 1:  # Grayscale
            axes[0, i].imshow(inputs[i].squeeze(), cmap='gray')
        else:  # RGB
            axes[0, i].imshow(np.transpose(inputs[i], (1, 2, 0)))
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
    
    for i in range(num_examples):
        if reconstructions.shape[1] == 1:  # Grayscale
            axes[1, i].imshow(reconstructions[i].squeeze(), cmap='gray')
        else:  # RGB
            axes[1, i].imshow(np.transpose(reconstructions[i], (1, 2, 0)))
        axes[1, i].set_title('Reconstruction')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_samples(model, num_samples=16, device='cpu'):
    """
    Plot samples generated from the model
    """
    model.eval()
    
    with torch.no_grad():
        samples = model.generate_samples(num_samples=num_samples)
    
    samples = samples.cpu().numpy()
    
    nrows = int(np.sqrt(num_samples))
    ncols = int(np.ceil(num_samples / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows))
    axes = axes.flatten()
    
    for i in range(num_samples):
        if samples.shape[1] == 1:  # Grayscale
            axes[i].imshow(samples[i].squeeze(), cmap='gray')
        else:  # RGB
            axes[i].imshow(np.transpose(samples[i], (1, 2, 0)))
        axes[i].axis('off')
    
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    return fig

def interpolate_latent(model, data_loader, num_points=10, device='cuda'):
    """
    Generate latent space interpolation between two random examples
    
    Args:
        model: The VAE model
        data_loader: DataLoader containing the data
        num_points: Number of interpolation points
        device: Device to use
    """
    model.eval()
    
    for data, _ in data_loader:
        data = data.to(device)
        if data.shape[0] >= 2:
            img1, img2 = data[0:1], data[1:2]
            break
    
    with torch.no_grad():
        mu1, _ = model.encode(img1)
        mu2, _ = model.encode(img2)
        
        t = torch.linspace(0, 1, num_points, device=device)
        
        omega = torch.acos(torch.clamp((mu1 * mu2).sum(dim=1), -1.0, 1.0))
        sin_omega = torch.sin(omega)
        
        interp_mus = []
        for ti in t:
            weight1 = torch.sin((1.0 - ti) * omega) / sin_omega
            weight2 = torch.sin(ti * omega) / sin_omega
            interp_mu = weight1 * mu1 + weight2 * mu2
            interp_mu = F.normalize(interp_mu, dim=1)
            interp_mus.append(interp_mu)
        
        interp_mus = torch.cat(interp_mus, dim=0)
        interp_imgs = model.decode(interp_mus)
        
        grid = make_grid(interp_imgs, nrow=num_points)
        grid_np = grid.cpu().permute(1, 2, 0).numpy()
        
        fig, ax = plt.subplots(figsize=(15, 3))
        ax.imshow(grid_np)
        ax.set_title('Latent Space Interpolation')
        ax.axis('off')
        
        return fig

class ImageDatasetPreprocessor:
    """
    Helper class for preprocessing image datasets
    """
    def __init__(self, img_size=(64, 64), normalize=True):
        self.img_size = img_size
        self.normalize = normalize
    
    def preprocess_batch(self, batch):
        """
        Preprocess a batch of images
        
        Args:
            batch: Tensor of shape [B, C, H, W]
            
        Returns:
            Preprocessed batch
        """
        if batch.shape[2:] != self.img_size:
            batch = F.interpolate(batch, size=self.img_size, mode='bilinear', align_corners=False)
        
        if self.normalize and batch.max() > 1.0:
            batch = batch / 255.0
        
        return batch
