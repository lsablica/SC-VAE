# Spherical Cauchy Variational Autoencoder (SC-VAE) 

![SC-VAE](https://img.shields.io/badge/SC--VAE-v0.1-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A novel variational autoencoder (VAE) architecture that employs a **spherical Cauchy** (spCauchy) latent distribution. This approach provides a numerically stable alternative to common distributions like Gaussian or von Mises-Fisher (vMF). 
 
## Features 

- **Hyperspherical Latent Space**: Naturally represents directional or cyclic data
- **Numerical Stability**: Avoids issues with modified Bessel functions found in vMF distributions
- **Closed-Form KL Divergence**: Efficient computation via rapidly convergent Gauss hypergeometric series
- **Differentiable Sampling**: Clean implementation of the reparameterization trick using Möbius transformations
- **Multiple Architectures**: Support for MLP, CNN, and Transformer-based encoders and decoders

## Installation

```bash
git clone https://github.com/lukassablica/SC-VAE.git
cd SC-VAE

pip install -r requirements.txt
```

## Quick Start

### Training a model

```python
from src.model import SpCauchyVAE
from src.config import SpCauchyVAEConfig
from src.trainer import Trainer

# Configure the model
config = SpCauchyVAEConfig(
    input_dim=[1, 28, 28],  # For MNIST
    latent_dim=20,
    encoder_type="cnn",
    decoder_type="cnn"
)

# Create the model
model = SpCauchyVAE(config)

# Train the model
trainer = Trainer(
    model=model,
    dataset="mnist",
    batch_size=128,
    learning_rate=1e-3,
    num_epochs=100
)
trainer.train()
```

### Command-line helper

You can also use the lightweight CLI in `main.py`:

```bash
# Train
python main.py train --dataset mnist --latent-dim 3 --hidden-dims 32 64 128 --num-epochs 2

# Generate samples from a checkpoint
python main.py generate --checkpoint checkpoints/best_model.pt

# Reconstruct a few examples
python main.py reconstruct --checkpoint checkpoints/best_model.pt --dataset mnist
```

### Training with a transformer architecture

```python
# Configure a transformer-based model
config = SpCauchyVAEConfig(
    input_dim=[1, 28, 28],  # For MNIST
    latent_dim=20,
    encoder_type="transformer",
    decoder_type="transformer",
    num_heads=8,
    num_layers=4,
    dropout=0.1
)

# Create and train the model
model = SpCauchyVAE(config)
# ... rest of training code
```

### Generating samples

```python
# Load a pretrained model
checkpoint_path = "path/to/checkpoint.pt"
model = SpCauchyVAE.load_from_checkpoint(checkpoint_path)

# Generate samples
samples = model.generate_samples(num_samples=16)
```

## Mathematical Background

SC-VAE employs the spherical Cauchy distribution as a latent prior. This distribution is defined on the unit sphere S^(d-1) and is constructed via a Möbius transformation of uniformly distributed points on the sphere:

p(x|μ,ρ) ∝ (1-ρ²)^(d-1) / (1+ρ²-2ρμᵀx)^(d-1)

Where:
- x ∈ S^(d-1) is a point on the unit sphere
- μ ∈ S^(d-1) is the mean direction
- ρ ∈ (0,1) is the concentration parameter

The KL divergence with the uniform prior is computed using a Gauss hypergeometric function with argument in (0,1), ensuring rapid convergence and numerical stability.

## Results

SC-VAE has been tested on various datasets including MNIST, SMILES and directional data. It shows particular advantages for:

- Data with inherent cyclical or directional structure
- High-dimensional latent spaces where numerical stability is crucial
- Applications requiring interpretable latent representations

## Documentation

- Paper: https://arxiv.org/abs/2506.21278
- Examples: `examples/` (MNIST tutorial, SMILES)
- Benchmarks: `benchmark/` (spCauchy vs vMF stress tests)


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{sablica2025scvae,
      title={Hyperspherical Variational Autoencoders Using Efficient Spherical Cauchy Distribution}, 
      author={Lukas Sablica and Kurt Hornik},
      year={2025},
      eprint={2506.21278},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      doi={10.48550/arXiv.2506.21278},
      url={https://arxiv.org/abs/2506.21278}, 
}
```
