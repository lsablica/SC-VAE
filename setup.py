from setuptools import setup, find_packages

setup(
    name="spcauchy-vae",
    version="0.1.0",
    description="Spherical Cauchy Variational Autoencoder",
    author="Lukas Sablica",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
    ],
    python_requires=">=3.8",
)
