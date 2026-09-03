"""Default grids and method selections for the latent-layer benchmark."""

DEFAULT_ACCURACY_DIMS = [2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 32, 33, 64, 128, 256, 512, 1024, 2048]
DEFAULT_RUNTIME_DIMS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
DEFAULT_NEIGHBOR_RUNTIME_DIMS = [9, 17, 33, 65, 129, 257, 513, 1025, 2049]
DEFAULT_SPCAUCHY_RHO_GRID = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
DEFAULT_VMF_KAPPA_GRID = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

DEFAULT_SPCAUCHY_ACCURACY_METHODS = [
    "direct",
    "neighbor",
    "laplace",
]
DEFAULT_SPCAUCHY_RUNTIME_METHODS = [
    "spcauchy_direct",
    "spcauchy_direct_fixed",
    "spcauchy_neighbor",
    "spcauchy_laplace",
    "spcauchy_direct_autograd",
]
DEFAULT_VMF_RUNTIME_METHODS = [
    "vmf_official",
    "vmf_robust",
]
DEFAULT_POWER_RUNTIME_METHODS = ["power_spherical"]
DEFAULT_SPCAUCHY_ROBUSTNESS_METHODS = [
    "direct",
    "neighbor",
    "laplace",
]
DEFAULT_VMF_ROBUSTNESS_METHODS = [
    "vmf_official",
    "vmf_robust",
]
