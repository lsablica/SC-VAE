"""Default grids and method selections for the latent-layer benchmark."""

DEFAULT_ACCURACY_DIMS = [2, 3, 4, 5, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
DEFAULT_RUNTIME_DIMS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
DEFAULT_SPCAUCHY_RHO_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995]
DEFAULT_VMF_KAPPA_GRID = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

DEFAULT_SPCAUCHY_ACCURACY_METHODS = [
    "series",
    "combined",
    "asymptotic_high_rho",
    "hybrid",
]
DEFAULT_SPCAUCHY_RUNTIME_METHODS = [
    "spcauchy_combined",
    "spcauchy_hybrid",
]
DEFAULT_VMF_RUNTIME_METHODS = [
    "vmf_official",
    "vmf_robust",
]
DEFAULT_SPCAUCHY_ROBUSTNESS_METHODS = [
    "series",
    "combined",
    "asymptotic_high_rho",
    "hybrid",
]
DEFAULT_VMF_ROBUSTNESS_METHODS = [
    "vmf_official",
    "vmf_robust",
]
