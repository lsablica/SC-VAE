import os
import subprocess
import sys

import spherical_cauchy

EXPECTED_PUBLIC_API = {
    "HypersphericalUniform",
    "SphericalCauchy",
    "mobius_transform",
    "pseudohyperbolic_distance",
    "sample_uniform_sphere",
    "spherical_cauchy_kl",
    "spherical_cauchy_kl_fixed",
    "spherical_cauchy_laplace_kl",
    "spherical_cauchy_neighbor_kl",
    "spherical_cauchy_pairwise_kl",
}
FORBIDDEN = {
    "combined",
    "series_old",
    "hybrid",
    "midpoint",
    "asymptotic_high_rho",
    "direct_autograd",
}


def test_public_exports_are_intentional():
    assert set(spherical_cauchy.__all__) == EXPECTED_PUBLIC_API
    exported_lower = {name.lower() for name in dir(spherical_cauchy)}
    assert FORBIDDEN.isdisjoint(exported_lower)


def test_import_does_not_eagerly_import_triton():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src"
    command = "import sys; import spherical_cauchy; assert 'triton' not in sys.modules"
    subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=environment,
    )
