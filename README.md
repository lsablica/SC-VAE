# Spherical Cauchy for PyTorch

`spherical_cauchy` provides reparameterized spherical Cauchy distributions and
their exact KL divergence in an idiomatic PyTorch API. The density is supported
on a unit hypersphere, while location and concentration are represented by a
unit direction and a scalar `rho` in `[0, 1)`.

## Install

From a checkout:

```bash
git clone https://github.com/lsablica/SC-VAE.git
cd SC-VAE
python -m pip install .
```

Install the development and experiment dependencies with:

```bash
python -m pip install '.[dev,experiments]'
```

## Distribution API

```python
import torch
from torch.distributions import kl_divergence
from spherical_cauchy import HypersphericalUniform, SphericalCauchy

loc = torch.nn.functional.normalize(torch.randn(32, 33), dim=-1)
rho = torch.full((32,), 0.7)
q = SphericalCauchy(loc, rho)
p = HypersphericalUniform(33, batch_shape=q.batch_shape)

z = q.rsample()
log_qz = q.log_prob(z)
kl_to_uniform = kl_divergence(q, p)
```

Pairwise KL is registered with PyTorch as well:

```python
q1 = SphericalCauchy(loc, rho)
q2 = SphericalCauchy(-loc, 0.4)
pairwise_kl = kl_divergence(q1, q2)
```

See [`examples/`](examples/) and [`docs/package_api.md`](docs/package_api.md)
for complete runnable examples.


## Repository layout

- `src/spherical_cauchy/`: installable PyTorch package
- `benchmark/`: benchmark-only competitor implementations and licenses
- `experiments/`: final experiment code and compact artifacts
- `tests/`: distribution, numerical, public API, and result checks

## Citation and license

Citation metadata is in [`CITATION.cff`](CITATION.cff). The code is available
under the [MIT License](LICENSE); vendored benchmark code retains its own
attribution and license notice under [`benchmark/`](benchmark/).
