# Package API

## Distributions

`SphericalCauchy(loc, concentration, validate_args=None)` is a reparameterized
`torch.distributions.Distribution` with event shape `(ambient_dim,)`. The batch
shape broadcasts `loc.shape[:-1]` and the concentration shape. A trailing
singleton concentration dimension is accepted and canonicalized.

The distribution exposes `loc`, `concentration`/`rho`, `ambient_dim`,
`ball_parameter`, and `mode`, together with `rsample`, `log_prob`, `entropy`,
and `expand`. Mean and variance are not implemented.

`HypersphericalUniform(ambient_dim, batch_shape=..., device=..., dtype=...)`
provides the corresponding uniform prior with the same distribution semantics.

PyTorch's `kl_divergence` supports spherical Cauchy to uniform, uniform to
spherical Cauchy, pairwise spherical Cauchy, and uniform to uniform pairs.

## Functional routes

- `spherical_cauchy_kl`: exact finite/certified direct evaluation
- `spherical_cauchy_kl_fixed`: certified fixed-budget evaluation
- `spherical_cauchy_neighbor_kl`: finite odd-dimensional approximation with
  exact dispatch in supported cases
- `spherical_cauchy_laplace_kl`: constant-cost approximation with an analytic
  custom backward
- `spherical_cauchy_pairwise_kl`: exact pairwise KL from open-ball parameters
- `pseudohyperbolic_distance`: stable open-ball distance
- `mobius_transform`: deterministic spherical Cauchy transport
- `sample_uniform_sphere`: reparameterized uniform sphere sampling

Exact direct functions accept `backend="auto"`, `"vectorized"`, `"compiled"`,
or `"triton"`. Triton remains optional and lazy.

The Laplace function accepts `backend="auto"`, `"eager"`, `"compiled"`, or
`"triton"`. Automatic selection uses a compiled PyTorch CPU kernel and a fused
Triton CUDA kernel when available.
