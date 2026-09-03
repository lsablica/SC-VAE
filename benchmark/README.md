# Benchmark-only competitor implementations

These modules are retained only to reproduce the comparisons in the paper.
They are not imported by the `spherical_cauchy` package.

- `vendor_power_spherical.py` is based on Nicola De Cao and Wilker Aziz's
  [Power Spherical](https://github.com/nicola-decao/power_spherical) source at
  commit `3d4619a9d6c01bc9b427533d386271a233e304cd`. Its MIT license is reproduced
  in `POWER_SPHERICAL_LICENSE`. Compatibility changes preserve tensor device
  and dtype and handle the uniform-limit scale.
- `vendor_vmf.py` is the original SciPy-backed vMF baseline used for the
  recorded failure comparison.
- `vendor_vmf_robust.py` replaces the unstable special-function path with
  torch-native continued-fraction and log-space evaluations. It remains a
  benchmark baseline, not part of the installable distribution package.

The smallNORB-specific vMF repair stays under that experiment because its
parameterization is frozen with the final run and must not affect the MNIST or
latent-layer baselines.
