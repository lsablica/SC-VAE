# MNIST Experiment Report

## Scope

This folder contains two MNIST workflows for Section 5.3:

- A dedicated qualitative `S^2` visualization run for figures.
- A multi-seed benchmark comparison across `gaussian`, `vmf`, and `spcauchy`.

All runs use the same CNN backbone:

- hidden dims: `[32, 64, 128]`
- activation: `ReLU`
- dropout: `0.1`
- optimizer: `AdamW`
- scheduler: `ReduceLROnPlateau(factor=0.5, patience=5)`
- batch size: `128`
- KL weight: `1.0`
- train split: official MNIST train
- eval split: official MNIST test
- checkpoint selection: lowest `eval_recon_loss`

## Qualitative `S^2` Run

Purpose: figure generation for the low-dimensional spherical latent space.

- preset: `qualitative_s2`
- model: `spcauchy`
- reported dim: `2`
- ambient latent dim: `3`
- seed: `1`
- epochs: `50`
- learning rate: `5e-4`
- warmup steps: `0`

Selected checkpoint:

- selected epoch: `47`
- selected eval recon: `128.6037`
- selected eval total: `136.4512`
- selected eval KL: `7.8475`

Epoch-40 reference values for direct comparison to the 40-epoch benchmark runs:

- eval recon at epoch 40: `129.6483`
- eval total at epoch 40: `137.4532`
- eval KL at epoch 40: `7.8049`

## Benchmark Protocol

Purpose: quantitative cross-model comparison.

- preset: `benchmark_comparison`
- models: `gaussian`, `vmf`, `spcauchy`
- reported dims: `{2, 3, 5, 10, 20}`
- seeds: `{1, 2, 3, 4, 5}`
- epochs: `40`

Dimension-specific benchmark settings:

- reported dims `2, 3`: learning rate `3e-4`, warmup `200` optimizer steps
- reported dims `5, 10, 20`: learning rate `1e-4`, warmup `0`

Latent-space convention:

- Gaussian: ambient latent dim = reported dim
- vMF and spCauchy: ambient latent dim = reported dim + 1

## Benchmark Results

Metrics below are aggregated over 5 seeds and reported as mean ± std from the best-eval-reconstruction checkpoint.

| Model | Reported Dim | Ambient Dim | Eval Recon | Eval Total | Eval KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| gaussian | 2 | 2 | 132.7879 ± 0.3613 | 139.7490 ± 0.3838 | 6.9611 ± 0.1207 |
| spcauchy | 2 | 3 | 130.1811 ± 0.6347 | 137.8593 ± 0.6009 | 7.6782 ± 0.0616 |
| vmf | 2 | 3 | 131.3819 ± 0.6982 | 138.1959 ± 0.6461 | 6.8140 ± 0.0592 |
| gaussian | 3 | 3 | 118.9629 ± 0.8584 | 128.1316 ± 0.8319 | 9.1688 ± 0.1625 |
| spcauchy | 3 | 4 | 116.4215 ± 0.4698 | 126.1765 ± 0.4747 | 9.7550 ± 0.1222 |
| vmf | 3 | 4 | 118.1335 ± 0.6299 | 127.1682 ± 0.5836 | 9.0346 ± 0.0498 |
| gaussian | 5 | 5 | 102.0180 ± 1.2672 | 114.6440 ± 1.1262 | 12.6260 ± 0.1439 |
| spcauchy | 5 | 6 | 100.0110 ± 0.4645 | 113.2527 ± 0.4323 | 13.2417 ± 0.0408 |
| vmf | 5 | 6 | 104.6356 ± 0.3483 | 116.9024 ± 0.3115 | 12.2668 ± 0.0584 |
| gaussian | 10 | 10 | 81.2162 ± 0.2902 | 100.6140 ± 0.2137 | 19.3977 ± 0.2171 |
| spcauchy | 10 | 11 | 79.9697 ± 0.3551 | 100.3966 ± 0.2757 | 20.4269 ± 0.0866 |
| vmf | 10 | 11 | 83.8876 ± 0.2726 | 103.4043 ± 0.2690 | 19.5167 ± 0.0570 |
| gaussian | 20 | 20 | 74.9125 ± 0.4550 | 98.0599 ± 0.2241 | 23.1475 ± 0.3123 |
| spcauchy | 20 | 21 | 74.5688 ± 0.1511 | 101.8710 ± 0.2846 | 27.3021 ± 0.1802 |
| vmf | 20 | 21 | 77.0952 ± 0.2850 | 103.5171 ± 0.2554 | 26.4219 ± 0.0629 |

## Headline Comparison

Primary ranking metric: validation/eval reconstruction loss.

- `d=2`: `spcauchy` best
- `d=3`: `spcauchy` best
- `d=5`: `spcauchy` best
- `d=10`: `spcauchy` best
- `d=20`: `spcauchy` best

## Useful Figure Runs

Dedicated qualitative figure run:

- `outputs/qualitative/spcauchy_s2/seed_1`

Benchmark `spcauchy` figure-capable runs on `S^2`:

- `outputs/benchmark/spcauchy/dim_2/seed_1`
- `outputs/benchmark/spcauchy/dim_2/seed_2`
- `outputs/benchmark/spcauchy/dim_2/seed_3`
- `outputs/benchmark/spcauchy/dim_2/seed_4`
- `outputs/benchmark/spcauchy/dim_2/seed_5`

Best `spcauchy` benchmark seed at reported dim `2` by eval reconstruction:

- seed `3`
- best eval recon: `129.5370`
- best eval total: `137.2839`
- best eval KL: `7.7470`
- selected epoch: `36`

## Artifact Sources

Main source files used for the numbers above:

- `outputs/aggregate/benchmark_summary.csv`
- `outputs/aggregate/benchmark_seed_level.csv`
- `outputs/qualitative/spcauchy_s2/seed_1/selection_summary.json`
- `outputs/qualitative/spcauchy_s2/seed_1/history.csv`
