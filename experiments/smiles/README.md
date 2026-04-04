# SMILES Benchmark Pipeline

This directory contains the script-based molecular benchmark pipeline for the Section 5.4 SMILES experiment. The benchmark is now centered on `ZINC-250k`, with a deterministic random `80/10/10` train/validation/test split and matched baselines that keep the sequence backbone fixed across latent geometries.

## Dataset

The expected raw file is:

- `250k_rndm_zinc_drugs_clean_3.csv`

By default it lives in:

`experiments/smiles/datasets/zinc250k/raw`

Download it with:

```bash
python -m experiments.smiles.get_data
```

Or download and build the processed cache in one step:

```bash
python -m experiments.smiles.preprocess
```

The preprocessing pipeline:

- loads the single raw ZINC-250k CSV
- detects the SMILES column automatically
- canonicalizes molecules with RDKit
- drops invalid and duplicate molecules
- filters canonicalized SMILES to a maximum length of `68` characters before splitting
- creates a deterministic `80/10/10` split with split seed `13`
- builds the character vocabulary from the train split only
- caches token tensors plus processed CSV splits in:

`experiments/smiles/datasets/zinc250k/processed`

## Main Models

- `spcauchy-128`: spherical reference model
- `gaussian-64`: matched posterior-parameter budget baseline
- `gaussian-128`: matched latent-size baseline

Each run manifest records the fairness regime explicitly.

## Training

Train one model and one seed:

```bash
python -m experiments.smiles.train --model-name spcauchy-128 --seed 0
python -m experiments.smiles.train --model-name gaussian-64 --seed 0
python -m experiments.smiles.train --model-name gaussian-128 --seed 0
```

Useful overrides:

- `--data-root` and `--processed-root`
- `--output-root` to force a specific run directory
- `--epochs`, `--batch-size`, `--device`
- `--embedding-dim`, `--hidden-dim`, `--num-layers`, `--num-heads`, `--dropout`
- `--max-train-samples`, `--max-val-samples`, `--max-test-samples` for pilots

The benchmark currently defaults to:

- 300 epochs
- AdamW
- learning rate `1e-4`
- weight decay `0.01`
- batch size `256`
- KL schedule: 1 zero-KL epoch, then a linear ramp from `0.0` to `0.015` over 20 epochs, then fixed
- `spcauchy_rho_bias_init = 0.0`
- gradient clipping with `max_norm = 1.0`
- AMP disabled by default
- maximum canonical SMILES length `68`

Each run writes:

- `checkpoints/best-val-elbo.pt`
- `checkpoints/last.pt`
- `metrics/train_history.csv`
- `metrics/train_history.json`
- `run_manifest.json`

## Evaluation

Evaluate reconstruction and prior-sample metrics from the best checkpoint:

```bash
python -m experiments.smiles.evaluate --checkpoint experiments/smiles/runs/zinc250k/spcauchy-128/latent_128/seed_0/<run_id>/checkpoints/best-val-elbo.pt
```

Saved outputs include:

- reconstruction ELBO / recon / KL
- exact reconstruction accuracy
- token-level reconstruction accuracy
- canonical molecule reconstruction accuracy
- prior-sample validity / uniqueness / novelty / internal diversity
- RDKit property summaries and Wasserstein distances against the held-out test split
- generated molecules CSV

## Interpolation Analysis

Run the spherical-vs-Euclidean interpolation analysis:

```bash
python -m experiments.smiles.interpolate --checkpoint <run>/checkpoints/best-val-elbo.pt
```

Important behavior:

- endpoint pairs are selected from the ZINC test split using the reference `spCauchy` latent directions
- the same pair file can be reused across baselines with `--pairs-file`
- `spCauchy` uses SLERP on the unit sphere
- Gaussian baselines use linear interpolation between latent means

Saved outputs include:

- `interpolation/selected_pairs.json`
- `interpolation/interpolation_steps.csv`
- `interpolation/interpolation_summary.csv`
- `tables/interpolation_by_bin.csv`

The analysis reports per-path validity, fully-valid-path rate, uniqueness, novelty, endpoint fingerprint trajectories, property trajectories, and a simple smoothness score.

## Aggregation And Plots

Aggregate multiple seeds after training/evaluation/interpolation runs are available:

```bash
python -m experiments.smiles.aggregate --runs-root experiments/smiles/runs --dataset-name zinc250k --output-dir experiments/smiles/aggregated
```

The aggregator writes:

- `benchmark_seed_metrics.csv`
- `benchmark_mean_std.csv`
- `interpolation_seed_metrics.csv`
- `interpolation_mean_std.csv`
- training curve plots
- property distribution plots
- interpolation summary plots
- representative interpolation figures per run

Use `--run-id` to aggregate one exact experiment family.

## End-To-End Runs

Use `run_all.py` for download + preprocessing + training + evaluation + interpolation + aggregation.

Recommended practical default:

```bash
python -m experiments.smiles.run_all --device cuda
```

`run_all.py` defaults to `--scale paper_subset`, which means:

- seeds: `0 1 2`
- epochs: `120`
- batch size: `384`
- train cap: `100000`
- val cap: `10000`
- test cap: `10000`

Available presets:

- `--scale pilot`
  - `25k / 2.5k / 2.5k`
  - `1` seed
  - `20` epochs
- `--scale paper_subset`
  - `100k / 10k / 10k`
  - `3` seeds
  - `120` epochs
- `--scale full`
  - full ZINC-250k split
  - `5` seeds
  - `300` epochs

Any explicit command-line override such as `--epochs`, `--batch-size`, or `--max-train-samples` takes precedence over the preset.

## Slurm

Cluster launch templates are included in:

- `experiments/smiles/slurm/preprocess_zinc250k.sbatch`
- `experiments/smiles/slurm/train_single_zinc250k.sbatch`
- `experiments/smiles/slurm/finalize_zinc250k.sbatch`
- `experiments/smiles/slurm/submit_preprocess_single.sh`
- `experiments/smiles/slurm/submit_train_single.sh`
- `experiments/smiles/slurm/submit_finalize_single.sh`

The hardcoded cluster scripts target:

- data root: `~/projects/SC-VAE-data/zinc250k/raw`
- processed root: `~/projects/SC-VAE-data/zinc250k/processed`
- runs root: `~/projects/SC-VAE-runs-zinc`
- aggregated root: `~/projects/SC-VAE-aggregated-zinc`

## Output Layout

Default run layout:

`experiments/smiles/runs/{dataset}/{model}/latent_{d}/seed_{seed}/{run_id}/`

Subdirectories:

- `checkpoints`
- `metrics`
- `samples`
- `interpolation`
- `plots`
- `tables`

## Notes

- The main decode policy is deterministic greedy decoding for reproducibility.
- The implementation is RDKit-first and does not require `guacamol` or `fcd_torch`.
- The benchmark assumes ZINC-250k only; MOSES is intentionally not part of this pipeline anymore.
