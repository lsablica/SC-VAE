from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from experiments.smiles.config import ExperimentConfig, MODEL_SPECS
from experiments.smiles.data import find_zinc250k_raw_file, prepare_zinc250k_dataset
from experiments.smiles.get_data import download_dataset


SCALE_PRESETS = {
    "full": {
        "seeds": [0, 1, 2, 3, 4],
        "epochs": 300,
        "batch_size": 384,
        "max_train_samples": None,
        "max_val_samples": None,
        "max_test_samples": None,
    },
    "paper_subset": {
        "seeds": [0, 1, 2],
        "epochs": 120,
        "batch_size": 384,
        "max_train_samples": 100_000,
        "max_val_samples": 10_000,
        "max_test_samples": 10_000,
    },
    "pilot": {
        "seeds": [0],
        "epochs": 20,
        "batch_size": 384,
        "max_train_samples": 25_000,
        "max_val_samples": 2_500,
        "max_test_samples": 2_500,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full ZINC-250k benchmark pipeline end-to-end.")
    parser.add_argument(
        "--scale",
        choices=sorted(SCALE_PRESETS),
        default="paper_subset",
        help="Benchmark preset. User-supplied seed/epoch/batch/subset flags override the preset.",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision on CUDA jobs. Leave off unless you have verified numerical stability.",
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["spcauchy-128", "gaussian-64", "gaussian-128"],
        default=["spcauchy-128", "gaussian-64", "gaussian-128"],
    )
    parser.add_argument("--run-id", default=None, help="Shared run id. Defaults to a UTC timestamp.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta-start", type=float, default=0.0)
    parser.add_argument("--beta-target", type=float, default=0.015)
    parser.add_argument("--beta-zero-epochs", type=int, default=1)
    parser.add_argument("--beta-warmup-epochs", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--spcauchy-rho-bias-init", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--max-smiles-length", type=int, default=68)
    parser.add_argument("--data-root", default="experiments/smiles/datasets/zinc250k/raw")
    parser.add_argument("--processed-root", default="experiments/smiles/datasets/zinc250k/processed")
    parser.add_argument("--runs-root", default="experiments/smiles/runs")
    parser.add_argument("--aggregated-root", default="experiments/smiles/aggregated")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-prior-samples", type=int, default=10000)
    parser.add_argument("--interpolation-pool-size", type=int, default=512)
    parser.add_argument("--interpolation-pairs-per-bin", type=int, default=25)
    parser.add_argument("--interpolation-steps", type=int, default=11)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-reprocess", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args()
    apply_scale_preset(args)
    return args


def apply_scale_preset(args: argparse.Namespace) -> None:
    preset = SCALE_PRESETS[args.scale]
    if args.seeds is None:
        args.seeds = list(preset["seeds"])
    if args.epochs is None:
        args.epochs = int(preset["epochs"])
    if args.batch_size is None:
        args.batch_size = int(preset["batch_size"])
    if args.max_train_samples is None:
        args.max_train_samples = preset["max_train_samples"]
    if args.max_val_samples is None:
        args.max_val_samples = preset["max_val_samples"]
    if args.max_test_samples is None:
        args.max_test_samples = preset["max_test_samples"]


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [run_all] {message}", flush=True)


def ensure_data(args: argparse.Namespace) -> None:
    raw_root = Path(args.data_root)
    if args.force_download:
        log("Force-download requested. Fetching the public ZINC-250k CSV.")
        download_dataset(raw_root, force=True)
    else:
        try:
            raw_path = find_zinc250k_raw_file(raw_root)
            log(f"Found existing ZINC-250k raw file at {raw_path}")
        except FileNotFoundError:
            log("ZINC-250k raw file missing. Downloading the public benchmark CSV.")
            download_dataset(raw_root)

    bundle = prepare_zinc250k_dataset(
        raw_dir=args.data_root,
        processed_dir=args.processed_root,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        max_smiles_length=args.max_smiles_length,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        force_reprocess=args.force_reprocess,
    )
    log(
        "Prepared dataset cache: "
        f"train={bundle.metadata['num_train']} val={bundle.metadata['num_val']} "
        f"test={bundle.metadata['num_test']} vocab={bundle.metadata['vocab_size']} "
        f"max_seq_len={bundle.metadata['max_seq_len']} max_smiles_length={bundle.metadata['max_smiles_length']}"
    )
    stats_path = Path(args.aggregated_root) / "latest_preprocessing_summary.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(bundle.metadata, indent=2), encoding="utf-8")


def run_command(command: list[str]) -> None:
    log("Running: " + " ".join(command))
    subprocess.run(command, check=True)


def resolve_run_root(args: argparse.Namespace, model_name: str, seed: int, run_id: str) -> Path:
    exp = ExperimentConfig(
        dataset_name="zinc250k",
        model_name=model_name,
        seed=seed,
        runs_root=args.runs_root,
        run_id=run_id,
        output_root=None,
    )
    return exp.resolve_output_root()


def build_train_command(args: argparse.Namespace, model_name: str, seed: int, run_id: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.smiles.train",
        "--dataset-name",
        "zinc250k",
        "--model-name",
        model_name,
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--beta-target",
        str(args.beta_target),
        "--beta-start",
        str(args.beta_start),
        "--beta-zero-epochs",
        str(args.beta_zero_epochs),
        "--beta-warmup-epochs",
        str(args.beta_warmup_epochs),
        "--validation-fraction",
        str(args.validation_fraction),
        "--test-fraction",
        str(args.test_fraction),
        "--split-seed",
        str(args.split_seed),
        "--max-smiles-length",
        str(args.max_smiles_length),
        "--data-root",
        args.data_root,
        "--processed-root",
        args.processed_root,
        "--runs-root",
        args.runs_root,
        "--run-id",
        run_id,
        "--embedding-dim",
        str(args.embedding_dim),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-heads",
        str(args.num_heads),
        "--num-layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--spcauchy-rho-bias-init",
        str(args.spcauchy_rho_bias_init),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.amp:
        command.append("--amp")
    for flag, value in [
        ("--max-train-samples", args.max_train_samples),
        ("--max-val-samples", args.max_val_samples),
        ("--max-test-samples", args.max_test_samples),
    ]:
        if value is not None:
            command.extend([flag, str(value)])
    return command


def build_eval_command(args: argparse.Namespace, checkpoint: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.smiles.evaluate",
        "--checkpoint",
        str(checkpoint),
        "--data-root",
        args.data_root,
        "--processed-root",
        args.processed_root,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--num-prior-samples",
        str(args.num_prior_samples),
        "--device",
        args.device,
    ]
    if args.amp:
        command.append("--amp")
    for flag, value in [
        ("--max-train-samples", args.max_train_samples),
        ("--max-val-samples", args.max_val_samples),
        ("--max-test-samples", args.max_test_samples),
    ]:
        if value is not None:
            command.extend([flag, str(value)])
    return command


def build_interpolate_command(
    args: argparse.Namespace,
    checkpoint: Path,
    *,
    reference_checkpoint: Path,
    pairs_file: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.smiles.interpolate",
        "--checkpoint",
        str(checkpoint),
        "--reference-checkpoint",
        str(reference_checkpoint),
        "--data-root",
        args.data_root,
        "--processed-root",
        args.processed_root,
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--pool-size",
        str(args.interpolation_pool_size),
        "--pairs-per-bin",
        str(args.interpolation_pairs_per_bin),
        "--steps",
        str(args.interpolation_steps),
        "--seed",
        "0",
        "--num-workers",
        str(args.num_workers),
    ]
    if args.amp:
        command.append("--amp")
    for flag, value in [
        ("--max-train-samples", args.max_train_samples),
        ("--max-val-samples", args.max_val_samples),
        ("--max-test-samples", args.max_test_samples),
    ]:
        if value is not None:
            command.extend([flag, str(value)])
    if pairs_file is not None:
        command.extend(["--pairs-file", str(pairs_file)])
    return command


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log(f"Starting full benchmark pipeline with run_id={run_id}")
    log(f"Scale={args.scale} Models={args.models} Seeds={args.seeds} Device={args.device}")
    log(
        "Effective training budget: "
        f"epochs={args.epochs} batch_size={args.batch_size} "
        f"train_cap={args.max_train_samples} val_cap={args.max_val_samples} "
        f"test_cap={args.max_test_samples}"
    )
    log(
        "Optimization defaults: "
        f"beta_start={args.beta_start} beta_target={args.beta_target} beta_zero_epochs={args.beta_zero_epochs} "
        f"beta_warmup_epochs={args.beta_warmup_epochs} spcauchy_rho_bias_init={args.spcauchy_rho_bias_init} "
        f"grad_clip_norm={args.grad_clip_norm} amp={args.amp} max_smiles_length={args.max_smiles_length}"
    )
    if args.device == "cuda" and not args.amp:
        log("AMP is disabled. This is the recommended default for the current SMILES benchmark unless ZINC pilots prove it stable.")

    ensure_data(args)

    checkpoints: dict[tuple[str, int], Path] = {}
    for seed in args.seeds:
        for model_name in args.models:
            output_root = resolve_run_root(args, model_name, seed, run_id)
            checkpoint = output_root / "checkpoints" / "best-val-elbo.pt"
            if args.skip_existing and checkpoint.exists():
                log(f"Skipping existing training run for model={model_name} seed={seed}")
            else:
                log(
                    f"Training model={model_name} seed={seed} "
                    f"(distribution={MODEL_SPECS[model_name].distribution_type}, latent_dim={MODEL_SPECS[model_name].latent_dim})"
                )
                run_command(build_train_command(args, model_name, seed, run_id))
            checkpoints[(model_name, seed)] = checkpoint

    for seed in args.seeds:
        for model_name in args.models:
            checkpoint = checkpoints[(model_name, seed)]
            log(f"Evaluating model={model_name} seed={seed}")
            run_command(build_eval_command(args, checkpoint))

    for seed in args.seeds:
        if ("spcauchy-128", seed) not in checkpoints:
            continue
        reference_checkpoint = checkpoints[("spcauchy-128", seed)]
        reference_run_root = reference_checkpoint.parents[1]
        reference_pairs = reference_run_root / "interpolation" / "selected_pairs.json"

        log(f"Running reference interpolation for seed={seed}")
        run_command(
            build_interpolate_command(
                args,
                reference_checkpoint,
                reference_checkpoint=reference_checkpoint,
                pairs_file=None,
            )
        )

        for model_name in args.models:
            if model_name == "spcauchy-128":
                continue
            checkpoint = checkpoints[(model_name, seed)]
            log(f"Running matched interpolation for model={model_name} seed={seed}")
            run_command(
                build_interpolate_command(
                    args,
                    checkpoint,
                    reference_checkpoint=reference_checkpoint,
                    pairs_file=reference_pairs,
                )
            )

    log("Aggregating benchmark outputs")
    run_command(
        [
            sys.executable,
            "-m",
            "experiments.smiles.aggregate",
            "--runs-root",
            args.runs_root,
            "--dataset-name",
            "zinc250k",
            "--output-dir",
            str(Path(args.aggregated_root) / run_id),
            "--eval-split",
            "test",
            "--run-id",
            run_id,
        ]
    )
    log(f"Full pipeline completed for run_id={run_id}")


if __name__ == "__main__":
    main()
