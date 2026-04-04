from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from experiments.smiles.data import find_zinc250k_raw_file, prepare_zinc250k_dataset
from experiments.smiles.get_data import download_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and preprocess ZINC-250k for the SMILES benchmark.")
    parser.add_argument("--data-root", default="experiments/smiles/datasets/zinc250k/raw")
    parser.add_argument("--processed-root", default="experiments/smiles/datasets/zinc250k/processed")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--max-smiles-length", type=int, default=68)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-reprocess", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    return parser.parse_args()


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [preprocess] {message}", flush=True)


def main() -> None:
    args = parse_args()
    raw_root = Path(args.data_root)
    if args.force_download:
        log("Force-download requested. Fetching the public ZINC-250k CSV.")
        download_dataset(raw_root)
    else:
        try:
            raw_path = find_zinc250k_raw_file(raw_root)
            log(f"Using existing ZINC-250k raw file at {raw_path}")
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
    summary_path = Path(args.processed_root) / "preprocessing_summary.json"
    summary_path.write_text(json.dumps(bundle.metadata, indent=2), encoding="utf-8")
    log(
        "Prepared dataset cache: "
        f"train={bundle.metadata['num_train']} val={bundle.metadata['num_val']} "
        f"test={bundle.metadata['num_test']} vocab={bundle.metadata['vocab_size']} "
        f"max_seq_len={bundle.metadata['max_seq_len']} max_smiles_length={bundle.metadata['max_smiles_length']}"
    )
    log(f"Saved preprocessing summary to {summary_path}")


if __name__ == "__main__":
    main()
