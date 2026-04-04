from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
from tqdm import tqdm

from experiments.smiles.chemistry import canonicalize_smiles, export_property_histograms, property_distance_summary, summarize_generation
from experiments.smiles.data import build_dataloader, load_split_frame, prepare_zinc250k_dataset
from experiments.smiles.decoding import deterministic_recon_logits, logits_to_smiles_batch
from experiments.smiles.model_factory import load_model_from_checkpoint
from src.utils import set_all_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ZINC-250k SMILES VAE run.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="experiments/smiles/datasets/zinc250k/raw")
    parser.add_argument("--processed-root", default="experiments/smiles/datasets/zinc250k/processed")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-prior-samples", type=int, default=10000)
    parser.add_argument("--output-root")
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision on CUDA.")
    parser.add_argument("--force-reprocess", action="store_true")
    parser.add_argument("--max-smiles-length", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [evaluate] {message}", flush=True)


def reconstruction_metrics(model, loader, idx_to_token: dict[int, str], device: str, amp_enabled: bool) -> tuple[dict, pd.DataFrame]:
    total_examples = 0
    exact_matches = 0
    canonical_matches = 0
    token_correct = 0
    token_total = 0
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    rows = []

    model.eval()
    for batch in tqdm(loader, desc="reconstruction", leave=False):
        token_ids = batch["token_ids"].to(device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits, mu, second_param = deterministic_recon_logits(model, token_ids)
                loss, recon, kl = model.loss_function(token_ids, logits, mu, second_param)
        decoded_smiles = logits_to_smiles_batch(logits.cpu(), idx_to_token)
        predicted_token_ids = torch.argmax(logits, dim=-1)
        mask = token_ids != model.config.pad_token_id
        token_correct += int(((predicted_token_ids == token_ids) & mask).sum().item())
        token_total += int(mask.sum().item())
        batch_size = token_ids.size(0)
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_recon += float(recon.item()) * batch_size
        total_kl += float(kl.item()) * batch_size

        for original, decoded in zip(batch["canonical_smiles"], decoded_smiles):
            original_canonical = canonicalize_smiles(original)
            decoded_canonical = canonicalize_smiles(decoded)
            exact_match = original == decoded
            canonical_match = original_canonical is not None and original_canonical == decoded_canonical
            exact_matches += int(exact_match)
            canonical_matches += int(canonical_match)
            rows.append(
                {
                    "original_smiles": original,
                    "decoded_smiles": decoded,
                    "exact_match": exact_match,
                    "canonical_match": canonical_match,
                    "decoded_is_valid": decoded_canonical is not None,
                }
            )

    metrics = {
        "num_examples": float(total_examples),
        "exact_reconstruction_accuracy": (exact_matches / total_examples) if total_examples else 0.0,
        "token_reconstruction_accuracy": (token_correct / token_total) if token_total else 0.0,
        "canonical_reconstruction_accuracy": (canonical_matches / total_examples) if total_examples else 0.0,
        "reconstruction_loss": (total_recon / total_examples) if total_examples else 0.0,
        "kl_divergence": (total_kl / total_examples) if total_examples else 0.0,
        "elbo": (total_loss / total_examples) if total_examples else 0.0,
    }
    return metrics, pd.DataFrame(rows)


def sample_prior_smiles(model, idx_to_token: dict[int, str], num_samples: int, device: str, batch_size: int = 512, amp_enabled: bool = False) -> list[str]:
    decoded = []
    model.eval()
    remaining = num_samples
    with torch.no_grad():
        while remaining > 0:
            current = min(batch_size, remaining)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model.generate_samples(num_samples=current, device=device)
            decoded.extend(logits_to_smiles_batch(logits.cpu(), idx_to_token))
            remaining -= current
    return decoded


def main() -> None:
    args = parse_args()
    model, model_config, checkpoint = load_model_from_checkpoint(args.checkpoint, device=args.device)
    amp_enabled = bool(args.amp and model_config.device == "cuda")
    log(f"Loaded checkpoint {Path(args.checkpoint).resolve()}")
    manifest = checkpoint.get("manifest", {})
    set_all_seeds(int(manifest.get("seed", 0)))
    max_train_samples = args.max_train_samples if args.max_train_samples is not None else manifest.get("max_train_samples")
    max_val_samples = args.max_val_samples if args.max_val_samples is not None else manifest.get("max_val_samples")
    max_test_samples = args.max_test_samples if args.max_test_samples is not None else manifest.get("max_test_samples")
    max_smiles_length = args.max_smiles_length if args.max_smiles_length is not None else manifest.get("max_smiles_length", 68)
    test_fraction = float(manifest.get("test_fraction", 0.1))
    split_seed = int(manifest.get("split_seed", 13))

    bundle = prepare_zinc250k_dataset(
        raw_dir=args.data_root,
        processed_dir=args.processed_root,
        validation_fraction=float(manifest.get("validation_fraction", 0.1)),
        test_fraction=test_fraction,
        split_seed=split_seed,
        max_smiles_length=int(max_smiles_length),
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
        force_reprocess=args.force_reprocess,
    )
    output_root = Path(args.output_root) if args.output_root else Path(args.checkpoint).resolve().parents[1]
    metrics_dir = output_root / "metrics"
    samples_dir = output_root / "samples"
    tables_dir = output_root / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    loader = build_dataloader(bundle, args.split, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    reconstruction, reconstruction_rows = reconstruction_metrics(
        model,
        loader,
        bundle.vocabulary.idx_to_token,
        model_config.device,
        amp_enabled,
    )
    reconstruction_rows.to_csv(metrics_dir / f"reconstruction_{args.split}.csv", index=False)
    log(
        "Reconstruction metrics: "
        f"exact={reconstruction['exact_reconstruction_accuracy']:.4f} "
        f"token={reconstruction['token_reconstruction_accuracy']:.4f} "
        f"canonical={reconstruction['canonical_reconstruction_accuracy']:.4f} "
        f"elbo={reconstruction['elbo']:.4f}"
    )

    generated_smiles = sample_prior_smiles(
        model,
        bundle.vocabulary.idx_to_token,
        args.num_prior_samples,
        model_config.device,
        amp_enabled=amp_enabled,
    )
    train_reference = set(load_split_frame(bundle, "train")["canonical_smiles"].tolist())
    generation_metrics, generation_rows = summarize_generation(generated_smiles, train_reference, seed=int(manifest.get("seed", 0)))
    generation_rows.to_csv(samples_dir / "prior_samples.csv", index=False)
    log(
        "Generation metrics: "
        f"validity={generation_metrics['validity']:.4f} "
        f"uniqueness={generation_metrics['uniqueness']:.4f} "
        f"novelty={generation_metrics['novelty']:.4f} "
        f"intdiv={generation_metrics['internal_diversity']:.4f}"
    )

    reference_smiles = load_split_frame(bundle, args.split)["canonical_smiles"].tolist()
    valid_generated = generation_rows.loc[generation_rows["is_valid"], "canonical_smiles"].dropna().tolist()
    property_metrics, generated_properties, reference_properties = property_distance_summary(valid_generated, reference_smiles)
    generated_properties.to_csv(metrics_dir / "generated_properties.csv", index=False)
    reference_properties.to_csv(metrics_dir / f"reference_properties_{args.split}.csv", index=False)
    export_property_histograms(generated_properties, reference_properties, tables_dir / f"property_histograms_{args.split}.csv")

    metrics = {
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "reconstruction": reconstruction,
        "generation": generation_metrics,
        "property_distribution": property_metrics,
    }
    save_json(metrics_dir / f"eval_{args.split}.json", metrics)
    log(f"Saved evaluation outputs to {output_root}")


if __name__ == "__main__":
    main()
