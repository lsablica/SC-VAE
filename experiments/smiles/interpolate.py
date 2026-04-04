from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from experiments.smiles.chemistry import canonicalize_smiles, compute_properties, tanimoto_similarity
from experiments.smiles.data import build_dataloader, load_split_frame, prepare_zinc250k_dataset
from experiments.smiles.decoding import logits_to_smiles_batch
from experiments.smiles.model_factory import load_model_from_checkpoint
from src.utils import set_all_seeds


BIN_DEFINITIONS = {
    "near": lambda cosine: cosine > 0.7,
    "medium": lambda cosine: 0.2 <= cosine <= 0.5,
    "orthogonal": lambda cosine: -0.1 <= cosine <= 0.1,
    "opposite": lambda cosine: cosine <= 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interpolation analysis for a trained ZINC-250k SMILES model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-checkpoint")
    parser.add_argument("--data-root", default="experiments/smiles/datasets/zinc250k/raw")
    parser.add_argument("--processed-root", default="experiments/smiles/datasets/zinc250k/processed")
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument("--pairs-per-bin", type=int, default=25)
    parser.add_argument("--steps", type=int, default=11)
    parser.add_argument("--pairs-file")
    parser.add_argument("--output-root")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision on CUDA.")
    parser.add_argument("--num-workers", type=int, default=0)
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
    print(f"[{timestamp}] [interpolate] {message}", flush=True)


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, dot_threshold: float = 0.9995) -> torch.Tensor:
    v0 = F.normalize(v0, dim=-1)
    v1 = F.normalize(v1, dim=-1)
    dot = torch.clamp(torch.sum(v0 * v1, dim=-1, keepdim=True), -1.0, 1.0)
    if torch.all(torch.abs(dot) > dot_threshold):
        return F.normalize((1.0 - t) * v0 + t * v1, dim=-1)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    return torch.sin((1.0 - t) * omega) / sin_omega * v0 + torch.sin(t * omega) / sin_omega * v1


def encode_split(model, loader, device: str, amp_enabled: bool) -> tuple[list[str], torch.Tensor]:
    smiles = []
    latents = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="encode", leave=False):
            token_ids = batch["token_ids"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                mu, _ = model.encode(token_ids)
            smiles.extend(batch["canonical_smiles"])
            latents.append(mu.cpu())
    return smiles, torch.cat(latents, dim=0)


def build_pair_file(
    reference_smiles: list[str],
    reference_latents: torch.Tensor,
    *,
    output_path: Path,
    pool_size: int,
    pairs_per_bin: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    candidate_count = min(pool_size, len(reference_smiles))
    if candidate_count < 2:
        raise ValueError("Need at least two reference molecules to form interpolation pairs.")
    if len(reference_smiles) > candidate_count:
        selected_indices = np.sort(rng.choice(len(reference_smiles), size=candidate_count, replace=False))
    else:
        selected_indices = np.arange(candidate_count)

    selected_latents = F.normalize(reference_latents[selected_indices], dim=1)
    cosine_matrix = torch.matmul(selected_latents, selected_latents.T).numpy()

    pairs = []
    for bin_name, predicate in BIN_DEFINITIONS.items():
        candidates = []
        for i in range(candidate_count):
            for j in range(i + 1, candidate_count):
                cosine = float(cosine_matrix[i, j])
                if predicate(cosine):
                    candidates.append((selected_indices[i], selected_indices[j], cosine))
        rng.shuffle(candidates)
        for idx_a, idx_b, cosine in candidates[:pairs_per_bin]:
            pairs.append(
                {
                    "bin": bin_name,
                    "index_a": int(idx_a),
                    "index_b": int(idx_b),
                    "smiles_a": reference_smiles[int(idx_a)],
                    "smiles_b": reference_smiles[int(idx_b)],
                    "reference_cosine": cosine,
                }
            )
    save_json(output_path, {"pairs": pairs})
    return pairs


def load_or_create_pairs(
    *,
    pairs_file: str | None,
    output_root: Path,
    reference_smiles: list[str],
    reference_latents: torch.Tensor,
    pool_size: int,
    pairs_per_bin: int,
    seed: int,
) -> list[dict]:
    if pairs_file:
        return json.loads(Path(pairs_file).read_text(encoding="utf-8"))["pairs"]
    selected_path = output_root / "interpolation" / "selected_pairs.json"
    if selected_path.exists():
        return json.loads(selected_path.read_text(encoding="utf-8"))["pairs"]
    return build_pair_file(
        reference_smiles,
        reference_latents,
        output_path=selected_path,
        pool_size=pool_size,
        pairs_per_bin=pairs_per_bin,
        seed=seed,
    )


def interpolate_latents(model_name: str, z_a: torch.Tensor, z_b: torch.Tensor, steps: int) -> torch.Tensor:
    ts = torch.linspace(0.0, 1.0, steps, device=z_a.device)
    if model_name.startswith("spcauchy"):
        return torch.cat([slerp(z_a, z_b, float(t.item())) for t in ts], dim=0)
    return torch.cat([(1.0 - t) * z_a + t * z_b for t in ts], dim=0)


def smoothness_score(sim_to_a: list[float | None], sim_to_b: list[float | None]) -> float:
    comparisons = []
    for first, second in zip(sim_to_a[:-1], sim_to_a[1:]):
        if first is not None and second is not None:
            comparisons.append(float(second <= first))
    for first, second in zip(sim_to_b[:-1], sim_to_b[1:]):
        if first is not None and second is not None:
            comparisons.append(float(second >= first))
    return float(np.mean(comparisons)) if comparisons else 0.0


def smiles_to_token_tensor(smiles: str, bundle, device: str) -> torch.Tensor:
    token_ids = [bundle.vocabulary.sos_token_id]
    token_ids.extend(bundle.vocabulary.token_to_idx[ch] for ch in smiles)
    token_ids.append(bundle.vocabulary.eos_token_id)
    token_ids.extend([bundle.vocabulary.pad_token_id] * (bundle.max_seq_len - len(token_ids)))
    return torch.tensor(token_ids, dtype=torch.long, device=device)


def evaluate_paths(
    *,
    model,
    model_name: str,
    pairs: list[dict],
    bundle,
    split_name: str,
    device: str,
    steps: int,
    train_reference: set[str],
    amp_enabled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_smiles = load_split_frame(bundle, split_name)["canonical_smiles"].tolist()
    idx_to_token = bundle.vocabulary.idx_to_token
    step_rows = []
    summary_rows = []

    model.eval()
    with torch.no_grad():
        for pair_id, pair in enumerate(tqdm(pairs, desc="interpolate", leave=False)):
            endpoint_a = split_smiles[pair["index_a"]]
            endpoint_b = split_smiles[pair["index_b"]]
            tokens = torch.stack(
                [
                    smiles_to_token_tensor(endpoint_a, bundle, device),
                    smiles_to_token_tensor(endpoint_b, bundle, device),
                ],
                dim=0,
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                mu, _ = model.encode(tokens)
            path_latents = interpolate_latents(model_name, mu[0:1], mu[1:2], steps)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model.decode(path_latents)
            decoded_smiles = logits_to_smiles_batch(logits.cpu(), idx_to_token)

            valid_flags = []
            novel_flags = []
            sim_to_a = []
            sim_to_b = []
            unique_valid = set()

            for step_idx, decoded in enumerate(decoded_smiles):
                canonical = canonicalize_smiles(decoded)
                is_valid = canonical is not None
                is_novel = bool(is_valid and canonical not in train_reference)
                if canonical is not None:
                    unique_valid.add(canonical)
                sim_a = tanimoto_similarity(endpoint_a, decoded)
                sim_b = tanimoto_similarity(endpoint_b, decoded)
                properties = compute_properties(decoded) if is_valid else None
                row = {
                    "pair_id": pair_id,
                    "bin": pair["bin"],
                    "step": step_idx,
                    "endpoint_a": endpoint_a,
                    "endpoint_b": endpoint_b,
                    "decoded_smiles": decoded,
                    "canonical_smiles": canonical,
                    "is_valid": is_valid,
                    "is_novel": is_novel,
                    "sim_to_a": sim_a,
                    "sim_to_b": sim_b,
                    "reference_cosine": pair["reference_cosine"],
                }
                if properties:
                    row.update(properties)
                step_rows.append(row)
                valid_flags.append(is_valid)
                novel_flags.append(is_novel)
                sim_to_a.append(sim_a)
                sim_to_b.append(sim_b)

            valid_count = sum(valid_flags)
            summary_rows.append(
                {
                    "pair_id": pair_id,
                    "bin": pair["bin"],
                    "reference_cosine": pair["reference_cosine"],
                    "valid_fraction": valid_count / len(valid_flags) if valid_flags else 0.0,
                    "fully_valid_path": float(all(valid_flags)),
                    "path_uniqueness": len(unique_valid) / valid_count if valid_count else 0.0,
                    "path_novelty": sum(novel_flags) / valid_count if valid_count else 0.0,
                    "smoothness": smoothness_score(sim_to_a, sim_to_b),
                    "endpoint_a": endpoint_a,
                    "endpoint_b": endpoint_b,
                }
            )
    return pd.DataFrame(step_rows), pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    reference_checkpoint = args.reference_checkpoint or args.checkpoint
    model, _, checkpoint = load_model_from_checkpoint(args.checkpoint, device=args.device)
    reference_model, _, reference_checkpoint_payload = load_model_from_checkpoint(reference_checkpoint, device=args.device)
    amp_enabled = bool(args.amp and model.config.device == "cuda")
    log(f"Loaded target checkpoint {Path(args.checkpoint).resolve()}")
    log(f"Loaded reference checkpoint {Path(reference_checkpoint).resolve()}")

    manifest = checkpoint.get("manifest", {})
    reference_manifest = reference_checkpoint_payload.get("manifest", {})
    set_all_seeds(args.seed or int(manifest.get("seed", 0)))
    max_train_samples = args.max_train_samples if args.max_train_samples is not None else reference_manifest.get("max_train_samples")
    max_val_samples = args.max_val_samples if args.max_val_samples is not None else reference_manifest.get("max_val_samples")
    max_test_samples = args.max_test_samples if args.max_test_samples is not None else reference_manifest.get("max_test_samples")
    max_smiles_length = args.max_smiles_length if args.max_smiles_length is not None else reference_manifest.get("max_smiles_length", 68)
    bundle = prepare_zinc250k_dataset(
        raw_dir=args.data_root,
        processed_dir=args.processed_root,
        validation_fraction=float(reference_manifest.get("validation_fraction", 0.1)),
        test_fraction=float(reference_manifest.get("test_fraction", 0.1)),
        split_seed=int(reference_manifest.get("split_seed", 13)),
        max_smiles_length=int(max_smiles_length),
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
        force_reprocess=args.force_reprocess,
    )

    output_root = Path(args.output_root) if args.output_root else Path(args.checkpoint).resolve().parents[1]
    (output_root / "interpolation").mkdir(parents=True, exist_ok=True)
    (output_root / "tables").mkdir(parents=True, exist_ok=True)

    reference_loader = build_dataloader(bundle, args.split, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    reference_smiles, reference_latents = encode_split(reference_model, reference_loader, reference_model.config.device, amp_enabled)
    pairs = load_or_create_pairs(
        pairs_file=args.pairs_file,
        output_root=output_root,
        reference_smiles=reference_smiles,
        reference_latents=reference_latents,
        pool_size=args.pool_size,
        pairs_per_bin=args.pairs_per_bin,
        seed=args.seed,
    )
    bin_counts = {}
    for pair in pairs:
        bin_counts[pair["bin"]] = bin_counts.get(pair["bin"], 0) + 1
    log(f"Using {len(pairs)} interpolation pairs with bin counts: {bin_counts}")

    train_reference = set(load_split_frame(bundle, "train")["canonical_smiles"].tolist())
    step_rows, summary_rows = evaluate_paths(
        model=model,
        model_name=str(manifest.get("model_name", "")),
        pairs=pairs,
        bundle=bundle,
        split_name=args.split,
        device=model.config.device,
        steps=args.steps,
        train_reference=train_reference,
        amp_enabled=amp_enabled,
    )
    step_rows.to_csv(output_root / "interpolation" / "interpolation_steps.csv", index=False)
    summary_rows.to_csv(output_root / "interpolation" / "interpolation_summary.csv", index=False)

    grouped = summary_rows.groupby("bin")[["valid_fraction", "fully_valid_path", "path_uniqueness", "path_novelty", "smoothness"]].agg(["mean", "std"])
    grouped.columns = ["_".join(parts).strip() for parts in grouped.columns.to_flat_index()]
    grouped = grouped.reset_index()
    grouped.to_csv(output_root / "tables" / "interpolation_by_bin.csv", index=False)
    save_json(
        output_root / "interpolation" / "interpolation_summary.json",
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "reference_checkpoint": str(Path(reference_checkpoint).resolve()),
            "split": args.split,
            "steps": args.steps,
            "pairs_per_bin": args.pairs_per_bin,
            "bins": grouped.to_dict(orient="records"),
        },
    )
    log(f"Saved interpolation outputs to {output_root}")


if __name__ == "__main__":
    main()
