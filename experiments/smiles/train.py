from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm

from experiments.smiles.config import ExperimentConfig, beta_for_epoch, ensure_output_dirs
from experiments.smiles.data import build_dataloader, prepare_zinc250k_dataset
from experiments.smiles.model_factory import build_model
from src.utils import set_all_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ZINC-250k SMILES VAE benchmark run.")
    parser.add_argument("--dataset-name", default="zinc250k")
    parser.add_argument("--model-name", choices=["spcauchy-128", "gaussian-64", "gaussian-128"], default="spcauchy-128")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta-start", type=float, default=0.0)
    parser.add_argument("--beta-target", type=float, default=0.015)
    parser.add_argument("--beta-zero-epochs", type=int, default=1)
    parser.add_argument("--beta-warmup-epochs", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--max-smiles-length", type=int, default=68)
    parser.add_argument("--data-root", default="experiments/smiles/datasets/zinc250k/raw")
    parser.add_argument("--processed-root", default="experiments/smiles/datasets/zinc250k/processed")
    parser.add_argument("--runs-root", default="experiments/smiles/runs")
    parser.add_argument("--output-root")
    parser.add_argument("--run-id")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--spcauchy-rho-bias-init", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision on CUDA. Leave off for this SMILES benchmark unless you have verified numerical stability.")
    parser.add_argument("--force-reprocess", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [train] {message}", flush=True)


def warn_on_suspicious_runs_root(experiment: ExperimentConfig) -> None:
    runs_root = Path(experiment.runs_root)
    tail = list(runs_root.parts)
    suspicious_tokens = {
        experiment.dataset_name,
        experiment.model_spec.name,
        f"latent_{experiment.model_spec.latent_dim}",
        f"seed_{experiment.seed}",
    }
    overlapping = [part for part in tail if part in suspicious_tokens]
    if overlapping:
        log(
            "runs_root already contains benchmark path components "
            f"{overlapping}. This can create duplicated empty folders. "
            "Prefer pointing runs_root at a clean top-level directory such as SC-VAE-runs2."
        )


def run_epoch(
    model,
    loader,
    optimizer,
    device: str,
    beta: float,
    train: bool,
    amp_enabled: bool,
    scaler,
    grad_clip_norm: float | None,
) -> dict[str, float]:
    model.train(mode=train)
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_examples = 0
    total_rho_mean = 0.0
    total_rho_max = 0.0
    iterator = tqdm(loader, leave=False, desc="train" if train else "val")

    for batch in iterator:
        token_ids = batch["token_ids"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)

        model.config.kl_weight = beta
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits, mu, second_param = model(token_ids)
                loss, recon, kl = model.loss_function(token_ids, logits, mu, second_param)
            if train:
                if amp_enabled:
                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

        batch_size = token_ids.size(0)
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_recon += float(recon.item()) * batch_size
        total_kl += float(kl.item()) * batch_size
        if model.distribution_type == "spcauchy":
            rho = second_param.detach()
            total_rho_mean += float(rho.mean().item()) * batch_size
            total_rho_max += float(rho.max().item()) * batch_size
        iterator.set_postfix(loss=f"{loss.item():.3f}", recon=f"{recon.item():.3f}", kl=f"{kl.item():.3f}")

    if total_examples == 0:
        raise ValueError("Encountered an empty dataloader.")
    metrics = {
        "loss": total_loss / total_examples,
        "recon_loss": total_recon / total_examples,
        "kl_loss": total_kl / total_examples,
    }
    if model.distribution_type == "spcauchy":
        metrics["rho_mean"] = total_rho_mean / total_examples
        metrics["rho_max"] = total_rho_max / total_examples
    return metrics


def save_checkpoint(path: Path, *, epoch: int, model, optimizer, val_loss: float, manifest: dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": model.config,
            "manifest": manifest,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    experiment = ExperimentConfig(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta_start=args.beta_start,
        beta_target=args.beta_target,
        beta_zero_epochs=args.beta_zero_epochs,
        beta_warmup_epochs=args.beta_warmup_epochs,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        max_smiles_length=args.max_smiles_length,
        data_root=args.data_root,
        processed_root=args.processed_root,
        runs_root=args.runs_root,
        output_root=args.output_root,
        run_id=args.run_id,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        spcauchy_rho_bias_init=args.spcauchy_rho_bias_init,
        grad_clip_norm=args.grad_clip_norm,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )
    set_all_seeds(experiment.seed)
    warn_on_suspicious_runs_root(experiment)

    output_root = experiment.resolve_output_root()
    output_dirs = ensure_output_dirs(output_root)
    log(f"Starting training run for model={experiment.model_name} seed={experiment.seed} device={experiment.resolved_device()}")
    log(f"Outputs will be written to {output_root}")
    bundle = prepare_zinc250k_dataset(
        raw_dir=experiment.data_root,
        processed_dir=experiment.processed_root,
        validation_fraction=experiment.validation_fraction,
        test_fraction=experiment.test_fraction,
        split_seed=experiment.split_seed,
        max_smiles_length=experiment.max_smiles_length,
        max_train_samples=experiment.max_train_samples,
        max_val_samples=experiment.max_val_samples,
        max_test_samples=experiment.max_test_samples,
        force_reprocess=args.force_reprocess,
    )
    model, model_config = build_model(
        experiment,
        vocab_size=bundle.vocabulary.size,
        max_seq_len=bundle.max_seq_len,
        pad_token_id=bundle.vocabulary.pad_token_id,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=experiment.learning_rate, weight_decay=experiment.weight_decay)
    amp_enabled = bool(args.amp and model_config.device == "cuda")
    if args.amp and not amp_enabled:
        log("AMP was requested but is unavailable on the resolved device; continuing with AMP disabled.")
    elif not args.amp and model_config.device == "cuda":
        log("AMP is disabled. This is the recommended default for the current SMILES benchmark because mixed precision produced NaNs during tuning.")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_loader = build_dataloader(bundle, "train", batch_size=experiment.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = build_dataloader(bundle, "val", batch_size=experiment.batch_size, shuffle=False, num_workers=args.num_workers)
    log(
        "Prepared data with "
        f"train={bundle.metadata['num_train']} val={bundle.metadata['num_val']} "
        f"test={bundle.metadata['num_test']} "
        f"vocab={bundle.vocabulary.size} max_seq_len={bundle.max_seq_len} "
        f"max_smiles_length={bundle.metadata['max_smiles_length']}"
    )
    log(
        "Model/fairness setup: "
        f"distribution={experiment.model_spec.distribution_type} latent_dim={experiment.model_spec.latent_dim} "
        f"fairness={experiment.model_spec.fairness_regime} "
        f"rho_bias_init={experiment.spcauchy_rho_bias_init:.3f} "
        f"grad_clip_norm={experiment.grad_clip_norm}"
    )

    manifest = experiment.to_manifest_dict()
    manifest["data_metadata"] = bundle.metadata
    manifest["vocab_size"] = bundle.vocabulary.size
    manifest["max_seq_len"] = bundle.max_seq_len
    manifest["pad_token_id"] = bundle.vocabulary.pad_token_id
    manifest["model_config"] = vars(model_config)
    manifest["amp"] = amp_enabled
    save_json(output_root / "run_manifest.json", manifest)

    metrics_path = output_dirs["metrics"] / "train_history.csv"
    history_rows = []
    best_val_loss = float("inf")
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "beta",
                "learning_rate",
                "train_loss",
                "train_recon_loss",
                "train_kl_loss",
                "val_loss",
                "val_recon_loss",
                "val_kl_loss",
                "val_to_train_kl_ratio",
                "train_rho_mean",
                "train_rho_max",
                "val_rho_mean",
                "val_rho_max",
            ],
        )
        writer.writeheader()
        for epoch in range(1, experiment.epochs + 1):
            beta = beta_for_epoch(epoch, experiment)
            log(f"Epoch {epoch}/{experiment.epochs} started with beta={beta:.4f}")
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                model_config.device,
                beta=beta,
                train=True,
                amp_enabled=amp_enabled,
                scaler=scaler,
                grad_clip_norm=experiment.grad_clip_norm,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer,
                model_config.device,
                beta=beta,
                train=False,
                amp_enabled=amp_enabled,
                scaler=scaler,
                grad_clip_norm=experiment.grad_clip_norm,
            )
            row = {
                "epoch": epoch,
                "beta": beta,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
                "train_recon_loss": train_metrics["recon_loss"],
                "train_kl_loss": train_metrics["kl_loss"],
                "val_loss": val_metrics["loss"],
                "val_recon_loss": val_metrics["recon_loss"],
                "val_kl_loss": val_metrics["kl_loss"],
                "val_to_train_kl_ratio": (
                    val_metrics["kl_loss"] / train_metrics["kl_loss"] if train_metrics["kl_loss"] > 0 else None
                ),
                "train_rho_mean": train_metrics.get("rho_mean"),
                "train_rho_max": train_metrics.get("rho_max"),
                "val_rho_mean": val_metrics.get("rho_mean"),
                "val_rho_max": val_metrics.get("rho_max"),
            }
            writer.writerow(row)
            history_rows.append(row)
            log(
                "Epoch summary: "
                f"train_loss={row['train_loss']:.4f} train_recon={row['train_recon_loss']:.4f} train_kl={row['train_kl_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} val_recon={row['val_recon_loss']:.4f} val_kl={row['val_kl_loss']:.4f} "
                f"val/train_kl_ratio={row['val_to_train_kl_ratio']:.4f}"
                + (
                    f" train_rho_mean={row['train_rho_mean']:.4f} train_rho_max={row['train_rho_max']:.4f} "
                    f"val_rho_mean={row['val_rho_mean']:.4f} val_rho_max={row['val_rho_max']:.4f}"
                    if row["train_rho_mean"] is not None
                    else ""
                )
            )

            save_checkpoint(output_dirs["checkpoints"] / "last.pt", epoch=epoch, model=model, optimizer=optimizer, val_loss=val_metrics["loss"], manifest=manifest)
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(output_dirs["checkpoints"] / "best-val-elbo.pt", epoch=epoch, model=model, optimizer=optimizer, val_loss=val_metrics["loss"], manifest=manifest)
                log(f"Saved new best checkpoint at epoch {epoch} with val_loss={best_val_loss:.4f}")

    save_json(output_dirs["metrics"] / "train_history.json", {"history": history_rows, "best_val_loss": best_val_loss})
    log(f"Finished training. Saved outputs to {output_root}")


if __name__ == "__main__":
    main()
