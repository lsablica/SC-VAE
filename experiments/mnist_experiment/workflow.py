from __future__ import annotations

from pathlib import Path

import torch
from tqdm.auto import tqdm

from .config import RunSpec
from .models import create_model


def train_one_epoch(
    model,
    loader,
    optimizer,
    device: str,
    epoch: int,
    total_epochs: int,
    base_learning_rate: float,
    warmup_steps: int = 0,
    global_step: int = 0,
) -> tuple[dict, int]:
    model.train()
    total_loss = 0.0
    recon_loss = 0.0
    kl_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for batch, _ in progress:
        global_step += 1
        if warmup_steps > 0 and global_step <= warmup_steps:
            current_lr = base_learning_rate * global_step / warmup_steps
        else:
            current_lr = base_learning_rate
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        batch = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, second_param = model(batch)
        loss, batch_recon, batch_kl = model.loss_function(batch, recon_batch, mu, second_param)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        recon_loss += float(batch_recon.item())
        kl_loss += float(batch_kl.item())
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            recon=f"{batch_recon.item():.4f}",
            kl=f"{batch_kl.item():.4f}",
        )
    progress.close()
    denom = max(len(loader), 1)
    return (
        {
            "total_loss": total_loss / denom,
            "recon_loss": recon_loss / denom,
            "kl_loss": kl_loss / denom,
        },
        global_step,
    )


def evaluate_model(model, loader, device: str, desc: str = "Eval") -> dict:
    model.eval()
    total_loss = 0.0
    recon_loss = 0.0
    kl_loss = 0.0
    with torch.no_grad():
        progress = tqdm(loader, desc=desc, leave=False)
        for batch, _ in progress:
            batch = batch.to(device)
            recon_batch, mu, second_param = model(batch)
            loss, batch_recon, batch_kl = model.loss_function(batch, recon_batch, mu, second_param)
            total_loss += float(loss.item())
            recon_loss += float(batch_recon.item())
            kl_loss += float(batch_kl.item())
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                recon=f"{batch_recon.item():.4f}",
                kl=f"{batch_kl.item():.4f}",
            )
        progress.close()
    denom = max(len(loader), 1)
    return {
        "total_loss": total_loss / denom,
        "recon_loss": recon_loss / denom,
        "kl_loss": kl_loss / denom,
    }


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    epoch: int,
    run_spec: RunSpec,
) -> dict:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "run_config": run_spec.to_dict(),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload


def save_checkpoint(path: str | Path, payload: dict) -> None:
    torch.save(payload, Path(path))


def load_model_from_checkpoint(checkpoint_path: str | Path, device: str) -> tuple[torch.nn.Module, dict, RunSpec]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    run_spec = RunSpec.from_dict(checkpoint["run_config"])
    model = create_model(run_spec, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint, run_spec
