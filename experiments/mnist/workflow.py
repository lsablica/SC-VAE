from __future__ import annotations

from pathlib import Path
import sys

import torch
from tqdm.auto import tqdm

from .config import RunSpec
from .models import create_model


class NonFiniteTrainingError(RuntimeError):
    """Raised when a run produces a nonfinite objective or gradient."""


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
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} [Train]",
        leave=False,
        disable=not sys.stderr.isatty(),
    )
    for batch, _ in progress:
        global_step += 1
        if warmup_steps > 0 and global_step <= warmup_steps:
            current_lr = base_learning_rate * global_step / warmup_steps
        else:
            current_lr = base_learning_rate
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        recon_batch, mu, second_param = model(batch)
        loss, batch_recon, batch_kl = model.loss_function(batch, recon_batch, mu, second_param)
        if not bool(torch.isfinite(loss)):
            raise NonFiniteTrainingError(
                f"Nonfinite training loss at epoch {epoch}, step {global_step}"
            )
        loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf"),
                error_if_nonfinite=True,
                foreach=True,
            )
        except RuntimeError as exc:
            raise NonFiniteTrainingError(
                f"Nonfinite gradient at epoch {epoch}, step {global_step}"
            ) from exc
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
    concentrations: list[torch.Tensor] = []
    with torch.no_grad():
        progress = tqdm(
            loader,
            desc=desc,
            leave=False,
            disable=not sys.stderr.isatty(),
        )
        for batch, _ in progress:
            batch = batch.to(device, non_blocking=True)
            recon_batch, mu, second_param = model(batch)
            loss, batch_recon, batch_kl = model.loss_function(batch, recon_batch, mu, second_param)
            if not bool(torch.isfinite(loss)):
                raise NonFiniteTrainingError(
                    f"Nonfinite evaluation loss in {desc}"
                )
            if getattr(model, "distribution_type", "") in {
                "spcauchy",
                "vmf",
                "powerspherical",
            }:
                concentrations.append(
                    second_param.detach().reshape(-1).cpu()
                )
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
    metrics = {
        "total_loss": total_loss / denom,
        "recon_loss": recon_loss / denom,
        "kl_loss": kl_loss / denom,
    }
    if concentrations:
        values = torch.cat(concentrations).to(torch.float64)
        quantiles = torch.quantile(
            values,
            torch.tensor(
                [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0],
                dtype=torch.float64,
            ),
        )
        metrics["concentration"] = {
            "parameter": {
                "spcauchy": "rho",
                "vmf": "kappa",
                "powerspherical": "lambda",
            }[getattr(model, "distribution_type", "")],
            "count": int(values.numel()),
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
            "min": float(quantiles[0]),
            "q25": float(quantiles[1]),
            "median": float(quantiles[2]),
            "q75": float(quantiles[3]),
            "q90": float(quantiles[4]),
            "q95": float(quantiles[5]),
            "q99": float(quantiles[6]),
            "max": float(quantiles[7]),
            "fraction_above_0_9": float((values > 0.9).double().mean()),
        }
    else:
        metrics["concentration"] = None
    return metrics


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
