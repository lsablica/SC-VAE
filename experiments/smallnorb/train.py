"""Resumable single-process trainer for the controlled smallNORB study."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import (
    ALL_FAMILIES,
    ARCHITECTURES,
    RunConfig,
    beta_for_epoch,
    learning_rate_for_epoch,
)
from .data import (
    assert_expected_split_counts,
    build_dataloaders,
)
from .evaluate import evaluate_partitions
from .metrics import summarize_tensor
from .models import SmallNORBViewVAE, build_model
from .utils import (
    command_output,
    ensure_dir,
    read_json,
    repo_relative,
    resolve_device,
    set_global_seed,
    sha256_file,
    write_commands,
    write_csv,
    write_environment,
    write_json,
)


class NonFiniteTrainingError(RuntimeError):
    pass


def _optimizer(
    model: SmallNORBViewVAE,
    config: RunConfig,
) -> torch.optim.Optimizer:
    concentration_ids = {
        id(parameter)
        for parameter in model.posterior.concentration_head.parameters()
    }
    shared = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in concentration_ids
    ]
    concentration = list(
        model.posterior.concentration_head.parameters()
    )
    return torch.optim.AdamW(
        [
            {"params": shared, "lr": config.learning_rate},
            {
                "params": concentration,
                "lr": (
                    config.learning_rate
                    * config.concentration_learning_rate_multiplier
                ),
                "name": "concentration_head",
            },
        ],
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
    )


def _set_learning_rate(
    optimizer: torch.optim.Optimizer,
    config: RunConfig,
    epoch: int,
) -> float:
    base = learning_rate_for_epoch(config, epoch)
    for group in optimizer.param_groups:
        multiplier = (
            config.concentration_learning_rate_multiplier
            if group.get("name") == "concentration_head"
            else 1.0
        )
        group["lr"] = base * multiplier
    return base


def _checkpoint_payload(
    model: SmallNORBViewVAE,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: RunConfig,
    epoch: int,
    selection_metric: float,
    data_loader_generator_state: torch.Tensor,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config.to_dict(),
        "selection_metric": selection_metric,
        "data_loader_generator_state": data_loader_generator_state,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".temporary")
    torch.save(payload, temporary)
    temporary.replace(path)


def _gpu_telemetry(epoch: int) -> dict[str, Any]:
    query = command_output(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,clocks.current.sm,"
            "clocks.current.memory,power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    values = [value.strip() for value in query.split(",")]
    row: dict[str, Any] = {"epoch": epoch, "raw": query}
    keys = (
        "temperature_c",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "power_w",
        "memory_used_mib",
    )
    if len(values) == len(keys):
        for key, value in zip(keys, values):
            try:
                row[key] = float(value)
            except ValueError:
                row[key] = value
    return row


@torch.no_grad()
def _initial_diagnostics(
    model: SmallNORBViewVAE,
    loader,
    device: torch.device,
) -> dict[str, Any]:
    images, _ = next(iter(loader))
    images = images.to(device, non_blocking=True)
    parameters = model.encode(images)
    kl = model.posterior.kl(parameters)
    result = {
        "kl": summarize_tensor(kl),
        "posterior_parameter": summarize_tensor(parameters.scale),
        "posterior_parameter_name": {
            "spcauchy": "rho",
            "vmf_robust": "kappa",
            "powerspherical": "lambda",
            "gaussian_isotropic": "sigma",
            "gaussian_diagonal": "sigma",
        }[model.family],
    }
    if model.family == "spcauchy":
        result["direct_kl"] = model.posterior.term_diagnostics(device)
    return result


def _train_epoch(
    model: SmallNORBViewVAE,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: RunConfig,
    epoch: int,
) -> dict[str, Any]:
    model.train()
    beta = beta_for_epoch(config, epoch)
    amp_enabled = config.mixed_precision and device.type == "cuda"
    totals = {
        "total": 0.0,
        "reconstruction_nll": 0.0,
        "kl": 0.0,
        "pixel_mse": 0.0,
    }
    example_count = 0
    gradient_norms: list[float] = []
    amp_skipped_steps = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            loss, components, _ = model.loss(
                images,
                beta=beta,
                sigma_x=config.sigma_x,
            )
        if not bool(torch.isfinite(loss)):
            raise NonFiniteTrainingError(
                f"Nonfinite loss at epoch {epoch}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip_norm,
            error_if_nonfinite=False,
            foreach=True,
        )
        if not bool(torch.isfinite(gradient_norm)):
            if amp_enabled:
                # GradScaler detects the same overflow and skips this update.
                # Record it explicitly instead of misclassifying an AMP
                # scaling overflow as a model-level numerical failure.
                scaler.step(optimizer)
                scaler.update()
                amp_skipped_steps += 1
                continue
            raise NonFiniteTrainingError(
                f"Nonfinite gradient at epoch {epoch}"
            )
        gradient_norms.append(float(gradient_norm))
        scaler.step(optimizer)
        scaler.update()
        batch_size = images.shape[0]
        example_count += batch_size
        for key in totals:
            totals[key] += (
                float(components[key].detach().sum()) if key != "total"
                else float(components["total"].detach().sum())
            )
    result = {
        key: value / max(example_count, 1)
        for key, value in totals.items()
    }
    result["beta"] = beta
    result["gradient_norm"] = summarize_tensor(
        torch.tensor(gradient_norms)
    )
    result["amp_skipped_steps"] = amp_skipped_steps
    result["peak_cuda_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    return result


def _flatten_history_row(
    epoch: int,
    learning_rate: float,
    train: dict[str, Any],
    validation: dict[str, Any],
    wall_clock_epoch_s: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "beta": train["beta"],
        "train_total": train["total"],
        "train_reconstruction_nll": train["reconstruction_nll"],
        "train_kl": train["kl"],
        "train_pixel_mse": train["pixel_mse"],
        "gradient_norm_mean": train["gradient_norm"].get("mean"),
        "gradient_norm_q90": train["gradient_norm"].get("q90"),
        "gradient_norm_max": train["gradient_norm"].get("max"),
        "peak_cuda_memory_bytes": train["peak_cuda_memory_bytes"],
        "amp_skipped_steps": train["amp_skipped_steps"],
        "wall_clock_epoch_s": wall_clock_epoch_s,
    }
    for partition, metrics in validation.items():
        for metric, summary in metrics.items():
            if isinstance(summary, dict) and "mean" in summary:
                row[f"{partition}_{metric}_mean"] = summary["mean"]
                if metric in {"kl", "posterior_scale"}:
                    row[f"{partition}_{metric}_median"] = summary[
                        "median"
                    ]
                    row[f"{partition}_{metric}_q95"] = summary["q95"]
                    row[f"{partition}_{metric}_max"] = summary["max"]
    return row


def _write_progress(
    run_dir: Path,
    history_rows: list[dict[str, Any]],
    history_details: list[dict[str, Any]],
    telemetry: list[dict[str, Any]],
) -> None:
    write_csv(run_dir / "history.csv", history_rows)
    write_json(run_dir / "history.json", history_details)
    write_csv(run_dir / "gpu_telemetry.csv", telemetry)


def train_run(
    config: RunConfig,
    device: torch.device,
    *,
    command: str,
    force: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Train one family without touching official test instances."""

    run_dir = ensure_dir(config.run_dir)
    best_path = run_dir / "checkpoint_best.pt"
    last_path = run_dir / "checkpoint_last.pt"
    if best_path.exists() and not force and not resume:
        raise FileExistsError(
            f"Run already exists: {run_dir}. Use --resume or --force."
        )

    set_global_seed(config.seed)
    write_json(run_dir / "config.json", config.to_dict())
    write_environment(run_dir / "environment.txt")
    write_commands(run_dir / "commands.txt", [command])
    split_counts = assert_expected_split_counts(config.data_root)
    loaders = build_dataloaders(config, include_test=False)
    model = build_model(config, device)
    optimizer = _optimizer(model, config)
    amp_enabled = config.mixed_precision and device.type == "cuda"
    # The reconstruction term sums 4096 pixels and starts in the thousands.
    # AMP's generic 65536 initial scale can therefore overflow otherwise
    # finite early gradients before dynamic backoff gets a chance to act.
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled,
        init_scale=1.0,
        growth_interval=10_000,
    )

    history_rows: list[dict[str, Any]] = []
    history_details: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    start_epoch = 1
    best_reconstruction = math.inf
    best_total = math.inf
    selected: dict[str, Any] | None = None
    if resume and last_path.exists():
        payload = torch.load(
            last_path, map_location=device, weights_only=False
        )
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scaler.load_state_dict(payload.get("scaler_state_dict", {}))
        if "data_loader_generator_state" in payload:
            loaders["train"].generator.set_state(
                payload["data_loader_generator_state"].cpu()
            )
        if "rng_state" in payload:
            rng_state = payload["rng_state"]
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch_cpu"].cpu())
            if (
                device.type == "cuda"
                and rng_state.get("torch_cuda") is not None
            ):
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in rng_state["torch_cuda"]]
                )
        start_epoch = int(payload["epoch"]) + 1
        history_details = (
            read_json(run_dir / "history.json")
            if (run_dir / "history.json").exists()
            else []
        )
        history_rows = [
            detail["flat"] for detail in history_details
        ]
        if (run_dir / "selection_summary.json").exists():
            selected = read_json(run_dir / "selection_summary.json")
            best_reconstruction = float(
                selected["validation_reconstruction_nll"]
            )
            best_total = float(selected["validation_total_objective"])

    initial = _initial_diagnostics(
        model, loaders["validation"], device
    )
    cache_manifest = (
        Path(config.data_root)
        / "processed"
        / "cache_manifest.json"
    )
    seed_manifest = {
        "schema_version": 1,
        "seed": config.seed,
        "family": config.family,
        "stage": config.stage,
        "run_name": config.run_name,
        "data_order": (
            "torch DataLoader shuffle generator seeded with the run seed; "
            "identical config seed gives identical order across families"
        ),
        "split_counts": split_counts,
        "effective_split_counts": {
            split: len(loader.dataset)
            for split, loader in loaders.items()
        },
        "data_cache_manifest": repo_relative(cache_manifest),
        "data_cache_manifest_sha256": sha256_file(cache_manifest),
        "parameter_counts": model.parameter_summary(),
        "initial_diagnostics": initial,
        "official_test_accessed": False,
    }
    write_json(run_dir / "seed_manifest.json", seed_manifest)

    training_started = time.perf_counter()
    try:
        for epoch in range(start_epoch, config.epochs + 1):
            epoch_started = time.perf_counter()
            learning_rate = _set_learning_rate(
                optimizer, config, epoch
            )
            train_metrics = _train_epoch(
                model,
                loaders["train"],
                optimizer,
                scaler,
                device,
                config,
                epoch,
            )
            include_ssim = (
                epoch % config.evaluate_ssim_every_epochs == 0
                or epoch == config.epochs
            )
            validation, _ = evaluate_partitions(
                model,
                loaders["validation"],
                device,
                sigma_x=config.sigma_x,
                beta=beta_for_epoch(config, epoch),
                include_ssim=include_ssim,
                prefix="validation",
                return_records=False,
            )
            elapsed = time.perf_counter() - epoch_started
            flat = _flatten_history_row(
                epoch,
                learning_rate,
                train_metrics,
                validation,
                elapsed,
            )
            detail = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train": train_metrics,
                "validation": validation,
                "wall_clock_epoch_s": elapsed,
                "flat": flat,
            }
            history_rows.append(flat)
            history_details.append(detail)
            if device.type == "cuda":
                telemetry.append(_gpu_telemetry(epoch))

            all_validation = validation["validation"]
            reconstruction = float(
                all_validation["reconstruction_nll"]["mean"]
            )
            total = float(all_validation["total_objective"]["mean"])
            is_better = (
                reconstruction < best_reconstruction
                or (
                    math.isclose(
                        reconstruction,
                        best_reconstruction,
                        rel_tol=0.0,
                        abs_tol=1e-10,
                    )
                    and total < best_total
                )
            )
            payload = _checkpoint_payload(
                model,
                optimizer,
                scaler,
                config,
                epoch,
                reconstruction,
                loaders["train"].generator.get_state(),
            )
            _save_checkpoint(last_path, payload)
            if is_better:
                best_reconstruction = reconstruction
                best_total = total
                _save_checkpoint(best_path, payload)
                selected = {
                    "selection_metric": (
                        "validation reconstruction NLL over all azimuths"
                    ),
                    "tie_breaker": "validation total objective",
                    "selected_epoch": epoch,
                    "validation_reconstruction_nll": reconstruction,
                    "validation_total_objective": total,
                    "validation": validation,
                    "checkpoint": repo_relative(best_path),
                }
                write_json(
                    run_dir / "selection_summary.json", selected
                )
            _write_progress(
                run_dir, history_rows, history_details, telemetry
            )
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_reconstruction_nll": train_metrics[
                            "reconstruction_nll"
                        ],
                        "train_kl": train_metrics["kl"],
                        "validation_reconstruction_nll": reconstruction,
                        "validation_gap_reconstruction_nll": validation[
                            "validation_gap"
                        ]["reconstruction_nll"]["mean"],
                        "validation_kl": all_validation["kl"]["mean"],
                        "beta": train_metrics["beta"],
                        "elapsed_s": elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    except Exception as error:
        _write_progress(
            run_dir, history_rows, history_details, telemetry
        )
        failure = {
            "status": "failed",
            "failure_type": type(error).__name__,
            "message": str(error),
            "completed_epochs": len(history_rows),
            "wall_clock_training_s": (
                time.perf_counter() - training_started
            ),
        }
        write_json(run_dir / "failure_summary.json", failure)
        raise

    if selected is None:
        raise RuntimeError("No checkpoint was selected")
    selected["wall_clock_training_s"] = (
        time.perf_counter() - training_started
    )
    selected["peak_cuda_memory_bytes"] = max(
        (
            row.get("peak_cuda_memory_bytes", 0)
            for row in history_rows
        ),
        default=0,
    )
    write_json(run_dir / "selection_summary.json", selected)
    return {
        "status": "completed",
        "run_dir": repo_relative(run_dir),
        "selected_epoch": selected["selected_epoch"],
        "validation_reconstruction_nll": best_reconstruction,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one controlled smallNORB VAE run."
    )
    parser.add_argument("--family", choices=ALL_FAMILIES, required=True)
    parser.add_argument(
        "--architecture",
        choices=ARCHITECTURES,
        default="baseline_cnn",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", default="search")
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--beta-target", type=float, default=0.5)
    parser.add_argument("--beta-warmup-epochs", type=int, default=20)
    parser.add_argument("--sigma-x", type=float, default=0.20)
    parser.add_argument(
        "--concentration-learning-rate-multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--notes", default="")
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    values = {
        "family": args.family,
        "architecture": args.architecture,
        "seed": args.seed,
        "stage": args.stage,
        "run_name": args.run_name,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "minimum_learning_rate": args.minimum_learning_rate,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "beta_target": args.beta_target,
        "beta_warmup_epochs": args.beta_warmup_epochs,
        "sigma_x": args.sigma_x,
        "concentration_learning_rate_multiplier": (
            args.concentration_learning_rate_multiplier
        ),
        "train_limit": args.train_limit,
        "validation_limit": args.validation_limit,
        "output_root": args.output_root,
        "mixed_precision": not args.disable_amp,
        "notes": args.notes,
    }
    if args.data_root is not None:
        values["data_root"] = args.data_root
    return RunConfig(**values)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    result = train_run(
        config,
        resolve_device(args.device),
        command="python -m experiments.smallnorb.train "
        + " ".join(
            argument
            for argument in __import__("sys").argv[1:]
        ),
        force=args.force,
        resume=args.resume,
    )
    print(result)


if __name__ == "__main__":
    main()
