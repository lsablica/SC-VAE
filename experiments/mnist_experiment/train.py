from __future__ import annotations

import argparse
from pathlib import Path

import torch

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mnist_experiment.config import (  # noqa: E402
    BENCHMARK_PRESET,
    OUTPUT_ROOT,
    QUALITATIVE_PRESET,
    DEFAULT_DATA_DIR,
    build_specs_for_preset,
)
from experiments.mnist_experiment.data import build_mnist_dataloaders  # noqa: E402
from experiments.mnist_experiment.models import create_model  # noqa: E402
from experiments.mnist_experiment.utils import (  # noqa: E402
    checkpoint_exists,
    ensure_dir,
    repo_relative_path,
    resolve_device,
    set_global_seed,
    write_csv,
    write_json,
)
from experiments.mnist_experiment.workflow import (  # noqa: E402
    checkpoint_payload,
    evaluate_model,
    save_checkpoint,
    train_one_epoch,
)


def _history_fieldnames() -> list[str]:
    return [
        "epoch",
        "train_total_loss",
        "train_recon_loss",
        "train_kl",
        "eval_total_loss",
        "eval_recon_loss",
        "eval_kl",
        "learning_rate",
    ]


def train_run(run_spec, data_dir: Path, output_root: Path, device: str, force: bool = False) -> dict:
    run_dir = ensure_dir(run_spec.output_dir(output_root))
    if checkpoint_exists(run_dir) and not force:
        return {
            "status": "skipped",
            "run_dir": repo_relative_path(run_dir),
            "reason": "existing_artifacts",
        }

    set_global_seed(run_spec.seed)
    write_json(run_dir / "run_config.json", run_spec.to_dict())

    train_loader, eval_loader = build_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=run_spec.batch_size,
        seed=run_spec.seed,
    )
    model = create_model(run_spec, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=run_spec.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=run_spec.scheduler_factor,
        patience=run_spec.scheduler_patience,
    )

    history_rows: list[dict] = []
    best_summary: dict | None = None
    best_eval_recon = float("inf")
    global_step = 0

    for epoch in range(1, run_spec.epochs + 1):
        train_metrics, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            run_spec.epochs,
            base_learning_rate=run_spec.learning_rate,
            warmup_steps=run_spec.warmup_steps,
            global_step=global_step,
        )
        eval_metrics = evaluate_model(model, eval_loader, device, desc=f"Epoch {epoch}/{run_spec.epochs} [Eval]")
        learning_rate = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "train_total_loss": train_metrics["total_loss"],
            "train_recon_loss": train_metrics["recon_loss"],
            "train_kl": train_metrics["kl_loss"],
            "eval_total_loss": eval_metrics["total_loss"],
            "eval_recon_loss": eval_metrics["recon_loss"],
            "eval_kl": eval_metrics["kl_loss"],
            "learning_rate": learning_rate,
        }
        history_rows.append(row)
        # Match the scheduler target to the benchmark's primary selection metric.
        scheduler.step(eval_metrics["recon_loss"])

        if eval_metrics["recon_loss"] < best_eval_recon:
            best_eval_recon = eval_metrics["recon_loss"]
            checkpoint_path = run_dir / "best_recon_checkpoint.pt"
            save_checkpoint(checkpoint_path, checkpoint_payload(model, optimizer, scheduler, epoch, run_spec))
            best_summary = {
                "selection_metric": run_spec.selection_metric,
                "selected_epoch": epoch,
                "selected_eval_recon_loss": eval_metrics["recon_loss"],
                "selected_eval_total_loss": eval_metrics["total_loss"],
                "selected_eval_kl": eval_metrics["kl_loss"],
                "checkpoint_path": repo_relative_path(checkpoint_path),
            }

    final_checkpoint_path = run_dir / "final_checkpoint.pt"
    save_checkpoint(final_checkpoint_path, checkpoint_payload(model, optimizer, scheduler, run_spec.epochs, run_spec))
    write_csv(run_dir / "history.csv", history_rows, fieldnames=_history_fieldnames())
    write_json(
        run_dir / "selection_summary.json",
        best_summary
        if best_summary is not None
        else {
            "selection_metric": run_spec.selection_metric,
            "selected_epoch": None,
            "selected_eval_recon_loss": None,
            "selected_eval_total_loss": None,
            "selected_eval_kl": None,
            "checkpoint_path": "",
        },
    )

    return {
        "status": "trained",
        "run_dir": repo_relative_path(run_dir),
        "best_checkpoint": repo_relative_path(run_dir / "best_recon_checkpoint.pt"),
        "final_checkpoint": repo_relative_path(final_checkpoint_path),
    }


def run_training_jobs(
    preset: str,
    data_dir: str | Path,
    output_root: str | Path,
    device: str,
    models: list[str] | None = None,
    reported_dims: list[int] | None = None,
    seeds: list[int] | None = None,
    epochs: int | None = None,
    force: bool = False,
) -> list[dict]:
    specs = build_specs_for_preset(
        preset,
        model_families=models,
        reported_dims=reported_dims,
        seeds=seeds,
        epochs=epochs,
    )
    results = []
    for run_spec in specs:
        results.append(
            train_run(
                run_spec=run_spec,
                data_dir=Path(data_dir),
                output_root=Path(output_root),
                device=device,
                force=force,
            )
        )
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train reproducible MNIST benchmark runs.")
    parser.add_argument("--preset", choices=[BENCHMARK_PRESET, QUALITATIVE_PRESET], required=True)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--models", nargs="+", choices=["gaussian", "vmf", "spcauchy"])
    parser.add_argument("--reported-dims", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)
    results = run_training_jobs(
        preset=args.preset,
        data_dir=args.data_dir,
        output_root=args.output_root,
        device=device,
        models=args.models,
        reported_dims=args.reported_dims,
        seeds=args.seeds,
        epochs=args.epochs,
        force=args.force,
    )
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
