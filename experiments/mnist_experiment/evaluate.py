from __future__ import annotations

import argparse
from pathlib import Path

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
from experiments.mnist_experiment.utils import (  # noqa: E402
    checkpoint_exists,
    ensure_dir,
    repo_relative_path,
    resolve_device,
    write_csv,
    write_json,
)
from experiments.mnist_experiment.workflow import evaluate_model, load_model_from_checkpoint  # noqa: E402


def evaluate_run(run_dir: Path, data_dir: Path, device: str) -> dict:
    best_checkpoint = run_dir / "best_recon_checkpoint.pt"
    final_checkpoint = run_dir / "final_checkpoint.pt"

    if not best_checkpoint.exists() or not final_checkpoint.exists():
        return {"status": "missing_checkpoints", "run_dir": repo_relative_path(run_dir)}

    _, _, run_spec = load_model_from_checkpoint(best_checkpoint, device=device)
    _, eval_loader = build_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=run_spec.batch_size,
        seed=run_spec.seed,
    )

    best_model, best_payload, _ = load_model_from_checkpoint(best_checkpoint, device=device)
    final_model, final_payload, _ = load_model_from_checkpoint(final_checkpoint, device=device)

    best_metrics = evaluate_model(best_model, eval_loader, device, desc=f"{run_spec.model_family} best [Eval]")
    final_metrics = evaluate_model(final_model, eval_loader, device, desc=f"{run_spec.model_family} final [Eval]")

    summary = {
        "run_config": run_spec.to_dict(),
        "best_recon_checkpoint": {
            "epoch": int(best_payload["epoch"]),
            "eval_total_loss": best_metrics["total_loss"],
            "eval_recon_loss": best_metrics["recon_loss"],
            "eval_kl": best_metrics["kl_loss"],
            "checkpoint_path": repo_relative_path(best_checkpoint),
        },
        "final_checkpoint": {
            "epoch": int(final_payload["epoch"]),
            "eval_total_loss": final_metrics["total_loss"],
            "eval_recon_loss": final_metrics["recon_loss"],
            "eval_kl": final_metrics["kl_loss"],
            "checkpoint_path": repo_relative_path(final_checkpoint),
        },
    }
    write_json(run_dir / "evaluation_summary.json", summary)
    write_csv(
        run_dir / "evaluation_summary.csv",
        [
            {
                "checkpoint_type": "best_recon_checkpoint",
                "epoch": summary["best_recon_checkpoint"]["epoch"],
                "eval_total_loss": summary["best_recon_checkpoint"]["eval_total_loss"],
                "eval_recon_loss": summary["best_recon_checkpoint"]["eval_recon_loss"],
                "eval_kl": summary["best_recon_checkpoint"]["eval_kl"],
            },
            {
                "checkpoint_type": "final_checkpoint",
                "epoch": summary["final_checkpoint"]["epoch"],
                "eval_total_loss": summary["final_checkpoint"]["eval_total_loss"],
                "eval_recon_loss": summary["final_checkpoint"]["eval_recon_loss"],
                "eval_kl": summary["final_checkpoint"]["eval_kl"],
            },
        ],
    )
    return {
        "status": "evaluated",
        "run_dir": repo_relative_path(run_dir),
        "summary_path": repo_relative_path(run_dir / "evaluation_summary.json"),
    }


def run_evaluation_jobs(
    preset: str,
    data_dir: str | Path,
    output_root: str | Path,
    device: str,
    models: list[str] | None = None,
    reported_dims: list[int] | None = None,
    seeds: list[int] | None = None,
    epochs: int | None = None,
) -> list[dict]:
    results = []
    for run_spec in build_specs_for_preset(preset, model_families=models, reported_dims=reported_dims, seeds=seeds, epochs=epochs):
        run_dir = run_spec.output_dir(Path(output_root))
        ensure_dir(run_dir)
        if not checkpoint_exists(run_dir):
            results.append({"status": "skipped", "run_dir": repo_relative_path(run_dir), "reason": "training_artifacts_missing"})
            continue
        results.append(evaluate_run(run_dir, Path(data_dir), device))
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved MNIST experiment checkpoints.")
    parser.add_argument("--preset", choices=[BENCHMARK_PRESET, QUALITATIVE_PRESET], required=True)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--models", nargs="+", choices=["gaussian", "vmf", "spcauchy"])
    parser.add_argument("--reported-dims", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)
    results = run_evaluation_jobs(
        preset=args.preset,
        data_dir=args.data_dir,
        output_root=args.output_root,
        device=device,
        models=args.models,
        reported_dims=args.reported_dims,
        seeds=args.seeds,
        epochs=args.epochs,
    )
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
