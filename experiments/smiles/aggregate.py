from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from pandas.errors import EmptyDataError

from experiments.smiles.plots import (
    plot_interpolation_bin_summary,
    plot_property_histograms,
    plot_representative_interpolations,
    plot_training_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed ZINC-250k SMILES benchmark outputs.")
    parser.add_argument("--runs-root", default="experiments/smiles/runs")
    parser.add_argument("--dataset-name", default="zinc250k")
    parser.add_argument("--output-dir", default="experiments/smiles/aggregated")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--run-id", default=None, help="Restrict aggregation to a specific run_id.")
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [aggregate] {message}", flush=True)


def collect_manifests(root: Path) -> list[dict]:
    manifests = []
    for manifest_path in root.rglob("run_manifest.json"):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["run_path"] = str(manifest_path.parent)
        manifests.append(payload)
    return manifests


def filter_manifests_by_run_id(manifests: list[dict], run_id: str | None) -> list[dict]:
    if run_id is None:
        return manifests
    return [manifest for manifest in manifests if manifest.get("run_id") == run_id]


def pick_latest_runs(manifests: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], dict] = {}
    for manifest in manifests:
        key = (manifest["model_name"], int(manifest["seed"]))
        previous = grouped.get(key)
        if previous is None or manifest["run_id"] > previous["run_id"]:
            grouped[key] = manifest
    return list(grouped.values())


def flatten_eval_metrics(manifest: dict, eval_payload: dict) -> dict:
    row = {
        "model_name": manifest["model_name"],
        "seed": manifest["seed"],
        "fairness_regime": manifest["fairness_regime"],
    }
    for section, metrics in eval_payload.items():
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                row[f"{section}_{key}"] = value
    return row


def aggregate_mean_std(frame: pd.DataFrame, group_column: str = "model_name") -> pd.DataFrame:
    numeric_columns = [column for column in frame.select_dtypes(include=["number"]).columns if column != "seed"]
    aggregated = frame.groupby(group_column)[numeric_columns].agg(["mean", "std"])
    aggregated.columns = ["_".join(parts).strip() for parts in aggregated.columns.to_flat_index()]
    return aggregated.reset_index()


def main() -> None:
    args = parse_args()
    base_root = Path(args.runs_root)
    runs_root = base_root / args.dataset_name
    if not runs_root.exists():
        runs_root = base_root
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Aggregating runs from {runs_root}")

    manifests = collect_manifests(runs_root)
    manifests = filter_manifests_by_run_id(manifests, args.run_id)
    selected_runs = pick_latest_runs(manifests)
    log(f"Found {len(selected_runs)} latest runs across model/seed combinations")
    benchmark_rows = []
    interpolation_frames = []

    for manifest in selected_runs:
        run_path = Path(manifest["run_path"])
        eval_path = run_path / "metrics" / f"eval_{args.eval_split}.json"
        interpolation_path = run_path / "interpolation" / "interpolation_summary.csv"
        interpolation_steps_path = run_path / "interpolation" / "interpolation_steps.csv"
        history_path = run_path / "metrics" / "train_history.csv"
        histogram_path = run_path / "tables" / f"property_histograms_{args.eval_split}.csv"

        if eval_path.exists():
            benchmark_rows.append(flatten_eval_metrics(manifest, json.loads(eval_path.read_text(encoding="utf-8"))))
        if interpolation_path.exists():
            frame = pd.read_csv(interpolation_path)
            frame["model_name"] = manifest["model_name"]
            frame["seed"] = manifest["seed"]
            interpolation_frames.append(frame)
            if interpolation_steps_path.exists():
                plot_representative_interpolations(
                    pd.read_csv(interpolation_steps_path),
                    frame,
                    run_path / "plots" / "representative_interpolations.png",
                )
        if history_path.exists():
            plot_training_curves(pd.read_csv(history_path), run_path / "plots" / "training_curves.png")
        if histogram_path.exists():
            try:
                histogram_frame = pd.read_csv(histogram_path)
            except EmptyDataError:
                histogram_frame = pd.DataFrame()
            if not histogram_frame.empty:
                plot_property_histograms(histogram_frame, run_path / "plots" / f"property_histograms_{args.eval_split}.png")

    benchmark_frame = pd.DataFrame(benchmark_rows)
    if not benchmark_frame.empty:
        benchmark_frame.to_csv(output_dir / "benchmark_seed_metrics.csv", index=False)
        aggregate_mean_std(benchmark_frame).to_csv(output_dir / "benchmark_mean_std.csv", index=False)

    if interpolation_frames:
        interpolation_frame = pd.concat(interpolation_frames, ignore_index=True)
        interpolation_frame.to_csv(output_dir / "interpolation_seed_metrics.csv", index=False)
        grouped = interpolation_frame.groupby(["model_name", "bin"])[["valid_fraction", "fully_valid_path", "path_uniqueness", "path_novelty", "smoothness"]].agg(["mean", "std"])
        grouped.columns = ["_".join(parts).strip() for parts in grouped.columns.to_flat_index()]
        grouped = grouped.reset_index()
        grouped.to_csv(output_dir / "interpolation_mean_std.csv", index=False)
        for model_name, frame in grouped.groupby("model_name"):
            plot_interpolation_bin_summary(frame.reset_index(drop=True), output_dir / f"{model_name}_interpolation_summary.png")

    save_json(output_dir / "aggregation_manifest.json", {"selected_runs": selected_runs})
    log(f"Saved aggregated outputs to {output_dir}")


if __name__ == "__main__":
    main()
