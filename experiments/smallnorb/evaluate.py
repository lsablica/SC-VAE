"""Deterministic reconstruction evaluation and posterior diagnostics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .config import RunConfig
from .data import build_dataloaders
from .metrics import (
    opposite_hemisphere_fraction,
    reconstruction_metric_vectors,
    summarize_tensor,
)
from .models import SmallNORBViewVAE, build_model
from .utils import (
    read_json,
    repo_relative,
    resolve_device,
    set_global_seed,
    write_csv,
    write_json,
)


def _parameter_name(family: str) -> str:
    return {
        "spcauchy": "rho",
        "vmf_robust": "kappa",
        "powerspherical": "lambda",
        "gaussian_isotropic": "sigma",
        "gaussian_diagonal": "sigma",
    }[family]


def _append_masked(
    destination: dict[str, list[torch.Tensor]],
    source: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> None:
    for key, values in source.items():
        destination[key].append(values[mask].detach().cpu())


def _summary_from_accumulator(
    metrics: dict[str, list[torch.Tensor]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, chunks in metrics.items():
        if not chunks:
            continue
        result[key] = summarize_tensor(torch.cat(chunks))
    return result


@torch.no_grad()
def evaluate_partitions(
    model: SmallNORBViewVAE,
    loader,
    device: torch.device,
    *,
    sigma_x: float,
    beta: float,
    include_ssim: bool,
    prefix: str,
    return_records: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate all, observed, and gap partitions in one deterministic pass."""

    model.eval()
    accumulators: dict[str, dict[str, list[torch.Tensor]]] = {
        "all": defaultdict(list),
        "observed": defaultdict(list),
        "gap": defaultdict(list),
    }
    records: list[dict[str, Any]] = []
    for images, metadata in loader:
        images = images.to(device, non_blocking=True)
        reconstructions, parameters = model.deterministic_reconstruction(
            images
        )
        kl = model.posterior.kl(parameters).float()
        vectors = reconstruction_metric_vectors(
            images,
            reconstructions,
            sigma_x,
            include_ssim=include_ssim,
        )
        vectors["kl"] = kl
        vectors["total_objective"] = (
            vectors["reconstruction_nll"] + beta * kl
        )
        scale = parameters.scale.reshape(parameters.scale.shape[0], -1)
        if scale.shape[1] == 1:
            vectors["posterior_scale"] = scale[:, 0]
        else:
            vectors["posterior_scale"] = scale.mean(dim=1)
            vectors["posterior_scale_max"] = scale.max(dim=1).values
        expected_cosine = model.posterior.expected_cosine(parameters)
        if expected_cosine is not None:
            vectors["expected_cosine_to_mode"] = expected_cosine

        gap_mask = metadata["is_gap"].to(torch.bool)
        masks = {
            "all": torch.ones_like(gap_mask),
            "observed": ~gap_mask,
            "gap": gap_mask,
        }
        for partition, mask in masks.items():
            _append_masked(accumulators[partition], vectors, mask)

        if return_records:
            cpu_vectors = {
                key: value.detach().cpu().tolist()
                for key, value in vectors.items()
            }
            batch_size = images.shape[0]
            for row_index in range(batch_size):
                row = {
                    "partition": (
                        f"{prefix}_gap"
                        if int(metadata["is_gap"][row_index])
                        else f"{prefix}_observed"
                    ),
                    "category": int(metadata["category"][row_index]),
                    "instance": int(metadata["instance"][row_index]),
                    "elevation": int(metadata["elevation"][row_index]),
                    "azimuth_index": int(
                        metadata["azimuth_index"][row_index]
                    ),
                    "azimuth_degrees": int(
                        metadata["azimuth_degrees"][row_index]
                    ),
                    "lighting": int(metadata["lighting"][row_index]),
                    "source_index": int(
                        metadata["source_index"][row_index]
                    ),
                }
                row.update(
                    {
                        key: values[row_index]
                        for key, values in cpu_vectors.items()
                    }
                )
                records.append(row)

    summaries = {
        prefix: _summary_from_accumulator(accumulators["all"]),
        f"{prefix}_observed": _summary_from_accumulator(
            accumulators["observed"]
        ),
        f"{prefix}_gap": _summary_from_accumulator(
            accumulators["gap"]
        ),
    }
    for summary in summaries.values():
        summary["posterior_parameter"] = _parameter_name(model.family)
        summary["deterministic_reconstruction"] = (
            "decoder evaluated at posterior location or Gaussian mean"
        )
    return summaries, records


@torch.no_grad()
def posterior_sample_diagnostics(
    model: SmallNORBViewVAE,
    loader,
    device: torch.device,
    *,
    max_examples: int = 256,
    samples_per_example: int = 64,
    seed: int = 13_579,
) -> dict[str, Any]:
    """Estimate remote mass with a fixed subset and Monte Carlo budget."""

    if not model.posterior.is_spherical:
        return {
            "applicable": False,
            "reason": "opposite hemisphere is a spherical diagnostic",
        }
    set_global_seed(seed)
    locations = []
    parameter_batches = []
    count = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        parameters = model.encode(images)
        remaining = max_examples - count
        take = min(remaining, images.shape[0])
        locations.append(parameters.location[:take])
        parameter_batches.append(
            type(parameters)(
                location=parameters.location[:take],
                scale=parameters.scale[:take],
                raw_scale=parameters.raw_scale[:take],
            )
        )
        count += take
        if count >= max_examples:
            break
    if not locations:
        return {"applicable": True, "count": 0}
    all_locations = torch.cat(locations, dim=0)
    samples = []
    for parameters in parameter_batches:
        draws = torch.stack(
            [
                model.posterior.sample(parameters)
                for _ in range(samples_per_example)
            ],
            dim=0,
        )
        samples.append(draws)
    all_samples = torch.cat(samples, dim=1)
    return {
        "applicable": True,
        "examples": int(all_locations.shape[0]),
        "samples_per_example": samples_per_example,
        "seed": seed,
        "fraction_in_opposite_hemisphere": (
            opposite_hemisphere_fraction(all_samples, all_locations)
        ),
    }


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[SmallNORBViewVAE, dict[str, Any], RunConfig]:
    payload = torch.load(
        Path(checkpoint_path),
        map_location=device,
        weights_only=False,
    )
    config = RunConfig.from_dict(payload["config"])
    model = build_model(config, device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload, config


def evaluate_run(
    run_dir: str | Path,
    device: torch.device,
    *,
    include_test: bool,
    include_train: bool = False,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    checkpoint = run_path / "checkpoint_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model, payload, config = load_checkpoint(checkpoint, device)
    loaders = build_dataloaders(config, include_test=include_test)
    set_global_seed(config.seed)
    validation, validation_records = evaluate_partitions(
        model,
        loaders["validation"],
        device,
        sigma_x=config.sigma_x,
        beta=config.beta_target,
        include_ssim=True,
        prefix="validation",
        return_records=True,
    )
    summary: dict[str, Any] = {
        "config": config.to_dict(),
        "checkpoint": repo_relative(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "selection_rule": (
            "lowest deterministic validation reconstruction NLL over all "
            "validation azimuths, total objective as tie breaker"
        ),
        "validation": validation,
        "test_was_accessed": bool(include_test),
    }
    write_csv(
        run_path / "evaluation_records_validation.csv",
        validation_records,
    )
    if include_train:
        set_global_seed(config.seed)
        train, train_records = evaluate_partitions(
            model,
            loaders["train"],
            device,
            sigma_x=config.sigma_x,
            beta=config.beta_target,
            include_ssim=True,
            prefix="train",
            return_records=True,
        )
        summary["train"] = train
        write_csv(
            run_path / "evaluation_records_train.csv", train_records
        )
    diagnostic_loader = loaders["validation"]
    if include_test:
        set_global_seed(config.seed)
        test, test_records = evaluate_partitions(
            model,
            loaders["test"],
            device,
            sigma_x=config.sigma_x,
            beta=config.beta_target,
            include_ssim=True,
            prefix="test",
            return_records=True,
        )
        summary["test"] = test
        write_csv(
            run_path / "evaluation_records_test.csv", test_records
        )
        diagnostic_loader = loaders["test"]
    summary["posterior_sample_diagnostics"] = (
        posterior_sample_diagnostics(
            model, diagnostic_loader, device
        )
    )
    output = run_path / "evaluation_summary.json"
    write_json(output, summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a smallNORB run without test leakage."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Access official test instances only after setup is frozen.",
    )
    parser.add_argument(
        "--include-train",
        action="store_true",
        help="Also report deterministic metrics on the training split.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = evaluate_run(
        args.run_dir,
        resolve_device(args.device),
        include_test=args.include_test,
        include_train=args.include_train,
    )
    print(
        {
            "checkpoint_epoch": summary["checkpoint_epoch"],
            "test_was_accessed": summary["test_was_accessed"],
        }
    )


if __name__ == "__main__":
    main()
