from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.smallnorb.config import RunConfig
from experiments.smallnorb.evaluate import evaluate_run
from experiments.smallnorb.train import train_run
from experiments.smallnorb.utils import sha256_file, write_json


def _write_source(
    processed: Path,
    source: str,
    instances: tuple[int, ...],
) -> None:
    rows = []
    for index in range(36):
        instance = instances[index % len(instances)]
        azimuth = index % 18
        rows.append(
            {
                "category": index % 5,
                "instance": instance,
                "elevation": index % 9,
                "azimuth_index": azimuth,
                "azimuth_degrees": azimuth * 20,
                "lighting": index % 6,
                "source_index": index,
            }
        )
    images = np.random.default_rng(0).integers(
        0, 256, size=(len(rows), 64, 64), dtype=np.uint8
    )
    np.save(processed / f"{source}_left_64_uint8.npy", images)
    np.savez_compressed(
        processed / f"{source}_metadata.npz",
        **{
            key: np.asarray([row[key] for row in rows], dtype=np.int16)
            for key in rows[0]
        },
    )


def test_two_epoch_smoke_pipeline_writes_required_artifacts(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "data"
    processed = data_root / "processed"
    processed.mkdir(parents=True)
    _write_source(processed, "training", (4, 6, 7, 8, 9))
    _write_source(processed, "testing", (0, 1, 2, 3, 5))
    cache_manifest = processed / "cache_manifest.json"
    write_json(cache_manifest, {"synthetic": True})
    monkeypatch.setattr(
        "experiments.smallnorb.train.assert_expected_split_counts",
        lambda _: {"synthetic": 72},
    )
    config = RunConfig(
        family="spcauchy",
        seed=0,
        stage="stage0",
        run_name="smoke",
        epochs=2,
        batch_size=8,
        num_workers=0,
        mixed_precision=False,
        train_limit=8,
        validation_limit=8,
        data_root=str(data_root),
        output_root=str(tmp_path / "outputs"),
        evaluate_ssim_every_epochs=1,
    )
    result = train_run(
        config,
        torch.device("cpu"),
        command="unit-test smoke",
    )
    run_dir = config.run_dir
    assert result["status"] == "completed"
    evaluate_run(run_dir, torch.device("cpu"), include_test=False)
    required = {
        "config.json",
        "environment.txt",
        "commands.txt",
        "history.csv",
        "history.json",
        "selection_summary.json",
        "evaluation_summary.json",
        "checkpoint_best.pt",
        "checkpoint_last.pt",
        "seed_manifest.json",
    }
    assert required.issubset(
        {path.name for path in run_dir.iterdir()}
    )
    seed_manifest = json.loads(
        (run_dir / "seed_manifest.json").read_text()
    )
    assert seed_manifest["official_test_accessed"] is False
    assert (
        seed_manifest["data_cache_manifest_sha256"]
        == sha256_file(cache_manifest)
    )
