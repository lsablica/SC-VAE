from __future__ import annotations

import numpy as np

from experiments.smallnorb.probe import (
    fit_pose_probe,
    geometry_alignment,
)


def _circular_dataset(instances: tuple[int, ...]) -> dict[str, np.ndarray]:
    rows = [
        (instance, azimuth)
        for instance in instances
        for azimuth in range(18)
    ]
    azimuth_index = np.asarray(
        [azimuth for _, azimuth in rows], dtype=np.int16
    )
    degrees = azimuth_index * 20
    radians = np.deg2rad(degrees)
    location = np.stack(
        (np.cos(radians), np.sin(radians)), axis=1
    )
    return {
        "location": location,
        "category": np.zeros(len(rows), dtype=np.int16),
        "instance": np.asarray(
            [instance for instance, _ in rows], dtype=np.int16
        ),
        "elevation": np.zeros(len(rows), dtype=np.int16),
        "lighting": np.zeros(len(rows), dtype=np.int16),
        "azimuth_index": azimuth_index,
        "azimuth_degrees": degrees,
        "is_gap": np.isin(azimuth_index, (16, 17, 0, 1)).astype(
            np.int16
        ),
        "source_index": np.arange(len(rows), dtype=np.int32),
    }


def test_pose_probe_recovers_an_exact_circular_embedding():
    train = _circular_dataset((4, 6, 7, 8))
    observed = train["is_gap"] == 0
    train = {
        key: value[observed]
        for key, value in train.items()
    }
    validation = _circular_dataset((9,))
    summary, _ = fit_pose_probe(train, validation)
    assert summary["partitions"]["validation_gap"][
        "mean_absolute_error_degrees"
    ] < 1e-3


def test_geometry_alignment_is_exact_for_a_circular_embedding():
    validation = _circular_dataset((9,))
    summary, rows = geometry_alignment(
        validation,
        spherical=True,
        gaussian_mean=None,
        gaussian_std=None,
        seed=123,
        pair_count=1_000,
    )
    assert len(rows) == 153
    # Numerically equal circular distances can receive slightly different
    # floating-point ranks after arccos, but the ordering stays effectively
    # perfect.
    assert summary["all_pairs"]["spearman"] > 0.99
    assert summary["pairs_crossing_gap"]["spearman"] > 0.99
