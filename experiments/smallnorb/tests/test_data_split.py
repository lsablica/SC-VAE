from __future__ import annotations

import itertools

import numpy as np

from experiments.smallnorb.config import (
    DEFAULT_DATA_ROOT,
    EXPECTED_SPLIT_COUNTS,
    GAP_AZIMUTH_INDICES,
    RunConfig,
)
from experiments.smallnorb.data import (
    build_datasets,
    read_matrix,
    split_indices,
    synthetic_matrix_bytes,
)


def _metadata(instances: tuple[int, ...]) -> dict[str, np.ndarray]:
    rows = list(
        itertools.product(
            range(5),
            instances,
            range(9),
            range(18),
            range(6),
        )
    )
    values = np.asarray(rows, dtype=np.int16)
    return {
        "category": values[:, 0],
        "instance": values[:, 1],
        "elevation": values[:, 2],
        "azimuth_index": values[:, 3],
        "azimuth_degrees": values[:, 3] * 20,
        "lighting": values[:, 4],
        "source_index": np.arange(len(values), dtype=np.int32),
    }


def test_binary_matrix_parser_round_trip():
    integer = np.arange(24, dtype=np.int32).reshape(6, 4)
    image = np.arange(2 * 3 * 4 * 5, dtype=np.uint8).reshape(2, 3, 4, 5)
    assert np.array_equal(
        read_matrix(synthetic_matrix_bytes(integer, 0x1E3D4C54)),
        integer,
    )
    assert np.array_equal(
        read_matrix(synthetic_matrix_bytes(image, 0x1E3D4C55)),
        image,
    )


def test_official_instance_and_gap_split_counts():
    training = _metadata((4, 6, 7, 8, 9))
    testing = _metadata((0, 1, 2, 3, 5))
    counts = {
        split: len(split_indices("training", training, split))
        for split in (
            "train",
            "validation",
            "validation_observed",
            "validation_gap",
        )
    }
    counts.update(
        {
            split: len(split_indices("testing", testing, split))
            for split in ("test", "test_observed", "test_gap")
        }
    )
    assert counts == EXPECTED_SPLIT_COUNTS


def test_training_excludes_every_gap_image_and_validation_retains_them():
    metadata = _metadata((4, 6, 7, 8, 9))
    train = split_indices("training", metadata, "train")
    validation_gap = split_indices(
        "training", metadata, "validation_gap"
    )
    assert not np.isin(
        metadata["azimuth_index"][train], GAP_AZIMUTH_INDICES
    ).any()
    assert np.isin(
        metadata["azimuth_index"][validation_gap],
        GAP_AZIMUTH_INDICES,
    ).all()
    assert set(np.unique(metadata["instance"][train])) == {4, 6, 7, 8}
    assert set(np.unique(metadata["instance"][validation_gap])) == {9}


def test_archived_data_path_falls_back_to_clone_cache():
    config = RunConfig.from_dict(
        {
            "family": "spcauchy",
            "seed": 0,
            "data_root": (
                "/definitely/not/the/current/clone/data/smallnorb"
            ),
        }
    )
    assert config.data_root == str(DEFAULT_DATA_ROOT)


def test_validation_only_dataset_build_never_constructs_test_split(
    monkeypatch,
):
    constructed: list[str] = []

    class RecordingDataset:
        def __init__(self, _root, split, **_kwargs):
            constructed.append(split)

    monkeypatch.setattr(
        "experiments.smallnorb.data.SmallNORBViewDataset",
        RecordingDataset,
    )
    config = RunConfig(family="spcauchy", seed=0)
    datasets = build_datasets(config, include_test=False)
    assert set(datasets) == {
        "train",
        "validation",
        "validation_observed",
        "validation_gap",
    }
    assert all(not split.startswith("test") for split in constructed)

    constructed.clear()
    datasets = build_datasets(config, include_test=True)
    assert {"test", "test_observed", "test_gap"} <= set(datasets)
