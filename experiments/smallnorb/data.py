"""Official smallNORB parser, preprocessing cache, and fixed angular split."""

from __future__ import annotations

import gzip
import io
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import (
    DEFAULT_DATA_ROOT,
    EXPECTED_SPLIT_COUNTS,
    GAP_AZIMUTH_INDICES,
    TEST_INSTANCES,
    TRAIN_INSTANCES,
    VALIDATION_INSTANCES,
    RunConfig,
)
from .utils import ensure_dir, sha256_file, write_json


OFFICIAL_BASE_URL = (
    "https://cs.nyu.edu/~yann/data/norb-v1.0-small"
)
FILE_STEMS = {
    "training": "smallnorb-5x46789x9x18x6x2x96x96-training",
    "testing": "smallnorb-5x01235x9x18x6x2x96x96-testing",
}
MATRIX_DTYPES = {
    0x1E3D4C51: np.dtype("<f4"),
    0x1E3D4C53: np.dtype("<f8"),
    0x1E3D4C54: np.dtype("<i4"),
    0x1E3D4C55: np.dtype("u1"),
    0x1E3D4C56: np.dtype("<i2"),
}


@dataclass(frozen=True)
class MatrixHeader:
    magic: int
    dtype: np.dtype
    shape: tuple[int, ...]
    header_bytes: int


def read_matrix_header(handle: BinaryIO) -> MatrixHeader:
    """Parse the official little-endian smallNORB matrix header."""

    prefix = handle.read(8)
    if len(prefix) != 8:
        raise ValueError("Truncated smallNORB matrix header")
    magic, ndim = struct.unpack("<ii", prefix)
    if magic not in MATRIX_DTYPES:
        raise ValueError(f"Unsupported smallNORB magic number: {magic:#x}")
    if not 1 <= ndim <= 8:
        raise ValueError(f"Invalid smallNORB matrix rank: {ndim}")
    stored_dimensions = max(ndim, 3)
    dimensions_raw = handle.read(4 * stored_dimensions)
    if len(dimensions_raw) != 4 * stored_dimensions:
        raise ValueError("Truncated smallNORB dimension header")
    dimensions = struct.unpack(
        f"<{stored_dimensions}i", dimensions_raw
    )
    shape = tuple(int(value) for value in dimensions[:ndim])
    if any(value <= 0 for value in shape):
        raise ValueError(f"Invalid smallNORB matrix shape: {shape}")
    return MatrixHeader(
        magic=magic,
        dtype=MATRIX_DTYPES[magic],
        shape=shape,
        header_bytes=8 + 4 * stored_dimensions,
    )


def read_matrix(path_or_handle: str | Path | BinaryIO) -> np.ndarray:
    """Read a complete smallNORB matrix, including gzip files."""

    close = False
    if hasattr(path_or_handle, "read"):
        handle = path_or_handle
    else:
        path = Path(path_or_handle)
        handle = (
            gzip.open(path, "rb")
            if path.suffix == ".gz"
            else path.open("rb")
        )
        close = True
    try:
        header = read_matrix_header(handle)
        count = int(np.prod(header.shape))
        payload = handle.read()
        expected = count * header.dtype.itemsize
        if len(payload) != expected:
            raise ValueError(
                f"Matrix payload has {len(payload)} bytes, expected {expected}"
            )
        return np.frombuffer(payload, dtype=header.dtype).reshape(
            header.shape
        )
    finally:
        if close:
            handle.close()


def _raw_path(data_root: Path, source: str, kind: str) -> Path:
    return (
        data_root
        / "raw"
        / f"{FILE_STEMS[source]}-{kind}.mat.gz"
    )


def download_official_smallnorb(
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> list[Path]:
    """Download missing official files without replacing existing data."""

    root = Path(data_root)
    ensure_dir(root / "raw")
    paths: list[Path] = []
    for source in ("training", "testing"):
        for kind in ("dat", "cat", "info"):
            path = _raw_path(root, source, kind)
            paths.append(path)
            if path.exists() and path.stat().st_size > 0:
                continue
            url = f"{OFFICIAL_BASE_URL}/{path.name}"
            temporary = path.with_suffix(path.suffix + ".partial")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
    return paths


def _stream_preprocessed_images(
    raw_path: Path,
    output_path: Path,
    chunk_size: int = 256,
) -> tuple[int, tuple[int, ...]]:
    """Stream stereo images and cache antialiased left-camera tensors."""

    with gzip.open(raw_path, "rb") as handle:
        header = read_matrix_header(handle)
        if len(header.shape) != 4 or header.shape[1:] != (2, 96, 96):
            raise ValueError(
                f"Unexpected smallNORB image shape: {header.shape}"
            )
        count = header.shape[0]
        output = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.uint8,
            shape=(count, 64, 64),
        )
        pair_bytes = 2 * 96 * 96
        for start in range(0, count, chunk_size):
            current = min(chunk_size, count - start)
            payload = handle.read(current * pair_bytes)
            if len(payload) != current * pair_bytes:
                raise ValueError(
                    f"Truncated image payload at record {start}"
                )
            stereo = np.frombuffer(payload, dtype=np.uint8).reshape(
                current, 2, 96, 96
            )
            left = torch.from_numpy(stereo[:, :1].copy()).float()
            cropped = left[:, :, 8:88, 8:88] / 255.0
            resized = F.interpolate(
                cropped,
                size=(64, 64),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            output[start : start + current] = (
                resized[:, 0].mul(255.0).round().clamp(0, 255)
                .to(torch.uint8)
                .numpy()
            )
        trailing = handle.read(1)
        if trailing:
            raise ValueError("Unexpected trailing image bytes")
        output.flush()
    return count, header.shape


def _normalized_metadata(
    categories: np.ndarray,
    information: np.ndarray,
) -> dict[str, np.ndarray]:
    categories = np.asarray(categories).reshape(-1).astype(np.int16)
    information = np.asarray(information)
    if information.shape != (categories.size, 4):
        raise ValueError(
            "smallNORB category and information matrices are misaligned"
        )
    raw_azimuth = information[:, 2].astype(np.int16)
    if np.any(raw_azimuth % 2):
        raise ValueError("smallNORB azimuth codes must be even")
    azimuth_index = raw_azimuth // 2
    metadata = {
        "category": categories,
        "instance": information[:, 0].astype(np.int16),
        "elevation": information[:, 1].astype(np.int16),
        "azimuth_index": azimuth_index,
        "azimuth_degrees": (raw_azimuth * 10).astype(np.int16),
        "lighting": information[:, 3].astype(np.int16),
        "source_index": np.arange(categories.size, dtype=np.int32),
    }
    return metadata


def prepare_smallnorb_cache(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    *,
    download: bool = False,
    force: bool = False,
) -> Path:
    """Create verified uint8 image and metadata caches."""

    root = Path(data_root)
    if download:
        download_official_smallnorb(root)
    cache_root = ensure_dir(root / "processed")
    source_records: dict[str, dict] = {}
    for source in ("training", "testing"):
        raw_paths = {
            kind: _raw_path(root, source, kind)
            for kind in ("dat", "cat", "info")
        }
        missing = [path for path in raw_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing official smallNORB files: "
                + ", ".join(str(path) for path in missing)
            )
        image_path = cache_root / f"{source}_left_64_uint8.npy"
        metadata_path = cache_root / f"{source}_metadata.npz"
        if force or not image_path.exists() or not metadata_path.exists():
            count, original_shape = _stream_preprocessed_images(
                raw_paths["dat"], image_path
            )
            categories = read_matrix(raw_paths["cat"])
            information = read_matrix(raw_paths["info"])
            metadata = _normalized_metadata(categories, information)
            if count != len(metadata["category"]):
                raise ValueError(
                    "Processed images and metadata have different lengths"
                )
            np.savez_compressed(metadata_path, **metadata)
        images = np.load(image_path, mmap_mode="r")
        with np.load(metadata_path) as archive:
            metadata_count = len(archive["category"])
            alignment_digest = _metadata_alignment_digest(archive)
        if images.shape != (metadata_count, 64, 64):
            raise ValueError(
                f"Invalid processed cache alignment for {source}"
            )
        source_records[source] = {
            "count": metadata_count,
            "processed_image_shape": list(images.shape),
            "image_sha256": sha256_file(image_path),
            "metadata_sha256": sha256_file(metadata_path),
            "metadata_alignment_sha256": alignment_digest,
            "raw": {
                kind: {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "source_url": f"{OFFICIAL_BASE_URL}/{path.name}",
                }
                for kind, path in raw_paths.items()
            },
        }
    manifest = {
        "schema_version": 1,
        "camera": "left",
        "input_range": [0.0, 1.0],
        "preprocessing": {
            "center_crop": [8, 8, 80, 80],
            "resize": [64, 64],
            "resize_mode": "bilinear",
            "antialias": True,
            "cache_dtype": "uint8",
        },
        "sources": source_records,
    }
    manifest_path = cache_root / "cache_manifest.json"
    write_json(manifest_path, manifest)
    assert_expected_split_counts(root)
    return manifest_path


def _metadata_alignment_digest(archive) -> str:
    import hashlib

    digest = hashlib.sha256()
    keys = (
        "category",
        "instance",
        "elevation",
        "azimuth_index",
        "azimuth_degrees",
        "lighting",
        "source_index",
    )
    for key in keys:
        values = np.asarray(archive[key])
        digest.update(key.encode("ascii"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def split_indices(
    source: str,
    metadata: dict[str, np.ndarray],
    split: str,
    gap_azimuth_indices: tuple[int, ...] = GAP_AZIMUTH_INDICES,
) -> np.ndarray:
    """Return the authoritative instance and angular split indices."""

    instances = np.asarray(metadata["instance"])
    azimuth = np.asarray(metadata["azimuth_index"])
    in_gap = np.isin(azimuth, np.asarray(gap_azimuth_indices))
    if source == "training":
        if split == "train":
            mask = np.isin(instances, TRAIN_INSTANCES) & ~in_gap
        elif split == "validation":
            mask = np.isin(instances, VALIDATION_INSTANCES)
        elif split == "validation_observed":
            mask = np.isin(instances, VALIDATION_INSTANCES) & ~in_gap
        elif split == "validation_gap":
            mask = np.isin(instances, VALIDATION_INSTANCES) & in_gap
        else:
            raise KeyError(f"Unsupported training-source split: {split}")
    elif source == "testing":
        base = np.isin(instances, TEST_INSTANCES)
        if split == "test":
            mask = base
        elif split == "test_observed":
            mask = base & ~in_gap
        elif split == "test_gap":
            mask = base & in_gap
        else:
            raise KeyError(f"Unsupported testing-source split: {split}")
    else:
        raise KeyError(source)
    return np.flatnonzero(mask).astype(np.int64)


def _load_metadata(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key].copy() for key in archive.files}


def assert_expected_split_counts(
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> dict[str, int]:
    root = Path(data_root) / "processed"
    training = _load_metadata(root / "training_metadata.npz")
    testing = _load_metadata(root / "testing_metadata.npz")
    counts = {}
    for split in (
        "train",
        "validation",
        "validation_observed",
        "validation_gap",
    ):
        counts[split] = len(split_indices("training", training, split))
    for split in ("test", "test_observed", "test_gap"):
        counts[split] = len(split_indices("testing", testing, split))
    if counts != EXPECTED_SPLIT_COUNTS:
        raise AssertionError(
            f"smallNORB split counts differ: {counts}"
        )
    return counts


def deterministic_subset(
    indices: np.ndarray,
    limit: int | None,
    seed: int,
) -> np.ndarray:
    if limit is None or limit >= len(indices):
        return indices
    if limit <= 0:
        raise ValueError("subset limit must be positive")
    generator = np.random.default_rng(seed)
    selected = generator.permutation(indices)[:limit]
    return np.sort(selected)


class SmallNORBViewDataset(Dataset):
    """Memory-mapped left-camera images with aligned viewpoint metadata."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        *,
        gap_azimuth_indices: tuple[int, ...] = GAP_AZIMUTH_INDICES,
        limit: int | None = None,
        subset_seed: int = 0,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.source = (
            "testing" if split.startswith("test") else "training"
        )
        processed = self.data_root / "processed"
        self.images = np.load(
            processed / f"{self.source}_left_64_uint8.npy",
            mmap_mode="r",
        )
        self.metadata = _load_metadata(
            processed / f"{self.source}_metadata.npz"
        )
        indices = split_indices(
            self.source,
            self.metadata,
            split,
            gap_azimuth_indices=gap_azimuth_indices,
        )
        self.indices = deterministic_subset(indices, limit, subset_seed)
        self.gap_azimuth_indices = tuple(gap_azimuth_indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source_index = int(self.indices[index])
        image = (
            torch.from_numpy(
                np.asarray(self.images[source_index]).copy()
            )
            .unsqueeze(0)
            .float()
            .div_(255.0)
        )
        metadata = {
            key: int(values[source_index])
            for key, values in self.metadata.items()
        }
        metadata["is_gap"] = int(
            metadata["azimuth_index"] in self.gap_azimuth_indices
        )
        return image, metadata


def build_datasets(
    config: RunConfig,
    *,
    include_test: bool = False,
) -> dict[str, SmallNORBViewDataset]:
    limits = {
        "train": config.train_limit,
        "validation": config.validation_limit,
        "validation_observed": config.validation_limit,
        "validation_gap": config.validation_limit,
        "test": config.test_limit,
        "test_observed": config.test_limit,
        "test_gap": config.test_limit,
    }
    splits = [
        "train",
        "validation",
        "validation_observed",
        "validation_gap",
    ]
    if include_test:
        splits.extend(["test", "test_observed", "test_gap"])
    return {
        split: SmallNORBViewDataset(
            config.data_root,
            split,
            gap_azimuth_indices=config.gap_azimuth_indices,
            limit=limits[split],
            subset_seed=config.seed,
        )
        for split in splits
    }


def build_dataloaders(
    config: RunConfig,
    *,
    include_test: bool = False,
) -> dict[str, DataLoader]:
    datasets = build_datasets(config, include_test=include_test)
    splits = [
        "train",
        "validation",
        "validation_observed",
        "validation_gap",
    ]
    if include_test:
        splits.extend(["test", "test_observed", "test_gap"])
    loaders: dict[str, DataLoader] = {}
    for split in splits:
        generator = torch.Generator().manual_seed(config.seed)
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=config.batch_size,
            shuffle=split == "train",
            generator=generator,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=(
                config.persistent_workers and config.num_workers > 0
            ),
            drop_last=False,
        )
    return loaders


def synthetic_matrix_bytes(
    array: np.ndarray,
    magic: int,
) -> io.BytesIO:
    """Build an in-memory matrix used by parser unit tests."""

    rank = array.ndim
    dimensions = list(array.shape)
    while len(dimensions) < 3:
        dimensions.append(1)
    payload = (
        struct.pack("<ii", magic, rank)
        + struct.pack(f"<{len(dimensions)}i", *dimensions)
        + np.asarray(array, dtype=MATRIX_DTYPES[magic]).tobytes(order="C")
    )
    return io.BytesIO(payload)
