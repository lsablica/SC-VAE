from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.smiles.chemistry import canonicalize_smiles
from experiments.smiles.config import SPECIAL_TOKENS


RAW_DATASET_FILENAMES = (
    "250k_rndm_zinc_drugs_clean_3.csv",
    "zinc250k.csv",
    "zinc250k.smi",
    "zinc250k.txt",
)
SMILES_COLUMN_CANDIDATES = ("smiles", "SMILES", "canon_smiles", "canonical_smiles", "mol_smiles")


@dataclass
class Vocabulary:
    token_to_idx: dict[str, int]

    @property
    def idx_to_token(self) -> dict[int, str]:
        return {idx: token for token, idx in self.token_to_idx.items()}

    @property
    def pad_token_id(self) -> int:
        return self.token_to_idx["<pad>"]

    @property
    def sos_token_id(self) -> int:
        return self.token_to_idx["<sos>"]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_idx["<eos>"]

    @property
    def size(self) -> int:
        return len(self.token_to_idx)


@dataclass
class PreparedDataBundle:
    raw_dir: Path
    processed_dir: Path
    split_paths: dict[str, Path]
    token_paths: dict[str, Path]
    vocabulary: Vocabulary
    max_seq_len: int
    metadata: dict


class SmilesDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, token_ids: np.ndarray):
        self.frame = frame.reset_index(drop=True).copy()
        self.token_ids = token_ids
        if len(self.frame) != len(self.token_ids):
            raise ValueError("Frame and token cache have different lengths.")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        token_ids = torch.as_tensor(np.asarray(self.token_ids[index], dtype=np.int64), dtype=torch.long)
        return {
            "token_ids": token_ids,
            "canonical_smiles": row["canonical_smiles"],
            "source_smiles": row.get("source_smiles", row["canonical_smiles"]),
        }


def find_zinc250k_raw_file(raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    for filename in RAW_DATASET_FILENAMES:
        candidate = raw_dir / filename
        if candidate.exists():
            return candidate
    supported = ", ".join(RAW_DATASET_FILENAMES)
    raise FileNotFoundError(f"Could not find ZINC-250k raw file in {raw_dir}. Expected one of: {supported}")


def _infer_smiles_column(frame: pd.DataFrame) -> str:
    for candidate in SMILES_COLUMN_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    if frame.shape[1] == 1:
        return str(frame.columns[0])
    object_columns = [column for column in frame.columns if frame[column].dtype == object]
    if len(object_columns) == 1:
        return object_columns[0]
    raise ValueError(
        "Unable to infer the SMILES column in the ZINC-250k CSV. "
        f"Columns were: {list(frame.columns)}"
    )


def read_zinc250k_smiles(raw_path: str | Path) -> list[str]:
    raw_path = Path(raw_path)
    if raw_path.suffix.lower() == ".csv":
        frame = pd.read_csv(raw_path)
        smiles_column = _infer_smiles_column(frame)
        return frame[smiles_column].astype(str).tolist()
    return [line.strip() for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _canonicalize_smiles_rows(smiles_list: Iterable[str]) -> pd.DataFrame:
    rows = []
    for smiles in smiles_list:
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            continue
        rows.append({"source_smiles": str(smiles), "canonical_smiles": canonical})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No valid SMILES remained after RDKit canonicalization.")
    frame = frame.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
    return frame


def build_vocabulary(smiles_iterable: Iterable[str]) -> Vocabulary:
    tokens = list(SPECIAL_TOKENS)
    charset = sorted({char for smiles in smiles_iterable for char in smiles})
    tokens.extend(charset)
    return Vocabulary(token_to_idx={token: idx for idx, token in enumerate(tokens)})


def _validate_vocab_coverage(frame: pd.DataFrame, vocabulary: Vocabulary) -> pd.DataFrame:
    allowed = set(vocabulary.token_to_idx)
    mask = frame["canonical_smiles"].map(lambda smiles: set(smiles).issubset(allowed))
    return frame.loc[mask].reset_index(drop=True)


def _subset_frame(frame: pd.DataFrame, max_samples: int | None) -> pd.DataFrame:
    if max_samples is None:
        return frame
    return frame.iloc[:max_samples].reset_index(drop=True)


def _filter_by_max_smiles_length(frame: pd.DataFrame, max_smiles_length: int | None) -> pd.DataFrame:
    if max_smiles_length is None:
        return frame
    mask = frame["canonical_smiles"].map(len) <= max_smiles_length
    return frame.loc[mask].reset_index(drop=True)


def encode_smiles_to_array(smiles_iterable: Iterable[str], vocabulary: Vocabulary, max_seq_len: int) -> np.ndarray:
    smiles_list = list(smiles_iterable)
    encoded = np.full((len(smiles_list), max_seq_len), vocabulary.pad_token_id, dtype=np.int32)
    for row_idx, smiles in enumerate(smiles_list):
        token_ids = [vocabulary.sos_token_id]
        token_ids.extend(vocabulary.token_to_idx[ch] for ch in smiles)
        token_ids.append(vocabulary.eos_token_id)
        if len(token_ids) > max_seq_len:
            raise ValueError(f"SMILES exceeds configured max_seq_len={max_seq_len}: {smiles}")
        encoded[row_idx, : len(token_ids)] = token_ids
    return encoded


def _save_split_artifacts(
    frame: pd.DataFrame,
    *,
    csv_path: Path,
    token_path: Path,
    vocabulary: Vocabulary,
    max_seq_len: int,
) -> None:
    frame.to_csv(csv_path, index=False)
    np.save(token_path, encode_smiles_to_array(frame["canonical_smiles"].tolist(), vocabulary, max_seq_len))


def _build_requested_cache_settings(
    *,
    validation_fraction: float,
    test_fraction: float,
    split_seed: int,
    max_smiles_length: int | None,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
) -> dict:
    return {
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "split_seed": split_seed,
        "max_smiles_length": max_smiles_length,
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "max_test_samples": max_test_samples,
    }


def _split_dataset(
    frame: pd.DataFrame,
    *,
    validation_fraction: float,
    test_fraction: float,
    split_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation_fraction and test_fraction must both be positive for the ZINC split.")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than 1.0.")

    rng = np.random.default_rng(split_seed)
    indices = np.arange(len(frame))
    rng.shuffle(indices)

    total_count = len(indices)
    test_size = max(1, int(round(total_count * test_fraction)))
    val_size = max(1, int(round(total_count * validation_fraction)))
    if test_size + val_size >= total_count:
        raise ValueError("Split fractions left no room for the training set.")

    test_indices = np.sort(indices[:test_size])
    val_indices = np.sort(indices[test_size : test_size + val_size])
    train_indices = np.sort(indices[test_size + val_size :])
    return (
        frame.iloc[train_indices].reset_index(drop=True),
        frame.iloc[val_indices].reset_index(drop=True),
        frame.iloc[test_indices].reset_index(drop=True),
    )


def prepare_zinc250k_dataset(
    raw_dir: str | Path,
    processed_dir: str | Path,
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    split_seed: int = 13,
    max_smiles_length: int | None = 68,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    force_reprocess: bool = False,
) -> PreparedDataBundle:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = processed_dir / "metadata.json"
    vocab_path = processed_dir / "vocab.json"
    split_paths = {
        "train": processed_dir / "train.csv",
        "val": processed_dir / "val.csv",
        "test": processed_dir / "test.csv",
    }
    token_paths = {
        "train": processed_dir / "train_tokens.npy",
        "val": processed_dir / "val_tokens.npy",
        "test": processed_dir / "test_tokens.npy",
    }

    requested_cache_settings = _build_requested_cache_settings(
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        split_seed=split_seed,
        max_smiles_length=max_smiles_length,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
    )

    if (
        not force_reprocess
        and metadata_path.exists()
        and vocab_path.exists()
        and all(path.exists() for path in split_paths.values())
        and all(path.exists() for path in token_paths.values())
    ):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_settings", {}) == requested_cache_settings:
            vocabulary = Vocabulary(token_to_idx=json.loads(vocab_path.read_text(encoding="utf-8")))
            return PreparedDataBundle(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                split_paths=split_paths,
                token_paths=token_paths,
                vocabulary=vocabulary,
                max_seq_len=int(metadata["max_seq_len"]),
                metadata=metadata,
            )

    raw_path = find_zinc250k_raw_file(raw_dir)
    raw_smiles = read_zinc250k_smiles(raw_path)
    canonical_frame = _canonicalize_smiles_rows(raw_smiles)
    canonical_frame = _filter_by_max_smiles_length(canonical_frame, max_smiles_length)
    if canonical_frame.empty:
        raise ValueError("No molecules remained after applying the max_smiles_length filter.")
    train_frame, val_frame, test_frame = _split_dataset(
        canonical_frame,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        split_seed=split_seed,
    )

    vocabulary = build_vocabulary(train_frame["canonical_smiles"])
    train_frame = _validate_vocab_coverage(train_frame, vocabulary)
    val_frame = _validate_vocab_coverage(val_frame, vocabulary)
    test_frame = _validate_vocab_coverage(test_frame, vocabulary)

    train_frame = _subset_frame(train_frame, max_train_samples)
    val_frame = _subset_frame(val_frame, max_val_samples)
    test_frame = _subset_frame(test_frame, max_test_samples)

    max_seq_len = max(
        max(train_frame["canonical_smiles"].map(len), default=0),
        max(val_frame["canonical_smiles"].map(len), default=0),
        max(test_frame["canonical_smiles"].map(len), default=0),
    ) + 2

    for split_name, frame in (("train", train_frame), ("val", val_frame), ("test", test_frame)):
        _save_split_artifacts(
            frame,
            csv_path=split_paths[split_name],
            token_path=token_paths[split_name],
            vocabulary=vocabulary,
            max_seq_len=max_seq_len,
        )

    metadata = {
        "dataset_name": "zinc250k",
        "raw_dir": str(raw_dir),
        "raw_file": str(raw_path),
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "split_seed": split_seed,
        "max_smiles_length": max_smiles_length,
        "max_seq_len": int(max_seq_len),
        "num_raw_rows": int(len(raw_smiles)),
        "num_valid_unique_rows": int(len(canonical_frame)),
        "num_train": int(len(train_frame)),
        "num_val": int(len(val_frame)),
        "num_test": int(len(test_frame)),
        "vocab_size": vocabulary.size,
        "special_tokens": list(SPECIAL_TOKENS),
        "token_cache_dtype": "int32",
        "cache_settings": requested_cache_settings,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    vocab_path.write_text(json.dumps(vocabulary.token_to_idx, indent=2), encoding="utf-8")

    return PreparedDataBundle(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        split_paths=split_paths,
        token_paths=token_paths,
        vocabulary=vocabulary,
        max_seq_len=max_seq_len,
        metadata=metadata,
    )


def load_split_frame(bundle: PreparedDataBundle, split_name: str) -> pd.DataFrame:
    if split_name not in bundle.split_paths:
        raise KeyError(f"Unknown split {split_name}. Available: {sorted(bundle.split_paths)}")
    return pd.read_csv(bundle.split_paths[split_name])


def build_dataset(bundle: PreparedDataBundle, split_name: str) -> SmilesDataset:
    frame = load_split_frame(bundle, split_name)
    if split_name not in bundle.token_paths:
        raise KeyError(f"Unknown token cache for split {split_name}. Available: {sorted(bundle.token_paths)}")
    token_ids = np.load(bundle.token_paths[split_name], mmap_mode="r")
    return SmilesDataset(frame=frame, token_ids=token_ids)


def collate_smiles_batch(batch: list[dict]) -> dict[str, object]:
    token_ids = torch.stack([item["token_ids"] for item in batch], dim=0)
    canonical_smiles = [item["canonical_smiles"] for item in batch]
    source_smiles = [item["source_smiles"] for item in batch]
    return {
        "token_ids": token_ids,
        "canonical_smiles": canonical_smiles,
        "source_smiles": source_smiles,
    }


def build_dataloader(
    bundle: PreparedDataBundle,
    split_name: str,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = build_dataset(bundle, split_name)
    pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers > 0),
        collate_fn=collate_smiles_batch,
    )
