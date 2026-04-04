from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from scipy.stats import wasserstein_distance

RDLogger.DisableLog("rdApp.*")

PROPERTY_COLUMNS = [
    "molecular_weight",
    "logp",
    "tpsa",
    "qed",
    "ring_count",
    "heavy_atom_count",
]


@dataclass
class MoleculeRecord:
    smiles: str
    canonical_smiles: str | None
    is_valid: bool


def canonicalize_smiles(smiles: str) -> str | None:
    smiles = str(smiles).strip()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def sanitize_smiles_series(smiles_iterable: Iterable[str]) -> list[MoleculeRecord]:
    records: list[MoleculeRecord] = []
    for smiles in smiles_iterable:
        canonical = canonicalize_smiles(smiles)
        records.append(
            MoleculeRecord(
                smiles=str(smiles),
                canonical_smiles=canonical,
                is_valid=canonical is not None,
            )
        )
    return records


def smiles_to_mol(smiles: str):
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return None
    return Chem.MolFromSmiles(canonical, sanitize=True)


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto_similarity(smiles_a: str, smiles_b: str) -> float | None:
    fp_a = morgan_fingerprint(smiles_a)
    fp_b = morgan_fingerprint(smiles_b)
    if fp_a is None or fp_b is None:
        return None
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def compute_properties(smiles: str) -> dict[str, float] | None:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return {
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "qed": float(QED.qed(mol)),
        "ring_count": float(Lipinski.RingCount(mol)),
        "heavy_atom_count": float(mol.GetNumHeavyAtoms()),
    }


def property_frame(smiles_list: Iterable[str]) -> pd.DataFrame:
    rows = []
    for smiles in smiles_list:
        properties = compute_properties(smiles)
        if properties is None:
            continue
        properties["canonical_smiles"] = canonicalize_smiles(smiles)
        rows.append(properties)
    return pd.DataFrame(rows)


def summarize_generation(
    generated_smiles: list[str],
    train_reference: set[str],
    *,
    diversity_pairs: int = 2_000,
    seed: int = 0,
) -> tuple[dict[str, float], pd.DataFrame]:
    records = []
    valid_canonical = []
    novel_count = 0
    for smiles in generated_smiles:
        canonical = canonicalize_smiles(smiles)
        is_valid = canonical is not None
        is_novel = bool(is_valid and canonical not in train_reference)
        if is_valid:
            valid_canonical.append(canonical)
            novel_count += int(is_novel)
        records.append(
            {
                "decoded_smiles": smiles,
                "canonical_smiles": canonical,
                "is_valid": is_valid,
                "is_novel": is_novel,
            }
        )

    valid_count = len(valid_canonical)
    unique_valid_count = len(set(valid_canonical))
    metrics = {
        "num_samples": float(len(generated_smiles)),
        "validity": (valid_count / len(generated_smiles)) if generated_smiles else 0.0,
        "uniqueness": (unique_valid_count / valid_count) if valid_count else 0.0,
        "novelty": (novel_count / valid_count) if valid_count else 0.0,
        "valid_count": float(valid_count),
        "unique_valid_count": float(unique_valid_count),
        "novel_valid_count": float(novel_count),
        "internal_diversity": estimate_internal_diversity(
            valid_canonical,
            n_pairs=diversity_pairs,
            seed=seed,
        ),
    }
    return metrics, pd.DataFrame(records)


def estimate_internal_diversity(
    canonical_smiles: list[str],
    *,
    n_pairs: int = 2_000,
    seed: int = 0,
) -> float:
    if len(canonical_smiles) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    fps = [morgan_fingerprint(smiles) for smiles in canonical_smiles]
    fps = [fp for fp in fps if fp is not None]
    if len(fps) < 2:
        return 0.0
    similarities = []
    for _ in range(min(n_pairs, len(fps) * (len(fps) - 1) // 2)):
        i, j = rng.choice(len(fps), size=2, replace=False)
        similarities.append(float(DataStructs.TanimotoSimilarity(fps[i], fps[j])))
    if not similarities:
        return 0.0
    return float(1.0 - np.mean(similarities))


def property_distance_summary(
    generated_smiles: list[str],
    reference_smiles: list[str],
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    generated_properties = property_frame(generated_smiles)
    reference_properties = property_frame(reference_smiles)
    summary: dict[str, float] = {}
    for property_name in PROPERTY_COLUMNS:
        gen_values = generated_properties[property_name].to_numpy() if not generated_properties.empty else np.array([])
        ref_values = reference_properties[property_name].to_numpy() if not reference_properties.empty else np.array([])
        if len(gen_values) == 0 or len(ref_values) == 0:
            summary[f"{property_name}_wasserstein"] = math.nan
            summary[f"{property_name}_generated_mean"] = math.nan
            summary[f"{property_name}_reference_mean"] = math.nan
            continue
        summary[f"{property_name}_wasserstein"] = float(wasserstein_distance(gen_values, ref_values))
        summary[f"{property_name}_generated_mean"] = float(np.mean(gen_values))
        summary[f"{property_name}_reference_mean"] = float(np.mean(ref_values))
    return summary, generated_properties, reference_properties


def export_property_histograms(
    generated_properties: pd.DataFrame,
    reference_properties: pd.DataFrame,
    output_path: str | Path,
    bins: int = 30,
) -> pd.DataFrame:
    columns = [
        "property",
        "bin_left",
        "bin_right",
        "generated_density",
        "reference_density",
    ]
    rows = []
    for property_name in PROPERTY_COLUMNS:
        if property_name not in generated_properties or property_name not in reference_properties:
            continue
        combined = pd.concat(
            [
                generated_properties[property_name].dropna(),
                reference_properties[property_name].dropna(),
            ],
            ignore_index=True,
        )
        if combined.empty:
            continue
        hist_bins = np.histogram_bin_edges(combined.to_numpy(), bins=bins)
        gen_hist, _ = np.histogram(generated_properties[property_name].dropna().to_numpy(), bins=hist_bins, density=True)
        ref_hist, _ = np.histogram(reference_properties[property_name].dropna().to_numpy(), bins=hist_bins, density=True)
        for idx in range(len(hist_bins) - 1):
            rows.append(
                {
                    "property": property_name,
                    "bin_left": hist_bins[idx],
                    "bin_right": hist_bins[idx + 1],
                    "generated_density": gen_hist[idx],
                    "reference_density": ref_hist[idx],
                }
            )
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(output_path, index=False)
    return frame
