from __future__ import annotations

import unittest
import shutil
from pathlib import Path

from experiments.smiles.chemistry import canonicalize_smiles, compute_properties, summarize_generation
from experiments.smiles.data import build_dataset, prepare_zinc250k_dataset
from experiments.smiles.decoding import token_ids_to_smiles


def write_csv(path: Path, smiles: list[str]) -> None:
    path.write_text("SMILES\n" + "\n".join(smiles) + "\n", encoding="utf-8")


class SmilesUtilsTest(unittest.TestCase):
    def test_canonicalization_and_properties(self):
        self.assertEqual(canonicalize_smiles("C(C)O"), "CCO")
        properties = compute_properties("CCO")
        self.assertIsNotNone(properties)
        self.assertGreater(properties["molecular_weight"], 0)
        self.assertEqual(properties["heavy_atom_count"], 3)

    def test_generation_summary_metrics(self):
        metrics, rows = summarize_generation(
            ["CCO", "invalid", "CCO", "CCC"],
            train_reference={"CCO"},
            seed=0,
        )
        self.assertEqual(metrics["validity"], 0.75)
        self.assertAlmostEqual(metrics["uniqueness"], 2 / 3)
        self.assertAlmostEqual(metrics["novelty"], 1 / 3)
        self.assertEqual(len(rows), 4)

    def test_dataset_tokenization_roundtrip(self):
        workspace_tmp = Path(__file__).resolve().parents[1] / "_test_artifacts"
        tmp_path = workspace_tmp / "utils_case"
        shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.mkdir(parents=True, exist_ok=True)
        try:
            raw_dir = tmp_path / "raw"
            processed_dir = tmp_path / "processed"
            raw_dir.mkdir()
            write_csv(
                raw_dir / "250k_rndm_zinc_drugs_clean_3.csv",
                ["CCO", "CCC", "CCN", "C1CC1", "CCCl", "CCBr", "CCF", "CCS", "CCCO", "COC"],
            )

            bundle = prepare_zinc250k_dataset(raw_dir=raw_dir, processed_dir=processed_dir, force_reprocess=True)
            dataset = build_dataset(bundle, "train")
            item = dataset[0]
            decoded = token_ids_to_smiles(item["token_ids"].tolist(), bundle.vocabulary.idx_to_token)
            self.assertEqual(decoded, item["canonical_smiles"])
            self.assertEqual(bundle.metadata["dataset_name"], "zinc250k")
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
