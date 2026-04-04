from __future__ import annotations

import json
import subprocess
import sys
import unittest
import shutil
from pathlib import Path


def write_csv(path: Path, smiles: list[str]) -> None:
    path.write_text("smiles\n" + "\n".join(smiles) + "\n", encoding="utf-8")


def run_command(args: list[str], cwd: Path) -> None:
    completed = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(args)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")


class SmilesPipelineSmokeTest(unittest.TestCase):
    def test_smiles_pipeline_smoke(self):
        repo_root = Path(__file__).resolve().parents[3]
        workspace_tmp = repo_root / "experiments" / "smiles" / "_test_artifacts"
        tmp_path = workspace_tmp / "smoke_case"
        shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.mkdir(parents=True, exist_ok=True)
        try:
            raw_dir = tmp_path / "raw"
            processed_dir = tmp_path / "processed"
            output_root = tmp_path / "run_spc"
            aggregate_dir = tmp_path / "aggregated"
            raw_dir.mkdir()

            write_csv(
                raw_dir / "250k_rndm_zinc_drugs_clean_3.csv",
                [
                    "CCO",
                    "CCC",
                    "CCN",
                    "CCCl",
                    "C1CC1",
                    "CC(=O)O",
                    "c1ccccc1",
                    "CCCO",
                    "CC(C)O",
                    "CC(C)N",
                    "CCBr",
                    "COC",
                    "CCF",
                    "CCS",
                    "c1ccccc1O",
                    "CC=O",
                ],
            )

            run_command(
                [
                    sys.executable,
                    "-m",
                    "experiments.smiles.train",
                    "--data-root",
                    str(raw_dir),
                    "--processed-root",
                    str(processed_dir),
                    "--output-root",
                    str(output_root),
                    "--model-name",
                    "spcauchy-128",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--device",
                    "cpu",
                    "--embedding-dim",
                    "32",
                    "--hidden-dim",
                    "32",
                    "--num-layers",
                    "1",
                    "--num-heads",
                    "4",
                    "--dropout",
                    "0.1",
                    "--max-train-samples",
                    "10",
                    "--max-val-samples",
                    "2",
                    "--max-test-samples",
                    "6",
                    "--force-reprocess",
                ],
                repo_root,
            )

            checkpoint = output_root / "checkpoints" / "best-val-elbo.pt"
            run_command(
                [
                    sys.executable,
                    "-m",
                    "experiments.smiles.evaluate",
                    "--checkpoint",
                    str(checkpoint),
                    "--data-root",
                    str(raw_dir),
                    "--processed-root",
                    str(processed_dir),
                    "--split",
                    "test",
                    "--batch-size",
                    "4",
                    "--device",
                    "cpu",
                    "--num-prior-samples",
                    "16",
                ],
                repo_root,
            )
            run_command(
                [
                    sys.executable,
                    "-m",
                    "experiments.smiles.interpolate",
                    "--checkpoint",
                    str(checkpoint),
                    "--data-root",
                    str(raw_dir),
                    "--processed-root",
                    str(processed_dir),
                    "--split",
                    "test",
                    "--batch-size",
                    "4",
                    "--device",
                    "cpu",
                    "--pool-size",
                    "6",
                    "--pairs-per-bin",
                    "1",
                    "--steps",
                    "5",
                    "--seed",
                    "0",
                ],
                repo_root,
            )
            run_command(
                [
                    sys.executable,
                    "-m",
                    "experiments.smiles.aggregate",
                    "--runs-root",
                    str(tmp_path),
                    "--dataset-name",
                    "zinc250k",
                    "--output-dir",
                    str(aggregate_dir),
                    "--eval-split",
                    "test",
                ],
                repo_root,
            )

            self.assertTrue(checkpoint.exists())
            self.assertTrue((output_root / "metrics" / "eval_test.json").exists())
            self.assertTrue((output_root / "interpolation" / "interpolation_summary.csv").exists())
            self.assertTrue((aggregate_dir / "benchmark_mean_std.csv").exists())

            manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["fairness_regime"], "spherical_reference")
            self.assertEqual(manifest["dataset_name"], "zinc250k")
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
