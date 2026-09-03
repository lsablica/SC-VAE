from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

from experiments.mnist.generate_qualitative_figures import OUTPUT_NAMES, generate_all

REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = REPO_ROOT / "experiments" / "mnist" / "final"


def test_qualitative_inputs_follow_fixed_selection_protocol() -> None:
    provenance = json.loads(
        (FINAL_ROOT / "qualitative_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["selected_seed"] == 0
    assert provenance["selected_epoch"] == 40
    assert provenance["checkpoint"]["sha256"] == (
        "69ce5d6e87b95b4945e3c74cc41e4487b25c705725b6146bccc864ea28b4f91a"
    )
    assert provenance["run_config"]["sha256"] == (
        "4bc677977f34589e9ec2517ba59336a6163edb3c3c8c7ff15076e42f8d757b86"
    )
    assert provenance["interpolation_endpoint_indices"] == [2, 0]
    assert provenance["interpolation_endpoint_labels"] == [1, 7]

    with np.load(FINAL_ROOT / "qualitative_inputs.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["posterior_indices"], np.arange(500))
        assert data["posterior_images"].shape == (500, 1, 28, 28)
        assert data["posterior_images"].dtype == np.float32
        np.testing.assert_array_equal(data["reconstruction_indices"], np.arange(10))
        assert data["uniform_points"].shape == (50, 3)
        assert data["interpolation_locations"].shape == (10, 3)
        assert data["history_values"].shape[0] == 40
        np.testing.assert_allclose(
            np.linalg.norm(data["posterior_locations"], axis=-1),
            1.0,
            rtol=2e-6,
            atol=2e-6,
        )


def test_compact_inputs_rerender_every_figure(tmp_path: Path) -> None:
    shutil.copy2(FINAL_ROOT / "qualitative_inputs.npz", tmp_path)
    shutil.copy2(FINAL_ROOT / "qualitative_provenance.json", tmp_path)
    provenance = generate_all(
        tmp_path,
        REPO_ROOT / "data",
        "cpu",
        refresh_from_checkpoint=False,
    )
    assert set(provenance["figure_sha256"]) == set(OUTPUT_NAMES)
    for name in OUTPUT_NAMES:
        actual = Image.open(tmp_path / name).convert("RGBA")
        expected = Image.open(FINAL_ROOT / name).convert("RGBA")
        assert actual.size == expected.size
        assert ImageChops.difference(actual, expected).getbbox() is None
