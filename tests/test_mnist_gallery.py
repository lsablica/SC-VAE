from __future__ import annotations

import base64
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np

from experiments.mnist.interactive.build_site import build

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gallery_uses_the_paper_posterior(tmp_path):
    output = build(tmp_path / "site")
    payload = json.loads(
        (output / "data/posterior.json").read_text(encoding="utf-8")
    )
    with np.load(
        ROOT / "experiments/mnist/final/qualitative_inputs.npz",
        allow_pickle=False,
    ) as inputs:
        expected_locations = inputs["posterior_locations"]
        expected_images = inputs["posterior_images"]
        expected_labels = inputs["posterior_labels"]
        expected_indices = inputs["posterior_indices"]

    points = payload["points"]
    actual_locations = np.asarray(
        [[point["x"], point["y"], point["z"]] for point in points]
    )
    assert len(points) == 500
    np.testing.assert_allclose(actual_locations, expected_locations, rtol=0, atol=0)
    np.testing.assert_array_equal(
        [point["label"] for point in points], expected_labels
    )
    np.testing.assert_array_equal(
        [point["test_index"] for point in points], expected_indices
    )
    assert expected_images.shape == (500, 1, 28, 28)
    assert [point["subset_index"] for point in points] == list(range(500))
    assert all(point["label_name"] == str(point["label"]) for point in points)
    image_bytes = base64.b64decode(points[0]["image_uri"].split(",", 1)[1])
    assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    offset = 8
    compressed = bytearray()
    while offset < len(image_bytes):
        length = int.from_bytes(image_bytes[offset : offset + 4], "big")
        kind = image_bytes[offset + 4 : offset + 8]
        if kind == b"IDAT":
            compressed.extend(image_bytes[offset + 8 : offset + 8 + length])
        offset += length + 12
    scanlines = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(28, 29)
    np.testing.assert_array_equal(scanlines[:, 0], 0)
    np.testing.assert_array_equal(
        scanlines[:, 1:], np.rint(expected_images[0, 0] * 255).astype(np.uint8)
    )
    assert payload["schema_version"] == 2
    assert payload["label_names"] == [str(digit) for digit in range(10)]
    assert payload["model"] == {
        "family": "spcauchy",
        "intrinsic_dimension": 2,
        "ambient_dimension": 3,
        "kl_method": "direct",
        "seed": 0,
        "epoch": 40,
    }


def test_gallery_copies_the_final_posterior_figure(tmp_path):
    output = build(tmp_path / "site")
    paper_figure = ROOT / "experiments/mnist/final/posterior_sphere_trimmed.png"
    site_figure = output / "assets/posterior_sphere_trimmed.png"
    assert _sha256(site_figure) == _sha256(paper_figure)
    assert (output / "index.html").is_file()
    assert (output / "app.js").is_file()
    assert (output / "styles.css").is_file()
    assert (output / ".nojekyll").is_file()
