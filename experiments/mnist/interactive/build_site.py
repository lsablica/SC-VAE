"""Build the static MNIST posterior viewer used by GitHub Pages."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import shutil
import struct
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = Path(__file__).resolve().parent / "site"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "_site"
INPUTS_PATH = REPO_ROOT / "experiments/mnist/final/qualitative_inputs.npz"
PROVENANCE_PATH = REPO_ROOT / "experiments/mnist/final/qualitative_provenance.json"
EVALUATION_PATH = (
    REPO_ROOT / "experiments/mnist/final/selected_run/evaluation_summary.json"
)
SOURCE_FIGURE = REPO_ROOT / "experiments/mnist/final/posterior_sphere_trimmed.png"
PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _camera(elevation: float, azimuth: float) -> dict[str, dict[str, float]]:
    elevation_radians = math.radians(elevation)
    azimuth_radians = math.radians(azimuth)
    distance = 1.65
    return {
        "eye": {
            "x": distance * math.cos(elevation_radians) * math.cos(azimuth_radians),
            "y": distance * math.cos(elevation_radians) * math.sin(azimuth_radians),
            "z": distance * math.sin(elevation_radians),
        }
    }


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def _image_uri(image: np.ndarray) -> str:
    """Encode a single 28x28 MNIST image without adding a site dependency."""
    pixels = np.asarray(image)
    if pixels.shape == (1, 28, 28):
        pixels = pixels[0]
    if pixels.shape != (28, 28):
        raise ValueError(f"Expected a 28x28 MNIST image, got {pixels.shape}")
    pixels = np.clip(np.rint(pixels * 255.0), 0, 255).astype(np.uint8)
    scanlines = b"".join(b"\x00" + row.tobytes() for row in pixels)
    png = b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 28, 28, 8, 0, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(scanlines, level=9)),
            _png_chunk(b"IEND", b""),
        )
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _payload() -> dict[str, object]:
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    if _sha256(INPUTS_PATH) != provenance["inputs"]["sha256"]:
        raise ValueError("The posterior input array does not match its provenance record")
    with np.load(INPUTS_PATH, allow_pickle=False) as arrays:
        indices = arrays["posterior_indices"].astype(np.int64)
        images = arrays["posterior_images"].astype(np.float32)
        labels = arrays["posterior_labels"].astype(np.int64)
        locations = arrays["posterior_locations"].astype(np.float64)

    if locations.shape != (500, 3):
        raise ValueError(f"Expected 500 three-dimensional locations, got {locations.shape}")
    if indices.shape != (500,) or labels.shape != (500,):
        raise ValueError("Posterior indices and labels must each contain 500 values")
    if images.shape != (500, 1, 28, 28):
        raise ValueError(f"Expected 500 MNIST images, got {images.shape}")
    if not np.isfinite(images).all() or images.min() < 0.0 or images.max() > 1.0:
        raise ValueError("Posterior images must contain finite values in [0, 1]")
    if not np.allclose(np.linalg.norm(locations, axis=1), 1.0, atol=2e-5):
        raise ValueError("Posterior locations do not lie on the unit sphere")

    recorded_figure_hash = provenance["figure_sha256"][SOURCE_FIGURE.name]
    if _sha256(SOURCE_FIGURE) != recorded_figure_hash:
        raise ValueError("The posterior figure does not match its provenance record")

    selected = evaluation["best_recon_checkpoint"]
    config = evaluation["run_config"]
    if (
        config["seed"] != provenance["selected_seed"]
        or selected["epoch"] != provenance["selected_epoch"]
        or config["spcauchy_kl_method"] != "direct"
    ):
        raise ValueError("The selected evaluation and qualitative provenance disagree")
    points = [
        {
            "subset_index": subset_index,
            "test_index": int(index),
            "label": int(label),
            "label_name": str(int(label)),
            "image_uri": _image_uri(image),
            "x": float(location[0]),
            "y": float(location[1]),
            "z": float(location[2]),
        }
        for subset_index, (index, image, label, location) in enumerate(
            zip(indices, images, labels, locations, strict=True)
        )
    ]
    camera = provenance["plotting"]["posterior_camera"]
    return {
        "schema_version": 2,
        "title": "MNIST posterior on the sphere",
        "description": (
            "Deterministic posterior locations for the 500 test images used in "
            "the paper's spherical Cauchy visualization."
        ),
        "model": {
            "family": config["model_family"],
            "intrinsic_dimension": config["reported_dim"],
            "ambient_dimension": config["ambient_latent_dim"],
            "kl_method": config["spcauchy_kl_method"],
            "seed": config["seed"],
            "epoch": selected["epoch"],
        },
        "metrics": {
            "reconstruction_loss": selected["eval_recon_loss"],
            "kl": selected["eval_kl"],
            "total_loss": selected["eval_total_loss"],
        },
        "label_names": [str(digit) for digit in range(10)],
        "num_points": len(points),
        "selection": {
            "rule": provenance["selection_rule"],
            "checkpoint_sha256": provenance["checkpoint"]["sha256"],
        },
        "paper_figure": {
            "path": f"assets/{SOURCE_FIGURE.name}",
            "sha256": recorded_figure_hash,
        },
        "input_sha256": provenance["inputs"]["sha256"],
        "default_camera": _camera(camera["elevation"], camera["azimuth"]),
        "palette": list(PALETTE),
        "points": points,
    }


def build(output: Path = DEFAULT_OUTPUT) -> Path:
    """Write a self-contained static site, except for the pinned Plotly CDN."""
    output = output.resolve()
    payload = _payload()
    data_dir = output / "data"
    assets_dir = output / "assets"
    data_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("index.html", "app.js", "styles.css"):
        shutil.copy2(STATIC_DIR / filename, output / filename)
    shutil.copy2(SOURCE_FIGURE, assets_dir / SOURCE_FIGURE.name)
    (data_dir / "posterior.json").write_text(
        json.dumps(payload, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (output / ".nojekyll").touch()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = build(args.output)
    print(f"Built MNIST posterior viewer at {output}")


if __name__ == "__main__":
    main()
