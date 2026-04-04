from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = ROOT / "experiments" / "mnist_experiment"
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
SITE_DIR = EXPERIMENT_ROOT / "interactive_gallery" / "site"
SITE_DATA_DIR = SITE_DIR / "data"
DEFAULT_RUN_DIR = OUTPUT_ROOT / "qualitative" / "spcauchy_s2" / "seed_1"
DEFAULT_PAYLOAD_FILENAME = "mnist_spcauchy_s2.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_experiment.config import DEFAULT_DATA_DIR  # noqa: E402


DEFAULT_CAMERA = {
    "eye": {"x": 1.06, "y": 0.96, "z": 0.64},
    "up": {"x": 0.0, "y": 0.0, "z": 1.0},
    "center": {"x": 0.0, "y": 0.0, "z": 0.0},
    "projection": {"type": "perspective"},
}

PALETTE = [
    "#ff6b6b",
    "#ffd166",
    "#06d6a0",
    "#118ab2",
    "#7b61ff",
    "#ff85b3",
    "#5bc0ff",
    "#8bd450",
    "#f97316",
    "#e879f9",
]


def ensure_site_data_dir() -> Path:
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return SITE_DATA_DIR


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mnist_image_data_uri(image: np.ndarray) -> str:
    image_uint8 = np.clip(np.asarray(image) * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image_uint8, mode="L").resize((112, 112), resample=Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
