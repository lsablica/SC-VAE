from __future__ import annotations

from pathlib import Path

if __package__ in (None, ""):
    from common import DEFAULT_PAYLOAD_FILENAME, DEFAULT_RUN_DIR, SITE_DATA_DIR, ensure_site_data_dir
    from export_mnist_qualitative_s2 import build_payload
else:
    from .common import DEFAULT_PAYLOAD_FILENAME, DEFAULT_RUN_DIR, SITE_DATA_DIR, ensure_site_data_dir
    from .export_mnist_qualitative_s2 import build_payload

from experiments.mnist_experiment.config import DEFAULT_DATA_DIR


def build() -> None:
    output_dir = ensure_site_data_dir()
    output_path = output_dir / DEFAULT_PAYLOAD_FILENAME
    checkpoint_path = DEFAULT_RUN_DIR / "best_recon_checkpoint.pt"
    selection_path = DEFAULT_RUN_DIR / "selection_summary.json"

    if checkpoint_path.exists() and selection_path.exists():
        build_payload(
            run_dir=DEFAULT_RUN_DIR,
            data_dir=DEFAULT_DATA_DIR,
            output_path=output_path,
            num_points=1000,
            seed=1,
            device="cpu",
        )
        print(f"Built interactive payload from {DEFAULT_RUN_DIR}")
        return

    if output_path.exists():
        print(
            "Source qualitative run is not available in this checkout; "
            f"reusing committed payload at {output_path}"
        )
        return

    raise FileNotFoundError(
        "Could not build the MNIST gallery payload because the qualitative run "
        "artifacts are missing and no prebuilt JSON payload is committed."
    )


if __name__ == "__main__":
    build()
