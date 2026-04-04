from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen


RAW_FILENAME = "250k_rndm_zinc_drugs_clean_3.csv"
SOURCE_URLS = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv",
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/main/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the public ZINC-250k CSV used by ChemicalVAE-style benchmarks.")
    parser.add_argument(
        "--output-dir",
        default="experiments/smiles/datasets/zinc250k/raw",
        help="Directory where the ZINC-250k CSV will be written.",
    )
    parser.add_argument(
        "--filename",
        default=RAW_FILENAME,
        help="Local filename for the downloaded raw CSV.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing local raw CSV.",
    )
    return parser.parse_args()


def download_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=120) as response:
        return response.read()


def download_dataset(output_dir: str | Path, *, filename: str = RAW_FILENAME, force: bool = False) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    if output_path.exists() and not force:
        print(f"ZINC-250k raw file already present at {output_path}")
        return output_path

    last_error = None
    for url in SOURCE_URLS:
        try:
            payload = download_bytes(url)
        except Exception as exc:  # pragma: no cover - network-path fallback
            last_error = exc
            print(f"Failed {url}: {exc}")
            continue
        if not payload.startswith(b"smiles") and b"smiles" not in payload[:512].lower():
            print(f"Skipping unexpected response from {url}")
            continue
        output_path.write_bytes(payload)
        print(f"Downloaded ZINC-250k from {url}")
        print(f"Saved -> {output_path}")
        return output_path

    raise RuntimeError(
        "Could not download the ZINC-250k CSV automatically. "
        f"Last error: {last_error}. Please place {RAW_FILENAME} in {output_dir} manually."
    )


def main() -> None:
    args = parse_args()
    download_dataset(args.output_dir, filename=args.filename, force=args.force)


if __name__ == "__main__":
    main()
