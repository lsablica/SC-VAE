"""Finalize provenance and checksums for the locked smallNORB study."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from benchmark.vendor_power_spherical import PowerSpherical

from .config import (
    EXPERIMENT_ROOT,
    FINAL_ROOT,
    MAIN_FAMILIES,
    REPO_ROOT,
    RUNS_ROOT,
    SEARCH_ROOT,
    SEEDS,
)
from .utils import (
    capture_environment,
    ensure_dir,
    read_json,
    repo_relative,
    sha256_file,
    write_json,
)


POWER_SPHERICAL_COMMIT = "3d4619a9d6c01bc9b427533d386271a233e304cd"
SMALLNORB_VMF_PATH = (
    EXPERIMENT_ROOT / "vendor_vmf_smallnorb.py"
)
SHARED_VMF_PATH = REPO_ROOT / "benchmark" / "vendor_vmf_robust.py"
POWER_SPHERICAL_PATH = (
    REPO_ROOT / "benchmark" / "vendor_power_spherical.py"
)


def _command(arguments: list[str]) -> str:
    result = subprocess.run(
        arguments,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or result.stderr.strip()


def _source_digest() -> tuple[str, int]:
    paths_raw = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    digest = hashlib.sha256()
    count = 0
    excluded = (
        Path("data"),
        Path("experiments/smallnorb/runs"),
        Path("experiments/smallnorb/search"),
        Path("experiments/smallnorb/final"),
        Path("experiments/contaminated_directional/final"),
    )
    for encoded in sorted(value for value in paths_raw.split(b"\0") if value):
        relative = Path(encoded.decode("utf-8"))
        if any(
            relative == prefix or prefix in relative.parents
            for prefix in excluded
        ):
            continue
        path = REPO_ROOT / relative
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
        count += 1
    return digest.hexdigest(), count


def _completed_run_records() -> list[dict[str, Any]]:
    records = []
    for seed_manifest in sorted(
        RUNS_ROOT.glob("final/*/*/seed_*/seed_manifest.json")
    ):
        run_dir = seed_manifest.parent
        config = read_json(run_dir / "config.json")
        evaluation = read_json(run_dir / "evaluation_summary.json")
        records.append(
            {
                "run_dir": repo_relative(run_dir),
                "family": config["family"],
                "seed": config["seed"],
                "config_sha256": sha256_file(run_dir / "config.json"),
                "seed_manifest_sha256": sha256_file(seed_manifest),
                "history_sha256": sha256_file(run_dir / "history.csv"),
                "evaluation_sha256": sha256_file(
                    run_dir / "evaluation_summary.json"
                ),
                "probe_sha256": sha256_file(
                    run_dir / "probe_summary.json"
                ),
                "interpolation_sha256": sha256_file(
                    run_dir / "interpolation_summary_test.json"
                ),
                "test_was_accessed": evaluation["test_was_accessed"],
                "checkpoints_git_ignored": True,
            }
        )
    return records


def _all_commands(run_records: list[dict[str, Any]]) -> list[str]:
    commands = []
    seen = set()
    for record in run_records:
        command_path = REPO_ROOT / record["run_dir"] / "commands.txt"
        for command in command_path.read_text(encoding="utf-8").splitlines():
            if command and command not in seen:
                commands.append(command)
                seen.add(command)
    commands.extend(
        [
            "python -m experiments.smallnorb.aggregate",
            "python -m experiments.smallnorb.figures --device cuda",
            "python -m experiments.smallnorb.record_manifest --finalize",
        ]
    )
    return commands


def _protocol_audit() -> dict[str, Any]:
    run_dirs = sorted(
        path.parent
        for path in RUNS_ROOT.glob(
            "final/*/*/seed_*/evaluation_summary.json"
        )
    )
    configs = [read_json(path / "config.json") for path in run_dirs]
    manifests = [
        read_json(path / "seed_manifest.json") for path in run_dirs
    ]
    evaluations = [
        read_json(path / "evaluation_summary.json") for path in run_dirs
    ]
    expected_pairs = {
        (family, seed) for family in MAIN_FAMILIES for seed in SEEDS
    }
    observed_pairs = {
        (config["family"], int(config["seed"])) for config in configs
    }
    ignored_shared_keys = {
        "family",
        "seed",
        "notes",
        "tags",
    }
    normalized_configs = {
        json.dumps(
            {
                key: value
                for key, value in config.items()
                if key not in ignored_shared_keys
            },
            sort_keys=True,
        )
        for config in configs
    }
    decoder_counts = {
        manifest["parameter_counts"]["shared_decoder"]
        for manifest in manifests
    }
    encoder_counts = {
        manifest["parameter_counts"]["shared_encoder"]
        for manifest in manifests
    }
    search_test_flags = []
    for path in sorted(
        SEARCH_ROOT.glob("*/spcauchy/seed_0/evaluation_summary.json")
    ):
        search_test_flags.append(
            bool(read_json(path).get("test_was_accessed", False))
        )
    checks = {
        "all_20_main_runs_present": observed_pairs == expected_pairs,
        "shared_config_identical_except_family_seed_and_labels": (
            len(normalized_configs) == 1
        ),
        "shared_encoder_parameter_count_identical": (
            len(encoder_counts) == 1
        ),
        "shared_decoder_parameter_count_identical": (
            len(decoder_counts) == 1
        ),
        "all_final_evaluations_accessed_test": all(
            evaluation["test_was_accessed"]
            for evaluation in evaluations
        ),
        "no_search_evaluation_accessed_test": not any(
            search_test_flags
        ),
        "all_configs_use_baseline_cnn": all(
            config["architecture"] == "baseline_cnn"
            for config in configs
        ),
        "all_configs_use_direct_spcauchy_route": all(
            config["spcauchy_kl_method"] == "direct"
            for config in configs
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_family_seed_pairs": sorted(expected_pairs),
        "observed_family_seed_pairs": sorted(observed_pairs),
        "shared_encoder_parameter_counts": sorted(encoder_counts),
        "shared_decoder_parameter_counts": sorted(decoder_counts),
    }


def _artifact_rows(root: Path) -> list[dict[str, Any]]:
    excluded = {
        root / "manifest.json",
        root / "artifact_checksums.sha256",
    }
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path in excluded or path.suffix in {".pt", ".pth", ".ckpt"}:
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def record_manifest(finalize: bool = True) -> Path:
    ensure_dir(FINAL_ROOT)
    run_records = _completed_run_records()
    commands = _all_commands(run_records)
    (FINAL_ROOT / "commands.txt").write_text(
        "\n".join(commands) + "\n", encoding="utf-8"
    )
    environment = capture_environment()
    (FINAL_ROOT / "environment.txt").write_text(
        "\n".join(
            f"{key}={json.dumps(value, sort_keys=True)}"
            for key, value in environment.items()
        )
        + "\n",
        encoding="utf-8",
    )
    source_sha256, source_count = _source_digest()
    audit = _protocol_audit()
    if not audit["passed"]:
        raise RuntimeError(f"Protocol audit failed: {audit['checks']}")
    write_json(FINAL_ROOT / "protocol_audit.json", audit)
    data_manifest = (
        REPO_ROOT
        / "data"
        / "smallnorb"
        / "processed"
        / "cache_manifest.json"
    )
    artifacts = _artifact_rows(FINAL_ROOT) if finalize else []
    manifest = {
        "schema_version": 1,
        "experiment": "smallnorb_viewpoint_generalization",
        "source": {
            "commit": _command(["git", "rev-parse", "HEAD"]),
            "branch": _command(["git", "branch", "--show-current"]),
            "status": _command(["git", "status", "--short"]).splitlines(),
            "source_tree_sha256": source_sha256,
            "source_file_count": source_count,
        },
        "environment": environment,
        "data": {
            "cache_manifest": repo_relative(data_manifest),
            "cache_manifest_sha256": sha256_file(data_manifest),
            "cache": read_json(data_manifest),
            "official_test_access_policy": (
                "The shared setup and initial validation gate were frozen "
                "before SC test evaluation. A subsequent mathematical audit "
                "found a vMF normalizer truncation error. The local baseline "
                "was corrected and its validation-only gate rerun before any "
                "vMF test evaluation. No test metric informed the repair."
            ),
        },
        "selection": {
            "search_report": repo_relative(
                SEARCH_ROOT / "SEARCH_REPORT.md"
            ),
            "frozen_setup": read_json(
                SEARCH_ROOT / "frozen_setup.json"
            ),
            "smoke_gate": read_json(
                RUNS_ROOT / "smoke" / "smoke_gate.json"
            ),
        },
        "protocol_audit": audit,
        "dependencies": {
            "power_spherical": {
                "upstream_commit": POWER_SPHERICAL_COMMIT,
                "vendored_path": "benchmark/vendor_power_spherical.py",
                "license_path": "benchmark/POWER_SPHERICAL_LICENSE",
                "class": PowerSpherical.__name__,
                "source_sha256": sha256_file(POWER_SPHERICAL_PATH),
                "modified_for_smallnorb": False,
            },
            "robust_vmf": {
                "smallnorb_local_path": repo_relative(
                    SMALLNORB_VMF_PATH
                ),
                "smallnorb_local_sha256": sha256_file(
                    SMALLNORB_VMF_PATH
                ),
                "shared_benchmark_path": repo_relative(
                    SHARED_VMF_PATH
                ),
                "shared_benchmark_sha256": sha256_file(
                    SHARED_VMF_PATH
                ),
                "isolation": (
                    "The rationalized rejection sampler and full-clamp "
                    "normalizer series are local to smallNORB. MNIST and "
                    "shared benchmark code is unchanged."
                ),
            }
        },
        "runs": run_records,
        "commands": commands,
        "artifacts": artifacts,
    }
    path = FINAL_ROOT / "manifest.json"
    write_json(path, manifest)
    if finalize:
        (FINAL_ROOT / "artifact_checksums.sha256").write_text(
            "\n".join(
                f"{row['sha256']}  {row['path']}" for row in artifacts
            )
            + "\n",
            encoding="utf-8",
        )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    print(record_manifest(finalize=args.finalize))


if __name__ == "__main__":
    main()
