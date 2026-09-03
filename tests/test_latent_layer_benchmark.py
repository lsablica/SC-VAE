import math
import shutil
from pathlib import Path

import pytest
import torch
from matplotlib.figure import Figure

from experiments.latent_layer.accuracy import (
    AccuracyConfig,
    compute_reference_value,
    save_accuracy_outputs,
)
from experiments.latent_layer.methods import (
    get_latent_step_method,
    get_spcauchy_kl_method,
)
from experiments.latent_layer.robustness import (
    RobustnessConfig,
    run_robustness_benchmark,
)
from experiments.latent_layer.runtime import (
    RuntimeConfig,
    run_runtime_benchmark,
    summarize_runtime_records,
)


def test_method_registries_contain_only_supported_routes():
    assert {
        name for name in ("direct", "direct_fixed", "neighbor", "laplace")
        if get_spcauchy_kl_method(name).name == name
    } == {"direct", "direct_fixed", "neighbor", "laplace"}
    assert get_latent_step_method("power_spherical").family == "powerspherical"
    assert get_latent_step_method("vmf_official").family == "vmf"
    with pytest.raises(KeyError):
        get_spcauchy_kl_method("removed-route")


def test_reference_uses_certified_direct_evaluator():
    reference = compute_reference_value(0.5, 8, AccuracyConfig())
    direct = get_spcauchy_kl_method("direct").evaluator(
        torch.tensor([[0.5]], dtype=torch.float64),
        8,
        direct_backend="vectorized",
    )
    assert reference.success
    assert reference.source == "direct_certified"
    assert math.isclose(reference.value, float(direct), rel_tol=1e-13)


def test_runtime_benchmark_smoke_runs_on_cpu():
    config = RuntimeConfig(
        dims=[8],
        spcauchy_methods=["spcauchy_direct"],
        vmf_methods=[],
        power_methods=[],
        device_name="cpu",
        dtype=torch.float32,
        seed=1,
        batch_size=2,
        warmup_iters=0,
        measure_iters=1,
        repeats=1,
        timeout_s=5.0,
        concentration_mode="direct-rho",
        rho_values=[0.5],
    )
    records = run_runtime_benchmark(config)
    assert len(records) == 1
    assert records[0]["method"] == "spcauchy_direct"
    assert records[0]["benchmark"] == "runtime"
    assert records[0]["success"]
    summary = summarize_runtime_records(records)
    assert summary[0]["evaluations"] == 1
    assert summary[0]["successes"] == 1


def test_robustness_benchmark_smoke_records_result():
    config = RobustnessConfig(
        spcauchy_dims=[8],
        vmf_dims=[],
        rho_grid=[0.5],
        spcauchy_methods=["direct"],
        vmf_methods=[],
        power_methods=[],
        device_name="cpu",
        dtype=torch.float32,
        seed=1,
        batch_size=2,
        timeout_s=5.0,
    )
    records = run_robustness_benchmark(config)
    assert len(records) == 1
    assert records[0]["benchmark"] == "robustness"
    assert records[0]["method"] == "direct"
    assert records[0]["success"]


def test_accuracy_plotting_smoke_writes_outputs():
    output = Path.cwd() / "latent_benchmark_test_output"
    shutil.rmtree(output, ignore_errors=True)
    original_savefig = Figure.savefig

    def fake_savefig(self, filename, *args, **kwargs):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).touch()

    Figure.savefig = fake_savefig
    try:
        records = [
            {
                "method": method,
                "success": True,
                "dim": dimension,
                "rho": rho,
                "abs_error": 1e-10 if method == "direct" else 1e-4,
                "eval_time_s": 1e-4 * dimension,
            }
            for method in ("direct", "laplace")
            for dimension in (8, 16)
            for rho in (0.5, 0.9)
        ]
        _, csv_path, figures = save_accuracy_outputs(records, out_dir=str(output))
        assert csv_path.endswith("spcauchy_kl_accuracy.csv")
        assert figures and all(Path(path).exists() for path in figures)
    finally:
        Figure.savefig = original_savefig
        shutil.rmtree(output, ignore_errors=True)
