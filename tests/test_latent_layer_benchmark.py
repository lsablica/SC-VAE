import math
import shutil
from pathlib import Path

import torch
from matplotlib.figure import Figure

from experiments.latent_layer_benchmark import accuracy as accuracy_module
from experiments.latent_layer_benchmark.accuracy import AccuracyConfig, compute_reference_value, save_accuracy_outputs
from experiments.latent_layer_benchmark.methods import get_latent_step_method, get_spcauchy_kl_method
from experiments.latent_layer_benchmark.robustness import RobustnessConfig, run_robustness_benchmark
from experiments.latent_layer_benchmark.runtime import RuntimeConfig, run_runtime_benchmark


def test_method_registries_resolve_expected_entries():
    assert get_spcauchy_kl_method("series").name == "series"
    assert get_spcauchy_kl_method("quadrature").name == "quadrature"
    assert get_spcauchy_kl_method("asymptotic_high_rho").name == "asymptotic_high_rho"
    assert get_spcauchy_kl_method("hybrid").name == "hybrid"
    assert get_spcauchy_kl_method("auto").name == "auto"

    assert get_latent_step_method("spcauchy_hybrid").family == "spcauchy"
    assert get_latent_step_method("vmf_official").family == "vmf"


def test_low_dim_reference_uses_exact_hybrid_branch():
    config = AccuracyConfig(dims=[2], rho_grid=[0.5], methods=["hybrid"])
    reference = compute_reference_value(0.5, 2, config)
    hybrid = get_spcauchy_kl_method("hybrid").evaluator(torch.tensor([[0.5]], dtype=torch.float64), 2)

    assert reference.success
    assert reference.source == "exact_lowd_hybrid"
    assert math.isclose(reference.value, float(hybrid.item()), rel_tol=1e-10, abs_tol=1e-10)


def test_reference_fallback_uses_series_and_optional_mpmath(monkeypatch):
    config = AccuracyConfig(reference_nodes=64, reference_disagreement_tol=1e-8)

    calls = []

    def fake_eval(_evaluator, _rho_value, _latent_dim, _device, _dtype, **_kwargs):
        calls.append("called")
        if len(calls) == 1:
            return None, 0.0, False, "runtime_error", "quadrature failed"
        return 1.25, 0.0, True, None, None

    monkeypatch.setattr(accuracy_module, "_evaluate_kl_method", fake_eval)
    reference = compute_reference_value(0.8, 8, config)
    assert reference.success
    assert reference.source == "long_series"
    assert reference.value == 1.25

    responses = iter(
        [
            (1.0, 0.0, True, None, None),
            (2.0, 0.0, True, None, None),
        ]
    )

    def disagreeing_eval(_evaluator, _rho_value, _latent_dim, _device, _dtype, **_kwargs):
        return next(responses)

    monkeypatch.setattr(accuracy_module, "_evaluate_kl_method", disagreeing_eval)
    monkeypatch.setattr(
        accuracy_module,
        "_mpmath_reference",
        lambda _rho_value, _latent_dim: accuracy_module.ReferenceResult(
            value=3.0,
            source="mpmath",
            success=True,
        ),
    )
    reference = compute_reference_value(0.95, 16, config)
    assert reference.success
    assert reference.source == "mpmath"
    assert reference.value == 3.0


def test_runtime_benchmark_smoke_runs_on_cpu():
    config = RuntimeConfig(
        dims=[8],
        spcauchy_methods=["spcauchy_hybrid"],
        vmf_methods=[],
        device_name="cpu",
        dtype=torch.float32,
        seed=1,
        batch_size=2,
        warmup_iters=0,
        measure_iters=1,
        timeout_s=5.0,
        concentration_mode="direct-rho",
        rho_values=[0.5],
    )
    records = run_runtime_benchmark(config)

    assert len(records) == 1
    record = records[0]
    assert record["method"] == "spcauchy_hybrid"
    assert record["benchmark"] == "runtime"
    assert "total_mean_s" in record


def test_robustness_benchmark_smoke_records_result():
    config = RobustnessConfig(
        spcauchy_dims=[2],
        vmf_dims=[],
        rho_grid=[0.5],
        spcauchy_methods=["hybrid"],
        vmf_methods=[],
        device_name="cpu",
        dtype=torch.float32,
        seed=1,
        batch_size=2,
        timeout_s=5.0,
    )
    records = run_robustness_benchmark(config)

    assert len(records) == 1
    record = records[0]
    assert record["benchmark"] == "robustness"
    assert record["method"] == "hybrid"
    assert "kl_error_threshold_exceeded" in record


def test_accuracy_plotting_smoke_writes_outputs():
    tmp_path = Path.cwd() / "latent_benchmark_test_output"
    shutil.rmtree(tmp_path, ignore_errors=True)
    original_savefig = Figure.savefig

    def fake_savefig(self, fname, *args, **kwargs):
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        Path(fname).touch()

    Figure.savefig = fake_savefig
    try:
        records = [
            {
                "benchmark": "accuracy",
                "method": method,
                "family": "spcauchy",
                "device": "cpu",
                "dtype": "float64",
                "seed": 0,
                "dim": dim,
                "rho": rho,
                "kappa": None,
                "batch_size": None,
                "success": True,
                "failure_type": None,
                "error_message": None,
                "kl_value": 0.2 + 0.05 * dim + rho,
                "eval_time_s": 1e-4 * (dim + 1) * (1.0 if method == "hybrid" else 1.5),
                "reference_value": 0.2 + 0.05 * dim + rho,
                "reference_source": "synthetic",
                "reference_success": True,
                "reference_failure_type": None,
                "reference_error_message": None,
                "abs_error": 1e-8 if method == "hybrid" else 1e-5,
                "rel_error": 1e-8 if method == "hybrid" else 1e-5,
            }
            for method in ["hybrid", "quadrature"]
            for dim in [2, 8]
            for rho in [0.5, 0.9]
        ]
        _, csv_path, figures = save_accuracy_outputs(records, out_dir=str(tmp_path))

        assert tmp_path.joinpath("results", "spcauchy_kl_accuracy.csv").exists()
        assert csv_path.endswith("spcauchy_kl_accuracy.csv")
        assert figures
        for figure in figures:
            assert Path(figure).exists()
    finally:
        Figure.savefig = original_savefig
        shutil.rmtree(tmp_path, ignore_errors=True)
