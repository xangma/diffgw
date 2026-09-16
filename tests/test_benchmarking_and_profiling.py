"""Tests for TorchWave benchmarking and profiling scripts."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

import torchwave
from torchwave.benchmark import (
    DEFAULT_MODEL_SPECS,
    _has_lal,
    benchmark_torchwave_model,
    format_markdown_table,
    format_table,
    resolve_device,
    resolve_dtype,
    run_benchmark_suite,
)
from torchwave.profile import (
    run_cprofile,
    run_memory_profile,
    run_torch_profiler,
)
from scripts.benchmark import run_benchmark_suite as script_run_benchmark_suite
from scripts.profile_waveform import run_torch_profiler as script_run_torch_profiler


def test_package_exports():
    """Verify that benchmark and profile modules are exported at top-level torchwave."""
    assert hasattr(torchwave, "benchmark")
    assert hasattr(torchwave, "profile")
    assert "benchmark" in torchwave.__all__
    assert "profile" in torchwave.__all__
    assert callable(torchwave.benchmark.run_benchmark_suite)
    assert callable(torchwave.profile.run_torch_profiler)
    assert callable(script_run_benchmark_suite)
    assert callable(script_run_torch_profiler)


def test_benchmark_suite_api():
    """Verify that run_benchmark_suite executes properly and returns structured metrics."""
    models = ["TaylorF2", "IMRPhenomD"]
    batch_sizes = [1, 2]
    device = torch.device("cpu")
    dtype = torch.float64

    res = run_benchmark_suite(
        models=models,
        batch_sizes=batch_sizes,
        device=device,
        dtype=dtype,
        iterations=2,
        warmup=1,
        measure_backward=True,
        compare_lal=_has_lal(),
    )

    assert "metadata" in res
    assert "benchmarks" in res
    assert len(res["benchmarks"]) == len(models) * len(batch_sizes)

    for b in res["benchmarks"]:
        assert "error" not in b
        assert b["mean_ms"] > 0.0
        assert b["wf_per_sec"] > 0.0
        assert "forward_mean_ms" in b
        assert "backward_mean_ms" in b
        if _has_lal() and b["model"] in ("TaylorF2", "IMRPhenomD"):
            assert "lal_mean_ms" in b
            assert "speedup_vs_lal" in b

    # Verify table formatters
    ascii_table = format_table(res)
    assert "TaylorF2" in ascii_table
    assert "IMRPhenomD" in ascii_table

    md_table = format_markdown_table(res)
    assert "| Model |" in md_table
    assert "TaylorF2" in md_table


def test_benchmark_cli_json_and_markdown(tmp_path: Path):
    """Verify benchmark.py runs via CLI and exports valid JSON and Markdown files."""
    json_path = tmp_path / "bench.json"
    md_path = tmp_path / "bench.md"

    cmd = [
        sys.executable,
        "scripts/benchmark.py",
        "--models", "TaylorF2",
        "--batch-sizes", "1",
        "--device", "cpu",
        "--dtype", "float64",
        "--iterations", "2",
        "--warmup", "1",
        "--json-out", str(json_path),
        "--markdown-out", str(md_path),
    ]

    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="src"))
    assert p.returncode == 0, f"Benchmark CLI failed: {p.stderr}"

    assert json_path.exists()
    with open(json_path, "r") as f:
        data = json.load(f)
    assert "benchmarks" in data
    assert len(data["benchmarks"]) == 1
    assert data["benchmarks"][0]["model"] == "TaylorF2"

    assert md_path.exists()
    with open(md_path, "r") as f:
        md_text = f.read()
    assert "TaylorF2" in md_text


def test_profiler_api():
    """Verify that profiler functions execute without errors."""
    spec = DEFAULT_MODEL_SPECS["IMRPhenomD"]
    device = torch.device("cpu")
    dtype = torch.float64

    # 1. PyTorch Profiler
    torch_table = run_torch_profiler(
        model_name="IMRPhenomD",
        batch_size=1,
        device=device,
        dtype=dtype,
        grid=spec["grid"],
        params=spec["params"],
        top_k=5,
        measure_backward=True,
    )
    assert "Self CPU" in torch_table

    # 2. cProfile
    cprof_table = run_cprofile(
        model_name="IMRPhenomD",
        batch_size=1,
        device=device,
        dtype=dtype,
        grid=spec["grid"],
        params=spec["params"],
        top_k=5,
        iterations=2,
    )
    assert "function calls" in cprof_table

    # 3. Memory Profile
    mem_info = run_memory_profile(
        model_name="IMRPhenomD",
        batch_size=1,
        device=device,
        dtype=dtype,
        grid=spec["grid"],
        params=spec["params"],
        top_k=5,
    )
    assert "output_tensor_shape" in mem_info
    assert mem_info["output_tensor_memory_kb"] > 0.0
    assert "top_allocations" in mem_info


def test_profile_waveform_cli(tmp_path: Path):
    """Verify profile_waveform.py runs via CLI and exports trace and JSON outputs."""
    trace_path = tmp_path / "trace.json"
    json_path = tmp_path / "prof.json"

    cmd = [
        sys.executable,
        "scripts/profile_waveform.py",
        "--model", "TaylorF2",
        "--batch-size", "1",
        "--device", "cpu",
        "--dtype", "float64",
        "--mode", "all",
        "--top-k", "5",
        "--trace-out", str(trace_path),
        "--json-out", str(json_path),
    ]

    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="src"))
    assert p.returncode == 0, f"Profiler CLI failed: {p.stderr}"

    assert trace_path.exists()
    with open(trace_path, "r") as f:
        trace_data = json.load(f)
    assert "traceEvents" in trace_data

    assert json_path.exists()
    with open(json_path, "r") as f:
        prof_data = json.load(f)
    assert prof_data["model"] == "TaylorF2"
    assert "torch_profiler_output" in prof_data
    assert "cprofile_output" in prof_data
    assert "memory_profile" in prof_data


def test_module_execution_benchmark(tmp_path: Path):
    """Verify python -m torchwave.benchmark executes properly."""
    json_path = tmp_path / "bench_module.json"
    cmd = [
        sys.executable,
        "-m", "torchwave.benchmark",
        "--models", "TaylorF2",
        "--batch-sizes", "1",
        "--device", "cpu",
        "--dtype", "float64",
        "--iterations", "2",
        "--warmup", "1",
        "--json-out", str(json_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="src"))
    assert p.returncode == 0, f"Module execution failed: {p.stderr}"
    assert json_path.exists()
    with open(json_path, "r") as f:
        data = json.load(f)
    assert "benchmarks" in data
    assert data["benchmarks"][0]["model"] == "TaylorF2"


def test_module_execution_profile(tmp_path: Path):
    """Verify python -m torchwave.profile executes properly."""
    json_path = tmp_path / "prof_module.json"
    cmd = [
        sys.executable,
        "-m", "torchwave.profile",
        "--model", "TaylorF2",
        "--batch-size", "1",
        "--device", "cpu",
        "--dtype", "float64",
        "--mode", "memory",
        "--top-k", "5",
        "--json-out", str(json_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="src"))
    assert p.returncode == 0, f"Module execution failed: {p.stderr}"
    assert json_path.exists()
    with open(json_path, "r") as f:
        data = json.load(f)
    assert data["model"] == "TaylorF2"
    assert "memory_profile" in data

