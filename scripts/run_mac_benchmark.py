"""Comprehensive benchmarking and profiling suite for TorchWave and JAXWave on mac."""

import argparse
import io
import json
import os
import pstats
import cProfile
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Ensure diffgw packages are importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torchwave
from torchwave.benchmark import (
    DEFAULT_MODEL_SPECS,
    benchmark_torchwave_model,
    format_markdown_table,
    format_table,
    resolve_device,
    resolve_dtype,
    sync_device,
)
from torchwave.profile import run_torch_profiler, run_cprofile, run_memory_profile

HAS_JAX = None

def _get_jax():
    global HAS_JAX
    if HAS_JAX is None:
        try:
            import jax
            import jax.numpy as jnp
            import jaxwave
            HAS_JAX = True
        except ImportError:
            HAS_JAX = False
    return HAS_JAX



def benchmark_jaxwave_model(
    model_name: str,
    batch_size: int,
    grid: Dict[str, float],
    params: Dict[str, Any],
    iterations: int = 20,
    warmup: int = 5,
    use_jit: bool = True,
) -> Dict[str, Any]:
    """Benchmark jaxwave waveform generation."""
    flow = float(grid["flow"])
    fhigh = float(grid["fhigh"])
    df = float(grid["df"])
    n_freqs = int(round((fhigh - flow) / df)) + 1
    freqs = jnp.linspace(flow, fhigh, n_freqs, dtype=jnp.float64)

    # Build inputs
    kw = {}
    for k, v in params.items():
        if isinstance(v, (int, float)):
            if batch_size > 1:
                kw[k] = jnp.full((batch_size,), float(v), dtype=jnp.float64)
            else:
                kw[k] = float(v)
        else:
            kw[k] = v

    def eval_fn(frequencies, **kwargs):
        return jaxwave.get_fd_waveform(model_name, sample_frequencies=frequencies, **kwargs)

    if use_jit:
        eval_fn = jax.jit(eval_fn)

    # Warmup / JIT compile
    t0_compile = time.perf_counter()
    for _ in range(warmup):
        res = eval_fn(freqs, **kw)
        if isinstance(res, tuple):
            res[0].block_until_ready()
            res[1].block_until_ready()
    t_compile = (time.perf_counter() - t0_compile) * 1000.0

    # Timing
    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        res = eval_fn(freqs, **kw)
        if isinstance(res, tuple):
            res[0].block_until_ready()
            res[1].block_until_ready()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    timings_arr = np.array(timings)
    mean_ms = float(np.mean(timings_arr))
    wf_per_sec = (batch_size * 1000.0) / mean_ms if mean_ms > 0 else 0.0

    return {
        "framework": "jax",
        "model": model_name,
        "batch_size": batch_size,
        "device": "cpu",
        "dtype": "float64",
        "jit": use_jit,
        "n_freq_bins": n_freqs,
        "iterations": iterations,
        "compile_warmup_ms": t_compile,
        "mean_ms": mean_ms,
        "std_ms": float(np.std(timings_arr)),
        "median_ms": float(np.median(timings_arr)),
        "min_ms": float(np.min(timings_arr)),
        "max_ms": float(np.max(timings_arr)),
        "p95_ms": float(np.percentile(timings_arr, 95)),
        "wf_per_sec": wf_per_sec,
    }


def main():
    parser = argparse.ArgumentParser(description="Run TorchWave & JAXWave benchmarks on mac")
    parser.add_argument("--models", type=str, default="TaylorF2,IMRPhenomD,IMRPhenomXAS,IMRPhenomXP,IMRPhenomD_NRTidalv2")
    parser.add_argument("--devices", type=str, default="cuda,cpu")
    parser.add_argument("--cuda-batch-sizes", type=str, default="1,10,100,1000,10000")
    parser.add_argument("--cpu-batch-sizes", type=str, default="1,10,100,1000")
    parser.add_argument("--dtypes", type=str, default="float64,float32")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--skip-jax", action="store_true")
    parser.add_argument("--run-profiles", action="store_true")
    parser.add_argument("--output", type=str, default="mac_benchmark_results.json")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    cuda_batches = [int(b.strip()) for b in args.cuda_batch_sizes.split(",") if b.strip()]
    cpu_batches = [int(b.strip()) for b in args.cpu_batch_sizes.split(",") if b.strip()]
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]

    all_results = []
    profiles = {}
    print("=" * 80)
    print(f"Starting diffgw Benchmarking Suite on {os.environ.get('COMPUTERNAME', 'mac')}")
    print(f"PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    has_jax = False if args.skip_jax else _get_jax()
    print(f"JAX Available: {has_jax}")
    print("=" * 80, flush=True)

    # 1. PyTorch Benchmarks
    for dev_str in devices:
        if dev_str == "cuda" and not torch.cuda.is_available():
            print("CUDA not available; skipping CUDA benchmarks.")
            continue
        dev = torch.device(dev_str)
        batch_list = cuda_batches if dev_str == "cuda" else cpu_batches

        for dt_str in dtypes:
            dt = getattr(torch, dt_str)
            print(f"\n--- Framework: PyTorch | Device: {dev_str.upper()} | Precision: {dt_str} ---", flush=True)

            for model_name in models:
                spec = DEFAULT_MODEL_SPECS.get(model_name)
                if not spec:
                    print(f"Unknown model: {model_name}; skipping.")
                    continue

                for bsz in batch_list:
                    try:
                        if dev_str == "cuda":
                            torch.cuda.empty_cache()
                        res = benchmark_torchwave_model(
                            model_name=model_name,
                            batch_size=bsz,
                            device=dev,
                            dtype=dt,
                            grid=spec["grid"],
                            params=spec["params"],
                            iterations=args.iterations,
                            warmup=args.warmup,
                            measure_backward=False,
                        )
                        if dev_str == "cuda":
                            torch.cuda.empty_cache()
                        res["framework"] = "torchwave"
                        all_results.append(res)
                        print(
                            f"[{dev_str.upper():4s} | {dt_str:7s}] {model_name:22s} | B={bsz:<5d} | "
                            f"Mean: {res['mean_ms']:8.3f} ms | Med: {res['median_ms']:8.3f} ms | "
                            f"Throughput: {res['wf_per_sec']:10,.1f} wf/s",
                            flush=True,
                        )
                    except Exception as e:
                        if dev_str == "cuda":
                            torch.cuda.empty_cache()
                        print(f"[{dev_str.upper():4s} | {dt_str:7s}] {model_name:22s} | B={bsz:<5d} | ERROR: {e}", flush=True)

    def save_output():
        output_path = Path(args.output)
        output_data = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "host": os.environ.get("COMPUTERNAME", "mac"),
                "python": sys.version,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "has_jax": has_jax,
            },
            "benchmarks": all_results,
            "profiles": profiles,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"[Incremental save to {output_path.resolve()}]", flush=True)

    save_output()

    # 2. JAX Benchmarks (CPU)
    if has_jax and not args.skip_jax:
        print("\n--- Framework: JAX | Device: CPU | Precision: float64 (with JIT) ---", flush=True)
        for model_name in models:
            spec = DEFAULT_MODEL_SPECS.get(model_name)
            if not spec:
                continue
            for bsz in cpu_batches:
                try:
                    res = benchmark_jaxwave_model(
                        model_name=model_name,
                        batch_size=bsz,
                        grid=spec["grid"],
                        params=spec["params"],
                        iterations=args.iterations,
                        warmup=args.warmup,
                        use_jit=True,
                    )
                    all_results.append(res)
                    print(
                        f"[JAX  | float64] {model_name:22s} | B={bsz:<5d} | "
                        f"Mean: {res['mean_ms']:8.3f} ms | Med: {res['median_ms']:8.3f} ms | "
                        f"Throughput: {res['wf_per_sec']:10,.1f} wf/s | JIT: {res['compile_warmup_ms']:6.1f} ms",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[JAX  | float64] {model_name:22s} | B={bsz:<5d} | ERROR: {e}", flush=True)
        save_output()

    # 3. Profiling
    if args.run_profiles and torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("Running Deep Profiling (PyTorch Profiler & cProfile) on RTX 5090 CUDA...")
        print("=" * 80, flush=True)

        for prof_model in ["IMRPhenomD", "TaylorF2", "IMRPhenomXAS", "IMRPhenomXP"]:
            if prof_model not in DEFAULT_MODEL_SPECS:
                continue
            spec = DEFAULT_MODEL_SPECS[prof_model]
            print(f"\n>>> Profiling {prof_model} (CUDA, B=1) Operator Table:")
            op_table_b1 = run_torch_profiler(
                model_name=prof_model,
                batch_size=1,
                device=torch.device("cuda"),
                dtype=torch.float64,
                grid=spec["grid"],
                params=spec["params"],
                top_k=15,
                sort_by="cuda_time_total",
                profile_memory=False,
            )
            print(op_table_b1, flush=True)

            bsz_high = 10 if prof_model == "IMRPhenomXP" else 1000
            print(f"\n>>> Profiling {prof_model} (CUDA, B={bsz_high}) Operator Table:")
            op_table_b_high = run_torch_profiler(
                model_name=prof_model,
                batch_size=bsz_high,
                device=torch.device("cuda"),
                dtype=torch.float64,
                grid=spec["grid"],
                params=spec["params"],
                top_k=15,
                sort_by="cuda_time_total",
                profile_memory=False,
            )
            print(op_table_b_high, flush=True)

            # cProfile to see CPU bottleneck
            print(f"\n>>> cProfile for {prof_model} (CUDA, B=1, 10 iterations):")
            cprof_out = run_cprofile(
                model_name=prof_model,
                batch_size=1,
                device=torch.device("cuda"),
                dtype=torch.float64,
                grid=spec["grid"],
                params=spec["params"],
                top_k=15,
                sort_by="cumtime",
                iterations=10,
            )
            print(cprof_out, flush=True)

            profiles[prof_model] = {
                "b1_op_table": op_table_b1,
                f"b{bsz_high}_op_table": op_table_b_high,
                "cprofile": cprof_out,
            }
            save_output()

    print(f"\n[Benchmarking & Profiling Complete]", flush=True)


if __name__ == "__main__":
    main()
