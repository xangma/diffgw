#!/usr/bin/env python3
"""Comprehensive profiling and benchmark suite for TorchWave on CPU and GPU."""

import argparse
import time
import sys
import os
from typing import Tuple

import torch
import numpy as np

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from torchwave.waveforms import get_fd_waveform

try:
    import lalsimulation as lalsim
    import lal
    HAS_LAL = True
except ImportError:
    HAS_LAL = False


def benchmark_lal(approximant: str, mass1: float, mass2: float, spin1z: float, spin2z: float,
                  f_lower: float, f_final: float, delta_f: float, n_runs: int = 50) -> float:
    """Benchmark LALSimulation single-waveform generation."""
    if not HAS_LAL:
        return float("nan")

    approx_enum = lalsim.GetApproximantFromString(approximant)
    
    # Warmup
    for _ in range(5):
        hp, hc = lalsim.SimInspiralChooseFDWaveform(
            mass1 * lal.MSUN_SI, mass2 * lal.MSUN_SI,
            0.0, 0.0, spin1z,
            0.0, 0.0, spin2z,
            100.0 * 1e6 * lal.PC_SI,
            0.0, 0.0, 0.0, 0.0, 0.0,
            delta_f, f_lower, f_final, f_lower,
            None, approx_enum,
        )

    t0 = time.perf_counter()
    for _ in range(n_runs):
        hp, hc = lalsim.SimInspiralChooseFDWaveform(
            mass1 * lal.MSUN_SI, mass2 * lal.MSUN_SI,
            0.0, 0.0, spin1z,
            0.0, 0.0, spin2z,
            100.0 * 1e6 * lal.PC_SI,
            0.0, 0.0, 0.0, 0.0, 0.0,
            delta_f, f_lower, f_final, f_lower,
            None, approx_enum,
        )
    t1 = time.perf_counter()
    return (t1 - t0) / n_runs


def generate_parameter_batch(batch_size: int, device: str = "cpu", dtype: torch.dtype = torch.float64):
    """Generate reproducible parameter tensors for batched evaluation."""
    torch.manual_seed(42)
    m1 = torch.linspace(20.0, 60.0, batch_size, device=device, dtype=dtype)
    m2 = torch.linspace(10.0, 30.0, batch_size, device=device, dtype=dtype)
    s1z = torch.linspace(-0.8, 0.8, batch_size, device=device, dtype=dtype)
    s2z = torch.linspace(-0.8, 0.8, batch_size, device=device, dtype=dtype)
    s1x = torch.linspace(0.01, 0.4, batch_size, device=device, dtype=dtype)
    s1y = torch.linspace(0.01, 0.4, batch_size, device=device, dtype=dtype)
    s2x = torch.linspace(-0.4, -0.01, batch_size, device=device, dtype=dtype)
    s2y = torch.linspace(-0.4, -0.01, batch_size, device=device, dtype=dtype)
    return m1, m2, s1x, s1y, s1z, s2x, s2y, s2z


def benchmark_torchwave(approximant: str, batch_size: int, device: str = "cpu",
                       dtype: torch.dtype = torch.float64, f_lower: float = 20.0,
                       f_final: float = 1024.0, delta_f: float = 0.25,
                       n_warmup: int = 5, n_runs: int = 20) -> Tuple[float, float]:
    """Benchmark TorchWave for a given model, batch size, and device.
    
    Returns (latency_seconds, throughput_wf_per_sec).
    """
    f = torch.arange(f_lower, f_final, delta_f, device=device, dtype=dtype)
    m1, m2, s1x, s1y, s1z, s2x, s2y, s2z = generate_parameter_batch(batch_size, device=device, dtype=dtype)
    
    kwargs = {
        "sample_frequencies": f,
        "mass1": m1 if batch_size > 1 else m1[0],
        "mass2": m2 if batch_size > 1 else m2[0],
        "distance": 100.0,
        "inclination": 0.5,
        "phic": 0.2,
        "tc": 0.0,
        "device": device,
        "dtype": dtype,
    }
    
    if approximant in ("IMRPhenomXP", "IMRPhenomXPHM", "IMRPhenomPv2"):
        kwargs.update({
            "spin1x": s1x if batch_size > 1 else s1x[0],
            "spin1y": s1y if batch_size > 1 else s1y[0],
            "spin1z": s1z if batch_size > 1 else s1z[0],
            "spin2x": s2x if batch_size > 1 else s2x[0],
            "spin2y": s2y if batch_size > 1 else s2y[0],
            "spin2z": s2z if batch_size > 1 else s2z[0],
        })
    else:
        kwargs.update({
            "spin1z": s1z if batch_size > 1 else s1z[0],
            "spin2z": s2z if batch_size > 1 else s2z[0],
        })

    # Warmup
    for _ in range(n_warmup):
        hp, hc = get_fd_waveform(approximant=approximant, **kwargs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        del hp, hc

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        hp, hc = get_fd_waveform(approximant=approximant, **kwargs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        del hp, hc
    t1 = time.perf_counter()

    latency = (t1 - t0) / n_runs
    throughput = batch_size / latency
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return latency, throughput


def run_profiler(approximant: str, batch_size: int, device: str = "cuda"):
    """Profile CUDA or CPU kernels using torch.profiler."""
    print(f"\n{'='*70}\nPROFILING {approximant} (batch_size={batch_size}, device={device})\n{'='*70}", flush=True)
    f = torch.arange(20.0, 1024.0, 0.25, device=device, dtype=torch.float64)
    m1, m2, s1x, s1y, s1z, s2x, s2y, s2z = generate_parameter_batch(batch_size, device=device, dtype=torch.float64)
    
    kwargs = {
        "sample_frequencies": f,
        "mass1": m1,
        "mass2": m2,
        "distance": 100.0,
        "inclination": 0.5,
        "phic": 0.2,
        "tc": 0.0,
        "device": device,
        "dtype": torch.float64,
    }
    if approximant in ("IMRPhenomXP", "IMRPhenomXPHM", "IMRPhenomPv2"):
        kwargs.update({"spin1x": s1x, "spin1y": s1y, "spin1z": s1z, "spin2x": s2x, "spin2y": s2y, "spin2z": s2z})
    else:
        kwargs.update({"spin1z": s1z, "spin2z": s2z})

    # Warmup
    for _ in range(3):
        get_fd_waveform(approximant=approximant, **kwargs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(5):
            get_fd_waveform(approximant=approximant, **kwargs)
            if device.startswith("cuda"):
                torch.cuda.synchronize()

    sort_by = "cuda_time_total" if device.startswith("cuda") else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_by, row_limit=15), flush=True)


def main():
    parser = argparse.ArgumentParser(description="TorchWave Profiler & Benchmark")
    parser.add_argument("--profile", action="store_true", help="Run torch.profiler on key models")
    parser.add_argument("--device", default="all", choices=["all", "cpu", "cuda"])
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with fewer models/sizes")
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"], help="Data type to benchmark")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    cuda_available = torch.cuda.is_available()
    print("Hardware & Environment:", flush=True)
    print(f"  PyTorch version: {torch.__version__}", flush=True)
    print(f"  CPU threads: {torch.get_num_threads()}", flush=True)
    print(f"  CUDA available: {cuda_available}", flush=True)
    if cuda_available:
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB", flush=True)
    print(f"  Benchmark DType: {dtype}", flush=True)

    models = ["IMRPhenomD", "IMRPhenomXAS", "IMRPhenomXHM", "IMRPhenomXP", "IMRPhenomXPHM", "TaylorF2"]
    if args.quick:
        models = ["IMRPhenomD", "IMRPhenomXAS", "IMRPhenomXHM"]

    # 1. Single waveform comparison against LAL
    print(f"\n{'='*85}", flush=True)
    print("SINGLE WAVEFORM LATENCY ($N=1$, 20-1024 Hz, df=0.25 Hz)", flush=True)
    print(f"{'='*85}", flush=True)
    print(f"{'Model':<16} | {'LAL CPU (ms)':<14} | {'TW CPU N=1 (ms)':<16} | {'TW CPU vs LAL':<14} | {'TW GPU N=1 (ms)':<15}", flush=True)
    print(f"{'-'*85}", flush=True)
    
    for m in models:
        lal_name = m
        t_lal = benchmark_lal(lal_name, 35.0, 30.0, 0.3, -0.2, 20.0, 1024.0, 0.25, n_runs=20)
        t_cpu, _ = benchmark_torchwave(m, batch_size=1, device="cpu", dtype=dtype, n_runs=20)
        speedup = t_lal / t_cpu if (not np.isnan(t_lal) and t_cpu > 0) else float("nan")
        
        t_gpu_str = "N/A"
        if cuda_available and args.device in ("all", "cuda"):
            t_gpu, _ = benchmark_torchwave(m, batch_size=1, device="cuda", dtype=dtype, n_runs=20)
            t_gpu_str = f"{t_gpu*1e3:.3f}"
            
        lal_str = f"{t_lal*1e3:.3f}" if not np.isnan(t_lal) else "N/A"
        speedup_str = f"{speedup:.2f}x" if not np.isnan(speedup) else "N/A"
        print(f"{m:<16} | {lal_str:<14} | {t_cpu*1e3:<16.3f} | {speedup_str:<14} | {t_gpu_str:<15}", flush=True)

    # 2. Batched Throughput Scaling on CPU
    if args.device in ("all", "cpu"):
        print(f"\n{'='*85}", flush=True)
        print("BATCHED THROUGHPUT SCALING ON CPU (waveforms / second, 64 cores)", flush=True)
        print(f"{'='*85}", flush=True)
        cpu_batch_sizes = [1, 16, 64, 256, 1024] if not args.quick else [1, 64, 256]
        header = f"{'Model':<16} | " + " | ".join([f"N={bs:<6}" for bs in cpu_batch_sizes])
        print(header, flush=True)
        print(f"{'-'*len(header)}", flush=True)
        for m in models:
            row = [f"{m:<16}"]
            is_serial = m in ("IMRPhenomXP", "IMRPhenomXPHM")
            for bs in cpu_batch_sizes:
                if is_serial and bs > 256:
                    row.append(f"{'N/A(loop)':>8}")
                    continue
                try:
                    n_runs = 3 if bs >= 256 else (5 if bs >= 64 else 10)
                    _, thr = benchmark_torchwave(m, batch_size=bs, device="cpu", dtype=dtype, n_runs=n_runs)
                    row.append(f"{thr:>8.1f}")
                except Exception:
                    row.append(f"{'ERR':>8}")
            print(" | ".join(row), flush=True)

    # 3. Batched Throughput Scaling on GPU
    if cuda_available and args.device in ("all", "cuda"):
        print(f"\n{'='*95}", flush=True)
        print("BATCHED THROUGHPUT SCALING ON GPU (waveforms / second) [RTX 4090 24GB]", flush=True)
        print(f"{'='*95}", flush=True)
        gpu_batch_sizes = [1, 64, 256, 1024, 4096, 16384, 65536] if not args.quick else [1, 256, 1024, 4096]
        header = f"{'Model':<16} | " + " | ".join([f"N={bs:<7}" for bs in gpu_batch_sizes])
        print(header, flush=True)
        print(f"{'-'*len(header)}", flush=True)
        for m in models:
            row = [f"{m:<16}"]
            is_serial = m in ("IMRPhenomXP", "IMRPhenomXPHM")
            for bs in gpu_batch_sizes:
                if is_serial and bs > 256:
                    row.append(f"{'N/A(loop)':>9}")
                    continue
                try:
                    # Clean up GPU memory before big batches
                    torch.cuda.empty_cache()
                    n_runs = 3 if bs >= 16384 else (8 if bs >= 1024 else 15)
                    _, thr = benchmark_torchwave(m, batch_size=bs, device="cuda", dtype=dtype, n_runs=n_runs)
                    row.append(f"{thr:>9.1f}")
                except Exception:
                    row.append(f"{'OOM/ERR':>9}")
            print(" | ".join(row), flush=True)

    # 4. Profiler breakdown
    if args.profile:
        for m in (["IMRPhenomD", "IMRPhenomXHM", "IMRPhenomXPHM"] if not args.quick else ["IMRPhenomXHM"]):
            run_profiler(m, batch_size=1024, device="cuda" if cuda_available else "cpu")


if __name__ == "__main__":
    main()
