#!/usr/bin/env python3
"""Diagnostic benchmark comparing torch eager vs torch.compile."""

import argparse
import sys
import time
from pathlib import Path
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torchwave
from torchwave.waveforms import get_fd_waveform


def bench_model(model_name="TaylorF2", batch_size=1000, device_str="cuda", dtype_str="float64"):
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)
    freqs = torch.linspace(20.0, 512.0, 1969, dtype=dtype, device=device)

    m1 = torch.full((batch_size,), 30.0, dtype=dtype, device=device)
    m2 = torch.full((batch_size,), 20.0, dtype=dtype, device=device)

    def eval_fn(mass1, mass2):
        return get_fd_waveform(model_name, sample_frequencies=freqs, mass1=mass1, mass2=mass2, device=device, dtype=dtype)

    # 1. Eager Benchmark
    for _ in range(5):
        _ = eval_fn(m1, m2)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(25):
        _ = eval_fn(m1, m2)
    if device.type == "cuda":
        torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) / 25 * 1000.0
    eager_tps = batch_size / (eager_ms / 1000.0)

    # 2. Compiled Benchmark
    compiled_eval = torch.compile(eval_fn, fullgraph=False)
    t0_compile = time.perf_counter()
    _ = compiled_eval(m1, m2)
    if device.type == "cuda":
        torch.cuda.synchronize()
    compile_warmup_ms = (time.perf_counter() - t0_compile) * 1000.0

    for _ in range(5):
        _ = compiled_eval(m1, m2)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(25):
        _ = compiled_eval(m1, m2)
    if device.type == "cuda":
        torch.cuda.synchronize()
    compiled_ms = (time.perf_counter() - t0) / 25 * 1000.0
    compiled_tps = batch_size / (compiled_ms / 1000.0)

    speedup = eager_ms / compiled_ms
    print(f"[{model_name:12s}] B={batch_size:<5d} | Eager: {eager_ms:7.3f} ms ({eager_tps:10,.1f} wf/s) | Compiled: {compiled_ms:7.3f} ms ({compiled_tps:10,.1f} wf/s) | Speedup: {speedup:5.2f}x | Warmup: {compile_warmup_ms:6.1f} ms")
    return {
        "model": model_name,
        "batch_size": batch_size,
        "eager_ms": eager_ms,
        "eager_tps": eager_tps,
        "compiled_ms": compiled_ms,
        "compiled_tps": compiled_tps,
        "speedup": speedup,
        "compile_warmup_ms": compile_warmup_ms,
    }


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Torch Eager vs Torch Compiled Benchmark on {dev} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})...\n")
    for model in ["TaylorF2", "IMRPhenomD", "IMRPhenomXAS"]:
        for bs in [1, 10, 100, 1000, 5000]:
            try:
                bench_model(model, batch_size=bs, device_str=dev)
            except Exception as e:
                print(f"[{model:12s}] B={bs:<5d} | ERROR: {e}")


if __name__ == "__main__":
    main()
