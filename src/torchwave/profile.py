"""Deep Performance & Memory Profiler for TorchWave Waveforms.

Provides operator-level and function-level profiling of waveform generation
using:
1. PyTorch Profiler (CPU/MPS/CUDA kernel timings, operator shapes, tensor memory)
2. Python cProfile (function call overhead, pure-Python bottlenecks)
3. Memory Profiler (tracemalloc, peak allocation, tensor footprints, leak checks)
4. Chrome Trace JSON export for visualization in chrome://tracing or Perfetto.

Can be run programmatically or via CLI:
    python -m torchwave.profile --model IMRPhenomD --mode all
    torchwave-profile --model TaylorF2 --backward
"""

import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time
import tracemalloc
from typing import Any, Dict, List, Optional

import torch

from torchwave.waveforms import get_fd_waveform
from torchwave.benchmark import (
    DEFAULT_MODEL_SPECS,
    resolve_device,
    resolve_dtype,
    sync_device,
)


def run_torch_profiler(
    model_name: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    grid: Dict[str, float],
    params: Dict[str, Any],
    top_k: int = 20,
    sort_by: str = "cpu_time_total",
    measure_backward: bool = False,
    trace_path: Optional[str] = None,
) -> str:
    """Profile waveform generation using PyTorch's native profiler."""
    flow = float(grid["flow"])
    fhigh = float(grid["fhigh"])
    df = float(grid["df"])
    n_freqs = int(round((fhigh - flow) / df)) + 1
    freqs = torch.linspace(flow, fhigh, n_freqs, dtype=dtype, device=device)

    # Build inputs
    kw = {}
    for k, v in params.items():
        if isinstance(v, (int, float)):
            if batch_size > 1:
                t = torch.full((batch_size,), float(v), dtype=dtype, device=device)
            else:
                t = torch.tensor(float(v), dtype=dtype, device=device)
            if measure_backward and k in ("mass1", "mass2", "chi1", "chi2", "spin1z", "spin2z"):
                t = t.clone().detach().requires_grad_(True)
            kw[k] = t
        else:
            kw[k] = v

    # Warmup
    hp, hc = get_fd_waveform(
        model_name, sample_frequencies=freqs, device=device, dtype=dtype, **kw
    )
    if measure_backward:
        (hp.abs().sum() + hc.abs().sum()).backward()
    sync_device(device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.profiler.record_function(f"{model_name}_generation"):
            hp, hc = get_fd_waveform(
                model_name, sample_frequencies=freqs, device=device, dtype=dtype, **kw
            )
            if measure_backward:
                with torch.profiler.record_function(f"{model_name}_backward"):
                    loss = (hp.abs().sum() + hc.abs().sum())
                    loss.backward()
            sync_device(device)

    if trace_path:
        prof.export_chrome_trace(trace_path)
        print(f"Chrome trace successfully saved to: {trace_path}")
        print("  -> Open in Chrome at chrome://tracing or https://ui.perfetto.dev")

    table = prof.key_averages().table(sort_by=sort_by, row_limit=top_k)
    return table


def run_cprofile(
    model_name: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    grid: Dict[str, float],
    params: Dict[str, Any],
    top_k: int = 20,
    sort_by: str = "cumtime",
    iterations: int = 10,
    measure_backward: bool = False,
) -> str:
    """Profile Python-level execution using cProfile."""
    flow = float(grid["flow"])
    fhigh = float(grid["fhigh"])
    df = float(grid["df"])
    n_freqs = int(round((fhigh - flow) / df)) + 1
    freqs = torch.linspace(flow, fhigh, n_freqs, dtype=dtype, device=device)

    def run_eval():
        for _ in range(iterations):
            kw = {}
            for k, v in params.items():
                if isinstance(v, (int, float)):
                    if batch_size > 1:
                        t = torch.full((batch_size,), float(v), dtype=dtype, device=device)
                    else:
                        t = torch.tensor(float(v), dtype=dtype, device=device)
                    if measure_backward and k in ("mass1", "mass2", "chi1", "chi2"):
                        t = t.clone().detach().requires_grad_(True)
                    kw[k] = t
                else:
                    kw[k] = v

            hp, hc = get_fd_waveform(
                model_name, sample_frequencies=freqs, device=device, dtype=dtype, **kw
            )
            if measure_backward:
                loss = (hp.abs().sum() + hc.abs().sum())
                loss.backward()
            sync_device(device)

    # Warmup once
    run_eval()

    profiler = cProfile.Profile()
    profiler.enable()
    run_eval()
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by)
    ps.print_stats(top_k)
    return s.getvalue()


def run_memory_profile(
    model_name: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    grid: Dict[str, float],
    params: Dict[str, Any],
    top_k: int = 10,
    measure_backward: bool = False,
) -> Dict[str, Any]:
    """Profile memory usage, peak allocations, and check for memory leaks."""
    flow = float(grid["flow"])
    fhigh = float(grid["fhigh"])
    df = float(grid["df"])
    n_freqs = int(round((fhigh - flow) / df)) + 1
    freqs = torch.linspace(flow, fhigh, n_freqs, dtype=dtype, device=device)

    kw = {}
    for k, v in params.items():
        if isinstance(v, (int, float)):
            if batch_size > 1:
                t = torch.full((batch_size,), float(v), dtype=dtype, device=device)
            else:
                t = torch.tensor(float(v), dtype=dtype, device=device)
            if measure_backward and k in ("mass1", "mass2", "chi1", "chi2"):
                t = t.clone().detach().requires_grad_(True)
            kw[k] = t
        else:
            kw[k] = v

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    hp, hc = get_fd_waveform(
        model_name, sample_frequencies=freqs, device=device, dtype=dtype, **kw
    )
    if measure_backward:
        loss = (hp.abs().sum() + hc.abs().sum())
        loss.backward()
    sync_device(device)

    snapshot_after = tracemalloc.take_snapshot()
    current_size, peak_size = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Calculate tensor sizes
    hp_bytes = hp.element_size() * hp.nelement()
    hc_bytes = hc.element_size() * hc.nelement()
    output_tensor_bytes = hp_bytes + hc_bytes

    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
    top_allocations = []
    for stat in top_stats[:top_k]:
        top_allocations.append({
            "trace": str(stat),
            "size_diff_kb": stat.size_diff / 1024.0,
            "count_diff": stat.count_diff,
        })

    # Leak test over 5 iterations
    tracemalloc.start()
    mem_snapshots = []
    for _ in range(5):
        hp, hc = get_fd_waveform(
            model_name, sample_frequencies=freqs, device=device, dtype=dtype, **kw
        )
        if measure_backward:
            loss = (hp.abs().sum() + hc.abs().sum())
            loss.backward()
        sync_device(device)
        mem_snapshots.append(tracemalloc.get_traced_memory()[0])
    tracemalloc.stop()

    leak_delta_kb = (mem_snapshots[-1] - mem_snapshots[0]) / 1024.0

    return {
        "output_tensor_shape": list(hp.shape),
        "output_tensor_memory_kb": output_tensor_bytes / 1024.0,
        "peak_traced_memory_kb": peak_size / 1024.0,
        "leak_growth_over_5_runs_kb": leak_delta_kb,
        "top_allocations": top_allocations,
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="TorchWave Waveform Profiling and Diagnostics Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="IMRPhenomD",
        help="Model approximant to profile (e.g. TaylorF2, IMRPhenomD, IMRPhenomXAS, etc.).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for profiling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run profile on ('cpu', 'mps', 'cuda', 'auto').",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        help="Floating point precision ('float32', 'float64').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="torch",
        choices=["torch", "cprofile", "memory", "all"],
        help="Profiling engine to run.",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Profile autograd backward pass in addition to forward pass.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top lines/operators to display.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="cpu_time_total",
        help="Sort key for PyTorch profiler (or cumtime/tottime for cProfile).",
    )
    parser.add_argument(
        "--trace-out",
        type=str,
        default=None,
        help="Path to export Chrome Trace JSON file.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Output path for JSON profiling results.",
    )

    args = parser.parse_args(argv)

    spec = DEFAULT_MODEL_SPECS.get(args.model)
    if spec is None:
        print(f"Error: model '{args.model}' not recognized. Available models:")
        print(", ".join(sorted(DEFAULT_MODEL_SPECS.keys())))
        sys.exit(1)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print("=" * 80)
    print(f"TORCHWAVE PROFILER: {args.model}")
    print(f"Device: {device} | Dtype: {dtype} | Batch Size: {args.batch_size} | Mode: {args.mode}")
    print(f"Autograd Backward: {args.backward}")
    print("=" * 80)

    json_report: Dict[str, Any] = {
        "model": args.model,
        "batch_size": args.batch_size,
        "device": str(device),
        "dtype": str(dtype),
        "backward": args.backward,
    }

    if args.mode in ("torch", "all"):
        print("\n--- [PyTorch Operator Profiler] ---")
        torch_table = run_torch_profiler(
            model_name=args.model,
            batch_size=args.batch_size,
            device=device,
            dtype=dtype,
            grid=spec["grid"],
            params=spec["params"],
            top_k=args.top_k,
            sort_by=args.sort_by,
            measure_backward=args.backward,
            trace_path=args.trace_out,
        )
        print(torch_table)
        json_report["torch_profiler_output"] = torch_table

    if args.mode in ("cprofile", "all"):
        print("\n--- [Python cProfile Stats] ---")
        cprof_sort = args.sort_by if args.sort_by in ("cumtime", "tottime", "calls") else "cumtime"
        cprof_table = run_cprofile(
            model_name=args.model,
            batch_size=args.batch_size,
            device=device,
            dtype=dtype,
            grid=spec["grid"],
            params=spec["params"],
            top_k=args.top_k,
            sort_by=cprof_sort,
            iterations=10,
            measure_backward=args.backward,
        )
        print(cprof_table)
        json_report["cprofile_output"] = cprof_table

    if args.mode in ("memory", "all"):
        print("\n--- [Memory & Tensor Footprint] ---")
        mem_info = run_memory_profile(
            model_name=args.model,
            batch_size=args.batch_size,
            device=device,
            dtype=dtype,
            grid=spec["grid"],
            params=spec["params"],
            top_k=args.top_k,
            measure_backward=args.backward,
        )
        print(f"Output Tensor Shape: {mem_info['output_tensor_shape']}")
        print(f"Output Polarization Memory: {mem_info['output_tensor_memory_kb']:.2f} KB")
        print(f"Peak Traced Process Memory: {mem_info['peak_traced_memory_kb']:.2f} KB")
        print(f"Memory Growth Across 5 Iterations (Leak Test): {mem_info['leak_growth_over_5_runs_kb']:.2f} KB")
        print("\nTop Memory Allocations (by line):")
        for alloc in mem_info["top_allocations"]:
            print(f"  +{alloc['size_diff_kb']:.1f} KB ({alloc['count_diff']} allocs): {alloc['trace']}")
        json_report["memory_profile"] = mem_info

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(json_report, f, indent=2)
        print(f"\nSaved profiling report to {args.json_out}")


if __name__ == "__main__":
    main()
