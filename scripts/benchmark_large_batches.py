#!/usr/bin/env python3
"""Ultra-Large Batch Scaling and Memory Profiling for DiffGW.

Evaluates waveform throughput, latency, and GPU memory utilization
across extreme batch scales (B = 1,000 to 50,000+) on modern accelerator hardware.
Tracks peak CUDA memory, per-waveform overhead, and gradient backprop scaling.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
for d in (str(_REPO_ROOT), str(_SRC_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)

import numpy as np
import torch

import torchwave
from torchwave.benchmark import (
    DEFAULT_MODEL_SPECS,
    calculate_sample_statistics,
    resolve_device,
    resolve_dtype,
    sync_device,
)
from torchwave.protocol import collect_system_provenance, seal_artifact


def measure_batch_scaling(
    model_name: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    grid: Dict[str, float],
    params: Dict[str, Any],
    iterations: int = 10,
    warmup: int = 3,
    measure_backward: bool = False,
) -> Dict[str, Any]:
    """Profile latency and VRAM for a single model at a specified batch size."""
    from torchwave.waveforms import get_fd_waveform

    flow = float(grid["flow"])
    fhigh = float(grid["fhigh"])
    df = float(grid["df"])
    n_freqs = int(round((fhigh - flow) / df)) + 1
    freqs = torch.linspace(flow, fhigh, n_freqs, device=device, dtype=dtype)

    def _make_inputs(req_grad: bool = False):
        inputs = {}
        for k, v in params.items():
            if isinstance(v, (int, float)):
                t = torch.full((batch_size,), float(v), device=device, dtype=dtype)
                if req_grad and k in ("mass1", "mass2", "chi1", "chi2", "spin1z", "spin2z"):
                    t.requires_grad_(True)
                inputs[k] = t
            else:
                inputs[k] = v
        return inputs

    # Clean memory before run
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    vram_before_mb = (
        torch.cuda.memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
    )

    ctx = torch.no_grad() if not measure_backward else torch.enable_grad()

    try:
        with ctx:
            # Warmup
            for _ in range(warmup):
                inp = _make_inputs(req_grad=measure_backward)
                hp, hc = get_fd_waveform(
                    model_name,
                    sample_frequencies=freqs,
                    device=device,
                    dtype=dtype,
                    **inp,
                )
                if measure_backward:
                    loss = hp.abs().sum() + hc.abs().sum()
                    loss.backward()
                sync_device(device)

            # Measurement
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            fwd_times: List[float] = []
            bwd_times: List[float] = []
            total_times: List[float] = []

            for _ in range(iterations):
                inp = _make_inputs(req_grad=measure_backward)
                sync_device(device)
                t0 = time.perf_counter()

                hp, hc = get_fd_waveform(
                    model_name,
                    sample_frequencies=freqs,
                    device=device,
                    dtype=dtype,
                    **inp,
                )
                sync_device(device)
                t_fwd = time.perf_counter()

                if measure_backward:
                    loss = hp.abs().sum() + hc.abs().sum()
                    loss.backward()
                    sync_device(device)
                    t_bwd = time.perf_counter()
                    bwd_times.append((t_bwd - t_fwd) * 1000.0)
                    total_times.append((t_bwd - t0) * 1000.0)
                else:
                    total_times.append((t_fwd - t0) * 1000.0)

                fwd_times.append((t_fwd - t0) * 1000.0)

        peak_vram_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
        )
        peak_res_mb = (
            torch.cuda.max_memory_reserved(device) / (1024 ** 2) if device.type == "cuda" else 0.0
        )

        stats = calculate_sample_statistics(total_times, batch_size)
        stats.update({
            "status": "SUCCESS",
            "model": model_name,
            "batch_size": batch_size,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "n_freq_bins": n_freqs,
            "measure_backward": measure_backward,
            "vram_before_mb": vram_before_mb,
            "peak_vram_mb": peak_vram_mb,
            "peak_reserved_mb": peak_res_mb,
            "vram_per_waveform_kb": (peak_vram_mb * 1024.0) / batch_size if batch_size > 0 else 0.0,
            "fwd_mean_ms": float(np.mean(fwd_times)),
        })
        if measure_backward:
            stats["bwd_mean_ms"] = float(np.mean(bwd_times))

        return stats

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        error_msg = str(e)
        is_oom = "out of memory" in error_msg.lower() or isinstance(e, torch.cuda.OutOfMemoryError)
        return {
            "status": "OOM" if is_oom else "ERROR",
            "model": model_name,
            "batch_size": batch_size,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "error": error_msg,
        }


def main():
    parser = argparse.ArgumentParser(description="DiffGW Ultra-Large Batch Scaling Benchmark")
    parser.add_argument(
        "--models",
        type=str,
        default="TaylorF2,IMRPhenomD,IMRPhenomXAS,IMRPhenomXP,IMRPhenomXHM",
        help="Comma-separated model names.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1000,5000,10000,25000,50000",
        help="Comma-separated batch sizes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Target device (cuda, mps, cpu, auto).",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        default="float64,float32",
        help="Comma-separated dtypes (float64, float32).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Timing iterations per batch size.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations.",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Benchmark backward pass (gradient backprop) alongside forward.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to output sealed receipt JSON.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate publication-grade plots (Fig 5 & 6) for VRAM and throughput.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Optional path to output plots directory.",
    )

    args = parser.parse_args()

    device = resolve_device(args.device)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",") if b.strip()]
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]

    provenance = collect_system_provenance()

    print("=" * 95)
    print("DIFFGW ULTRA-LARGE BATCH SCALING & MEMORY BENCHMARK")
    print(f"Host: {provenance['host']['hostname']} ({provenance['host']['os']})")
    gpu_devs = provenance.get("gpus", [])
    if gpu_devs:
        print(f"GPU:  {gpu_devs[0]['name']} ({gpu_devs[0]['total_memory_gb']} GB VRAM)")
    else:
        print(f"Device: {device}")
    print(f"Models:      {', '.join(models)}")
    print(f"Batch Sizes: {batch_sizes}")
    print(f"Precisions:  {dtypes}")
    print(f"Grad Backprop: {args.backward}")
    print("=" * 95)

    results = []

    for dt_str in dtypes:
        dt = resolve_dtype(dt_str, device)
        print(f"\n--- Precision: {dt_str} | Device: {device} ---")
        print(f"{'Model':<16s} | {'Batch':<7s} | {'Latency (ms)':<14s} | {'Throughput (wf/s)':<18s} | {'Peak VRAM':<12s} | {'VRAM/wf':<10s} | {'Status'}")
        print("-" * 95)

        for model in models:
            spec = DEFAULT_MODEL_SPECS.get(model)
            if not spec:
                print(f"Unknown model: {model}")
                continue

            for bsz in batch_sizes:
                res = measure_batch_scaling(
                    model_name=model,
                    batch_size=bsz,
                    device=device,
                    dtype=dt,
                    grid=spec["grid"],
                    params=spec["params"],
                    iterations=args.iterations,
                    warmup=args.warmup,
                    measure_backward=args.backward,
                )
                results.append(res)

                if res["status"] == "SUCCESS":
                    lat_str = f"{res['median_ms']:8.2f} ms"
                    th_str = f"{res['wf_per_sec']:12,.1f} wf/s"
                    vram_str = f"{res['peak_vram_mb']:8.1f} MB"
                    per_wf_str = f"{res['vram_per_waveform_kb']:6.2f} KB"
                    print(f"{model:<16s} | B={bsz:<5d} | {lat_str:<14s} | {th_str:<18s} | {vram_str:<12s} | {per_wf_str:<10s} | PASS")
                elif res["status"] == "OOM":
                    print(f"{model:<16s} | B={bsz:<5d} | {'--':<14s} | {'--':<18s} | {'OOM':<12s} | {'--':<10s} | OUT OF MEMORY")
                    break
                else:
                    print(f"{model:<16s} | B={bsz:<5d} | {'--':<14s} | {'--':<18s} | {'ERROR':<12s} | {'--':<10s} | {res.get('error', 'FAILED')}")

    # Seal results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    payload = {
        "benchmark_type": "ultra_large_batch_scaling",
        "provenance": provenance,
        "configuration": {
            "models": models,
            "batch_sizes": batch_sizes,
            "device": str(device),
            "dtypes": dtypes,
            "backward": args.backward,
            "iterations": args.iterations,
            "warmup": args.warmup,
        },
        "results": results,
    }

    sealed = seal_artifact(payload)
    out_path = Path(args.output) if args.output else _REPO_ROOT / "artifacts" / "benchmarks" / f"batch_scaling_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sealed, f, indent=2)

    print("\n" + "=" * 95)
    print(f"Benchmark finished! Sealed receipt saved to: {out_path.resolve()}")
    print(f"Content Seal SHA-256: {sealed['content_sha256']}")
    print("=" * 95)

    if args.plot:
        plot_dir = Path(args.plot_dir) if args.plot_dir else _REPO_ROOT / "artifacts" / "benchmarks" / "plots"
        print(f"\nGenerating publication-quality figures in: {plot_dir}")
        plot_large_batch_figures(results, plot_dir, provenance)


def plot_large_batch_figures(results: List[Dict[str, Any]], output_dir: Path, provenance: Optional[Dict[str, Any]] = None):
    """Plot Fig 5 (VRAM Scaling) and Fig 6 (Ultra-Large Batch Throughput)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    models = ["TaylorF2", "IMRPhenomD", "IMRPhenomXAS", "IMRPhenomXP", "IMRPhenomXHM"]
    model_colors = {
        "TaylorF2": "#1f77b4",
        "IMRPhenomD": "#ff7f0e",
        "IMRPhenomXAS": "#2ca02c",
        "IMRPhenomXP": "#d62728",
        "IMRPhenomXHM": "#9467bd",
    }
    model_markers = {
        "TaylorF2": "o",
        "IMRPhenomD": "s",
        "IMRPhenomXAS": "^",
        "IMRPhenomXP": "D",
        "IMRPhenomXHM": "v",
    }

    # Fig 5: VRAM Scaling (2 subplots: Peak VRAM MB & Memory per waveform KB)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for m in models:
        for dt_str, ls, alpha in [("float64", "-", 1.0), ("float32", "--", 0.85)]:
            pts = [r for r in results if r.get("model") == m and r.get("dtype") == dt_str and r.get("status") == "SUCCESS"]
            if not pts:
                continue
            pts.sort(key=lambda x: x["batch_size"])
            bs = [p["batch_size"] for p in pts]
            peak_vram = [p["peak_vram_mb"] for p in pts]
            vram_per_wf = [p["vram_per_waveform_kb"] for p in pts]
            lbl = f"{m} ({'FP64' if dt_str == 'float64' else 'FP32'})"
            c = model_colors.get(m, "#333333")
            mk = model_markers.get(m, "o")

            ax1.plot(bs, peak_vram, linestyle=ls, marker=mk, color=c, alpha=alpha, label=lbl)
            ax2.plot(bs, vram_per_wf, linestyle=ls, marker=mk, color=c, alpha=alpha, label=lbl)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Batch Size (B)")
    ax1.set_ylabel("Peak Allocated VRAM (MB)")
    ax1.set_title("Peak VRAM Scaling Across Batch Sizes")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.set_xscale("log")
    ax2.set_xlabel("Batch Size (B)")
    ax2.set_ylabel("VRAM per Waveform (KB / wf)")
    ax2.set_title("Amortized Memory Footprint per Waveform")
    ax2.grid(True, which="both", alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1), frameon=True)
    fig.suptitle("DiffGW GPU Memory Footprint & Scaling on NVIDIA RTX 5090 (32GB GDDR7)", fontsize=14, y=1.02)
    fig.tight_layout()

    f5_png = output_dir / "fig5_vram_scaling.png"
    f5_pdf = output_dir / "fig5_vram_scaling.pdf"
    fig.savefig(f5_png, bbox_inches="tight")
    fig.savefig(f5_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Generated: {f5_png}")

    # Fig 6: Ultra-Large Batch Throughput Saturation
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        for dt_str, ls, alpha in [("float64", "-", 1.0), ("float32", "--", 0.85)]:
            pts = [r for r in results if r.get("model") == m and r.get("dtype") == dt_str and r.get("status") == "SUCCESS"]
            if not pts:
                continue
            pts.sort(key=lambda x: x["batch_size"])
            bs = [p["batch_size"] for p in pts]
            tps = [p["wf_per_sec"] for p in pts]
            lbl = f"{m} ({'FP64' if dt_str == 'float64' else 'FP32'})"
            c = model_colors.get(m, "#333333")
            mk = model_markers.get(m, "o")
            ax.plot(bs, tps, linestyle=ls, marker=mk, color=c, alpha=alpha, label=lbl)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch Size (B)")
    ax.set_ylabel("Waveform Throughput (waveforms / sec)")
    ax.set_title("DiffGW Ultra-Large Batch Throughput Saturation (NVIDIA RTX 5090 32GB)", pad=12)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=True)
    fig.tight_layout()

    f6_png = output_dir / "fig6_ultra_batch_throughput.png"
    f6_pdf = output_dir / "fig6_ultra_batch_throughput.pdf"
    fig.savefig(f6_png, bbox_inches="tight")
    fig.savefig(f6_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Generated: {f6_png}")


if __name__ == "__main__":
    main()
