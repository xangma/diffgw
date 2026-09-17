#!/usr/bin/env python3
"""Publication-Quality Benchmark Plotting Suite for DiffGW.

Generates standardized figures from sealed DiffGW benchmark receipts:
1. Fig 1: Throughput vs. Batch Size (log-log curves across backends & devices)
2. Fig 2: Per-Waveform Latency vs. Batch Size (CPU vs GPU crossover)
3. Fig 3: Speedup vs. LALSimulation baseline
4. Fig 4: Cross-Model Approximant Throughput comparison
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def configure_matplotlib_style():
    """Apply clean, modern publication style settings."""
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


def load_benchmark_data(receipt_paths: Union[Path, List[Path]]) -> Dict[str, Any]:
    """Load benchmark receipts and merge them with deduplication."""
    if isinstance(receipt_paths, Path):
        receipt_paths = [receipt_paths]

    all_benchmarks_map = {}
    merged_provenance = {}

    for p in receipt_paths:
        with open(p, "r") as f:
            data = json.load(f)
        if not merged_provenance:
            merged_provenance = data.get("provenance", data.get("metadata", {}))
        for b in data.get("benchmarks", []):
            key = (
                b.get("framework"),
                b.get("model"),
                b.get("batch_size"),
                b.get("device"),
                b.get("dtype"),
            )
            all_benchmarks_map[key] = b

    return {"benchmarks": list(all_benchmarks_map.values()), "provenance": merged_provenance}


def plot_throughput_vs_batch(
    benchmarks: List[Dict[str, Any]],
    output_dir: Path,
    provenance: Optional[Dict[str, Any]] = None,
):
    """Figure 1: Throughput (wf/s) vs. Batch Size for all models."""
    models = sorted(list(set(b["model"] for b in benchmarks if "model" in b)))
    if not models:
        return

    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5), squeeze=False)
    axes = axes[0]

    colors = {
        ("torchwave", "cuda", "float64"): ("#1f77b4", "o-", "Torch CUDA FP64"),
        ("torchwave", "cuda", "float32"): ("#17becf", "s--", "Torch CUDA FP32"),
        ("torchwave", "cpu", "float64"): ("#ff7f0e", "^-", "Torch CPU FP64"),
        ("torchwave", "cpu", "float32"): ("#bcbd22", "v--", "Torch CPU FP32"),
        ("jaxwave", "cpu", "float64"): ("#2ca02c", "d-", "JAX CPU FP64"),
        ("jaxwave", "cuda", "float64"): ("#98df8a", "D-", "JAX CUDA FP64"),
        ("ripple", "cuda", "float64"): ("#9467bd", "p-", "Ripple CUDA"),
        ("ripple", "cpu", "float64"): ("#8c564b", "v-", "Ripple CPU FP64"),
    }

    gpu_devs = provenance.get("gpu", {}).get("devices", []) if provenance else []
    gpu_name = gpu_devs[0].get("name", "GPU") if gpu_devs else "Host Platform"

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_benches = [b for b in benchmarks if b.get("model") == model]

        # Find LAL single-core baseline if present
        lal_entry = next((b for b in model_benches if b.get("framework") == "lalsimulation"), None)
        if lal_entry:
            lal_tps = lal_entry.get("wf_per_sec", 0.0)
            ax.axhline(lal_tps, color="#d62728", linestyle=":", linewidth=2, label=f"LAL C Baseline ({lal_tps:.1f} wf/s)")

        # Plot each framework / device / dtype curve
        curves = {}
        for b in model_benches:
            fw = b.get("framework", "torchwave")
            if fw == "lalsimulation":
                continue
            dev = b.get("device", "cpu")
            dt = b.get("dtype", "float64")
            key = (fw, dev, dt)
            if key not in curves:
                curves[key] = []
            curves[key].append((b.get("batch_size", 1), b.get("wf_per_sec", 0.0), b.get("wf_per_sec_ci_95")))

        for key, points in curves.items():
            points.sort(key=lambda x: x[0])
            bs = [p[0] for p in points]
            tps = [p[1] for p in points]

            style = colors.get(key, ("#7f7f7f", "x-", f"{key[0]} {key[1]}"))
            col, mark, lbl = style

            ax.plot(bs, tps, mark, color=col, label=lbl)

            # Draw confidence band if available
            has_ci = all(p[2] is not None for p in points)
            if has_ci:
                ci_low = [p[2][0] for p in points]
                ci_high = [p[2][1] for p in points]
                ax.fill_between(bs, ci_low, ci_high, color=col, alpha=0.15)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Batch Size (B)")
        if idx == 0:
            ax.set_ylabel("Throughput (waveforms / sec)")
        ax.set_title(f"{model}")
        ax.legend(loc="upper left", framealpha=0.9)

    fig.suptitle(f"Gravitational-Wave Waveform Throughput Scaling ({gpu_name})", y=1.02)
    fig.tight_layout()

    out_png = output_dir / "fig1_throughput_vs_batch.png"
    out_pdf = output_dir / "fig1_throughput_vs_batch.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"  -> Generated: {out_png}")


def plot_latency_per_waveform(
    benchmarks: List[Dict[str, Any]],
    output_dir: Path,
):
    """Figure 2: Per-Waveform Latency vs. Batch Size showing GPU Amortization Crossover."""
    models = sorted(list(set(b["model"] for b in benchmarks if "model" in b)))
    if not models:
        return

    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5), squeeze=False)
    axes = axes[0]

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_benches = [b for b in benchmarks if b.get("model") == model]

        for dev, col, mark, lbl in [("cuda", "#1f77b4", "o-", "CUDA (FP64)"), ("cpu", "#ff7f0e", "^-", "CPU (FP64)")]:
            points = [b for b in model_benches if b.get("device") == dev and b.get("dtype") == "float64" and b.get("framework") == "torchwave"]
            if not points:
                continue
            points.sort(key=lambda x: x.get("batch_size", 1))
            bs = [p.get("batch_size", 1) for p in points]
            # per-waveform latency in ms
            per_wf_ms = [p.get("median_ms", 1.0) / p.get("batch_size", 1) for p in points]
            ax.plot(bs, per_wf_ms, mark, color=col, label=lbl)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Batch Size (B)")
        if idx == 0:
            ax.set_ylabel("Per-Waveform Latency (ms / wf)")
        ax.set_title(f"{model}: Crossover")
        ax.legend(loc="upper right")

    fig.suptitle("Per-Waveform Generation Latency (GPU Batch Amortization Crossover)", y=1.02)
    fig.tight_layout()

    out_png = output_dir / "fig2_latency_vs_batch.png"
    out_pdf = output_dir / "fig2_latency_vs_batch.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"  -> Generated: {out_png}")


def plot_speedup_vs_lal(
    benchmarks: List[Dict[str, Any]],
    output_dir: Path,
):
    """Figure 3: Speedup Factor relative to LALSimulation Single-Core Baseline."""
    models = sorted(list(set(b["model"] for b in benchmarks if "model" in b)))
    lal_baselines = {}
    for b in benchmarks:
        if b.get("framework") == "lalsimulation":
            lal_baselines[b.get("model")] = b.get("median_ms", 1.0)

    if not lal_baselines:
        print("  [Notice] No LALSimulation baselines in receipt; skipping Fig 3 (Speedup vs LAL).")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, lal_ms in lal_baselines.items():
        cuda_points = [
            b for b in benchmarks
            if b.get("model") == model and b.get("framework") == "torchwave" and b.get("device") == "cuda" and b.get("dtype") == "float64"
        ]
        if not cuda_points:
            continue
        cuda_points.sort(key=lambda x: x.get("batch_size", 1))
        bs = [p.get("batch_size", 1) for p in cuda_points]
        # Speedup vs LAL single-waveform = (B * lal_ms) / torch_batch_ms = torch_wf_per_sec / lal_wf_per_sec
        speedups = [(p.get("wf_per_sec", 1.0) / (1000.0 / lal_ms)) for p in cuda_points]
        ax.plot(bs, speedups, "o-", label=f"{model} (CUDA vs LAL)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="1x Baseline (Parity with LAL)")
    ax.set_xlabel("Batch Size (B)")
    ax.set_ylabel("Speedup Factor vs LALSimulation C CPU (x)")
    ax.set_title("TorchWave GPU Speedup vs. Single-Core LALSimulation Baseline")
    ax.legend(loc="upper left")

    fig.tight_layout()
    out_png = output_dir / "fig3_speedup_vs_lal.png"
    out_pdf = output_dir / "fig3_speedup_vs_lal.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"  -> Generated: {out_png}")


def plot_model_comparison_bar(
    benchmarks: List[Dict[str, Any]],
    output_dir: Path,
):
    """Figure 4: Cross-Model Throughput Comparison Bar Chart (B=1 vs B=1000)."""
    models = ["TaylorF2", "IMRPhenomD", "IMRPhenomXAS", "IMRPhenomXP"]
    active_models = [m for m in models if any(b.get("model") == m for b in benchmarks)]
    if not active_models:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(active_models))
    width = 0.25

    b1_tps = []
    b100_tps = []
    b1000_tps = []

    for m in active_models:
        # Fallback to CPU if CUDA is not available
        cuda_present = any(b.get("device") == "cuda" for b in benchmarks)
        target_dev = "cuda" if cuda_present else "cpu"
        dev_label = target_dev.upper()

        def get_tps(bsz):
            matches = [
                b.get("wf_per_sec", 0.0) for b in benchmarks
                if b.get("model") == m and b.get("device") == target_dev and b.get("dtype") == "float64" and b.get("batch_size") == bsz
            ]
            return matches[0] if matches else 0.0

        b1_tps.append(get_tps(1))
        b100_tps.append(get_tps(100 if cuda_present else 5))
        b1000_tps.append(get_tps(1000 if cuda_present else 10))

    label_mid = "Batch 100" if cuda_present else "Batch 5"
    label_high = "Batch 1,000" if cuda_present else "Batch 10"

    r1 = ax.bar(x - width, b1_tps, width, label=f"Batch 1 ({dev_label} FP64)", color="#a1c9f4")
    r2 = ax.bar(x, b100_tps, width, label=f"{label_mid} ({dev_label} FP64)", color="#4878d0")
    r3 = ax.bar(x + width, b1000_tps, width, label=f"{label_high} ({dev_label} FP64)", color="#1b458b")

    ax.set_ylabel("Throughput (waveforms / sec)")
    ax.set_title(f"TorchWave Approximant Throughput Across Workload Scales ({dev_label})")
    ax.set_xticks(x)
    ax.set_xticklabels(active_models)
    if any(v > 0 for v in b1_tps + b100_tps + b1000_tps):
        ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    out_png = output_dir / "fig4_model_breakdown_bar.png"
    out_pdf = output_dir / "fig4_model_breakdown_bar.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"  -> Generated: {out_png}")


def plot_all_figures(receipt_paths: Union[Path, List[Path]], output_dir: Path):
    """Load benchmark receipt(s) and render all four publication-grade figures."""
    configure_matplotlib_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_benchmark_data(receipt_paths)
    benchmarks = data.get("benchmarks", [])
    provenance = data.get("provenance", data.get("metadata", {}))

    print(f"Rendering benchmark figures to: {output_dir.resolve()}")
    plot_throughput_vs_batch(benchmarks, output_dir, provenance)
    plot_latency_per_waveform(benchmarks, output_dir)
    plot_speedup_vs_lal(benchmarks, output_dir)
    plot_model_comparison_bar(benchmarks, output_dir)
    print("All figures successfully rendered!")


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures from DiffGW benchmark receipts.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path(s) to benchmark JSON receipt(s), comma-separated.")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Directory to save output figures.")
    args = parser.parse_args()

    raw_inputs = [p.strip() for p in args.input.split(",") if p.strip()]
    input_paths = [Path(p) for p in raw_inputs]
    for p in input_paths:
        if not p.exists():
            print(f"Error: Input file not found: {p}")
            sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else input_paths[0].parent / "plots"
    plot_all_figures(input_paths, out_dir)


if __name__ == "__main__":
    main()
