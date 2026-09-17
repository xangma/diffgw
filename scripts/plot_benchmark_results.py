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


CANONICAL_MODEL_ORDER = ["TaylorF2", "IMRPhenomD", "IMRPhenomXAS", "IMRPhenomXP", "IMRPhenomXHM"]


def sort_models(models: List[str]) -> List[str]:
    """Sort models in canonical physics progression order."""
    ordered = [m for m in CANONICAL_MODEL_ORDER if m in models]
    remaining = sorted([m for m in models if m not in CANONICAL_MODEL_ORDER])
    return ordered + remaining


def plot_throughput_vs_batch(
    benchmarks: List[Dict[str, Any]],
    output_dir: Path,
    provenance: Optional[Dict[str, Any]] = None,
):
    """Figure 1: Throughput (wf/s) vs. Batch Size for all models."""
    available_models = list(set(b["model"] for b in benchmarks if "model" in b))
    models = sort_models(available_models)
    if not models:
        return

    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 5.5), squeeze=False)
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

    legend_entries = {}
    lal_line_entry = None

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_benches = [b for b in benchmarks if b.get("model") == model]

        # Find LAL single-core baseline if present
        lal_entry = next((b for b in model_benches if b.get("framework") == "lalsimulation"), None)
        if lal_entry:
            lal_tps = lal_entry.get("wf_per_sec", 0.0)
            line = ax.axhline(lal_tps, color="#d62728", linestyle=":", linewidth=2.0, alpha=0.85)
            if lal_line_entry is None:
                lal_line_entry = (line, "LAL C Baseline (CPU single-core)")
            # Annotate LAL baseline on plot edge without occluding data
            ax.text(
                0.96, lal_tps * 1.18,
                f"LAL: {lal_tps:,.0f} wf/s",
                color="#d62728",
                fontsize=9,
                fontweight="semibold",
                ha="right",
                va="bottom",
                transform=ax.get_yaxis_transform(),
                alpha=0.95,
            )

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

            line, = ax.plot(bs, tps, mark, color=col, label=lbl)
            if lbl not in legend_entries:
                legend_entries[lbl] = line

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
        ax.set_title(f"{model}", pad=10)

    # Build unified figure-level legend positioned cleanly above subplots
    handles = []
    labels = []
    if lal_line_entry:
        handles.append(lal_line_entry[0])
        labels.append(lal_line_entry[1])
    for lbl, h in legend_entries.items():
        handles.append(h)
        labels.append(lbl)

    fig.suptitle(f"Gravitational-Wave Waveform Throughput Scaling ({gpu_name})", y=0.98, fontsize=15)
    ncols = min(len(handles), 5)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=ncols,
        frameon=True,
        framealpha=0.9,
        facecolor="#fcfcfc",
        edgecolor="#d0d0d0",
        fontsize=9.5,
    )
    fig.subplots_adjust(top=0.76, bottom=0.14, left=0.05, right=0.98, wspace=0.22)

    out_png = output_dir / "fig1_throughput_vs_batch.png"
    out_pdf = output_dir / "fig1_throughput_vs_batch.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Generated: {out_png}")


def plot_latency_per_waveform(
    benchmarks: List[Dict[str, Any]],
    output_dir: Path,
):
    """Figure 2: Per-Waveform Latency vs. Batch Size showing GPU Amortization Crossover."""
    available_models = list(set(b["model"] for b in benchmarks if "model" in b))
    models = sort_models(available_models)
    if not models:
        return

    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 5.2), squeeze=False)
    axes = axes[0]

    legend_entries = {}

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
            line, = ax.plot(bs, per_wf_ms, mark, color=col, label=lbl)
            if lbl not in legend_entries:
                legend_entries[lbl] = line

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Batch Size (B)")
        if idx == 0:
            ax.set_ylabel("Per-Waveform Latency (ms / wf)")
        ax.set_title(f"{model}: Crossover", pad=10)

    fig.suptitle("Per-Waveform Generation Latency (GPU Batch Amortization Crossover)", y=0.98, fontsize=15)
    handles = list(legend_entries.values())
    labels = list(legend_entries.keys())
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=len(labels),
        frameon=True,
        framealpha=0.9,
        facecolor="#fcfcfc",
        edgecolor="#d0d0d0",
        fontsize=10,
    )
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.05, right=0.98, wspace=0.22)

    out_png = output_dir / "fig2_latency_vs_batch.png"
    out_pdf = output_dir / "fig2_latency_vs_batch.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Generated: {out_png}")


def plot_speedup_vs_lal(
    benchmarks: List[Dict[str, Any]],
    output_dir: Path,
):
    """Figure 3: Speedup Factor relative to LALSimulation Single-Core Baseline."""
    available_models = list(set(b["model"] for b in benchmarks if "model" in b))
    models = sort_models(available_models)
    lal_baselines = {}
    for b in benchmarks:
        if b.get("framework") == "lalsimulation":
            lal_baselines[b.get("model")] = b.get("median_ms", 1.0)

    if not lal_baselines:
        print("  [Notice] No LALSimulation baselines in receipt; skipping Fig 3 (Speedup vs LAL).")
        return

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    
    model_palette = {
        "TaylorF2": "#1f77b4",
        "IMRPhenomD": "#ff7f0e",
        "IMRPhenomXAS": "#2ca02c",
        "IMRPhenomXP": "#d62728",
        "IMRPhenomXHM": "#9467bd",
    }

    for model in models:
        if model not in lal_baselines:
            continue
        lal_ms = lal_baselines[model]
        cuda_points = [
            b for b in benchmarks
            if b.get("model") == model and b.get("framework") == "torchwave" and b.get("device") == "cuda" and b.get("dtype") == "float64"
        ]
        if not cuda_points:
            continue
        cuda_points.sort(key=lambda x: x.get("batch_size", 1))
        bs = [p.get("batch_size", 1) for p in cuda_points]
        speedups = [(p.get("wf_per_sec", 1.0) / (1000.0 / lal_ms)) for p in cuda_points]
        col = model_palette.get(model, "#333333")
        ax.plot(bs, speedups, "o-", color=col, linewidth=2.2, label=f"{model} (TorchWave CUDA)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7, linewidth=1.8, label="1x Baseline (Parity with LAL)")
    ax.set_xlabel("Batch Size (B)")
    ax.set_ylabel("Speedup Factor vs LALSimulation C CPU (x)")
    ax.set_title("GPU Speedup vs. Single-Core LALSimulation Baseline", pad=12)
    # Move legend completely outside axes to the right
    ax.legend(
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.95,
        facecolor="#fcfcfc",
        edgecolor="#d0d0d0",
    )

    out_png = output_dir / "fig3_speedup_vs_lal.png"
    out_pdf = output_dir / "fig3_speedup_vs_lal.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Generated: {out_png}")


def plot_model_comparison_bar(
    benchmarks: List[Dict[str, Any]],
    output_dir: Path,
):
    """Figure 4: Cross-Model Throughput Comparison Bar Chart (B=1 vs B=100 vs B=1000)."""
    models = ["TaylorF2", "IMRPhenomD", "IMRPhenomXAS", "IMRPhenomXP", "IMRPhenomXHM"]
    active_models = [m for m in models if any(b.get("model") == m for b in benchmarks)]
    if not active_models:
        return

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    x = np.arange(len(active_models))
    width = 0.24

    b1_tps = []
    b100_tps = []
    b1000_tps = []

    cuda_present = any(b.get("device") == "cuda" for b in benchmarks)
    target_dev = "cuda" if cuda_present else "cpu"
    dev_label = target_dev.upper()

    for m in active_models:
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

    r1 = ax.bar(x - width, b1_tps, width, label=f"Batch 1 ({dev_label} FP64)", color="#a1c9f4", edgecolor="#6baed6", linewidth=0.8)
    r2 = ax.bar(x, b100_tps, width, label=f"{label_mid} ({dev_label} FP64)", color="#4878d0", edgecolor="#2171b5", linewidth=0.8)
    r3 = ax.bar(x + width, b1000_tps, width, label=f"{label_high} ({dev_label} FP64)", color="#1b458b", edgecolor="#08306b", linewidth=0.8)

    ax.set_ylabel("Throughput (waveforms / sec)")
    ax.set_title(f"TorchWave Approximant Throughput Across Workload Scales ({dev_label})", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(active_models)
    if any(v > 0 for v in b1_tps + b100_tps + b1000_tps):
        ax.set_yscale("log")
        # Ensure log scale bottom doesn't clip small values
        all_vals = [v for v in b1_tps + b100_tps + b1000_tps if v > 0]
        if all_vals:
            min_val = min(all_vals)
            max_val = max(all_vals)
            ax.set_ylim(min_val * 0.4, max_val * 4.0)

    # Move legend completely outside axes to the right
    ax.legend(
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.95,
        facecolor="#fcfcfc",
        edgecolor="#d0d0d0",
    )

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
