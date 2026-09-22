#!/usr/bin/env python3
"""Render the published package-comparison receipts without running benchmarks.

Requires matplotlib. The default output is the maintained SVG documentation
figures; use --format png or pdf to export another format.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

ROOT = Path(__file__).resolve().parents[1]
STYLES = {
    "lal": ("LALSimulation", "#4b5563", "D"),
    "torch": ("DiffGW Torch eager", "#008a91", "o"),
    "torch_compiled": ("DiffGW Torch compiled", "#1764b0", "^"),
    "jax": ("DiffGW JAX JIT", "#7855ab", "s"),
    "ripple": ("Ripple JAX JIT", "#c56812", "v"),
}


def load_receipt(path: Path, device: str) -> dict:
    """Reject altered, incomplete or inconsistent evidence instead of hiding it."""
    data = json.loads(path.read_text())
    content_hash = data.pop("content_sha256")
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(payload.encode()).hexdigest() != content_hash:
        raise ValueError(f"Receipt content hash does not match: {path}")
    if data["schema"] != "diffgw-package-comparison/v1" or data["configuration"]["device"] != device:
        raise ValueError(f"Unexpected receipt schema/device: {path}")
    config, gate = data["configuration"], data["qualification"]
    expected = set(itertools.product(config["models"], config["backends"], config["batches"]))
    seen = set()
    for row in data["results"]:
        key = (row["model"], row["backend"], row["batch"])
        if key in seen or row["backend"] not in STYLES:
            raise ValueError(f"Duplicate or unsupported series: {key}")
        seen.add(key)
        if row["device"] != device or row["dtype"] != "float64" or row.get("output_dtype") != "complex128":
            raise ValueError(f"Expected {device} FP64 result: {key}")
        accuracy = row["accuracy"]
        qualified = (
            accuracy["max_relative_l2"] <= gate["max_relative_l2"]
            and accuracy["max_mismatch"] <= gate["max_mismatch"]
        )
        if accuracy["qualified"] is not qualified or row["status"] != ("qualified" if qualified else "unqualified"):
            raise ValueError(f"Inconsistent accuracy qualification: {key}")
        if "timing" not in row:
            raise ValueError(f"No measured timing for {key}; cannot render complete comparison")
        timing = row["timing"]
        low, high = timing["median_ci95_ms"]
        values = [*timing["raw_ms"], low, timing["median_ms"], high]
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError(f"Nonpositive or nonfinite timing: {key}")
        if not low <= timing["median_ms"] <= high or median(timing["raw_ms"]) != timing["median_ms"]:
            raise ValueError(f"Inconsistent timing summary: {key}")
    if seen != expected:
        raise ValueError(f"Receipt does not contain its declared configurations: {path}")
    return data


def check_matched_campaign(cpu: dict, gpu: dict) -> None:
    for key in ("models", "batches", "iterations"):
        if cpu["configuration"][key] != gpu["configuration"][key]:
            raise ValueError(f"CPU/GPU configurations differ in {key}")
    for key in ("source_commit", "frequency_grid", "fixed_parameters", "qualification", "timing_boundary"):
        if cpu[key] != gpu[key]:
            raise ValueError(f"CPU/GPU campaigns differ in {key}")
    for key in ("host", "cpu", "gpu", "packages", "cpu_affinity", "source_sha256", "script_sha256"):
        if cpu["provenance"][key] != gpu["provenance"][key]:
            raise ValueError(f"CPU/GPU provenance differs in {key}")
    parameters = {}
    for receipt in (cpu, gpu):
        for row in receipt["results"]:
            key = (row["model"], row["batch"])
            if parameters.setdefault(key, row["parameters"]) != row["parameters"]:
                raise ValueError(f"Parameter populations differ for {key}")


def hardware(receipt: dict) -> str:
    provenance = receipt["provenance"]
    if receipt["configuration"]["device"] == "cpu":
        return provenance["cpu"].replace(" 16-Core Processor", "") + " · one logical CPU"
    return provenance["gpu"].split(",")[0] + " · CUDA"


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
        "axes.labelsize": 11, "axes.edgecolor": "#c4cbd4", "text.color": "#253247",
        "axes.labelcolor": "#253247", "xtick.color": "#475569", "ytick.color": "#475569",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "svg.fonttype": "none", "svg.hashsalt": "diffgw-package-benchmarks",
        "pdf.fonttype": 42,
    })


def footnotes(fig, receipt: dict) -> None:
    gate = receipt["qualification"]
    provenance = receipt["provenance"]
    packages = provenance["packages"]
    fig.text(0.04, 0.080, "Filled: passes LAL agreement gate.  Hollow / †: fails gate; diagnostic timing.", fontsize=10)
    fig.text(0.04, 0.049,
             f"Gate: relative L2 ≤ {gate['max_relative_l2']:.0e}, mismatch ≤ {gate['max_mismatch']:.0e}. "
             "Error bars: 95% bootstrap intervals within this run.", fontsize=9)
    fig.text(0.04, 0.018,
             f"{provenance['timestamp'][:10]} · DiffGW {receipt['source_commit'][:7]} · "
             f"Ripple {packages['rippleGW']} · LALSuite {packages['lalsuite']} · "
             "Resident inputs; setup, transfers and derivatives excluded.", fontsize=8.5)


def save(fig, output: Path, name: str, fmt: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}.{fmt}"
    metadata = {"Date": None} if fmt == "svg" else {}
    fig.savefig(path, format=fmt, dpi=180, metadata=metadata)
    if fmt == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(path)


def plot_scaling(receipt: dict, output: Path, fmt: str) -> None:
    config = receipt["configuration"]
    models = config["models"]
    fig, axes = plt.subplots(1, len(models), figsize=(12, 5.5), squeeze=False, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.69, bottom=0.24, wspace=0.15)
    fig.suptitle("Warmed waveform generation · " + hardware(receipt), y=0.96, fontsize=15, weight="bold")
    fig.text(0.5, 0.88,
             f"FP64 · {receipt['frequency_grid']['bins']:,} frequencies · both polarizations · "
             f"median of {config['iterations']} calls · lower is faster", ha="center", fontsize=11)
    handles = [Line2D([], [], color=color, marker=marker, label=label, linewidth=1.6)
               for backend, (label, color, marker) in STYLES.items() if backend in config["backends"]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.84), ncol=3, frameon=False, fontsize=10)
    for ax, model in zip(axes[0], models):
        for backend, (_, color, marker) in STYLES.items():
            rows = sorted((r for r in receipt["results"] if r["model"] == model and r["backend"] == backend),
                          key=lambda r: r["batch"])
            for left, right in zip(rows, rows[1:]):
                qualified = left["accuracy"]["qualified"] and right["accuracy"]["qualified"]
                ax.plot([left["batch"], right["batch"]], [left["timing"]["median_ms"], right["timing"]["median_ms"]],
                        color=color, linestyle="-" if qualified else "--", linewidth=1.5, alpha=0.8)
            for row in rows:
                timing = row["timing"]
                mid = timing["median_ms"]
                low, high = timing["median_ci95_ms"]
                ax.errorbar(row["batch"], mid, yerr=[[mid - low], [high - mid]], fmt=marker,
                            color=color, markerfacecolor=color if row["accuracy"]["qualified"] else "white",
                            markeredgewidth=1.3, markersize=6.5, capsize=3, linewidth=1.2, zorder=3)
        ax.set(xscale="log", yscale="log", title=model, xlabel="Binaries per batch",
               xlim=(min(config["batches"]) * 0.7, max(config["batches"]) * 1.45))
        ax.set_xticks(config["batches"], [f"{b:,}" for b in config["batches"]])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(which="major", color="#e5e9ef", linewidth=0.7)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[0][0].set_ylabel("Milliseconds per complete batch (log scale)")
    footnotes(fig, receipt)
    suffix = "cpu" if config["device"] == "cpu" else "gpu"
    save(fig, output, f"package-latency-{suffix}", fmt)


def plot_summary(cpu: dict, gpu: dict, output: Path, fmt: str) -> None:
    """Show all implementations for D/XAS, without selecting the fastest backend."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    fig.subplots_adjust(left=0.235, right=0.96, top=0.82, bottom=0.21, hspace=0.63, wspace=0.22)
    fig.suptitle("Package comparison · 1,000 binaries per call", y=0.96, fontsize=17, weight="bold")
    fig.text(0.5, 0.913, f"FP64 · {cpu['frequency_grid']['bins']:,} frequencies · both polarizations · "
             "warmed median and 95% interval", ha="center", fontsize=11)
    for row_index, receipt in enumerate((cpu, gpu)):
        backends = [b for b in STYLES if b in receipt["configuration"]["backends"]]
        intervals = [r["timing"]["median_ci95_ms"] for r in receipt["results"]
                     if r["model"] in ("IMRPhenomD", "IMRPhenomXAS") and r["batch"] == 1000]
        limits = (min(low for low, _ in intervals) * 0.7, max(high for _, high in intervals) * 2.1)
        row_top = axes[row_index][0].get_position().y1
        fig.text(0.235, row_top + 0.050, hardware(receipt), fontsize=11, weight="bold")
        for column, model in enumerate(("IMRPhenomD", "IMRPhenomXAS")):
            ax = axes[row_index][column]
            for position, backend in enumerate(backends):
                matches = [r for r in receipt["results"]
                           if (r["model"], r["backend"], r["batch"]) == (model, backend, 1000)]
                if len(matches) != 1:
                    raise ValueError(f"Summary requires {model}/{backend} at batch 1,000")
                result = matches[0]
                timing = result["timing"]
                mid = timing["median_ms"]
                low, high = timing["median_ci95_ms"]
                _, color, marker = STYLES[backend]
                qualified = result["accuracy"]["qualified"]
                ax.errorbar(mid, position, xerr=[[mid - low], [high - mid]], fmt=marker, color=color,
                            markerfacecolor=color if qualified else "white", markeredgewidth=1.5,
                            markersize=8, capsize=4, linewidth=1.5, zorder=3)
                ax.annotate(f"{mid:.3g}" + ("" if qualified else "†"), (high, position),
                            xytext=(8, 0), textcoords="offset points", va="center", fontsize=11)
            ax.set(xscale="log", title=model, xlabel="Complete batch latency (ms, log scale)",
                   xlim=limits, ylim=(len(backends) - 0.5, -0.6))
            ax.set_yticks(range(len(backends)), [STYLES[b][0] for b in backends] if column == 0 else [""] * len(backends))
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(axis="x", color="#e5e9ef", linewidth=0.8)
            ax.tick_params(axis="y", length=0)
    footnotes(fig, cpu)
    save(fig, output, "package-latency-summary", fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, default=ROOT / "docs/benchmarks/mac-20260921-cpu.json")
    parser.add_argument("--gpu", type=Path, default=ROOT / "docs/benchmarks/mac-20260921-gpu.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/benchmarks/plots")
    parser.add_argument("--format", choices=("svg", "png", "pdf"), default="svg")
    args = parser.parse_args()
    cpu, gpu = load_receipt(args.cpu, "cpu"), load_receipt(args.gpu, "cuda")
    check_matched_campaign(cpu, gpu)
    configure_style()
    plot_summary(cpu, gpu, args.output_dir, args.format)
    for receipt in (cpu, gpu):
        plot_scaling(receipt, args.output_dir, args.format)


if __name__ == "__main__":
    main()
