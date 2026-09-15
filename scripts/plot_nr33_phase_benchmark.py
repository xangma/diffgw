#!/usr/bin/env python3
"""Plot the frozen NR--LAL phase benchmark report.

This is a read-only report renderer.  It verifies every saved result-array
hash before plotting and never recalculates or reruns the benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MASSES = (20.0, 60.0, 120.0)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load(report_path: Path):
    report = json.loads(report_path.read_text())
    cases = report.get("cases", [])
    if len(cases) != 7:
        raise ValueError(f"expected seven benchmark cases, found {len(cases)}")
    loaded = []
    for case in cases:
        ref = case["arrays"]
        path = Path(ref["path"])
        actual = _sha256(path)
        if actual.lower() != ref["sha256"].lower():
            raise ValueError(f"saved array hash changed: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            required = {"primary", "lal_M60", "frequency_Mf", "phase_mask"}
            missing = required.difference(arrays.files)
            if missing:
                raise ValueError(f"{path} lacks {sorted(missing)}")
            data = {key: np.asarray(arrays[key]).copy() for key in required}
        n = data["frequency_Mf"].size
        if any(data[key].size != n for key in ("primary", "lal_M60", "phase_mask")):
            raise ValueError(f"inconsistent saved array lengths in {path}")
        if not (np.all(np.isfinite(data["frequency_Mf"])) and
                np.all(np.isfinite(data["primary"])) and
                np.all(np.isfinite(data["lal_M60"]))):
            raise ValueError(f"non-finite saved array in {path}")
        loaded.append((case, data))
    return report, loaded


def _affine_residual(case, data):
    """Return target-model phase after the report's frozen affine correction."""
    f = data["frequency_Mf"]
    target = data["primary"]
    model = data["lal_M60"]
    metrics = next(x for x in case["lal_comparisons"]
                   if float(x["total_mass_msun"]) == 60.0)["metrics"]
    phase = np.unwrap(np.angle(target * np.conj(model)))
    # ``affine_phase`` is the fitted intercept; ``constant_phase`` is the
    # separate constant-only alignment reported alongside it.
    residual = phase - (float(metrics["affine_phase"]) +
                        2.0 * np.pi * f * float(metrics["time_shift"]))
    mask = data["phase_mask"].astype(bool)
    return residual, mask


def _style(ax):
    ax.grid(True, alpha=.22, linewidth=.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure(report_path: Path, output: Path) -> None:
    report, loaded = _load(report_path)
    fig = plt.figure(figsize=(18, 14))
    grid = fig.add_gridspec(3, 4, height_ratios=(1.0, 1.0, 1.18),
                            hspace=.32, wspace=.25)
    colors = plt.get_cmap("viridis")(np.linspace(.08, .92, len(loaded)))

    # Seven panels keep every frozen q visible while sharing the same units.
    phase_positions = [(0, col) for col in range(4)] + [(1, col) for col in range(3)]
    phase_axes = [fig.add_subplot(grid[row, col]) for row, col in phase_positions]
    core_ax = fig.add_subplot(grid[1, 3])
    all_residuals = []
    for case, data in loaded:
        residual, mask = _affine_residual(case, data)
        valid = mask & np.isfinite(residual)
        all_residuals.append(residual[valid])
    scale = max(float(np.max(np.abs(x))) for x in all_residuals)
    phase_limit = max(.005, 1.08 * scale)
    for ax, (case, data), color in zip(phase_axes, loaded, colors):
        f = data["frequency_Mf"]
        residual, mask = _affine_residual(case, data)
        valid = mask & np.isfinite(residual)
        plotted = np.where(valid, residual, np.nan)
        ax.plot(f, plotted, color=color, linewidth=1.05,
                label="NR − LAL after affine fit")
        if np.any(~mask):
            ax.fill_between(f, -phase_limit, phase_limit, where=~mask,
                            color="0.82", alpha=.38, step="mid",
                            label="outside phase mask")
        # Amplitude context uses a second scale so weak-tail support is visible.
        amp = np.abs(data["primary"])
        amp /= np.max(amp)
        twin = ax.twinx()
        twin.plot(f, amp, color="0.25", alpha=.32, linewidth=.65,
                  label="normalized |NR|")
        twin.set_ylim(0, 1.08)
        twin.set_yticks([0, 1])
        twin.set_yticklabels(["0", "1"], color="0.35", fontsize=7)
        twin.spines["top"].set_visible(False)
        twin.spines["right"].set_alpha(.25)
        ax.set_xlim(*case["phase_band_Mf"])
        ax.set_ylim(-phase_limit, phase_limit)
        q = case["target_q"]
        status = "PASS" if case["phase_quality_pass"] else "FAIL"
        status_color = "#176b3a" if case["phase_quality_pass"] else "#b2182b"
        ax.set_title(f"q = {q:g}   {status}", color=status_color,
                     fontsize=10, fontweight="bold")
        ax.text(.03, .08, f"mask {int(mask.sum())}/{len(mask)} bins",
                transform=ax.transAxes, fontsize=7.5, color="0.28")
        _style(ax)
        if ax in phase_axes[-4:]:
            ax.set_xlabel(r"frequency $Mf$")
        if ax in (phase_axes[0], phase_axes[4]):
            ax.set_ylabel("affine phase residual [rad]")

    # The eighth panel makes the amplitude-weighted core legible while the
    # seven preceding panels retain the full masked band and its tail.
    core_scale = 0.0
    for case_i, (case, data) in enumerate(loaded):
        residual, mask = _affine_residual(case, data)
        amp = np.abs(data["primary"])
        core = mask & (amp >= .5 * np.max(amp))
        core_scale = max(core_scale, float(np.max(np.abs(residual[core]))))
        core_ax.plot(data["frequency_Mf"], np.where(core, residual, np.nan),
                     color=colors[case_i], linewidth=1.0,
                     label=f'q={case["target_q"]:g}')
    core_ax.set_ylim(-max(.006, 1.12 * core_scale), max(.006, 1.12 * core_scale))
    core_ax.set_title(r"Amplitude-weighted core ($|NR|\geq0.5\,max$)", fontsize=10,
                      fontweight="bold")
    core_ax.set_xlabel(r"frequency $Mf$")
    core_ax.set_ylabel("affine residual [rad]")
    core_ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="upper left")
    _style(core_ax)

    # Summary panels use the report's scalar metrics and show all three masses.
    l2_ax = fig.add_subplot(grid[2, :2])
    mm_ax = fig.add_subplot(grid[2, 2:])
    x = np.arange(len(loaded), dtype=float)
    mass_colors = {20.0: "#1b9e77", 60.0: "#d95f02", 120.0: "#7570b3"}
    styles = {"raw_relative_l2": ("raw", ":"),
              "constant_relative_l2": ("constant phase", "--"),
              "affine_relative_l2": ("affine phase", "-")}
    # Complex L2 is mass-independent in the frozen unweighted comparison;
    # show its three alignment variants once at the requested 60 Msun.
    values = []
    for case, _ in loaded:
        entry = next(v for v in case["lal_comparisons"]
                     if float(v["total_mass_msun"]) == 60.0)
        values.append(entry["metrics"])
    for key, (label, linestyle) in styles.items():
        l2_ax.plot(x, [float(v[key]) for v in values], linestyle=linestyle,
                   marker="o", markersize=3.5, linewidth=1.25,
                   color="#d95f02", label=f"{label}, 60 $M_\\odot$")

    for mass in MASSES:
        mismatches = []
        for case, _ in loaded:
            entry = next(v for v in case["lal_comparisons"]
                         if float(v["total_mass_msun"]) == mass)
            mismatches.append(float(entry["detector_optimized"]["mismatch"]))
        mm_ax.plot(x, mismatches, marker="o", markersize=4, linewidth=1.35,
                   color=mass_colors[mass], label=f"{int(mass)} $M_\u2609$")
    for case_i, (case, _) in enumerate(loaded):
        if not case["phase_quality_pass"]:
            l2_ax.plot(x[case_i], np.nan, marker="X", color="#b2182b",
                       markersize=9, label="quality failure" if case_i == 0 else None)
            mm_ax.plot(x[case_i], np.nan, marker="X", color="#b2182b",
                       markersize=9)
    for ax in (l2_ax, mm_ax):
        ax.set_xticks(x)
        ax.set_xticklabels([f'{c["target_q"]:g}' for c, _ in loaded])
        ax.set_xlabel("target mass ratio q")
        ax.set_yscale("log")
        _style(ax)
    l2_ax.set_title("Complex L2 summary", fontweight="bold")
    l2_ax.set_ylabel("relative complex L2")
    mm_ax.set_title("Detector-weighted optimized mismatch summary", fontweight="bold")
    mm_ax.set_ylabel("1 − optimized overlap")
    l2_ax.legend(fontsize=8, ncol=1, loc="upper left", frameon=False)
    mm_ax.legend(fontsize=8, ncol=3, loc="upper left", frameon=False)

    fig.suptitle("NR33 phase benchmark: frozen ringdown-only single-mode comparison",
                 fontsize=16, fontweight="bold", y=.988)
    fig.text(.5, .958,
             "NR reference vs public LAL PhenomXHM (release 122022), 60 Msun phase curves; "
             "curves end at the phase mask; gray line = normalized NR amplitude",
             ha="center", va="top", fontsize=9.5, color="0.25")
    fig.text(.5, .018,
             "Affine phase fit is NR-amplitude-squared weighted on the preregistered mask. "
             "Summary metrics are read from the frozen report for 20, 60, and 120 Msun.",
             ha="center", fontsize=9, color="0.28")
    fig.subplots_adjust(top=.88, bottom=.09, left=.05, right=.98,
                        hspace=.42, wspace=.30)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output.with_suffix('.png')}")
    print(f"wrote {output.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
                        default=Path("docs/reviews/2026-09-15-independent-phase-benchmark/results/report.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/reviews/2026-09-15-independent-phase-benchmark/figures/nr33-phase-benchmark"))
    args = parser.parse_args()
    make_figure(args.report, args.output)


if __name__ == "__main__":
    main()
