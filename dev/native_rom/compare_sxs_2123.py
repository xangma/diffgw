"""Frozen, development-only SEOBNRv4 (2,2) comparison with SXS:BBH:2123.

Requires decoded modes from extract_sxs_2123.py. No waveform or parameter fit
is performed; the first M omega_22=0.09 crossing fixes time and phase.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import socket

import numpy as np
from scipy.signal import savgol_filter

from . import qualify_generator as q


def load_mode(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        time = source["time_M"]
        mode = source["mode"]
    if (time.ndim != 1 or mode.shape != time.shape or len(time) < 100
            or not np.isfinite(time).all() or not np.isfinite(mode).all()
            or not np.all(np.diff(time) > 0) or max(abs(mode)) <= 0):
        raise ValueError(f"invalid waveform: {path}")
    return dict(time=time, mode=mode)


def anchor(waveform: dict, *, target: float = 0.09, dt: float = 0.25) -> dict:
    """Use a fixed frequency crossing; never minimize the pairwise residual."""
    time, mode = waveform["time"], waveform["mode"]
    phase = np.unwrap(np.angle(mode))
    peak_index = int(np.argmax(abs(mode)))
    peak_time = float(time[peak_index])
    start = max(float(time[0]), peak_time - 1000.)
    stop = min(float(time[-1]), peak_time - 100.)
    grid = np.arange(math.ceil(start / dt), math.floor(stop / dt) + 1) * dt
    if len(grid) < 81:
        raise ValueError("insufficient inspiral support for frequency anchor")
    phase_grid = np.interp(grid, time, phase)
    frequency = -savgol_filter(phase_grid, 81, 3, deriv=1, delta=dt)
    crossings = np.flatnonzero((frequency[:-1] < target) & (frequency[1:] >= target))
    if len(crossings) != 1:
        raise ValueError(f"expected one inspiral frequency crossing, got {len(crossings)}")
    index = int(crossings[0])
    fraction = (target-frequency[index])/(frequency[index+1]-frequency[index])
    anchor_time = float(grid[index]+fraction*dt)
    anchor_phase = float(np.interp(anchor_time, time, phase))
    return dict(time=time-anchor_time, mode=mode*np.exp(-1j*anchor_phase),
                anchor_time_M=anchor_time, anchor_phase_rad=anchor_phase,
                peak_after_anchor_M=peak_time-anchor_time,
                peak_amplitude=float(abs(mode[peak_index])),
                frequency_at_grid_crossing=[float(frequency[index]),
                                             float(frequency[index+1])])


def compare(candidate: dict, reference: dict, windows: dict, dt: float,
            *, support: tuple[float, float] | None = None) -> dict:
    lower = max(float(candidate["time"][0]), float(reference["time"][0]))
    upper = min(float(candidate["time"][-1]), float(reference["time"][-1]))
    if support is not None:
        lower = max(lower, support[0])
        upper = min(upper, support[1])
    grid = np.arange(math.ceil(lower/dt), math.floor(upper/dt)+1)*dt
    if len(grid) < 100:
        raise ValueError("insufficient common waveform support")
    def sampled(waveform):
        mode = waveform["mode"]
        return np.interp(grid, waveform["time"], mode.real) + 1j*np.interp(
            grid, waveform["time"], mode.imag)
    actual, expected = sampled(candidate), sampled(reference)
    reference_peak = reference["peak_after_anchor_M"]
    sections = {}
    for name, (start, end) in windows.items():
        mask = (grid-reference_peak >= start) & (grid-reference_peak < end)
        if int(mask.sum()) >= 3:
            sections[name] = dict(samples=int(mask.sum()), metrics=q.mode_metrics(
                actual[mask], expected[mask]))
        else:
            sections[name] = dict(samples=int(mask.sum()), unavailable=True)
    phase = np.unwrap(np.angle(actual*np.conj(expected)))
    return dict(common_support_M=[float(grid[0]), float(grid[-1])],
                common_samples=len(grid), metrics=q.mode_metrics(actual, expected),
                windows=sections, phase_residual_rad=dict(
                    start=float(phase[0]), end=float(phase[-1]),
                    range=float(np.ptp(phase))),
                peak_lag_difference_M=float(candidate["peak_after_anchor_M"]
                                            -reference_peak),
                peak_amplitude_ratio=float(candidate["peak_amplitude"]
                                           / reference["peak_amplitude"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["schema"] != "diffgw.seobnrv4.sxs-2123-plan.v1":
        raise ValueError("unexpected SXS comparison plan")
    extraction_path = args.decoded_dir / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    if (extraction["schema"] != "diffgw.seobnrv4.sxs-2123-extraction.v1"
            or extraction["plan_sha256"] != q.sha256(args.plan)):
        raise ValueError("SXS extraction does not match frozen plan")
    for archive_name, details in plan["catalog"]["waveforms"].items():
        if extraction["archive_files"][archive_name]["md5"] != details["md5"]:
            raise ValueError(f"SXS archive identity mismatch: {archive_name}")
    nr = {}
    for lev in (3, 4):
        record = extraction["extracted"][f"Lev{lev}"]
        path = args.decoded_dir / record["path"]
        if q.sha256(path) != record["sha256"]:
            raise ValueError(f"decoded mode checksum mismatch: Lev{lev}")
        nr[f"Lev{lev}"] = anchor(load_mode(path))
    import torch
    torch.set_num_threads(1)
    method = plan["method"]
    settings = q.Settings(max_run_seconds=600.)
    runs = {}
    for name, step in (("native", method["grid_step_M"]),
                       ("native_half_grid", method["grid_step_M"]/2)):
        runs[name] = q.native_run(plan["row"],
                                  dict(name=name, atol=method["native_atol"],
                                       rtol=method["native_rtol"], dt_geom=step), settings)
    aligned = {name: anchor(dict(time=run["time"], mode=run["mode"]))
               for name, run in runs.items()}
    windows = method["windows_relative_to_Lev4_amplitude_peak_M"]
    result = dict(schema="diffgw.seobnrv4.sxs-2123-comparison.v1",
                  plan_sha256=q.sha256(args.plan), extraction_sha256=q.sha256(extraction_path),
                  script_sha256=q.sha256(__file__),
                  generator_source_manifest_sha256=q.canonical_hash(q.source_snapshot()),
                  process=dict(host=socket.gethostname(), python=platform.python_version()),
                  row=plan["row"], numerical_choice=dict(
                      atol=method["native_atol"], rtol=method["native_rtol"],
                      grid_step_M=method["grid_step_M"]),
                  anchor={name: {key: value for key, value in item.items()
                                 if key not in ("time", "mode")}
                          for name, item in {**nr, **aligned}.items()},
                  native={name: q.summarize_run(run) for name, run in runs.items()},
                  native_grid_sensitivity=q.compare_runs(runs["native"],
                                                         runs["native_half_grid"]),
                  comparisons=dict(native_vs_Lev4=compare(aligned["native"], nr["Lev4"],
                                                          windows, method["grid_step_M"]),
                                   Lev3_vs_Lev4=compare(nr["Lev3"], nr["Lev4"],
                                                        windows, method["grid_step_M"]),
                                   Lev3_vs_Lev4_on_native_support=compare(
                                       nr["Lev3"], nr["Lev4"], windows,
                                       method["grid_step_M"],
                                       support=(float(aligned["native"]["time"][0]),
                                                float(aligned["native"]["time"][-1]))),
                                   native_half_grid_vs_Lev4=compare(
                                       aligned["native_half_grid"], nr["Lev4"], windows,
                                       method["grid_step_M"])),
                  physical_validation_passed=False, release_qualified=False,
                  limitations=["One post-publication NR configuration; no domain-wide validation.",
                               "N2 extrapolation at two resolutions does not bound all NR error.",
                               "No detector-noise-weighted faithfulness or independent holdout gate.",
                               "Native ODE tolerance convergence is not established on this row."])
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(evidence=str(args.json_out),
                          native_complex_l2=result["comparisons"]["native_vs_Lev4"]
                              ["metrics"]["complex_relative_l2"],
                          nr_complex_l2=result["comparisons"]["Lev3_vs_Lev4"]
                              ["metrics"]["complex_relative_l2"])))


if __name__ == "__main__":
    main()
