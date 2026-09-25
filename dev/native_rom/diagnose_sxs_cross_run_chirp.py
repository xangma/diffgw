"""Post-hoc fixed-frequency phasing control for two failed antialigned SXS rows."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from . import qualify_generator as q


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/sxs_cross_run_chirp_plan.json"
OUTPUT = q.ROOT / "dev/native_rom/evidence/2026-09-25/sxs_cross_run_chirp.json"


def _crossings(wave: dict, plan: dict, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return time and carrier phase at unique pre-peak frequency crossings."""
    dt = plan["sample_step_M"]
    time = wave["time"]
    grid = np.arange(math.ceil(time[0] / dt), math.floor(time[-1] / dt) + 1) * dt
    phase = np.interp(grid, time, np.unwrap(-np.angle(wave["mode"])))
    points = round(width / dt) + 1
    if points % 2 != 1 or not math.isclose((points - 1) * dt, width):
        raise ValueError("smoothing width must give an odd sample count")
    omega = savgol_filter(phase, points, plan["polynomial_degree"],
                         deriv=1, delta=dt)
    start, end = plan["crossing_time_window_M"]
    times, phases = [], []
    for frequency in plan["mode_angular_frequencies_M_inv"]:
        hits = np.flatnonzero((omega[:-1] < frequency)
                              & (omega[1:] >= frequency)
                              & (grid[:-1] >= start) & (grid[:-1] < end))
        if len(hits) != 1:
            raise ValueError(f"expected unique crossing of {frequency}: {len(hits)}")
        index = int(hits[0])
        fraction = (frequency - omega[index]) / (omega[index + 1] - omega[index])
        times.append(float(grid[index] + fraction * dt))
        phases.append(float(phase[index] + fraction * (phase[index + 1] - phase[index])))
    return np.array(times), np.array(phases)


def _residual(candidate: tuple, reference: tuple, anchor_index: int) -> dict:
    candidate_time, candidate_phase = candidate
    reference_time, reference_phase = reference
    time = ((candidate_time - candidate_time[anchor_index])
            - (reference_time - reference_time[anchor_index]))
    phase = ((candidate_phase - candidate_phase[anchor_index])
             - (reference_phase - reference_phase[anchor_index]))
    return dict(time_difference_M=time.tolist(),
                phase_difference_from_anchor_rad=phase.tolist())


def main() -> None:
    if OUTPUT.exists():
        raise ValueError("output already exists")
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.sxs-cross-run-chirp-plan.v1"
            or plan["release_qualified"] is not False
            or plan["posthoc"] is not True
            or plan["sample_step_M"] != 0.25
            or plan["polynomial_degree"] != 3
            or plan["smoothing_widths_M"] != [10, 20, 40]
            or plan["mode_angular_frequencies_M_inv"]
            != [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12]
            or plan["phase_anchor_mode_frequency_M_inv"] != 0.09
            or plan["crossing_time_window_M"] != [-2000, 100]
            or plan["cases"] != {"third": "SXS:BBH:2569", "paper": "SXS:BBH:0260"}):
        raise ValueError("frozen fixed-frequency protocol changed")
    prior_path = Path(plan["prior_cross_run"]["path"])
    if q.sha256(prior_path) != plan["prior_cross_run"]["sha256"]:
        raise ValueError("prior cross-run comparison changed")
    prior = json.loads(prior_path.read_text())
    if (prior["status"] != "diagnostic_complete"
            or prior["release_qualified"] is not False):
        raise ValueError("prior failed-gate control unavailable")
    waves = {}
    for label, simulation in plan["cases"].items():
        entry = plan["arrays"][label]
        path = Path(entry["path"])
        if q.sha256(path) != entry["sha256"]:
            raise ValueError(f"frozen {simulation} array changed")
        with np.load(path, allow_pickle=False) as arrays:
            for name in ("native", "nr_low", "nr_high"):
                waves[label + "_" + name] = dict(
                    time=arrays[name + "_time"], mode=arrays[name + "_mode"])
    anchor_index = plan["mode_angular_frequencies_M_inv"].index(
        plan["phase_anchor_mode_frequency_M_inv"])
    result = dict(schema="diffgw.seobnrv4.sxs-cross-run-chirp.v1",
                  status="running", posthoc=True, release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  mode_angular_frequencies_M_inv=plan["mode_angular_frequencies_M_inv"],
                  phase_anchor_mode_frequency_M_inv=plan[
                      "phase_anchor_mode_frequency_M_inv"],
                  cases=plan["cases"], smoothing={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for width in plan["smoothing_widths_M"]:
        crossed = {name: _crossings(wave, plan, width) for name, wave in waves.items()}
        entries = {}
        for label in plan["cases"]:
            high = crossed[label + "_nr_high"]
            entries[label] = dict(
                native_vs_high_nr=_residual(crossed[label + "_native"], high, anchor_index),
                low_vs_high_nr=_residual(crossed[label + "_nr_low"], high, anchor_index))
        entries["cross_run"] = dict(
            high_nr=_residual(crossed["third_nr_high"], crossed["paper_nr_high"],
                              anchor_index),
            native=_residual(crossed["third_native"], crossed["paper_native"],
                             anchor_index))
        result["smoothing"][str(width)] = entries
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    for label in plan["cases"]:
        row = result["smoothing"]["20"][label]
        print(label, row["native_vs_high_nr"]["time_difference_M"][0],
              row["native_vs_high_nr"]["phase_difference_from_anchor_rad"][0],
              flush=True)


if __name__ == "__main__":
    main()
