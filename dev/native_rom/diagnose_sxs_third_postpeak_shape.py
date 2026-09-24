"""Post-hoc peak-relative native/public shape diagnostic for two frozen SXS rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import compare_sxs_2123 as compare
from . import compare_sxs_2123_public_control as public
from . import qualify_generator as q


def sample(wave: dict, times: np.ndarray) -> np.ndarray:
    return (np.interp(times, wave["time"], wave["mode"].real)
            + 1j * np.interp(times, wave["time"], wave["mode"].imag))


def peak_relative(wave: dict, offsets: np.ndarray) -> np.ndarray:
    peak_time = wave["peak_after_anchor_M"]
    peak_mode = sample(wave, np.array([peak_time]))[0]
    return sample(wave, peak_time + offsets) / peak_mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--case-plan", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--parity-plan", type=Path, required=True)
    parser.add_argument("--parity-result", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.sxs-third-postpeak-shape-plan.v1"
            or plan["qualification_claim"] is not False
            or any(q.sha256(getattr(args, name)) != plan[f"{name.replace('-', '_')}_sha256"]
                   for name in ("case_plan", "comparison", "parity_plan", "parity_result"))):
        raise ValueError("frozen postpeak input changed")
    case_plan = json.loads(args.case_plan.read_text())
    comparison = json.loads(args.comparison.read_text())
    parity_plan = json.loads(args.parity_plan.read_text())
    parity = json.loads(args.parity_result.read_text())
    if set(plan["simulations"]) - set(case_plan["cases"]):
        raise ValueError("unexpected SXS row")
    if parity["plan_sha256"] != q.sha256(args.parity_plan):
        raise ValueError("public parity plan changed")
    identity = q.lal_binary_identity(parity_plan["binary_versions"]["lal"],
                                     parity_plan["binary_versions"]["lalsimulation"])
    from .diagnose_q4_q8_qnm_parity import _public_qnm
    from torchwave.approximants.kerr_qnm_table_convention import (
        signed_qnm_220_table_convention,
    )
    import torch
    method = plan["method"]
    dt = method["grid_step_M"]
    start, stop = method["peak_relative_interval_M"]
    offsets = np.arange(math.ceil(start / dt), math.floor(stop / dt) + 1) * dt
    cases = {}
    for simulation in plan["simulations"]:
        entry = comparison["cases"][simulation]
        path = Path(entry["anchored_array"]["path"])
        if q.sha256(path) != entry["anchored_array"]["sha256"]:
            raise ValueError(f"{simulation} native array changed")
        with np.load(path, allow_pickle=False) as arrays:
            native = dict(time=arrays["native_time"], mode=arrays["native_mode"],
                          **entry["anchor"]["native"])
        public_wave = compare.anchor(public.public_mode(
            case_plan["cases"][simulation]["row"], dt,
            parity_plan["public_call"]), dt=dt)
        old_public = parity["cases"][simulation]["public_anchor"]
        if (abs(public_wave["peak_after_anchor_M"]
                - old_public["peak_after_anchor_M"]) > 1e-8
                or abs(public_wave["peak_amplitude"]
                       - old_public["peak_amplitude"]) > 1e-8):
            raise ValueError(f"{simulation} public waveform changed")
        if any(wave["time"][0] > wave["peak_after_anchor_M"] + start
               or wave["time"][-1] < wave["peak_after_anchor_M"] + stop
               for wave in (native, public_wave)):
            raise ValueError(f"{simulation} lacks frozen postpeak support")
        nshape = peak_relative(native, offsets)
        pshape = peak_relative(public_wave, offsets)
        intervals = {}
        for lower, upper in method["postpeak_intervals_M"]:
            selected = (offsets >= lower) & (offsets < upper)
            intervals[f"{lower:g}:{upper:g}"] = q.mode_metrics(
                nshape[selected], pshape[selected])
        checkpoints = {}
        for offset in method["sample_offsets_M"]:
            index = int(round((offset - offsets[0]) / dt))
            checkpoints[str(offset)] = dict(
                amplitude_ratio=float(abs(nshape[index]) / abs(pshape[index])),
                phase_difference_rad=float(np.angle(nshape[index] / pshape[index])))
        final_mass = entry["native"]["final_mass_fraction"]
        final_spin = entry["native"]["final_spin"]
        omega, damping = signed_qnm_220_table_convention(
            torch.tensor(final_spin, dtype=torch.float64))
        native_qnm = dict(frequency_M=float(omega / final_mass),
                          damping_M=float(damping / final_mass))
        row = case_plan["cases"][simulation]["row"]
        public_qnm = _public_qnm(row)
        public_from_final = _public_qnm(
            row, final_mass_solar=final_mass * (row["mass1"] + row["mass2"]),
            final_spin=final_spin)
        qnm_differences = {
            "binary": max(abs(native_qnm[key] - public_qnm[key]) for key in native_qnm),
            "same_remnant": max(abs(native_qnm[key] - public_from_final[key])
                                for key in native_qnm),
        }
        cases[simulation] = dict(
            anchored_array_sha256=q.sha256(path),
            original_fixed_postpeak_complex_relative_l2=parity["cases"][simulation][
                "fixed_native_vs_public"]["windows"]["post_peak"]["metrics"][
                    "complex_relative_l2"],
            peak_lag_difference_M=parity["cases"][simulation][
                "fixed_native_vs_public"]["peak_lag_difference_M"],
            peak_relative_shape=q.mode_metrics(nshape, pshape),
            postpeak_intervals=intervals,
            checkpoints=checkpoints,
            qnm_control=dict(native=native_qnm, public_binary=public_qnm,
                             public_at_native_remnant=public_from_final,
                             max_absolute_difference_M=qnm_differences,
                             passed=all(value <= plan["qnm_control"]["absolute_frequency_budget_M"]
                                        for value in qnm_differences.values())))
        print(json.dumps(dict(simulation=simulation,
                              peak_relative_complex_l2=cases[simulation][
                                  "peak_relative_shape"]["complex_relative_l2"])),
              flush=True)
    result = dict(schema="diffgw.seobnrv4.sxs-third-postpeak-shape.v1",
                  plan_sha256=q.sha256(args.plan), script_sha256=q.sha256(__file__),
                  lal_binary=identity, posthoc=True, release_qualified=False,
                  cases=cases)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
