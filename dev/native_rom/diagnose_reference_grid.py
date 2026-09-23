"""Check whether the public SEOBNRv4 peak anchor is stable across output grids."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import sys

import lal
import lalsimulation as lalsim
import numpy as np
from scipy.interpolate import make_interp_spline

from . import qualify_generator as q
from .diagnose_peak_anchor import fitted_peak


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/reference_grid_plan.json"


def public_mode(row, step_M):
    from scripts.verify_eob_published_waveforms import _invert_22
    from torchwave.constants import MSUN_SEC

    mass_seconds = (row["mass1"] + row["mass2"]) * float(MSUN_SEC)
    dt = step_M * mass_seconds
    hp, hc = lalsim.SimInspiralChooseTDWaveform(
        row["mass1"] * lal.MSUN_SI, row["mass2"] * lal.MSUN_SI,
        0., 0., row["chi1"], 0., 0., row["chi2"],
        100e6 * lal.PC_SI, 0.8, 0., 0., 0., 0.,
        dt, row["omega_start"] / (math.pi * mass_seconds), 0.,
        lal.CreateDict(), lalsim.GetApproximantFromString("SEOBNRv4"))
    if len(hp.data.data) != len(hc.data.data) or hp.epoch != hc.epoch or hp.deltaT != hc.deltaT:
        raise ValueError("inconsistent public polarization grids")
    time = float(hp.epoch) + np.arange(len(hp.data.data)) * float(hp.deltaT)
    mode = _invert_22(np.asarray(hp.data.data), np.asarray(hc.data.data), 0.8)
    return time, mode, mass_seconds


def compare(time_coarse, mode_coarse, time_fine, mode_fine, mass_seconds, anchors,
            windows_M):
    coarse_peak = fitted_peak(time_coarse, mode_coarse, "quadratic_power")
    fine_peak = fitted_peak(time_fine, mode_fine, "quadratic_power")
    spline = make_interp_spline(time_coarse, mode_coarse, k=5)
    overlap = (time_fine >= time_coarse[0]) & (time_fine <= time_coarse[-1])
    raw_candidate = spline(time_fine[overlap])
    raw_reference = mode_fine[overlap]
    raw = q.mode_metrics(raw_candidate, raw_reference)
    raw_time_M = (time_fine[overlap] - fine_peak["time_s"]) / mass_seconds
    post_mask = raw_time_M >= 0
    post_candidate, post_reference = raw_candidate[post_mask], raw_reference[post_mask]
    scalar = np.vdot(post_candidate, post_reference) / np.vdot(post_candidate, post_candidate)
    scalar_diagnostic = dict(amplitude=float(abs(scalar)), phase_rad=float(np.angle(scalar)),
                             post_peak_after_fit=q.mode_metrics(scalar * post_candidate,
                                                                post_reference),
                             whole_after_fit=q.mode_metrics(scalar * raw_candidate,
                                                            raw_reference))
    raw_windows = []
    total_squared_error = float(np.vdot(raw_candidate - raw_reference,
                                       raw_candidate - raw_reference).real)
    for low, high in windows_M:
        mask = (raw_time_M >= low) & (raw_time_M < high)
        if mask.sum() < 3:
            raise ValueError("too few samples in frozen reference window")
        local_error = raw_candidate[mask] - raw_reference[mask]
        raw_windows.append(dict(time_M=[low, high], samples=int(mask.sum()),
                                metrics=q.mode_metrics(raw_candidate[mask], raw_reference[mask]),
                                squared_error_fraction=float(np.vdot(local_error, local_error).real
                                                             / total_squared_error)))
    output = {}
    for anchor in anchors:
        if anchor == "independent_peaks":
            coarse_origin, fine_origin = coarse_peak["time_s"], fine_peak["time_s"]
            coarse_phase, fine_phase = coarse_peak["phase_rad"], fine_peak["phase_rad"]
        elif anchor == "shared_fine_peak":
            coarse_origin = fine_origin = fine_peak["time_s"]
            coarse_phase = fine_phase = fine_peak["phase_rad"]
        else:
            raise ValueError(f"unknown anchor: {anchor}")
        sample_at = time_fine - fine_origin + coarse_origin
        valid = (sample_at >= time_coarse[0]) & (sample_at <= time_coarse[-1])
        candidate = spline(sample_at[valid]) * np.exp(-1j * coarse_phase)
        reference = mode_fine[valid] * np.exp(-1j * fine_phase)
        peak_relative = (time_fine[valid] - fine_origin) / mass_seconds
        output[anchor] = dict(whole=q.mode_metrics(candidate, reference),
                              post_peak=q.mode_metrics(candidate[peak_relative >= 0],
                                                       reference[peak_relative >= 0]),
                              samples=int(valid.sum()))
    return dict(raw_epoch=raw, raw_windows=raw_windows,
                post_peak_complex_scalar_diagnostic=scalar_diagnostic,
                anchors=output, coarse_peak=coarse_peak,
                fine_peak=fine_peak,
                peak_time_difference_M=(coarse_peak["time_s"] - fine_peak["time_s"]) / mass_seconds,
                peak_phase_difference_rad=coarse_peak["phase_rad"] - fine_peak["phase_rad"],
                coarse_epoch_s=float(time_coarse[0]), fine_epoch_s=float(time_fine[0]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_whole_waveform"]
    prior = json.loads(prior_path.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.reference-grid-plan.v1"
            or q.sha256(prior_path) != plan["prior_whole_waveform_sha256"]
            or prior["status"] != "diagnostic_complete"
            or prior["rows"][plan["row_index"]] != plan["row"]
            or plan["steps_M"] != [0.5, 0.125, 0.03125]
            or q.lal_binary_identity(prior["lal_binary"]["lal_version"],
                                     prior["lal_binary"]["lalsimulation_version"])
            != prior["lal_binary"]):
        parser.error("frozen plan or public binary differs")
    report = dict(schema="diffgw.seobnrv4.reference-grid.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  whole_waveform_parity_validated=False, plan_sha256=q.sha256(PLAN),
                  prior_sha256=q.sha256(prior_path), script_sha256=q.sha256(__file__),
                  lal_binary=prior["lal_binary"], process=dict(
                      host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
                  row=plan["row"], result=None, errors=[])
    q.write_evidence(args.json_out, report)
    try:
        modes = [public_mode(plan["row"], step) for step in plan["steps_M"]]
        if len({item[2] for item in modes}) != 1:
            raise ValueError("mass seconds differ")
        report["result"] = {f"{plan['steps_M'][i]}_to_{plan['steps_M'][i+1]}":
                            compare(modes[i][0], modes[i][1], modes[i+1][0],
                                    modes[i+1][1], modes[i][2], plan["anchors"],
                                    plan["windows_M"])
                            for i in range(len(modes) - 1)}
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          evidence=str(args.json_out.resolve()))), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
