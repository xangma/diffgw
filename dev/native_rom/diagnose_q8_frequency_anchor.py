"""Compare frozen q8 modes using an independently defined frequency anchor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_peak_anchor import fitted_metrics, fitted_peak, public_mode
from .diagnose_q8_mode_phase import _replay


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/q8_frequency_anchor_plan.json"


def frequency_anchor(time_s, mode, mass_seconds, target, peak_time_s):
    """Interpolate the first pre-peak crossing of a mode's own GW frequency."""
    time_M = np.asarray(time_s) / mass_seconds
    unwrapped_phase = np.unwrap(np.angle(mode))
    omega_M = -np.gradient(unwrapped_phase, time_M, edge_order=2)
    crossing = np.flatnonzero((omega_M[:-1] < target)
                              & (omega_M[1:] >= target)
                              & (time_s[:-1] < peak_time_s))
    if crossing.size != 1:
        raise ValueError(f"expected one pre-peak frequency crossing, found {crossing.size}")
    i = int(crossing[0])
    fraction = float((target-omega_M[i])/(omega_M[i+1]-omega_M[i]))
    crossing_time_s = float(time_s[i]+fraction*(time_s[i+1]-time_s[i]))
    crossing_phase_rad = float(unwrapped_phase[i]
                               + fraction*(unwrapped_phase[i+1]-unwrapped_phase[i]))
    return dict(time_s=crossing_time_s, phase_rad=crossing_phase_rad,
                bracket_indices=[i, i+1], bracket_angular_frequency_M=[
                    float(omega_M[i]), float(omega_M[i+1])],
                peak_lag_M=float((peak_time_s-crossing_time_s)/mass_seconds))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    campaign_path = q.ROOT / plan["campaign"]
    phase_path = q.ROOT / plan["mode_phase"]
    if (plan["schema"] != "diffgw.seobnrv4.q8-frequency-anchor-plan.v1"
            or plan["row_name"] != "q8_high_aligned"
            or plan["level_name"] != "ode_2"
            or plan["peak_fit"] != "quadratic_power"
            or plan["anchor_angular_frequency_M"] != 0.1
            or plan["peak_relative_window_boundaries_M"] != [-20., 0.]
            or plan["probe_times_from_reference_peak_M"] != [
                -1100., -1000., -800., -600., -400., -200., -100., -20.,
                -10., 0., 10., 30.]
            or plan["release_qualified"] is not False
            or q.sha256(campaign_path) != plan["campaign_sha256"]
            or q.sha256(phase_path) != plan["mode_phase_sha256"]):
        parser.error("frozen plan or inputs differ")
    campaign = json.loads(campaign_path.read_text())
    prior = json.loads(phase_path.read_text())
    row_item = campaign["rows"][plan["row_name"]]
    row, choice = row_item["row"], campaign["choice"]
    pin = campaign["lal_binary"]
    if (campaign["status"] != "diagnostic_complete"
            or prior["status"] != "diagnostic_complete"
            or prior["row"] != row or prior["choice"] != choice
            or row_item["runs"]["ode_2"]["level"]["name"] != "ode_2"
            or q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin):
        parser.error("campaign or pinned public binary differs")
    torch.set_num_threads(1)
    source_paths = (
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/qualify_generator.py",
        "dev/native_rom/diagnose_peak_anchor.py",
        "dev/native_rom/diagnose_q8_mode_phase.py",
        "src/torchwave/approximants/eob/seobnrv4_flux.py",
    )
    report = dict(
        schema="diffgw.seobnrv4.q8-frequency-anchor.v1", scope=q.SCOPE,
        status="running", release_qualified=False, plan_sha256=q.sha256(PLAN),
        script_sha256=q.sha256(__file__), campaign_sha256=q.sha256(campaign_path),
        mode_phase_sha256=q.sha256(phase_path),
        source_sha256={path: q.sha256(q.ROOT / path) for path in source_paths},
        lal_binary=pin, row=row, choice=choice, errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                     pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        run = _replay(row, row_item["runs"]["ode_2"]["level"],
                      q.Settings(**campaign["settings"]), choice,
                      row_item["fixed_restart_time_M"])
        summary = q.summarize_run(run)
        if summary["array_sha256"] != row_item["runs"]["ode_2"]["native"]["array_sha256"]:
            raise ValueError("native fine waveform differs from frozen campaign")
        report["native"] = dict(array_sha256=summary["array_sha256"],
                                elapsed_seconds=time.monotonic()-started)
        q.write_evidence(args.json_out, report)
        print("native_waveform_replayed", flush=True)
        nt, nm, rt, rm, mass_seconds = public_mode(row, run)
        fitted = fitted_metrics(nt, nm, rt, rm, mass_seconds, plan["peak_fit"])
        if not np.isclose(fitted["metrics"]["complex_relative_l2"],
                          prior["peak_fitted"]["metrics"]["complex_relative_l2"],
                          atol=1e-12, rtol=0):
            raise ValueError("peak-fitted public mode differs from frozen comparison")
        npk = fitted_peak(nt, nm, plan["peak_fit"])
        rpk = fitted_peak(rt, rm, plan["peak_fit"])
        target = plan["anchor_angular_frequency_M"]
        na = frequency_anchor(nt, nm, mass_seconds, target, npk["time_s"])
        ra = frequency_anchor(rt, rm, mass_seconds, target, rpk["time_s"])
        ntime = (nt-na["time_s"])/mass_seconds
        rtime = (rt-ra["time_s"])/mass_seconds
        nmode = nm*np.exp(-1j*na["phase_rad"])
        rmode = rm*np.exp(-1j*ra["phase_rad"])
        common = (rtime >= ntime[0]) & (rtime <= ntime[-1])
        times = rtime[common]
        reference = rmode[common]
        candidate = (np.interp(times, ntime, nmode.real)
                     + 1j*np.interp(times, ntime, nmode.imag))
        if times.size < 3:
            raise ValueError("too few common frequency-anchored samples")
        relative_peak_time = times-ra["peak_lag_M"]
        windows = {}
        lower, upper = plan["peak_relative_window_boundaries_M"]
        for name, mask in (("inspiral", relative_peak_time < lower),
                           ("late_inspiral", (relative_peak_time >= lower)
                            & (relative_peak_time < upper)),
                           ("post_peak", relative_peak_time >= upper)):
            if mask.sum() >= 3:
                windows[name] = dict(
                    time_from_reference_peak_M=[float(relative_peak_time[mask][0]),
                                                float(relative_peak_time[mask][-1])],
                    metrics=q.mode_metrics(candidate[mask], reference[mask]))
        active = ((abs(candidate) > max(abs(candidate))*1e-4)
                  & (abs(reference) > max(abs(reference))*1e-4))
        active_time = relative_peak_time[active]
        active_phase = np.unwrap(np.angle(candidate[active]*reference[active].conj()))
        probes = []
        if (active_time[0] > min(plan["probe_times_from_reference_peak_M"])
                or active_time[-1] < max(plan["probe_times_from_reference_peak_M"])):
            raise ValueError("probes exceed active waveform support")
        for t in plan["probe_times_from_reference_peak_M"]:
            c = (np.interp(t, relative_peak_time, candidate.real)
                 + 1j*np.interp(t, relative_peak_time, candidate.imag))
            r = (np.interp(t, relative_peak_time, reference.real)
                 + 1j*np.interp(t, relative_peak_time, reference.imag))
            probes.append(dict(time_from_reference_peak_M=t,
                               phase_difference_rad=float(np.interp(t, active_time, active_phase)),
                               amplitude_ratio=float(abs(c)/abs(r))))
        report["peak_fitted"] = fitted
        report["frequency_anchored"] = dict(
            target_angular_frequency_M=target, native_anchor=na,
            reference_anchor=ra,
            native_minus_reference_peak_lag_M=na["peak_lag_M"]-ra["peak_lag_M"],
            common_support_from_anchor_M=[float(times[0]), float(times[-1])],
            metrics=q.mode_metrics(candidate, reference), windows=windows,
            probes=probes)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
