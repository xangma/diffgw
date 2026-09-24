"""Replay q8 and localize peak-anchored public-mode phase discrepancy."""

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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/q8_mode_phase_plan.json"


def _replay(row, level, settings, choice, fixed_restart, *, return_boundary_data=False):
    numerical, flux, physical = choice["numerical"], choice["flux"], choice["physical"]
    return q.native_run(
        row, level, settings,
        diagnostic_hermite_resampling=numerical["hermite_resampling"],
        diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
        diagnostic_matching_second_derivatives=True,
        diagnostic_restart_time_M=fixed_restart,
        diagnostic_coarse_initial_step_M=0.125,
        diagnostic_warm_start_Momega=0.03162277660168379,
        diagnostic_nqc_phase_window_M=choice["nqc_phase_window_M"],
        diagnostic_mode_21_higher_spin=flux["mode_21_higher_spin"],
        diagnostic_mode_31_higher_spin=flux["mode_31_higher_spin"],
        diagnostic_mode_32_spin_square=flux["mode_32_spin_square"],
        diagnostic_mode_33_higher_spin=physical["mode_33_higher_spin"],
        diagnostic_mode_43_higher_spin=choice["mode_43_higher_spin"],
        diagnostic_spin_truncation=choice["spin_truncation"],
        diagnostic_hamiltonian_log_derivative=physical["hamiltonian_log_derivative"],
        diagnostic_hamiltonian_calibration=choice["calibration"],
        return_boundary_data=return_boundary_data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    campaign_path = q.ROOT / plan["campaign"]
    orbital_path = q.ROOT / plan["orbital_phase"]
    if (plan["schema"] != "diffgw.seobnrv4.q8-mode-phase-plan.v1"
            or plan["row_name"] != "q8_high_aligned" or plan["level_name"] != "ode_2"
            or plan["peak_fit"] != "quadratic_power"
            or plan["early_phase_anchor_M"] != -1000.0
            or plan["probe_times_M"] != [-1100., -1000., -800., -600., -400.,
                                         -200., -100., -20., -10., 0., 10., 30.]
            or plan["release_qualified"] is not False
            or q.sha256(campaign_path) != plan["campaign_sha256"]
            or q.sha256(orbital_path) != plan["orbital_phase_sha256"]):
        parser.error("frozen plan or inputs differ")
    campaign = json.loads(campaign_path.read_text())
    orbital = json.loads(orbital_path.read_text())
    row_item = campaign["rows"][plan["row_name"]]
    row, choice = row_item["row"], campaign["choice"]
    pin = campaign["lal_binary"]
    if (campaign["status"] != "diagnostic_complete"
            or orbital["status"] != "diagnostic_complete"
            or orbital["row"] != row or orbital["choice"] != choice
            or row_item["runs"]["ode_2"]["level"]["name"] != "ode_2"
            or q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin):
        parser.error("campaign or pinned public binary differs")
    torch.set_num_threads(1)
    source_paths = (
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/qualify_generator.py",
        "dev/native_rom/diagnose_peak_anchor.py",
        "src/torchwave/approximants/eob/seobnrv4_flux.py",
    )
    report = dict(
        schema="diffgw.seobnrv4.q8-mode-phase.v1", scope=q.SCOPE,
        status="running", release_qualified=False, plan_sha256=q.sha256(PLAN),
        script_sha256=q.sha256(__file__), campaign_sha256=q.sha256(campaign_path),
        orbital_phase_sha256=q.sha256(orbital_path),
        source_sha256={path: q.sha256(q.ROOT / path) for path in source_paths},
        lal_binary=pin, row=row, choice=choice, probes=[], errors=[],
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
        baseline = row_item["fitted_peak"]
        if not np.isclose(fitted["metrics"]["complex_relative_l2"],
                          baseline["metrics"]["complex_relative_l2"], atol=1e-12, rtol=0):
            raise ValueError("peak-fitted public mode differs from frozen campaign")
        npk = fitted_peak(nt, nm, plan["peak_fit"])
        rpk = fitted_peak(rt, rm, plan["peak_fit"])
        nt = (nt-npk["time_s"])/mass_seconds
        rt = (rt-rpk["time_s"])/mass_seconds
        nm = nm*np.exp(-1j*npk["phase_rad"])
        rm = rm*np.exp(-1j*rpk["phase_rad"])
        common = (rt >= nt[0]) & (rt <= nt[-1])
        times = rt[common]
        reference = rm[common]
        candidate = (np.interp(times, nt, nm.real)
                     + 1j*np.interp(times, nt, nm.imag))
        active = ((abs(candidate) > max(abs(candidate))*1e-4)
                  & (abs(reference) > max(abs(reference))*1e-4))
        phase = np.unwrap(np.angle(candidate[active]*reference[active].conj()))
        active_times = times[active]
        anchor = plan["early_phase_anchor_M"]
        if not (active_times[0] <= min(plan["probe_times_M"])
                and max(plan["probe_times_M"]) <= active_times[-1]):
            raise ValueError("mode phase probes exceed active waveform support")
        early_phase = float(np.interp(anchor, active_times, phase))
        reanchored = candidate*np.exp(-1j*early_phase)
        report["peak_fitted"] = fitted
        report["early_phase_anchor"] = dict(
            time_M=anchor, peak_anchored_phase_difference_rad=early_phase,
            diagnostic_inspiral_complex_relative_l2=q.mode_metrics(
                reanchored[times < -20.], reference[times < -20.])[
                    "complex_relative_l2"],
            diagnostic_whole_complex_relative_l2=q.mode_metrics(
                reanchored, reference)["complex_relative_l2"])
        for t in plan["probe_times_M"]:
            c = (np.interp(t, times, candidate.real)
                 + 1j*np.interp(t, times, candidate.imag))
            r = (np.interp(t, times, reference.real)
                 + 1j*np.interp(t, times, reference.imag))
            report["probes"].append(dict(
                time_from_fitted_peak_M=t,
                peak_anchored_phase_difference_rad=float(np.interp(t, active_times, phase)),
                phase_difference_from_early_anchor_rad=float(
                    np.interp(t, active_times, phase)-early_phase),
                amplitude_ratio=float(abs(c)/abs(r))))
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
