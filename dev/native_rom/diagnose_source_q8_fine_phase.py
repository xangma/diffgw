"""Audit q8 phase reconstruction on the native attachment's fine trajectory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_source_q8_raw_phase import (
    _anchored, _array, _digest, _nqc_phase, _relative_raw_phase, _state_at,
)
from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
from .eob.torch.source_calibration_hamiltonian import SourceCalibratedHistoricalHamiltonian
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_mode22 import (
    seobnrv4_published_mode22_from_hamiltonian,
)


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_q8_fine_phase_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / path for name, path in plan["inputs"].items()}
    probes = np.asarray(plan["probe_times_M"], dtype=float)
    if (plan["schema"] != "diffgw.seobnrv4.source-q8-fine-phase-plan.v1"
            or plan["row"] != "q8_high_aligned" or plan["level"] != "ode_2"
            or probes.tolist() != [-120., -100., -80., -60., -40., -25.]
            or plan["phase_anchor_M"] != -100.
            or plan["release_qualified"] is not False
            or {name: q.sha256(path) for name, path in paths.items()}
               != plan["input_sha256"]):
        parser.error("frozen fine-phase plan or inputs differ")
    campaign = json.loads(paths["campaign"].read_text())
    raw = json.loads(paths["raw_phase"].read_text())
    interpolation = json.loads(paths["output_interpolation"].read_text())
    previous = json.loads(paths["source_native_mode"].read_text())
    row = campaign["rows"][plan["row"]]["row"]
    if (any(x["status"] != "diagnostic_complete"
            for x in (campaign, raw, interpolation, previous))
            or raw["row"] != row or raw["probe_times_M"] != probes.tolist()
            or interpolation["probe_times_M"] != probes.tolist()):
        parser.error("pinned q8 evidence changed")
    scope = dict(q.SCOPE)
    scope["description"] = "Post-hoc fine-trajectory phase reconstruction on one q8 row."
    scope["known_limitations"] = [
        "Selected after the coarse-state reconstruction residual was observed.",
        "The native attachment spline is an internal consistency control, not an independent reference.",
        "One q8 row cannot qualify generator or ROM physics.",
    ]
    report = dict(schema="diffgw.seobnrv4.source-q8-fine-phase.v1",
                  status="running", release_qualified=False, scope=scope,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256=plan["input_sha256"], row=row, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        torch.set_num_threads(1)
        settings = q.Settings(**campaign["settings"])
        choice = campaign["choice"]
        level = campaign["rows"][plan["row"]]["runs"][plan["level"]]["level"]
        numerical, flux_choice, physical = (choice[key] for key in
                                            ("numerical", "flux", "physical"))
        with q.bounded_run(settings.max_run_seconds), torch.no_grad():
            native = reconstruct_seobnrv4_22_adaptive(
                torch.tensor(row["omega_start"], dtype=torch.float64),
                **{key: row[key] for key in ("mass1", "mass2", "chi1", "chi2")},
                **dict(q.CHOICES, spin_truncation=choice["spin_truncation"]),
                **{key: level[key] for key in ("atol", "rtol", "dt_geom")},
                diagnostic_hermite_resampling=numerical["hermite_resampling"],
                diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
                diagnostic_matching_second_derivatives=True,
                diagnostic_restart_time_M=campaign["rows"][plan["row"]]["fixed_restart_time_M"],
                diagnostic_coarse_initial_step_M=0.125,
                diagnostic_warm_start_Momega=0.03162277660168379,
                diagnostic_nqc_phase_window_M=choice["nqc_phase_window_M"],
                diagnostic_mode_21_higher_spin=flux_choice["mode_21_higher_spin"],
                diagnostic_mode_31_higher_spin=flux_choice["mode_31_higher_spin"],
                diagnostic_mode_32_spin_square=flux_choice["mode_32_spin_square"],
                diagnostic_mode_33_higher_spin=physical["mode_33_higher_spin"],
                diagnostic_mode_43_higher_spin=choice["mode_43_higher_spin"],
                diagnostic_hamiltonian_log_derivative=physical["hamiltonian_log_derivative"],
                diagnostic_hamiltonian_calibration=choice["calibration"],
                ringdown_duration=settings.ringdown_duration_M,
                num_steps=settings.max_accepted_steps,
                max_accepted_steps=settings.max_accepted_steps,
                max_evaluations=settings.max_evaluations,
                max_rejected_steps=settings.max_rejected_steps)
        hashes = dict(time=_digest(_array(native.time)), mode=_digest(_array(native.mode)))
        if hashes != previous["rows"][plan["row"]]["native_array_sha256"]:
            raise ValueError("native waveform replay changed")
        report["native_array_sha256"] = hashes
        q.write_evidence(args.json_out, report)
        print("native_q8_replayed", flush=True)
        fine = native.trajectory.fine
        valid = _array(fine.output_valid).astype(bool)
        count = int(valid.sum())
        if not np.all(valid[:count]) or np.any(valid[count:]):
            raise ValueError("fine validity is not a prefix")
        peak = float(_array(native.trajectory.peak.time).reshape(-1)[0])
        restart = float(_array(native.trajectory.restart_time).reshape(-1)[0])
        fine_t = _array(fine.output_time)[:count] + restart - peak
        fine_states = _array(fine.output_state)[:count]
        if not np.all(np.diff(fine_t) > 0) or probes[0] < fine_t[0] or probes[-1] > fine_t[-1]:
            raise ValueError("fine trajectory does not cover probes")
        fine_probes = np.asarray([_state_at(fine_t, fine_states, p) for p in probes])
        coarse_probes = np.asarray(raw["native_state"])
        x1 = row["mass1"] / (row["mass1"] + row["mass2"])
        def scalar(value):
            return torch.tensor(value, dtype=torch.float64)
        ham = SourceCalibratedHistoricalHamiltonian(
            scalar(x1*(1-x1)), scalar(row["chi1"]), scalar(row["chi2"]),
            delta=scalar(2*x1-1))
        rr = SEOBNRv4PublishedFlux(
            spin_truncation=choice["spin_truncation"], retain_pan_cubic_spin=True,
            mode_21_higher_spin=flux_choice["mode_21_higher_spin"],
            mode_31_higher_spin=flux_choice["mode_31_higher_spin"],
            mode_32_spin_square=flux_choice["mode_32_spin_square"],
            mode_33_higher_spin=physical["mode_33_higher_spin"],
            mode_43_higher_spin=choice["mode_43_higher_spin"])
        field = EOBTrajectoryIntegrator(ham, rr).vector_field
        coeff = np.asarray(previous["rows"][plan["row"]]["nqc_coefficients"]["native"][-2:])
        def reconstruct(states, omega):
            modes = []
            for state, w in zip(states, omega):
                with torch.no_grad():
                    mode = seobnrv4_published_mode22_from_hamiltonian(
                        torch.as_tensor(state, dtype=torch.float64), ham, rr,
                        omega=scalar(w), spin_v6=q.CHOICES["spin_v6"],
                        include_pan_v8=False, include_positive_chi_540=False,
                        retain_pan_cubic_spin=True)
                modes.append(complex(mode.item()))
            nqc = np.asarray([_nqc_phase(state, w, coeff)
                              for state, w in zip(states, omega)])
            raw_phase = _anchored(_relative_raw_phase(modes, states), anchor)
            nqc_phase = _anchored(nqc, anchor)
            return raw_phase+nqc_phase, modes, raw_phase, nqc_phase
        anchor = int(np.flatnonzero(probes == plan["phase_anchor_M"])[0])
        fine_omega = _array(field(torch.as_tensor(fine_probes, dtype=torch.float64)))[:, 1]
        coarse_omega = np.asarray(raw["native_omega_on_native_state"])
        coarse_phase, coarse_modes, coarse_raw, coarse_nqc = reconstruct(
            coarse_probes, coarse_omega)
        if np.max(abs(np.asarray([[z.real, z.imag] for z in coarse_modes])
                      -np.asarray(raw["native_raw_on_native_state"]))) > 1e-11:
            raise ValueError("coarse raw-mode replay changed")
        fine_phase, _, fine_raw, fine_nqc = reconstruct(fine_probes, fine_omega)
        nqc = native.attachment.fine.nqc
        query = torch.as_tensor(probes + peak - restart, dtype=torch.float64)
        spline_raw = _array(nqc.waveform_phase_spline.evaluate(query))
        spline_basis = _array(nqc.phase_basis_spline.evaluate(query))
        spline_coeff = _array(nqc.calibration.coefficients.phase)
        spline_phase = _anchored(-spline_raw + spline_basis @ spline_coeff, anchor)
        coarse_error = np.asarray(raw["native_reconstruction_error_rad"])
        fine_error = coarse_error - (fine_phase-coarse_phase)
        spline_error = coarse_error - (spline_phase-coarse_phase)
        cubic_error = np.asarray(interpolation["results"]["cubic"]["native_reconstruction_error_rad"])
        if np.max(abs(coarse_phase + coarse_error
                      - (fine_phase + fine_error))) > 1e-10:
            raise ValueError("fine phase closure failed")
        report.update(probe_times_M=probes.tolist(),
                      fine_step_M=float(np.median(np.diff(fine_t))),
                      fine_samples=count,
                      fine_time_support_M=[float(fine_t[0]), float(fine_t[-1])],
                      coarse_to_fine_state_max_abs=np.max(abs(fine_probes-coarse_probes), axis=0).tolist(),
                      fine_omega=fine_omega.tolist(),
                      coarse_reconstructed_phase_rad=coarse_phase.tolist(),
                      fine_reconstructed_phase_rad=fine_phase.tolist(),
                      attachment_spline_phase_rad=spline_phase.tolist(),
                      fine_minus_coarse_raw_phase_rad=(fine_raw-coarse_raw).tolist(),
                      fine_minus_coarse_nqc_phase_rad=(fine_nqc-coarse_nqc).tolist(),
                      fine_minus_coarse_reconstruction_rad=(fine_phase-coarse_phase).tolist(),
                      spline_minus_fine_reconstruction_rad=(spline_phase-fine_phase).tolist(),
                      output_linear_minus_fine_reconstruction_rad=fine_error.tolist(),
                      output_linear_minus_attachment_spline_rad=spline_error.tolist(),
                      output_cubic_minus_attachment_spline_rad=(cubic_error-(spline_phase-coarse_phase)).tolist(),
                      elapsed_seconds=time.monotonic()-started)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc),
                                     traceback=traceback.format_exc()))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
