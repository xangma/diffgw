"""Frozen whole-waveform sensitivity to the historical Delta_t log derivative."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import sys
import time

import torch

from . import qualify_generator as q
from .compare_whole_waveforms import aligned_metrics
from .diagnose_peak_anchor import fitted_metrics, public_mode


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/hamiltonian_whole_waveform_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_mode_33_whole_waveform"]
    state_path = q.ROOT / plan["same_state_hamiltonian"]
    ham_path = q.ROOT / "dev/native_rom/eob/torch/historical_hamiltonian.py"
    if (plan["schema"] != "diffgw.seobnrv4.hamiltonian-whole-waveform-plan.v1"
            or q.sha256(prior_path) != plan["prior_mode_33_whole_waveform_sha256"]
            or q.sha256(state_path) != plan["same_state_hamiltonian_sha256"]
            or q.sha256(ham_path) != plan["diagnostic_hamiltonian_sha256"]
            or plan["baseline_hamiltonian"] != "analytic"
            or plan["candidate_hamiltonian"] != "lal_v4"
            or plan["mode_33_higher_spin"] != "lal_v4_rho"
            or plan["peak_fit"] != "quadratic_power"):
        parser.error("frozen plan or prior evidence differs")
    prior = json.loads(prior_path.read_text())
    state = json.loads(state_path.read_text())
    if (prior["status"] != "diagnostic_complete"
            or state["status"] != "diagnostic_complete"):
        parser.error("prior diagnostic did not complete")
    row, level = prior["row"], prior["level"]
    settings, choice = q.Settings(**prior["settings"]), prior["numerical_choice"]
    pin = prior["lal_binary"]
    if (q.canonical_hash(q.lal_binary_identity(
            pin["lal_version"], pin["lalsimulation_version"])["binaries"])
            != pin["binary_manifest_sha256"]):
        parser.error("public binary identity differs from frozen comparison")
    torch.set_num_threads(1)
    sources = ["src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
               "src/torchwave/approximants/eob/seobnrv4_flux.py",
               "dev/native_rom/eob/torch/historical_hamiltonian.py",
               "dev/native_rom/eob/torch/adaptive_driver_clean.py",
               "dev/native_rom/qualify_generator.py"]
    report = dict(
        schema="diffgw.seobnrv4.hamiltonian-whole-waveform.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        historical_parity_established=False, plan_sha256=q.sha256(PLAN),
        prior_sha256=q.sha256(prior_path), state_sha256=q.sha256(state_path),
        source_sha256={name: q.sha256(q.ROOT / name) for name in sources},
        script_sha256=q.sha256(__file__), row=row, level=level,
        settings=asdict(settings), numerical_choice=choice,
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                     argv=sys.argv), results={}, errors=[])
    q.write_evidence(args.json_out, report)
    try:
        for label, convention in (("baseline", "analytic"), ("candidate", "lal_v4")):
            print(f"{label}: hamiltonian_log_derivative={convention}", flush=True)
            started = time.monotonic()
            run = q.native_run(
                row, level, settings,
                diagnostic_hermite_resampling=choice["hermite_resampling"],
                diagnostic_local_nqc_derivatives=choice["local_nqc_phase_derivatives"],
                diagnostic_restart_time_M=choice["fixed_restart_time_M"],
                diagnostic_mode_33_higher_spin="lal_v4_rho",
                diagnostic_hamiltonian_log_derivative=convention)
            native = q.summarize_run(run)
            if (label == "baseline"
                    and native["array_sha256"] != prior["results"]["candidate"]["native"]["array_sha256"]):
                raise ValueError("baseline waveform differs from frozen prior")
            nt, nm, rt, rm, mass_seconds = public_mode(row, run)
            report["results"][label] = dict(
                hamiltonian_log_derivative=convention, native=native,
                sampled_peak=aligned_metrics(nt, nm, rt, rm, mass_seconds),
                fitted_peak=fitted_metrics(nt, nm, rt, rm, mass_seconds, "quadratic_power"),
                elapsed_seconds=time.monotonic()-started)
            q.write_evidence(args.json_out, report)
        report["sensitivity"] = {
            anchoring: {
                key: {label: report["results"][label][anchoring]["metrics"][key]
                      for label in ("baseline", "candidate")}
                for key in ("complex_relative_l2", "amplitude_relative_l2",
                            "phase_rms_rad", "unmaximized_real_overlap")}
            for anchoring in ("sampled_peak", "fitted_peak")}
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], sensitivity=report.get("sensitivity"),
                          errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
