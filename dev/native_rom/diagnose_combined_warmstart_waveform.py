"""Frozen one-row full-waveform sensitivity after a wholly native warm start."""

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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/combined_warmstart_waveform_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    warm_path = q.ROOT / plan["prior_warmstart"]
    ham_path = q.ROOT / plan["prior_hamiltonian_waveform"]
    if (plan["schema"] != "diffgw.seobnrv4.combined-warmstart-waveform-plan.v1"
            or q.sha256(warm_path) != plan["prior_warmstart_sha256"]
            or q.sha256(ham_path) != plan["prior_hamiltonian_waveform_sha256"]
            or plan["row_index"] != 1
            or plan["peak_fit"] != "quadratic_power"
            or [item["name"] for item in plan["choices"]] != [
                "baseline", "mode_33", "hamiltonian", "combined"]):
        parser.error("frozen plan or evidence differs")
    warm, ham = (json.loads(path.read_text()) for path in (warm_path, ham_path))
    row = warm["results"][1]["row"]
    if (warm["status"] != "diagnostic_complete"
            or ham["status"] != "diagnostic_complete"
            or row != ham["row"]):
        parser.error("prior diagnostic differs")
    old_plan = json.loads((q.ROOT / "dev/native_rom/evidence/2026-09-23/"
                           "native_warmstart_whole_waveform_plan.json").read_text())
    prior = json.loads((q.ROOT / old_plan["prior_whole_waveform"]).read_text())
    pin = prior["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("public binary identity changed")
    level, choice = ham["level"], ham["numerical_choice"]
    if level != prior["level"] or choice != prior["numerical_choice"]:
        parser.error("numerical choices differ from frozen warm start")
    settings = q.Settings(**ham["settings"])
    torch.set_num_threads(1)
    sources = ["src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
               "src/torchwave/approximants/eob/seobnrv4_flux.py",
               "dev/native_rom/eob/torch/historical_hamiltonian.py",
               "dev/native_rom/eob/torch/warm_start.py",
               "dev/native_rom/eob/torch/adaptive_driver_clean.py",
               "dev/native_rom/qualify_generator.py"]
    report = dict(
        schema="diffgw.seobnrv4.combined-warmstart-waveform.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), prior_sha256=dict(warm=q.sha256(warm_path),
                                                      hamiltonian=q.sha256(ham_path)),
        source_sha256={name: q.sha256(q.ROOT / name) for name in sources},
        script_sha256=q.sha256(__file__), lal_binary=pin, row=row, level=level,
        settings=asdict(settings), numerical_choice=choice,
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                     argv=sys.argv), results={}, errors=[])
    q.write_evidence(args.json_out, report)
    try:
        for item in plan["choices"]:
            print(f"choice={item['name']}", flush=True)
            started = time.monotonic()
            run = q.native_run(
                row, level, settings,
                diagnostic_hermite_resampling=choice["hermite_resampling"],
                diagnostic_local_nqc_derivatives=choice["local_nqc_phase_derivatives"],
                diagnostic_restart_time_M=choice["fixed_restart_time_M"],
                diagnostic_warm_start_Momega=plan["warm_start_Momega"],
                diagnostic_mode_33_higher_spin=item["mode_33_higher_spin"],
                diagnostic_hamiltonian_log_derivative=item["hamiltonian_log_derivative"])
            native = q.summarize_run(run)
            if (item["name"] == "baseline"
                    and native["array_sha256"] != warm["results"][1]["waveform"]["array_sha256"]):
                raise ValueError("baseline differs from frozen native warm start")
            nt, nm, rt, rm, mass_seconds = public_mode(row, run)
            report["results"][item["name"]] = dict(
                choice=item, native=native,
                sampled_peak=aligned_metrics(nt, nm, rt, rm, mass_seconds),
                fitted_peak=fitted_metrics(nt, nm, rt, rm, mass_seconds,
                                           "quadratic_power"),
                elapsed_seconds=time.monotonic()-started)
            q.write_evidence(args.json_out, report)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          fitted_l2={name: item["fitted_peak"]["metrics"][
                              "complex_relative_l2"] for name, item in
                              report["results"].items()})), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
