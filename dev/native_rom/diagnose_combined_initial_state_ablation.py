"""Causal early-state ablation of the remaining combined v4 waveform residual."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch

import torch

from . import qualify_generator as q
from .diagnose_peak_anchor import fitted_metrics, public_mode
from .eob.torch import warm_start


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/combined_initial_state_ablation_plan.json"
WAVEFORM_PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/combined_warmstart_waveform_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    wave_path, pre_path = (q.ROOT / plan[key] for key in
                           ("prior_waveform", "prior_pre_evolution"))
    if (plan["schema"] != "diffgw.seobnrv4.combined-initial-state-ablation-plan.v1"
            or q.sha256(wave_path) != plan["prior_waveform_sha256"]
            or q.sha256(pre_path) != plan["prior_pre_evolution_sha256"]
            or plan["row_index"] != 1 or plan["state"] != "reference_initial"
            or plan["peak_fit"] != "quadratic_power"):
        parser.error("frozen plan or evidence differs")
    prior, pre = (json.loads(path.read_text()) for path in (wave_path, pre_path))
    wave_plan = json.loads(WAVEFORM_PLAN.read_text())
    if (prior["status"] != "diagnostic_complete"
            or pre["status"] != "diagnostic_complete"
            or prior["plan_sha256"] != q.sha256(WAVEFORM_PLAN)
            or prior["row"] != pre["results"][1]["row"]):
        parser.error("prior diagnostics differ")
    pin = prior["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("public binary identity changed")
    row, level, choice = prior["row"], prior["level"], prior["numerical_choice"]
    settings = q.Settings(**prior["settings"])
    mode = prior["results"]["combined"]["choice"]
    reference_initial = pre["results"][1]["reference_initial"]
    torch.set_num_threads(1)
    sources = ["src/torchwave/approximants/eob/initial_conditions.py",
               "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
               "src/torchwave/approximants/eob/seobnrv4_flux.py",
               "dev/native_rom/eob/torch/historical_hamiltonian.py",
               "dev/native_rom/eob/torch/warm_start.py",
               "dev/native_rom/eob/torch/adaptive_driver_clean.py",
               "dev/native_rom/qualify_generator.py"]
    report = dict(
        schema="diffgw.seobnrv4.combined-initial-state-ablation.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        reference_state_runtime_dependency=False,
        plan_sha256=q.sha256(PLAN), prior_sha256=dict(waveform=q.sha256(wave_path),
                                                      pre_evolution=q.sha256(pre_path)),
        source_sha256={name: q.sha256(q.ROOT / name) for name in sources},
        script_sha256=q.sha256(__file__), row=row, level=level,
        numerical_choice=choice, candidate_choice=mode, settings=asdict(settings),
        public_initial=reference_initial,
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                     argv=sys.argv), results={}, errors=[])
    q.write_evidence(args.json_out, report)

    def solve():
        return q.native_run(
            row, level, settings,
            diagnostic_hermite_resampling=choice["hermite_resampling"],
            diagnostic_local_nqc_derivatives=choice["local_nqc_phase_derivatives"],
            diagnostic_restart_time_M=choice["fixed_restart_time_M"],
            diagnostic_warm_start_Momega=wave_plan["warm_start_Momega"],
            diagnostic_mode_33_higher_spin=mode["mode_33_higher_spin"],
            diagnostic_hamiltonian_log_derivative=mode["hamiltonian_log_derivative"])

    try:
        print("choice=baseline", flush=True)
        started = time.monotonic()
        baseline = solve()
        native = q.summarize_run(baseline)
        if native["array_sha256"] != prior["results"]["combined"]["native"]["array_sha256"]:
            raise ValueError("combined baseline differs from frozen waveform")
        report["results"]["baseline"] = dict(native=native,
                                              fitted_peak=prior["results"]["combined"]["fitted_peak"],
                                              elapsed_seconds=time.monotonic()-started)
        q.write_evidence(args.json_out, report)

        original = warm_start.find_quasicircular_initial_conditions
        measured = []

        def substitute(*args, **kwargs):
            r, pr_star, pphi = original(*args, **kwargs)
            measured.append(dict(r=float(r), pr_star=float(pr_star), pphi=float(pphi)))
            return (r.new_tensor(reference_initial["r"]),
                    pr_star.new_tensor(reference_initial["pr_star"]),
                    pphi.new_tensor(reference_initial["pphi"]))

        print("choice=public_initial_state", flush=True)
        started = time.monotonic()
        with patch.object(warm_start, "find_quasicircular_initial_conditions", substitute):
            candidate = solve()
        if len(measured) != 1:
            raise ValueError("expected one warm-start initial-state intervention")
        nt, nm, rt, rm, mass_seconds = public_mode(row, candidate)
        report["native_initial"] = measured[0]
        report["results"]["public_initial_state"] = dict(
            native=q.summarize_run(candidate),
            fitted_peak=fitted_metrics(nt, nm, rt, rm, mass_seconds,
                                       "quadratic_power"),
            elapsed_seconds=time.monotonic()-started)
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
