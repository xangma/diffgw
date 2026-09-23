"""Dev-only whole-waveform sensitivity to the compiled v4 orbital state.

This intervention is a diagnostic probe, never a native initial-condition
prescription or a route to training the ROM.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_lal_dynamics import _reference, array_hash
from .diagnose_peak_anchor import fitted_metrics, public_mode


EVIDENCE = q.ROOT / "dev/native_rom/evidence/2026-09-23"
PLAN = EVIDENCE / "lal_initial_state_ablation_plan.json"


def _reference_initial(row, dynamics_row, step):
    values = _reference(row, step)
    expected = dynamics_row["sampled_references"][str(step)]["reference_array_sha256"]
    if array_hash(values) != expected:
        raise ValueError("sampled compiled dynamics differs from frozen diagnostic")
    frequency = np.gradient(values[2], values[0], edge_order=2)
    index = int(np.argmin(abs(frequency-row["omega_start"])))
    return dict(index=index, time_M=float(values[0, index]),
                frequency_Momega=float(frequency[index]),
                r=float(values[1, index]),
                pr_star=float(values[3, index]),
                pphi=float(values[4, index]))


def _run(row, level, settings, choice):
    return q.native_run(
        row, level, settings,
        diagnostic_hermite_resampling=choice["hermite_resampling"],
        diagnostic_local_nqc_derivatives=choice["local_nqc_phase_derivatives"],
        diagnostic_restart_time_M=choice["fixed_restart_time_M"])


def _row(index, prior, peaks, dynamics, plan):
    import torchwave.approximants.eob.initial_conditions as initial_conditions

    row = prior["rows"][index]
    level, choice = prior["level"], prior["numerical_choice"]
    settings = q.Settings(**prior["settings"])
    reference = _reference_initial(row, dynamics["results"][index], plan["reference_step_M"])
    baseline = _run(row, level, settings, choice)
    baseline_summary = q.summarize_run(baseline)
    expected_hashes = prior["results"][index]["native"]["array_sha256"]
    if baseline_summary["array_sha256"] != expected_hashes:
        raise ValueError("baseline native waveform differs from frozen prior")
    old_metrics = peaks["results"][index]["fitted_peaks"]["quadratic_power"]["metrics"]

    original = initial_conditions.find_quasicircular_initial_conditions
    replacement = {}

    def substitute(*positional, **named):
        r, pr_star, pphi = original(*positional, **named)
        replacement.update(native=dict(r=float(r), pr_star=float(pr_star), pphi=float(pphi)))
        return (r.new_tensor(reference["r"]), pr_star.new_tensor(reference["pr_star"]),
                pphi.new_tensor(reference["pphi"]))

    with patch.object(initial_conditions, "find_quasicircular_initial_conditions", substitute):
        candidate = _run(row, level, settings, choice)
    if not replacement:
        raise ValueError("diagnostic initial-state intervention did not run")
    candidate_summary = q.summarize_run(candidate)
    native_time, native_mode, reference_time, reference_mode, mass_seconds = public_mode(
        row, candidate)
    comparison = fitted_metrics(native_time, native_mode, reference_time,
                                reference_mode, mass_seconds, "quadratic_power")
    return dict(row=row, reference_initial=reference, baseline_initial=replacement["native"],
                baseline=dict(array_sha256=baseline_summary["array_sha256"],
                              fitted_peak_metrics=old_metrics),
                intervention=dict(native=candidate_summary, fitted_peak=comparison),
                complex_relative_l2_change=(comparison["metrics"]["complex_relative_l2"]
                                            - old_metrics["complex_relative_l2"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    paths = {key: q.ROOT / plan[key] for key in (
        "prior_whole_waveform", "prior_peak_anchor", "dynamics_diagnostic")}
    if (plan["schema"] != "diffgw.seobnrv4.lal-initial-state-ablation-plan.v1"
            or plan["rows"] != [0, 1]
            or plan["reference_step_M"] != 0.125
            or q.sha256(paths["prior_whole_waveform"]) != plan["prior_sha256"]
            or q.sha256(paths["prior_peak_anchor"]) != plan["prior_peak_anchor_sha256"]
            or q.sha256(paths["dynamics_diagnostic"]) != plan["dynamics_sha256"]):
        parser.error("frozen plan or input evidence hash mismatch")
    prior, peaks, dynamics = (json.loads(paths[key].read_text()) for key in paths)
    if (prior["status"] != "diagnostic_complete"
            or peaks["status"] != "diagnostic_complete"
            or dynamics["status"] != "diagnostic_complete"
            or peaks["prior_whole_waveform"]["sha256"] != plan["prior_sha256"]
            or prior["rows"] != [dynamics["results"][i]["row"] for i in plan["rows"]]
            or prior["rows"] != [peaks["results"][i]["row"] for i in plan["rows"]]):
        parser.error("incompatible source diagnostics")
    pin = prior["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("compiled LALSuite binary differs from frozen evidence")
    report = dict(schema="diffgw.seobnrv4.lal-initial-state-ablation.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  whole_waveform_parity_validated=False,
                  plan_sha256=q.sha256(PLAN),
                  input_sha256={key: q.sha256(path) for key, path in paths.items()},
                  script_sha256=q.sha256(__file__),
                  source_sha256={path: q.sha256(q.ROOT / path) for path in (
                      "dev/native_rom/eob/torch/adaptive_driver_clean.py",
                      "src/torchwave/approximants/eob/initial_conditions.py",
                      "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
                      "src/torchwave/approximants/eob/seobnrv4_flux.py")},
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), results=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        for index in plan["rows"]:
            started = time.monotonic()
            result = _row(index, prior, peaks, dynamics, plan)
            result["elapsed_seconds"] = time.monotonic()-started
            report["results"].append(result)
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(index=index, elapsed_seconds=result["elapsed_seconds"],
                                  baseline_l2=result["baseline"]["fitted_peak_metrics"]["complex_relative_l2"],
                                  intervened_l2=result["intervention"]["fitted_peak"]["metrics"]["complex_relative_l2"])),
                  flush=True)
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
