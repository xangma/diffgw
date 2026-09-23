"""Dev-only waveform probe using a fully native pre-evolved starting state."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch

import torch

from . import qualify_generator as q
from .diagnose_lal_initial_state_ablation import _run
from .diagnose_peak_anchor import fitted_metrics, public_mode


EVIDENCE = q.ROOT / "dev/native_rom/evidence/2026-09-23"
PLAN = EVIDENCE / "native_warmstart_whole_waveform_plan.json"


def _row(index, prior, peaks, pre_evolution, lal_ablation):
    import torchwave.approximants.eob.initial_conditions as initial_conditions

    row = prior["rows"][index]
    state = pre_evolution["results"][index]["native_cutoff_crossing_state"]
    original = initial_conditions.find_quasicircular_initial_conditions
    replacement = {}

    def substitute(*positional, **named):
        r, pr_star, pphi = original(*positional, **named)
        replacement.update(native=dict(r=float(r), pr_star=float(pr_star), pphi=float(pphi)))
        return (r.new_tensor(state["r"]), pr_star.new_tensor(state["pr_star"]),
                pphi.new_tensor(state["pphi"]))

    with patch.object(initial_conditions, "find_quasicircular_initial_conditions", substitute):
        candidate = _run(row, prior["level"], q.Settings(**prior["settings"]),
                         prior["numerical_choice"])
    if not replacement:
        raise ValueError("native warm-start intervention did not run")
    native_time, native_mode, reference_time, reference_mode, mass_seconds = public_mode(
        row, candidate)
    comparison = fitted_metrics(native_time, native_mode, reference_time,
                                reference_mode, mass_seconds, "quadratic_power")
    baseline = peaks["results"][index]["fitted_peaks"]["quadratic_power"]["metrics"]
    lal_state = lal_ablation["results"][index]["intervention"]["fitted_peak"]["metrics"]
    return dict(row=row, native_pre_evolution_state=state,
                original_native_initial=replacement["native"],
                native=q.summarize_run(candidate), fitted_peak=comparison,
                baseline_complex_relative_l2=baseline["complex_relative_l2"],
                lal_state_complex_relative_l2=lal_state["complex_relative_l2"],
                native_warmstart_complex_relative_l2=comparison["metrics"]["complex_relative_l2"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    names = ("prior_whole_waveform", "prior_peak_anchor", "native_pre_evolution",
             "lal_state_ablation")
    paths = {key: q.ROOT / plan[key] for key in names}
    hashes = ("prior_sha256", "prior_peak_anchor_sha256",
              "native_pre_evolution_sha256", "lal_state_ablation_sha256")
    if (plan["schema"] != "diffgw.seobnrv4.native-warmstart-whole-waveform-plan.v1"
            or plan["rows"] != [0, 1]
            or any(q.sha256(paths[key]) != plan[hash_key]
                   for key, hash_key in zip(names, hashes))):
        parser.error("frozen native warm-start plan or source hash mismatch")
    prior, peaks, pre_evolution, lal_ablation = (
        json.loads(paths[key].read_text()) for key in names)
    if (any(record["status"] != "diagnostic_complete" for record in
            (prior, peaks, pre_evolution, lal_ablation))
            or peaks["prior_whole_waveform"]["sha256"] != plan["prior_sha256"]
            or any(prior["rows"][i] != record["results"][i]["row"]
                   for record in (peaks, pre_evolution, lal_ablation)
                   for i in plan["rows"])):
        parser.error("incompatible frozen diagnostic rows")
    pin = prior["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("pinned compiled LALSuite binary changed")
    report = dict(schema="diffgw.seobnrv4.native-warmstart-waveform-diagnostic.v1",
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
            result = _row(index, prior, peaks, pre_evolution, lal_ablation)
            result["elapsed_seconds"] = time.monotonic()-started
            report["results"].append(result)
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(index=index, elapsed_seconds=result["elapsed_seconds"],
                                  baseline_l2=result["baseline_complex_relative_l2"],
                                  lal_state_l2=result["lal_state_complex_relative_l2"],
                                  native_warmstart_l2=result["native_warmstart_complex_relative_l2"])),
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
