"""Check inline native frequency warm starts against frozen native evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import torch

from . import qualify_generator as q
from .diagnose_peak_anchor import fitted_metrics, public_mode
from .eob.torch.warm_start import native_frequency_warm_start


EVIDENCE = q.ROOT / "dev/native_rom/evidence/2026-09-23"
PLAN = EVIDENCE / "inline_warmstart_plan.json"
COORDINATES = ("r", "phi", "pr_star", "pphi")


def _row(index, pre, frozen, prior, plan):
    from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
    from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

    row = pre["results"][index]["row"]
    m1, m2, s1, s2 = (torch.tensor(row[key], dtype=torch.float64)
                      for key in ("mass1", "mass2", "chi1", "chi2"))
    x1, x2 = m1/(m1+m2), m2/(m1+m2)
    ham = SEOBNRv4Hamiltonian(x1*x2, s1, s2, delta=x1-x2)
    flux = SEOBNRv4PublishedFlux(spin_truncation=q.CHOICES["spin_truncation"],
                                 retain_pan_cubic_spin=True)
    choices = plan["integration"]
    with torch.no_grad():
        warm = native_frequency_warm_start(
            ham, flux, torch.tensor(plan["native_start_Momega"], dtype=torch.float64),
            torch.tensor(row["omega_start"], dtype=torch.float64),
            atol=choices["atol"], rtol=choices["rtol"],
            first_step_M=choices["first_step_M"],
            frequency_tolerance=choices["crossing_frequency_tolerance"])
    expected = pre["results"][index]["native_cutoff_crossing_state"]
    difference = {key: float(warm.state[i])-expected[key]
                  for i, key in enumerate(COORDINATES) if key != "phi"}
    crossing_time_difference = (float(warm.crossing_time)
                                - pre["results"][index]["native_cutoff_crossing_time_M"])
    level, choice = prior["level"], prior["numerical_choice"]
    started = time.monotonic()
    run = q.native_run(
        row, level, q.Settings(**prior["settings"]),
        diagnostic_hermite_resampling=choice["hermite_resampling"],
        diagnostic_local_nqc_derivatives=choice["local_nqc_phase_derivatives"],
        diagnostic_restart_time_M=choice["fixed_restart_time_M"],
        diagnostic_warm_start_Momega=plan["native_start_Momega"])
    native_time, native_mode, reference_time, reference_mode, mass_seconds = public_mode(
        row, run)
    comparison = fitted_metrics(native_time, native_mode, reference_time,
                                reference_mode, mass_seconds, "quadratic_power")
    l2 = comparison["metrics"]["complex_relative_l2"]
    frozen_l2 = frozen["results"][index]["native_warmstart_complex_relative_l2"]
    gates = plan["diagnostic_gates"]
    checks = dict(
        state=max(abs(value) for value in difference.values())
        <= gates["max_abs_state_difference_from_native_DOP853"],
        crossing_time=abs(crossing_time_difference)
        <= gates["max_abs_crossing_time_difference_M"],
        frequency=abs(float(warm.frequency_residual))
        <= gates["max_abs_frequency_residual"],
        waveform=abs(l2-frozen_l2)
        <= gates["max_absolute_complex_relative_l2_change_from_frozen_intervention"])
    return dict(row=row, warm_start=dict(
                    initial_state=dict(zip(COORDINATES, warm.initial_state.tolist())),
                    crossing_state=dict(zip(COORDINATES, warm.state.tolist())),
                    crossing_time_M=float(warm.crossing_time),
                    accepted_steps=warm.accepted_steps, evaluations=warm.evaluations,
                    frequency_residual=float(warm.frequency_residual)),
                native_rkf45_minus_native_DOP853=difference,
                crossing_time_difference_M=crossing_time_difference,
                waveform=q.summarize_run(run), fitted_peak=comparison,
                frozen_intervention_l2=frozen_l2,
                inline_l2=l2, diagnostic_gates=checks,
                diagnostic_passed=all(checks.values()),
                waveform_elapsed_seconds=time.monotonic()-started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    pre_path = q.ROOT / plan["native_pre_evolution"]
    frozen_path = q.ROOT / plan["frozen_warmstart_waveform"]
    if (plan["schema"] != "diffgw.seobnrv4.inline-warmstart-plan.v1"
            or plan["rows"] != [0, 1]
            or q.sha256(pre_path) != plan["native_pre_evolution_sha256"]
            or q.sha256(frozen_path) != plan["frozen_warmstart_waveform_sha256"]):
        parser.error("inline warm-start plan or inputs differ")
    pre, frozen = json.loads(pre_path.read_text()), json.loads(frozen_path.read_text())
    old_plan_path = EVIDENCE / "native_warmstart_whole_waveform_plan.json"
    if q.sha256(old_plan_path) != frozen["plan_sha256"]:
        parser.error("frozen warm-start plan changed")
    old_plan = json.loads(old_plan_path.read_text())
    prior_path = q.ROOT / old_plan["prior_whole_waveform"]
    if q.sha256(prior_path) != old_plan["prior_sha256"]:
        parser.error("prior whole waveform changed")
    prior = json.loads(prior_path.read_text())
    if (any(record["status"] != "diagnostic_complete"
            for record in (pre, frozen, prior))
            or pre["results"][0]["row"] != prior["rows"][0]
            or pre["results"][1]["row"] != prior["rows"][1]):
        parser.error("incompatible prior evidence")
    pin = prior["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("pinned compiled LALSuite binary changed")
    report = dict(schema="diffgw.seobnrv4.inline-warmstart-diagnostic.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  whole_waveform_parity_validated=False,
                  plan_sha256=q.sha256(PLAN),
                  input_sha256=dict(pre=q.sha256(pre_path), frozen=q.sha256(frozen_path),
                                    prior=q.sha256(prior_path)),
                  script_sha256=q.sha256(__file__),
                  source_sha256={path: q.sha256(q.ROOT / path) for path in (
                      "dev/native_rom/eob/torch/warm_start.py",
                      "dev/native_rom/eob/torch/adaptive_driver_clean.py",
                      "src/torchwave/approximants/eob/initial_conditions.py",
                      "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
                      "src/torchwave/approximants/eob/seobnrv4_flux.py")},
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), results=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        for index in plan["rows"]:
            result = _row(index, pre, frozen, prior, plan)
            report["results"].append(result)
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(index=index, inline_l2=result["inline_l2"],
                                  frozen_l2=result["frozen_intervention_l2"],
                                  diagnostic_gates=result["diagnostic_gates"])), flush=True)
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
