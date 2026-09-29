"""Run the frozen two-row gates with one opt-in continuous NQC candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

from dev.native_rom import qualify_generator as q
from dev.native_rom.eob.torch import adaptive_driver_clean as driver


ROOT = Path(__file__).resolve().parents[5]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-29"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", choices=("nonspin", "spin"), required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    receipt_path = EVIDENCE / "ode_gate_slice" / f"rerun_{args.row}.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt["config"]["scientific_choices"]["spin_truncation"] != "tabulated"
            or receipt["results"][0]["inputs"] != {
                "mass1": 30.0, "mass2": 20.0,
                "chi1": 0.0 if args.row == "nonspin" else 0.2,
                "chi2": 0.0 if args.row == "nonspin" else -0.1,
                "omega_start": 0.04,
            }):
        parser.error("frozen row or choices changed")
    settings = q.Settings(**receipt["config"]["settings"])
    plans = {item["name"]: item for item in q.run_plan(settings)}
    prior = {item["name"]: item for item in receipt["results"][0]["runs"]}
    if set(plans) != set(prior):
        parser.error("frozen run levels changed")
    for name, plan in plans.items():
        if any(plan[key] != prior[name][key] for key in ("atol", "rtol", "dt_geom")):
            parser.error("frozen run tolerances or grid changed")
    baseline = json.loads((EVIDENCE / "ode_chain_derivatives" / f"{args.row}.json").read_text())
    if baseline["script_sha256"] != q.sha256(EVIDENCE / "ode_chain_derivatives" / "diagnose.py"):
        parser.error("chain-rule diagnostic script changed")
    row = receipt["results"][0]["inputs"]
    source_before = q.source_snapshot()
    capture = []
    original = driver.attach_seobnrv4_22_numerical_compatibility

    def capture_baseline(*positional, **keyword):
        result = original(*positional, **keyword)
        if not keyword:
            coefficients = result.nqc.calibration.coefficients
            capture.append(dict(
                amplitude=coefficients.amplitude.detach().cpu().numpy().tolist(),
                phase=coefficients.phase.detach().cpu().numpy().tolist(),
                match_time_M=float(result.match_time.detach())))
        return result

    driver.attach_seobnrv4_22_numerical_compatibility = capture_baseline
    raw = {}
    rows = []
    output = dict(schema="diffgw.seobnrv4.ode-chain-gate.v1",
                  status="running", scope=dict(q.SCOPE, development_only=True), row=row,
                  rule="ODE chain-rule first and second derivatives at continuous match state; spline values retained",
                  plan_sha256=q.sha256(EVIDENCE / "ode_chain_derivatives" / "PLAN.md"),
                  diagnostic_sha256=q.sha256(EVIDENCE / "ode_chain_derivatives" / f"{args.row}.json"),
                  frozen_receipt_sha256=q.sha256(receipt_path),
                  source_sha256_before=source_before,
                  source_manifest_sha256=q.canonical_hash(source_before),
                  baseline_checks=receipt["results"][0]["checks"],
                  baseline_runs={name: prior[name] for name in plans},
                  runs=rows,
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv))
    # Check the exact match-time coefficient transformation before spending
    # time on the remaining four levels; the gate definitions are unchanged.
    order = ("ode_2", "ode_0", "ode_1", "grid_1", "grid_2")
    try:
        for name in order:
            capture.clear()
            start = time.monotonic()
            run = q.native_run(
                row, plans[name], settings,
                diagnostic_spin_truncation="tabulated",
                diagnostic_chain_rule_nqc_derivatives=True,
                return_nqc_coefficients=True, return_accepted_hashes=True)
            if len(capture) != 1:
                raise ValueError(f"expected one unchanged provisional baseline for {name}")
            baseline_coefficient = capture[0]
            if name == "ode_2":
                for label in ("amplitude", "phase"):
                    old = baseline["native"]["spline_system"][label + "_coefficients"]
                    new = baseline["native"]["chain_system"][label + "_coefficients"]
                    if np.max(np.abs(np.asarray(baseline_coefficient[label]) - old)) > 1e-8:
                        raise ValueError("provisional coefficient does not replay frozen spline solve")
                    if np.max(np.abs(np.asarray(run["nqc_coefficients"][label]) - new)) > 1e-7:
                        raise ValueError("candidate coefficient does not replay validated chain solve")
            if abs(baseline_coefficient["match_time_M"] + run["restart_time_M"] - run["match_time_M"]) > 1e-8:
                raise ValueError("candidate match time differs from provisional baseline")
            raw[name] = run
            summary = q.summarize_run(run)
            summary["baseline_nqc_coefficients"] = {
                key: baseline_coefficient[key] for key in ("amplitude", "phase")}
            summary["baseline_array_sha256"] = prior[name]["array_sha256"]
            summary["candidate_seconds"] = time.monotonic()-start
            rows.append(summary)
            q.write_evidence(args.json_out, output)
            print(json.dumps(dict(run=name, seconds=summary["candidate_seconds"],
                                  baseline_nqc=summary["baseline_nqc_coefficients"],
                                  candidate_nqc=summary["nqc_coefficients"])), flush=True)
        ode = q.convergence_check(raw["ode_0"], raw["ode_1"], raw["ode_2"], settings)
        grid = q.convergence_check(raw["ode_2"], raw["grid_1"], raw["grid_2"], settings)
        output["candidate_checks"] = dict(ode=ode, grid=grid)
        output["candidate_numerical_checks_passed"] = (
            ode["numerical_check_passed"] and grid["numerical_check_passed"])
        output["source_sha256_after"] = q.source_snapshot()
        output["sources_unchanged"] = source_before == output["source_sha256_after"]
        output["status"] = "diagnostic_complete"
        q.write_evidence(args.json_out, output)
        if not output["sources_unchanged"]:
            raise ValueError("candidate source changed during gate")
        print(json.dumps(dict(status=output["status"], gates={
            name: check["gates"] for name, check in output["candidate_checks"].items()})),
              flush=True)
    finally:
        driver.attach_seobnrv4_22_numerical_compatibility = original


if __name__ == "__main__":
    main()
