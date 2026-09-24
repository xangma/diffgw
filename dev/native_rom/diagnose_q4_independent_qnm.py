"""Recompute the frozen q4 mode with the independently generated Kerr QNM."""

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
from .diagnose_peak_anchor import fitted_metrics, public_mode
from .diagnose_q8_mode_phase import _replay
from .diagnose_q8_ringdown_peak_bound import _frequency_metrics


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/q4_independent_qnm_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {key: q.ROOT / plan[key] for key in (
        "campaign", "baseline_frequency_anchor", "independent_qnm_source",
        "compatibility_qnm_source")}
    if (plan["schema"] != "diffgw.seobnrv4.q4-independent-qnm-plan.v1"
            or plan["row_name"] != "q4_mixed_spin"
            or plan["level_name"] != "ode_2"
            or plan["anchor_angular_frequency_M"] != .1
            or plan["peak_fit"] != "quadratic_power"
            or plan["production_change"] is not False
            or plan["release_qualified"] is not False
            or any(q.sha256(path) != plan[key + "_sha256"] for key, path in paths.items())):
        parser.error("frozen plan or inputs differ")
    campaign = json.loads(paths["campaign"].read_text())
    baseline = json.loads(paths["baseline_frequency_anchor"].read_text())
    item = campaign["rows"][plan["row_name"]]
    row, choice = item["row"], campaign["choice"]
    pin = campaign["lal_binary"]
    if (campaign["status"] != "diagnostic_complete"
            or baseline["status"] != "diagnostic_complete"
            or baseline["row"] != row or baseline["choice"] != choice
            or q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin):
        parser.error("frozen q4 row or public binary differs")
    torch.set_num_threads(1)
    source_paths = ("dev/native_rom/eob/torch/adaptive_driver_clean.py",
                    "dev/native_rom/qualify_generator.py",
                    "dev/native_rom/diagnose_q8_mode_phase.py",
                    "dev/native_rom/diagnose_q8_ringdown_peak_bound.py")
    report = dict(schema="diffgw.seobnrv4.q4-independent-qnm.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256={key: q.sha256(path) for key, path in paths.items()},
                  source_sha256={path: q.sha256(q.ROOT / path) for path in source_paths},
                  lal_binary=pin, row=row, choice=choice, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        from torchwave.approximants import kerr_qnm_table_convention as compatibility
        from torchwave.approximants.kerr_qnm_generated import signed_qnm_220

        started = time.monotonic()
        with patch.object(compatibility, "signed_qnm_220_table_convention", signed_qnm_220):
            run = _replay(row, item["runs"]["ode_2"]["level"],
                          q.Settings(**campaign["settings"]), choice,
                          item["fixed_restart_time_M"], return_boundary_data=True)
        summary = q.summarize_run(run)
        if (summary["array_sha256"] == item["runs"]["ode_2"]["native"]["array_sha256"]
                or summary["final_spin"] != item["runs"]["ode_2"]["native"]["final_spin"]
                or summary["final_mass_fraction"] != item["runs"]["ode_2"]["native"]["final_mass_fraction"]):
            raise ValueError("independent QNM did not isolate the frozen native model")
        report["native"] = dict(array_sha256=summary["array_sha256"],
                                elapsed_seconds=time.monotonic()-started,
                                final_spin=summary["final_spin"],
                                final_mass_fraction=summary["final_mass_fraction"],
                                boundary_time_M=run["boundary_time_M"],
                                boundary_inputs=run["boundary_data"])
        q.write_evidence(args.json_out, report)
        print("independent_qnm_waveform_complete", flush=True)
        nt, nm, rt, rm, mass_seconds = public_mode(row, run)
        report["peak_fitted"] = fitted_metrics(nt, nm, rt, rm, mass_seconds, plan["peak_fit"])
        report["frequency_anchored"] = _frequency_metrics(
            nt, nm, rt, rm, mass_seconds, plan["anchor_angular_frequency_M"])
        report["baseline"] = dict(
            peak_fitted_complex_relative_l2=baseline["peak_fitted"]["metrics"]["complex_relative_l2"],
            frequency_anchored_complex_relative_l2=baseline["frequency_anchored"]["metrics"]["complex_relative_l2"],
            frequency_anchored_postpeak_complex_relative_l2=baseline["frequency_anchored"]["windows"]["post_peak"]["complex_relative_l2"])
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
