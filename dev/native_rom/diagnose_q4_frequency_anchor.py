"""Compare the frozen q4 mixed-spin waveform at an independent frequency anchor."""

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
from .diagnose_peak_anchor import fitted_metrics, public_mode
from .diagnose_q8_mode_phase import _replay
from .diagnose_q8_ringdown_peak_bound import _frequency_metrics


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/q4_frequency_anchor_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    campaign_path = q.ROOT / plan["campaign"]
    if (plan["schema"] != "diffgw.seobnrv4.q4-frequency-anchor-plan.v1"
            or plan["row_name"] != "q4_mixed_spin"
            or plan["level_name"] != "ode_2"
            or plan["anchor_angular_frequency_M"] != 0.1
            or plan["peak_fit"] != "quadratic_power"
            or plan["release_qualified"] is not False
            or q.sha256(campaign_path) != plan["campaign_sha256"]):
        parser.error("frozen plan or campaign differs")
    campaign = json.loads(campaign_path.read_text())
    item = campaign["rows"][plan["row_name"]]
    row, choice = item["row"], campaign["choice"]
    pin = campaign["lal_binary"]
    if (campaign["status"] != "diagnostic_complete"
            or q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin):
        parser.error("campaign or pinned public binary differs")
    torch.set_num_threads(1)
    source_paths = (
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/qualify_generator.py",
        "dev/native_rom/diagnose_peak_anchor.py",
        "dev/native_rom/diagnose_q8_frequency_anchor.py",
        "dev/native_rom/diagnose_q8_ringdown_peak_bound.py",
        "src/torchwave/approximants/eob/merger_ringdown.py",
    )
    report = dict(schema="diffgw.seobnrv4.q4-frequency-anchor.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  campaign_sha256=q.sha256(campaign_path),
                  source_sha256={path: q.sha256(q.ROOT / path) for path in source_paths},
                  lal_binary=pin, row=row, choice=choice, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        run = _replay(row, item["runs"]["ode_2"]["level"],
                      q.Settings(**campaign["settings"]), choice,
                      item["fixed_restart_time_M"])
        summary = q.summarize_run(run)
        if summary["array_sha256"] != item["runs"]["ode_2"]["native"]["array_sha256"]:
            raise ValueError("native fine waveform differs from frozen campaign")
        report["native"] = dict(array_sha256=summary["array_sha256"],
                                elapsed_seconds=time.monotonic()-started)
        q.write_evidence(args.json_out, report)
        print("native_waveform_replayed", flush=True)
        nt, nm, rt, rm, mass_seconds = public_mode(row, run)
        fitted = fitted_metrics(nt, nm, rt, rm, mass_seconds, plan["peak_fit"])
        expected = item["fitted_peak"]["metrics"]["complex_relative_l2"]
        if not np.isclose(fitted["metrics"]["complex_relative_l2"], expected,
                          atol=1e-12, rtol=0):
            raise ValueError("pinned public comparison changed")
        report["peak_fitted"] = fitted
        report["frequency_anchored"] = _frequency_metrics(
            nt, nm, rt, rm, mass_seconds, plan["anchor_angular_frequency_M"])
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
