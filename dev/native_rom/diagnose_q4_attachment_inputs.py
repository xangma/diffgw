"""Capture native q4 ringdown inputs and observable public peak-grid proxies."""

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
from .diagnose_peak_anchor import fitted_peak, public_mode
from .diagnose_q8_mode_phase import _replay


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/q4_attachment_inputs_plan.json"


def _sampled_peak_proxy(time_s, mode, mass_seconds):
    peak_index = int(np.argmax(abs(mode)))
    if peak_index < 1:
        raise ValueError("peak has no preceding sample")
    phase = -np.unwrap(np.angle(mode))
    step_M = float((time_s[peak_index]-time_s[peak_index-1])/mass_seconds)
    return dict(index=peak_index, time_s=float(time_s[peak_index]),
                amplitude=float(abs(mode[peak_index])),
                phase_rad=float(phase[peak_index]),
                backward_frequency_M=float((phase[peak_index]-phase[peak_index-1])/step_M),
                output_step_M=step_M)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    campaign_path = q.ROOT / plan["campaign"]
    profile_path = q.ROOT / plan["postpeak_profile"]
    if (plan["schema"] != "diffgw.seobnrv4.q4-attachment-inputs-plan.v1"
            or plan["row_name"] != "q4_mixed_spin"
            or plan["level_name"] != "ode_2"
            or plan["public_peak_proxy"] != "largest output-grid amplitude sample"
            or plan["production_change"] is not False
            or plan["release_qualified"] is not False
            or q.sha256(campaign_path) != plan["campaign_sha256"]
            or q.sha256(profile_path) != plan["postpeak_profile_sha256"]):
        parser.error("frozen plan or inputs differ")
    campaign = json.loads(campaign_path.read_text())
    profile = json.loads(profile_path.read_text())
    item = campaign["rows"][plan["row_name"]]
    row, choice = item["row"], campaign["choice"]
    if (campaign["status"] != "diagnostic_complete"
            or profile["status"] != "diagnostic_complete"
            or profile["row"] != row or profile["choice"] != choice):
        parser.error("frozen campaign or profile differs")
    torch.set_num_threads(1)
    source_paths = ("dev/native_rom/qualify_generator.py",
                    "dev/native_rom/diagnose_q8_mode_phase.py",
                    "src/torchwave/approximants/eob/numerical_attachment.py",
                    "src/torchwave/approximants/eob/merger_ringdown.py")
    report = dict(schema="diffgw.seobnrv4.q4-attachment-inputs.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  campaign_sha256=q.sha256(campaign_path),
                  postpeak_profile_sha256=q.sha256(profile_path),
                  source_sha256={path: q.sha256(q.ROOT / path) for path in source_paths},
                  row=row, choice=choice, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        run = _replay(row, item["runs"]["ode_2"]["level"],
                      q.Settings(**campaign["settings"]), choice,
                      item["fixed_restart_time_M"], return_boundary_data=True)
        summary = q.summarize_run(run)
        if summary["array_sha256"] != item["runs"]["ode_2"]["native"]["array_sha256"]:
            raise ValueError("native waveform differs from frozen campaign")
        report["native"] = dict(array_sha256=summary["array_sha256"],
                                elapsed_seconds=time.monotonic()-started,
                                boundary_time_M=run["boundary_time_M"],
                                match_time_M=run["match_time_M"],
                                boundary_inputs=run["boundary_data"])
        q.write_evidence(args.json_out, report)
        print("native_waveform_replayed", flush=True)
        nt, nm, rt, rm, mass_seconds = public_mode(row, run)
        native_peak = fitted_peak(nt, nm, "quadratic_power")
        public_peak = fitted_peak(rt, rm, "quadratic_power")
        report["sampled_peak_proxies"] = dict(
            native=_sampled_peak_proxy(nt, nm, mass_seconds),
            public=_sampled_peak_proxy(rt, rm, mass_seconds),
            native_fitted_peak_time_s=native_peak["time_s"],
            public_fitted_peak_time_s=public_peak["time_s"],
            limitation="Public output-grid samples do not expose the hidden high-rate LAL ringdown boundary; their backward frequency is not an attachment input.")
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
