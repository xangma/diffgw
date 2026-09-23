"""Replay the candidate's tight-grid waveforms for peak-anchored comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import torch

from . import qualify_generator as q
from .compare_whole_waveforms import aligned_metrics
from .diagnose_peak_anchor import fitted_metrics, public_mode


BASELINE = Path("dev/native_rom/evidence/2026-09-23/peak_anchor_diagnostic.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = json.loads(args.campaign.read_text())
    baseline = json.loads(BASELINE.read_text())
    if (campaign.get("schema") != "diffgw.seobnrv4.nqc-second-derivative-campaign.v1"
            or not campaign.get("sources_unchanged")
            or len(campaign.get("rows", {})) != 2
            or baseline.get("lal_binary") != campaign.get("lal_binary")):
        parser.error("candidate campaign or public-binary pin differs")
    torch.set_num_threads(1)
    settings = q.Settings(atol_levels=tuple(campaign["settings"]["atol_levels"]),
                          rtol_levels=tuple(campaign["settings"]["rtol_levels"]))
    before = q.source_snapshot()
    if before != {key: value for key, value in campaign["source_sha256_before"].items()
                  if key in before}:
        parser.error("candidate generator sources differ from campaign")
    if q.lal_binary_identity("7.7.1", "6.2.1") != campaign["lal_binary"]:
        parser.error("public binary differs from campaign")
    output = dict(schema="diffgw.seobnrv4.nqc-second-derivative-waveforms.v1",
                  scope=dict(**q.SCOPE, development_only=True,
                             whole_waveform_parity_validated=False,
                             numerical_generator_qualified=False),
                  campaign_sha256=q.sha256(args.campaign),
                  baseline_sha256=q.sha256(BASELINE),
                  source_sha256_before=before,
                  process=dict(argv=sys.argv, cwd=os.getcwd(),
                               host=socket.gethostname(), pid=os.getpid()),
                  rows={}, status="running")
    q.write_evidence(args.json_out, output)
    for name, result in campaign["rows"].items():
        level = next(run for run in result["runs"] if run["name"] == "grid_2")
        row = result["inputs"]
        run = q.native_run(
            row, {key: level[key] for key in ("name", "atol", "rtol", "dt_geom")}, settings,
            diagnostic_hermite_resampling=True,
            diagnostic_local_nqc_derivatives=True,
            diagnostic_matching_second_derivatives=True,
            diagnostic_restart_time_M=165.)
        hashes = q.summarize_run(run)["array_sha256"]
        if hashes != level["array_sha256"]:
            raise ValueError(f"{name} candidate waveform differs from campaign")
        nt, nm, rt, rm, mass_seconds = public_mode(row, run)
        prior = next(item for item in baseline["results"] if item["row"] == row)
        output["rows"][name] = dict(
            row=row, hashes=hashes,
            candidate=dict(sampled_peak=aligned_metrics(nt, nm, rt, rm, mass_seconds),
                           quadratic_power=fitted_metrics(nt, nm, rt, rm, mass_seconds,
                                                          "quadratic_power")),
            frozen_baseline=dict(quadratic_power=prior["fitted_peaks"]["quadratic_power"]["metrics"]))
        q.write_evidence(args.json_out, output)
        print(f"row={name} hash_verified=True", flush=True)
    output["source_sha256_after"] = q.source_snapshot()
    output["sources_unchanged"] = before == output["source_sha256_after"]
    output["status"] = "diagnostic_complete" if output["sources_unchanged"] else "failed"
    q.write_evidence(args.json_out, output)
    print(json.dumps(dict(status=output["status"], evidence=str(args.json_out.resolve()))),
          flush=True)
    return 0 if output["sources_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
