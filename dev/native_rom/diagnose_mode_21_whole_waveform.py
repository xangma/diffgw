"""Frozen whole-waveform sensitivity to the published (2,1) rho hybrid.

This is a development diagnostic, not a release or historical-parity gate.
Both modes use the same pinned numerical choices and public binary. Their
amplitude peaks are found independently; no relative fit is optimized.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import torch

from . import qualify_generator as q
from .compare_whole_waveforms import aligned_metrics
from .diagnose_peak_anchor import fitted_metrics, public_mode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    if plan.get("schema") != "diffgw.seobnrv4.mode21-whole-waveform-plan.v1":
        parser.error("unsupported frozen plan")
    prior_path = q.ROOT / plan["prior_whole_waveform"]
    if q.sha256(prior_path) != plan["prior_sha256"]:
        parser.error("prior whole-waveform evidence hash differs from plan")
    prior = json.loads(prior_path.read_text())
    index = plan["row_index"]
    if (prior.get("schema") != "diffgw.whole-waveform-diagnostic.v1"
            or prior.get("status") != "diagnostic_complete"
            or index != 1 or len(prior["rows"]) <= index
            or prior["rows"][index] != prior["results"][index]["row"]
            or plan["baseline_mode_21_higher_spin"] != "taylor_f"
            or plan["candidate_mode_21_higher_spin"] != "rho_hybrid"
            or plan["peak_fit"] != "quadratic_power"):
        parser.error("incompatible frozen plan or prior evidence")
    pin = prior["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("public binary identity differs from frozen comparison")
    row, level = prior["rows"][index], prior["level"]
    settings, choice = q.Settings(**prior["settings"]), prior["numerical_choice"]
    torch.set_num_threads(1)
    sources = ["src/torchwave/approximants/eob/seobnrv4_flux.py",
               "dev/native_rom/eob/torch/adaptive_driver_clean.py",
               "dev/native_rom/qualify_generator.py", "dev/native_rom/diagnose_peak_anchor.py"]
    output = dict(
        schema="diffgw.seobnrv4.mode21-whole-waveform-diagnostic.v1",
        scope=q.SCOPE, status="running", historical_parity_established=False,
        whole_waveform_parity_validated=False, release_qualified=False,
        plan=dict(path=str(args.plan), sha256=q.sha256(args.plan)),
        prior=dict(path=str(prior_path), sha256=q.sha256(prior_path)),
        lal_binary=dict(lal_version=pin["lal_version"],
                        lalsimulation_version=pin["lalsimulation_version"],
                        distribution=pin["lalsuite_distribution"],
                        binary_manifest_sha256=q.canonical_hash(pin["binaries"])),
        source_sha256={path: q.sha256(q.ROOT / path) for path in sources},
        script_sha256=q.sha256(__file__), row=row, level=level,
        settings=asdict(settings), numerical_choice=choice,
        process=dict(host=socket.gethostname(), cwd=str(Path.cwd()),
                     pid=os.getpid(), argv=sys.argv), results={}, errors=[])
    q.write_evidence(args.json_out, output)
    try:
        for label, mode_21 in (("baseline", "taylor_f"), ("candidate", "rho_hybrid")):
            print(f"{label}: mode_21_higher_spin={mode_21}", flush=True)
            start = time.monotonic()
            run = q.native_run(
                row, level, settings,
                diagnostic_hermite_resampling=choice["hermite_resampling"],
                diagnostic_local_nqc_derivatives=choice["local_nqc_phase_derivatives"],
                diagnostic_restart_time_M=choice["fixed_restart_time_M"],
                diagnostic_mode_21_higher_spin=mode_21)
            native = q.summarize_run(run)
            if label == "baseline" and native["array_sha256"] != prior["results"][index]["native"]["array_sha256"]:
                raise ValueError("baseline waveform differs from frozen prior")
            nt, nm, rt, rm, mass_seconds = public_mode(row, run)
            sampled = aligned_metrics(nt, nm, rt, rm, mass_seconds)
            if label == "baseline":
                old = prior["results"][index]["comparison"]["amplitude_peak_anchored"]["metrics"]
                if not math.isclose(sampled["metrics"]["complex_relative_l2"],
                                    old["complex_relative_l2"], rel_tol=0, abs_tol=1e-12):
                    raise ValueError("baseline sampled-peak metric differs from frozen prior")
            output["results"][label] = dict(
                mode_21_higher_spin=mode_21, native=native, sampled_peak=sampled,
                fitted_peak=fitted_metrics(nt, nm, rt, rm, mass_seconds, plan["peak_fit"]),
                elapsed_seconds=time.monotonic()-start)
            q.write_evidence(args.json_out, output)
        a = output["results"]["baseline"]["fitted_peak"]["metrics"]
        b = output["results"]["candidate"]["fitted_peak"]["metrics"]
        output["sensitivity"] = {
            key: dict(baseline=a[key], candidate=b[key], difference=b[key]-a[key])
            for key in ("complex_relative_l2", "amplitude_relative_l2",
                        "phase_rms_rad", "unmaximized_real_overlap")}
        output["status"] = "diagnostic_complete"
    except Exception as exc:
        output["status"] = "failed"
        output["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, output)
    print(json.dumps(dict(status=output["status"],
                          sensitivity=output.get("sensitivity"), errors=output["errors"])),
          flush=True)
    return 0 if output["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
