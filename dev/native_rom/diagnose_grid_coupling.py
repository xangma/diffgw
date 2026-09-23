"""Development-only controlled grid/restart diagnostic; no model qualification.

Run from the repository root with ``PYTHONPATH=src:. python -m
dev.native_rom.diagnose_grid_coupling --json-out /tmp/grid.json``.
"""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import numpy as np
import torch

from dev.native_rom.eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
from dev.native_rom.qualify_generator import (
    canonical_hash, compare_runs, sha256, source_snapshot,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--json-out", required=True, type=Path)
args = parser.parse_args()

def snapshot():
    sources = source_snapshot()
    sources[str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[2]))] = sha256(__file__)
    return canonical_hash(sources)

source_before = snapshot()

torch.set_num_threads(1)
specs = [
    ("baseline", 0.5, None, None),
    ("fine_default", 0.125, None, None),
    ("fine_fixed_initial", 0.125, 0.5, None),
    ("fine_fixed_restart", 0.125, None, 165.0),
    ("fine_both_fixed", 0.125, 0.5, 165.0),
    ("medium_both_fixed", 0.25, 0.5, 165.0),
]
runs = {}
comparisons = {}
for name, dt, initial, restart in specs:
    before = time.monotonic()
    with torch.no_grad():
        result = reconstruct_seobnrv4_22_adaptive(
            torch.tensor(0.04, dtype=torch.float64), 30.0, 20.0,
            chi1=0.0, chi2=0.0, dt_geom=dt, atol=1e-14, rtol=1e-13,
            diagnostic_coarse_initial_step_M=initial,
            diagnostic_restart_time_M=restart,
        )
    def scalar(x):
        return float(x.detach().cpu())
    run = dict(
        name=name, dt_geom=dt, time=result.time.detach().cpu().numpy(),
        mode=result.mode.detach().cpu().numpy(),
        peak_time_M=scalar(result.trajectory.peak.time),
        match_time_M=scalar(result.attachment.match_time),
        boundary_time_M=scalar(result.attachment.boundary_time),
        restart_time_M=scalar(result.trajectory.restart_time),
        coarse_event_time_M=scalar(result.trajectory.coarse.statistics.event_time),
        fine_event_time_M=scalar(result.trajectory.fine.statistics.event_time),
    )
    runs[name] = run
    print(json.dumps({"run": name, "seconds": time.monotonic()-before,
                      "samples": len(run["time"]),
                      "restart_time_M": run["restart_time_M"],
                      "coarse_event_time_M": run["coarse_event_time_M"],
                      "fine_event_time_M": run["fine_event_time_M"],
                      "peak_time_M": run["peak_time_M"]}), flush=True)
    if name != "baseline":
        metrics = compare_runs(run, runs["baseline"])
        comparisons[name] = metrics
        print(json.dumps({"comparison": name, "complex_relative_l2": metrics["complex_relative_l2"],
                          "amplitude_relative_l2": metrics["amplitude_relative_l2"],
                          "peak_time_M_abs_difference": metrics["peak_time_M_abs_difference"],
                          "match_time_M_abs_difference": metrics["match_time_M_abs_difference"],
                          "support_end_difference_M": metrics["support_end_difference_M"]}), flush=True)
comparisons["medium_to_fine_both_fixed"] = compare_runs(
    runs["fine_both_fixed"], runs["medium_both_fixed"])
source_after = snapshot()
report = dict(
    schema="diffgw.seobnrv4.grid_coupling_diagnostic.v1",
    scope=dict(physics_validated=False, default_generator_qualified=False,
               development_only=True, no_coefficient_fitting=True),
    row=dict(omega_start=0.04, mass1=30.0, mass2=20.0, chi1=0.0, chi2=0.0),
    tolerances=dict(atol=1e-14, rtol=1e-13),
    cases={name: {key: value for key, value in run.items()
                  if key not in ("time", "mode")}
           | {"samples": len(run["time"])}
           for name, run in runs.items()},
    comparisons=comparisons,
    source_sha256_before=source_before, source_sha256_after=source_after,
    sources_unchanged=source_before == source_after,
    process=dict(argv=sys.argv, cwd=os.getcwd(), host=socket.gethostname(), pid=os.getpid()),
    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    versions=dict(torch=torch.__version__, numpy=np.__version__),
)
args.json_out.parent.mkdir(parents=True, exist_ok=True)
args.json_out.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
print(f"wrote {args.json_out}", flush=True)
if source_before != source_after:
    raise RuntimeError("sources changed during the diagnostic")
