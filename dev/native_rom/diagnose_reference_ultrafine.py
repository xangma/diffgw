"""Check public SEOBNRv4 grid self-consistency below 0.03125 M."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

from . import qualify_generator as q
from .diagnose_reference_grid import compare, public_mode


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/reference_ultrafine_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_reference_grid"]
    prior = json.loads(prior_path.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.reference-ultrafine-plan.v1"
            or q.sha256(prior_path) != plan["prior_reference_grid_sha256"]
            or prior["status"] != "diagnostic_complete"
            or prior["row"] != plan["row"]
            or plan["steps_M"] != [0.03125, 0.0078125, 0.001953125]
            or q.lal_binary_identity(prior["lal_binary"]["lal_version"],
                                     prior["lal_binary"]["lalsimulation_version"])
            != prior["lal_binary"]):
        parser.error("frozen plan or public binary differs")
    report = dict(schema="diffgw.seobnrv4.reference-ultrafine.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  whole_waveform_parity_validated=False, plan_sha256=q.sha256(PLAN),
                  prior_sha256=q.sha256(prior_path), script_sha256=q.sha256(__file__),
                  lal_binary=prior["lal_binary"], process=dict(
                      host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
                  row=plan["row"], result=None, errors=[])
    q.write_evidence(args.json_out, report)
    start = time.monotonic()
    try:
        modes = [public_mode(plan["row"], step) for step in plan["steps_M"]]
        if len({item[2] for item in modes}) != 1:
            raise ValueError("mass seconds differ")
        result = {}
        for i in range(len(modes) - 1):
            if not np.isfinite(modes[i][1]).all() or not np.isfinite(modes[i+1][1]).all():
                raise ValueError("nonfinite public mode")
            result[f"{plan['steps_M'][i]}_to_{plan['steps_M'][i+1]}"] = compare(
                modes[i][0], modes[i][1], modes[i+1][0], modes[i+1][1],
                modes[i][2], plan["anchors"], plan["windows_M"])
        report["result"] = dict(comparisons=result,
                                sample_counts=[int(len(item[0])) for item in modes],
                                elapsed_seconds=time.monotonic() - start)
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
