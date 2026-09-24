"""Measure pinned-source NQC sensitivity to controlled compiler flags."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import socket
import sys
import tempfile

import numpy as np

from . import qualify_generator as q
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_wheel_mode import _source_nqc


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_nqc_build_controls_plan.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {key: q.ROOT / plan[key] for key in
             ("source_flux_plan", "source_wheel_mode", "adapter")}
    variants = [("O0", "-O0", ()), ("O2", "-O2", ()),
                ("O3", "-O3", ()), ("O2_fast_math", "-O2", ("-ffast-math",))]
    if (plan["schema"] != "diffgw.seobnrv4.source-nqc-build-controls-plan.v1"
            or plan["variants"] != [name for name, _, _ in variants]
            or plan["step_M"] != 0.125
            or any(q.sha256(paths[key]) != plan[key+"_sha256"] for key in paths)):
        parser.error("Frozen plan or inputs changed")
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    prior = json.loads(paths["source_wheel_mode"].read_text())
    if (prior["status"] != "diagnostic_complete"
            or [item["name"] for item in prior["rows"]] != plan["rows"]):
        parser.error("Prior full-mode comparison unavailable")
    libraries = _libraries(source_plan["pinned_wheel"])
    report = dict(schema="diffgw.seobnrv4.source-nqc-build-controls.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256={key: q.sha256(path) for key, path in paths.items()},
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), variants=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-nqc-build-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            for name, optimization, extra_flags in variants:
                dylib = Path(tmp) / f"source_nqc_{name}.dylib"
                _compile(private, args.headers, args.gsl_include, libraries,
                         paths["adapter"], dylib,
                         source_plan["full_generator_source_includes"],
                         source_plan["full_generator_public_includes"],
                         optimization=optimization, extra_flags=extra_flags,
                         require_wheel_flux=False)
                function = ctypes.CDLL(str(dylib)).diffgw_v4_full_source_nqc
                function.argtypes = [ctypes.c_double]*6 + [
                    ctypes.POINTER(ctypes.c_double)]
                function.restype = ctypes.c_int
                rows = []
                for item in prior["rows"]:
                    source = _source_nqc(function, item["row"], plan["step_M"])
                    wheel = np.asarray(item["wheel_nqc_coefficients"])
                    baseline = np.asarray(item["source_nqc_coefficients"])
                    rows.append(dict(name=item["name"],
                                     coefficients=source.tolist(),
                                     minus_O2_baseline=(source-baseline).tolist(),
                                     minus_wheel=(source-wheel).tolist(),
                                     active_l2_distance_from_wheel=float(
                                         np.linalg.norm((source-wheel)[[0, 1, 2, 6, 7]]))))
                report["variants"].append(dict(name=name, optimization=optimization,
                                               extra_flags=list(extra_flags), rows=rows))
                q.write_evidence(args.json_out, report)
                print(f"variant={name}", flush=True)
        if any(np.max(np.abs(row["minus_O2_baseline"])) > 1e-11
               for variant in report["variants"] if variant["name"] == "O2"
               for row in variant["rows"]):
            raise ValueError("O2 baseline did not reproduce")
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])),
          flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
