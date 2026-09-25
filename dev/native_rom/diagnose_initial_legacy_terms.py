"""Trace the pinned source's legacy and optimized initial-momentum terms."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile

from . import qualify_generator as q
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/initial_legacy_terms_plan.json"
SOURCE = "lalsimulation/lib/LALSimIMRSpinEOBInitialConditions.c"
TERMS = ("dHdpphi", "d2Hdr2", "d2Hdrdpphi", "dEdr", "flux_over_nu",
         "dHdpr_at_0p001", "pr_non_tortoise")
INSERTIONS = (
    ("  dEdr  = - dHdpphi * d2Hdr2 / d2Hdrdpphi;",
     """  dEdr  = - dHdpphi * d2Hdr2 / d2Hdrdpphi;
  diffgw_ic_trace[0] = dHdpphi;
  diffgw_ic_trace[1] = d2Hdr2;
  diffgw_ic_trace[2] = d2Hdrdpphi;
  diffgw_ic_trace[3] = dEdr;"""),
    ("    /*printf( \"Ingredients going into prDot:",
     """    diffgw_ic_trace[4] = flux;
    diffgw_ic_trace[5] = dHdpr;
    /*printf( \"Ingredients going into prDot:"""),
    ("    pSph[0] = rDot / (dHdpr / cartValues[3] );",
     """    pSph[0] = rDot / (dHdpr / cartValues[3] );
    diffgw_ic_trace[6] = pSph[0];"""),
)


def _instrument(path: Path) -> str:
    source = path.read_text()
    for before, after in INSERTIONS:
        if source.count(before) != 1:
            raise ValueError(f"Pinned source trace anchor differs: {before[:48]}")
        source = source.replace(before, after, 1)
    path.write_text(source)
    return q.sha256(path)


def _relative(a: float, b: float, floor: float) -> float:
    return abs(a-b) / max(abs(b), floor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("pinned wheel trace requires macOS")
    plan = json.loads(PLAN.read_text())
    paths = {key: q.ROOT / plan[key] for key in (
        "source_flux_plan", "boundary_evidence", "adapter", "base_adapter")}
    if (plan["schema"] != "diffgw.seobnrv4.initial-legacy-terms-plan.v1"
            or plan["release_qualified"] is not False
            or plan["source_choices"] != ["legacy", "optimized"]
            or q.sha256(__file__) != plan["script_sha256"]
            or any(q.sha256(paths[key]) != plan[f"{key}_sha256"]
                   for key in paths)):
        parser.error("frozen plan or pinned input changed")
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    boundary = json.loads(paths["boundary_evidence"].read_text())
    if (boundary["status"] != "diagnostic_complete"
            or len(boundary["points"]) != 8):
        parser.error("boundary evidence is incomplete")
    report = {
        "schema": "diffgw.seobnrv4.initial-legacy-terms.v1",
        "status": "running", "release_qualified": False, "scope": q.SCOPE,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "terms": TERMS, "points": [], "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        libraries = _libraries(source_plan["pinned_wheel"])
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-ic-terms-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            report["instrumented_source_sha256"] = _instrument(private / SOURCE)
            if report["instrumented_source_sha256"] != plan[
                    "instrumented_source_sha256"]:
                raise ValueError("Instrumented source hash differs")
            library_path = Path(tmp) / "trace.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["adapter"], library_path,
                     source_plan["flux_source_includes"],
                     source_plan["flux_public_includes"])
            library = ctypes.CDLL(str(library_path))
            function = library.diffgw_v4_initial_conditions
            function.argtypes = [ctypes.c_double] * 5 + [
                ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            trace = (ctypes.c_double * len(TERMS)).in_dll(
                library, "diffgw_ic_trace")
            for point in boundary["points"]:
                row = point["physical"]
                sources = {}
                for choice in plan["source_choices"]:
                    for index in range(len(TERMS)):
                        trace[index] = math.nan
                    values = (ctypes.c_double * 3)()
                    rc = function(row["mass1"], row["mass2"], row["chi1"],
                                  row["chi2"], point["frequency_Momega"],
                                  int(choice == "optimized"), values)
                    if rc:
                        raise RuntimeError(f"{choice} source IC failed: {rc}")
                    term_values = dict(zip(TERMS, trace))
                    if not all(math.isfinite(x) for x in term_values.values()):
                        raise ValueError("Missing or nonfinite source trace")
                    expected = point["sources"][choice]["source"]
                    measured = dict(zip(("r", "pr_star", "pphi"), values))
                    differences = {key: _relative(measured[key], expected[key],
                                   plan["relative_denominator_floor"])
                                   for key in expected}
                    if max(differences.values()) > plan["replay_relative_gate"]:
                        raise ValueError(f"Trace changed source IC: {differences}")
                    reconstructed = (
                        -term_values["flux_over_nu"] / term_values["dEdr"]
                        / (term_values["dHdpr_at_0p001"] / 0.001))
                    if _relative(reconstructed,
                                 term_values["pr_non_tortoise"], 0.001) > 1e-12:
                        raise ValueError("Traced momentum terms do not compose")
                    sources[choice] = {
                        "initial_state": measured, "terms": term_values,
                        "replay_relative_difference": differences,
                        "tortoise_ratio": measured["pr_star"] /
                        term_values["pr_non_tortoise"],
                    }
                report["points"].append({
                    "name": point["name"],
                    "frequency_Momega": point["frequency_Momega"],
                    "sources": sources,
                    "legacy_over_optimized": {
                        key: sources["legacy"]["terms"][key]
                        / sources["optimized"]["terms"][key]
                        for key in TERMS},
                })
                q.write_evidence(args.json_out, report)
                print(point["name"], point["frequency_Momega"], flush=True)
        report["max_fractional_branch_change"] = {
            key: max(abs(point["legacy_over_optimized"][key]-1.)
                     for point in report["points"]) for key in TERMS}
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__,
                                 "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({key: report.get(key) for key in (
        "status", "max_fractional_branch_change", "errors")}), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
