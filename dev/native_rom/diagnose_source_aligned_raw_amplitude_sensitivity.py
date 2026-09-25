"""Separate raw-mode state and orbital-frequency sensitivity at one NQC event."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np

from . import qualify_generator as q
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_aligned_raw_amplitude_curvature import _key


PLAN = (q.ROOT / "dev/native_rom/evidence/2026-09-25/"
        "source_aligned_raw_amplitude_sensitivity_plan.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    names = ("curvature_evidence", "source_flux_plan", "mode_adapter")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-raw-amplitude-sensitivity-plan.v1"
            or plan["release_qualified"] is not False
            or plan["integrator_pairs"] != [["adaptive_dense_tight", "bounded_dense_tight"],
                                             ["bounded_dense_tight", "bounded_endpoint_tight"]]
            or plan["offsets_M"] != [-0.2, -0.1, 0.0, 0.1, 0.2]
            or plan["center_perturbations"] != {
                "coordinates": ["radius", "radial_momentum", "angular_momentum", "omega"],
                "absolute_steps": [1e-8, 1e-9, 1e-10, 1e-11],
            }
            or any(q.sha256(path) != plan[name + "_sha256"]
                   for name, path in paths.items())):
        parser.error("Frozen sensitivity plan or input changed")
    prior = json.loads(paths["curvature_evidence"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (prior["status"] != "diagnostic_complete"
            or prior["release_qualified"] is not False
            or prior["historical_0p01_gate"] is not False
            or prior["event_M"] != 137.59240346159578):
        parser.error("Frozen curvature result changed")
    row = prior["row"]
    report = {
        "schema": "diffgw.seobnrv4.source-aligned-raw-amplitude-sensitivity.v1",
        "scope": dict(**q.SCOPE, development_only=True,
                      fixed_source_state_and_event=True),
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {name: q.sha256(path) for name, path in paths.items()},
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "historical_0p01_gate": False, "event_M": prior["event_M"],
        "row": row, "pairs": {}, "center_perturbations": {}, "errors": [],
        "limitations": plan["interpretation"],
    }
    q.write_evidence(args.json_out, report)
    try:
        libraries = _libraries(source_plan["pinned_wheel"])
        with tempfile.TemporaryDirectory(prefix="diffgw-source-raw-amp-sensitivity-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            mode_path = Path(tmp) / "mode.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["mode_adapter"], mode_path,
                     source_plan["flux_source_includes"],
                     source_plan["flux_public_includes"])
            function = ctypes.CDLL(str(mode_path)).diffgw_v4_mode22_same_state
            function.argtypes = [ctypes.c_double] * 9 + [ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            x1 = row["mass1"] / (row["mass1"] + row["mass2"])

            def amplitude(state: np.ndarray, omega: float) -> float:
                output = (ctypes.c_double * 39)()
                rc = function(x1, 1 - x1, row["chi1"], row["chi2"],
                              *state, omega, output)
                if rc:
                    raise RuntimeError(f"Pinned raw mode failed: {rc}")
                return abs(complex(output[0], output[1]))

            for first, second in plan["integrator_pairs"]:
                pair = {}
                for offset in plan["offsets_M"]:
                    key = _key(offset)
                    sample_a = prior["integrators"][first]["samples"][key]
                    sample_b = prior["integrators"][second]["samples"][key]
                    state_a = np.asarray(sample_a["state"])
                    state_b = np.asarray(sample_b["state"])
                    omega_a = sample_a["omega"]
                    omega_b = sample_b["omega"]
                    amplitude_aa = amplitude(state_a, omega_a)
                    amplitude_ba = amplitude(state_b, omega_a)
                    amplitude_bb = amplitude(state_b, omega_b)
                    pair[key] = {
                        "state_abs_linf": float(np.max(abs(state_b - state_a))),
                        "omega_delta": omega_b - omega_a,
                        "total_amplitude_delta": amplitude_bb - amplitude_aa,
                        "state_at_fixed_omega_delta": amplitude_ba - amplitude_aa,
                        "omega_at_fixed_state_delta": amplitude_bb - amplitude_ba,
                        "saved_amplitude_replay_abs_linf": max(
                            abs(amplitude_aa - sample_a["amplitude"]),
                            abs(amplitude_bb - sample_b["amplitude"])),
                    }
                report["pairs"][f"{first}_to_{second}"] = pair
                q.write_evidence(args.json_out, report)

            center = prior["integrators"]["adaptive_dense_tight"]["samples"][_key(0.0)]
            state = np.asarray(center["state"])
            omega = center["omega"]
            baseline = amplitude(state, omega)
            coordinates = {"radius": 0, "radial_momentum": 2,
                           "angular_momentum": 3, "omega": None}
            for name, index in coordinates.items():
                records = {}
                for step in plan["center_perturbations"]["absolute_steps"]:
                    if index is None:
                        plus = amplitude(state, omega + step)
                        minus = amplitude(state, omega - step)
                    else:
                        forward, backward = state.copy(), state.copy()
                        forward[index] += step
                        backward[index] -= step
                        plus = amplitude(forward, omega)
                        minus = amplitude(backward, omega)
                    records[str(step)] = {
                        "plus_minus_baseline": plus - baseline,
                        "minus_minus_baseline": minus - baseline,
                        "central_slope": (plus - minus) / (2 * step),
                        "even_residual": plus + minus - 2 * baseline,
                    }
                report["center_perturbations"][name] = records
                q.write_evidence(args.json_out, report)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({"status": report["status"], "errors": report["errors"],
                      "pairs": report["pairs"],
                      "center_perturbations": report["center_perturbations"]}),
          flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
