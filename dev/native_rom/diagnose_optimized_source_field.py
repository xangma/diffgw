"""Compare the pinned v4 optimized evolution field at explicit states."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import socket
import sys
import tempfile

import torch

from . import qualify_generator as q
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .eob.torch.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian,
)
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/optimized_source_field_plan.json"
FIELDS = ("r", "phi", "pr_star", "pphi")


def _native(row: dict, state: dict, choice: dict) -> list[float]:
    m1, m2, chi1, chi2 = [torch.tensor(row[name], dtype=torch.float64)
                          for name in ("mass1", "mass2", "chi1", "chi2")]
    x1, x2 = m1/(m1+m2), m2/(m1+m2)
    ham = SourceCalibratedHistoricalHamiltonian(
        x1*x2, chi1, chi2, delta=x1-x2)
    flux = SEOBNRv4PublishedFlux(**choice)
    values = torch.tensor([state[name] for name in FIELDS], dtype=torch.float64)
    with torch.no_grad():
        field = EOBTrajectoryIntegrator(ham, flux).vector_field(values)
    return field.detach().cpu().tolist()


def _point(function, name: str, row: dict, state: dict, choice: dict) -> dict:
    out = (ctypes.c_double*6)()
    rc = function(*[row[key] for key in ("mass1", "mass2", "chi1", "chi2")],
                  *[state[key] for key in FIELDS], out)
    if rc:
        raise RuntimeError(f"Optimized source field failed at {name}: {rc}")
    native = _native(row, state, choice)
    source = list(out[:4])
    return dict(name=name, row=row, state=state,
                source_optimized_derivative=dict(zip(FIELDS, source)),
                native_derivative=dict(zip(FIELDS, native)),
                native_minus_source=dict(zip(
                    FIELDS, [a-b for a, b in zip(native, source)])),
                native_minus_source_relative=dict(zip(
                    FIELDS, [a/b-1. for a, b in zip(native, source)])),
                source_regular_flux_over_nu=out[4],
                source_optimized_flux_over_nu=out[5],
                regular_minus_optimized_flux_relative=out[4]/out[5]-1.,
                native_minus_source_optimized_torque_relative=native[3]/source[3]-1.)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {key: q.ROOT / plan[key] for key in
             ("source_flux_result", "source_flux_plan",
              "source_wheel_dynamics", "adapter")}
    if (plan["schema"] != "diffgw.seobnrv4.optimized-source-field-plan.v1"
            or any(q.sha256(path) != plan[key+"_sha256"]
                   for key, path in paths.items())
            or plan["release_qualified"] is not False):
        parser.error("Frozen plan or prerequisites changed")
    prior = json.loads(paths["source_flux_result"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    dynamics = json.loads(paths["source_wheel_dynamics"].read_text())
    trace_spec = source_plan["mode_trajectory_evidence"]
    trace_path = q.ROOT / trace_spec["path"]
    if q.sha256(trace_path) != trace_spec["sha256"]:
        parser.error("Frozen trajectory states changed")
    trace = json.loads(trace_path.read_text())
    if (prior["status"] != "diagnostic_complete"
            or dynamics["status"] != "diagnostic_complete"
            or len(prior["rows"]) != 2
            or [item["name"] for item in prior["mode_trajectory"]]
            != ["moderate_spin_early", "strong_aligned_spin"]
            or len(prior["mode_trajectory"][1]["points"]) != 4
            or [item["name"] for item in trace["rows"]]
            != trace_spec["row_names"]
            or prior["source_placement_flux_choice"]
            != plan["source_placement_flux_choice"]):
        parser.error("Pinned source results are incompatible")
    libraries = _libraries(source_plan["pinned_wheel"])
    torch.set_num_threads(1)
    code = ("dev/native_rom/eob/torch/source_calibration_hamiltonian.py",
            "src/torchwave/approximants/eob/integrator.py",
            "src/torchwave/approximants/eob/seobnrv4_flux.py")
    report = dict(schema="diffgw.seobnrv4.optimized-source-field.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256={key: q.sha256(path) for key, path in paths.items()},
                  trajectory_sha256=q.sha256(trace_path),
                  source_sha256={key: q.sha256(q.ROOT/key) for key in code},
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv),
                  points=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-optimized-field-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            dylib = Path(tmp) / "optimized_source_field.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["adapter"], dylib,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            library = ctypes.CDLL(str(dylib))
            function = library.diffgw_v4_optimized_source_field
            function.argtypes = [ctypes.c_double]*8 + [ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            for index, item in enumerate(prior["rows"]):
                initial = item["source_legacy_initial"]
                state = dict(r=initial["r"], phi=0.,
                             pr_star=initial["pr_star"], pphi=initial["pphi"])
                report["points"].append(_point(
                    function, f"source_initial_{index}", item["row"], state,
                    plan["source_placement_flux_choice"]))
                q.write_evidence(args.json_out, report)
            row = trace["rows"][1]["physical"]
            if row != dynamics["rows"][2]["row"]:
                raise ValueError("Strong-spin row changed")
            for index, item in enumerate(trace["rows"][1]["points"]):
                report["points"].append(_point(
                    function, f"strong_aligned_{index}", row, item["state"],
                    plan["source_placement_flux_choice"]))
                q.write_evidence(args.json_out, report)
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
