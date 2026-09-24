"""Check whether fine public output can proxy a coarse run's hidden boundary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import lalsimulation as lalsim
import numpy as np

from . import qualify_generator as q
from .diagnose_public_nqc_mode_boundary import public_call


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/q4_public_grid_proxy_plan.json"
ACTIVE = (0, 1, 2, 6, 7)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    attachment_path = q.ROOT / plan["attachment_inputs"]
    if (plan["schema"] != "diffgw.seobnrv4.q4-public-grid-proxy-plan.v1"
            or plan["coarse_output_step_M"] != .5
            or plan["fine_output_step_M"] != .015625
            or plan["late_high_dynamics_window_M"] != 30.
            or plan["production_change"] is not False
            or plan["release_qualified"] is not False
            or q.sha256(attachment_path) != plan["attachment_inputs_sha256"]):
        parser.error("frozen plan or attachment input evidence differs")
    prior = json.loads(attachment_path.read_text())
    campaign = json.loads((q.ROOT / "dev/native_rom/evidence/2026-09-24/source_candidate_new_rows.json").read_text())
    pin = campaign["lal_binary"]
    if (prior["status"] != "diagnostic_complete"
            or prior["row"] != campaign["rows"]["q4_mixed_spin"]["row"]
            or q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin):
        parser.error("frozen q4 row or public binary differs")
    report = dict(schema="diffgw.seobnrv4.q4-public-grid-proxy.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  attachment_inputs_sha256=q.sha256(attachment_path),
                  lal_binary=pin, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        data = {}
        for name, step in (("coarse", .5), ("fine", .015625)):
            empty, coefficients = public_call(prior["row"], step, 1)
            if list(empty) != [None, None, None]:
                raise ValueError("coefficient-only public call returned waveforms")
            (modes, _, high), _ = public_call(prior["row"], step, 0)
            mode = lalsim.SphHarmTimeSeriesGetMode(modes, 2, 2)
            data[name] = dict(step_M=step, high=np.asarray(high.data).reshape(5, -1),
                              coefficients=coefficients[list(ACTIVE)],
                              mode_samples=len(mode.data.data),
                              mode_deltaT_s=float(mode.deltaT))
        coarse, fine = data["coarse"], data["fine"]
        common, ci, fi = np.intersect1d(coarse["high"][0], fine["high"][0],
                                       return_indices=True)
        if len(common) < 100 or not np.allclose(np.diff(common), .015625, rtol=0, atol=1e-10):
            raise ValueError("public high dynamics have no common uniform grid")
        selected = common >= common[-1]-plan["late_high_dynamics_window_M"]
        delta = coarse["high"][1:, ci[selected]]-fine["high"][1:, fi[selected]]
        ccoef, fcoef = coarse["coefficients"], fine["coefficients"]
        report["public_comparison"] = dict(
            coarse=dict(output_step_M=coarse["step_M"],
                        high_dynamics_samples=coarse["high"].shape[1],
                        mode_samples=coarse["mode_samples"],
                        nqc_active_coefficients=ccoef.tolist()),
            fine=dict(output_step_M=fine["step_M"],
                      high_dynamics_samples=fine["high"].shape[1],
                      mode_samples=fine["mode_samples"],
                      nqc_active_coefficients=fcoef.tolist()),
            common_high_dynamics_samples=len(common),
            common_late_samples=int(selected.sum()),
            late_state_max_abs_difference=abs(delta).max(axis=1).tolist(),
            nqc_active_max_relative_change=float(np.max(abs(fcoef-ccoef)/np.maximum(abs(ccoef), 1))),
            fine_output_valid_as_coarse_boundary_proxy=bool(np.array_equal(ccoef, fcoef)
                                                             and not np.any(delta)))
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
