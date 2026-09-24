"""Test whether q8 NQC coefficient differences explain measured phase drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys

import lal
import numpy as np

from . import qualify_generator as q
from .diagnose_source_wheel_mode import _public


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_q8_nqc_phase_counterfactual_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / path for name, path in plan["inputs"].items()}
    if (plan["schema"] != "diffgw.seobnrv4.source-q8-nqc-phase-counterfactual-plan.v1"
            or plan["row"] != "q8_high_aligned" or plan["output_step_M"] != 0.5
            or plan["phase_anchor_M"] != -100.0
            or plan["last_probe_M"] != -25.0
            or plan["release_qualified"] is not False
            or {name: q.sha256(path) for name, path in paths.items()}
               != plan["input_sha256"]):
        parser.error("frozen NQC counterfactual plan or inputs differ")
    campaign = json.loads(paths["campaign"].read_text())
    phase_profile = json.loads(paths["phase_profile"].read_text())
    source_native = json.loads(paths["source_native_mode"].read_text())
    source_wheel = json.loads(paths["source_wheel_mode"].read_text())
    row = campaign["rows"][plan["row"]]["row"]
    if (any(x["status"] != "diagnostic_complete" for x in
            (campaign, phase_profile, source_native, source_wheel))
            or phase_profile["row"] != row
            or q.lal_binary_identity(campaign["lal_binary"]["lal_version"],
                                     campaign["lal_binary"]["lalsimulation_version"])
               != campaign["lal_binary"]):
        parser.error("pinned q8 profile or binary differs")
    scope = dict(q.SCOPE)
    scope["description"] = (
        "Coefficient-only NQC phase counterfactual on one public-wheel trajectory.")
    scope["known_limitations"] = [
        item for item in q.SCOPE["known_limitations"]
        if item != "No time, phase, amplitude, or coefficient fitting is performed."]
    scope["known_limitations"].append(
        "Uses one predeclared phase anchor and no fitted parameters; common wheel "
        "states are a counterfactual, not the native or separate-source trajectory.")
    report = dict(schema="diffgw.seobnrv4.source-q8-nqc-phase-counterfactual.v1",
                  status="running", release_qualified=False, scope=scope,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256=plan["input_sha256"], row=row,
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), errors=[])
    q.write_evidence(args.json_out, report)
    try:
        wheel_epoch, wheel_dt, wheel_mode, low = _public(row, plan["output_step_M"])
        mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
        if (not math.isclose(wheel_dt/mass_seconds, 0.5, rel_tol=0, abs_tol=1e-10)
                or hashlib.sha256(wheel_mode.tobytes()).hexdigest()
                   != source_wheel["rows"][plan["row"]]["wheel_mode_sha256"]):
            raise ValueError("wheel mode changed")
        frequency = np.gradient(low[2], plan["output_step_M"], edge_order=2)
        peak = int(np.argmax(frequency))
        low_t = (np.arange(low.shape[1])-peak)*plan["output_step_M"]
        offset_real = peak+wheel_epoch/wheel_dt
        if abs(offset_real-round(offset_real)) > 0.5:
            raise ValueError("wheel mode/low dynamics mapping invalid")
        probes = np.asarray(phase_profile["probe_times_M"], dtype=float)
        selected = probes <= plan["last_probe_M"]
        if (not selected.any() or plan["phase_anchor_M"] not in probes[selected]
                or probes[selected][0] < low_t[0]
                or probes[selected][-1] > low_t[-1]):
            raise ValueError("phase probes exceed common public inspiral")
        probes = probes[selected]
        r, pr = low[1], low[3]
        ratio = pr/(r*frequency)
        if not np.isfinite(ratio).all():
            raise ValueError("invalid NQC basis")
        basis = np.stack((ratio, ratio*pr**2), axis=-1)
        coefficients = source_native["rows"][plan["row"]]["nqc_coefficients"]
        phase_coeff = {key: np.asarray(value[-2:], dtype=float)
                       for key, value in coefficients.items()
                       if key in ("native", "source", "wheel")}
        if set(phase_coeff) != {"native", "source", "wheel"}:
            raise ValueError("missing NQC phase coefficients")
        anchor = int(np.flatnonzero(probes == plan["phase_anchor_M"])[0])
        pairs = {}
        for candidate, reference in (("native", "source"),
                                     ("wheel", "source"),
                                     ("native", "wheel")):
            delta = phase_coeff[candidate]-phase_coeff[reference]
            values = basis @ delta
            predicted = np.interp(probes, low_t, values)
            predicted -= predicted[anchor]
            name = f"{candidate}_minus_{reference}"
            observed = np.asarray(phase_profile["pairs"][name][
                "phase_from_anchor_rad"], dtype=float)[selected]
            residual = observed-predicted
            pairs[name] = dict(delta_phase_coefficients=delta.tolist(),
                               observed_phase_rad=observed.tolist(),
                               predicted_nqc_only_phase_rad=predicted.tolist(),
                               residual_phase_rad=residual.tolist(),
                               residual_rms_rad=float(np.sqrt(np.mean(residual**2))),
                               residual_max_abs_rad=float(np.max(np.abs(residual))))
        report.update(probe_times_M=probes.tolist(), pairs=pairs,
                      wheel_mode_sha256=hashlib.sha256(wheel_mode.tobytes()).hexdigest(),
                      wheel_low_peak_index=peak,
                      wheel_mode_to_low_offset_remainder=float(
                          offset_real-round(offset_real)))
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
