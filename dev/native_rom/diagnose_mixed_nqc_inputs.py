"""Attribute a pinned strong-spin NQC input gap to mode and orbital state."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
from scipy.optimize import brentq

from . import qualify_generator as q
from .diagnose_native_source_nqc_inputs import _fit, _source_samples
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_source_flux_initial import _libraries


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/mixed_nqc_inputs_plan.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    names = ("source_evidence", "native_evidence", "source_array", "native_array",
             "source_flux_plan")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.mixed-nqc-inputs-plan.v1"
            or plan["row_name"] != "strong_aligned_spin"
            or plan["equal_orbital_frequency_control"] is not True
            or any(q.sha256(paths[name]) != plan[name+"_sha256"] for name in names)):
        parser.error("Frozen plan or prerequisites changed")
    source = json.loads(paths["source_evidence"].read_text())
    native = json.loads(paths["native_evidence"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (source["status"] != "diagnostic_complete"
            or native["status"] != "diagnostic_complete"
            or source["rows"][0]["name"] != plan["row_name"]
            or source["array_asset_sha256"] != q.sha256(paths["source_array"])
            or native["native_array_asset_sha256"] != q.sha256(paths["native_array"])):
        parser.error("Captured source/native NQC arrays changed")
    gsl_path = _libraries(source_plan["pinned_wheel"])["libgsl.28.dylib"]
    gsl = GSLCubic(gsl_path)
    report = dict(schema="diffgw.seobnrv4.mixed-nqc-inputs.v1", scope=q.SCOPE,
                  status="running", release_qualified=False, plan_sha256=q.sha256(PLAN),
                  script_sha256=q.sha256(__file__),
                  input_sha256={name: q.sha256(path) for name, path in paths.items()},
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                               argv=sys.argv), comparison=None, errors=[])
    q.write_evidence(args.json_out, report)
    try:
        source_time, source_values, source_meta, source_coefficients = _source_samples(
            paths["source_array"])
        with np.load(paths["native_array"]) as data:
            native_time = data["time"].copy()
            native_values = data["values"].copy()
            native_match = float(data["match"])
            native_coefficients = data["coefficients"].copy()
        source_match = float(source_meta[2])
        targets = source_meta[3:]
        step = float(source_meta[1])
        shift = int(round((native_match-source_match)/step))
        count = min(len(source_time), len(native_time)-shift)
        if (shift < 0 or count < 9000 or native_values.shape != (5, len(native_time))
                or source_values.shape != (5, len(source_time))
                or abs((native_match-source_match)-shift*step) > 1e-8
                or not np.allclose(native_time[shift:shift+count]-source_time[:count],
                                   shift*step, atol=1e-8, rtol=0)):
            raise ValueError("Source and native NQC sample grids are not index-aligned")
        time = source_time[:count]
        source_values = source_values[:, :count]
        native_values = native_values[:, shift:shift+count]
        fits = {}
        for mode_name, mode in (("source", source_values), ("native", native_values)):
            for state_name, state in (("source", source_values), ("native", native_values)):
                key = f"{mode_name}_mode_{state_name}_state"
                fits[key] = _fit(gsl, time, mode[0], mode[1], state[2], state[3],
                                 state[4], source_match, targets)
        full_source = np.asarray(native["comparison"]["source_fit"]["coefficients"])
        full_native = np.asarray(native["comparison"]["native_fit_with_source_targets"]
                                 ["coefficients"])
        source_fit = np.asarray(fits["source_mode_source_state"]["coefficients"])
        native_fit = np.asarray(fits["native_mode_native_state"]["coefficients"])
        if (np.max(abs(full_source-source_fit)) > 1e-3
                or np.max(abs(full_native-native_fit)) > 1e-3):
            raise ValueError("Common-grid truncation changed source or native NQC fit")
        source_active = source_coefficients[[0, 1, 2, 6, 7]]
        if (np.max(abs(source_fit-source_active)) > 1e-3
                or np.max(abs(native_coefficients-np.asarray(
                    native["comparison"]["native_active"]))) > 1e-12):
            raise ValueError("Captured NQC coefficients changed")
        source_mode_native_state = np.asarray(fits["source_mode_native_state"]["coefficients"])
        native_mode_source_state = np.asarray(fits["native_mode_source_state"]["coefficients"])
        source_frequency_at_match = gsl.derivatives(time, source_values[4], source_match)[0]
        equal_frequency_offset = brentq(
            lambda offset: gsl.derivatives(time, native_values[4],
                                           source_match+offset)[0]-source_frequency_at_match,
            -0.1, 0.1, xtol=1e-12)
        equal_frequency_fit = _fit(gsl, time, *native_values,
                                   source_match+equal_frequency_offset, targets)
        equal_frequency_derivatives = {
            name: dict(source=gsl.derivatives(time, source_values[j], source_match),
                       native_shifted=gsl.derivatives(time, native_values[j],
                                                      source_match+equal_frequency_offset))
            for j, name in enumerate(("amplitude", "phase", "radius", "radial_momentum",
                                      "orbital_frequency"))
        }
        report["comparison"] = dict(
            aligned_sample_count=count, shift_samples=shift,
            coordinate_shift_M=shift*step, source_match_time_M=source_match,
            native_match_time_M=native_match, source_targets=targets.tolist(),
            fits=fits, source_active=source_active.tolist(),
            native_active=native_coefficients.tolist(),
            truncation_max_abs_error=dict(source=float(np.max(abs(full_source-source_fit))),
                                          native=float(np.max(abs(full_native-native_fit)))),
            coefficient_delta_native_minus_source=(native_fit-source_fit).tolist(),
            state_only_delta=(source_mode_native_state-source_fit).tolist(),
            mode_only_delta=(native_mode_source_state-source_fit).tolist(),
            interaction_delta=(native_fit-source_mode_native_state
                               -native_mode_source_state+source_fit).tolist(),
            equal_orbital_frequency=dict(
                native_match_shift_M=equal_frequency_offset,
                source_frequency_at_match=source_frequency_at_match,
                derivatives=equal_frequency_derivatives,
                native_fit=equal_frequency_fit,
                native_fit_minus_source=(np.asarray(equal_frequency_fit["coefficients"])
                                         -source_fit).tolist()),
            method_limit="Crossed mode/state samples are counterfactual inputs to the NQC fit; they are not self-consistent physical waveforms.")
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
