"""Dev-only causal native whole-mode replay with pinned-source NQC coefficients."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from unittest.mock import patch

import lal
import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_peak_anchor import fitted_metrics
from .diagnose_q4_legacy_initial_intervention import _native_chart
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_q4_ultrafine_chart import _native
from .diagnose_source_wheel_mode import _source
from torchwave.approximants.eob import numerical_attachment as attachment
from torchwave.approximants.eob.nqc import NQCCoefficients


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/q4_native_source_nqc_intervention_plan.json"


def _with_coefficients(values):
    """Patch only the solved coefficients and their sampled corrected series."""
    original = attachment._full_series_nqc
    calls = []

    def substitute(*args, **kwargs):
        nqc = original(*args, **kwargs)
        a = torch.as_tensor(values[:3], dtype=nqc.sampled_amplitude.dtype,
                            device=nqc.sampled_amplitude.device)
        b = torch.as_tensor(values[3:], dtype=a.dtype, device=a.device)
        coefficients = NQCCoefficients(a, b)
        amplitude = nqc.sampled_amplitude + (
            nqc.sampled_products * a.unsqueeze(-2)).sum(-1)
        phase = nqc.sampled_waveform_phase - (
            nqc.sampled_phase_basis * b.unsqueeze(-2)).sum(-1)
        corrected = amplitude * torch.exp(-1j * phase)
        calls.append({"native": torch.cat(nqc.coefficients).detach().tolist(),
                      "injected": list(values)})
        return replace(
            nqc,
            calibration=nqc.calibration._replace(coefficients=coefficients),
            sampled_corrected_amplitude=amplitude,
            sampled_corrected_phase=attachment._unwrap_mode_phase(corrected),
        )

    return patch.object(attachment, "_full_series_nqc", substitute), calls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("Pinned source adapter requires macOS")
    plan = json.loads(PLAN.read_text())
    inputs = {key: q.ROOT / plan[key] for key in (
        "prior_chart", "source_coefficients", "source_flux_plan",
        "mode_adapter", "settings")}
    if (plan["schema"] != "diffgw.seobnrv4.q4-native-source-nqc-intervention-plan.v1"
            or plan["release_qualified"] is not False
            or plan["row_name"] != "q4_nonspinning"
            or plan["output_steps_M"] != [0.03125, 0.015625]
            or any(q.sha256(path) != plan[key + "_sha256"]
                   for key, path in inputs.items())):
        parser.error("Frozen plan or prerequisites changed")
    prior = json.loads(inputs["prior_chart"].read_text())
    captured = json.loads(inputs["source_coefficients"].read_text())
    source_plan = json.loads(inputs["source_flux_plan"].read_text())
    if (prior["status"] != "diagnostic_complete"
            or captured["status"] != "diagnostic_complete"
            or prior["row"] != captured["row"]):
        parser.error("Frozen source coefficient capture changed")
    row = prior["row"]
    settings = q.Settings(**json.loads(inputs["settings"].read_text())["settings"])
    torch.set_num_threads(1)
    report = {
        "schema": "diffgw.seobnrv4.q4-native-source-nqc-intervention.v1",
        "scope": q.SCOPE, "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {key: q.sha256(path) for key, path in inputs.items()},
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": row, "grids": {}, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-q4-native-nqc-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            library_path = Path(tmp) / "source_mode.dylib"
            _compile(private, args.headers, args.gsl_include,
                     _libraries(source_plan["pinned_wheel"]),
                     inputs["mode_adapter"], library_path,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            source_fn = ctypes.CDLL(str(library_path)).diffgw_v4_full_source_mode
            source_fn.argtypes = [ctypes.c_double] * 6 + [
                ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double)]
            source_fn.restype = ctypes.c_int
            for step in plan["output_steps_M"]:
                started = time.monotonic()
                key = str(step)
                old = prior["grids"][key]
                coefficients = captured["grids"][key]["source_active"]
                epoch, delta_t, source_mode = _source(
                    source_fn, row, step, plan["capacity"])
                if (hashlib.sha256(source_mode.tobytes()).hexdigest()
                        != old["source"]["mode_sha256"]
                        or hashlib.sha256(source_mode.tobytes()).hexdigest()
                        != captured["grids"][key]["source_mode_sha256"]):
                    raise ValueError(f"Pinned source mode changed at {step} M")
                source_time = epoch + np.arange(source_mode.size) * delta_t
                mass_seconds = (row["mass1"] + row["mass2"]) * float(lal.MTSUN_SI)
                level = {"atol": 1e-14, "rtol": 1e-13}
                baseline = _native(row, step, level, settings,
                                   return_nqc_coefficients=True)
                summary = q.summarize_run(baseline)
                if summary["array_sha256"] != old["native"]["array_sha256"]:
                    raise ValueError(f"Native baseline changed at {step} M")
                native_coefficients = (baseline["nqc_coefficients"]["amplitude"]
                                       + baseline["nqc_coefficients"]["phase"])
                no_op_patch, no_op_calls = _with_coefficients(native_coefficients)
                with no_op_patch:
                    no_op = _native(row, step, level, settings)
                if (len(no_op_calls) != 2
                        or not np.array_equal(no_op["mode"], baseline["mode"])
                        or not np.array_equal(no_op["time"], baseline["time"])):
                    raise ValueError(f"Same-native-coefficient replay changed at {step} M")
                source_patch, source_calls = _with_coefficients(coefficients)
                with source_patch:
                    changed = _native(row, step, level, settings)
                if (len(source_calls) != 2
                        or not np.array_equal(changed["time"], baseline["time"])):
                    raise ValueError(f"NQC intervention changed trajectory chart at {step} M")

                def compare(run):
                    return fitted_metrics(*_native_chart(row, run), source_time,
                                          source_mode, mass_seconds, plan["peak_fit"])

                baseline_metric = compare(baseline)
                changed_metric = compare(changed)
                if abs(baseline_metric["metrics"]["complex_relative_l2"] - old[
                        "native_vs_source_fitted"]["metrics"]["complex_relative_l2"]) > 1e-12:
                    raise ValueError(f"Native/source baseline metric changed at {step} M")
                report["grids"][key] = {
                    "source_mode_sha256": hashlib.sha256(source_mode.tobytes()).hexdigest(),
                    "native_coefficients": native_coefficients,
                    "source_coefficients": coefficients,
                    "native_baseline": summary,
                    "same_coefficients_no_op": True,
                    "source_coefficient_run": q.summarize_run(changed),
                    "baseline_vs_source": baseline_metric,
                    "intervention_vs_source": changed_metric,
                    "complex_relative_l2_change": changed_metric["metrics"][
                        "complex_relative_l2"] - baseline_metric["metrics"][
                            "complex_relative_l2"],
                    "elapsed_seconds": time.monotonic() - started,
                }
                q.write_evidence(args.json_out, report)
                print("grid", step, report["grids"][key]["complex_relative_l2_change"],
                      flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({"status": report["status"], "errors": report["errors"]}), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
