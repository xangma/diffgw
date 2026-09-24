"""Evaluate pinned GSL NQC spline matching on frozen native input samples."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch

from . import qualify_generator as q


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/nqc_gsl_same_input_plan.json"


class GSLCubic:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        lib = self.lib
        void = ctypes.c_void_p
        double = ctypes.c_double
        pointer = ctypes.POINTER(double)
        lib.gsl_spline_alloc.argtypes = [void, ctypes.c_size_t]
        lib.gsl_spline_alloc.restype = void
        lib.gsl_spline_init.argtypes = [void, pointer, pointer, ctypes.c_size_t]
        lib.gsl_spline_init.restype = ctypes.c_int
        lib.gsl_spline_free.argtypes = [void]
        lib.gsl_interp_accel_alloc.restype = void
        lib.gsl_interp_accel_free.argtypes = [void]
        for name in ("gsl_spline_eval", "gsl_spline_eval_deriv", "gsl_spline_eval_deriv2"):
            function = getattr(lib, name)
            function.argtypes = [void, double, void]
            function.restype = double
        self.kind = void.in_dll(lib, "gsl_interp_cspline")

    def derivatives(self, time, values, match):
        time = np.ascontiguousarray(time, dtype=np.float64)
        values = np.ascontiguousarray(values, dtype=np.float64)
        if time.ndim != 1 or values.shape != time.shape or len(time) < 3:
            raise ValueError("GSL spline needs equal one-dimensional samples")
        spline = self.lib.gsl_spline_alloc(self.kind, len(time))
        accel = self.lib.gsl_interp_accel_alloc()
        if not spline or not accel:
            raise MemoryError("GSL spline allocation failed")
        try:
            code = self.lib.gsl_spline_init(
                spline, time.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(time))
            if code:
                raise ValueError(f"GSL spline initialization failed: {code}")
            output = [getattr(self.lib, name)(spline, match, accel) for name in
                      ("gsl_spline_eval", "gsl_spline_eval_deriv", "gsl_spline_eval_deriv2")]
            if not np.isfinite(output).all():
                raise ValueError("nonfinite GSL spline derivative")
            return output
        finally:
            self.lib.gsl_interp_accel_free(accel)
            self.lib.gsl_spline_free(spline)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.nqc-gsl-same-input-plan.v1"
            or plan["rows"] != ["strong_aligned_spin", "strong_antialigned_spin"]):
        parser.error("frozen GSL plan differs")
    coefficient_path = q.ROOT / plan["coefficient_evidence"]
    if q.sha256(coefficient_path) != plan["coefficient_evidence_sha256"]:
        parser.error("coefficient evidence differs")
    coefficients = json.loads(coefficient_path.read_text())
    waveform_plan = json.loads((q.ROOT / plan["waveform_plan"]).read_text())
    if (coefficients["status"] != "diagnostic_complete"
            or [row["name"] for row in coefficients["rows"]] != plan["rows"]
            or [row["name"] for row in waveform_plan["rows"]] != plan["rows"]):
        parser.error("frozen prerequisites differ")
    binary = coefficients["lal_binary"]
    if q.lal_binary_identity(binary["lal_version"], binary["lalsimulation_version"]) != binary:
        parser.error("public binary identity differs")
    gsl_paths = [Path(path) for path in binary["binaries"] if Path(path).name.startswith("libgsl.")]
    if len(gsl_paths) != 1:
        parser.error("pinned GSL library is ambiguous")
    gsl = GSLCubic(gsl_paths[0])
    if not np.allclose(gsl.derivatives(np.arange(5.), 2*np.arange(5.)+3, 2.25),
                       [7.5, 2., 0.], rtol=0, atol=1e-12):
        parser.error("GSL binding self-check failed")
    torch.set_num_threads(1)
    report = dict(
        schema="diffgw.seobnrv4.nqc-gsl-same-input.v1", scope=q.SCOPE,
        status="running", release_qualified=False, plan_sha256=q.sha256(PLAN),
        script_sha256=q.sha256(__file__), coefficient_evidence_sha256=q.sha256(coefficient_path),
        source_sha256={path: q.sha256(q.ROOT / path) for path in (
            "dev/native_rom/qualify_generator.py",
            "dev/native_rom/eob/torch/adaptive_driver_clean.py",
            "src/torchwave/approximants/eob/nqc.py",
            "src/torchwave/approximants/eob/numerical_attachment.py",
        )},
        lal_binary=binary, gsl_library=str(gsl_paths[0]), rows=[], errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
    )
    q.write_evidence(args.json_out, report)
    try:
        for item, prior in zip(waveform_plan["rows"], coefficients["rows"]):
            waveform_path = q.ROOT / item["waveform"]
            if q.sha256(waveform_path) != item["waveform_sha256"]:
                raise ValueError(f"{item['name']} waveform evidence differs")
            waveform = json.loads(waveform_path.read_text())
            level = waveform["candidate"][waveform_plan["level"]]["level"]
            numerical = waveform["choice"]["numerical"]
            physical = waveform["choice"]["physical"]
            flux = waveform["choice"]["flux_placements"]
            run = q.native_run(
                waveform["row"], level, q.Settings(**waveform["settings"]),
                diagnostic_hermite_resampling=numerical["hermite_resampling"],
                diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
                diagnostic_matching_second_derivatives=True,
                diagnostic_restart_time_M=(
                    numerical["fixed_restart_time_M"] if item["name"] == "strong_aligned_spin"
                    else waveform["choice"]["fixed_restart_time_M"]),
                diagnostic_coarse_initial_step_M=0.125,
                diagnostic_warm_start_Momega=0.03162277660168379,
                diagnostic_nqc_phase_window_M=item["nqc_phase_window_M"],
                diagnostic_mode_33_higher_spin=physical["mode_33_higher_spin"],
                diagnostic_hamiltonian_log_derivative=physical["hamiltonian_log_derivative"],
                diagnostic_mode_21_higher_spin=flux["mode_21_higher_spin"],
                diagnostic_mode_31_higher_spin=flux["mode_31_higher_spin"],
                diagnostic_mode_32_spin_square=flux["mode_32_spin_square"],
                return_nqc_coefficients=True, return_nqc_samples=True,
            )
            replay = q.summarize_run(run)["array_sha256"]
            if replay != prior["native_waveform_sha256"]:
                raise ValueError(f"{item['name']} native waveform no longer reproduces")
            native = np.asarray(run["nqc_coefficients"]["amplitude"]
                                + run["nqc_coefficients"]["phase"])
            if not np.allclose(native, prior["native_coefficients"], rtol=0, atol=1e-10):
                raise ValueError(f"{item['name']} native coefficients no longer reproduce")
            samples = run["nqc_samples"]
            time = np.asarray(samples["time"])
            match = samples["match_time"]
            if not np.allclose(np.diff(time), np.diff(time)[0], rtol=0, atol=1e-9):
                raise ValueError(f"{item['name']} NQC input grid is not uniform")
            if not 0 < match < time[-1]:
                raise ValueError(f"{item['name']} match lies outside native NQC samples")
            amplitude = gsl.derivatives(time, samples["amplitude"], match)
            products = np.stack([gsl.derivatives(time, np.asarray(samples["products"])[:, j], match)
                                 for j in range(3)], axis=-1)
            phase_basis = np.stack([gsl.derivatives(time, np.asarray(samples["phase_basis"])[:, j], match)
                                    for j in range(2)], axis=-1)
            source_phase = gsl.derivatives(time, -np.asarray(samples["waveform_phase"]), match)
            omega, omega_dot = source_phase[1:]
            if omega * omega_dot > 0:
                omega, omega_dot = abs(omega), abs(omega_dot)
            else:
                omega, omega_dot = abs(omega), -abs(omega_dot)
            targets = np.asarray(prior["target_counterfactual"]["source_targets"])
            amp_matrix = products
            amp_rhs = targets[:3] - amplitude
            phase_matrix = -phase_basis[1:]
            phase_rhs = targets[3:] - [omega, omega_dot]
            gsl_coefficients = np.concatenate((np.linalg.solve(amp_matrix, amp_rhs),
                                               np.linalg.solve(phase_matrix, phase_rhs)))
            full_spline_phase = np.asarray(run["nqc_full_spline_phase_matrix"])
            public = np.asarray(prior["public_coefficients"])
            report["rows"].append(dict(
                name=item["name"], waveform_sha256=replay, sample_count=len(time),
                sample_sha256=q.canonical_hash({
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in samples.items()
                }),
                match_time_M=match, source_targets=targets.tolist(),
                gsl_base_amplitude_derivatives=amplitude,
                gsl_product_matrix=amp_matrix.tolist(),
                gsl_phase_basis_matrix=phase_basis[1:].tolist(),
                gsl_source_waveform_phase_derivatives=source_phase,
                gsl_source_frequency_derivatives=[omega, omega_dot],
                gsl_amplitude_rhs=amp_rhs.tolist(), gsl_phase_rhs=phase_rhs.tolist(),
                gsl_coefficients=gsl_coefficients.tolist(),
                native_coefficients=native.tolist(), public_coefficients=public.tolist(),
                gsl_minus_native=(gsl_coefficients-native).tolist(),
                gsl_minus_public=(gsl_coefficients-public).tolist(),
                gsl_public_gap_l2=float(np.linalg.norm(gsl_coefficients-public)),
                native_public_gap_l2=float(np.linalg.norm(native-public)),
                gsl_phase_matrix_minus_native_full_spline=(phase_basis[1:]-full_spline_phase).tolist(),
            ))
            q.write_evidence(args.json_out, report)
            print(f"row={item['name']}", flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
