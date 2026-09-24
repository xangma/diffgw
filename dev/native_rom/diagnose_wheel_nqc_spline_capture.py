"""Capture pinned macOS wheel NQC spline inputs and replay their source C fit.

The interposer observes GSL inputs in a separate process. It does not alter
the installed wheel and is only a development diagnostic, not model data.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np

from . import qualify_generator as q
from .diagnose_hamiltonian_same_state import compile_adapter as compile_hamiltonian_adapter
from .diagnose_nqc_c_same_input import bind, compile_wrapper
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_public_nqc_mode_boundary import public_call
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_wheel_state_fit import ACTIVE, local_indices, source_peak_rule


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/wheel_nqc_spline_capture_plan.json"


def _capture_child(name: str, step: float) -> int:
    frozen = json.loads((q.ROOT / "dev/native_rom/evidence/2026-09-24/source_wheel_mode.json").read_text())
    row = next(item["row"] for item in frozen["rows"] if item["name"] == name)
    _, coefficients = public_call(row, step, 1)
    print(json.dumps(np.asarray(coefficients, dtype=np.float64).tolist()))
    return 0


def _compile_interposer(source: Path, gsl: Path, output: Path) -> None:
    subprocess.run(
        ["cc", "-dynamiclib", "-Wl,-headerpad_max_install_names", "-o",
         str(output), str(source), str(gsl)],
        check=True, capture_output=True, text=True,
    )
    install_name = subprocess.check_output(
        ["otool", "-D", str(gsl)], text=True,
    ).splitlines()[1].strip()
    subprocess.run(
        ["install_name_tool", "-change", install_name, str(gsl), str(output)],
        check=True, capture_output=True, text=True,
    )


def _read_capture(path: Path, count: int, n: int, dt: float) -> tuple[list[np.ndarray], str]:
    data = path.read_bytes()
    position = 0
    columns = []
    expected_axis = np.arange(n, dtype=np.float64) * dt
    for _ in range(count):
        if position + 8 > len(data):
            raise ValueError("Wheel GSL spline capture is incomplete")
        length = struct.unpack_from("<Q", data, position)[0]
        position += 8
        if length != n or position + 16 * n > len(data):
            raise ValueError("Wheel GSL spline capture has an unexpected array length")
        axis = np.frombuffer(data, dtype="<f8", count=n, offset=position)
        position += 8 * n
        values = np.frombuffer(data, dtype="<f8", count=n, offset=position).copy()
        position += 8 * n
        if np.max(abs(axis - expected_axis)) > 1e-11:
            raise ValueError("Wheel GSL spline capture has an unexpected time axis")
        columns.append(values)
    if position != len(data):
        raise ValueError("Wheel GSL spline capture has extra calls")
    return columns, hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--headers", type=Path)
    parser.add_argument("--gsl-include", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--capture-child", choices=["nonspinning", "mild_mixed_spin", "strong_aligned_spin"])
    parser.add_argument("--step-M", type=float, default=0.125)
    args = parser.parse_args(argv)
    if args.capture_child:
        return _capture_child(args.capture_child, args.step_M)
    if sys.platform != "darwin":
        parser.error("The pinned wheel spline interposer requires macOS")
    if any(value is None for value in (args.source_root, args.headers, args.gsl_include, args.json_out)):
        parser.error("--source-root, --headers, --gsl-include, and --json-out are required")
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / entry["path"] for name, entry in plan["inputs"].items()}
    if (plan["schema"] != "diffgw.seobnrv4.wheel-nqc-spline-capture-plan.v1"
            or plan["release_qualified"] is not False
            or plan["rows"] != ["nonspinning", "mild_mixed_spin", "strong_aligned_spin"]
            or plan["step_M"] != 0.125 or plan["expected_spline_calls"] != 10
            or plan["compiler_controls"] != ["-O0", "-O2", "-O3", "-O2 -ffast-math"]
            or plan["local_half_width_M"] != 4.0
            or any(q.sha256(paths[name]) != entry["sha256"]
                   for name, entry in plan["inputs"].items())):
        parser.error("Frozen wheel spline capture plan or inputs changed")
    frozen = json.loads(paths["source_wheel_mode"].read_text())
    exact_rows = json.loads(paths["exact_rows"].read_text())["rows"]
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    libraries = _libraries(source_plan["pinned_wheel"])
    gsl_library = libraries["libgsl.28.dylib"]
    gsl = GSLCubic(gsl_library)
    report = {
        "schema": "diffgw.seobnrv4.wheel-nqc-spline-capture.v1",
        "scope": dict(
            q.SCOPE, development_only=True,
            description="Pinned-wheel private GSL NQC inputs, source-C replay, and source-mode input ablations.",
            known_limitations=[
                "The captured wheel arrays are black-box reference outputs, not native model data.",
                "The interposer is specific to the pinned macOS wheel and linked GSL binary.",
                "Three diagnostic rows and source-wheel parity do not establish physical accuracy or release qualification.",
            ],
        ),
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {name: q.sha256(path) for name, path in paths.items()},
        "pinned_libraries_sha256": {name: q.sha256(path) for name, path in libraries.items()},
        "rows": {}, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-wheel-nqc-capture-") as temp:
            directory = Path(temp)
            shim = directory / "capture.dylib"
            _compile_interposer(paths["interposer"], gsl_library, shim)
            private = directory / "source"
            _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
            exact_path = directory / "exact.dylib"
            _compile(
                private, args.headers, args.gsl_include, libraries,
                paths["exact_state_adapter"], exact_path,
                source_plan["full_generator_source_includes"],
                source_plan["full_generator_public_includes"],
                require_wheel_flux=False,
            )
            # A controlled one-call ablation: keep the source state, H, and
            # coefficients, but feed the wheel's captured omega to cbrt(v^3).
            omega_adapter = directory / "wheel_omega_mode_adapter.c"
            omega_source = paths["exact_state_adapter"].read_text()
            substitutions = (
                ("int diffgw_v4_source_nqc_state(",
                 "int diffgw_v4_source_nqc_state_wheel_omega("),
                ("double r, double phi, double pr_star, double pphi, double *out) {",
                 "double r, double phi, double pr_star, double pphi, "
                 "double omega_override, double *out) {"),
                ("&mode, &values, cbrt(out[0]), H, 2, 2, &params, 0)",
                 "&mode, &values, cbrt(omega_override), H, 2, 2, &params, 0)"),
                ("nu, &x, &p, &s1, &s2, &sk, &ss, 1, &coeffs);\n  REAL8Vector values",
                 "nu, &x, &p, &s1, &s2, &sk, &ss, 1, &coeffs);\n"
                 "  out[3] = H;\n  REAL8Vector values"),
            )
            for old, new in substitutions:
                if omega_source.count(old) != 1:
                    raise ValueError("Pinned explicit-state adapter changed")
                omega_source = omega_source.replace(old, new)
            omega_adapter.write_text(omega_source)
            report["wheel_omega_adapter_sha256"] = q.sha256(omega_adapter)
            omega_path = directory / "wheel_omega_mode.dylib"
            _compile(
                private, args.headers, args.gsl_include, libraries,
                omega_adapter, omega_path,
                source_plan["full_generator_source_includes"],
                source_plan["full_generator_public_includes"],
                require_wheel_flux=False,
            )
            omega_fast_path = directory / "wheel_omega_mode_fast.dylib"
            _compile(
                private, args.headers, args.gsl_include, libraries,
                omega_adapter, omega_fast_path,
                source_plan["full_generator_source_includes"],
                source_plan["full_generator_public_includes"],
                require_wheel_flux=False, extra_flags=("-ffast-math",),
            )
            hamiltonian_path = directory / "source_wheel_hamiltonian.dylib"
            compile_hamiltonian_adapter(
                args.headers, args.gsl_include, private / "lalsimulation/lib", libraries,
                hamiltonian_path,
            )
            high_path = directory / "source_high.dylib"
            _compile(
                private, args.headers, args.gsl_include, libraries,
                paths["source_high_adapter"], high_path,
                source_plan["full_generator_source_includes"],
                source_plan["full_generator_public_includes"],
                require_wheel_flux=False,
            )
            fit_path = directory / "fit.dylib"
            compile_wrapper(
                private, args.headers, args.gsl_include,
                [str(path) for path in libraries.values()], fit_path,
            )
            exact = ctypes.CDLL(str(exact_path)).diffgw_v4_source_nqc_state
            exact.argtypes = [ctypes.c_double] * 8 + [ctypes.POINTER(ctypes.c_double)]
            exact.restype = ctypes.c_int
            wheel_omega_mode = ctypes.CDLL(str(omega_path)).diffgw_v4_source_nqc_state_wheel_omega
            wheel_omega_mode.argtypes = [ctypes.c_double] * 9 + [ctypes.POINTER(ctypes.c_double)]
            wheel_omega_mode.restype = ctypes.c_int
            wheel_omega_mode_fast = ctypes.CDLL(
                str(omega_fast_path)).diffgw_v4_source_nqc_state_wheel_omega
            wheel_omega_mode_fast.argtypes = wheel_omega_mode.argtypes
            wheel_omega_mode_fast.restype = ctypes.c_int
            source_wheel_hamiltonian = ctypes.CDLL(
                str(hamiltonian_path)).diffgw_v4_hamiltonian_same_state
            source_wheel_hamiltonian.argtypes = [
                ctypes.c_double] * 7 + [ctypes.POINTER(ctypes.c_double)]
            source_wheel_hamiltonian.restype = ctypes.c_int
            compiler_adapters = {"-O2": exact}
            for label, optimization, flags in (
                ("-O0", "-O0", ()),
                ("-O3", "-O3", ()),
                ("-O2 -ffast-math", "-O2", ("-ffast-math",)),
            ):
                variant_path = directory / f"exact_{label.split()[0][1:]}{'_fast' if flags else ''}.dylib"
                _compile(
                    private, args.headers, args.gsl_include, libraries,
                    paths["exact_state_adapter"], variant_path,
                    source_plan["full_generator_source_includes"],
                    source_plan["full_generator_public_includes"],
                    optimization=optimization, require_wheel_flux=False,
                    extra_flags=flags,
                )
                function = ctypes.CDLL(str(variant_path)).diffgw_v4_source_nqc_state
                function.argtypes = exact.argtypes
                function.restype = ctypes.c_int
                compiler_adapters[label] = function
            source_high = ctypes.CDLL(str(high_path)).diffgw_v4_full_source_high_dynamics
            source_high.argtypes = [
                ctypes.c_double] * 6 + [ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double)]
            source_high.restype = ctypes.c_int
            c_fit, c_peak_offset = bind(fit_path)
            pointer = ctypes.POINTER(ctypes.c_double)

            for name in plan["rows"]:
                row_data = next(item for item in frozen["rows"] if item["name"] == name)
                row = row_data["row"]
                public, _ = public_call(row, plan["step_M"], 0)
                high = np.asarray(public[2].data, dtype=np.float64).reshape(5, -1)
                high_hash = q.canonical_hash(high.tolist())
                expected = exact_rows[name]
                if (high_hash != plan["public_high_sha256"][name]
                        or high.shape[1] != expected["wheel"]["high_samples"]):
                    raise ValueError(f"Pinned public high-rate dynamics changed: {name}")
                capture = directory / f"{name}.bin"
                env = os.environ.copy()
                env["DYLD_INSERT_LIBRARIES"] = str(shim)
                env["DIFFGW_GSL_CAPTURE_PATH"] = str(capture)
                env["DIFFGW_GSL_CAPTURE_SIZE"] = str(high.shape[1])
                env["PYTHONPATH"] = os.pathsep.join((str(q.ROOT / "src"), str(q.ROOT)))
                child = subprocess.run(
                    [sys.executable, "-m", "dev.native_rom.diagnose_wheel_nqc_spline_capture",
                     "--capture-child", name, "--step-M", str(plan["step_M"])],
                    cwd=q.ROOT, env=env, check=True, capture_output=True, text=True,
                )
                reported = np.asarray(json.loads(child.stdout), dtype=np.float64)[list(ACTIVE)]
                frozen_coefficients = np.asarray(row_data["wheel_nqc_coefficients"])[list(ACTIVE)]
                if np.max(abs(reported - frozen_coefficients)) > 1e-9:
                    raise ValueError(f"Pinned wheel NQC coefficients changed: {name}")
                columns, capture_hash = _read_capture(
                    capture, plan["expected_spline_calls"], high.shape[1],
                    plan["step_M"] / 8.0,
                )
                # Source call order: orbital-frequency peak, Q3/Q4/Q5,
                # amplitude, zero QNS, P3/P4, mode phase, zero PNS.
                omega, q3, q4, q5, amp, qns, p3, p4, phase, pns = columns
                radius = q3 / q4
                radial_momentum = p3 * radius * omega
                radius_gap = float(np.max(abs(radius - high[1])))
                radial_gap = float(np.max(abs(radial_momentum - high[3])))
                if (radius_gap > 1e-12 or radial_gap > 1e-12
                        or np.max(abs(qns)) != 0 or np.max(abs(pns)) != 0):
                    raise ValueError(f"Wheel private NQC states differ from public states: {name}")
                axis = np.arange(high.shape[1], dtype=np.float64) * (plan["step_M"] / 8.0)
                peak_indices = local_indices(
                    axis, expected["wheel"]["peak_rule"]["time_M"],
                    plan["local_half_width_M"],
                )
                peak_result = source_peak_rule(gsl, axis[peak_indices], omega[peak_indices])
                peak = float(peak_result["time_M"])
                masses = [row[key] for key in ("mass1", "mass2", "chi1", "chi2")]
                offset = c_peak_offset(
                    masses[0], masses[1], (masses[2] - masses[3]) / 2,
                    (masses[2] + masses[3]) / 2,
                )
                match = peak - offset

                def fit(inputs: dict[str, np.ndarray]) -> np.ndarray:
                    arrays = [np.ascontiguousarray(value) for value in (
                        inputs["amp"], inputs["phase"], high[1], high[3], inputs["omega"],
                    )]
                    result = np.empty(5, dtype=np.float64)
                    rc = c_fit(
                        *(array.ctypes.data_as(pointer) for array in arrays),
                        len(amp), peak, plan["step_M"] / 8.0,
                        masses[0], masses[1], (masses[2] - masses[3]) / 2,
                        (masses[2] + masses[3]) / 2,
                        result.ctypes.data_as(pointer),
                    )
                    if rc or not np.isfinite(result).all():
                        raise RuntimeError(f"Pinned source C NQC fit failed: {name}, {rc}")
                    return result

                wheel_inputs = {"omega": omega, "amp": amp, "phase": phase}
                wheel_fit = fit(wheel_inputs)
                wheel_l2 = float(np.linalg.norm(wheel_fit - reported))
                if wheel_l2 > 1e-6:
                    raise ValueError(f"Captured wheel NQC fit does not replay: {name}, {wheel_l2}")
                values = (ctypes.c_double * 4)()
                raw_mode = np.empty(high.shape[1], dtype=np.complex128)
                source_omega = np.empty(high.shape[1], dtype=np.float64)
                for index in range(high.shape[1]):
                    if exact(*masses, *high[1:, index], values):
                        raise RuntimeError(f"Pinned source raw mode failed: {name}/{index}")
                    source_omega[index] = values[0]
                    raw_mode[index] = complex(values[1], values[2])
                source_phase = np.unwrap(np.angle(raw_mode))
                source_phase += phase[0] - source_phase[0]
                source_inputs = {
                    "omega": source_omega, "amp": abs(raw_mode), "phase": source_phase,
                }
                wheel_omega_raw_mode = np.empty(high.shape[1], dtype=np.complex128)
                wheel_omega_fast_mode_amplitude = np.empty(high.shape[1], dtype=np.float64)
                wheel_omega_source_frequency = np.empty(high.shape[1], dtype=np.float64)
                wheel_omega_source_hamiltonian = np.empty(high.shape[1], dtype=np.float64)
                for index in range(high.shape[1]):
                    if wheel_omega_mode(*masses, *high[1:, index], float(omega[index]), values):
                        raise RuntimeError(f"Wheel-omega source mode failed: {name}/{index}")
                    wheel_omega_source_frequency[index] = values[0]
                    wheel_omega_raw_mode[index] = complex(values[1], values[2])
                    wheel_omega_source_hamiltonian[index] = values[3]
                    if wheel_omega_mode_fast(*masses, *high[1:, index],
                                             float(omega[index]), values):
                        raise RuntimeError(f"Fast wheel-omega mode failed: {name}/{index}")
                    wheel_omega_fast_mode_amplitude[index] = abs(
                        complex(values[1], values[2]))
                if np.max(abs(wheel_omega_source_frequency - source_omega)) > 1e-14:
                    raise ValueError(f"Wheel-omega ablation changed source frequency: {name}")
                wheel_omega_mode_amplitude = abs(wheel_omega_raw_mode)
                wheel_omega_mode_phase = np.unwrap(np.angle(wheel_omega_raw_mode))
                wheel_omega_mode_phase += phase[0] - wheel_omega_mode_phase[0]
                source_internal_control = None
                if name == "strong_aligned_spin":
                    with np.load(paths["source_nqc_inputs"]) as source_npz:
                        captured_source = source_npz["values"]
                    capacity = 50000
                    source_buffer = np.empty(5 * capacity, dtype=np.float64)
                    source_count = ctypes.c_size_t()
                    rc = source_high(
                        *masses, row["omega_start"], plan["step_M"], capacity,
                        ctypes.byref(source_count), source_buffer.ctypes.data_as(pointer),
                    )
                    if rc or captured_source.shape != (source_count.value, 5):
                        raise RuntimeError("Pinned source internal control changed")
                    source_states = source_buffer[:5 * source_count.value].reshape(5, -1)
                    source_indices = np.flatnonzero(
                        abs(np.arange(source_count.value) * (plan["step_M"] / 8.0)
                            - expected["source"]["match_time_M"])
                        <= plan["local_half_width_M"],
                    )
                    source_input_error = np.empty((len(source_indices), 2))
                    for j, index in enumerate(source_indices):
                        if exact(*masses, *source_states[1:, index], values):
                            raise RuntimeError(f"Pinned source control failed at {index}")
                        source_input_error[j] = (
                            abs(complex(values[1], values[2])) - captured_source[index, 0],
                            values[0] - captured_source[index, 4],
                        )
                    source_internal_control = {
                        "sample_count": len(source_indices),
                        "amplitude_max_abs": float(np.max(abs(source_input_error[:, 0]))),
                        "orbital_frequency_max_abs": float(np.max(abs(source_input_error[:, 1]))),
                    }
                    if max(source_internal_control["amplitude_max_abs"],
                           source_internal_control["orbital_frequency_max_abs"]) > 1e-13:
                        raise ValueError("Explicit-state adapter differs from source generator")
                variants = {}
                for label, replaced in (
                    ("wheel", ()), ("amplitude", ("amp",)),
                    ("phase", ("phase",)), ("orbital_frequency", ("omega",)),
                    ("all_source_inputs", ("amp", "phase", "omega")),
                ):
                    inputs = {key: (source_inputs[key] if key in replaced else wheel_inputs[key])
                              for key in wheel_inputs}
                    coefficients = fit(inputs)
                    variants[label] = {
                        "coefficients": coefficients.tolist(),
                        "minus_wheel_reported": (coefficients - reported).tolist(),
                        "minus_wheel_reported_l2": float(np.linalg.norm(coefficients - reported)),
                    }
                near = abs(axis - match) <= plan["local_half_width_M"]
                total_mass = masses[0] + masses[1]
                x1, x2 = masses[0] / total_mass, masses[1] / total_mass
                nu = x1 * x2
                hamiltonian_comparison = np.empty((int(np.count_nonzero(near)), 2))
                hamiltonian_values = (ctypes.c_double * 5)()
                for offset, index in enumerate(np.flatnonzero(near)):
                    rc = source_wheel_hamiltonian(
                        x1, x2, masses[2], masses[3],
                        float(high[1, index]), float(high[3, index]),
                        float(high[4, index]), hamiltonian_values,
                    )
                    if rc:
                        raise RuntimeError(f"Wheel Hamiltonian control failed: {name}/{index}/{rc}")
                    source_mode_h_over_nu = wheel_omega_source_hamiltonian[index] / nu
                    hamiltonian_comparison[offset] = (
                        hamiltonian_values[0] - source_mode_h_over_nu,
                        hamiltonian_values[1] - source_mode_h_over_nu,
                    )
                wheel_omega_amplitude_fit = fit({
                    "omega": omega, "amp": wheel_omega_mode_amplitude, "phase": phase,
                })
                wheel_omega_mode_control = {
                    "sample_count": int(np.count_nonzero(near)),
                    "amplitude_minus_source_max_abs": float(np.max(abs(
                        wheel_omega_mode_amplitude[near] - source_inputs["amp"][near]))),
                    "amplitude_minus_wheel_max_abs": float(np.max(abs(
                        wheel_omega_mode_amplitude[near] - amp[near]))),
                    "amplitude_minus_wheel_rms": float(np.sqrt(np.mean(np.square(
                        wheel_omega_mode_amplitude[near] - amp[near])))),
                    "fast_math_amplitude_minus_wheel_max_abs": float(np.max(abs(
                        wheel_omega_fast_mode_amplitude[near] - amp[near]))),
                    "fast_math_amplitude_minus_wheel_rms": float(np.sqrt(np.mean(np.square(
                        wheel_omega_fast_mode_amplitude[near] - amp[near])))),
                    "phase_minus_wheel_max_abs": float(np.max(abs(
                        wheel_omega_mode_phase[near] - phase[near]))),
                    "amplitude_only_fit_minus_wheel_l2": float(np.linalg.norm(
                        wheel_omega_amplitude_fit - reported)),
                    "source_hamiltonian_wrapper_minus_source_mode_max_abs": float(
                        np.max(abs(hamiltonian_comparison[:, 0]))),
                    "wheel_hamiltonian_wrapper_minus_source_mode_max_abs": float(
                        np.max(abs(hamiltonian_comparison[:, 1]))),
                }
                compiler_controls = {}
                near_indices = np.flatnonzero(near)
                for label in plan["compiler_controls"]:
                    function = compiler_adapters[label]
                    amplitude_error = []
                    orbital_frequency_error = []
                    amplitude_shift_from_O2 = []
                    for index in near_indices:
                        if function(*masses, *high[1:, index], values):
                            raise RuntimeError(f"Compiler control failed: {name}/{label}/{index}")
                        variant_amplitude = abs(complex(values[1], values[2]))
                        amplitude_error.append(variant_amplitude - amp[index])
                        orbital_frequency_error.append(values[0] - omega[index])
                        amplitude_shift_from_O2.append(variant_amplitude - source_inputs["amp"][index])
                    amplitude_error = np.asarray(amplitude_error)
                    orbital_frequency_error = np.asarray(orbital_frequency_error)
                    amplitude_shift_from_O2 = np.asarray(amplitude_shift_from_O2)
                    compiler_controls[label] = {
                        "sample_count": len(near_indices),
                        "amplitude_minus_wheel_max_abs": float(np.max(abs(amplitude_error))),
                        "amplitude_minus_wheel_rms": float(np.sqrt(np.mean(np.square(amplitude_error)))),
                        "orbital_frequency_minus_wheel_max_abs": float(np.max(abs(orbital_frequency_error))),
                        "orbital_frequency_minus_wheel_rms": float(np.sqrt(np.mean(np.square(orbital_frequency_error)))),
                        "amplitude_minus_O2_max_abs": float(np.max(abs(amplitude_shift_from_O2))),
                    }
                report["rows"][name] = {
                    "row": row,
                    "public_high_dynamics_sha256": high_hash,
                    "private_gsl_capture_sha256": capture_hash,
                    "private_gsl_spline_length": int(high.shape[1]),
                    "private_minus_public_radius_max_abs": radius_gap,
                    "private_minus_public_radial_momentum_max_abs": radial_gap,
                    "wheel_peak_rule": peak_result,
                    "wheel_match_time_M": match,
                    "wheel_reported_coefficients": reported.tolist(),
                    "wheel_fit_replay_l2": wheel_l2,
                    "same_state_source_minus_wheel_input_max_abs": {
                        key: float(np.max(abs(source_inputs[key][near] - wheel_inputs[key][near])))
                        for key in wheel_inputs
                    },
                    "source_internal_control": source_internal_control,
                    "wheel_omega_mode_control": wheel_omega_mode_control,
                    "compiler_controls": compiler_controls,
                    "fit_ablations": variants,
                }
                q.write_evidence(args.json_out, report)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "diagnostic_failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        q.write_evidence(args.json_out, report)
        raise
    q.write_evidence(args.json_out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
