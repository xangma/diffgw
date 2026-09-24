"""Decompose q8 late-inspiral phase on source and native trajectory states.

All source calls are compiled from the pinned source snapshot.  This is an
audit-only state/raw-mode counterfactual, not a generator implementation.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time

import numpy as np
import torch
import lal

from . import qualify_generator as q
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
from .eob.torch.source_calibration_hamiltonian import SourceCalibratedHistoricalHamiltonian
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_mode22 import (
    seobnrv4_published_mode22_from_hamiltonian,
)


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_q8_raw_phase_plan.json"


def _array(value):
    result = value.detach().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("nonfinite native array")
    return result


def _digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _state_at(times, states, probe):
    if probe < times[0] or probe > times[-1] or not np.all(np.diff(times) > 0):
        raise ValueError("state probe outside ordered trajectory")
    return np.asarray([np.interp(probe, times, states[:, i]) for i in range(4)])


def _relative_raw_phase(modes, states):
    """Unwrap slowly varying residual after removing the known -2 phi factor."""
    modes = np.asarray(modes)
    phi = np.asarray(states)[:, 1]
    residual = np.unwrap(np.angle(modes*np.exp(2j*phi)))
    return -2*phi + residual


def _anchored(values, index):
    values = np.asarray(values, dtype=float)
    return values-values[index]


def _nqc_phase(state, omega, coeff):
    r, _, pr, _ = state
    ratio = pr/(r*omega)
    return float(coeff[0]*ratio + coeff[1]*ratio*pr*pr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT/path for name, path in plan["inputs"].items()}
    probes = np.asarray(plan["probe_times_M"], dtype=float)
    if (plan["schema"] != "diffgw.seobnrv4.source-q8-raw-phase-plan.v1"
            or plan["row"] != "q8_high_aligned" or plan["level"] != "ode_2"
            or plan["output_step_M"] != 0.5
            or plan["phase_anchor_M"] != -100.
            or probes.tolist() != [-120., -100., -80., -60., -40., -25.]
            or plan["release_qualified"] is not False
            or {name: q.sha256(path) for name, path in paths.items()}
               != plan["input_sha256"]):
        parser.error("frozen raw-phase plan or inputs differ")
    campaign = json.loads(paths["campaign"].read_text())
    profile = json.loads(paths["phase_profile"].read_text())
    previous = json.loads(paths["source_native_mode"].read_text())
    attachment = json.loads(paths["source_attachment"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    row = campaign["rows"][plan["row"]]["row"]
    if (any(x["status"] != "diagnostic_complete" for x in
            (campaign, profile, previous, attachment))
            or profile["row"] != row
            or q.lal_binary_identity(campaign["lal_binary"]["lal_version"],
                                     campaign["lal_binary"]["lalsimulation_version"])
               != campaign["lal_binary"]):
        parser.error("pinned q8 evidence or wheel changed")
    scope = dict(q.SCOPE)
    scope["description"] = "Diagnostic q8 late-inspiral raw-mode and trajectory phase decomposition."
    scope["known_limitations"] = [
        "Six late-inspiral probes on one q8 high-spin row do not qualify the model.",
        "One fixed phase anchor removes the arbitrary phase gauge; no parameters are fit.",
        "Source adapters are separately compiled and are not native generator dependencies.",
        "Mode values are sampled from linearly interpolated trajectory states.",
    ]
    report = dict(schema="diffgw.seobnrv4.source-q8-raw-phase.v1",
                  status="running", release_qualified=False, scope=scope,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256=plan["input_sha256"], row=row, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        torch.set_num_threads(1)
        started = time.monotonic()
        settings = q.Settings(**campaign["settings"])
        choice = campaign["choice"]
        level = campaign["rows"][plan["row"]]["runs"][plan["level"]]["level"]
        numerical, flux_choice, physical = (choice[key] for key in
                                            ("numerical", "flux", "physical"))
        with q.bounded_run(settings.max_run_seconds), torch.no_grad():
            native = reconstruct_seobnrv4_22_adaptive(
                torch.tensor(row["omega_start"], dtype=torch.float64),
                **{key: row[key] for key in ("mass1", "mass2", "chi1", "chi2")},
                **dict(q.CHOICES, spin_truncation=choice["spin_truncation"]),
                **{key: level[key] for key in ("atol", "rtol", "dt_geom")},
                diagnostic_hermite_resampling=numerical["hermite_resampling"],
                diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
                diagnostic_matching_second_derivatives=True,
                diagnostic_restart_time_M=campaign["rows"][plan["row"]][
                    "fixed_restart_time_M"],
                diagnostic_coarse_initial_step_M=0.125,
                diagnostic_warm_start_Momega=0.03162277660168379,
                diagnostic_nqc_phase_window_M=choice["nqc_phase_window_M"],
                diagnostic_mode_21_higher_spin=flux_choice["mode_21_higher_spin"],
                diagnostic_mode_31_higher_spin=flux_choice["mode_31_higher_spin"],
                diagnostic_mode_32_spin_square=flux_choice["mode_32_spin_square"],
                diagnostic_mode_33_higher_spin=physical["mode_33_higher_spin"],
                diagnostic_mode_43_higher_spin=choice["mode_43_higher_spin"],
                diagnostic_hamiltonian_log_derivative=physical[
                    "hamiltonian_log_derivative"],
                diagnostic_hamiltonian_calibration=choice["calibration"],
                ringdown_duration=settings.ringdown_duration_M,
                num_steps=settings.max_accepted_steps,
                max_accepted_steps=settings.max_accepted_steps,
                max_evaluations=settings.max_evaluations,
                max_rejected_steps=settings.max_rejected_steps)
        hashes = dict(time=_digest(_array(native.time)), mode=_digest(_array(native.mode)))
        if hashes != previous["rows"][plan["row"]]["native_array_sha256"]:
            raise ValueError("native waveform replay changed")
        trajectory = native.trajectory
        valid = _array(trajectory.valid).astype(bool)
        n = int(valid.sum())
        if not np.all(valid[:n]) or np.any(valid[n:]):
            raise ValueError("native trajectory validity is not a prefix")
        native_t = _array(trajectory.time)[:n]-float(_array(trajectory.peak.time))
        native_states = _array(trajectory.state)[:n]
        native_omega = _array(trajectory.frequency)[:n]
        report["native_array_sha256"] = hashes
        report["native_valid_samples"] = n
        q.write_evidence(args.json_out, report)
        print("native_q8_replayed", flush=True)
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-q8-raw-phase-") as tmp:
            private = Path(tmp)/"source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            libraries = _libraries(source_plan["pinned_wheel"])
            for name, includes, public in (
                    ("high_adapter", "full_generator_source_includes",
                     "full_generator_public_includes"),
                    ("exact_state_adapter", "full_generator_source_includes",
                     "full_generator_public_includes"),
                    ("mode_adapter", "full_generator_source_includes",
                     "full_generator_public_includes")):
                _compile(private, args.headers, args.gsl_include, libraries,
                         paths[name], Path(tmp)/(name+".dylib"),
                         source_plan[includes], source_plan[public],
                         require_wheel_flux=False)
            pointer = ctypes.POINTER(ctypes.c_double)
            high = ctypes.CDLL(str(Path(tmp)/"high_adapter.dylib")).diffgw_v4_full_source_high_dynamics
            high.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                             ctypes.POINTER(ctypes.c_size_t), pointer]
            high.restype = ctypes.c_int
            exact = ctypes.CDLL(str(Path(tmp)/"exact_state_adapter.dylib")).diffgw_v4_source_nqc_state
            exact.argtypes = [ctypes.c_double]*8 + [pointer]
            exact.restype = ctypes.c_int
            full = ctypes.CDLL(str(Path(tmp)/"mode_adapter.dylib")).diffgw_v4_full_source_mode
            full.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                             ctypes.POINTER(ctypes.c_size_t), pointer, pointer, pointer]
            full.restype = ctypes.c_int
            masses = [row[key] for key in ("mass1", "mass2", "chi1", "chi2")]
            capacity = plan["capacity"]
            buffer = np.empty(5*capacity, dtype=np.float64)
            count = ctypes.c_size_t()
            rc = high(*masses, row["omega_start"], plan["output_step_M"],
                      capacity, ctypes.byref(count), buffer.ctypes.data_as(pointer))
            if rc or count.value < 100:
                raise RuntimeError(f"pinned source high dynamics failed: {rc}")
            source_high = buffer[:5*count.value].reshape(5, -1).copy()
            source_peak = attachment["rows"][plan["row"]]["source"][
                "orbital_peak_time_M"]
            source_t = source_high[0]-source_high[0, 0]-source_peak
            if not np.all(np.diff(source_t) > 0):
                raise ValueError("source high-rate time is not increasing")
            mode_buffer = np.empty(2*capacity, dtype=np.float64)
            mode_count = ctypes.c_size_t()
            epoch, delta_t = ctypes.c_double(), ctypes.c_double()
            rc = full(*masses, row["omega_start"], plan["output_step_M"],
                      capacity, ctypes.byref(mode_count), ctypes.byref(epoch),
                      ctypes.byref(delta_t), mode_buffer.ctypes.data_as(pointer))
            if rc or mode_count.value < 100:
                raise RuntimeError(f"pinned source full mode failed: {rc}")
            source_mode = mode_buffer[:2*mode_count.value].copy().view(np.complex128)
            if _digest(source_mode) != previous["rows"][plan["row"]]["source_mode_sha256"]:
                raise ValueError("source full-mode replay changed")
            mass_seconds = (masses[0]+masses[1])*lal.MTSUN_SI
            source_mode_t = (epoch.value+np.arange(source_mode.size)*delta_t.value)/mass_seconds

            x1 = row["mass1"]/(row["mass1"]+row["mass2"])
            def scalar(x):
                return torch.tensor(x, dtype=torch.float64)
            ham = SourceCalibratedHistoricalHamiltonian(
                scalar(x1*(1-x1)), scalar(row["chi1"]), scalar(row["chi2"]),
                delta=scalar(2*x1-1))
            rr = SEOBNRv4PublishedFlux(
                spin_truncation=choice["spin_truncation"],
                retain_pan_cubic_spin=True,
                mode_21_higher_spin=flux_choice["mode_21_higher_spin"],
                mode_31_higher_spin=flux_choice["mode_31_higher_spin"],
                mode_32_spin_square=flux_choice["mode_32_spin_square"],
                mode_33_higher_spin=physical["mode_33_higher_spin"],
                mode_43_higher_spin=choice["mode_43_higher_spin"])
            def source_raw(state):
                values = (ctypes.c_double*3)()
                rc = exact(*masses, *state.tolist(), values)
                if rc:
                    raise RuntimeError(f"pinned source raw-state call failed: {rc}")
                return float(values[0]), complex(values[1], values[2])
            def native_raw(state, omega):
                with torch.no_grad():
                    mode = seobnrv4_published_mode22_from_hamiltonian(
                        torch.as_tensor(state, dtype=torch.float64), ham, rr,
                        omega=scalar(omega), spin_v6=q.CHOICES["spin_v6"],
                        include_pan_v8=False, include_positive_chi_540=False,
                        retain_pan_cubic_spin=True)
                return complex(mode.item())

            states_source, states_native = [], []
            mode_source_source, mode_source_native = [], []
            mode_native_native, mode_native_forced = [], []
            omega_source_source, omega_source_native, omega_native = [], [], []
            nqc_source, nqc_native = [], []
            coefficients = previous["rows"][plan["row"]]["nqc_coefficients"]
            source_coeff = np.asarray(coefficients["source"][-2:])
            native_coeff = np.asarray(coefficients["native"][-2:])
            for probe in probes:
                ss = _state_at(source_t, source_high[1:].T, probe)
                sn = _state_at(native_t, native_states, probe)
                ws, hs = source_raw(ss)
                wn_source, hn_source = source_raw(sn)
                wn = float(np.interp(probe, native_t, native_omega))
                hn = native_raw(sn, wn)
                hn_forced = native_raw(sn, wn_source)
                states_source.append(ss)
                states_native.append(sn)
                mode_source_source.append(hs)
                mode_source_native.append(hn_source)
                mode_native_native.append(hn)
                mode_native_forced.append(hn_forced)
                omega_source_source.append(ws)
                omega_source_native.append(wn_source)
                omega_native.append(wn)
                nqc_source.append(_nqc_phase(ss, ws, source_coeff))
                nqc_native.append(_nqc_phase(sn, wn, native_coeff))
            anchor = int(np.flatnonzero(probes == plan["phase_anchor_M"])[0])
            ps = _relative_raw_phase(mode_source_source, states_source)
            psn = _relative_raw_phase(mode_source_native, states_native)
            pn = _relative_raw_phase(mode_native_native, states_native)
            pnf = _relative_raw_phase(mode_native_forced, states_native)
            trajectory_term = _anchored(psn-ps, anchor)
            same_state_term = _anchored(pnf-psn, anchor)
            frequency_term = _anchored(pn-pnf, anchor)
            nqc_term = _anchored(np.asarray(nqc_native)-nqc_source, anchor)
            modeled = trajectory_term+same_state_term+frequency_term+nqc_term
            reconstructed_source = _anchored(ps+np.asarray(nqc_source), anchor)
            reconstructed_native = _anchored(pn+np.asarray(nqc_native), anchor)
            source_mode_phase = _anchored(np.interp(
                probes, source_mode_t, np.unwrap(np.angle(source_mode))), anchor)
            native_mode_t = _array(native.time)-float(_array(trajectory.peak.time))
            native_mode_phase = _anchored(np.interp(
                probes, native_mode_t, np.unwrap(np.angle(_array(native.mode)))),
                anchor)
            source_reconstruction_error = source_mode_phase-reconstructed_source
            native_reconstruction_error = native_mode_phase-reconstructed_native
            observed_times = np.asarray(profile["probe_times_M"])
            observed = np.asarray(profile["pairs"]["native_minus_source"][
                "phase_from_anchor_rad"])
            indices = [int(np.flatnonzero(observed_times == p)[0]) for p in probes]
            observed = observed[indices]
            residual = observed-modeled
            actual_pair = native_mode_phase-source_mode_phase
            if np.max(abs(actual_pair-observed)) > 1e-10:
                raise ValueError("full-mode phase replay differs from pinned profile")
            if np.max(abs(residual-(native_reconstruction_error
                                    -source_reconstruction_error))) > 1e-10:
                raise ValueError("phase reconstruction residual does not close")
            report.update(
                probe_times_M=probes.tolist(),
                source_high_array_sha256=_digest(source_high),
                source_mode_sha256=_digest(source_mode),
                source_high_samples=int(source_high.shape[1]),
                source_high_time_support_M=[float(source_t[0]), float(source_t[-1])],
                source_state=[state.tolist() for state in states_source],
                native_state=[state.tolist() for state in states_native],
                source_omega_on_source_state=omega_source_source,
                source_omega_on_native_state=omega_source_native,
                native_omega_on_native_state=omega_native,
                source_raw_on_source_state=[[z.real, z.imag] for z in mode_source_source],
                source_raw_on_native_state=[[z.real, z.imag] for z in mode_source_native],
                native_raw_on_native_state=[[z.real, z.imag] for z in mode_native_native],
                trajectory_phase_rad=trajectory_term.tolist(),
                same_state_raw_phase_rad=same_state_term.tolist(),
                native_frequency_phase_rad=frequency_term.tolist(),
                nqc_phase_rad=nqc_term.tolist(),
                modeled_phase_rad=modeled.tolist(),
                observed_phase_rad=observed.tolist(),
                source_reconstruction_error_rad=source_reconstruction_error.tolist(),
                native_reconstruction_error_rad=native_reconstruction_error.tolist(),
                residual_phase_rad=residual.tolist(),
                residual_rms_rad=float(np.sqrt(np.mean(residual**2))),
                residual_max_abs_rad=float(np.max(abs(residual))),
                elapsed_seconds=time.monotonic()-started)
            if not np.isfinite(modeled).all() or not np.isfinite(residual).all():
                raise ValueError("nonfinite phase decomposition")
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          residual_max_abs_rad=report.get("residual_max_abs_rad"))),
          flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
