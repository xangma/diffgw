"""Development-only exact-row source/native early raw-mode crossover."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import lal
import numpy as np
import torch

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_wheel_dynamics import _source as source_dynamics
from dev.native_rom.diagnose_source_wheel_mode import _source as source_mode, _source_nqc
from torchwave.approximants.eob.nqc import NQCCoefficients, nqc_22_factor


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_aligned_early_raw_crossover_plan.json"
OUTPUT = EVIDENCE / "source_aligned_early_raw_crossover.json"
SOURCE_ROOT = Path("/Users/xangma/repos/lalsuite-git")
HEADERS = Path("/tmp/diffgw-lalsuite-pinned-53acc0c/merged-include")
GSL_INCLUDE = Path("/usr/local/include")
ROW = (30.0, 20.0, 0.8, 0.6, 0.04)


def as_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def rel_l2(left, right):
    return float(np.linalg.norm(left - right) / np.linalg.norm(right))


def source_raw(function, row, states):
    out = (ctypes.c_double * 3)()
    values = np.empty((len(states), 3), dtype=np.float64)
    for i, state in enumerate(states):
        rc = function(*row[:4], *map(float, state), out)
        if rc:
            raise RuntimeError(f"source raw mode failed at {i}: {rc}")
        values[i] = out[:]
    return values[:, 0], values[:, 1] + 1j * values[:, 2]


def native_raw(hamiltonian, flux, states):
    parts = []
    for start in range(0, len(states), 64):
        tensor = torch.as_tensor(states[start:start + 64], dtype=torch.float64)
        mode, _ = driver.torch_mode(
            tensor, hamiltonian, flux,
            **(driver.MODE_OPTIONS | {"retain_pan_cubic_spin": True}),
        )
        parts.append(as_numpy(mode))
    return np.concatenate(parts)


def frequency_states(omega, time, state, target):
    crossings = np.flatnonzero((omega[:-1] <= target[-1]) & (omega[1:] >= target[-1]))
    if len(crossings) != 1 or omega[0] >= target[0]:
        raise ValueError("fixed-frequency window has no unique early crossing")
    end = int(crossings[0])+2
    if not np.all(np.diff(omega[:end]) > 0):
        raise ValueError("early fixed-frequency chart is outside monotone trajectory")
    chart = np.column_stack([np.interp(target, omega[:end], state[:end, i])
                             for i in range(4)])
    return chart, np.interp(target, omega[:end], time[:end]), end


def normalized(mode):
    return mode * np.conj(mode[0]) / abs(mode[0])


def main():
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-early-raw-crossover-plan.v1"
            or plan["release_qualified"] is not False
            or tuple(plan["row_tuple"]) != ROW
            or plan["script_sha256"] != q.sha256(Path(__file__))
            or plan["step_M"] != 0.5):
        raise ValueError("frozen plan changed")
    for item in plan["inputs"].values():
        if q.sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen input changed: {item['path']}")
    source_plan = json.loads((ROOT / plan["inputs"]["source_plan"]["path"]).read_text())
    ladder = json.loads((ROOT / plan["inputs"]["ladder_plan"]["path"]).read_text())
    prior = json.loads((ROOT / plan["inputs"]["same_row_source_result"]["path"]).read_text())
    if (tuple(prior["row_tuple"]) != ROW or prior["source_step_M"] != plan["step_M"]
            or prior["source_nqc_coefficients_at_step_M"] != plan["source_coefficients"]):
        raise ValueError("same-row source coefficients disagree with frozen plan")
    level = next(item for item in ladder["levels"] if item["name"] == "refined")
    libraries = _libraries(source_plan["pinned_wheel"])
    torch.set_num_threads(1)
    m1, m2, chi1, chi2, omega_start = ROW
    row = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, omega_start=omega_start)
    started = time.monotonic()
    result = dict(schema="diffgw.seobnrv4.source-aligned-early-raw-crossover.v1",
                  status="running", release_qualified=False,
                  row=row, host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(Path(__file__)),
                  checks={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)
    nu = m1*m2/(m1+m2)**2
    ham = driver.TorchSourceHam(tensor(nu), tensor(chi1), tensor(chi2),
                                delta=tensor((m1-m2)/(m1+m2)))
    flux = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
    r0, pr0, pphi0 = driver.find_quasicircular_initial_conditions(
        ham, flux, tensor(omega_start))
    initial = torch.stack((r0, torch.zeros_like(r0), pr0, pphi0)).detach()
    field = driver.TorchIntegrator(ham, flux).vector_field
    coarse = driver.integrate_adaptive(
        field, initial, plan["step_M"], t_end=1000.0,
        event=driver.AlignedSpinV4Event("coarse"), initial_step=plan["step_M"],
        max_step=ladder["max_step_M"], controller="gsl_state_scaled",
        atol=level["atol"], rtol=level["rtol"])
    native_time = as_numpy(coarse.output_time)
    native_state = as_numpy(coarse.output_state)
    with tempfile.TemporaryDirectory(prefix="diffgw-v4-early-crossover-") as tmp:
        private = Path(tmp) / "source"
        _snapshot(SOURCE_ROOT, source_plan["pinned_source"], private, HEADERS)
        def build(filename, adapter):
            library = Path(tmp) / filename
            _compile(private, HEADERS, GSL_INCLUDE, libraries, ROOT / adapter, library,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            return ctypes.CDLL(str(library))
        dynamics = build("dynamics.dylib", plan["inputs"]["dynamics_adapter"]["path"])
        dyn = dynamics.diffgw_v4_full_source_dynamics
        dyn.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                       ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double)]
        dyn.restype = ctypes.c_int
        source_row = source_dynamics(dyn, row, plan["step_M"], 50000)
        exact = build("exact.dylib", plan["inputs"]["exact_adapter"]["path"])
        exact_fn = exact.diffgw_v4_source_nqc_state
        exact_fn.argtypes = [ctypes.c_double]*8 + [ctypes.POINTER(ctypes.c_double)]
        exact_fn.restype = ctypes.c_int
        source_time, source_state = source_row[0], source_row[1:].T
        source_omega, source_mode_raw = source_raw(exact_fn, ROW, source_state)
        native_source_omega, _ = source_raw(exact_fn, ROW, native_state)
        target = np.asarray(plan["fixed_orbital_frequency_Momega"], dtype=np.float64)
        source_chart_state, source_chart_time, source_early_count = frequency_states(
            source_omega, source_time, source_state, target)
        native_chart_state, native_chart_time, native_early_count = frequency_states(
            native_source_omega, native_time, native_state, target)
        verified_omega, source_on_source = source_raw(exact_fn, ROW, source_chart_state)
        native_on_source = native_raw(ham, flux, source_chart_state)
        native_on_native = native_raw(ham, flux, native_chart_state)
        full = build("full.dylib", plan["inputs"]["full_mode_adapter"]["path"])
        full_fn = full.diffgw_v4_full_source_mode
        full_fn.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        full_fn.restype = ctypes.c_int
        epoch, delta_t, corrected = source_mode(full_fn, row, plan["step_M"], 50000)
        full_nqc = full.diffgw_v4_full_source_nqc
        full_nqc.argtypes = [ctypes.c_double]*6 + [ctypes.POINTER(ctypes.c_double)]
        full_nqc.restype = ctypes.c_int
        actual_coefficients = _source_nqc(full_nqc, row, plan["step_M"])
    # The full-mode output starts at a separately defined epoch. Compare only
    # its first coarse samples, and retain this as a conditional mapping check.
    coeff = np.asarray(plan["source_coefficients"], dtype=np.float64)
    coeff_t = NQCCoefficients(tensor(coeff[:3]), tensor(coeff[6:8]))
    count = min(len(source_state), len(corrected), 256)
    factor = as_numpy(nqc_22_factor(tensor(source_state[:count, 0]),
                                      tensor(source_state[:count, 2]),
                                      tensor(source_omega[:count]), coeff_t,
                                      sample_dims=1))
    scale = (m1+m2)*lal.MRSUN_SI/(100e6*lal.PC_SI)
    reconstructed = source_mode_raw[:count]*factor*scale
    source_full_control = {
        "samples": count, "direct_first_samples_l2": rel_l2(reconstructed, corrected[:count]),
        "direct_first_samples_amplitude_l2": rel_l2(abs(reconstructed), abs(corrected[:count])),
        "mapping_assumption": "Full source mode index zero corresponds to coarse dynamics index zero; no fitted index shift.",
        "epoch_s": epoch, "delta_t_s": delta_t,
    }
    checks = {
        "exact_row": tuple(row.values()) == ROW,
        "finite": all(np.isfinite(x).all() for x in (
            source_row, native_state, source_chart_state, native_chart_state,
            source_on_source, native_on_source, native_on_native)),
        "source_cadence": np.max(abs(np.diff(source_time)-plan["step_M"])) < 1e-10,
        "native_cadence": np.max(abs(np.diff(native_time)-plan["step_M"])) < 1e-10,
        "interpolated_source_omega": np.max(abs(verified_omega-target))
            < plan["gates"]["interpolated_source_omega_max_abs"],
        "early_time_margin": max(source_chart_time[-1], native_chart_time[-1])
            < min(source_time[-1], native_time[-1])-plan["gates"]["terminal_margin_M"],
        "source_nqc_coefficient_replay": np.max(abs(actual_coefficients-coeff))
            < plan["gates"]["source_nqc_coefficient_max_abs"],
    }
    result.update(status="diagnostic_complete" if all(checks.values()) else "failed",
                  elapsed_s=time.monotonic()-started, checks=checks,
                  diagnostic_checks_passed=all(checks.values()),
                  source_samples=int(len(source_time)), native_samples=int(len(native_time)),
                  source_full_mode_samples=int(len(corrected)),
                  source_full_control=source_full_control,
                  source_nqc_coefficient_replay_max_abs=float(np.max(abs(actual_coefficients-coeff))),
                  source_early_count=source_early_count, native_early_count=native_early_count,
                  chart={"Momega": target.tolist(),
                         "source_time_M": source_chart_time.tolist(),
                         "native_time_M": native_chart_time.tolist(),
                         "source_state": source_chart_state.tolist(),
                         "native_state": native_chart_state.tolist(),
                         "source_omega_recomputed": verified_omega.tolist()},
                  comparisons={
                      "same_source_state_unanchored_l2": rel_l2(native_on_source, source_on_source),
                      "same_source_state_fixed_anchor_l2": rel_l2(normalized(native_on_source), normalized(source_on_source)),
                      "native_on_two_trajectories_unanchored_l2": rel_l2(native_on_native, native_on_source),
                      "native_on_two_trajectories_fixed_anchor_l2": rel_l2(normalized(native_on_native), normalized(native_on_source)),
                      "same_source_state_amplitude_l2": rel_l2(abs(native_on_source), abs(source_on_source)),
                      "native_on_two_trajectories_amplitude_l2": rel_l2(abs(native_on_native), abs(native_on_source)),
                      "time_difference_first_last_M": [float(native_chart_time[i]-source_chart_time[i]) for i in (0, -1)],
                      "state_difference_max_abs": np.max(abs(native_chart_state-source_chart_state), axis=0).tolist(),
                  },
                  modes={"source_on_source": np.column_stack((source_on_source.real, source_on_source.imag)).tolist(),
                         "native_on_source": np.column_stack((native_on_source.real, native_on_source.imag)).tolist(),
                         "native_on_native": np.column_stack((native_on_native.real, native_on_native.imag)).tolist()})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "checks": checks,
                      "comparisons": result["comparisons"],
                      "full_control": source_full_control}), flush=True)


if __name__ == "__main__":
    main()
