"""Dev-only physical full-series NQC comparison on fine EOB trajectories."""

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from dev.native_rom.diagnose_jax_adaptive_physical_event import CASES
from dev.native_rom.diagnose_jax_two_stage_trajectory import construct_case
from dev.native_rom.eob.jax.numerical_attachment_clean import (
    calibrate_nqc_22_full_series as jax_nqc,
    full_orbital_peak,
)
from dev.native_rom.eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.nqc import (
    seobnrv4_nqc_parameters as jax_parameters,
    seobnrv4_nqc_targets as jax_targets,
    seobnrv4_nqc_time_offset_numerical_compatibility as jax_offset,
)
from jaxwave.approximants.eob.published_waveform import published_mode22_and_frequency as jax_mode
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as JaxFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as JaxHam
from torchwave.approximants.eob.adaptive_driver_clean import AlignedSpinV4Event, integrate_adaptive
from torchwave.approximants.eob.nqc import (
    seobnrv4_nqc_parameters as torch_parameters,
    seobnrv4_nqc_targets as torch_targets,
    seobnrv4_nqc_time_offset_numerical_compatibility as torch_offset,
)
from torchwave.approximants.eob.numerical_attachment import (
    _full_orbital_peak,
    calibrate_nqc_22_full_series as torch_nqc,
)
from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency as torch_mode
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as TorchFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as TorchHam

jax.config.update("jax_enable_x64", True)

MODE_OPTIONS = dict(spin_v6="bohe_total_energy", include_pan_v8=False,
                    include_positive_chi_540=False, retain_pan_cubic_spin=False)


def _array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().numpy()
    return np.asarray(value)


def _max_abs(left, right):
    return float(np.max(np.abs(_array(left) - _array(right))))


def _relative_l2(left, right):
    delta = _array(left) - _array(right)
    return float(np.linalg.norm(delta) / np.linalg.norm(_array(right)))


def _compare(left, right):
    return {
        "amplitude_matrix_max_abs": _max_abs(left.amplitude_matching_matrix,
                                               right.amplitude_matching_matrix),
        "phase_matrix_max_abs": _max_abs(left.phase_matching_matrix,
                                           right.phase_matching_matrix),
        "amplitude_coeff_relative_l2": _relative_l2(left.coefficients.amplitude,
                                                       right.coefficients.amplitude),
        "phase_coeff_relative_l2": _relative_l2(left.coefficients.phase,
                                                   right.coefficients.phase),
        "corrected_amplitude_relative_l2": _relative_l2(left.sampled_corrected_amplitude,
                                                            right.sampled_corrected_amplitude),
        "corrected_mode_complex_relative_l2": _relative_l2(
            left.evaluate(left.amplitude_spline.knots),
            right.evaluate(right.amplitude_spline.knots)),
    }


def run(case_id):
    started = time.monotonic()
    m1, m2, chi1, chi2, _ = CASES[case_id]
    initial_torch, initial_jax, torch_field, jax_field = construct_case(case_id)
    dt, max_duration, max_step = 0.5, 1000.0, 2.0
    coarse = integrate_adaptive(
        torch_field, initial_torch, dt, t_end=max_duration,
        event=AlignedSpinV4Event("coarse"), initial_step=dt,
        max_step=max_step, controller="gsl_state_scaled", atol=1e-10, rtol=1e-9)
    n_coarse = len(coarse.output_time)
    restart = (n_coarse - math.ceil(150 / dt) if (n_coarse - 1) * dt >= 150
               else n_coarse - math.ceil(n_coarse / 2))
    fine_step = dt / 32
    fine = integrate_adaptive(
        torch_field, coarse.output_state[restart], fine_step,
        t_end=max_duration, event=AlignedSpinV4Event("fine"),
        initial_step=fine_step, max_step=max_step,
        controller="gsl_state_scaled", atol=1e-10, rtol=1e-9)
    jrun = integrate_two_stage_trajectory(
        jax_field, initial_jax, output_step=dt,
        max_duration=max_duration, coarse_initial_step=dt,
        coarse_max_step=max_step, fine_max_step=max_step,
        atol=1e-10, rtol=1e-9)
    torch_omega = torch_field(fine.output_state)[..., 1]
    jax_omega = jax.vmap(jax_field)(jrun.fine_state)[:, 1]
    print(case_id, "trajectories", len(fine.output_time), flush=True)

    nu = m1 * m2 / (m1 + m2)**2
    delta = (m1 - m2) / (m1 + m2)
    def tensor(value):
        return torch.tensor(value, dtype=torch.float64)
    th = TorchHam(tensor(nu), tensor(chi1), tensor(chi2), delta=tensor(delta))
    tf = TorchFlux(spin_truncation="tabulated", retain_pan_cubic_spin=False)
    jh = JaxHam(nu, chi1, chi2, delta=delta)
    jf = JaxFlux(spin_truncation="tabulated", retain_pan_cubic_spin=False)
    torch_base, torch_frequency = torch_mode(fine.output_state, th, tf, **MODE_OPTIONS)
    jax_mode_fn = jax.jit(lambda state: jax_mode(state, jh, jf, **MODE_OPTIONS))
    jax_base, jax_frequency = jax_mode_fn(jrun.fine_state)
    print(case_id, "base modes", flush=True)

    tnu, tchi = torch_parameters(tensor(m1), tensor(m2), tensor(chi1), tensor(chi2))
    jnu, jchi = jax_parameters(m1, m2, chi1, chi2)
    tpeak = _full_orbital_peak(fine.output_time, torch_omega)
    jpeak = full_orbital_peak(jrun.fine_time, jax_omega)
    tmatch = tpeak.time + torch_offset(tnu, tchi)
    jmatch = jpeak.time + jax_offset(jnu, jchi)
    tref = torch_nqc(
        fine.output_time, fine.output_state[:, 0], fine.output_state[:, 2],
        torch_omega, torch_base.abs(), torch_frequency, tmatch,
        torch_targets(tnu, tchi), base_mode=torch_base)
    ttime = jnp.asarray(fine.output_time.detach().numpy())
    tstate = jnp.asarray(fine.output_state.detach().numpy())
    shared = jax_nqc(
        ttime, tstate[:, 0], tstate[:, 2], jnp.asarray(torch_omega.detach().numpy()),
        jnp.asarray(torch_base.abs().detach().numpy()),
        jnp.asarray(torch_frequency.detach().numpy()), jnp.asarray(float(tmatch)),
        jax_targets(jnu, jchi), base_mode=jnp.asarray(torch_base.detach().numpy()))
    print(case_id, "same-input NQC", flush=True)
    end_to_end = jax_nqc(
        jrun.fine_time, jrun.fine_state[:, 0], jrun.fine_state[:, 2], jax_omega,
        jnp.abs(jax_base), jax_frequency, jmatch,
        jax_targets(jnu, jchi), base_mode=jax_base)
    print(case_id, "end-to-end NQC", flush=True)

    same = _compare(shared, tref)
    e2e = _compare(end_to_end, tref)
    result = {
        "schema": "diffgw.seobnrv4.jax-full-series-nqc-physical.v1",
        "case_id": case_id, "status": "diagnostic_complete",
        "scope": "Physical fine-series frontend NQC parity; no ringdown, independent reference, or release qualification",
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "input": {"case_parameters": CASES[case_id], "fine_step_M": fine_step,
                  "max_step_M": max_step, "atol": 1e-10, "rtol": 1e-9,
                  "fine_samples": len(fine.output_time)},
        "same_input": same,
        "end_to_end": e2e,
        "base_mode_complex_relative_l2": _relative_l2(jax_base, torch_base),
        "base_frequency_max_abs": _max_abs(jax_frequency, torch_frequency),
        "match_time_abs_M": abs(float(jmatch) - float(tmatch)),
        "torch": {"amplitude_condition": float(tref.calibration.amplitude_condition),
                  "phase_condition": float(tref.calibration.phase_condition)},
        "jax": {"amplitude_condition": float(end_to_end.calibration.amplitude_condition),
                "phase_condition": float(end_to_end.calibration.phase_condition)},
        "elapsed_s": time.monotonic() - started,
        "release_qualified": False,
    }
    result["checks"] = {
        "matching_matrix_max_abs_at_most": max(
            same["amplitude_matrix_max_abs"], same["phase_matrix_max_abs"],
            e2e["amplitude_matrix_max_abs"], e2e["phase_matrix_max_abs"]) <= 1e-8,
        "corrected_mode_complex_relative_l2_at_most": max(
            same["corrected_mode_complex_relative_l2"],
            e2e["corrected_mode_complex_relative_l2"]) <= 1e-5,
        "all_matching_conditions_finite": all(math.isfinite(value) for group in
                                               (result["torch"], result["jax"])
                                               for value in group.values()),
        "same_input_coefficient_relative_l2_at_most": max(
            same["amplitude_coeff_relative_l2"],
            same["phase_coeff_relative_l2"]) <= 1e-5,
    }
    result["diagnostic_checks_passed"] = all(result["checks"].values())
    output = Path(f"dev/native_rom/evidence/2026-09-25/jax_full_series_nqc_physical_{case_id}.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(case_id, json.dumps(result["checks"], sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(*CASES, "prospective"), required=True)
    selected = parser.parse_args().case
    names = (("q4_nonspinning", "strong_aligned", "strong_antialigned")
             if selected == "prospective" else (selected,))
    for name in names:
        run(name)
