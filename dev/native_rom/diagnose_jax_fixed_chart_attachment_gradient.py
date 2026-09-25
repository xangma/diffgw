"""Dev-only physical q4 gradient through NQC and continuous attachment."""

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_adaptive_physical_event import CASES
from dev.native_rom.diagnose_jax_full_series_nqc_physical import MODE_OPTIONS
from dev.native_rom.diagnose_jax_two_stage_trajectory import construct_case
from dev.native_rom.eob.jax.numerical_attachment_clean import (
    attach_seobnrv4_22_numerical_compatibility,
    full_orbital_peak,
    traceable_fixed_chart_attached_mode22,
    traceable_nqc_22_full_series_coefficients,
)
from dev.native_rom.eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.nqc import (
    _seobnrv4_nqc_targets_unchecked,
    _seobnrv4_nqc_time_offset_numerical_compatibility_unchecked,
    seobnrv4_nqc_parameters,
    seobnrv4_nqc_targets,
    seobnrv4_nqc_time_offset_numerical_compatibility,
)
from jaxwave.approximants.eob.published_waveform import published_mode22_and_frequency
from jaxwave.approximants.eob.remnant import eob_final_mass_fraction, hofmann_final_spin
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "jax_fixed_chart_attachment_gradient_plan.json"
OUTPUT = EVIDENCE / "jax_fixed_chart_attachment_gradient.json"


def relative_l2(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    return float(np.linalg.norm(actual - expected) /
                 max(np.linalg.norm(expected), 1e-30))


def main():
    plan = json.loads(PLAN.read_text())
    prior_path = EVIDENCE / "jax_full_series_nqc_physical_q4_nonspinning.json"
    if (plan["schema"] != "diffgw.seobnrv4.jax-fixed-chart-attachment-gradient-plan.v1"
            or plan["case"] != "q4_nonspinning" or plan["release_qualified"] is not False
            or q.sha256(prior_path) != plan["prior_physical_sha256"]):
        raise ValueError("frozen attachment derivative plan changed")
    prior = json.loads(prior_path.read_text())
    if prior["status"] != "diagnostic_complete" or not prior["diagnostic_checks_passed"]:
        raise ValueError("prior physical NQC parity diagnostic is incomplete")
    started = time.monotonic()
    m1, m2, chi1, chi2, _ = CASES[plan["case"]]
    _, initial, _, field = construct_case(plan["case"])
    trajectory = integrate_two_stage_trajectory(
        field, initial, output_step=0.5, max_duration=1000.0,
        coarse_initial_step=0.5, coarse_max_step=2.0, fine_max_step=2.0,
        atol=1e-10, rtol=1e-9)
    time_grid = trajectory.fine_time
    state = trajectory.fine_state
    orbital_frequency = jax.vmap(field)(state)[:, 1]
    nu = m1*m2/(m1+m2)**2
    delta = (m1-m2)/(m1+m2)
    ham = SEOBNRv4Hamiltonian(nu, chi1, chi2, delta=delta)
    flux = SEOBNRv4PublishedFlux(spin_truncation="tabulated",
                                 retain_pan_cubic_spin=False)
    base_mode, base_frequency = jax.jit(lambda series:
        published_mode22_and_frequency(series, ham, flux, **MODE_OPTIONS))(state)
    nqc_nu, nqc_chi = seobnrv4_nqc_parameters(m1, m2, chi1, chi2)
    peak_time = full_orbital_peak(time_grid, orbital_frequency).time
    match = peak_time + \
        seobnrv4_nqc_time_offset_numerical_compatibility(nqc_nu, nqc_chi)
    half_width = plan["matching_window_half_width_M"]
    selected = np.flatnonzero(np.abs(np.asarray(time_grid - match)) <= half_width)
    if (selected.size < plan["checks"]["matching_window_min_samples"]
            or selected[-1] - selected[0] + 1 != selected.size):
        raise ValueError("matching window is too short or noncontiguous")
    slc = slice(int(selected[0]), int(selected[-1])+1)
    time_grid, state = time_grid[slc], state[slc]
    orbital_frequency = orbital_frequency[slc]
    base_mode, base_frequency = base_mode[slc], base_frequency[slc]
    spacing = plan["attachment_output_step_M"]
    relative_output = np.arange(
        plan["attachment_output_start_relative_M"],
        plan["attachment_output_stop_relative_M"] + spacing/2, spacing)
    output_time = jnp.asarray(float(match) + relative_output)
    final_mass = eob_final_mass_fraction(m1, m2, chi1, chi2)
    final_spin = hofmann_final_spin(m1, m2, chi1, chi2, fit_order=(3, 4))
    qnm_frequency, qnm_damping = evaluate_kerr_qnm_220(final_spin)
    sigma = (qnm_frequency - 1j*qnm_damping)/final_mass
    checked = attach_seobnrv4_22_numerical_compatibility(
        time_grid, state, orbital_frequency, base_mode, base_frequency,
        nqc_nu, nqc_chi, sigma, output_time, match_time=match,
        diagnostic_continuous_boundary=True)
    print(f"physical attachment window ready: {len(time_grid)} samples, "
          f"{len(output_time)} outputs, elapsed={time.monotonic()-started:.1f}s",
          flush=True)

    def replay(perturbations):
        spin_shift, input_shift = perturbations
        spin = nqc_chi + spin_shift
        selected_match = peak_time + \
            _seobnrv4_nqc_time_offset_numerical_compatibility_unchecked(nqc_nu, spin)
        modified_r = state[:, 0] * (1 + 0.01*input_shift)
        modified_pr = state[:, 2] * (1 + 0.01*input_shift)
        modified_omega = orbital_frequency * (1 + 0.01*input_shift)
        modified_mode = base_mode * (1 + 0.01*input_shift) * \
            jnp.exp(-1j*0.01*input_shift*(time_grid - match))
        return traceable_fixed_chart_attached_mode22(
            time_grid, modified_r, modified_pr, modified_omega,
            modified_mode, selected_match,
            _seobnrv4_nqc_targets_unchecked(nqc_nu, spin),
            nqc_nu, spin, sigma, output_time)

    origin = jnp.zeros(2)
    eager = replay(origin)
    compiled_replay = jax.jit(replay)
    compiled = compiled_replay(origin)
    mode_derivative = jax.jit(jax.jacfwd(lambda x: replay(x)[0]))(origin)
    print(f"compiled full attachment derivative ready: "
          f"elapsed={time.monotonic()-started:.1f}s", flush=True)
    fd_error = {}
    for step in plan["finite_difference_steps"]:
        errors = []
        for component in range(2):
            delta = jnp.eye(2)[component] * step
            plus = compiled_replay(delta)[0]
            minus = compiled_replay(-delta)[0]
            errors.append(relative_l2(mode_derivative[:, component],
                                      (plus-minus)/(2*step)))
        fd_error[str(step)] = errors
    checked_mode_error = relative_l2(compiled[0], checked.mode)
    checked_corrected_error = relative_l2(compiled[1], checked.corrected_inspiral)
    eager_compiled_error = [relative_l2(compiled_value, eager_value)
                            for compiled_value, eager_value in zip(compiled, eager)]
    fit = traceable_nqc_22_full_series_coefficients(
        time_grid, state[:, 0], state[:, 2], orbital_frequency,
        base_mode, match, seobnrv4_nqc_targets(nqc_nu, nqc_chi))
    limits = plan["checks"]
    checks = {
        "checked_compiled_mode_relative_l2_at_most":
            checked_mode_error <= limits["checked_compiled_mode_relative_l2_at_most"],
        "checked_compiled_corrected_relative_l2_at_most":
            checked_corrected_error <= limits[
                "checked_compiled_corrected_relative_l2_at_most"],
        "derivative_finite_difference_relative_l2_at_most":
            max(max(errors) for errors in fd_error.values()) <= limits[
                "derivative_finite_difference_relative_l2_at_most"],
        "maximum_matching_condition":
            max(float(fit[4]), float(fit[5])) <= limits["maximum_matching_condition"],
        "finite_derivative": bool(np.isfinite(np.asarray(mode_derivative)).all()),
    }
    result = {
        "schema": "diffgw.seobnrv4.jax-fixed-chart-attachment-gradient.v1",
        "status": "diagnostic_complete", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN),
        "source_sha256": {
            "dev/native_rom/diagnose_jax_fixed_chart_attachment_gradient.py": q.sha256(__file__),
            "dev/native_rom/eob/jax/numerical_attachment_clean.py": q.sha256(
                ROOT / "dev/native_rom/eob/jax/numerical_attachment_clean.py"),
        },
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "case": plan["case"], "window_samples": len(time_grid),
        "output_samples": len(output_time), "match_time_M": float(match),
        "input_array_sha256": {
            name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for name, value in (("time", time_grid), ("state", state),
                                ("orbital_frequency", orbital_frequency),
                                ("base_mode", base_mode))
        },
        "checked_compiled_mode_relative_l2": checked_mode_error,
        "checked_compiled_corrected_relative_l2": checked_corrected_error,
        "eager_compiled_relative_l2": eager_compiled_error,
        "derivative_l2": [float(np.linalg.norm(np.asarray(mode_derivative)[:, i]))
                          for i in range(2)],
        "derivative_fd_relative_l2": fd_error,
        "matching_condition": [float(fit[4]), float(fit[5])],
        "checks": checks, "diagnostic_checks_passed": all(checks.values()),
        "elapsed_s": time.monotonic()-started,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"checks": checks, "fd_error": fd_error,
                      "checked_mode_error": checked_mode_error}, sort_keys=True),
          flush=True)
    return 0 if result["diagnostic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
