"""Dev-only physical q4 fixed-chart NQC target-derivative check."""

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
    calibrate_nqc_22_full_series,
    full_orbital_peak,
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
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "jax_traceable_full_series_nqc_plan.json"
OUTPUT = EVIDENCE / "jax_traceable_full_series_nqc.json"


def relative_error(actual, expected):
    return float(np.linalg.norm(np.asarray(actual) - np.asarray(expected)) /
                 max(np.linalg.norm(np.asarray(expected)), 1e-30))


def main():
    plan = json.loads(PLAN.read_text())
    prior_path = EVIDENCE / "jax_full_series_nqc_physical_q4_nonspinning.json"
    if (plan["schema"] != "diffgw.seobnrv4.jax-traceable-full-series-nqc-plan.v2"
            or plan["case"] != "q4_nonspinning" or plan["release_qualified"] is not False
            or plan["fine_output_step_M"] != 0.015625
            or plan["matching_window_half_width_M"] != 10.0
            or plan["chi_finite_difference_steps"] != [0.0001, 0.00001]
            or q.sha256(prior_path) != plan["prior_physical_sha256"]):
        raise ValueError("frozen physical NQC derivative plan changed")
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
    if selected.size < plan["checks"]["matching_window_min_samples"] or \
            selected[-1] - selected[0] + 1 != selected.size:
        raise ValueError("matching window is too short or noncontiguous")
    slc = slice(int(selected[0]), int(selected[-1])+1)
    time_grid, state = time_grid[slc], state[slc]
    orbital_frequency = orbital_frequency[slc]
    base_mode, base_frequency = base_mode[slc], base_frequency[slc]
    print(f"physical NQC window ready: {len(time_grid)} samples, "
          f"elapsed={time.monotonic()-started:.1f}s", flush=True)

    def fit(target_chi, move_match):
        selected_match = (peak_time +
            _seobnrv4_nqc_time_offset_numerical_compatibility_unchecked(
                nqc_nu, target_chi)) if move_match else match
        return traceable_nqc_22_full_series_coefficients(
            time_grid, state[:, 0], state[:, 2], orbital_frequency,
            base_mode, selected_match,
            _seobnrv4_nqc_targets_unchecked(nqc_nu, target_chi))

    checked = calibrate_nqc_22_full_series(
        time_grid, state[:, 0], state[:, 2], orbital_frequency,
        jnp.abs(base_mode), base_frequency, match,
        seobnrv4_nqc_targets(nqc_nu, nqc_chi), base_mode=base_mode)
    eager = fit(nqc_chi, False)
    compiled_fit = jax.jit(lambda chi: fit(chi, False))
    compiled = compiled_fit(nqc_chi)
    derivative = jax.jit(jax.jacfwd(lambda chi: fit(chi, False)[:2]))(nqc_chi)
    moving_fit = jax.jit(lambda chi: fit(chi, True))
    moving_base = moving_fit(nqc_chi)
    moving_derivative = jax.jit(jax.jacfwd(
        lambda chi: fit(chi, True)[:2]))(nqc_chi)
    print(f"compiled NQC derivative ready: elapsed={time.monotonic()-started:.1f}s",
          flush=True)
    differences = {}
    moving_differences = {}
    for step in plan["chi_finite_difference_steps"]:
        plus, minus = compiled_fit(nqc_chi+step), compiled_fit(nqc_chi-step)
        differences[str(step)] = [
            relative_error(tangent, (high-low)/(2*step))
            for tangent, high, low in zip(derivative, plus[:2], minus[:2])]
        moving_plus, moving_minus = moving_fit(nqc_chi+step), moving_fit(nqc_chi-step)
        moving_differences[str(step)] = [
            relative_error(tangent, (high-low)/(2*step))
            for tangent, high, low in zip(
                moving_derivative, moving_plus[:2], moving_minus[:2])]
    checked_coeff = checked.coefficients
    checked_error = [relative_error(eager[0], checked_coeff.amplitude),
                     relative_error(eager[1], checked_coeff.phase)]
    compiled_error = [relative_error(compiled[0], eager[0]),
                      relative_error(compiled[1], eager[1])]
    moving_baseline_error = [relative_error(left, right) for left, right in zip(
        moving_base[:2], compiled[:2])]
    limits = plan["checks"]
    checks = {
        "checked_traceable_coefficient_relative_error_at_most":
            max(checked_error) <= limits["checked_traceable_coefficient_relative_error_at_most"],
        "eager_compiled_coefficient_relative_error_at_most":
            max(compiled_error) <= limits["eager_compiled_coefficient_relative_error_at_most"],
        "chi_derivative_finite_difference_relative_error_at_most":
            max(max(value) for value in differences.values()) <= limits[
                "chi_derivative_finite_difference_relative_error_at_most"],
        "moving_match_chi_derivative_fd_relative_error_at_most":
            max(max(value) for value in moving_differences.values()) <= limits[
                "chi_derivative_finite_difference_relative_error_at_most"],
        "moving_match_baseline_coefficient_relative_error_at_most":
            max(moving_baseline_error) <= limits[
                    "eager_compiled_coefficient_relative_error_at_most"],
        "maximum_matching_condition":
            max(float(compiled[4]), float(compiled[5])) <= limits[
                "maximum_matching_condition"],
        "finite_derivative": all(np.isfinite(np.asarray(value)).all()
                                 for value in (*derivative, *moving_derivative)),
    }
    result = {
        "schema": "diffgw.seobnrv4.jax-traceable-full-series-nqc.v2",
        "status": "diagnostic_complete", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN),
        "source_sha256": {
            "dev/native_rom/diagnose_jax_traceable_full_series_nqc.py":
                q.sha256(__file__),
            "dev/native_rom/eob/jax/numerical_attachment_clean.py":
                q.sha256(ROOT / "dev/native_rom/eob/jax/numerical_attachment_clean.py"),
            "src/jaxwave/approximants/eob/nqc.py":
                q.sha256(ROOT / "src/jaxwave/approximants/eob/nqc.py"),
        },
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "case": plan["case"], "window_samples": len(time_grid),
        "input_array_sha256": {
            name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for name, value in (("time", time_grid), ("state", state),
                                ("orbital_frequency", orbital_frequency),
                                ("base_mode", base_mode))
        },
        "window_time_M": [float(time_grid[0]), float(time_grid[-1])],
        "match_time_M": float(match),
        "checked_traceable_coefficient_relative_error": checked_error,
        "eager_compiled_coefficient_relative_error": compiled_error,
        "moving_match_baseline_coefficient_relative_error": moving_baseline_error,
        "matching_condition": [float(compiled[4]), float(compiled[5])],
        "chi_target_derivative": [np.asarray(value).tolist() for value in derivative],
        "chi_derivative_fd_relative_error": differences,
        "chi_target_and_match_derivative": [
            np.asarray(value).tolist() for value in moving_derivative],
        "chi_target_and_match_fd_relative_error": moving_differences,
        "checks": checks, "diagnostic_checks_passed": all(checks.values()),
        "elapsed_s": time.monotonic()-started,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"checks": checks, "fd_error": differences,
                      "moving_fd_error": moving_differences}, sort_keys=True),
          flush=True)
    return 0 if result["diagnostic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
