"""Fixed-input spline and phase-row readout for the ODE Lie NQC diagnostic."""

from __future__ import annotations

import json
import os
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import qualify_generator as q
from .diagnose_jax_adaptive_physical_event import CASES
from .diagnose_jax_full_series_nqc_physical import MODE_OPTIONS
from .diagnose_jax_two_stage_trajectory import construct_case
from .eob.jax.numerical_attachment_clean import (
    _solve_traceable,
    _unwrap_mode_phase_unchecked,
)
from .eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.nqc import (
    _seobnrv4_nqc_targets_unchecked,
    nqc_22_basis,
    seobnrv4_nqc_parameters,
)
from jaxwave.approximants.eob.numerical_attachment import natural_cubic_spline
from jaxwave.approximants.eob.published_waveform import published_mode22_and_frequency
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

jax.config.update("jax_enable_x64", True)

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_nqc_fixed_rows_plan.json"
OUTPUT = PLAN.with_name("jax_nqc_fixed_rows.json")


def _plain(value):
    return np.asarray(value).tolist()


def _max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.jax-nqc-fixed-rows-plan.v1"
            or plan["case"] != "high_start_mixed_spin"
            or plan["release_qualified"] is not False):
        raise ValueError("frozen fixed-row plan changed")
    input_path = q.ROOT / plan["input_result"]
    if q.sha256(input_path) != plan["input_sha256"]:
        raise ValueError("pinned ODE Lie result changed")
    prior = json.loads(input_path.read_text())
    report = dict(schema="diffgw.seobnrv4.jax-nqc-fixed-rows.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256=plan["input_sha256"],
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid())
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")

    case = plan["case"]
    m1, m2, chi1, chi2, _ = CASES[case]
    _, initial, _, field = construct_case(case)
    cfg = plan["expected"]
    dt = cfg["output_step_M"]
    run = integrate_two_stage_trajectory(
        field, initial, output_step=dt, max_duration=1000.,
        coarse_initial_step=dt, coarse_max_step=2., fine_max_step=2.,
        fine_refinement_factor=cfg["fine_refinement_factor"],
        atol=cfg["adaptive_atol"], rtol=cfg["adaptive_rtol"])
    print("trajectory", run.coarse.accepted_steps, run.fine.accepted_steps,
          flush=True)
    nu, delta = m1*m2/(m1+m2)**2, (m1-m2)/(m1+m2)
    ham = SEOBNRv4Hamiltonian(nu, chi1, chi2, delta=delta)
    flux = SEOBNRv4PublishedFlux(spin_truncation="tabulated",
                                 retain_pan_cubic_spin=False)
    nqc_nu, nqc_chi = seobnrv4_nqc_parameters(m1, m2, chi1, chi2)
    targets = _seobnrv4_nqc_targets_unchecked(nqc_nu, nqc_chi)
    saved_center = jnp.asarray(prior["fixed_input"]["fixed_center"])
    tmatch = jnp.asarray(plan["match_time_M"])
    center = natural_cubic_spline(
        run.fine_time, run.fine_state, scalar_output=False).evaluate(tmatch)
    center_error = _max_abs(center, saved_center)
    omega = jax.vmap(field)(run.fine_state)[:, 1]
    mode, _ = published_mode22_and_frequency(run.fine_state, ham, flux,
                                              **MODE_OPTIONS)

    def spline_rows(fine, frequency, base_mode):
        amplitude = jnp.abs(base_mode)
        phase = _unwrap_mode_phase_unchecked(base_mode)
        q_basis, p_basis = nqc_22_basis(
            fine[:, 0], fine[:, 2], frequency)
        combined = jnp.concatenate((amplitude[:, None],
                                    amplitude[:, None]*q_basis, p_basis), axis=-1)
        spline = natural_cubic_spline(run.fine_time, combined,
                                      scalar_output=False)
        value = spline(tmatch)
        first = spline.derivative(tmatch)
        second = spline.derivative(tmatch, 2)
        amplitude_rows = jnp.stack((value[:4], first[:4], second[:4]))
        phase_matrix = jnp.stack((first[4:6], second[4:6]))
        phase_spline = natural_cubic_spline(run.fine_time, phase,
                                            scalar_output=True)
        phase_raw = jnp.stack((phase_spline.derivative(tmatch),
                               phase_spline.derivative(tmatch, 2)))
        phase_target = jnp.stack((targets.frequency,
                                  targets.frequency_derivative))
        phase_rhs = phase_raw-phase_target
        amplitude_rhs = jnp.stack(targets[:3])-amplitude_rows[:, 0]
        amplitude_coeff, _, _ = _solve_traceable(
            amplitude_rows[:, 1:4], amplitude_rhs)
        phase_coeff, _, phase_residual = _solve_traceable(
            phase_matrix, phase_rhs)
        return (amplitude_rows, phase_matrix, phase_raw, phase_target,
                phase_rhs, amplitude_coeff, phase_coeff, phase_residual)

    eager = spline_rows(run.fine_state, omega, mode)
    compiled = jax.jit(spline_rows)(run.fine_state, omega, mode)
    print("compiled spline rows", time.monotonic()-started, flush=True)
    lie_rows = np.asarray(prior["fixed_input"]["eager_raw_amplitude_match_rows"])
    direct_mode, _ = published_mode22_and_frequency(center, ham, flux,
                                                     **MODE_OPTIONS)
    sampled_mode_at_match = natural_cubic_spline(
        run.fine_time, jnp.stack((mode.real, mode.imag), axis=-1),
        scalar_output=False).evaluate(tmatch)
    sampled_mode_at_match = sampled_mode_at_match[0]+1j*sampled_mode_at_match[1]
    sampled_omega_at_match = natural_cubic_spline(
        run.fine_time, omega, scalar_output=True).evaluate(tmatch)
    report.update(
        status="diagnostic_complete", case=case,
        frozen_match_time_M=float(tmatch),
        center_replay_max_abs=center_error,
        eager_spline_amplitude_rows=_plain(eager[0]),
        jit_spline_amplitude_rows=_plain(compiled[0]),
        eager_jit_spline_amplitude_rows_max_abs=_max_abs(eager[0], compiled[0]),
        eager_lie_amplitude_rows=_plain(lie_rows),
        lie_minus_spline_amplitude_rows=_plain(lie_rows-np.asarray(eager[0])),
        lie_minus_spline_amplitude_rows_max_abs=_max_abs(lie_rows, eager[0]),
        eager_jit_spline_amplitude_coefficients_max_abs=_max_abs(eager[5], compiled[5]),
        spline_amplitude_coefficients=_plain(eager[5]),
        eager_phase_basis_derivative_rows=_plain(eager[1]),
        jit_phase_basis_derivative_rows=_plain(compiled[1]),
        eager_phase_raw_derivatives=_plain(eager[2]),
        jit_phase_raw_derivatives=_plain(compiled[2]),
        phase_targets=_plain(eager[3]),
        eager_phase_rhs=_plain(eager[4]),
        jit_phase_rhs=_plain(compiled[4]),
        eager_phase_coefficients=_plain(eager[6]),
        jit_phase_coefficients=_plain(compiled[6]),
        eager_jit_phase_coefficients_max_abs=_max_abs(eager[6], compiled[6]),
        eager_jit_phase_basis_rows_max_abs=_max_abs(eager[1], compiled[1]),
        eager_jit_phase_raw_derivatives_max_abs=_max_abs(eager[2], compiled[2]),
        eager_phase_solve_residual_relative_l2=float(
            jnp.linalg.norm(eager[7])/jnp.linalg.norm(eager[4])),
        direct_center_mode_vs_sampled_spline_absolute_error=float(
            jnp.abs(direct_mode-sampled_mode_at_match)),
        direct_center_omega_vs_sampled_spline_absolute_error=float(
            jnp.abs(field(center)[1]-sampled_omega_at_match)),
        mode_options=MODE_OPTIONS,
        elapsed_s=time.monotonic()-started)
    report["checks"] = dict(
        center_replay=center_error <= plan["gates"]["max_replayed_center_absolute_error"],
        spline_coefficients_replay=_max_abs(
            eager[5], prior["fixed_input"]["spline_amplitude_coefficients"]) <=
        plan["gates"]["max_replayed_spline_coefficient_absolute_error"])
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print("phase gap", report["eager_jit_phase_coefficients_max_abs"], flush=True)
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
