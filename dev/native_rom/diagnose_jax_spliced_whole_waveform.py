"""Dev-only full coarse/fine/ringdown v4 parity and public-binary diagnostic."""

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

from dev.native_rom import qualify_generator as qualification
from dev.native_rom.compare_whole_waveforms import compare_public_binary
from dev.native_rom.diagnose_jax_adaptive_physical_event import CASES
from dev.native_rom.diagnose_jax_full_series_nqc_physical import MODE_OPTIONS
from dev.native_rom.diagnose_jax_two_stage_trajectory import construct_case
from dev.native_rom.eob.jax.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian as JaxSourceHam,
)
from dev.native_rom.eob.torch.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian as TorchSourceHam,
)
from dev.native_rom.eob.jax.numerical_attachment_clean import (
    attach_seobnrv4_22_numerical_compatibility as jax_attach,
    full_orbital_peak as jax_peak,
)
from dev.native_rom.eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.nqc import (
    nqc_22_factor as jax_factor,
    seobnrv4_nqc_parameters as jax_parameters,
    seobnrv4_nqc_time_offset_numerical_compatibility as jax_offset,
)
from jaxwave.approximants.eob.published_waveform import published_mode22_and_frequency as jax_mode
from jaxwave.approximants.eob.integrator import EOBTrajectoryIntegrator as JaxIntegrator
from jaxwave.approximants.eob.initial_conditions import (
    find_quasicircular_initial_conditions as jax_initial_conditions,
)
from jaxwave.approximants.eob.remnant import (
    eob_final_mass_fraction as jax_final_mass,
    hofmann_final_spin as jax_final_spin,
)
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as JaxFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as JaxHam
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220 as jax_qnm
from torchwave.approximants.eob.adaptive_driver_clean import AlignedSpinV4Event, integrate_adaptive
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator as TorchIntegrator
from torchwave.approximants.eob.nqc import (
    nqc_22_factor as torch_factor,
    nqc_22_basis as torch_basis,
    seobnrv4_nqc_parameters as torch_parameters,
    seobnrv4_nqc_targets as torch_targets,
    seobnrv4_nqc_time_offset_numerical_compatibility as torch_offset,
)
from torchwave.approximants.eob.numerical_attachment import (
    _full_orbital_peak as torch_peak,
    _eval_batch_scalar,
    _unwrap_mode_phase,
    attach_seobnrv4_22_numerical_compatibility as torch_attach,
)
from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency as torch_mode
from torchwave.approximants.eob.remnant import (
    eob_final_mass_fraction as torch_final_mass,
    hofmann_final_spin as torch_final_spin,
)
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as TorchFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as TorchHam
from torchwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220 as torch_qnm

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "dev/native_rom/evidence/2026-09-25/jax_spliced_whole_waveform_plan.json"
SOURCE_PLAN = ROOT / "dev/native_rom/evidence/2026-09-25/jax_source_spliced_whole_waveform_plan.json"
INDEPENDENT_INIT_PLAN = ROOT / "dev/native_rom/evidence/2026-09-25/jax_source_independent_init_plan.json"

SOURCE_FLUX_OPTIONS = dict(
    spin_truncation="tabulated", retain_pan_cubic_spin=True,
    mode_21_higher_spin="rho_hybrid", mode_31_higher_spin="lal_v4_rho",
    mode_32_spin_square="lal_v4_zero", mode_33_higher_spin="lal_v4_rho",
    mode_43_higher_spin="lal_v4_rho",
)


def _array(value):
    return value.detach().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _relative_l2(left, right):
    return float(np.linalg.norm(left - right) / np.linalg.norm(right))


def _polynomial_matching_second(time_grid, state, orbital_frequency, base_mode,
                                match_time, *, half_width_M, degree):
    """Diagnostic local curvature for A, three A*q products, and mode phase."""
    if half_width_M <= 0 or degree < 2:
        raise ValueError("invalid diagnostic matching polynomial")
    selected = (time_grid >= match_time-half_width_M) & (
        time_grid <= match_time+half_width_M)
    if int(selected.sum()) < degree + 3:
        raise ValueError("insufficient NQC polynomial samples")
    q, _ = torch_basis(state[:, 0], state[:, 2], orbital_frequency)
    amplitude = base_mode.abs()
    values = torch.cat((amplitude[:, None], amplitude[:, None]*q,
                        _unwrap_mode_phase(base_mode)[:, None]), dim=-1)
    x = (time_grid[selected]-match_time)/half_width_M
    matrix = torch.stack(tuple(x**power for power in range(degree+1)), dim=-1)
    coefficients = torch.linalg.lstsq(matrix, values[selected], driver="gels").solution
    return 2*coefficients[2]/half_width_M**2


def _polynomial_phase_matrix(time_grid, state, orbital_frequency, match_time,
                             *, half_width_M, degree):
    """Diagnostic local first/second derivatives of the two NQC phase bases."""
    if half_width_M <= 0 or degree < 2:
        raise ValueError("invalid diagnostic phase-basis polynomial")
    selected = (time_grid >= match_time-half_width_M) & (
        time_grid <= match_time+half_width_M)
    if int(selected.sum()) < degree + 3:
        raise ValueError("insufficient NQC phase-basis samples")
    _, phase_basis = torch_basis(state[:, 0], state[:, 2], orbital_frequency)
    x = (time_grid[selected]-match_time)/half_width_M
    matrix = torch.stack(tuple(x**power for power in range(degree+1)), dim=-1)
    coefficients = torch.linalg.lstsq(matrix, phase_basis[selected], driver="gels").solution
    return torch.stack((coefficients[1]/half_width_M,
                        2*coefficients[2]/half_width_M**2))


def _prepare_public(plan):
    pin = plan["public_binary"]
    identity_plan = json.loads((ROOT / pin["identity_source"]).read_text())
    identity = qualification.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"])
    if identity != identity_plan["lal_binary"]:
        raise ValueError("public binary differs from the pinned earlier comparison")
    return identity


def run(case_id, plan, public_identity, *, variant="published", anchor_control=False,
        independent_init=False, numerical_settings=None, write_result=True,
        return_series=False, return_stages=False, diagnostic_nqc_second=None,
        diagnostic_nqc_phase_matrix=None):
    started = time.monotonic()
    m1, m2, chi1, chi2, omega_start = CASES[case_id]
    nu = m1 * m2 / (m1 + m2)**2
    delta = (m1 - m2) / (m1 + m2)

    def tensor(value):
        return torch.tensor(value, dtype=torch.float64)

    if variant == "source":
        th = TorchSourceHam(tensor(nu), tensor(chi1), tensor(chi2), delta=tensor(delta))
        tf = TorchFlux(**SOURCE_FLUX_OPTIONS)
        jh = JaxSourceHam(nu, chi1, chi2, delta=delta)
        jf = JaxFlux(**SOURCE_FLUX_OPTIONS)
        radius, momentum, angular_momentum = find_quasicircular_initial_conditions(
            th, tf, tensor(omega_start))
        initial_torch = torch.stack((radius, torch.zeros_like(radius), momentum,
                                     angular_momentum)).detach()
        if independent_init:
            jradius, jmomentum, jangular_momentum = jax_initial_conditions(
                jh, jf, jnp.asarray(omega_start))
            initial_jax = jnp.stack((jradius, jnp.zeros_like(jradius),
                                     jmomentum, jangular_momentum))
        else:
            initial_jax = jnp.asarray(initial_torch.numpy())
        torch_field = TorchIntegrator(th, tf).vector_field
        jax_field = jax.jit(JaxIntegrator(jh, jf).vector_field)
        mode_options = MODE_OPTIONS | {"retain_pan_cubic_spin": True}
    else:
        initial_torch, initial_jax, torch_field, jax_field = construct_case(case_id)
        th = TorchHam(tensor(nu), tensor(chi1), tensor(chi2), delta=tensor(delta))
        tf = TorchFlux(spin_truncation="tabulated", retain_pan_cubic_spin=False)
        jh = JaxHam(nu, chi1, chi2, delta=delta)
        jf = JaxFlux(spin_truncation="tabulated", retain_pan_cubic_spin=False)
        mode_options = MODE_OPTIONS
    numerical_settings = numerical_settings or {}
    dt = numerical_settings.get("coarse_output_step_M", 0.5)
    max_duration = numerical_settings.get("max_duration_M", 1000.0)
    max_step = numerical_settings.get("max_step_M", 2.0)
    atol = numerical_settings.get("atol", 1e-10)
    rtol = numerical_settings.get("rtol", 1e-9)
    coarse = integrate_adaptive(
        torch_field, initial_torch, dt, t_end=max_duration,
        event=AlignedSpinV4Event("coarse"), initial_step=dt,
        max_step=max_step, controller="gsl_state_scaled", atol=atol, rtol=rtol)
    n_coarse = len(coarse.output_time)
    restart = (n_coarse - math.ceil(150 / dt) if (n_coarse - 1) * dt >= 150
               else n_coarse - math.ceil(n_coarse / 2))
    fine_step = dt / 32
    fine = integrate_adaptive(
        torch_field, coarse.output_state[restart], fine_step,
        t_end=max_duration, event=AlignedSpinV4Event("fine"),
        initial_step=fine_step, max_step=max_step,
        controller="gsl_state_scaled", atol=atol, rtol=rtol)
    jrun = integrate_two_stage_trajectory(
        jax_field, initial_jax, output_step=dt,
        max_duration=max_duration, coarse_initial_step=dt,
        coarse_max_step=max_step, fine_max_step=max_step,
        atol=atol, rtol=rtol)
    print(case_id, "trajectories", len(fine.output_time), flush=True)

    torch_fine_omega = torch_field(fine.output_state)[:, 1]
    torch_coarse_omega = torch_field(coarse.output_state)[:, 1]
    jax_fine_omega = jax.vmap(jax_field)(jrun.fine_state)[:, 1]
    jax_coarse_omega = jax.vmap(jax_field)(jrun.coarse_state)[:, 1]
    torch_fine_mode, torch_fine_frequency = torch_mode(fine.output_state, th, tf, **mode_options)
    torch_coarse_mode, _ = torch_mode(coarse.output_state, th, tf, **mode_options)
    jax_mode_fn = jax.jit(lambda state: jax_mode(state, jh, jf, **mode_options))
    jax_fine_mode, jax_fine_frequency = jax_mode_fn(jrun.fine_state)
    jax_coarse_mode, _ = jax_mode_fn(jrun.coarse_state)
    print(case_id, "base modes", flush=True)

    tnu, tchi = torch_parameters(tensor(m1), tensor(m2), tensor(chi1), tensor(chi2))
    jnu, jchi = jax_parameters(m1, m2, chi1, chi2)
    tmass = torch_final_mass(*map(tensor, (m1, m2, chi1, chi2)))
    jmass = jax_final_mass(m1, m2, chi1, chi2)
    tspin = torch_final_spin(*map(tensor, (m1, m2, chi1, chi2)), fit_order=(3, 4))
    jspin = jax_final_spin(m1, m2, chi1, chi2, fit_order=(3, 4))
    tw, tdamp = torch_qnm(tspin)
    jw, jdamp = jax_qnm(jspin)
    tsigma = torch.complex(tw, -tdamp) / tmass
    jsigma = (jw - 1j * jdamp) / jmass
    tmatch = torch_peak(fine.output_time, torch_fine_omega).time + torch_offset(tnu, tchi)
    jmatch = jax_peak(jrun.fine_time, jax_fine_omega).time + jax_offset(jnu, jchi)
    fine_count = math.ceil((max(float(tmatch), float(jmatch)) + 100) / fine_step) + 1
    output_time = np.arange(fine_count, dtype=np.float64) * fine_step
    matching_second = None
    phase_matrix = None
    if diagnostic_nqc_second is not None:
        if variant != "source":
            raise ValueError("diagnostic NQC curvature is source-backed only")
        matching_second = _polynomial_matching_second(
            fine.output_time, fine.output_state, torch_fine_omega,
            torch_fine_mode, tmatch, **diagnostic_nqc_second)
    if diagnostic_nqc_phase_matrix is not None:
        if variant != "source":
            raise ValueError("diagnostic NQC phase matrix is source-backed only")
        phase_matrix = _polynomial_phase_matrix(
            fine.output_time, fine.output_state, torch_fine_omega,
            tmatch, **diagnostic_nqc_phase_matrix)
    tref = torch_attach(
        fine.output_time, fine.output_state, torch_fine_omega,
        torch_fine_mode, torch_fine_frequency, tnu, tchi, tsigma,
        torch.tensor(output_time, dtype=torch.float64),
        diagnostic_phase_matrix=phase_matrix,
        diagnostic_matching_second=matching_second)
    jref = jax_attach(
        jrun.fine_time, jrun.fine_state, jax_fine_omega,
        jax_fine_mode, jax_fine_frequency, jnu, jchi, jsigma,
        jnp.asarray(output_time))
    print(case_id, "fine attachments", flush=True)

    tcoarse = torch_coarse_mode * torch_factor(
        coarse.output_state[:, 0], coarse.output_state[:, 2],
        torch_coarse_omega, tref.calibration.coefficients)
    jcoarse = jax_coarse_mode * jax_factor(
        jrun.coarse_state[:, 0], jrun.coarse_state[:, 2],
        jax_coarse_omega, jref.calibration.coefficients)
    tsamples = _array(tref.mode)[::32]
    jsamples = _array(jref.mode)[::32]
    tmode = np.concatenate((_array(tcoarse[:restart]), tsamples))
    jmode = np.concatenate((_array(jcoarse[:jrun.restart_index]), jsamples))
    ttime = np.concatenate((_array(coarse.output_time[:restart]),
                            float(coarse.output_time[restart]) + output_time[::32]))
    jtime = np.concatenate((_array(jrun.coarse_time[:jrun.restart_index]),
                            float(jrun.coarse_time[jrun.restart_index]) + output_time[::32]))
    if tmode.shape != jmode.shape or ttime.shape != jtime.shape:
        raise ValueError("Torch and JAX whole-waveform output charts differ")
    checks = {
        "restart_index_equal": restart == jrun.restart_index,
        "whole_mode_relative_l2_at_most": _relative_l2(jmode, tmode) <= 1e-5,
        "coarse_prefix_relative_l2_at_most": _relative_l2(jmode[:restart], tmode[:restart]) <= 1e-5,
        "post_restart_relative_l2_at_most": _relative_l2(jmode[restart:], tmode[restart:]) <= 1e-5,
        "full_time_max_abs_at_most_M": float(np.max(np.abs(jtime - ttime))) <= 1e-8,
    }
    initial_state_max_abs = float(np.max(np.abs(np.asarray(initial_jax)
                                              - initial_torch.numpy())))
    if independent_init:
        control = json.loads(INDEPENDENT_INIT_PLAN.read_text())
        checks["independent_initial_state_max_abs_at_most"] = (
            initial_state_max_abs <= control["initial_state_max_abs_at_most"])
    row = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, omega_start=omega_start)
    torch_run = dict(time=ttime, mode=tmode, peak_time_M=float(tref.peak.time + coarse.output_time[restart]),
                     dt_geom=dt)
    jax_run = dict(time=jtime, mode=jmode, peak_time_M=float(jref.peak.time + jrun.coarse_time[jrun.restart_index]),
                   dt_geom=dt)
    torch_public = compare_public_binary(row, torch_run)
    jax_public = compare_public_binary(row, jax_run)
    print(case_id, "public comparisons", flush=True)
    result = {
        "schema": ("diffgw.seobnrv4.jax-source-independent-init.v1" if independent_init
                   else "diffgw.seobnrv4.jax-spliced-whole-waveform.v1"),
        "case_id": case_id, "variant": variant, "status": "diagnostic_complete",
        "scope": "Development-only full Torch/JAX waveform parity and descriptive public-binary comparison",
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "input": {"case_parameters": CASES[case_id], "coarse_output_step_M": dt,
                  "fine_output_step_M": fine_step, "fine_output_samples": fine_count,
                  "max_step_M": max_step, "atol": atol, "rtol": rtol},
        "public_binary": public_identity,
        "source_choices": SOURCE_FLUX_OPTIONS if variant == "source" else None,
        "diagnostic_nqc_second": diagnostic_nqc_second,
        "diagnostic_nqc_second_values": (_array(matching_second).tolist()
                                         if matching_second is not None else None),
        "diagnostic_nqc_phase_matrix": diagnostic_nqc_phase_matrix,
        "diagnostic_nqc_phase_matrix_values": (_array(phase_matrix).tolist()
                                               if phase_matrix is not None else None),
        "independent_initial_solve": independent_init,
        "initial_state_max_abs": initial_state_max_abs,
        "torch_initial_state": initial_torch.tolist(),
        "jax_initial_state": np.asarray(initial_jax).tolist(),
        "torch_restart_index": restart, "jax_restart_index": jrun.restart_index,
        "whole_output_samples": len(ttime),
        "whole_time_max_abs_M": float(np.max(np.abs(jtime - ttime))),
        "whole_mode_relative_l2": _relative_l2(jmode, tmode),
        "coarse_prefix_relative_l2": _relative_l2(jmode[:restart], tmode[:restart]),
        "post_restart_relative_l2": _relative_l2(jmode[restart:], tmode[restart:]),
        "torch_public": torch_public, "jax_public": jax_public,
        "checks": checks, "diagnostic_checks_passed": all(checks.values()),
        "elapsed_s": time.monotonic() - started,
        "release_qualified": False,
    }
    if anchor_control:
        from dev.native_rom.diagnose_peak_anchor import METHODS, fitted_metrics, public_mode

        nt, nm, rt, rm, mass_seconds = public_mode(row, torch_run)
        result["fitted_peak_public"] = {
            method: fitted_metrics(nt, nm, rt, rm, mass_seconds, method)
            for method in METHODS
        }
        result["anchor_control_plan_sha256"] = qualification.sha256(
            ROOT / "dev/native_rom/evidence/2026-09-25/jax_published_anchor_control_plan.json")
    if independent_init:
        result["independent_init_plan_sha256"] = qualification.sha256(INDEPENDENT_INIT_PLAN)
    name = ("jax_source_independent_init" if independent_init else
            "jax_published_anchor_control" if anchor_control else
            "jax_source_spliced_whole_waveform" if variant == "source" else
            "jax_spliced_whole_waveform")
    output = Path(f"dev/native_rom/evidence/2026-09-25/{name}_{case_id}.json")
    if write_result:
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(case_id, json.dumps({"checks": checks,
                              "torch_public_complex_l2": torch_public["amplitude_peak_anchored"]["metrics"]["complex_relative_l2"],
                              "jax_public_complex_l2": jax_public["amplitude_peak_anchored"]["metrics"]["complex_relative_l2"]},
                             sort_keys=True), flush=True)
    if return_series:
        if return_stages:
            targets = torch_targets(tnu, tchi)
            amplitude_rhs = torch.stack(targets[:3]) - torch.stack((
                _eval_batch_scalar(tref.nqc.amplitude_spline, tmatch),
                _eval_batch_scalar(tref.nqc.amplitude_spline, tmatch, derivative=1),
                matching_second[0] if matching_second is not None else
                _eval_batch_scalar(tref.nqc.amplitude_spline, tmatch, derivative=2)))
            phase_rhs = torch.stack((
                _eval_batch_scalar(tref.nqc.waveform_phase_spline, tmatch, derivative=1)
                - targets.frequency,
                (matching_second[4] if matching_second is not None else
                 _eval_batch_scalar(tref.nqc.waveform_phase_spline, tmatch, derivative=2))
                - targets.frequency_derivative))
            stage = {
                "coarse_time": _array(coarse.output_time),
                "coarse_state": _array(coarse.output_state),
                "fine_time": float(coarse.output_time[restart]) + _array(fine.output_time),
                "fine_state": _array(fine.output_state),
                "fine_base_mode": _array(torch_fine_mode),
                "fine_corrected_inspiral": _array(tref.corrected_inspiral),
                "fine_orbital_frequency": _array(torch_fine_omega),
                "nqc_coefficients": [_array(x) for x in tref.calibration.coefficients],
                "nqc_amplitude_matrix": _array(tref.nqc.amplitude_matching_matrix),
                "nqc_phase_matrix": _array(tref.nqc.phase_matching_matrix),
                "nqc_amplitude_rhs": _array(amplitude_rhs),
                "nqc_phase_rhs": _array(phase_rhs),
                "nqc_amplitude_condition": float(tref.calibration.amplitude_condition),
                "nqc_phase_condition": float(tref.calibration.phase_condition),
                "orbital_peak_time_M": float(coarse.output_time[restart] + tref.peak.time),
                "match_time_M": float(coarse.output_time[restart] + tref.match_time),
                "boundary_time_M": float(coarse.output_time[restart] + tref.boundary_time),
                "boundary_index": int(tref.boundary_index),
                "restart_index": restart,
            }
            return result, (ttime, tmode), (jtime, jmode), stage
        return result, (ttime, tmode), (jtime, jmode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(*CASES, "prospective"), required=True)
    parser.add_argument("--variant", choices=("published", "source"), default="published")
    parser.add_argument("--anchor-control", action="store_true")
    parser.add_argument("--independent-init", action="store_true")
    args = parser.parse_args()
    selected, variant = args.case, args.variant
    plan = json.loads((SOURCE_PLAN if variant == "source" else PLAN).read_text())
    expected_schema = ("diffgw.seobnrv4.jax-source-spliced-whole-waveform-plan.v1" if variant == "source"
                       else "diffgw.seobnrv4.jax-spliced-whole-waveform-plan.v1")
    if plan["schema"] != expected_schema:
        raise ValueError("wrong frozen plan schema")
    if args.independent_init:
        control = json.loads(INDEPENDENT_INIT_PLAN.read_text())
        baseline_paths = {
            name: ROOT / f"dev/native_rom/evidence/2026-09-25/"
                         f"jax_source_spliced_whole_waveform_{name}.json"
            for name in (control["pilot_case"], *control["prospective_cases"])
        }
        if (variant != "source" or args.anchor_control
                or control["schema"] != "diffgw.seobnrv4.jax-source-independent-init-plan.v1"
                or control["release_qualified"] is not False
                or qualification.sha256(SOURCE_PLAN) != control["source_plan_sha256"]
                or any(qualification.sha256(path) != control["source_results_sha256"][name]
                       for name, path in baseline_paths.items())):
            raise ValueError("independent-initial-state control differs from frozen inputs")
    if args.anchor_control:
        control = json.loads((ROOT / "dev/native_rom/evidence/2026-09-25/"
                              "jax_published_anchor_control_plan.json").read_text())
        if (variant != "published" or selected != control["case"]
                or control["schema"] != "diffgw.seobnrv4.jax-published-anchor-control-plan.v1"
                or control["release_qualified"] is not False
                or qualification.sha256(ROOT / "dev/native_rom/evidence/2026-09-25/"
                                        "jax_spliced_whole_waveform_strong_aligned.json")
                != control["published_prior_sha256"]
                or qualification.sha256(ROOT / "dev/native_rom/evidence/2026-09-25/"
                                        "source_spliced_anchor.json")
                != control["source_anchor_sha256"]):
            raise ValueError("frozen published anchor control differs")
    identity = _prepare_public(plan)
    names = (tuple(plan["prospective_cases"]) if selected == "prospective" else (selected,))
    for name in names:
        run(name, plan, identity, variant=variant, anchor_control=args.anchor_control,
            independent_init=args.independent_init)
