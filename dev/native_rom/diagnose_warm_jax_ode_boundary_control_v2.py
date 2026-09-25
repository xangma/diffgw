"""Dev-only replay of warm ODE matching and boundary frequency on both backends."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import torch

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from dev.native_rom.diagnose_source_aligned_ode_chain_amplitude_v2 import _continuous_amplitude
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_matrix import _phase_matrix
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_rhs import _rhs_derivatives
from dev.native_rom.eob.jax import numerical_attachment_clean as jax_attachment
from torchwave.approximants.eob.adaptive_driver_clean import AlignedSpinV4Event, integrate_adaptive
from torchwave.approximants.eob import numerical_attachment as torch_attachment


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "warm_jax_ode_boundary_control_v2_plan.json"
OUTPUT = EVIDENCE / "warm_jax_ode_boundary_control_v2.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arr(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def relative_l2(left, right):
    return float(np.linalg.norm(left-right)/np.linalg.norm(right))


def metric(left, right):
    left, right = arr(left), arr(right)
    if left.shape != right.shape:
        raise ValueError("attachment comparison charts differ")
    return {"relative_l2": relative_l2(left, right),
            "max_abs": float(np.max(abs(left-right))), "samples": len(left)}


def main():
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.warm-jax-ode-boundary-plan.v2"
            or plan["release_qualified"] is not False
            or sha(__file__) != plan["script_sha256"]):
        raise ValueError("frozen ODE boundary runner changed")
    for item in plan["inputs"].values():
        if sha(ROOT/item["path"]) != item["sha256"]:
            raise ValueError("frozen input changed")
    v1 = json.loads((ROOT/plan["inputs"]["trajectory_v1"]["path"]).read_text())
    v2 = json.loads((ROOT/plan["inputs"]["attachment_v2"]["path"]).read_text())
    default = json.loads((ROOT/plan["inputs"]["default_control"]["path"]).read_text())
    matched = json.loads((ROOT/plan["inputs"]["matched_source_control"]["path"]).read_text())
    historical = json.loads((ROOT/plan["inputs"]["warm_whole_v3"]["path"]).read_text())
    if (v1["status"] != "diagnostic_complete"
            or v2["status"] != "diagnostic_complete"
            or default["status"] != "diagnostic_complete"
            or matched["status"] != "diagnostic_complete"
            or historical["status"] != "diagnostic_overlap_only"
            or tuple(v1["row_tuple"]) != tuple(plan["row_tuple"])):
        raise ValueError("trajectory/attachment controls changed")
    torch.set_num_threads(1)
    m1, m2, chi1, chi2, _omega = plan["row_tuple"]
    nu, delta = m1*m2/(m1+m2)**2, (m1-m2)/(m1+m2)
    def tensor(value):
        return torch.tensor(value, dtype=torch.float64)
    ham = driver.TorchSourceHam(tensor(nu), tensor(chi1), tensor(chi2), delta=tensor(delta))
    flux = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
    field = driver.TorchIntegrator(ham, flux).vector_field
    settings = plan["settings"]
    dt = settings["coarse_output_step_M"]
    state = torch.tensor(v1["warm_state"], dtype=torch.float64)
    result = {
        "schema": "diffgw.seobnrv4.warm-jax-ode-boundary-control.v2",
        "status": "running", "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": sha(PLAN), "script_sha256": sha(__file__),
        "row_tuple": plan["row_tuple"], "settings": settings,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    started = time.monotonic()
    coarse = integrate_adaptive(
        field, state, dt, t_end=settings["max_duration_M"],
        event=AlignedSpinV4Event("coarse"), initial_step=dt,
        max_step=settings["max_step_M"], controller="gsl_state_scaled",
        atol=settings["atol"], rtol=settings["rtol"])
    restart = len(coarse.output_time)-math.ceil(150/dt)
    fine_dt = dt/settings["fine_refinement_factor"]
    fine = integrate_adaptive(
        field, coarse.output_state[restart], fine_dt,
        t_end=settings["max_duration_M"], event=AlignedSpinV4Event("fine"),
        initial_step=fine_dt, max_step=settings["max_step_M"],
        controller="gsl_state_scaled", atol=settings["atol"], rtol=settings["rtol"])
    if (restart != v1["stages"]["coarse"]["torch_restart_index"]
            or len(fine.output_time) != v1["stages"]["fine"]["torch_output_samples"]):
        raise ValueError("Torch fine trajectory does not replay v1")
    ftime, fstate = fine.output_time, fine.output_state
    omega = field(fstate)[:, 1]
    raw, raw_frequency = driver.torch_mode(
        fstate, ham, flux, **(driver.MODE_OPTIONS | {"retain_pan_cubic_spin": True}))
    tnu, tchi = driver.torch_parameters(*map(tensor, (m1, m2, chi1, chi2)))
    tspin = driver.torch_final_spin(*map(tensor, (m1, m2, chi1, chi2)), fit_order=(3, 4))
    tw, tdamp = driver.torch_qnm(tspin)
    tmass = driver.torch_final_mass(*map(tensor, (m1, m2, chi1, chi2)))
    sigma = torch.complex(tw, -tdamp)/tmass
    match = tensor(v2["stages"]["torch_match_time_M"])
    output = torch.arange(plan["output_samples"], dtype=torch.float64)*fine_dt
    phase_capture, rhs_capture = [], []
    phase_matrix = _phase_matrix(
        ftime, fstate, omega, match, half_width_M=0.0, degree=0,
        field=field, capture=phase_capture)
    center_state = phase_capture[0]["interpolated_state"]
    amplitude_rows, amplitude_raw = _continuous_amplitude(
        center_state, ham, flux, field)
    amp_deriv = amplitude_rows[1:, 1:4]
    rhs_deriv = _rhs_derivatives(
        ftime, raw, match, center_state=center_state,
        hamiltonian=ham, flux=flux, field=field, capture=rhs_capture)
    rhs_deriv[0, :] = amplitude_rows[1:, 0]
    controls = {
        "amplitude_derivatives": arr(amp_deriv).tolist(),
        "amplitude_ode_rows": arr(amplitude_rows).tolist(),
        "amplitude_ode_raw": amplitude_raw,
        "phase_matrix": arr(phase_matrix).tolist(),
        "phase_ode_capture": phase_capture[0],
        "matching_rhs_derivatives": arr(rhs_deriv).tolist(),
        "rhs_ode_capture": rhs_capture[0],
        "continuous_boundary": True,
        "match_time_M": float(match),
    }
    tref = driver.torch_attach(
        ftime, fstate, omega, raw, raw_frequency, tnu, tchi, sigma, output,
        match_time=match, diagnostic_amplitude_derivatives=amp_deriv,
        diagnostic_phase_matrix=phase_matrix,
        diagnostic_matching_rhs_derivatives=rhs_deriv,
        diagnostic_continuous_boundary=True)
    jref = driver.jax_attach(
        *map(lambda value: jnp.asarray(arr(value)),
             (ftime, fstate, omega, raw, raw_frequency, tnu, tchi, sigma, output)),
        match_time=jnp.asarray(arr(match)),
        diagnostic_amplitude_derivatives=jnp.asarray(arr(amp_deriv)),
        diagnostic_phase_matrix=jnp.asarray(arr(phase_matrix)),
        diagnostic_matching_rhs_derivatives=jnp.asarray(arr(rhs_deriv)),
        diagnostic_continuous_boundary=True)
    boundary_index = int(tref.boundary_index)
    if boundary_index != int(jref.boundary_index):
        raise ValueError("unmatched NQC boundary index")
    with torch.enable_grad():
        center = fstate[boundary_index].detach().clone().requires_grad_(True)
        vector = field(center)
        _, raw_frequency_at_boundary = driver.torch_mode(
            center, ham, flux,
            **(driver.MODE_OPTIONS | {"retain_pan_cubic_spin": True}))
        phase_basis = driver.torch_basis(center[0], center[2], vector[1])[1]
        basis_first = torch.stack(tuple(
            (torch.autograd.grad(phase_basis[i], center,
                                 retain_graph=True)[0]*vector).sum()
            for i in range(2)))
        ode_frequency = (raw_frequency_at_boundary
                         -basis_first.dot(tref.calibration.coefficients.phase.detach())).detach()
    old_controls = historical["levels"]["refined"]["ringdown_boundary_controls"]
    if (boundary_index != old_controls["boundary_index"]
            or abs(float(match)-old_controls["match_time_local_M"]) > 1e-12):
        raise ValueError("historical refined boundary does not replay")
    original_t = torch_attachment.merger_ringdown_22
    original_j = jax_attachment.merger_ringdown_22
    ringdown_calls = {}

    def torch_ringdown(elapsed, ring_sigma, amplitude, amplitude_derivative,
                       frequency, phase, shape):
        ringdown_calls["torch"] = {
            "original_frequency": float(frequency),
            "injected_frequency": float(ode_frequency)}
        return original_t(elapsed, ring_sigma, amplitude, amplitude_derivative,
                          ode_frequency, phase, shape)

    def jax_ringdown(elapsed, ring_sigma, amplitude, amplitude_derivative,
                     frequency, phase, shape):
        ringdown_calls["jax"] = {
            "original_frequency": float(frequency),
            "injected_frequency": float(ode_frequency)}
        return original_j(elapsed, ring_sigma, amplitude, amplitude_derivative,
                          jnp.asarray(float(ode_frequency)), phase, shape)

    with (patch.object(torch_attachment, "merger_ringdown_22", torch_ringdown),
          patch.object(jax_attachment, "merger_ringdown_22", jax_ringdown)):
        tode = driver.torch_attach(
            ftime, fstate, omega, raw, raw_frequency, tnu, tchi, sigma, output,
            match_time=match, diagnostic_amplitude_derivatives=amp_deriv,
            diagnostic_phase_matrix=phase_matrix,
            diagnostic_matching_rhs_derivatives=rhs_deriv,
            diagnostic_continuous_boundary=True)
        jode = driver.jax_attach(
            *map(lambda value: jnp.asarray(arr(value)),
                 (ftime, fstate, omega, raw, raw_frequency,
                  tnu, tchi, sigma, output)),
            match_time=jnp.asarray(arr(match)),
            diagnostic_amplitude_derivatives=jnp.asarray(arr(amp_deriv)),
            diagnostic_phase_matrix=jnp.asarray(arr(phase_matrix)),
            diagnostic_matching_rhs_derivatives=jnp.asarray(arr(rhs_deriv)),
            diagnostic_continuous_boundary=True)
    if set(ringdown_calls) != {"torch", "jax"}:
        raise ValueError("ODE boundary injection did not reach both backends")
    tcoef, jcoef = tref.calibration.coefficients, jref.calibration.coefficients
    tmode, jmode = arr(tref.mode), arr(jref.mode)
    boundary = min(float(tref.boundary_time), float(jref.boundary_time))
    before, after = arr(output) < boundary, arr(output) >= boundary
    result["controls"] = controls
    result["previous_attempt"] = {
        "status": "incomplete_historical_replay",
        "reason": "matched local-polynomial NQC phase controls differed from historical Hermite/ODE controls",
        "historical_frequency_abs": json.loads((ROOT/plan["inputs"]["ode_v1"]["path"]).read_text())
            ["stages"]["ode_vs_historical_frequency_abs"],
    }
    result["historical_refined_boundary_controls"] = old_controls
    result["ode_boundary_control"] = {
        "boundary_index": boundary_index,
        "input_boundary_time_M": float(ftime[boundary_index]),
        "continuous_attachment_boundary_time_M": float(tode.boundary_time),
        "raw_frequency_at_boundary": float(raw_frequency_at_boundary.detach()),
        "phase_basis_at_boundary": arr(phase_basis.detach()).tolist(),
        "ode_phase_basis_first": arr(basis_first.detach()).tolist(),
        "phase_coefficients": arr(tref.calibration.coefficients.phase).tolist(),
        "injected_frequency": float(ode_frequency),
        "torch_ringdown_call": ringdown_calls["torch"],
        "jax_ringdown_call": ringdown_calls["jax"],
    }
    result["stages"] = {
        "match_time_abs_M": abs(float(tref.match_time)-float(jref.match_time)),
        "boundary_time_abs_M": abs(float(tref.boundary_time)-float(jref.boundary_time)),
        "boundary_index_abs": abs(int(tref.boundary_index)-int(jref.boundary_index)),
        "torch_boundary_frequency": float(tref.boundary_frequency),
        "jax_boundary_frequency": float(jref.boundary_frequency),
        "boundary_frequency_abs": abs(float(tref.boundary_frequency)-float(jref.boundary_frequency)),
        "coefficient_amplitude_max_abs": float(np.max(abs(arr(tcoef.amplitude)-arr(jcoef.amplitude)))),
        "coefficient_phase_max_abs": float(np.max(abs(arr(tcoef.phase)-arr(jcoef.phase)))),
        "corrected_inspiral": metric(jref.corrected_inspiral, tref.corrected_inspiral),
        "whole_attached_mode": metric(jmode, tmode),
        "pre_boundary_mode": metric(jmode[before], tmode[before]),
        "post_boundary_mode": metric(jmode[after], tmode[after]),
        "matched_source_vs_asymmetric_v2_ratio":
            metric(jmode, tmode)["relative_l2"]/
            v2["stages"]["whole_attached_mode"]["relative_l2"],
        "matched_source_vs_default_control_ratio":
            metric(jmode, tmode)["relative_l2"]/
            default["stages"]["whole_attached_mode"]["relative_l2"],
        "ode_corrected_inspiral": metric(
            jode.corrected_inspiral, tode.corrected_inspiral),
        "ode_whole_attached_mode": metric(jode.mode, tode.mode),
        "ode_pre_boundary_mode": metric(arr(jode.mode)[before], arr(tode.mode)[before]),
        "ode_post_boundary_mode": metric(arr(jode.mode)[after], arr(tode.mode)[after]),
        "ode_vs_baseline_torch_pre_boundary": metric(
            arr(tode.mode)[before], tmode[before]),
        "ode_vs_baseline_torch_post_boundary": metric(
            arr(tode.mode)[after], tmode[after]),
        "ode_vs_historical_frequency_abs": abs(
            float(ode_frequency)-old_controls["continuous_frequency"]),
        "ode_vs_historical_raw_frequency_abs": abs(
            float(raw_frequency_at_boundary.detach())-old_controls["raw_frequency"]),
        "ode_vs_historical_basis_first_max_abs": float(np.max(abs(
            arr(basis_first.detach())-np.asarray(old_controls["ode_phase_basis_first"])))),
        "ode_vs_historical_phase_coefficients_max_abs": float(np.max(abs(
            arr(tref.calibration.coefficients.phase)
            -np.asarray(old_controls["phase_coefficients"])))),
        "ode_vs_historical_amplitude_coefficients_max_abs": float(np.max(abs(
            arr(tref.calibration.coefficients.amplitude)
            -np.asarray(v2["stages"]["torch_coefficients"]["amplitude"])))),
        "ode_backend_boundary_time_abs_M": abs(
            float(tode.boundary_time)-float(jode.boundary_time)),
        "ode_backend_boundary_index_abs": abs(
            int(tode.boundary_index)-int(jode.boundary_index)),
        "ode_backend_boundary_amplitude_abs": abs(
            float(tode.boundary_amplitude)-float(jode.boundary_amplitude)),
        "ode_backend_boundary_phase_abs": abs(
            float(tode.boundary_phase)-float(jode.boundary_phase)),
    }
    result["checks"] = {
        "same_boundary": result["stages"]["boundary_time_abs_M"] <= 1e-12
                         and result["stages"]["boundary_index_abs"] == 0,
        "corrected_inspiral_parity":
            result["stages"]["corrected_inspiral"]["relative_l2"] <= 1e-8,
        "pre_boundary_mode_parity":
            result["stages"]["pre_boundary_mode"]["relative_l2"] <= 1e-8,
        "whole_attached_mode_parity":
            result["stages"]["whole_attached_mode"]["relative_l2"] <= 1e-8,
        "historical_frequency_replayed":
            result["stages"]["ode_vs_historical_frequency_abs"] <= 1e-9,
        "historical_phase_coefficients_replayed":
            result["stages"]["ode_vs_historical_phase_coefficients_max_abs"] <= 1e-9,
        "same_injected_frequency":
            ringdown_calls["torch"]["injected_frequency"]
            == ringdown_calls["jax"]["injected_frequency"],
        "ode_same_boundary":
            result["stages"]["ode_backend_boundary_time_abs_M"] <= 1e-12
            and result["stages"]["ode_backend_boundary_index_abs"] == 0,
        "ode_whole_attached_mode_parity":
            result["stages"]["ode_whole_attached_mode"]["relative_l2"] <= 1e-8,
        "ode_pre_boundary_unchanged":
            result["stages"]["ode_vs_baseline_torch_pre_boundary"]["relative_l2"] <= 1e-12,
    }
    result["status"] = "diagnostic_complete"
    result["elapsed_s"] = time.monotonic()-started
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
