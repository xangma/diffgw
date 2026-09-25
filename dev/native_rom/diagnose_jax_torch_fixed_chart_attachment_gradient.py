"""Dev-only JAX/Torch full attachment directional derivative parity."""

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_adaptive_physical_event import CASES
from dev.native_rom.diagnose_jax_full_series_nqc_physical import MODE_OPTIONS
from dev.native_rom.diagnose_jax_two_stage_trajectory import construct_case
from dev.native_rom.eob.jax.numerical_attachment_clean import (
    full_orbital_peak,
    traceable_fixed_chart_attached_mode22,
)
from dev.native_rom.eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.nqc import (
    _seobnrv4_nqc_targets_unchecked,
    _seobnrv4_nqc_time_offset_numerical_compatibility_unchecked,
    seobnrv4_nqc_parameters,
    seobnrv4_nqc_time_offset_numerical_compatibility,
)
from jaxwave.approximants.eob.published_waveform import published_mode22_and_frequency
from jaxwave.approximants.eob.remnant import eob_final_mass_fraction, hofmann_final_spin
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220
from torchwave.approximants.eob.nqc import (
    seobnrv4_nqc_time_offset_numerical_compatibility as torch_time_offset,
)
from torchwave.approximants.eob.numerical_attachment import (
    attach_seobnrv4_22_numerical_compatibility as torch_attach,
)

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "jax_torch_fixed_chart_attachment_gradient_plan.json"
OUTPUT = EVIDENCE / "jax_torch_fixed_chart_attachment_gradient.json"


def as_torch(value):
    array = np.asarray(value)
    dtype = torch.complex128 if np.iscomplexobj(array) else torch.float64
    return torch.as_tensor(array.copy(), dtype=dtype)


def relative_l2(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    return float(np.linalg.norm(actual-expected) /
                 max(np.linalg.norm(expected), 1e-30))


def main():
    plan = json.loads(PLAN.read_text())
    previous_path = EVIDENCE / "jax_fixed_chart_attachment_gradient.json"
    previous = json.loads(previous_path.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.jax-torch-fixed-chart-attachment-gradient-plan.v1"
            or plan["case"] != "q4_nonspinning" or plan["release_qualified"] is not False
            or q.sha256(previous_path) != plan["prior_jax_evidence_sha256"]
            or not previous["diagnostic_checks_passed"]):
        raise ValueError("frozen cross-backend plan changed")
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
    match = peak_time + seobnrv4_nqc_time_offset_numerical_compatibility(
        nqc_nu, nqc_chi)
    selected = np.flatnonzero(np.abs(np.asarray(time_grid-match)) <= 10.0)
    if selected.size != previous["window_samples"]:
        raise ValueError("physical window changed")
    slc = slice(int(selected[0]), int(selected[-1])+1)
    time_grid, state = time_grid[slc], state[slc]
    orbital_frequency = orbital_frequency[slc]
    base_mode, base_frequency = base_mode[slc], base_frequency[slc]
    array_hashes = {
        name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for name, value in (("time", time_grid), ("state", state),
                            ("orbital_frequency", orbital_frequency),
                            ("base_mode", base_mode))
    }
    hashes_equal = array_hashes == previous["input_array_sha256"]
    if not hashes_equal or abs(float(match)-previous["match_time_M"]) > 1e-12:
        raise ValueError("physical baseline differs from prior JAX check")
    relative_output = np.arange(-7.875, 19.875+0.125, 0.25)
    output_time = jnp.asarray(float(match)+relative_output)
    if len(output_time) != previous["output_samples"]:
        raise ValueError("attachment output chart changed")
    final_mass = eob_final_mass_fraction(m1, m2, chi1, chi2)
    final_spin = hofmann_final_spin(m1, m2, chi1, chi2, fit_order=(3, 4))
    qnm_frequency, qnm_damping = evaluate_kerr_qnm_220(final_spin)
    sigma = (qnm_frequency-1j*qnm_damping)/final_mass
    print(f"physical chart ready: {len(time_grid)} fine samples, "
          f"elapsed={time.monotonic()-started:.1f}s", flush=True)

    def jax_mode(perturbations):
        spin_shift, input_shift = perturbations
        spin = nqc_chi+spin_shift
        selected_match = peak_time + \
            _seobnrv4_nqc_time_offset_numerical_compatibility_unchecked(nqc_nu, spin)
        modified_r = state[:, 0]*(1+0.01*input_shift)
        modified_pr = state[:, 2]*(1+0.01*input_shift)
        modified_omega = orbital_frequency*(1+0.01*input_shift)
        modified_mode = base_mode*(1+0.01*input_shift) * \
            jnp.exp(-1j*0.01*input_shift*(time_grid-match))
        return traceable_fixed_chart_attached_mode22(
            time_grid, modified_r, modified_pr, modified_omega, modified_mode,
            selected_match, _seobnrv4_nqc_targets_unchecked(nqc_nu, spin),
            nqc_nu, spin, sigma, output_time)[0]

    ttime, tstate, tomega, tbase, tfrequency, toutput = (
        as_torch(value) for value in
        (time_grid, state, orbital_frequency, base_mode, base_frequency, output_time))
    tnu, tnqc_chi, tpeak, tmatch, tsigma = (
        as_torch(value) for value in (nqc_nu, nqc_chi, peak_time, match, sigma))

    def torch_mode(perturbations):
        spin_shift, input_shift = perturbations.unbind()
        spin = tnqc_chi+spin_shift
        selected_match = tpeak+torch_time_offset(tnu, spin)
        modified_state = torch.stack((
            tstate[:, 0]*(1+0.01*input_shift), tstate[:, 1],
            tstate[:, 2]*(1+0.01*input_shift), tstate[:, 3]), dim=-1)
        modified_omega = tomega*(1+0.01*input_shift)
        modified_mode = tbase*(1+0.01*input_shift) * \
            torch.exp(-1j*0.01*input_shift*(ttime-tmatch))
        return torch_attach(
            ttime, modified_state, modified_omega, modified_mode, tfrequency,
            tnu, spin, tsigma, toutput, match_time=selected_match,
            diagnostic_continuous_boundary=True).mode

    origin_jax, origin_torch = jnp.zeros(2), torch.zeros(2, dtype=torch.float64)
    jax_value = jax.jit(jax_mode)(origin_jax)
    jax_tangent = jax.jit(jax.jacfwd(jax_mode))(origin_jax)
    torch_value = torch_mode(origin_torch)
    torch_tangents = []
    for component in range(2):
        tangent_input = torch.eye(2, dtype=torch.float64)[component]
        _, tangent = torch.autograd.functional.jvp(
            torch_mode, origin_torch, tangent_input)
        torch_tangents.append(tangent.detach().numpy())
        print(f"Torch JVP {component+1}/2 ready: "
              f"elapsed={time.monotonic()-started:.1f}s", flush=True)
    mode_error = relative_l2(torch_value.detach().numpy(), jax_value)
    tangent_error = [relative_l2(torch_tangents[i], jax_tangent[:, i])
                     for i in range(2)]
    limits = plan["checks"]
    checks = {
        "same_physical_input_hashes": hashes_equal,
        "baseline_mode_relative_l2_at_most":
            mode_error <= limits["baseline_mode_relative_l2_at_most"],
        "tangent_relative_l2_at_most":
            max(tangent_error) <= limits["tangent_relative_l2_at_most"],
        "finite_tangents": all(np.isfinite(value).all()
                               for value in (np.asarray(jax_tangent), *torch_tangents)),
    }
    result = {
        "schema": "diffgw.seobnrv4.jax-torch-fixed-chart-attachment-gradient.v1",
        "status": "diagnostic_complete", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN),
        "source_sha256": q.sha256(__file__),
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "case": plan["case"], "input_array_sha256": array_hashes,
        "window_samples": len(time_grid), "output_samples": len(output_time),
        "baseline_mode_relative_l2": mode_error,
        "tangent_relative_l2": tangent_error,
        "jax_tangent_l2": [float(np.linalg.norm(np.asarray(jax_tangent)[:, i]))
                           for i in range(2)],
        "torch_tangent_l2": [float(np.linalg.norm(value)) for value in torch_tangents],
        "checks": checks, "diagnostic_checks_passed": all(checks.values()),
        "elapsed_s": time.monotonic()-started,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"checks": checks, "mode_error": mode_error,
                      "tangent_error": tangent_error}, sort_keys=True), flush=True)
    return 0 if result["diagnostic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
