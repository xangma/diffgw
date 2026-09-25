"""Dev-only complete numerical-attachment parity on physical fine EOB rows."""

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
from dev.native_rom.diagnose_jax_full_series_nqc_physical import MODE_OPTIONS
from dev.native_rom.diagnose_jax_two_stage_trajectory import construct_case
from dev.native_rom.eob.jax.numerical_attachment_clean import (
    attach_seobnrv4_22_numerical_compatibility as jax_attach,
)
from dev.native_rom.eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.nqc import seobnrv4_nqc_parameters as jax_parameters
from jaxwave.approximants.eob.nqc import nqc_22_factor as jax_nqc_factor
from jaxwave.approximants.eob.published_waveform import published_mode22_and_frequency as jax_mode
from jaxwave.approximants.eob.remnant import (
    eob_final_mass_fraction as jax_final_mass,
    hofmann_final_spin as jax_final_spin,
)
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as JaxFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as JaxHam
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220 as jax_qnm
from torchwave.approximants.eob.adaptive_driver_clean import AlignedSpinV4Event, integrate_adaptive
from torchwave.approximants.eob.nqc import seobnrv4_nqc_parameters as torch_parameters
from torchwave.approximants.eob.nqc import nqc_22_factor as torch_nqc_factor
from torchwave.approximants.eob.numerical_attachment import (
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


def _array(value):
    return value.detach().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _relative_l2(left, right):
    return float(np.linalg.norm(_array(left) - _array(right)) / np.linalg.norm(_array(right)))


def _compare(left, right, output_time):
    boundary = float(right.boundary_time)
    tail = _array(output_time) >= boundary
    left_mode, right_mode = _array(left.mode), _array(right.mode)
    fields = ("boundary_time", "boundary_phase", "boundary_frequency",
              "boundary_amplitude", "match_time", "match_phase", "internal_step")
    metadata = {field + "_abs": abs(float(getattr(left, field)) - float(getattr(right, field)))
                for field in fields}
    return {
        "boundary_index_equal": int(left.boundary_index) == int(right.boundary_index),
        "boundary_index": int(left.boundary_index),
        "boundary_metadata": metadata,
        "boundary_metadata_max_abs": max(metadata.values()),
        "corrected_inspiral_relative_l2": _relative_l2(
            left.corrected_inspiral, right.corrected_inspiral),
        "full_mode_complex_relative_l2": _relative_l2(left_mode, right_mode),
        "ringdown_complex_relative_l2": _relative_l2(left_mode[tail], right_mode[tail]),
        "support_zero_pattern_equal": bool(np.array_equal(left_mode[tail] == 0, right_mode[tail] == 0)),
    }


def _spliced_comparison(plan, coarse, jrun, torch_field, jax_field,
                        th, tf, jh, jf, tref, independent, output_time):
    """Apply each fine calibration to its own coarse prefix before splicing."""
    restart = jrun.restart_index
    n_coarse = len(coarse.output_time)
    torch_restart = (n_coarse - math.ceil(150 / 0.5)
                     if (n_coarse - 1) * 0.5 >= 150 else n_coarse - math.ceil(n_coarse / 2))
    if restart != torch_restart:
        raise ValueError("Torch/JAX restart indices differ")
    tcoarse, _ = torch_mode(coarse.output_state, th, tf, **MODE_OPTIONS)
    jcoarse = jax.jit(lambda state: jax_mode(state, jh, jf, **MODE_OPTIONS)[0])(
        jrun.coarse_state)
    tomega = torch_field(coarse.output_state)[:, 1]
    jomega = jax.vmap(jax_field)(jrun.coarse_state)[:, 1]
    tcorrected = tcoarse * torch_nqc_factor(
        coarse.output_state[:, 0], coarse.output_state[:, 2], tomega,
        tref.calibration.coefficients)
    jcorrected = jcoarse * jax_nqc_factor(
        jrun.coarse_state[:, 0], jrun.coarse_state[:, 2], jomega,
        independent.calibration.coefficients)
    tmode = np.concatenate((_array(tcorrected[:restart]), _array(tref.mode)))
    jmode = np.concatenate((_array(jcorrected[:restart]), _array(independent.mode)))
    ttime = np.concatenate((_array(coarse.output_time[:restart]),
                            float(coarse.output_time[restart]) + output_time))
    jtime = np.concatenate((_array(jrun.coarse_time[:restart]),
                            float(jrun.coarse_time[restart]) + output_time))
    tmismatch = abs(complex(tcorrected[restart]) - complex(tref.corrected_inspiral[0]))
    jmismatch = abs(complex(jcorrected[restart]) - complex(independent.corrected_inspiral[0]))
    metrics = dict(
        restart_index=restart, prefix_samples=restart, full_samples=len(tmode),
        global_time_max_abs_M=float(np.max(abs(ttime-jtime))),
        prefix_complex_relative_l2=_relative_l2(jcorrected[:restart], tcorrected[:restart]),
        full_complex_relative_l2=_relative_l2(jmode, tmode),
        torch_restart_continuity_relative=float(tmismatch/abs(complex(tref.corrected_inspiral[0]))),
        jax_restart_continuity_relative=float(jmismatch/abs(complex(independent.corrected_inspiral[0]))),
        zero_pattern_equal=bool(np.array_equal(tmode == 0, jmode == 0)),
    )
    checks = dict(
        global_time_max_abs_M=metrics["global_time_max_abs_M"] <= plan["gates"]["max_global_time_difference_M"],
        prefix_complex_relative_l2=metrics["prefix_complex_relative_l2"] <= plan["gates"]["max_prefix_complex_relative_l2"],
        full_complex_relative_l2=metrics["full_complex_relative_l2"] <= plan["gates"]["max_full_complex_relative_l2"],
        torch_restart_continuity_relative=metrics["torch_restart_continuity_relative"] <= plan["gates"]["max_restart_continuity_relative"],
        jax_restart_continuity_relative=metrics["jax_restart_continuity_relative"] <= plan["gates"]["max_restart_continuity_relative"],
        zero_pattern_equal=metrics["zero_pattern_equal"],
    )
    return metrics, checks


def run(case_id, *, splice=False):
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
    tmass = torch_final_mass(*map(tensor, (m1, m2, chi1, chi2)))
    jmass = jax_final_mass(m1, m2, chi1, chi2)
    tspin = torch_final_spin(*map(tensor, (m1, m2, chi1, chi2)), fit_order=(3, 4))
    jspin = jax_final_spin(m1, m2, chi1, chi2, fit_order=(3, 4))
    tw, tdamp = torch_qnm(tspin)
    jw, jdamp = jax_qnm(jspin)
    tsigma = torch.complex(tw, -tdamp) / tmass
    jsigma = (jw - 1j * jdamp) / jmass
    end = max(float(fine.output_time[-1]), float(jrun.fine_time[-1])) + 150
    output_time = np.arange(0, end + 0.5, 0.5)
    toutput = torch.tensor(output_time, dtype=torch.float64)

    tref = torch_attach(
        fine.output_time, fine.output_state, torch_omega, torch_base, torch_frequency,
        tnu, tchi, tsigma, toutput)
    ttime = jnp.asarray(_array(fine.output_time))
    tstate = jnp.asarray(_array(fine.output_state))
    shared = jax_attach(
        ttime, tstate, jnp.asarray(_array(torch_omega)),
        jnp.asarray(_array(torch_base)), jnp.asarray(_array(torch_frequency)),
        jnu, jchi, jnp.asarray(complex(tsigma)), jnp.asarray(output_time))
    print(case_id, "same-input attachment", flush=True)
    independent = jax_attach(
        jrun.fine_time, jrun.fine_state, jax_omega, jax_base, jax_frequency,
        jnu, jchi, jsigma, jnp.asarray(output_time))
    print(case_id, "end-to-end attachment", flush=True)

    same = _compare(shared, tref, output_time)
    end_to_end = _compare(independent, tref, output_time)
    result = {
        "schema": "diffgw.seobnrv4.jax-numerical-attachment-physical.v1",
        "case_id": case_id, "status": "diagnostic_complete",
        "scope": "Torch/JAX full numerical attachment parity; no independent physical reference or release qualification",
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "input": {"case_parameters": CASES[case_id], "fine_step_M": fine_step,
                  "output_step_M": dt, "output_samples": len(output_time),
                  "max_step_M": max_step, "atol": 1e-10, "rtol": 1e-9,
                  "final_spin_fit_order": [3, 4]},
        "qnm_sigma_complex_abs": float(abs(complex(jsigma) - complex(tsigma))),
        "same_input": same, "end_to_end": end_to_end,
        "elapsed_s": time.monotonic() - started,
        "release_qualified": False,
    }
    result["checks"] = {
        "same_input_boundary_index_equal": same["boundary_index_equal"],
        "end_to_end_boundary_index_equal": end_to_end["boundary_index_equal"],
        "same_input_boundary_metadata_max_abs_at_most": same["boundary_metadata_max_abs"] <= 1e-7,
        "end_to_end_boundary_metadata_max_abs_at_most": end_to_end["boundary_metadata_max_abs"] <= 1e-5,
        "same_input_full_mode_relative_l2_at_most": same["full_mode_complex_relative_l2"] <= 1e-6,
        "end_to_end_full_mode_relative_l2_at_most": end_to_end["full_mode_complex_relative_l2"] <= 1e-5,
        "end_to_end_ringdown_relative_l2_at_most": end_to_end["ringdown_complex_relative_l2"] <= 1e-5,
        "same_input_support_zero_pattern_equal": same["support_zero_pattern_equal"],
        "end_to_end_support_zero_pattern_equal": end_to_end["support_zero_pattern_equal"],
    }
    result["diagnostic_checks_passed"] = all(result["checks"].values())
    if splice:
        from dev.native_rom.qualify_generator import sha256
        plan_path = Path("dev/native_rom/evidence/2026-09-25/jax_spliced_waveform_plan.json")
        plan = json.loads(plan_path.read_text())
        if (plan["schema"] != "diffgw.seobnrv4.jax-spliced-waveform-plan.v1"
                or case_id not in plan["cases"] or plan["release_qualified"] is not False):
            raise ValueError("frozen splice protocol changed")
        metrics, checks = _spliced_comparison(
            plan, coarse, jrun, torch_field, jax_field,
            th, tf, jh, jf, tref, independent, output_time)
        spliced = dict(schema="diffgw.seobnrv4.jax-spliced-waveform.v1",
                       case_id=case_id, status="diagnostic_complete",
                       release_qualified=False, plan_sha256=sha256(plan_path),
                       script_sha256=sha256(__file__),
                       scope="Torch/JAX coarse-prefix NQC and complete spliced waveform parity on a fixed output chart",
                       host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                       input=result["input"], metrics=metrics, checks=checks,
                       diagnostic_checks_passed=all(checks.values()),
                       elapsed_s=time.monotonic()-started)
        output = Path(f"dev/native_rom/evidence/2026-09-25/jax_spliced_waveform_{case_id}.json")
        output.write_text(json.dumps(spliced, indent=2, sort_keys=True) + "\n")
        print(case_id, json.dumps(spliced["checks"], sort_keys=True), flush=True)
    else:
        output = Path(f"dev/native_rom/evidence/2026-09-25/jax_numerical_attachment_physical_{case_id}.json")
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(case_id, json.dumps(result["checks"], sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(*CASES, "prospective"), required=True)
    parser.add_argument("--splice", action="store_true")
    args = parser.parse_args()
    selected = args.case
    names = (("q4_nonspinning", "strong_aligned", "strong_antialigned")
             if selected == "prospective" else (selected,))
    for name in names:
        run(name, splice=args.splice)
