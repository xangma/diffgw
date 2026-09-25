"""Dev-only Torch/JAX whole-mode parity after fine-grid attachment and decimation."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

from . import diagnose_jax_numerical_attachment_physical as base
from . import qualify_generator as q
from .diagnose_jax_adaptive_physical_event import CASES
from .diagnose_jax_two_stage_trajectory import construct_case
from .eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.nqc import nqc_22_factor as jax_nqc_factor
from jaxwave.approximants.eob.nqc import seobnrv4_nqc_parameters as jax_parameters
from jaxwave.approximants.eob.published_waveform import published_mode22_and_frequency as jax_mode
from jaxwave.approximants.eob.remnant import eob_final_mass_fraction as jax_final_mass
from jaxwave.approximants.eob.remnant import hofmann_final_spin as jax_final_spin
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220 as jax_qnm
from torchwave.approximants.eob.adaptive_driver_clean import AlignedSpinV4Event, integrate_adaptive
from torchwave.approximants.eob.nqc import nqc_22_factor as torch_nqc_factor
from torchwave.approximants.eob.nqc import seobnrv4_nqc_parameters as torch_parameters
from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency as torch_mode
from torchwave.approximants.eob.remnant import eob_final_mass_fraction as torch_final_mass
from torchwave.approximants.eob.remnant import hofmann_final_spin as torch_final_spin
from torchwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220 as torch_qnm

jax.config.update("jax_enable_x64", True)

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_fine_grid_splice_plan.json"
OUTPUT_ROOT = q.ROOT / "dev/native_rom/evidence/2026-09-25"


def _array(value):
    return value.detach().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _relative_l2(left, right):
    return float(np.linalg.norm(_array(left) - _array(right)) / np.linalg.norm(_array(right)))


def run(case_id: str) -> None:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.jax-fine-grid-splice-plan.v1"
            or case_id not in plan["cases"] or plan["release_qualified"] is not False):
        raise ValueError("frozen fine-grid splice plan changed")
    dt = plan["output_step_M"]
    refinement = plan["fine_refinement_factor"]
    fine_step = dt / refinement
    m1, m2, chi1, chi2, _ = CASES[case_id]
    initial_torch, initial_jax, torch_field, jax_field = construct_case(case_id)
    coarse = integrate_adaptive(
        torch_field, initial_torch, dt, t_end=1000.,
        event=AlignedSpinV4Event("coarse"), initial_step=dt, max_step=2.,
        controller="gsl_state_scaled", atol=1e-10, rtol=1e-9)
    n_coarse = len(coarse.output_time)
    restart = (n_coarse - math.ceil(150 / dt) if (n_coarse - 1) * dt >= 150
               else n_coarse - math.ceil(n_coarse / 2))
    fine = integrate_adaptive(
        torch_field, coarse.output_state[restart], fine_step, t_end=1000.,
        event=AlignedSpinV4Event("fine"), initial_step=fine_step, max_step=2.,
        controller="gsl_state_scaled", atol=1e-10, rtol=1e-9)
    jrun = integrate_two_stage_trajectory(
        jax_field, initial_jax, output_step=dt, max_duration=1000.,
        coarse_initial_step=dt, coarse_max_step=2., fine_max_step=2.,
        fine_refinement_factor=refinement, atol=1e-10, rtol=1e-9)
    if jrun.restart_index != restart or jrun.refinement_factor != refinement:
        raise ValueError("Torch/JAX restart or refinement differs")
    print(case_id, "trajectories", len(fine.output_time), flush=True)

    nu = m1 * m2 / (m1 + m2)**2
    delta = (m1 - m2) / (m1 + m2)
    def tensor(value):
        return torch.tensor(value, dtype=torch.float64)

    th = base.TorchHam(tensor(nu), tensor(chi1), tensor(chi2), delta=tensor(delta))
    tf = base.TorchFlux(spin_truncation="tabulated", retain_pan_cubic_spin=False)
    jh = base.JaxHam(nu, chi1, chi2, delta=delta)
    jf = base.JaxFlux(spin_truncation="tabulated", retain_pan_cubic_spin=False)
    tbase, tfreq = torch_mode(fine.output_state, th, tf, **base.MODE_OPTIONS)
    jbase, jfreq = jax.jit(lambda state: jax_mode(state, jh, jf, **base.MODE_OPTIONS))(
        jrun.fine_state)
    tomega = torch_field(fine.output_state)[:, 1]
    jomega = jax.vmap(jax_field)(jrun.fine_state)[:, 1]
    tnu, tchi = torch_parameters(tensor(m1), tensor(m2), tensor(chi1), tensor(chi2))
    jnu, jchi = jax_parameters(m1, m2, chi1, chi2)
    tspin = torch_final_spin(*map(tensor, (m1, m2, chi1, chi2)), fit_order=(3, 4))
    tw, tdamp = torch_qnm(tspin)
    tsigma = torch.complex(tw, -tdamp) / torch_final_mass(
        *map(tensor, (m1, m2, chi1, chi2)))
    jw, jdamp = jax_qnm(jax_final_spin(m1, m2, chi1, chi2, fit_order=(3, 4)))
    jsigma = (jw - 1j * jdamp) / jax_final_mass(m1, m2, chi1, chi2)

    # The match is intrinsic to the sampled fine trajectory, independent of
    # the later output chart. A short provisional call fixes a common extent.
    probe_time = np.arange(0, max(float(fine.output_time[-1]),
                                  float(jrun.fine_time[-1])) + dt, dt)
    tprobe = base.torch_attach(fine.output_time, fine.output_state, tomega,
                               tbase, tfreq, tnu, tchi, tsigma,
                               torch.tensor(probe_time, dtype=torch.float64))
    jprobe = base.jax_attach(jrun.fine_time, jrun.fine_state, jomega,
                             jbase, jfreq, jnu, jchi, jsigma, jnp.asarray(probe_time))
    count = max(math.ceil((float(tprobe.match_time)+plan["ringdown_duration_M"])
                          / fine_step),
                math.ceil((float(jprobe.match_time)+plan["ringdown_duration_M"])
                          / fine_step)) + 1
    output_time = np.arange(count) * fine_step
    attached_t = base.torch_attach(
        fine.output_time, fine.output_state, tomega, tbase, tfreq, tnu, tchi,
        tsigma, torch.tensor(output_time, dtype=torch.float64))
    attached_j = base.jax_attach(
        jrun.fine_time, jrun.fine_state, jomega, jbase, jfreq, jnu, jchi,
        jsigma, jnp.asarray(output_time))
    print(case_id, "fine attachments", count, flush=True)

    tcoarse, _ = torch_mode(coarse.output_state, th, tf, **base.MODE_OPTIONS)
    jcoarse = jax.jit(lambda state: jax_mode(state, jh, jf, **base.MODE_OPTIONS)[0])(
        jrun.coarse_state)
    tcorr = tcoarse * torch_nqc_factor(
        coarse.output_state[:, 0], coarse.output_state[:, 2],
        torch_field(coarse.output_state)[:, 1], attached_t.calibration.coefficients)
    jcorr = jcoarse * jax_nqc_factor(
        jrun.coarse_state[:, 0], jrun.coarse_state[:, 2],
        jax.vmap(jax_field)(jrun.coarse_state)[:, 1],
        attached_j.calibration.coefficients)
    indices = np.arange(0, count, refinement)
    tfine_mode = np.where(output_time <= float(attached_t.match_time)
                          + plan["ringdown_duration_M"], _array(attached_t.mode), 0.)
    jfine_mode = np.where(output_time <= float(attached_j.match_time)
                          + plan["ringdown_duration_M"], _array(attached_j.mode), 0.)
    tmode = np.concatenate((_array(tcorr[:restart]), tfine_mode[indices]))
    jmode = np.concatenate((_array(jcorr[:restart]), jfine_mode[indices]))
    ttime = np.concatenate((_array(coarse.output_time[:restart]),
                            float(coarse.output_time[restart])+output_time[indices]))
    jtime = np.concatenate((_array(jrun.coarse_time[:restart]),
                            float(jrun.coarse_time[restart])+output_time[indices]))
    metrics = dict(
        prefix_samples=restart, full_samples=len(tmode), fine_samples=count,
        max_global_time_difference_M=float(np.max(abs(ttime-jtime))),
        prefix_complex_relative_l2=_relative_l2(jcorr[:restart], tcorr[:restart]),
        full_complex_relative_l2=_relative_l2(jmode, tmode),
        fine_complex_relative_l2=_relative_l2(jfine_mode, tfine_mode),
        match_time_difference_M=abs(float(attached_t.match_time)-float(attached_j.match_time)),
        torch_restart_continuity_relative=float(abs(complex(tcorr[restart])-
            complex(attached_t.corrected_inspiral[0]))/abs(complex(attached_t.corrected_inspiral[0]))),
        jax_restart_continuity_relative=float(abs(complex(jcorr[restart])-
            complex(attached_j.corrected_inspiral[0]))/abs(complex(attached_j.corrected_inspiral[0]))),
        zero_pattern_equal=bool(np.array_equal(tmode == 0, jmode == 0)))
    checks = dict(
        global_time=metrics["max_global_time_difference_M"] <= plan["gates"]["max_global_time_difference_M"],
        prefix=metrics["prefix_complex_relative_l2"] <= plan["gates"]["max_prefix_complex_relative_l2"],
        full=metrics["full_complex_relative_l2"] <= plan["gates"]["max_full_complex_relative_l2"],
        fine=metrics["fine_complex_relative_l2"] <= plan["gates"]["max_fine_complex_relative_l2"],
        match_time=metrics["match_time_difference_M"] <= plan["gates"]["max_match_time_difference_M"],
        torch_restart=metrics["torch_restart_continuity_relative"] <= plan["gates"]["max_restart_continuity_relative"],
        jax_restart=metrics["jax_restart_continuity_relative"] <= plan["gates"]["max_restart_continuity_relative"],
        zero_pattern=metrics["zero_pattern_equal"])
    result = dict(schema="diffgw.seobnrv4.jax-fine-grid-splice.v1",
                  case_id=case_id, status="diagnostic_complete",
                  release_qualified=False, plan_sha256=q.sha256(PLAN),
                  script_sha256=q.sha256(__file__), host=platform.node(),
                  cwd=os.getcwd(), pid=os.getpid(), metrics=metrics, checks=checks,
                  diagnostic_checks_passed=all(checks.values()),
                  elapsed_s=time.monotonic()-started)
    path = OUTPUT_ROOT / f"jax_fine_grid_splice_{case_id}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(case_id, json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(*CASES, "prospective"), required=True)
    selected = parser.parse_args().case
    for name in (("q4_nonspinning", "strong_aligned", "strong_antialigned")
                 if selected == "prospective" else (selected,)):
        run(name)
