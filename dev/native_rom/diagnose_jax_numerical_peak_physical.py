"""Dev-only physical fine-trajectory orbital-peak comparison."""

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import jax
import numpy as np

from dev.native_rom.diagnose_jax_adaptive_physical_event import CASES
from dev.native_rom.diagnose_jax_two_stage_trajectory import construct_case
from dev.native_rom.eob.jax.numerical_attachment_clean import full_orbital_peak
from dev.native_rom.eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from torchwave.approximants.eob.adaptive_driver_clean import (
    AlignedSpinV4Event,
    integrate_adaptive,
)
from torchwave.approximants.eob.numerical_attachment import _full_orbital_peak

jax.config.update("jax_enable_x64", True)


def run(case_id):
    started = time.monotonic()
    initial_torch, initial_jax, torch_field, jax_field = construct_case(case_id)
    dt, max_duration, max_step = 0.5, 1000.0, 2.0
    coarse = integrate_adaptive(
        torch_field, initial_torch, dt, t_end=max_duration,
        event=AlignedSpinV4Event("coarse"), initial_step=dt,
        max_step=max_step, controller="gsl_state_scaled",
        atol=1e-10, rtol=1e-9,
    )
    n_coarse = len(coarse.output_time)
    restart = (n_coarse - math.ceil(150 / dt) if (n_coarse - 1) * dt >= 150
               else n_coarse - math.ceil(n_coarse / 2))
    fine_step = dt / 32
    fine = integrate_adaptive(
        torch_field, coarse.output_state[restart], fine_step,
        t_end=max_duration, event=AlignedSpinV4Event("fine"),
        initial_step=fine_step, max_step=max_step,
        controller="gsl_state_scaled", atol=1e-10, rtol=1e-9,
    )
    jrun = integrate_two_stage_trajectory(
        jax_field, initial_jax, output_step=dt,
        max_duration=max_duration, coarse_initial_step=dt,
        coarse_max_step=max_step, fine_max_step=max_step,
        atol=1e-10, rtol=1e-9,
    )
    torch_omega = torch_field(fine.output_state)[..., 1]
    jax_omega = jax.vmap(jax_field)(jrun.fine_state)[:, 1]
    torch_peak = _full_orbital_peak(fine.output_time, torch_omega)
    shared_peak = full_orbital_peak(fine.output_time.detach().numpy(),
                                   torch_omega.detach().numpy())
    jax_peak = full_orbital_peak(jrun.fine_time, jax_omega)

    def record(peak):
        return {"time_M": float(peak.time), "frequency": float(peak.frequency),
                "curvature": float(peak.curvature),
                "sample_index": int(peak.sample_index)}

    reference = record(torch_peak)
    same = record(shared_peak)
    end_to_end = record(jax_peak)
    result = {
        "schema": "diffgw.seobnrv4.jax-numerical-peak-physical.v1",
        "case_id": case_id,
        "status": "diagnostic_complete",
        "scope": "Scalar full-series peak rule on Torch fine samples and separately integrated JAX fine samples; no NQC or ringdown",
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "input": {"case_parameters": CASES[case_id],
                  "initial_state": initial_torch.tolist(),
                  "output_step_M": dt, "fine_step_M": fine_step,
                  "max_step_M": max_step, "atol": 1e-10, "rtol": 1e-9},
        "torch": reference,
        "jax_on_torch_samples": same,
        "jax_end_to_end": end_to_end,
        "comparison": {
            "same_input_peak_time_abs_M": abs(same["time_M"] - reference["time_M"]),
            "same_input_frequency_abs": abs(same["frequency"] - reference["frequency"]),
            "same_input_curvature_abs": abs(same["curvature"] - reference["curvature"]),
            "end_to_end_peak_time_abs_M": abs(end_to_end["time_M"] - reference["time_M"]),
            "end_to_end_frequency_abs": abs(end_to_end["frequency"] - reference["frequency"]),
            "end_to_end_curvature_abs": abs(end_to_end["curvature"] - reference["curvature"]),
            "fine_frequency_max_abs": float(np.max(np.abs(
                np.asarray(jax_omega) - torch_omega.detach().numpy()))),
        },
        "elapsed_s": time.monotonic() - started,
        "release_qualified": False,
    }
    comparison = result["comparison"]
    result["checks"] = {
        "all_sample_indices_equal": reference["sample_index"] == same["sample_index"] == end_to_end["sample_index"],
        "same_input_peak_time_abs_M_at_most": comparison["same_input_peak_time_abs_M"] <= 1e-5,
        "same_input_frequency_abs_at_most": comparison["same_input_frequency_abs"] <= 1e-8,
        "same_input_curvature_abs_at_most": comparison["same_input_curvature_abs"] <= 1e-7,
        "end_to_end_peak_time_abs_M_at_most": comparison["end_to_end_peak_time_abs_M"] <= 1e-4,
        "end_to_end_frequency_abs_at_most": comparison["end_to_end_frequency_abs"] <= 1e-8,
        "end_to_end_curvature_abs_at_most": comparison["end_to_end_curvature_abs"] <= 1e-7,
        "fine_frequency_max_abs_at_most": comparison["fine_frequency_max_abs"] <= 1e-8,
    }
    result["diagnostic_checks_passed"] = all(result["checks"].values())
    output = Path(f"dev/native_rom/evidence/2026-09-25/jax_numerical_peak_physical_{case_id}.json")
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(case_id, json.dumps(result["comparison"], sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    selected = parser.parse_args().case
    for name in (CASES if selected == "all" else (selected,)):
        run(name)
