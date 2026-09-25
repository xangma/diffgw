"""Dev-only attachment sampling control on one factor-256 Torch trajectory."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_adaptive_physical_event import CASES
from dev.native_rom.diagnose_jax_source_cadence import _shared_compare
from dev.native_rom.diagnose_jax_spliced_whole_waveform import (
    MODE_OPTIONS,
    SOURCE_FLUX_OPTIONS,
    SOURCE_PLAN,
    TorchFlux,
    TorchSourceHam,
    _array,
    _eval_batch_scalar,
    _prepare_public,
    _relative_l2,
    _stationary_orbital_peak_time,
    run,
    torch_attach,
    torch_final_mass,
    torch_final_spin,
    torch_mode,
    torch_natural_cubic_spline,
    torch_offset,
    torch_parameters,
    torch_qnm,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "jax_source_fixed_trajectory_sampling_plan.json"


def _tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def _attach(time_grid, state, orbital_frequency, base_mode, base_frequency,
            output_time, match_time, nu, chi, sigma):
    result = torch_attach(
        time_grid, state, orbital_frequency, base_mode, base_frequency,
        nu, chi, sigma, _tensor(output_time), match_time=_tensor(match_time))
    return result, _array(result.mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("pilot", "prospective"), required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    case = plan[f"{args.case}_case"]
    stationary_path = EVIDENCE / f"jax_source_stationary_peak_control_{case}.json"
    if (plan["schema"] != "diffgw.seobnrv4.jax-source-fixed-trajectory-sampling-plan.v1"
            or plan["release_qualified"] is not False
            or plan["integration_fine_refinement_factor"] != 256
            or plan["decimation"] != 2
            or q.sha256(SOURCE_PLAN) != plan["source_plan_sha256"]
            or q.sha256(stationary_path) != plan["stationary_control_sha256"][case]):
        raise ValueError("frozen fixed-trajectory plan or prerequisites changed")
    prior = json.loads(stationary_path.read_text())
    source_plan = json.loads(SOURCE_PLAN.read_text())
    identity = _prepare_public(source_plan)
    output = EVIDENCE / f"jax_source_fixed_trajectory_sampling_{case}.json"
    started = time.monotonic()
    result = {
        "schema": "diffgw.seobnrv4.jax-source-fixed-trajectory-sampling.v1",
        "status": "running", "case_id": case, "plan_sha256": q.sha256(PLAN),
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "release_qualified": False,
        "prior_independently_integrated_grid_l2": prior["comparisons"]["torch"]["complex_relative_l2"],
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    try:
        settings = {
            "coarse_output_step_M": plan["output_step_M"],
            "fine_refinement_factor": plan["integration_fine_refinement_factor"],
            "max_step_M": plan["max_step_M"],
            "atol": plan["atol"], "rtol": plan["rtol"],
        }
        _, high_series, _, stage = run(
            case, source_plan, identity, variant="source", independent_init=True,
            numerical_settings=settings, diagnostic_peak_policy="stationary",
            write_result=False, return_series=True, return_stages=True)
        m1, m2, chi1, chi2, _ = CASES[case]
        nu_scalar = m1*m2/(m1+m2)**2
        delta = (m1-m2)/(m1+m2)
        hamiltonian = TorchSourceHam(_tensor(nu_scalar), _tensor(chi1),
                                     _tensor(chi2), delta=_tensor(delta))
        flux = TorchFlux(**SOURCE_FLUX_OPTIONS)
        state = _tensor(stage["fine_state"])
        base_mode, base_frequency = torch_mode(
            state, hamiltonian, flux, **(MODE_OPTIONS | {"retain_pan_cubic_spin": True}))
        base_mode_error = _relative_l2(_array(base_mode), stage["fine_base_mode"])
        if base_mode_error > 1e-12:
            raise ValueError("reconstructed base mode differs from source run")
        time_grid = _tensor(np.arange(len(state))*plan["output_step_M"]
                            / plan["integration_fine_refinement_factor"])
        orbital_frequency = _tensor(stage["fine_orbital_frequency"])
        nu, chi = torch_parameters(*map(_tensor, (m1, m2, chi1, chi2)))
        mass = torch_final_mass(*map(_tensor, (m1, m2, chi1, chi2)))
        spin = torch_final_spin(*map(_tensor, (m1, m2, chi1, chi2)), fit_order=(3, 4))
        omega, damping = torch_qnm(spin)
        sigma = torch.complex(omega, -damping)/mass
        restart = stage["restart_index"]
        post_count = len(high_series[0]) - restart
        t0 = float(stage["fine_time"][0])
        match_high = stage["match_time_M"] - t0
        high_factor = plan["integration_fine_refinement_factor"]
        high_output = np.arange((post_count-1)*high_factor+1)*plan["output_step_M"]/high_factor
        high_attach, high_post = _attach(
            time_grid, state, orbital_frequency, base_mode, base_frequency,
            high_output, match_high, nu, chi, sigma)
        high_reconstructed = np.concatenate((high_series[1][:restart], high_post[::high_factor]))
        high_reproduction_l2 = _relative_l2(high_reconstructed, high_series[1])
        decimation = plan["decimation"]
        low_factor = high_factor//decimation
        low_time = time_grid[::decimation]
        low_state = state[::decimation]
        low_omega = orbital_frequency[::decimation]
        low_mode = base_mode[::decimation]
        low_frequency = base_frequency[::decimation]
        low_output = np.arange((post_count-1)*low_factor+1)*plan["output_step_M"]/low_factor
        peak_spline = torch_natural_cubic_spline(low_time, low_omega)
        peak_low = _stationary_orbital_peak_time(
            _array(low_time), _array(low_omega),
            lambda x: _eval_batch_scalar(peak_spline, _tensor(x), derivative=1),
            lambda x: _eval_batch_scalar(peak_spline, _tensor(x), derivative=2))
        match_low = peak_low + float(torch_offset(nu, chi))
        controls = {}
        for label, match in (("fixed_match", match_high),
                             ("recomputed_stationary_peak", match_low)):
            attached, post = _attach(
                low_time, low_state, low_omega, low_mode, low_frequency,
                low_output, match, nu, chi, sigma)
            series = (high_series[0], np.concatenate((high_series[1][:restart],
                                                      post[::low_factor])))
            controls[label] = {
                "comparison_to_full_sampling": _shared_compare(high_series, series),
                "match_time_M": t0 + match,
                "nqc_coefficients": [_array(x).tolist()
                                     for x in attached.calibration.coefficients],
                "amplitude_condition": float(attached.calibration.amplitude_condition),
                "phase_condition": float(attached.calibration.phase_condition),
            }
        result.update({
            "status": "diagnostic_complete", "settings": settings,
            "base_mode_reconstruction_l2": base_mode_error,
            "high_attachment_reconstruction_l2": high_reproduction_l2,
            "high_match_time_M": stage["match_time_M"],
            "recomputed_low_peak_slope": float(_eval_batch_scalar(
                peak_spline, _tensor(peak_low), derivative=1)),
            "recomputed_match_time_shift_M": match_low-match_high,
            "high_nqc_coefficients": [_array(x).tolist()
                                      for x in high_attach.calibration.coefficients],
            "controls": controls,
        })
        limits = plan["checks"]
        result["checks"] = {
            "reconstructed_high_complex_relative_l2_at_most": (
                high_reproduction_l2 <= limits["reconstructed_high_complex_relative_l2_at_most"]),
            "shared_time_max_abs_at_most_M": all(
                item["comparison_to_full_sampling"]["shared_time_max_abs_M"]
                <= limits["shared_time_max_abs_at_most_M"]
                for item in controls.values()),
            "stationary_peak_slope_max_abs_at_most": (
                abs(result["recomputed_low_peak_slope"])
                <= limits["stationary_peak_slope_max_abs_at_most"]),
        }
        result["diagnostic_checks_passed"] = all(result["checks"].values())
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    result["elapsed_s"] = time.monotonic()-started
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(case, json.dumps({"status": result["status"],
                            "checks": result.get("checks"),
                            "controls": result.get("controls"),
                            "error": result.get("error")}, sort_keys=True), flush=True)
    return 0 if result["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
