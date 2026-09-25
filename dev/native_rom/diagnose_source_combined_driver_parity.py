"""Replay the frozen combined local-NQC control through the long-waveform driver."""

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_spliced_whole_waveform import (
    CASES,
    SOURCE_PLAN,
    _prepare_public,
    run,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_combined_driver_parity_plan.json"
OUTPUT = EVIDENCE / "source_combined_driver_parity_q4_nonspinning.json"


def main():
    plan = json.loads(PLAN.read_text())
    previous = EVIDENCE / "jax_source_combined_backend_parity_q4_nonspinning.json"
    if (plan["schema"] != "diffgw.seobnrv4.source-combined-driver-parity-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "q4_nonspinning"
            or plan["polynomial"] != {"half_width_M": 2.0, "degree": 4}
            or q.sha256(SOURCE_PLAN) != plan["source_plan_sha256"]
            or q.sha256(previous) != plan["prior_combined_result_sha256"]):
        raise ValueError("frozen combined-driver plan or prerequisites changed")
    torch.set_num_threads(1)
    started = time.monotonic()
    result = dict(schema="diffgw.seobnrv4.source-combined-driver-parity.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), host=platform.node(), cwd=os.getcwd(),
                  pid=os.getpid(), checks={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    try:
        source_plan = json.loads(SOURCE_PLAN.read_text())
        identity = _prepare_public(source_plan)
        numerical = dict(coarse_output_step_M=plan["output_step_M"],
                         fine_refinement_factor=plan["fine_refinement_factor"],
                         max_step_M=plan["max_step_M"],
                         atol=plan["atol"], rtol=plan["rtol"])
        original, (original_time, original_mode), _, stage = run(
            plan["case"], source_plan, identity, variant="source",
            independent_init=True, numerical_settings=numerical,
            diagnostic_peak_policy="stationary",
            diagnostic_nqc_amplitude_derivatives=plan["polynomial"],
            diagnostic_nqc_phase_matrix=plan["polynomial"],
            diagnostic_nqc_rhs_derivatives=plan["polynomial"],
            diagnostic_continuous_boundary=True, diagnostic_jax_local=True,
            diagnostic_jax_local_fit="jax", write_result=False,
            return_series=True, return_stages=True)
        row = dict(zip(("mass1", "mass2", "chi1", "chi2", "omega_start"),
                       CASES[plan["case"]]))
        driver = q.native_run(
            row, dict(name="combined_driver", dt_geom=plan["output_step_M"],
                      atol=plan["atol"], rtol=plan["rtol"]),
            q.Settings(ringdown_duration_M=100.0),
            diagnostic_spin_truncation="tabulated",
            diagnostic_mode_21_higher_spin="rho_hybrid",
            diagnostic_mode_31_higher_spin="lal_v4_rho",
            diagnostic_mode_32_spin_square="lal_v4_zero",
            diagnostic_mode_33_higher_spin="lal_v4_rho",
            diagnostic_mode_43_higher_spin="lal_v4_rho",
            diagnostic_hamiltonian_log_derivative="lal_v4",
            diagnostic_hamiltonian_calibration="lal_v4_rounded",
            diagnostic_qnm_source="independent_kerr",
            diagnostic_coarse_initial_step_M=plan["output_step_M"],
            diagnostic_fine_max_step_M=plan["max_step_M"],
            diagnostic_fine_refinement_factor=plan["fine_refinement_factor"],
            diagnostic_combined_local_fit=plan["polynomial"],
            return_nqc_coefficients=True)
        count = min(len(driver["time"]), len(original_time))
        metrics = {
            "sample_count_difference": len(driver["time"]) - len(original_time),
            "whole_time_max_abs_M": float(np.max(np.abs(driver["time"][:count]
                                                       - original_time[:count]))),
            "whole_mode_complex_relative_l2": float(np.linalg.norm(
                driver["mode"][:count] - original_mode[:count])
                / np.linalg.norm(original_mode[:count])),
            "match_time_abs_M": abs(driver["match_time_M"] - stage["match_time_M"]),
            "boundary_time_abs_M": abs(driver["boundary_time_M"]
                                       - stage["boundary_time_M"]),
            "nqc_coefficient_max_abs": max(float(np.max(np.abs(
                np.asarray(driver["nqc_coefficients"][name]) - stage["nqc_coefficients"][i])))
                for i, name in enumerate(("amplitude", "phase"))),
        }
        limits = plan["checks"]
        result.update(metrics=metrics, original_torch_jax=original["whole_mode_relative_l2"],
                      original_torch_public=original["torch_public"]["amplitude_peak_anchored"]
                      ["metrics"]["complex_relative_l2"], driver=q.summarize_run(driver),
                      public_binary=identity)
        metric_names = {
            "whole_mode_complex_relative_l2_at_most": "whole_mode_complex_relative_l2",
            "whole_time_max_abs_at_most_M": "whole_time_max_abs_M",
            "match_time_abs_at_most_M": "match_time_abs_M",
            "boundary_time_abs_at_most_M": "boundary_time_abs_M",
            "nqc_coefficient_max_abs_at_most": "nqc_coefficient_max_abs",
        }
        result["checks"] = {
            "support_equal": metrics["sample_count_difference"] == 0,
            **{name: metrics[metric_names[name]] <= limit
               for name, limit in limits.items()},
        }
        result["diagnostic_checks_passed"] = all(result["checks"].values())
        result["status"] = "diagnostic_complete"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = dict(type=type(exc).__name__, message=str(exc))
    result["elapsed_s"] = time.monotonic() - started
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status=result["status"], checks=result["checks"],
                          metrics=result.get("metrics"), error=result.get("error")),
                     sort_keys=True), flush=True)
    if result["status"] != "diagnostic_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
