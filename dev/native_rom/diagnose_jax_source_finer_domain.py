"""Dev-only common factor-512 to factor-1024 combined-control refinement."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from dev.native_rom import qualify_generator as q
from dev.native_rom import diagnose_jax_spliced_whole_waveform as waveform
from dev.native_rom.diagnose_jax_source_cadence import _shared_compare
from dev.native_rom.diagnose_jax_source_tolerance_ladder import _compare
from dev.native_rom.diagnose_jax_spliced_whole_waveform import (
    SOURCE_PLAN,
    _prepare_public,
    run,
)

EVIDENCE = Path(__file__).resolve().parents[2] / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "jax_source_finer_domain_plan.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.jax-source-finer-domain-plan.v1"
            or plan["release_qualified"] is not False
            or args.case not in plan["cases"]
            or plan["fine_refinement_factors"] != [512, 1024]
            or plan["polynomial"] != {"half_width_M": 2.0, "degree": 4}
            or q.sha256(SOURCE_PLAN) != plan["source_plan_sha256"]
            or q.sha256(waveform.__file__) != plan["run_source_sha256"]):
        raise ValueError("frozen finer-domain plan or source changed")
    previous = plan["cases"][args.case]
    prior_path = EVIDENCE / previous["prior_result"]
    if q.sha256(prior_path) != previous["prior_sha256"]:
        raise ValueError("prior numerical evidence changed")
    prior = json.loads(prior_path.read_text())
    if (prior["status"] != "diagnostic_complete"
            or prior["levels"]["refined"]["settings"]["fine_refinement_factor"] != 512
            or prior["levels"]["refined"]["native_backend_checks_passed"] is not True):
        raise ValueError("prior factor-512 baseline changed")
    source_plan = json.loads(SOURCE_PLAN.read_text())
    identity = _prepare_public(source_plan)
    output = EVIDENCE / f"jax_source_finer_domain_{args.case}.json"
    started = time.monotonic()
    result = dict(schema="diffgw.seobnrv4.jax-source-finer-domain.v1",
                  status="running", case_id=args.case, plan_sha256=q.sha256(PLAN),
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                  levels={}, comparisons={}, checks={}, release_qualified=False)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    series = {}
    try:
        for label, factor in zip(("baseline", "refined"),
                                 plan["fine_refinement_factors"]):
            print(args.case, label, "started", flush=True)
            settings = dict(coarse_output_step_M=plan["output_step_M"],
                            fine_refinement_factor=factor,
                            max_step_M=plan["max_step_M"],
                            atol=plan["atol"], rtol=plan["rtol"])
            diagnostic, torch_series, jax_series, stage = run(
                args.case, source_plan, identity, variant="source",
                independent_init=True, numerical_settings=settings,
                diagnostic_peak_policy="stationary",
                diagnostic_nqc_amplitude_derivatives=plan["polynomial"],
                diagnostic_nqc_phase_matrix=plan["polynomial"],
                diagnostic_nqc_rhs_derivatives=plan["polynomial"],
                diagnostic_continuous_boundary=True, diagnostic_jax_local=True,
                diagnostic_jax_local_fit="jax", write_result=False,
                return_series=True, return_stages=True)
            if abs(stage["jax_match_time_M"] - stage["match_time_M"]) > 1e-7:
                raise ValueError("Torch/JAX stationary match changed")
            for grid, _ in (torch_series, jax_series):
                if np.max(np.abs(np.diff(grid) - plan["output_step_M"])) > 1e-8:
                    raise ValueError("whole output chart changed")
            series[label] = dict(torch=torch_series, jax=jax_series)
            result["levels"][label] = dict(
                settings=settings, torch_jax=_compare(torch_series, jax_series),
                native_backend_checks_passed=diagnostic["diagnostic_checks_passed"],
                torch_match_time_M=stage["match_time_M"],
                jax_match_time_M=stage["jax_match_time_M"],
                torch_boundary_time_M=stage["boundary_time_M"],
                jax_boundary_time_M=stage["jax_boundary_time_M"])
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(args.case, label, "finished", flush=True)
        result["comparisons"] = {
            backend: _shared_compare(series["baseline"][backend],
                                     series["refined"][backend])
            for backend in ("torch", "jax")}
        limits = plan["checks"]
        result["checks"] = {
            "each_grid_torch_jax_complex_relative_l2_at_most": all(
                level["torch_jax"]["complex_relative_l2"]
                <= limits["each_grid_torch_jax_complex_relative_l2_at_most"]
                for level in result["levels"].values()),
            "baseline_torch_jax_reproduced_abs_at_most": abs(
                result["levels"]["baseline"]["torch_jax"]["complex_relative_l2"]
                - prior["levels"]["refined"]["torch_jax"]["complex_relative_l2"])
                <= limits["baseline_torch_jax_reproduced_abs_at_most"],
            "torch_shared_sample_complex_relative_l2_at_most": (
                result["comparisons"]["torch"]["complex_relative_l2"]
                <= limits["torch_shared_sample_complex_relative_l2_at_most"]),
            "jax_shared_sample_complex_relative_l2_at_most": (
                result["comparisons"]["jax"]["complex_relative_l2"]
                <= limits["jax_shared_sample_complex_relative_l2_at_most"]),
            "shared_time_max_abs_at_most_M": all(
                item["shared_time_max_abs_M"] <= limits["shared_time_max_abs_at_most_M"]
                for item in result["comparisons"].values()),
            "native_backend_checks_passed": all(
                level["native_backend_checks_passed"]
                for level in result["levels"].values()),
        }
        result["diagnostic_checks_passed"] = all(result["checks"].values())
        result["status"] = "diagnostic_complete"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = dict(type=type(exc).__name__, message=str(exc))
    result["elapsed_s"] = time.monotonic() - started
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    print(args.case, json.dumps(dict(status=result["status"],
                                checks=result["checks"],
                                comparisons=result["comparisons"],
                                error=result.get("error")), sort_keys=True),
          flush=True)
    if result["status"] != "diagnostic_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
