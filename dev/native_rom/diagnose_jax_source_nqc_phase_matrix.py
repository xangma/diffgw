"""Dev-only aligned-spin NQC phase-basis matching-matrix control."""

import json
import os
import platform
import time
from pathlib import Path

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_source_tolerance_ladder import _compare, _series_summary
from dev.native_rom.diagnose_jax_spliced_whole_waveform import (
    SOURCE_PLAN,
    _prepare_public,
    run,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "jax_source_nqc_phase_matrix_plan.json"
LADDER_PLAN = EVIDENCE / "jax_source_tolerance_ladder_plan.json"


def main():
    plan = json.loads(PLAN.read_text())
    case = plan["case"]
    inputs = EVIDENCE / f"jax_source_nqc_matching_inputs_{case}.json"
    prior_path = EVIDENCE / f"jax_source_tolerance_ladder_{case}.json"
    if (plan["schema"] != "diffgw.seobnrv4.jax-source-nqc-phase-matrix-plan.v1"
            or plan["release_qualified"] is not False
            or case != "strong_aligned"
            or plan["levels"] != ["baseline", "refined"]
            or q.sha256(SOURCE_PLAN) != plan["source_plan_sha256"]
            or q.sha256(LADDER_PLAN) != plan["ladder_plan_sha256"]
            or q.sha256(inputs) != plan["matching_input_result_sha256"]
            or q.sha256(prior_path) != plan["natural_cubic_ladder_result_sha256"]):
        raise ValueError("frozen phase-matrix plan or prerequisites changed")
    prior = json.loads(prior_path.read_text())
    ladder = json.loads(LADDER_PLAN.read_text())
    source_plan = json.loads(SOURCE_PLAN.read_text())
    identity = _prepare_public(source_plan)
    started = time.monotonic()
    output = EVIDENCE / f"jax_source_nqc_phase_matrix_{case}.json"
    result = {
        "schema": "diffgw.seobnrv4.jax-source-nqc-phase-matrix.v1",
        "status": "running", "case_id": case,
        "scope": "Torch-only local-polynomial phase-basis matrix; JAX retains natural cubic",
        "plan_sha256": q.sha256(PLAN), "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "levels": {}, "comparisons": {}, "checks": {},
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    series = {}
    for name in plan["levels"]:
        level = next(value for value in ladder["levels"] if value["name"] == name)
        settings = {
            "coarse_output_step_M": ladder["coarse_output_step_M"],
            "max_step_M": ladder["max_step_M"],
            "atol": level["atol"], "rtol": level["rtol"],
        }
        diagnostic, torch_series, jax_series = run(
            case, source_plan, identity, variant="source", independent_init=True,
            numerical_settings=settings,
            diagnostic_nqc_phase_matrix=plan["candidate"],
            write_result=False, return_series=True)
        series[name] = torch_series
        result["levels"][name] = {
            "settings": settings,
            "torch": _series_summary(torch_series),
            "jax_natural_cubic": _series_summary(jax_series),
            "matching_phase_matrix": diagnostic["diagnostic_nqc_phase_matrix_values"],
            "torch_to_jax_natural_cubic": _compare(torch_series, jax_series),
            "torch_public_complex_relative_l2": diagnostic["torch_public"]["amplitude_peak_anchored"]["metrics"]["complex_relative_l2"],
            "runner_backend_parity_checks_passed": diagnostic["diagnostic_checks_passed"],
        }
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(case, name, result["levels"][name]["torch_to_jax_natural_cubic"], flush=True)
    candidate = _compare(series["baseline"], series["refined"])
    natural = prior["comparisons"]["torch_baseline_to_refined"]
    ratio = candidate["complex_relative_l2"] / natural["complex_relative_l2"]
    result["comparisons"] = {
        "candidate_torch_baseline_to_refined": candidate,
        "natural_cubic_torch_baseline_to_refined": natural,
        "candidate_to_natural_cubic_drift_ratio": ratio,
    }
    limits = plan["checks"]
    result["checks"] = {
        "candidate_baseline_to_refined_complex_relative_l2_at_most": (
            candidate["complex_relative_l2"]
            <= limits["candidate_baseline_to_refined_complex_relative_l2_at_most"]),
        "candidate_drift_at_most_one_tenth_of_natural_cubic": ratio <= 0.1,
    }
    result["counterfactual_checks_passed"] = all(result["checks"].values())
    result["elapsed_s"] = time.monotonic() - started
    result["status"] = "diagnostic_complete"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(case, json.dumps({"comparisons": result["comparisons"],
                            "checks": result["checks"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
