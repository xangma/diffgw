"""Dev-only frozen counterfactual for source-backed NQC matching curvature."""

import argparse
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
PLAN = EVIDENCE / "jax_source_nqc_curvature_plan.json"
LADDER_PLAN = EVIDENCE / "jax_source_tolerance_ladder_plan.json"


def _validate(plan, case):
    if (plan["schema"] != "diffgw.seobnrv4.jax-source-nqc-curvature-plan.v1"
            or plan["release_qualified"] is not False
            or q.sha256(SOURCE_PLAN) != plan["source_plan_sha256"]
            or q.sha256(LADDER_PLAN) != plan["ladder_plan_sha256"]
            or plan["levels"] != ["baseline", "refined"]
            or case not in (plan["pilot_case"], plan["prospective_case"])):
        raise ValueError("frozen NQC curvature plan changed")
    prior_path = EVIDENCE / f"jax_source_tolerance_ladder_{case}.json"
    if q.sha256(prior_path) != plan["ladder_result_sha256"][case]:
        raise ValueError(f"frozen natural-cubic tolerance result changed: {case}")
    ladder = json.loads(LADDER_PLAN.read_text())
    if (ladder["schema"] != "diffgw.seobnrv4.jax-source-tolerance-ladder-plan.v1"
            or ladder["release_qualified"] is not False):
        raise ValueError("frozen tolerance ladder plan changed")
    return ladder, json.loads(prior_path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("pilot", "prospective"), required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    case = plan[f"{args.case}_case"]
    ladder, prior = _validate(plan, case)
    source_plan = json.loads(SOURCE_PLAN.read_text())
    identity = _prepare_public(source_plan)
    started = time.monotonic()
    output = EVIDENCE / f"jax_source_nqc_curvature_{case}.json"
    result = {
        "schema": "diffgw.seobnrv4.jax-source-nqc-curvature.v1",
        "status": "running", "case_id": case, "release_qualified": False,
        "scope": "Torch-only local-polynomial NQC curvature counterfactual; JAX retains natural-cubic curvature",
        "plan_sha256": q.sha256(PLAN),
        "natural_cubic_result_sha256": q.sha256(EVIDENCE / f"jax_source_tolerance_ladder_{case}.json"),
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
            numerical_settings=settings, diagnostic_nqc_second=plan["candidate"],
            write_result=False, return_series=True)
        series[name] = torch_series
        result["levels"][name] = {
            "settings": settings,
            "torch": _series_summary(torch_series),
            "jax_natural_cubic": _series_summary(jax_series),
            "matching_second": diagnostic["diagnostic_nqc_second_values"],
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
        "candidate_drift_at_most_one_tenth_of_natural_cubic": (
            ratio <= 0.1),
    }
    result["counterfactual_checks_passed"] = all(result["checks"].values())
    result["elapsed_s"] = time.monotonic() - started
    result["status"] = "diagnostic_complete"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(case, json.dumps({"comparisons": result["comparisons"],
                            "checks": result["checks"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
