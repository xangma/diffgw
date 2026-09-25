"""Dev-only extended source-backed Torch/JAX whole-waveform ODE ladder."""

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
PLAN = EVIDENCE / "jax_source_extended_tolerance_plan.json"
LADDER_PLAN = EVIDENCE / "jax_source_tolerance_ladder_plan.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("pilot", "prospective"), required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    case = plan[f"{args.case}_case"]
    prior_path = EVIDENCE / f"jax_source_tolerance_ladder_{case}.json"
    phase_path = EVIDENCE / "jax_source_nqc_phase_matrix_strong_aligned.json"
    if (plan["schema"] != "diffgw.seobnrv4.jax-source-extended-tolerance-plan.v1"
            or plan["release_qualified"] is not False
            or [item["name"] for item in plan["levels"]]
            != ["finest_replay", "ultra", "maximum"]
            or q.sha256(SOURCE_PLAN) != plan["source_plan_sha256"]
            or q.sha256(LADDER_PLAN) != plan["ladder_plan_sha256"]
            or q.sha256(prior_path) != plan["ladder_result_sha256"][case]
            or q.sha256(phase_path) != plan["phase_matrix_control_sha256"]):
        raise ValueError("frozen extended tolerance plan or prerequisites changed")
    prior = json.loads(prior_path.read_text())
    ladder = json.loads(LADDER_PLAN.read_text())
    source_plan = json.loads(SOURCE_PLAN.read_text())
    identity = _prepare_public(source_plan)
    started = time.monotonic()
    output = EVIDENCE / f"jax_source_extended_tolerance_{case}.json"
    result = {
        "schema": "diffgw.seobnrv4.jax-source-extended-tolerance.v1",
        "status": "running", "case_id": case,
        "plan_sha256": q.sha256(PLAN), "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "levels": {}, "comparisons": {}, "checks": {},
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    series = {}
    try:
        for level in plan["levels"]:
            name = level["name"]
            settings = {
                "coarse_output_step_M": ladder["coarse_output_step_M"],
                "max_step_M": ladder["max_step_M"],
                "atol": level["atol"], "rtol": level["rtol"],
            }
            diagnostic, torch_series, jax_series = run(
                case, source_plan, identity, variant="source", independent_init=True,
                numerical_settings=settings, write_result=False, return_series=True)
            series[name] = {"torch": torch_series, "jax": jax_series}
            result["levels"][name] = {
                "settings": settings,
                "torch": _series_summary(torch_series),
                "jax": _series_summary(jax_series),
                "torch_jax": _compare(torch_series, jax_series),
                "diagnostic_checks_passed": diagnostic["diagnostic_checks_passed"],
            }
            if name == "finest_replay":
                observed = result["levels"][name]["torch_jax"]["complex_relative_l2"]
                expected = prior["levels"]["finest"]["torch_jax"]["complex_relative_l2"]
                if abs(observed - expected) > 1e-12:
                    raise ValueError(f"frozen finest backend comparison changed: {case}")
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(case, name, result["levels"][name]["torch_jax"], flush=True)
        for backend in ("torch", "jax"):
            for left, right in (("finest_replay", "ultra"), ("ultra", "maximum")):
                key = f"{backend}_{left}_to_{right}"
                result["comparisons"][key] = _compare(
                    series[left][backend], series[right][backend])
        limits = plan["checks"]
        result["checks"] = {
            "ultra_to_maximum_complex_relative_l2_at_most": all(
                result["comparisons"][f"{backend}_ultra_to_maximum"]["complex_relative_l2"]
                <= limits["ultra_to_maximum_complex_relative_l2_at_most"]
                for backend in ("torch", "jax")),
            "torch_jax_maximum_complex_relative_l2_at_most": (
                result["levels"]["maximum"]["torch_jax"]["complex_relative_l2"]
                <= limits["torch_jax_maximum_complex_relative_l2_at_most"]),
            "each_level_backend_checks_passed": all(
                value["diagnostic_checks_passed"] for value in result["levels"].values()),
        }
        result["diagnostic_checks_passed"] = all(result["checks"].values())
        result["status"] = "diagnostic_complete"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    result["elapsed_s"] = time.monotonic() - started
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(case, json.dumps({"status": result["status"],
                            "checks": result["checks"],
                            "comparisons": result["comparisons"],
                            "error": result.get("error")}, sort_keys=True), flush=True)
    return 0 if result["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
