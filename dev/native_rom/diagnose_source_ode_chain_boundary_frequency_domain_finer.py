"""Development-only third-tolerance check of continuous boundary frequency."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_source_tolerance_ladder import _compare, _series_summary
from dev.native_rom.diagnose_source_aligned_ode_chain_boundary_frequency import _capture_level
from dev.native_rom.diagnose_source_aligned_ode_chain_ringdown_inputs import _metrics, _mode
from dev.native_rom.diagnose_source_ode_chain_boundary_frequency_domain import _source_components
from torchwave.approximants.eob import numerical_attachment as attachment


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_ode_chain_boundary_frequency_domain_finer_plan.json"
OUTPUT = EVIDENCE / "source_ode_chain_boundary_frequency_domain_finer.json"


def _checked_plan() -> dict:
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-ode-chain-boundary-frequency-domain-finer-plan.v1"
            or plan["release_qualified"] is not False
            or plan["cases"] != ["q4_nonspinning", "high_start_mixed_spin", "strong_antialigned"]
            or plan["levels"] != ["refined", "finest"]
            or plan["runner_sentinel"] != {"half_width_M": 0.0, "degree": 0}
            or plan["fine_refinement_factor"] != 32):
        raise ValueError("frozen finer-domain plan changed")
    for name, frozen in plan["inputs"].items():
        if q.sha256(ROOT / frozen["path"]) != frozen["sha256"]:
            raise ValueError(f"frozen {name} input changed")
    if q.sha256(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("frozen finer-domain runner changed")
    return plan


def _run_case(case, plan, source_plan, ladder, identity, prior):
    field, hamiltonian, flux = _source_components(case)
    original_full_series = attachment._full_series_nqc
    original_eval = attachment._eval_batch_scalar
    original_ringdown = attachment.merger_ringdown_22
    runs, records = {}, {}
    for name in plan["levels"]:
        level = next(row for row in ladder["levels"] if row["name"] == name)
        settings = {"coarse_output_step_M": ladder["coarse_output_step_M"],
                    "max_step_M": ladder["max_step_M"],
                    "atol": level["atol"], "rtol": level["rtol"]}
        diagnostic, series, stage, inputs, controls = _capture_level(
            name, settings, source_plan, identity, {**plan, "case": case},
            field, hamiltonian, flux, original_full_series, original_eval)
        whole_time, candidate_mode = series
        whole_index = np.flatnonzero(np.asarray(whole_time) >= float(stage["boundary_time_M"]))
        chart_index = plan["fine_refinement_factor"]*(whole_index-int(stage["restart_index"]))
        candidate_inputs = dict(inputs)
        candidate_inputs["frequency"] = controls["continuous_frequency"]
        candidate_ring = _mode(candidate_inputs, original_ringdown=original_ringdown,
                               chart_index=chart_index)
        original_ring = _mode(inputs, original_ringdown=original_ringdown,
                              chart_index=chart_index)
        noop_mode = np.asarray(candidate_mode).copy()
        noop_mode[whole_index] = original_ring
        records[name] = {
            "settings": settings,
            "candidate": _series_summary(series),
            "noop_original_frequency": _series_summary((whole_time, noop_mode)),
            "match_time_M": float(stage["match_time_M"]),
            "boundary_time_M": float(stage["boundary_time_M"]),
            "boundary_index": int(stage["boundary_index"]),
            "restart_index": int(stage["restart_index"]),
            "ringdown_samples": int(len(whole_index)),
            "ringdown_reconstruction": _metrics(np.asarray(candidate_mode)[whole_index],
                                                candidate_ring),
            "support_count": inputs["support_count"],
            "support_end_M": inputs["support_end"],
            "frequency_original": controls["original_frequency"],
            "frequency_continuous": controls["continuous_frequency"],
            "frequency_raw": controls["raw_frequency"],
            "frequency_direct_raw_phase": controls["direct_raw_phase_first"],
            "frequency_backward_secant_replay_error": abs(
                controls["backward_secant"]-controls["original_frequency"]),
            "phase_basis_sample_replay_error": float(np.max(np.abs(
                np.asarray(controls["sampled_phase_basis"])
                - np.asarray(controls["ode_phase_basis"])))),
            "raw_frequency_sample_replay_error": abs(
                controls["raw_frequency"]-controls["sampled_raw_frequency"]),
            "public_complex_relative_l2": diagnostic["torch_public"]
                ["amplitude_peak_anchored"]["metrics"]["complex_relative_l2"],
        }
        runs[name] = {"candidate": series, "noop": (whole_time, noop_mode),
                      "ring_time": np.asarray(whole_time)[whole_index],
                      "ring_candidate": np.asarray(candidate_mode)[whole_index],
                      "ring_noop": original_ring}
        print(case, name, records[name]["candidate"]["mode_sha256"], flush=True)
    refined, finest = (runs[name] for name in plan["levels"])
    old = prior["cases"][case]
    replay = (records["refined"]["candidate"] == old["levels"]["refined"]["candidate"]
              and records["refined"]["noop_original_frequency"]
              == old["levels"]["refined"]["noop"])
    comparisons = {
        "candidate_baseline_to_refined": old["comparisons"]["candidate_whole"],
        "candidate_refined_to_finest": _compare(refined["candidate"], finest["candidate"]),
        "noop_baseline_to_refined": old["comparisons"]["noop_whole"],
        "noop_refined_to_finest": _compare(refined["noop"], finest["noop"]),
        "candidate_ringdown_refined_to_finest": _compare(
            (refined["ring_time"], refined["ring_candidate"]),
            (finest["ring_time"], finest["ring_candidate"])),
        "noop_ringdown_refined_to_finest": _compare(
            (refined["ring_time"], refined["ring_noop"]),
            (finest["ring_time"], finest["ring_noop"])),
    }
    comparisons["candidate_monotone"] = (
        comparisons["candidate_refined_to_finest"]["complex_relative_l2"]
        <= comparisons["candidate_baseline_to_refined"]["complex_relative_l2"])
    comparisons["noop_monotone"] = (
        comparisons["noop_refined_to_finest"]["complex_relative_l2"]
        <= comparisons["noop_baseline_to_refined"]["complex_relative_l2"])
    limits = plan["checks"]
    checks = {
        "refined_hash_replay_exact": replay,
        "candidate_refined_to_finest_at_most": comparisons["candidate_refined_to_finest"]
            ["complex_relative_l2"] <= limits["candidate_refined_to_finest_at_most"],
        "ringdown_reconstruction_at_most": all(
            records[name]["ringdown_reconstruction"]["max_absolute"]
            <= limits["ringdown_reconstruction_max_absolute_at_most"]
            for name in plan["levels"]),
        "direct_raw_phase_derivative_at_most": all(
            abs(records[name]["frequency_direct_raw_phase"]
                - records[name]["frequency_raw"])
            <= limits["direct_raw_phase_derivative_at_most"]
            for name in plan["levels"]),
        "backward_secant_replay_at_most": all(
            records[name]["frequency_backward_secant_replay_error"]
            <= limits["backward_secant_replay_at_most"] for name in plan["levels"]),
        "phase_basis_sample_replay_at_most": all(
            records[name]["phase_basis_sample_replay_error"]
            <= limits["phase_basis_sample_replay_at_most"] for name in plan["levels"]),
        "raw_frequency_sample_replay_at_most": all(
            records[name]["raw_frequency_sample_replay_error"]
            <= limits["raw_frequency_sample_replay_at_most"] for name in plan["levels"]),
        "same_boundary_chart_support": (
            records["refined"]["boundary_time_M"] == records["finest"]["boundary_time_M"]
            and records["refined"]["support_count"] == records["finest"]["support_count"]
            and comparisons["candidate_refined_to_finest"]["support_sample_difference"] == 0
            and comparisons["candidate_refined_to_finest"]["time_max_abs_M"] == 0.0
            and comparisons["noop_refined_to_finest"]["time_max_abs_M"] == 0.0),
        "finite_results": all(np.isfinite(value) for value in (
            records[name]["public_complex_relative_l2"]
            for name in plan["levels"])) and all(np.isfinite(
                comparisons[key]["complex_relative_l2"]) for key in (
                    "candidate_refined_to_finest", "noop_refined_to_finest",
                    "candidate_ringdown_refined_to_finest",
                    "noop_ringdown_refined_to_finest")),
    }
    return {"levels": {"baseline": old["levels"]["baseline"], **records},
            "comparisons": comparisons, "checks": checks,
            "prospective_checks_passed": all(checks.values())}


def main() -> None:
    plan = _checked_plan()
    source_plan = json.loads((ROOT / plan["inputs"]["source_plan"]["path"]).read_text())
    ladder = json.loads((ROOT / plan["inputs"]["ladder_plan"]["path"]).read_text())
    prior = json.loads((ROOT / plan["inputs"]["prior_result"]["path"]).read_text())
    identity = driver._prepare_public(source_plan)
    started = time.monotonic()
    result = {"schema": "diffgw.seobnrv4.source-ode-chain-boundary-frequency-domain-finer.v1",
              "status": "running", "release_qualified": False,
              "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
              "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
              "cases": {}}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for case in plan["cases"]:
        result["cases"][case] = _run_case(case, plan, source_plan, ladder, identity, prior)
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    result["prospective_checks_passed"] = all(
        row["prospective_checks_passed"] for row in result["cases"].values())
    result["elapsed_s"] = time.monotonic()-started
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({case: {"comparisons": row["comparisons"], "checks": row["checks"]}
                      for case, row in result["cases"].items()}), flush=True)


if __name__ == "__main__":
    main()
