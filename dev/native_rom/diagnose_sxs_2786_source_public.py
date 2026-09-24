"""Compare the post-hoc SXS:BBH:2786 source-choice waveform with pinned public v4."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import compare_sxs_2123 as c
from . import compare_sxs_2123_public_control as public
from . import qualify_generator as q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    root = q.ROOT / "dev/native_rom/evidence/2026-09-24"
    case_path = root / "sxs_second_holdout_plan.json"
    comparison_path = root / "sxs_second_holdout_comparison.json"
    public_plan_path = root / "sxs_2786_public_control_plan.json"
    public_result_path = root / "sxs_2786_public_control.json"
    source_result_path = root / "sxs_2786_source_choice_loose.json"
    case_plan = json.loads(case_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    public_plan = json.loads(public_plan_path.read_text())
    public_result = json.loads(public_result_path.read_text())
    source_result = json.loads(source_result_path.read_text())
    simulation = "SXS:BBH:2786"
    if (comparison["plan_sha256"] != q.sha256(case_path)
            or public_result["control_plan_sha256"] != q.sha256(public_plan_path)
            or public_result["comparison_sha256"] != q.sha256(comparison_path)
            or source_result["comparison_sha256"] != q.sha256(comparison_path)
            or source_result["release_qualified"] is not False):
        raise ValueError("source/public prerequisites changed")
    q.lal_binary_identity(public_plan["binary_versions"]["lal"],
                          public_plan["binary_versions"]["lalsimulation"])
    source_path = Path(source_result["array"]["path"])
    if q.sha256(source_path) != source_result["array"]["sha256"]:
        raise ValueError("source-choice waveform changed")
    with np.load(source_path, allow_pickle=False) as record:
        source = dict(time=record["source_choice_time"],
                      mode=record["source_choice_mode"])
    source.update(source_result["anchor"])
    dt = case_plan["method"]["grid_step_M"]
    generated = public.public_mode(case_plan["cases"][simulation]["row"], dt,
                                   public_plan["public_call"])
    reference = c.anchor(generated, dt=dt)
    for key in ("anchor_time_M", "peak_after_anchor_M", "peak_amplitude"):
        if not math.isclose(reference[key], public_result["anchors"]["public"][key],
                            rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"public replay changed: {key}")
    primary = comparison["cases"][simulation]
    fixed = c.compare(source, reference,
                      case_plan["method"][
                          "windows_relative_to_highest_resolution_amplitude_peak_M"],
                      dt, support=primary["comparisons"]["native_vs_high"][
                          "common_support_M"])
    result = dict(schema="diffgw.seobnrv4.sxs-2786-source-public.v1",
                  simulation=simulation, case_plan_sha256=q.sha256(case_path),
                  comparison_sha256=q.sha256(comparison_path),
                  public_plan_sha256=q.sha256(public_plan_path),
                  public_result_sha256=q.sha256(public_result_path),
                  source_result_sha256=q.sha256(source_result_path),
                  source_array_sha256=q.sha256(source_path),
                  script_sha256=q.sha256(__file__), fixed_crossing=fixed,
                  posthoc_selected_case=True, release_qualified=False)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
    print(json.dumps({"complex_relative_l2": fixed["metrics"]["complex_relative_l2"],
                      "peak_lag_difference_M": fixed["peak_lag_difference_M"],
                      "windows": {name: entry["metrics"]["complex_relative_l2"]
                                  for name, entry in fixed["windows"].items()}}))


if __name__ == "__main__":
    main()
