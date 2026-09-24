"""Post-hoc public SEOBNRv4 control for the frozen five-case SXS comparison.

This is a diagnostic of shared residuals, not an independent validation set
or a release faithfulness test. No time, phase, amplitude, or parameter fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from . import compare_sxs_2123 as c
from . import compare_sxs_2123_public_control as public
from . import qualify_generator as q


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--control-plan", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    control = json.loads(args.control_plan.read_text())
    primary = json.loads(args.primary.read_text())
    extraction_path = args.decoded_dir / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.sxs-five-case-physical-plan.v1"
            or control["schema"] != "diffgw.seobnrv4.sxs-five-case-public-control-plan.v1"
            or control["qualification_claim"] is not False
            or control["primary_comparison_sha256"] != q.sha256(args.primary)
            or primary["status"] != "diagnostic_complete"
            or not primary["all_cases_succeeded"]
            or primary["plan_sha256"] != q.sha256(args.plan)
            or primary["extraction_sha256"] != q.sha256(extraction_path)
            or extraction["plan_sha256"] != q.sha256(args.plan)
            or set(primary["cases"]) != set(plan["cases"])):
        raise ValueError("public control inputs do not match completed primary result")
    identity = q.lal_binary_identity(control["binary_versions"]["lal"],
                                     control["binary_versions"]["lalsimulation"])
    method = plan["method"]
    dt = method["grid_step_M"]
    windows = method["windows_relative_to_highest_resolution_amplitude_peak_M"]
    result = dict(schema="diffgw.seobnrv4.sxs-five-case-public-control.v1",
                  plan_sha256=q.sha256(args.plan),
                  control_plan_sha256=q.sha256(args.control_plan),
                  primary_comparison_sha256=q.sha256(args.primary),
                  extraction_sha256=q.sha256(extraction_path),
                  script_sha256=q.sha256(__file__), lal_binary=identity,
                  status="running", release_qualified=False,
                  cases={})
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    for simulation, case in plan["cases"].items():
        started = time.monotonic()
        entry = dict(errors=[])
        result["cases"][simulation] = entry
        try:
            high = max(case["levels"])
            record = extraction["cases"][simulation]["extracted"][f"Lev{high}"]
            nr_path = args.decoded_dir / record["path"]
            if q.sha256(nr_path) != record["sha256"]:
                raise ValueError("decoded NR mode changed")
            nr = c.anchor(c.load_mode(nr_path), dt=dt)
            waveform = public.public_mode(case["row"], dt, control["public_call"])
            anchored = c.anchor(waveform, dt=dt)
            native_support = tuple(primary["cases"][simulation]["comparisons"]
                                   ["native_vs_high"]["common_support_M"])
            entry.update(public_call={key: waveform[key] for key in (
                "f_lower_Hz", "delta_t_s", "distance_Mpc", "inclination_rad")},
                anchors={name: {key: value for key, value in item.items()
                                 if key not in ("time", "mode")}
                         for name, item in (("public", anchored), (f"Lev{high}", nr))},
                public_vs_high=c.compare(anchored, nr, windows, dt,
                                         support=native_support),
                native_vs_high_complex_l2_from_primary=primary["cases"][simulation]
                    ["comparisons"]["native_vs_high"]["metrics"]["complex_relative_l2"],
                elapsed_seconds=time.monotonic()-started)
            print(f"{simulation} complete", flush=True)
        except Exception as exc:
            entry["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
            entry["elapsed_seconds"] = time.monotonic()-started
            print(f"{simulation} failed: {exc}", file=sys.stderr, flush=True)
        args.json_out.write_text(json.dumps(result, indent=2, allow_nan=False,
                                            sort_keys=True) + "\n")
    result["all_cases_succeeded"] = all(
        "public_vs_high" in entry and not entry["errors"]
        for entry in result["cases"].values())
    result["status"] = "diagnostic_complete"
    args.json_out.write_text(json.dumps(result, indent=2, allow_nan=False,
                                        sort_keys=True) + "\n")
    return 0 if result["all_cases_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
