"""Post-hoc direct native/public control for a selected SXS case.

The selection follows inspection of NR residuals. This diagnoses a discrepancy;
it cannot serve as an independent release holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time

import torch

from . import compare_sxs_2123 as c
from . import compare_sxs_2123_public_control as public
from . import qualify_generator as q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--public-control", type=Path, required=True)
    parser.add_argument("--diagnostic-plan", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    primary = json.loads(args.primary.read_text())
    control = json.loads(args.public_control.read_text())
    diagnostic = json.loads(args.diagnostic_plan.read_text())
    simulation = diagnostic["simulation"]
    if (diagnostic["schema"] != "diffgw.seobnrv4.sxs-native-public-diagnostic-plan.v1"
            or diagnostic["qualification_claim"] is not False
            or diagnostic["primary_sha256"] != q.sha256(args.primary)
            or diagnostic["public_control_sha256"] != q.sha256(args.public_control)
            or primary["plan_sha256"] != q.sha256(args.plan)
            or control["plan_sha256"] != q.sha256(args.plan)
            or not primary["all_cases_succeeded"]
            or not control["all_cases_succeeded"]
            or simulation not in plan["cases"]):
        raise ValueError("diagnostic inputs changed")
    extraction_path = args.decoded_dir / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    if q.sha256(extraction_path) != primary["extraction_sha256"]:
        raise ValueError("NR extraction changed")
    torch.set_num_threads(1)
    case = plan["cases"][simulation]
    method = plan["method"]
    dt = method["grid_step_M"]
    windows = method["windows_relative_to_highest_resolution_amplitude_peak_M"]
    started = time.monotonic()
    native = q.native_run(case["row"], dict(name="native",
        atol=method["native_atol"], rtol=method["native_rtol"], dt_geom=dt),
        q.Settings(max_run_seconds=600.),
        diagnostic_fine_rkf45_resampling=method["diagnostic_fine_rkf45_resampling"])
    native_summary = q.summarize_run(native)
    if native_summary["array_sha256"] != primary["cases"][simulation]["native"]["array_sha256"]:
        raise ValueError("native waveform changed")
    public_wave = public.public_mode(case["row"], dt, diagnostic["public_call"])
    high = max(case["levels"])
    record = extraction["cases"][simulation]["extracted"][f"Lev{high}"]
    path = args.decoded_dir / record["path"]
    if q.sha256(path) != record["sha256"]:
        raise ValueError("NR mode changed")
    native_anchored = c.anchor(native, dt=dt)
    public_anchored = c.anchor(public_wave, dt=dt)
    nr_anchored = c.anchor(c.load_mode(path), dt=dt)
    support = tuple(primary["cases"][simulation]["comparisons"]
                    ["native_vs_high"]["common_support_M"])
    result = dict(schema="diffgw.seobnrv4.sxs-native-public-diagnostic.v1",
                  host=socket.gethostname(), simulation=simulation,
                  plan_sha256=q.sha256(args.plan), primary_sha256=q.sha256(args.primary),
                  public_control_sha256=q.sha256(args.public_control),
                  diagnostic_plan_sha256=q.sha256(args.diagnostic_plan),
                  script_sha256=q.sha256(__file__),
                  generator_source_manifest_sha256=q.canonical_hash(q.source_snapshot()),
                  native=native_summary,
                  anchors={name: {key: value for key, value in wave.items()
                                  if key not in ("time", "mode")}
                           for name, wave in (("native", native_anchored),
                                              ("public", public_anchored),
                                              ("NR", nr_anchored))},
                  native_vs_public=c.compare(native_anchored, public_anchored,
                                             windows, dt, support=support),
                  native_vs_nr=c.compare(native_anchored, nr_anchored,
                                         windows, dt, support=support),
                  public_vs_nr=c.compare(public_anchored, nr_anchored,
                                         windows, dt, support=support),
                  elapsed_seconds=time.monotonic()-started,
                  release_qualified=False)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, allow_nan=False,
                                        sort_keys=True) + "\n")
    print(json.dumps({name: result[name]["metrics"]["complex_relative_l2"]
                      for name in ("native_vs_public", "native_vs_nr", "public_vs_nr")}))


if __name__ == "__main__":
    main()
