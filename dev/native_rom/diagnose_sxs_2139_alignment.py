"""Post-hoc bounded time/phase fit for the antialigned SXS:BBH:2139 row.

This determines how much of the fixed-frequency residual is a global
alignment effect. It is unweighted and cannot establish release faithfulness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import compare_sxs_2123 as c
from . import compare_sxs_2123_public_control as public
from . import optimize_sxs_2123_alignment as fit
from . import qualify_generator as q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--alignment-plan", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--array-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    primary = json.loads(args.primary.read_text())
    direct = json.loads(args.direct.read_text())
    alignment = json.loads(args.alignment_plan.read_text())
    simulation = "SXS:BBH:2139"
    if (alignment["schema"] != "diffgw.seobnrv4.sxs-2139-alignment-plan.v1"
            or alignment["qualification_claim"] is not False
            or alignment["primary_sha256"] != q.sha256(args.primary)
            or alignment["direct_sha256"] != q.sha256(args.direct)
            or primary["plan_sha256"] != q.sha256(args.plan)
            or direct["simulation"] != simulation):
        raise ValueError("alignment prerequisites changed")
    extraction_path = args.decoded_dir / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    if q.sha256(extraction_path) != primary["extraction_sha256"]:
        raise ValueError("NR extraction changed")
    torch.set_num_threads(1)
    case = plan["cases"][simulation]
    method = plan["method"]
    dt = method["grid_step_M"]
    started = time.monotonic()
    native = q.native_run(case["row"], dict(name="native",
        atol=method["native_atol"], rtol=method["native_rtol"], dt_geom=dt),
        q.Settings(max_run_seconds=600.),
        diagnostic_fine_rkf45_resampling=method["diagnostic_fine_rkf45_resampling"])
    if q.summarize_run(native)["array_sha256"] != direct["native"]["array_sha256"]:
        raise ValueError("native waveform changed")
    high = max(case["levels"])
    record = extraction["cases"][simulation]["extracted"][f"Lev{high}"]
    nr_path = args.decoded_dir / record["path"]
    if q.sha256(nr_path) != record["sha256"]:
        raise ValueError("NR mode changed")
    public_wave = public.public_mode(case["row"], dt, alignment["public_call"])
    waves = dict(native=c.anchor(native, dt=dt),
                 public=c.anchor(public_wave, dt=dt),
                 nr=c.anchor(c.load_mode(nr_path), dt=dt))
    args.array_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.array_out, **{
        f"{name}_{key}": wave[key] for name, wave in waves.items()
        for key in ("time", "mode")})
    window = tuple(alignment["window_relative_to_reference_peak_M"])
    bounds = tuple(alignment["allowed_time_shift_M"])
    pairs = {
        name: fit.optimized_pair(waves[candidate], waves[reference],
                                 window=window, dt=dt, lag_bounds=bounds)
        for name, candidate, reference in (
            ("native_vs_public", "native", "public"),
            ("native_vs_nr", "native", "nr"),
            ("public_vs_nr", "public", "nr"))}
    result = dict(schema="diffgw.seobnrv4.sxs-2139-alignment.v1",
                  plan_sha256=q.sha256(args.plan),
                  primary_sha256=q.sha256(args.primary),
                  direct_sha256=q.sha256(args.direct),
                  alignment_plan_sha256=q.sha256(args.alignment_plan),
                  script_sha256=q.sha256(__file__),
                  array_sha256=q.sha256(args.array_out),
                  pairs=pairs, elapsed_seconds=time.monotonic()-started,
                  detector_noise_weighted=False, release_qualified=False)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, allow_nan=False,
                                        sort_keys=True) + "\n")
    print(json.dumps({name: pair["optimized_overlap"]
                      for name, pair in pairs.items()}))


if __name__ == "__main__":
    main()
