"""Check public ringdown origins against the pinned SEOBNRv4 high-rate grid.

The public origins were inferred independently from held-out postpeak modes.
This retrospective check does not fit a waveform or change the generator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import qualify_generator as q


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/public_ringdown_grid_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / value for name, value in plan["inputs"].items()}
    if (plan["schema"] != "diffgw.seobnrv4.public-ringdown-grid-plan.v1"
            or plan["output_step_M"] != 0.5
            or plan["resampling_threshold"] != 1/50
            or plan["grid_alignment_tolerance_M"] != 1e-6
            or plan["release_qualified"] is not False
            or {name: q.sha256(path) for name, path in paths.items()}
               != plan["input_sha256"]):
        parser.error("frozen grid plan or inverse inputs differ")
    ratio = 50*plan["output_step_M"]
    resample_factor = 1
    while resample_factor < ratio:
        resample_factor *= 2
    high_step_M = plan["output_step_M"]/resample_factor
    rows = {}
    for name, path in paths.items():
        inverse = json.loads(path.read_text())
        if (inverse["status"] != "diagnostic_complete"
                or not inverse["ansatz_holdout_passed"]
                or inverse["release_qualified"] is not False):
            parser.error(f"frozen {name} inverse is not valid")
        sampled_peak_M = -inverse["public_fitted_peak"]["offset_samples"]*plan["output_step_M"]
        public_t0_M = inverse["public_amplitude_fit"]["t0_M"]
        steps = (public_t0_M-sampled_peak_M)/high_step_M
        nearest = round(steps)
        residual_M = (steps-nearest)*high_step_M
        if abs(residual_M) > plan["grid_alignment_tolerance_M"]:
            raise ValueError(f"{name} public origin misses the high-rate grid")
        native_t0_M = inverse["native_boundary"]["time_from_fitted_peak_M"]
        rows[name] = dict(row=inverse["row"], public_sampled_peak_M=sampled_peak_M,
                          public_inferred_boundary_M=public_t0_M,
                          nearest_high_grid_offset_from_sampled_peak=nearest,
                          public_high_grid_residual_M=residual_M,
                          native_boundary_from_own_fitted_peak_M=native_t0_M,
                          native_minus_public_own_peak_boundary_M=native_t0_M-public_t0_M,
                          native_minus_public_boundary_in_high_steps=(native_t0_M-public_t0_M)/high_step_M)
    scope = dict(q.SCOPE)
    scope["description"] = "Retrospective grid alignment of two independently inverted public ringdowns; each waveform uses its own fitted peak chart."
    result = dict(schema="diffgw.seobnrv4.public-ringdown-grid.v1",
                  status="diagnostic_complete", release_qualified=False,
                  scope=scope,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256=plan["input_sha256"],
                  historical_source=plan["historical_source"],
                  output_step_M=plan["output_step_M"],
                  resample_factor=resample_factor, high_step_M=high_step_M,
                  rows=rows,
                  interpretation="Integer public offsets support an actual high-rate attachment knot. Native/public differences in separately peak-anchored charts do not by themselves locate the underlying dynamics error or establish release parity.")
    q.write_evidence(args.json_out, result)
    print(json.dumps(dict(status=result["status"], high_step_M=high_step_M,
                          rows={name: {"public_grid_offset": value["nearest_high_grid_offset_from_sampled_peak"],
                                       "public_grid_residual_M": value["public_high_grid_residual_M"]}
                                for name, value in rows.items()})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
