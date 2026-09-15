#!/usr/bin/env python3
"""Bounded smoke diagnostic for the clean adaptive EOB driver.

The defaults deliberately start at a high dimensionless orbital frequency so
the probe measures a short trajectory.  This is a cost/shape diagnostic, not
a production 20 Hz BNS run or an oracle comparison.
"""

from __future__ import annotations

import argparse
import json
import time as walltime

import torch

from torchwave.approximants.eob.adaptive_driver_clean import (
    reconstruct_seobnrv4_22_adaptive,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omega-start", type=float, default=0.05)
    parser.add_argument("--mass1", type=float, default=30.0)
    parser.add_argument("--mass2", type=float, default=30.0)
    parser.add_argument("--dt-geom", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=1000,
                        help="accepted-step bound; defaults to a short probe")
    parser.add_argument("--ringdown-duration", type=float, default=10.0)
    args = parser.parse_args()
    if args.num_steps < 4 or args.num_steps > 5000:
        parser.error("--num-steps must be between 4 and 5000 for this bounded diagnostic")

    dtype = torch.float64
    omega = torch.tensor(args.omega_start, dtype=dtype)
    started = walltime.perf_counter()
    result = reconstruct_seobnrv4_22_adaptive(
        omega,
        args.mass1,
        args.mass2,
        dt_geom=args.dt_geom,
        num_steps=args.num_steps,
        ringdown_duration=args.ringdown_duration,
    )
    elapsed = walltime.perf_counter() - started
    coarse = result.trajectory.coarse
    fine = result.trajectory.fine
    report = {
        "dtype": str(dtype),
        "controller": "gsl_state_scaled",
        "rhs_batch_contract": "scalar_per_row_eob",
        "ragged_waveform_policy": "explicit_equal_length_error",
        "omega_start": args.omega_start,
        "mass1": args.mass1,
        "mass2": args.mass2,
        "dt_geom": args.dt_geom,
        "fine_step": float(fine.output_step),
        "refinement_factor": result.attachment.refinement_factor,
        "coarse_accepted": coarse.statistics.total_accepted,
        "coarse_rejected": coarse.statistics.total_rejected,
        "coarse_evaluations": coarse.statistics.total_evaluations,
        "coarse_output_samples": int(coarse.output_valid.sum()),
        "fine_accepted": fine.statistics.total_accepted,
        "fine_rejected": fine.statistics.total_rejected,
        "fine_evaluations": fine.statistics.total_evaluations,
        "fine_output_samples": int(fine.output_valid.sum()),
        "restart_index": result.trajectory.restart_index,
        "splice_samples": int(result.time.numel()),
        "valid_splice_samples": int(result.trajectory.valid.sum()),
        "peak_time": float(result.trajectory.peak.time),
        "match_time": float(result.attachment.match_time),
        "elapsed_seconds": elapsed,
        "finite_mode": bool(torch.isfinite(result.mode).all()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
