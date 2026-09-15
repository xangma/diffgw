#!/usr/bin/env python3
"""Run the bounded adapter's short high-mass/high-start cost probe.

The script intentionally does not launch a long physical 20 Hz/BNS run.  A
plain invocation prints the proposed probe; pass ``--run`` only when that
small probe has been requested.  The native adaptive route is used when the
probe is run, and the output is limited to timing/grid/support metadata.
"""

# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch

from torchwave.approximants.eob.conditioned_waveform_clean import (
    conditioned_seobnrv4_fd,
)


def _scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return [_scalar(item) for item in value]
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute the proposed native probe")
    parser.add_argument("--mass1", type=float, default=80.0)
    parser.add_argument("--mass2", type=float, default=60.0)
    parser.add_argument("--chi1z", type=float, default=0.0)
    parser.add_argument("--chi2z", type=float, default=0.0)
    parser.add_argument("--f-low", type=float, default=80.0)
    parser.add_argument("--f-max", type=float, default=512.0)
    parser.add_argument("--delta-f", type=float, default=8.0)
    parser.add_argument("--distance", type=float, default=100.0)
    parser.add_argument("--inclination", type=float, default=0.0)
    parser.add_argument("--phic", type=float, default=0.0)
    parser.add_argument("--tc", type=float, default=0.0)
    parser.add_argument("--num-steps", type=int, default=2048)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    proposal = {
        "mass1": args.mass1,
        "mass2": args.mass2,
        "f_low": args.f_low,
        "f_max": args.f_max,
        "delta_f": args.delta_f,
        "num_steps": args.num_steps,
    }
    if not args.run:
        print(json.dumps({"status": "proposed", "probe": proposal}, sort_keys=True))
        return 0

    # Keep accidental invocations bounded to the intended high-mass/high-start
    # smoke probe.  A long low-frequency binary belongs to a separate review.
    if args.mass1 + args.mass2 < 80.0:
        raise SystemExit("refusing a low-mass run; use the high-mass probe defaults")
    if args.f_low < 64.0:
        raise SystemExit("refusing a low-start run; use f_low >= 64 Hz for this probe")
    if args.num_steps < 4:
        raise SystemExit("--num-steps must be at least four")

    started = time.perf_counter()
    result = conditioned_seobnrv4_fd(
        args.mass1,
        args.mass2,
        chi1z=args.chi1z,
        chi2z=args.chi2z,
        f_low=args.f_low,
        f_max=args.f_max,
        delta_f=args.delta_f,
        distance=args.distance,
        inclination=args.inclination,
        phic=args.phic,
        tc=args.tc,
        num_steps=args.num_steps,
        return_metadata=True,
    )
    elapsed = time.perf_counter() - started
    payload = {
        "status": "completed",
        "elapsed_seconds": elapsed,
        "probe": proposal,
        "sampling": {
            "n_time": result.sampling.n_time,
            "n_frequency": result.sampling.n_frequency,
            "delta_f": _scalar(result.sampling.delta_f),
            "dt_seconds": _scalar(result.dt_seconds),
        },
        "support": {
            "start_frequency": _scalar(result.support.start_frequency),
            "f_safe": _scalar(result.support.f_safe),
            "support_time": _scalar(result.support.support_time),
        },
        "adapter": {
            "omega_start": _scalar(result.omega_start),
            "dt_geom": _scalar(result.dt_geom),
            "physical_epoch": _scalar(result.physical_epoch),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
