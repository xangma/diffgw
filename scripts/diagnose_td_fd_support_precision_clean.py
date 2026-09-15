#!/usr/bin/env python3
"""Emit reproducible public-API controls for clean TD support arithmetic.

This diagnostic evaluates support plans only; it does not generate a native
trajectory or waveform.  The public calls are the four scalar timing APIs
listed in the numerical support contract.
"""

from __future__ import annotations

import json

import lal
import lalsimulation as lalsim
import numpy as np
import torch

from torchwave.approximants.eob.td_fd_conditioning_clean import plan_td_support


CASES = (
    (30.0, 20.0, 20.0, 0.0, 0.0),
    (1.4, 1.4, 30.0, 0.0, 0.0),
    (80.0, 60.0, 80.0, 0.0, 0.0),
    (42.0, 17.0, 35.0, -0.3, 0.2),
    (5.0, 4.0, 20.0, 0.7, 0.2),
    (100.0, 5.0, 10.0, -0.95, 0.0),
)


def ulp_delta(clean: float, public: float) -> int:
    """Return the binary64 bit-distance for positive finite values."""

    clean_bits = np.asarray(clean, dtype=np.float64).view(np.int64).item()
    public_bits = np.asarray(public, dtype=np.float64).view(np.int64).item()
    return int(clean_bits - public_bits)


def run_case(m1: float, m2: float, f_low: float, chi1: float, chi2: float) -> dict[str, object]:
    plan = plan_td_support(m1, m2, f_low, chi1, chi2, dtype=torch.float64)
    m1_si = m1 * lal.MSUN_SI
    m2_si = m2 * lal.MSUN_SI
    total_si = m1_si + m2_si
    safe = float(plan.f_safe)
    chirp = float(lalsim.SimInspiralChirpTimeBound(safe, m1_si, m2_si, chi1, chi2))
    merge = float(lalsim.SimInspiralMergeTimeBound(m1_si, m2_si))
    ringdown = float(
        lalsim.SimInspiralRingdownTimeBound(total_si, float(plan.spin_bound))
    )
    public_support = 1.1 * chirp + (merge + ringdown) + 3.0 / safe
    public_start = float(
        lalsim.SimInspiralChirpStartFrequencyBound(public_support, m1_si, m2_si)
    )
    clean_start = float(plan.start_frequency)
    return {
        "mass1_solar": m1,
        "mass2_solar": m2,
        "f_low_hz": f_low,
        "chi1z": chi1,
        "chi2z": chi2,
        "f_safe_hz": safe,
        "clean_support_time_seconds": float(plan.support_time),
        "public_support_time_seconds": public_support,
        "clean_start_frequency_hz": clean_start,
        "public_start_frequency_hz": public_start,
        "clean_start_frequency_hex": clean_start.hex(),
        "public_start_frequency_hex": public_start.hex(),
        "clean_minus_public_ulp": ulp_delta(clean_start, public_start),
        "public_controls": {
            "chirp_time_seconds": chirp,
            "merge_time_seconds": merge,
            "ringdown_time_seconds": ringdown,
            "calls": [
                "SimInspiralChirpTimeBound",
                "SimInspiralMergeTimeBound",
                "SimInspiralRingdownTimeBound",
                "SimInspiralChirpStartFrequencyBound",
            ],
        },
    }


def main() -> None:
    print(json.dumps({"cases": [run_case(*case) for case in CASES]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
