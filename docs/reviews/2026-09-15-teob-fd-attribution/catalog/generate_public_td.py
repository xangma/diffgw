#!/usr/bin/env python3
"""Reproduce the cached catalog SimIMRTEOBResumS exact-start TD capture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lal
import lalsimulation as lalsim
import numpy as np


M1 = 1.4
M2 = 1.4
FLOW = 30.0
FMAX = 500.0
DT = 1.0 / 1024.0
DF = 0.5
DISTANCE_MPC = 100.0
INCLINATION = 0.0
PHI_REF = 0.0
SPINS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def public_support_bounds() -> dict[str, float]:
    m1_si = M1 * lal.MSUN_SI
    m2_si = M2 * lal.MSUN_SI
    total_si = (M1 + M2) * lal.MSUN_SI
    chirp = float(lalsim.SimInspiralChirpTimeBound(FLOW, m1_si, m2_si, 0.0, 0.0))
    merge = float(lalsim.SimInspiralMergeTimeBound(m1_si, m2_si))
    spin_bound = 0.686
    ringdown = float(lalsim.SimInspiralRingdownTimeBound(total_si, spin_bound))
    plunge_ringdown = merge + ringdown
    support = 1.1 * chirp + plunge_ringdown + 3.0 / FLOW
    start = float(lalsim.SimInspiralChirpStartFrequencyBound(support, m1_si, m2_si))
    return {
        "chirp_time_seconds": chirp,
        "merge_time_seconds": merge,
        "ringdown_time_seconds": ringdown,
        "plunge_ringdown_time_seconds": plunge_ringdown,
        "support_time_seconds": support,
        "start_frequency_hz": start,
        "spin_bound": spin_bound,
    }


def capture(start_frequency: float) -> tuple[np.ndarray, np.ndarray, float]:
    options = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(options, 0)
    mode_array = lalsim.SimInspiralCreateModeArray()
    lalsim.SimInspiralModeArrayActivateMode(mode_array, 2, 2)
    lalsim.SimInspiralWaveformParamsInsertModeArray(options, mode_array)
    hp, hc = lalsim.SimIMRTEOBResumS(
        0.0, DT, M1 * lal.MSUN_SI, M2 * lal.MSUN_SI, *SPINS,
        0.0, 0.0, DISTANCE_MPC * 1.0e6 * lal.PC_SI,
        INCLINATION, PHI_REF, options, 0.0, 0.0, start_frequency, 0.0,
    )
    # SimIMRTEOBResumS returns the direct EOB frame; the generic cache uses
    # the reviewed paired minus sign, with no phase/time alignment.
    return (-np.asarray(hp.data.data, dtype=np.float64),
            -np.asarray(hc.data.data, dtype=np.float64), float(hp.epoch))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("public-td-reproduced.npz"))
    args = parser.parse_args()
    bounds = public_support_bounds()
    hp, hc, epoch = capture(bounds["start_frequency_hz"])
    np.savez_compressed(
        args.output, exact_start_raw_h_plus=hp, exact_start_raw_h_cross=hc,
        epoch_seconds=np.asarray(epoch), sampling_dt_seconds=np.asarray(DT),
        support_start_frequency=np.asarray(bounds["start_frequency_hz"]),
    )
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "samples": int(hp.size), "epoch_seconds": epoch, "dt_seconds": DT, "support": bounds}, sort_keys=True))


if __name__ == "__main__":
    main()
