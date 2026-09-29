#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare the two documented XHM releases through public binary APIs only.

This is an external-oracle diagnostic.  No output is used to fit or replace a
production coefficient.  In particular, ``Phase`` for the spherical (3,2)
mode includes mixing and need not be unchanged when its spheroidal phase is.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import lal
import lalsimulation as ls
import numpy as np


RELEASES = (122019, 122022)
MODES = ((2, -1), (3, -2), (3, -3), (4, -4))
CASES = (
    ("catalog", 40.0, 20.0, 0.3, -0.1, 500.0, 0.5, 20.0),
    ("holdout", 55.0, 11.0, -0.4, 0.2, 680.0, 0.71, 27.0),
)


def _evaluate(name: str, frequency, mode, case, release: int) -> np.ndarray:
    _, m1, m2, chi1, chi2, distance, phi_ref, f_ref = case
    params = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion(params, release)
    if "Spheroidal" in name:
        args = (frequency, *mode, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
                chi1, chi2, distance * 1.0e6 * lal.PC_SI, phi_ref, f_ref, params)
    else:
        args = (frequency, *mode, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
                0.0, 0.0, chi1, 0.0, 0.0, chi2,
                distance * 1.0e6 * lal.PC_SI, phi_ref, f_ref, params)
    return np.asarray(getattr(ls, name)(*args).data, dtype=np.float64).copy()


def _metrics(older: np.ndarray, current: np.ndarray, *, phase: bool) -> dict[str, float | bool]:
    delta = older - current
    if phase:
        # A release-dependent phase origin is a gauge, while its frequency
        # dependence changes the shape of the spherical mode.
        gauge_removed = np.angle(np.exp(1j * (delta - delta[0])))
        return {
            "bitwise_equal": bool(np.array_equal(older, current)),
            "max_abs_difference_rad": float(np.max(np.abs(delta))),
            "rms_wrapped_shape_after_reference_phase_rad": float(np.sqrt(np.mean(gauge_removed**2))),
        }
    return {
        "bitwise_equal": bool(np.array_equal(older, current)),
        "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(current)),
    }


def diagnose() -> dict[str, object]:
    frequency = lal.CreateREAL8Vector(201)
    frequency.data[:] = np.linspace(20.0, 500.0, 201)
    rows: dict[str, object] = {}
    for case in CASES:
        mode_rows: dict[str, object] = {}
        for mode in MODES:
            pieces: dict[str, object] = {}
            observables = [
                ("spherical_amplitude", "SimIMRPhenomXHMAmplitude", False),
                ("spherical_phase", "SimIMRPhenomXHMPhase", True),
            ]
            if mode == (3, -2):
                observables.extend((
                    ("spheroidal_amplitude", "SimIMRPhenomXHM_SpheroidalAmplitude", False),
                    ("spheroidal_phase", "SimIMRPhenomXHM_SpheroidalPhase", True),
                ))
            for key, api, phase in observables:
                old, new = (_evaluate(api, frequency, mode, case, release) for release in RELEASES)
                pieces[key] = _metrics(old, new, phase=phase)
            mode_rows[f"{mode[0]}{abs(mode[1])}"] = pieces
        rows[case[0]] = mode_rows
    return {
        "schema": "diffgw.xhm.release-boundary-diagnostic/v1",
        "evidence_category": "external-binary-numerical-oracle",
        "production_fit_policy": "release outputs are diagnostic only; no coefficient is fitted or transcribed",
        "release_versions": list(RELEASES),
        "frequency_grid_hz": {"start": 20.0, "stop": 500.0, "count": 201},
        "case_parameters": {
            case_id: {"mass1_msun": m1, "mass2_msun": m2, "chi1": chi1, "chi2": chi2,
                      "distance_mpc": distance, "phi_ref_rad": phi_ref, "f_ref_hz": f_ref}
            for case_id, m1, m2, chi1, chi2, distance, phi_ref, f_ref in CASES
        },
        "versions": {"lalsuite": importlib.metadata.version("lalsuite"),
                     "lalsimulation": ls.__version__, "numpy": np.__version__},
        "cases": rows,
        "qualification_implication": (
            "Unchanged 32 spheroidal phase does not imply unchanged spherical phase: "
            "the recalibrated spheroidal amplitude enters the published mixing transform. "
            "This binary comparison provides no release-122022 fit surfaces and does not relax frozen gates."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(diagnose(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
