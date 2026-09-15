"""Bounded black-box checks for the opt-in native MSA v223 composition.

The clean implementation is evaluated on the fixed 81-point probes from the
v223 contract. Raw positive-frequency hp/hc errors are also reported against
public IMRPhenomXP for the frozen XP catalog case, its holdout, and three fresh
systems. No phase, time, amplitude, or residual parameters are fitted.
"""
from __future__ import annotations

import json
import math

import lal
import lalsimulation as lalsim
import torch

from torchwave.approximants.precession_msa_angles_clean import (
    assemble_precession_msa_angles,
)
from torchwave.approximants.precession_msa_waveform_clean import waveform_v223


VERSION = 223
EXPANSION_ORDER = 5
MPRIME = 2
CONVENTION = 0
FINAL_SPIN_MOD = 3
FLOW = 20.0
FHIGH = 200.0
DISTANCE = 500.0
INCLINATION = 0.7


def _frozen_xp_case(case_id: str):
    """Read the existing frozen XP catalog/holdout record."""
    from parity_harness import get_canonical_catalog_cases, get_canonical_holdout_cases

    records = get_canonical_catalog_cases() + get_canonical_holdout_cases()
    for record in records:
        if record.get("id") == case_id:
            p = record["params"]
            return {
                "name": case_id,
                "mass1": float(p["mass1"]),
                "mass2": float(p["mass2"]),
                "spin1": (float(p["spin1x"]), float(p["spin1y"]), float(p["spin1z"])),
                "spin2": (float(p["spin2x"]), float(p["spin2y"]), float(p["spin2z"])),
                "distance": float(p.get("distance", DISTANCE)),
                "inclination": float(p.get("inclination", INCLINATION)),
                "phic": float(p.get("phic", 0.0)),
                "tc": float(p.get("tc", 0.0)),
                "f_ref": float(p.get("f_ref", FLOW)),
            }
    raise KeyError(f"frozen XP case not found: {case_id}")


CASES = (
    _frozen_xp_case("IMRPhenomXP_diagnostic"),
    _frozen_xp_case("IMRPhenomXP_holdout"),
    {
        "name": "fresh_anchor_A",
        "mass1": 35.0,
        "mass2": 25.0,
        "spin1": (0.2, 0.1, 0.2),
        "spin2": (-0.1, 0.15, -0.1),
        "distance": DISTANCE,
        "inclination": INCLINATION,
        "phic": 0.0,
        "tc": 0.0,
        "f_ref": 20.0,
    },
    {
        "name": "fresh_anchor_B",
        "mass1": 45.0,
        "mass2": 15.0,
        "spin1": (0.45, -0.2, 0.3),
        "spin2": (0.1, 0.2, -0.25),
        "distance": DISTANCE,
        "inclination": INCLINATION,
        "phic": 0.0,
        "tc": 0.0,
        "f_ref": 20.0,
    },
    {
        "name": "fresh_anchor_C",
        "mass1": 31.0,
        "mass2": 17.0,
        "spin1": (-0.13, 0.27, 0.21),
        "spin2": (0.19, -0.08, -0.16),
        "distance": DISTANCE,
        "inclination": INCLINATION,
        "phic": 0.0,
        "tc": 0.0,
        "f_ref": 27.0,
    },
)


def _options():
    options = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(options, VERSION)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPExpansionOrder(
        options, EXPANSION_ORDER
    )
    lalsim.SimInspiralWaveformParamsInsertPhenomXPConvention(options, CONVENTION)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(
        options, FINAL_SPIN_MOD
    )
    return options


def _native_frequencies(case):
    return torch.linspace(case["f_ref"], FHIGH, 81, dtype=torch.float64)


def _public_frequencies():
    return torch.arange(FLOW, FHIGH + 1.0, dtype=torch.float64)


def _native_angles(case, frequencies):
    with torch.no_grad():
        return assemble_precession_msa_angles(
            frequencies,
            case["mass1"],
            case["mass2"],
            torch.tensor(case["spin1"], dtype=torch.float64),
            torch.tensor(case["spin2"], dtype=torch.float64),
            f_ref=case["f_ref"],
            order=EXPANSION_ORDER,
            compatibility="v223",
        )


def _native_waveform(case, frequencies):
    s1, s2 = case["spin1"], case["spin2"]
    with torch.no_grad():
        return waveform_v223(
            frequencies,
            case["mass1"],
            case["mass2"],
            s1[0], s1[1], s1[2],
            s2[0], s2[1], s2[2],
            distance=case["distance"],
            inclination=case["inclination"],
            phic=case["phic"],
            tc=case["tc"],
            f_ref=case["f_ref"],
            convention=CONVENTION,
            order=EXPANSION_ORDER,
        )


def _public_angles(case, frequencies):
    vector = lal.CreateREAL8Vector(frequencies.numel())
    vector.data[:] = frequencies.tolist()
    s1, s2 = case["spin1"], case["spin2"]
    values = lalsim.SimIMRPhenomXPMSAAngles(
        vector,
        case["mass1"] * lal.MSUN_SI,
        case["mass2"] * lal.MSUN_SI,
        *s1,
        *s2,
        case["inclination"],
        case["f_ref"],
        MPRIME,
        _options(),
    )
    return tuple(
        torch.as_tensor(list(value.data), dtype=torch.float64) for value in values
    )


def _public_waveform(case):
    s1, s2 = case["spin1"], case["spin2"]
    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        case["mass1"] * lal.MSUN_SI,
        case["mass2"] * lal.MSUN_SI,
        *s1,
        *s2,
        case["distance"] * 1.0e6 * lal.PC_SI,
        case["inclination"],
        case["phic"],
        case["tc"],
        0.0,
        0.0,
        1.0,
        FLOW,
        FHIGH,
        case["f_ref"],
        _options(),
        lalsim.IMRPhenomXP,
    )
    start, stop = int(FLOW), int(FHIGH) + 1
    return (
        torch.as_tensor(hp.data.data[start:stop].copy(), dtype=torch.complex128),
        torch.as_tensor(hc.data.data[start:stop].copy(), dtype=torch.complex128),
    )


def _summary(values):
    values = tuple(torch.as_tensor(value) for value in values)
    return {
        "finite": all(bool(torch.isfinite(value).all()) for value in values),
        "max_abs": [float(value.abs().max()) for value in values],
    }


def _raw_metrics(actual, target):
    rows = {}
    for name, value, reference in zip(("hp", "hc"), actual, target):
        difference = value - reference
        norm = torch.linalg.vector_norm(reference)
        rows[name] = {
            "difference_l2": float(torch.linalg.vector_norm(difference)),
            "target_l2": float(norm),
            "relative_l2": float(torch.linalg.vector_norm(difference) / norm)
            if float(norm) else math.inf,
            "max_abs": float(difference.abs().max()),
        }
    return rows


def _one(case):
    native_f = _native_frequencies(case)
    public_f = _public_frequencies()
    angles = _native_angles(case, native_f)
    native_hp, native_hc = _native_waveform(case, public_f)
    public_hp, public_hc = _public_waveform(case)
    public_angles = _public_angles(case, native_f)
    return {
        "name": case["name"],
        "parameters": case,
        "native_grid_hz": native_f.tolist(),
        "native_angles": _summary(
            (angles["alpha"], angles["gamma"], angles["cos_beta"])
        ),
        "public_angles": _summary(public_angles),
        "native_waveform": _summary((native_hp, native_hc)),
        "public_raw_waveform": _summary((public_hp, public_hc)),
        "raw_hp_hc_metrics": _raw_metrics(
            (native_hp, native_hc), (public_hp, public_hc)
        ),
    }


def main():
    rows = []
    for case in CASES:
        print(f"running {case['name']}", flush=True)
        rows.append(_one(case))
    report = {
        "status": "complete",
        "contract": {
            "prec_version": VERSION,
            "expansion_order": EXPANSION_ORDER,
            "mprime": MPRIME,
            "convention": CONVENTION,
            "final_spin_mod": FINAL_SPIN_MOD,
            "native_frequency_count": 81,
            "native_frequency_range_hz": [FLOW, FHIGH],
            "public_waveform_grid_hz": [FLOW, FHIGH],
        },
        "raw_comparison_definition": (
            "native waveform_v223 and public IMRPhenomXP positive-frequency "
            "hp/hc on the common integer-Hz band; no phase/time/norm optimization"
        ),
        "rows": rows,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

