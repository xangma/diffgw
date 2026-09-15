"""Validate the clean Pv2 numerical conventions and public support semantics.

The clean waveform evaluates every requested positive frequency.  The public
``ChooseFDWaveform`` wrapper has a half-open ``[f_min, f_max)`` Pv2 support,
whereas ``SimIMRPhenomPFrequencySequence`` can return a value at an explicitly
requested final sample.  This diagnostic conditions the uniform comparison
only after checking those grid assumptions and reports the raw endpoint
separately.  No phase or amplitude alignment is applied.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from diagnose_phenompv2_waveform_clean import CASES
from torchwave.approximants.phenompv2_waveform_clean import waveform
from torchwave.waveforms import get_fd_waveform


DF = 0.25
FLOW = 20.0
FMAX = 512.0
DISTANCE_MPC = 500.0
DETECTOR_FACTORS = (0.7, -0.2)


def _metrics(test: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    delta = np.asarray(test) - np.asarray(reference)
    scale = max(float(np.max(np.abs(reference))), 1.0e-300)
    return {
        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1.0e-300)),
        "max_normalized_error": float(np.max(np.abs(delta)) / scale),
        "max_absolute_error": float(np.max(np.abs(delta))),
    }


def _uniform_contract(
    frequencies: Iterable[float], *, df: float, requested_fmax: float
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Return the verified half-open support mask for a uniform standard grid.

    A standard adapter may append power-of-two padding above the requested
    edge.  Those samples remain part of the validated grid and are excluded
    by the half-open mask; a nonuniform grid still uses sequence semantics.
    """
    f = np.asarray(frequencies, dtype=np.float64)
    if f.ndim != 1 or f.size < 2 or not np.isfinite(f).all():
        raise ValueError("uniform support requires a finite one-dimensional grid")
    if not np.isfinite(requested_fmax):
        raise ValueError("uniform support requires a finite requested fmax")
    diffs = np.diff(f)
    atol = 1.0e-12
    rtol = 1.0e-10
    uniform = bool(np.all(diffs > 0.0) and np.allclose(diffs, df, atol=atol, rtol=rtol))
    if not uniform:
        raise ValueError("uniform support requires a strictly increasing grid with the declared df")
    endpoint_tol = atol + rtol * max(abs(requested_fmax), 1.0)
    covers_edge = bool(float(f[-1]) + endpoint_tol >= requested_fmax)
    starts_before_edge = bool(requested_fmax + endpoint_tol >= float(f[0]))
    if not covers_edge or not starts_before_edge:
        raise ValueError(
            "uniform standard support requires a grid covering fmax from its first sample"
        )
    mask = f < requested_fmax
    return mask, {
        "semantics": "uniform_standard_half_open",
        "df_hz": float(df),
        "requested_fmax_hz": float(requested_fmax),
        "terminal_sample_hz": float(f[-1]),
        "padded_samples": int(np.count_nonzero(f >= requested_fmax)),
        "endpoint_excluded": True,
        "supported_samples": int(mask.sum()),
    }


def _sequence_contract(frequencies: Iterable[float]) -> Tuple[np.ndarray, Dict[str, object]]:
    """Return the direct sequence support mask, retaining each requested point."""
    f = np.asarray(frequencies, dtype=np.float64)
    if f.ndim != 1 or not np.isfinite(f).all() or np.any(f <= 0.0):
        raise ValueError("sequence support requires finite positive frequencies")
    return np.ones(f.shape, dtype=bool), {
        "semantics": "arbitrary_frequency_sequence",
        "requested_samples": int(f.size),
        "terminal_sample_hz": float(f[-1]),
        "endpoint_retained": True,
    }


def _clean(case: dict, frequencies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    kwargs = {k: v for k, v in case.items() if k not in ("label", "f_ref")}
    hp, hc = waveform(
        torch.as_tensor(frequencies, dtype=torch.float64),
        distance=DISTANCE_MPC,
        tc=0.0,
        f_ref=case.get("f_ref", FLOW),
        **kwargs,
    )
    return hp.detach().cpu().numpy(), hc.detach().cpu().numpy()


def _routed(
    case: dict, frequencies: np.ndarray, *, support: str, support_edge: float = FMAX
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the same clean driver through the public high-level route."""
    kwargs = {k: v for k, v in case.items() if k not in ("label", "f_ref")}
    hp, hc = get_fd_waveform(
        "IMRPhenomPv2",
        sample_frequencies=torch.as_tensor(frequencies, dtype=torch.float64),
        f_lower=FLOW,
        f_final=support_edge,
        frequency_support=support,
        distance=DISTANCE_MPC,
        tc=0.0,
        f_ref=case.get("f_ref", FLOW),
        dtype=torch.float64,
        **kwargs,
    )
    return hp.detach().cpu().numpy(), hc.detach().cpu().numpy()


def _public_choose(case: dict):
    import lal
    import lalsimulation as ls

    hp, hc = ls.SimInspiralChooseFDWaveform(
        case["mass1"] * lal.MSUN_SI,
        case["mass2"] * lal.MSUN_SI,
        case["spin1x"],
        case["spin1y"],
        case["spin1z"],
        case["spin2x"],
        case["spin2y"],
        case["spin2z"],
        DISTANCE_MPC * 1.0e6 * lal.PC_SI,
        case["inclination"],
        case["phic"],
        0.0,
        0.0,
        0.0,
        DF,
        FLOW,
        FMAX,
        case.get("f_ref", FLOW),
        lal.CreateDict(),
        ls.IMRPhenomPv2,
    )
    return np.asarray(hp.data.data).copy(), np.asarray(hc.data.data).copy()


def _public_sequence(case: dict, frequencies: np.ndarray):
    import lal
    import lalsimulation as ls

    ref = case.get("f_ref", FLOW)
    g = ls.SimIMRPhenomPCalculateModelParametersFromSourceFrame(
        case["mass1"] * lal.MSUN_SI,
        case["mass2"] * lal.MSUN_SI,
        ref,
        case["phic"],
        case["inclination"],
        case["spin1x"],
        case["spin1y"],
        case["spin1z"],
        case["spin2x"],
        case["spin2y"],
        case["spin2z"],
        ls.IMRPhenomPv2_V,
    )
    hp, hc = ls.SimIMRPhenomPFrequencySequence(
        np.asarray(frequencies, dtype=np.float64),
        *g[:4],
        case["mass1"] * lal.MSUN_SI,
        case["mass2"] * lal.MSUN_SI,
        DISTANCE_MPC * 1.0e6 * lal.PC_SI,
        g[4],
        g[5],
        ref,
        ls.IMRPhenomPv2_V,
        ls.NoNRT_V,
        lal.CreateDict(),
    )
    return np.asarray(hp.data.data).copy(), np.asarray(hc.data.data).copy()


def _one_case(case: dict, choose_available: bool) -> dict:
    uniform = np.arange(FLOW, FMAX + 0.5 * DF, DF, dtype=np.float64)
    sequence = np.asarray((FLOW, FLOW + DF, FMAX - DF, FMAX), dtype=np.float64)
    seq_indices = np.asarray((0, 1, uniform.size - 2, uniform.size - 1))
    clean_uniform_hp, clean_uniform_hc = _clean(case, uniform)
    clean_sequence_hp, clean_sequence_hc = _clean(case, sequence)
    routed_uniform_hp, routed_uniform_hc = _routed(
        case, uniform, support="uniform"
    )
    routed_sequence_hp, routed_sequence_hc = _routed(
        case, sequence, support="sequence"
    )
    sequence_mask, sequence_contract = _sequence_contract(sequence)
    uniform_mask, uniform_contract = _uniform_contract(
        uniform, df=DF, requested_fmax=FMAX
    )

    # Exercise the padded-grid branch with an edge that is neither a power of
    # two nor the terminal sample.  The public route must zero the edge and
    # all padding while retaining the interior waveform values.
    padded = np.arange(FLOW, 64.0001, DF, dtype=np.float64)
    padded_fmax = 47.5
    padded_mask, padded_contract = _uniform_contract(
        padded, df=DF, requested_fmax=padded_fmax
    )
    routed_padded_hp, routed_padded_hc = _routed(
        case, padded, support="uniform", support_edge=padded_fmax
    )
    clean_padded_hp, clean_padded_hc = _clean(case, padded)

    clean_sequence_control = {
        "hp": _metrics(clean_sequence_hp[sequence_mask], clean_uniform_hp[seq_indices]),
        "hc": _metrics(clean_sequence_hc[sequence_mask], clean_uniform_hc[seq_indices]),
        "last_requested_frequency_hz": float(sequence[-1]),
        "last_requested_hp_abs": float(abs(clean_sequence_hp[-1])),
        "last_requested_hc_abs": float(abs(clean_sequence_hc[-1])),
    }
    row = {
        "label": case["label"],
        "parameters": case,
        "clean_support_contract": {
            "uniform": uniform_contract,
            "sequence": sequence_contract,
            "sequence_vs_uniform": clean_sequence_control,
        },
        "routed_api": {
            "uniform": {
                "support_contract": uniform_contract,
                "hp_vs_clean_interior": _metrics(
                    routed_uniform_hp[uniform_mask], clean_uniform_hp[uniform_mask]
                ),
                "hc_vs_clean_interior": _metrics(
                    routed_uniform_hc[uniform_mask], clean_uniform_hc[uniform_mask]
                ),
                "endpoint_hz": FMAX,
                "endpoint_hp_abs": float(abs(routed_uniform_hp[-1])),
                "endpoint_hc_abs": float(abs(routed_uniform_hc[-1])),
                "endpoint_excluded": bool(
                    routed_uniform_hp[-1] == 0 and routed_uniform_hc[-1] == 0
                ),
            },
            "padded_uniform": {
                "support_contract": padded_contract,
                "hp_vs_clean_interior": _metrics(
                    routed_padded_hp[padded_mask], clean_padded_hp[padded_mask]
                ),
                "hc_vs_clean_interior": _metrics(
                    routed_padded_hc[padded_mask], clean_padded_hc[padded_mask]
                ),
                "edge_hz": padded_fmax,
                "terminal_hz": float(padded[-1]),
                "edge_excluded": bool(
                    routed_padded_hp[np.searchsorted(padded, padded_fmax)] == 0
                    and routed_padded_hc[np.searchsorted(padded, padded_fmax)] == 0
                ),
                "padding_zero": bool(
                    np.all(routed_padded_hp[~padded_mask] == 0)
                    and np.all(routed_padded_hc[~padded_mask] == 0)
                ),
            },
            "sequence": {
                "support_contract": sequence_contract,
                "hp_vs_clean": _metrics(routed_sequence_hp, clean_sequence_hp),
                "hc_vs_clean": _metrics(routed_sequence_hc, clean_sequence_hc),
                "endpoint_hz": FMAX,
                "endpoint_hp_abs": float(abs(routed_sequence_hp[-1])),
                "endpoint_hc_abs": float(abs(routed_sequence_hc[-1])),
                "endpoint_retained": bool(
                    abs(routed_sequence_hp[-1]) > 0.0
                    or abs(routed_sequence_hc[-1]) > 0.0
                ),
            },
        },
    }
    if not choose_available:
        row["public"] = {"status": "unavailable"}
        return row

    choose_hp, choose_hc = _public_choose(case)
    # ChooseFD returns the native deltaF grid from DC through the requested
    # upper endpoint.  The clean outputs above start at FLOW, so condition the
    # public series to that same physical band before computing raw metrics.
    first = int(round(FLOW / DF))
    last = int(round(FMAX / DF)) + 1
    choose_hp = choose_hp[first:last]
    choose_hc = choose_hc[first:last]
    detector_clean = DETECTOR_FACTORS[0] * clean_uniform_hp + DETECTOR_FACTORS[1] * clean_uniform_hc
    detector_routed = (
        DETECTOR_FACTORS[0] * routed_uniform_hp
        + DETECTOR_FACTORS[1] * routed_uniform_hc
    )
    detector_choose = DETECTOR_FACTORS[0] * choose_hp + DETECTOR_FACTORS[1] * choose_hc
    raw = {
        "hp": {
            "full": _metrics(clean_uniform_hp[uniform >= FLOW], choose_hp[uniform >= FLOW]),
            "interior": _metrics(clean_uniform_hp[uniform_mask], choose_hp[uniform_mask]),
        },
        "hc": {
            "full": _metrics(clean_uniform_hc[uniform >= FLOW], choose_hc[uniform >= FLOW]),
            "interior": _metrics(clean_uniform_hc[uniform_mask], choose_hc[uniform_mask]),
        },
        "detector": {
            "full": _metrics(detector_clean[uniform >= FLOW], detector_choose[uniform >= FLOW]),
            "interior": _metrics(detector_clean[uniform_mask], detector_choose[uniform_mask]),
        },
    }
    routed_raw = {
        "hp": {
            "full": _metrics(routed_uniform_hp, choose_hp),
            "interior": _metrics(routed_uniform_hp[uniform_mask], choose_hp[uniform_mask]),
        },
        "hc": {
            "full": _metrics(routed_uniform_hc, choose_hc),
            "interior": _metrics(routed_uniform_hc[uniform_mask], choose_hc[uniform_mask]),
        },
        "detector": {
            "full": _metrics(detector_routed, detector_choose),
            "interior": _metrics(detector_routed[uniform_mask], detector_choose[uniform_mask]),
        },
    }
    sequence_hp, sequence_hc = _public_sequence(case, sequence)
    row["public"] = {
        "choose_fd": {
            "support_contract": uniform_contract,
            "raw": raw,
            "routed_raw": routed_raw,
            "endpoint_hz": FMAX,
            "endpoint_hp_abs": float(abs(choose_hp[-1])),
            "endpoint_hc_abs": float(abs(choose_hc[-1])),
        },
        "frequency_sequence": {
            "support_contract": sequence_contract,
            "endpoint_hz": FMAX,
            "endpoint_hp_abs": float(abs(sequence_hp[-1])),
            "endpoint_hc_abs": float(abs(sequence_hc[-1])),
            "endpoint_retained": bool(abs(sequence_hp[-1]) > 0.0 or abs(sequence_hc[-1]) > 0.0),
        },
    }
    return row


def main() -> None:
    outdir = Path("verification/pv2-numerical-convention")
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        import lal  # noqa: F401
        import lalsimulation as ls
        versions = {"lal": getattr(lal, "__version__", "unknown"), "lalsimulation": getattr(ls, "__version__", "unknown")}
        choose_available = True
        unavailable_reason = None
    except Exception as exc:
        versions = {}
        choose_available = False
        unavailable_reason = str(exc)

    rows = [_one_case(case, choose_available) for case in CASES]
    report = {
        "status": "ok" if choose_available else "public_oracle_unavailable",
        "provenance": "MIT clean-room D carrier + independent cyclic six-significant-digit QNM sampling; raw hp/hc/detector metrics; no fitted phase removal",
        "convention": {
            "phase_samples": 10,
            "phase_interval": "0.8*fRD through min(1.2*fRD, 0.2/Msec)",
            "frequency_units": "Hz",
            "qnm_units": "cyclic Mf and Mf_damp",
            "uniform_standard_support": "strictly increasing uniform grid covering requested fmax, half-open f < fmax; padded samples above the edge are zeroed",
            "sequence_support": "retain every explicitly requested finite positive frequency",
        },
        "versions": versions,
        "unavailable_reason": unavailable_reason,
        "cases": rows,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    path = outdir / "diagnostic.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
