"""Clean-room diagnosis of the XP v223 exact-aligned chart boundary.

This diagnostic uses only public LALSimulation callables and the local clean
MSA/source-frame helpers.  The production endpoint correction lives in
``precession_msa_waveform_clean.py``; this diagnostic is read-only with
respect to runtime/helper physics and does not change that helper or the
production adapter.  The
three waveform constructions reported for each fixture are:

* the native clean ``waveform_v223`` composition;
* the same clean projector with the public source-frame/MSA angle chart
  injected; and
* the public ``IMRPhenomXP`` waveform, used as a black-box target.

The public chart injection is an equation-level decomposition.  It identifies
whether a boundary residual comes from the source-frame chart, the finite
opening angle, the remnant scalar, or the opaque XAS carrier phase.  No phase,
time, norm, coefficient, or threshold adjustment is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import lal
import lalsimulation as ls
import torch

from torchwave.approximants.imrphenom_xas import IMRPhenomXAS
from torchwave.approximants.precession_msa_angles_clean import (
    assemble_precession_msa_angles,
)
from torchwave.approximants.precession_msa_waveform_clean import (
    _native_msa_final_spin,
    compose_msa_waveform,
    transport_msa_angles_for_phi_ref,
    waveform_v223,
)
from torchwave.approximants.precession_source_frame_clean import (
    source_frame_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "docs/reviews/2026-09-13-parity-review/verification/xp-aligned-boundary-clean"
FREQUENCIES = torch.arange(20.0, 512.0, 0.25, dtype=torch.float64)
FLOW = 20.0
FHIGH = 512.0
DELTA_F = 0.25
F_REF = 20.0
DISTANCE = 500.0
INCLINATION = 0.7
PHIC = 0.3
EXPANSION_ORDER = 5
MPRIME = 2
STRICT_GATE = 1.0e-6
SAMPLE_INDICES = (0, 680, 1967)

# The first three are the requested audited probes.  The last two are fresh
# fixtures so the boundary contract is not tied to one mass ratio or one
# aligned-spin pair.  Every spin is represented by all three components.
FIXTURES = (
    {
        "id": "audited_aligned_35_25",
        "label": "audited prior exact-aligned probe",
        "mass1_msun": 35.0,
        "mass2_msun": 25.0,
        "spin1": (0.0, 0.0, 0.2),
        "spin2": (0.0, 0.0, -0.1),
    },
    {
        "id": "audited_zero_35_25",
        "label": "audited prior zero-spin probe",
        "mass1_msun": 35.0,
        "mass2_msun": 25.0,
        "spin1": (0.0, 0.0, 0.0),
        "spin2": (0.0, 0.0, 0.0),
    },
    {
        "id": "audited_tiny_neighbor_35_25",
        "label": "audited tiny nonzero transverse neighbor",
        "mass1_msun": 35.0,
        "mass2_msun": 25.0,
        "spin1": (1.0e-5, 0.0, 0.2),
        "spin2": (0.0, 1.0e-5, -0.1),
    },
    {
        "id": "fresh_aligned_45_15",
        "label": "fresh exact-aligned mass-ratio fixture",
        "mass1_msun": 45.0,
        "mass2_msun": 15.0,
        "spin1": (0.0, 0.0, 0.35),
        "spin2": (0.0, 0.0, -0.2),
    },
    {
        "id": "fresh_zero_equal_30_30",
        "label": "fresh equal-mass zero-spin fixture",
        "mass1_msun": 30.0,
        "mass2_msun": 30.0,
        "spin1": (0.0, 0.0, 0.0),
        "spin2": (0.0, 0.0, 0.0),
    },
)

CLEAN_FILES = (
    "precession_msa_waveform_clean.py",
    "precession_msa_angles_clean.py",
    "precession_source_frame_clean.py",
)


def _scalar(value) -> float:
    return float(torch.as_tensor(value).detach().cpu().reshape(-1)[0])


def _wrap(value: float) -> float:
    """Return a principal angle representative in ``[-pi, pi)``."""
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _sha256(path: Path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _helper_hashes():
    base = ROOT / "src/torchwave/approximants"
    return {name: _sha256(base / name) for name in CLEAN_FILES}


def _options(convention: int, final_spin_mod: int = 3):
    params = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(params, 223)
    ls.SimInspiralWaveformParamsInsertPhenomXPExpansionOrder(
        params, EXPANSION_ORDER
    )
    ls.SimInspiralWaveformParamsInsertPhenomXPConvention(
        params, int(convention)
    )
    ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(
        params, int(final_spin_mod)
    )
    return params


def _option_record(convention: int):
    params = _options(convention, final_spin_mod=3)
    return {
        "requested": {
            "prec_version": 223,
            "expansion_order": EXPANSION_ORDER,
            "convention": int(convention),
            "final_spin_mod": 3,
        },
        "looked_up": {
            "prec_version": int(ls.SimInspiralWaveformParamsLookupPhenomXPrecVersion(params)),
            "expansion_order": int(ls.SimInspiralWaveformParamsLookupPhenomXPExpansionOrder(params)),
            "convention": int(ls.SimInspiralWaveformParamsLookupPhenomXPConvention(params)),
            "final_spin_mod": int(ls.SimInspiralWaveformParamsLookupPhenomXPFinalSpinMod(params)),
        },
    }


def _physical(case):
    return {
        "mass1_si": float(case["mass1_msun"]) * lal.MSUN_SI,
        "mass2_si": float(case["mass2_msun"]) * lal.MSUN_SI,
        "spin1": tuple(float(value) for value in case["spin1"]),
        "spin2": tuple(float(value) for value in case["spin2"]),
    }


def _public_source(case, phi_ref: float, convention: int):
    physical = _physical(case)
    output = ls.SimIMRPhenomXPCalculateModelParametersFromSourceFrame(
        physical["mass1_si"],
        physical["mass2_si"],
        F_REF,
        float(phi_ref),
        INCLINATION,
        *physical["spin1"],
        *physical["spin2"],
        _options(convention),
    )
    values = tuple(float(value) for value in output)
    return {
        "tuple": list(values),
        "semantics": [
            "chi1", "chi2", "chip", "thetaJN", "alpha0", "phiJL0", "zeta"
        ],
        "phi_ref": float(phi_ref),
        "options": _option_record(convention),
    }


def _public_angles(case, convention: int):
    physical = _physical(case)
    vector = lal.CreateREAL8Vector(FREQUENCIES.numel())
    vector.data[:] = FREQUENCIES.tolist()
    output = ls.SimIMRPhenomXPMSAAngles(
        vector,
        physical["mass1_si"],
        physical["mass2_si"],
        *physical["spin1"],
        *physical["spin2"],
        INCLINATION,
        F_REF,
        MPRIME,
        _options(convention),
    )
    return {
        "alpha": torch.as_tensor(list(output[0].data), dtype=torch.float64),
        "gamma": torch.as_tensor(list(output[1].data), dtype=torch.float64),
        "cos_beta": torch.as_tensor(list(output[2].data), dtype=torch.float64),
    }


def _public_waveform(case, convention: int):
    physical = _physical(case)
    hp, hc = ls.SimInspiralChooseFDWaveform(
        physical["mass1_si"],
        physical["mass2_si"],
        *physical["spin1"],
        *physical["spin2"],
        DISTANCE * 1.0e6 * lal.PC_SI,
        INCLINATION,
        PHIC,
        0.0,
        0.0,
        0.0,
        DELTA_F,
        FLOW,
        FHIGH,
        F_REF,
        _options(convention),
        ls.IMRPhenomXP,
    )
    # The public uniform constructor begins at 0 Hz.  [80:2048] is exactly
    # 20 <= f < 512 on the authoritative quarter-Hz diagnostic grid.
    return (
        torch.as_tensor(hp.data.data[80:2048].copy(), dtype=torch.complex128),
        torch.as_tensor(hc.data.data[80:2048].copy(), dtype=torch.complex128),
    )


def _public_xas(case, phi_c: float):
    physical = _physical(case)
    hp, hc = ls.SimInspiralChooseFDWaveform(
        physical["mass1_si"],
        physical["mass2_si"],
        # XAS is the non-precessing carrier.  The clean composition passes
        # only the aligned z components to this opaque call; retain that
        # contract for the public carrier even for the tiny-spin fixture.
        0.0,
        0.0,
        physical["spin1"][2],
        0.0,
        0.0,
        physical["spin2"][2],
        DISTANCE * 1.0e6 * lal.PC_SI,
        0.0,
        float(phi_c),
        0.0,
        0.0,
        0.0,
        DELTA_F,
        FLOW,
        FHIGH,
        F_REF,
        lal.CreateDict(),
        ls.IMRPhenomXAS,
    )
    return (
        torch.as_tensor(hp.data.data[80:2048].copy(), dtype=torch.complex128),
        torch.as_tensor(hc.data.data[80:2048].copy(), dtype=torch.complex128),
    )


def _clean_geometry(case, phi_ref: float):
    s1 = torch.as_tensor(case["spin1"], dtype=torch.float64)
    s2 = torch.as_tensor(case["spin2"], dtype=torch.float64)
    return source_frame_geometry(
        torch.as_tensor(case["mass1_msun"], dtype=torch.float64),
        torch.as_tensor(case["mass2_msun"], dtype=torch.float64),
        s1[0], s1[1], s1[2], s2[0], s2[1], s2[2],
        torch.as_tensor(INCLINATION, dtype=torch.float64),
        torch.as_tensor(phi_ref, dtype=torch.float64),
        torch.as_tensor(F_REF, dtype=torch.float64),
        compatibility="v223",
    )


def _clean_angles(case):
    return assemble_precession_msa_angles(
        FREQUENCIES,
        torch.as_tensor(case["mass1_msun"], dtype=torch.float64),
        torch.as_tensor(case["mass2_msun"], dtype=torch.float64),
        torch.as_tensor(case["spin1"], dtype=torch.float64),
        torch.as_tensor(case["spin2"], dtype=torch.float64),
        f_ref=torch.as_tensor(F_REF, dtype=torch.float64),
        order=EXPANSION_ORDER,
        beta_pn=False,
        average_mode="leading",
        compatibility="v223",
    )


def _aligned(case) -> bool:
    s1 = case["spin1"]
    s2 = case["spin2"]
    transverse_sq = sum(float(value) ** 2 for value in s1[:2] + s2[:2])
    # Exact equality is intentional.  The tiny-neighbor fixture must remain
    # on the generic route and visible in the report.
    return transverse_sq == 0.0


def _chart_geometry(public_geometry, convention: int):
    values = public_geometry["tuple"]
    return {
        "alpha0": values[4],
        "theta_jn": values[3],
        # Convention 0 public metadata uses the basis before the clean
        # projector's Appendix-C pi/2 relation; convention 1 already carries
        # that metadata shift.  The composer applies its Table-IV shift once.
        "zeta": values[6] + (math.pi / 2.0 if int(convention) == 0 else 0.0),
        "phi_j": -values[5],
    }


def _public_transport(public_angles, public_zero, public_current, *, convention, aligned):
    zero = public_zero["tuple"]
    current = public_current["tuple"]
    delta_alpha = _wrap(current[4] - zero[4])
    delta_phi_j = _wrap((-current[5]) - (-zero[5]))
    omitted = bool(aligned and int(convention) == 1)
    delta_gamma = 0.0
    if int(convention) == 1 and not omitted:
        delta_gamma = -delta_phi_j
    return {
        "alpha": public_angles["alpha"] + delta_alpha,
        "gamma": public_angles["gamma"] + delta_gamma,
        "cos_beta": public_angles["cos_beta"],
    }, {
        "delta_alpha": delta_alpha,
        "delta_phi_j": delta_phi_j,
        "delta_gamma": delta_gamma,
        "aligned_gamma_transport_omitted": omitted,
        "source": "Appendix C/Table IV source-frame coordinate transport",
    }


def _native_state(case, convention: int, *, aligned: bool):
    angles = _clean_angles(case)
    geom_zero = _clean_geometry(case, 0.0)
    geom_current = _clean_geometry(case, PHIC)
    if aligned:
        # This is the clean v223 endpoint chart before the explicit mapping
        # below.  Keep it in the decomposition so the correction is visible.
        shift = torch.as_tensor(math.pi, dtype=torch.float64)
        geom_zero = (geom_zero[0] + shift, *geom_zero[1:])
        geom_current = (geom_current[0] + shift, *geom_current[1:])
    transported, anchor, details = transport_msa_angles_for_phi_ref(
        {"alpha": angles["alpha"], "gamma": angles["gamma"],
         "cos_beta": angles["cos_beta"]},
        geom_zero,
        geom_current,
        convention=convention,
        aligned=aligned,
    )
    final_spin, final_detail = _native_msa_final_spin(
        torch.as_tensor(case["mass1_msun"], dtype=torch.float64),
        torch.as_tensor(case["mass2_msun"], dtype=torch.float64),
        torch.as_tensor(case["spin1"], dtype=torch.float64),
        torch.as_tensor(case["spin2"], dtype=torch.float64),
        torch.as_tensor(F_REF, dtype=torch.float64),
        angles,
        compatibility="v223",
    )
    native_abs_before_endpoint_chart = {
        "alpha": transported["alpha"] + anchor["alpha0"],
        "gamma": transported["gamma"] + (
            math.pi - anchor["phi_j"] if int(convention) == 1 else 0.0
        ),
        "cos_beta": transported["cos_beta"],
    }
    native_abs = native_abs_before_endpoint_chart
    if int(convention) == 1 and aligned:
        # Mirror the production endpoint mapping so the post-correction chart
        # can be checked independently of the waveform residual.
        endpoint_phi_j = anchor["phi_j"]
        native_abs = {
            "alpha": native_abs["alpha"] - endpoint_phi_j,
            "gamma": native_abs["gamma"] + endpoint_phi_j,
            "cos_beta": native_abs["cos_beta"],
        }
    carrier_phase = 0.0 if int(convention) == 1 else -_scalar(anchor["phi_j"])
    return {
        "angles": angles,
        "geometry_zero": geom_zero,
        "geometry_current": geom_current,
        "transported": transported,
        "absolute_before_endpoint_chart": native_abs_before_endpoint_chart,
        "absolute": native_abs,
        "anchor": anchor,
        "transport_details": details,
        "final_spin": final_spin,
        "final_detail": final_detail,
        "carrier_phase": carrier_phase,
    }


def _metric(actual, target):
    actual = torch.as_tensor(actual, dtype=torch.complex128)
    target = torch.as_tensor(target, dtype=torch.complex128)
    finite = torch.isfinite(actual.real) & torch.isfinite(actual.imag)
    finite &= torch.isfinite(target.real) & torch.isfinite(target.imag)
    if not bool(torch.all(finite)):
        return {
            "support": int(finite.sum()),
            "nonfinite": int((~finite).sum()),
            "relative_l2": None,
            "max_abs": None,
        }
    norm = torch.linalg.vector_norm(target)
    return {
        "support": int(target.numel()),
        "nonfinite": 0,
        "relative_l2": float(torch.linalg.vector_norm(actual - target) / norm),
        "max_abs": float(torch.max(torch.abs(actual - target))),
    }


def _pair_metrics(actual, target):
    return {"h_plus": _metric(actual[0], target[0]),
            "h_cross": _metric(actual[1], target[1])}


def _max_pair_metric(metrics):
    values = [metrics[key]["relative_l2"] for key in ("h_plus", "h_cross")]
    return max(value for value in values if value is not None)


def _samples(values):
    values = torch.as_tensor(values).detach().cpu()
    return {
        f"{float(FREQUENCIES[index]):g}Hz": _scalar(values[index])
        for index in SAMPLE_INDICES
    }


def _angle_samples(values):
    return {
        "alpha": _samples(values["alpha"]),
        "gamma": _samples(values["gamma"]),
        "cos_beta": _samples(values["cos_beta"]),
    }


def _geometry_record(geometry):
    names = ("alpha0", "theta_jn", "zeta", "phi_j")
    return {name: _scalar(value) for name, value in zip(names, geometry)}


def _public_angle_geometry_record(public_geometry):
    values = public_geometry["tuple"]
    return {name: values[index] for index, name in enumerate(public_geometry["semantics"])}


def _chart_deltas(native_absolute, public_absolute):
    da = torch.atan2(
        torch.sin(public_absolute["alpha"] - native_absolute["alpha"]),
        torch.cos(public_absolute["alpha"] - native_absolute["alpha"]),
    )
    dg = torch.atan2(
        torch.sin(public_absolute["gamma"] - native_absolute["gamma"]),
        torch.cos(public_absolute["gamma"] - native_absolute["gamma"]),
    )
    dc = public_absolute["cos_beta"] - native_absolute["cos_beta"]
    return {
        "alpha_max_abs_wrapped": float(torch.max(torch.abs(da))),
        "gamma_max_abs_wrapped": float(torch.max(torch.abs(dg))),
        "alpha_plus_gamma_max_abs_wrapped": float(torch.max(torch.abs(
            torch.atan2(torch.sin(da + dg), torch.cos(da + dg))
        ))),
        "cos_beta_max_abs": float(torch.max(torch.abs(dc))),
        "samples": {
            "alpha": _samples(da),
            "gamma": _samples(dg),
            "alpha_plus_gamma": _samples(da + dg),
            "cos_beta": _samples(dc),
        },
    }


def _remnant_record(case, native):
    mass1 = torch.as_tensor(case["mass1_msun"], dtype=torch.float64)
    mass2 = torch.as_tensor(case["mass2_msun"], dtype=torch.float64)
    eta = _scalar(mass1 * mass2 / (mass1 + mass2).square())
    public = float(ls.SimIMRPhenomXFinalSpin2017(
        eta, float(case["spin1"][2]), float(case["spin2"][2])
    ))
    clean = _scalar(native["final_spin"])
    detail = native["final_detail"]
    return {
        "eta": eta,
        "clean_native_msa_v223": clean,
        "public_aligned_fit": public,
        "clean_minus_public": clean - public,
        "clean_detail": {
            key: _scalar(detail[key])
            for key in (
                "s_av_squared", "mean_s1_l", "mean_s2_l", "initial_s1_l",
                "initial_s2_l", "aligned_kerr_parameter", "remnant_mass_fraction",
                "l_fin",
            )
            if key in detail
        },
        "interpretation": (
            "The remnant scalar is recorded separately.  It is passed to the "
            "opaque carrier and is not fitted or phase-adjusted by this diagnostic."
        ),
    }


def _carrier_record(case, native):
    phase = native["carrier_phase"]
    public_fit = _remnant_record(case, native)["public_aligned_fit"]
    clean_fit = _scalar(native["final_spin"])
    local = IMRPhenomXAS()
    local_clean = local(
        f=FREQUENCIES,
        mass1=case["mass1_msun"],
        mass2=case["mass2_msun"],
        chi1=case["spin1"][2],
        chi2=case["spin2"][2],
        distance=DISTANCE,
        inclination=0.0,
        phi_c=phase,
        t_c=0.0,
        f_ref=F_REF,
        _final_spin_override=clean_fit,
    )
    local_public = local(
        f=FREQUENCIES,
        mass1=case["mass1_msun"],
        mass2=case["mass2_msun"],
        chi1=case["spin1"][2],
        chi2=case["spin2"][2],
        distance=DISTANCE,
        inclination=0.0,
        phi_c=phase,
        t_c=0.0,
        f_ref=F_REF,
        _final_spin_override=public_fit,
    )
    public = _public_xas(case, phase)
    public_tuple = _public_source(case, PHIC, 0)
    table_phase = 0.0 if abs(public_tuple["tuple"][5]) < 1.0e-15 else None
    # Convention-specific public phiJL0 is supplied by the caller below; this
    # default field is replaced in _run_one.  Keeping the carrier comparison
    # here avoids conflating it with the projected boundary residual.
    return {
        "clean_carrier_phase": phase,
        "public_xas_carrier_phase_input": phase,
        "public_table_phase_placeholder": table_phase,
        "local_xas_with_clean_remnant_vs_public_xas": _pair_metrics(local_clean, public),
        "local_xas_with_public_fit_vs_public_xas": _pair_metrics(local_public, public),
        "opaque_carrier_scope": (
            "XAS is called at inclination zero; the clean projector receives "
            "only its positive-frequency h22."
        ),
    }


def _run_one(case, convention: int):
    aligned = _aligned(case)
    public_source_zero = _public_source(case, 0.0, convention)
    public_source_current = _public_source(case, PHIC, convention)
    public_raw_angles = _public_angles(case, convention)
    public_angles, public_transport = _public_transport(
        public_raw_angles,
        public_source_zero,
        public_source_current,
        convention=convention,
        aligned=aligned,
    )
    public_geometry = _chart_geometry(public_source_current, convention)
    native = _native_state(case, convention, aligned=aligned)
    public_absolute = public_angles
    direct = waveform_v223(
        FREQUENCIES,
        case["mass1_msun"], case["mass2_msun"],
        *case["spin1"], *case["spin2"],
        distance=DISTANCE,
        inclination=INCLINATION,
        phic=PHIC,
        tc=0.0,
        f_ref=F_REF,
        convention=convention,
        final_spin_mod=3,
    )
    chart_injected = compose_msa_waveform(
        FREQUENCIES,
        case["mass1_msun"], case["mass2_msun"],
        case["spin1"][2], case["spin2"][2],
        public_absolute,
        public_geometry,
        native["final_spin"],
        distance=DISTANCE,
        tc=0.0,
        f_ref=F_REF,
        convention=convention,
        angles_are_absolute=True,
    )
    target = _public_waveform(case, convention)
    direct_metrics = _pair_metrics(direct, target)
    chart_metrics = _pair_metrics(chart_injected, target)

    native_geometry = native["geometry_current"]
    native_abs = native["absolute"]
    angle_chart = _chart_deltas(native_abs, public_absolute)
    angle_chart_before_endpoint = _chart_deltas(
        native["absolute_before_endpoint_chart"], public_absolute,
    )
    beta_native = torch.acos(torch.clamp(native_abs["cos_beta"], -1.0, 1.0))
    beta_public = torch.acos(torch.clamp(public_absolute["cos_beta"], -1.0, 1.0))
    public_values = public_source_current["tuple"]
    table_phase = 0.0 if convention == 1 else public_values[5]
    carrier = _carrier_record(case, native)
    carrier["public_table_carrier_phase"] = table_phase
    carrier["clean_minus_public_table_phase"] = native["carrier_phase"] - table_phase

    if aligned:
        status = (
            "endpoint_chart_corrected"
            if convention == 1 else "ordinary_aligned_control"
        )
    else:
        status = "known_tiny_spin_internal_failure" if "tiny" in case["id"] else "generic_boundary_neighbor"
    return {
        "fixture": {
            "id": case["id"],
            "label": case["label"],
            "mass1_msun": case["mass1_msun"],
            "mass2_msun": case["mass2_msun"],
            "spin1": list(case["spin1"]),
            "spin2": list(case["spin2"]),
            "distance_mpc": DISTANCE,
            "inclination_rad": INCLINATION,
            "phic_rad": PHIC,
            "f_ref_hz": F_REF,
            "full_three_component_spins": True,
            "exact_transverse_zero": aligned,
        },
        "convention": int(convention),
        "options": _option_record(convention),
        "status": status,
        "strict_gate": {
            "relative_complex_l2": STRICT_GATE,
            "direct_native_pass": _max_pair_metric(direct_metrics) < STRICT_GATE,
            "chart_injected_pass": _max_pair_metric(chart_metrics) < STRICT_GATE,
            "threshold_changed": False,
        },
        "source_frame": {
            "clean_zero": _geometry_record(native["geometry_zero"]),
            "clean_current_chart": _geometry_record(native_geometry),
            "public_zero": _public_angle_geometry_record(public_source_zero),
            "public_current": _public_angle_geometry_record(public_source_current),
            "public_current_mapped_for_clean_projector": public_geometry,
            "public_transport": public_transport,
            "public_tuple_contract": "(chi1, chi2, chip, thetaJN, alpha0, phiJL0, zeta)",
            "phi_j_mapping": "phi_j=-phiJL0",
            "zeta_mapping": "public convention 0 receives +pi/2 clean-basis shift; convention 1 receives no metadata shift before the composer's -pi/2 Table-IV shift",
        },
        "angles": {
            "public_raw": _angle_samples(public_raw_angles),
            "public_transported_absolute": _angle_samples(public_absolute),
            "clean_native_relative": _angle_samples(native["angles"]),
            "clean_native_absolute_before_endpoint_chart": _angle_samples(
                native["absolute_before_endpoint_chart"]
            ),
            "clean_native_absolute": _angle_samples(native_abs),
            "chart_deltas_before_endpoint_correction": angle_chart_before_endpoint,
            "chart_deltas_public_minus_clean": angle_chart,
            "beta_max_rad": {
                "public": float(torch.max(beta_public)),
                "clean": float(torch.max(beta_native)),
            },
            "mprime": MPRIME,
            "angle_api": "SimIMRPhenomXPMSAAngles",
        },
        "remnant": _remnant_record(case, native),
        "carrier": carrier,
        "waveform": {
            "direct_clean_native_vs_public_xp": direct_metrics,
            "public_chart_injected_clean_vs_public_xp": chart_metrics,
            "public_target": "SimInspiralChooseFDWaveform(..., IMRPhenomXP)",
            "grid_hz": {
                "flow": FLOW,
                "fhigh_exclusive": FHIGH,
                "delta_f": DELTA_F,
                "count": int(FREQUENCIES.numel()),
            },
        },
    }


def _equations():
    return {
        "source_frame_observer": "N=(sin(iota) sin(phiRef), sin(iota) cos(phiRef), cos(iota))",
        "source_frame_j_azimuth": "phiJ=atan2(J_y,J_x), with public phiJL0=-phiJ",
        "table_iv_convention_0": "phi_c=phiJL0=-phiJ; clean projector basis is related to public metadata by zeta_clean=zeta_public+pi/2",
        "table_iv_convention_1": "phi_c=0; public convention-1 zeta metadata is consumed as reported and the clean projector applies its -pi/2 Table-IV shift once",
        "eq45_endpoint": "The m'=2 term depends on alpha+gamma at beta=0; m'!=2 terms are proportional to finite-beta Wigner factors",
        "aligned_chart_relation": "For convention 1 at exact alignment, delta_alpha=phiRef-pi/2=-phiJ and delta_gamma=+phiJ, so delta_alpha+delta_gamma=0",
        "cause": (
            "Before the endpoint mapping, the clean endpoint retains the helper's "
            "finite root-regularized beta (about 3e-5--4e-5 rad).  Therefore the "
            "m'!=2 O(beta) terms expose the convention-1 chart offsets even though "
            "the leading m'=2 sum is unchanged.  The native v223 wrapper now applies "
            "the same public alpha/gamma chart relation; its direct waveform remains "
            "within the unchanged gate."
        ),
        "endpoint_correction": (
            "For exact transverse-zero v223 convention 1 only, subtract the current "
            "clean phi_J from relative alpha and add it to relative gamma before "
            "projection.  This preserves alpha+gamma and cos(beta), leaves tiny "
            "nonzero transverse rows on the legacy/generic behavior, and does not "
            "alter the carrier or remnant."
        ),
        "reusable_contract": (
            "At exact aligned v223 convention 1, apply the paired endpoint chart "
            "mapping before Eq. (45) projection: delta_alpha=-phi_J and "
            "delta_gamma=+phi_J.  Retain the current finite cos(beta) and root "
            "behavior, keep exact-zero selection separate from tiny nonzero "
            "transverse spins, and retain the Table-IV carrier phases."
        ),
        "scope": "This report diagnoses XP v223 convention 0/1 exact-aligned and zero-spin boundaries plus a tiny-spin neighbor; it does not broaden to XPHM or a tiny-spin physics correction.",
    }


def _minimal_failing_regression(rows):
    for row in rows:
        if row["fixture"]["id"] == "audited_tiny_neighbor_35_25" and row["convention"] == 1:
            return {
                "fixture": row["fixture"]["id"],
                "convention": 1,
                "direct_native_relative_complex_l2": max(
                    row["waveform"]["direct_clean_native_vs_public_xp"][key]["relative_l2"]
                    for key in ("h_plus", "h_cross")
                ),
                "public_chart_injected_relative_complex_l2": max(
                    row["waveform"]["public_chart_injected_clean_vs_public_xp"][key]["relative_l2"]
                    for key in ("h_plus", "h_cross")
                ),
                "expected": "tiny nonzero transverse route remains visibly above 1e-6; exact endpoint rows pass after chart correction",
                "reason": "the tiny-spin discrepancy is a separate MSA internal boundary failure and is not corrected here",
            }
    return {"status": "unavailable"}


def build_report():
    hashes_before = _helper_hashes()
    rows = []
    for case in FIXTURES:
        for convention in (0, 1):
            print(f"running fixture={case['id']} convention={convention}", flush=True)
            try:
                rows.append(_run_one(case, convention))
            except Exception as exc:
                rows.append({
                    "fixture": {
                        "id": case["id"],
                        "label": case["label"],
                        "mass1_msun": case["mass1_msun"],
                        "mass2_msun": case["mass2_msun"],
                        "spin1": list(case["spin1"]),
                        "spin2": list(case["spin2"]),
                    },
                    "convention": convention,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
    hashes_after = _helper_hashes()
    return {
        "provenance": (
            "Clean-room equation-level diagnosis. Public LALSimulation angle, "
            "source-frame, remnant, XAS, and XP waveform functions are black-box "
            "interfaces. Local clean MSA/source-frame helpers are treated as "
            "opaque where their physics is outside this boundary diagnosis."
        ),
        "equations": _equations(),
        "fixtures": "The audited and fresh fixture labels are immutable report metadata; no frozen catalog/holdout case is mutated or rerouted.",
        "grid": {
            "flow_hz": FLOW,
            "fhigh_exclusive_hz": FHIGH,
            "delta_f_hz": DELTA_F,
            "count": int(FREQUENCIES.numel()),
            "sample_frequencies_hz": [float(FREQUENCIES[index]) for index in SAMPLE_INDICES],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": getattr(torch, "__version__", None),
            "lal": getattr(lal, "__version__", None),
            "lalsimulation": getattr(ls, "__version__", None),
            "helper_hashes_before": hashes_before,
            "helper_hashes_after": hashes_after,
            "helper_hashes_stable_during_run": hashes_before == hashes_after,
            "diagnostic_script_sha256": _sha256(Path(__file__).resolve()),
        },
        "public_contract": {
            "prec_version": 223,
            "conventions": [0, 1],
            "final_spin_mod": 3,
            "expansion_order": EXPANSION_ORDER,
            "mprime": MPRIME,
            "source_frame_api": "SimIMRPhenomXPCalculateModelParametersFromSourceFrame",
            "angle_api": "SimIMRPhenomXPMSAAngles",
            "remnant_api": "SimIMRPhenomXFinalSpin2017",
            "carrier_api": "SimInspiralChooseFDWaveform(..., IMRPhenomXAS)",
            "waveform_api": "SimInspiralChooseFDWaveform(..., IMRPhenomXP)",
        },
        "strict_gate": {
            "relative_complex_l2": STRICT_GATE,
            "threshold_changed": False,
            "known_boundary_failures_are_visible": True,
        },
        "minimal_failing_regression": _minimal_failing_regression(rows),
        "rows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "report.json")
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return report


if __name__ == "__main__":
    main()
