"""Bounded black-box diagnostic for the clean-room MSA XP composition.

The report keeps three boundaries separate:

* public MSA angles injected into the fresh projector with the NNLO final-spin
  choice (public ``FinalSpinMod=0``),
* the clean native MSA angles with that same explicit scalar, and
* the clean MSA averaged final-spin formula against public ``FinalSpinMod=3``.

The installed LALSimulation functions are used as black-box references.  No
phase, time, or norm optimization is performed; every reported waveform
number is a raw complex polarization error.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import platform
import sys
from pathlib import Path

import lal
import lalsimulation as ls
import torch

from torchwave.approximants.imrphenom_xas import XASParameterFits
from torchwave.approximants.precession_msa_angles_clean import (
    assemble_precession_msa_angles,
)
from torchwave.approximants.precession_msa_waveform_clean import (
    _native_msa_final_spin,
    compose_msa_waveform,
    msa_final_spin_normalization_report,
)
from torchwave.approximants.precession_nnlo_clean import effective_chip
from torchwave.approximants.precession_beta_clean import orbital_angular_momentum
from torchwave.approximants.precession_source_frame_clean import (
    source_frame_geometry,
)
from torchwave.approximants.precession_twisting_clean import (
    project_twisted_positive_modes,
    twist_mode,
)
from torchwave.waveforms import get_fd_waveform


FREQUENCIES = torch.arange(20.0, 513.0, 1.0, dtype=torch.float64)
FLOW = 20.0
FHIGH = 512.0
F_REF = 20.0
DISTANCE = 500.0
INCLINATION = 0.7
PHASES = (0.0, 0.2)
VERSIONS = (220, 223, 224)
PRIMARY_VERSION = 223
EXPANSION_ORDER = 5

CASES = (
    {
        "name": "A",
        "mass1_msun": 35.0,
        "mass2_msun": 25.0,
        "chi1": (0.2, 0.1, 0.2),
        "chi2": (-0.1, 0.15, -0.1),
    },
    {
        "name": "B",
        "mass1_msun": 45.0,
        "mass2_msun": 15.0,
        "chi1": (0.45, -0.2, 0.3),
        "chi2": (0.1, 0.2, -0.25),
    },
    {
        "name": "C",
        "mass1_msun": 30.0,
        "mass2_msun": 10.0,
        "chi1": (0.12, -0.31, 0.44),
        "chi2": (-0.22, 0.18, -0.07),
    },
    {
        "name": "aligned_control",
        "mass1_msun": 35.0,
        "mass2_msun": 25.0,
        "chi1": (0.0, 0.0, 0.2),
        "chi2": (0.0, 0.0, -0.1),
    },
)


def _dict_options(*, version=223, convention=0, final_spin_mod=None):
    params = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(params, int(version))
    ls.SimInspiralWaveformParamsInsertPhenomXPExpansionOrder(
        params, EXPANSION_ORDER
    )
    ls.SimInspiralWaveformParamsInsertPhenomXPConvention(
        params, int(convention)
    )
    if final_spin_mod is not None:
        ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(
            params, int(final_spin_mod)
        )
    return params


def _looked_up_options(params):
    """Read back every public option used by this diagnostic call."""
    return {
        "prec_version": int(ls.SimInspiralWaveformParamsLookupPhenomXPrecVersion(params)),
        "expansion_order": int(ls.SimInspiralWaveformParamsLookupPhenomXPExpansionOrder(params)),
        "convention": int(ls.SimInspiralWaveformParamsLookupPhenomXPConvention(params)),
        "final_spin_mod": int(ls.SimInspiralWaveformParamsLookupPhenomXPFinalSpinMod(params)),
    }


def _public_reference_parameters(case):
    """Build the one physical-parameter record shared by public calls."""
    return {
        "mass1_si": float(case["mass1_msun"]) * lal.MSUN_SI,
        "mass2_si": float(case["mass2_msun"]) * lal.MSUN_SI,
        "spin1": tuple(float(value) for value in case["chi1"]),
        "spin2": tuple(float(value) for value in case["chi2"]),
    }


def _lal_angles(
    case, *, version=PRIMARY_VERSION, convention=0, physical_parameters=None
):
    physical = physical_parameters or _public_reference_parameters(case)
    fv = lal.CreateREAL8Vector(FREQUENCIES.numel())
    fv.data[:] = FREQUENCIES.tolist()
    out = ls.SimIMRPhenomXPMSAAngles(
        fv,
        physical["mass1_si"],
        physical["mass2_si"],
        *physical["spin1"],
        *physical["spin2"],
        INCLINATION,
        F_REF,
        2,
        _dict_options(version=version, convention=convention, final_spin_mod=0),
    )
    return {
        "alpha": torch.as_tensor(list(out[0].data), dtype=torch.float64),
        "gamma": torch.as_tensor(list(out[1].data), dtype=torch.float64),
        "cos_beta": torch.as_tensor(list(out[2].data), dtype=torch.float64),
    }


def _public_source_frame_geometry(
    case,
    phic,
    *,
    version=PRIMARY_VERSION,
    convention,
    final_spin_mod=0,
    physical_parameters=None,
):
    """Call the public source-frame metadata function with exact SI inputs."""
    physical = physical_parameters or _public_reference_parameters(case)
    params = _dict_options(
        version=version,
        convention=convention,
        final_spin_mod=final_spin_mod,
    )
    output = ls.SimIMRPhenomXPCalculateModelParametersFromSourceFrame(
        physical["mass1_si"],
        physical["mass2_si"],
        F_REF,
        float(phic),
        INCLINATION,
        *physical["spin1"],
        *physical["spin2"],
        params,
    )
    values = tuple(float(value) for value in output)
    return {
        "tuple": list(values),
        "semantics": [
            "chi1", "chi2", "chip", "thetaJN", "alpha0", "phiJL0", "zeta"
        ],
        "call": {
            "mass1_si": physical["mass1_si"],
            "mass2_si": physical["mass2_si"],
            "phiRef": float(phic),
            "inclination": INCLINATION,
            "spin1": list(physical["spin1"]),
            "spin2": list(physical["spin2"]),
            "f_ref": F_REF,
            "options_requested": {
                "prec_version": int(version),
                "expansion_order": EXPANSION_ORDER,
                "convention": int(convention),
                "final_spin_mod": int(final_spin_mod),
            },
            "options_looked_up": _looked_up_options(params),
        },
    }


def _public_geometry_for_composition(public_geometry, *, convention):
    """Map the seven public values to the clean ``(alpha,theta,zeta,phiJ)``.

    Appendix C uses the ABFO/Kidder polarization basis in the clean projector.
    The paper's basis relation is a ``pi/2`` rotation.  Public convention 0
    reports the unshifted metadata, while convention 1 already reports its
    convention basis shift; applying the relation a second time would flip the
    injected carrier.  The convention-1 composition step still applies its
    separate Table-IV ``-pi/2`` projector shift.
    """
    chi1, chi2, chip, theta_jn, alpha0, phi_jl0, zeta_public = public_geometry["tuple"]
    zeta_basis_shift = math.pi / 2.0 if int(convention) == 0 else 0.0
    return {
        "alpha0": alpha0,
        "theta_jn": theta_jn,
        "zeta": zeta_public + zeta_basis_shift,
        "phi_j": -phi_jl0,
        "metadata": {
            "public_tuple_semantics": public_geometry["semantics"],
            "phi_j_definition": "-phiJL0",
            "zeta_basis_shift_applied": zeta_basis_shift,
            "zeta_basis_shift_reason": "Appendix C Eq. (1782--1786) basis relation; convention 1 public metadata already includes it",
            "public_zeta_used_once": True,
        },
    }


def _angle_difference(a, b):
    """Return a wrapped angle difference in ``[-pi, pi)``."""
    return (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi


def _polarization_angle_difference(a, b):
    """Return a zeta difference modulo its physical ``pi`` period."""
    return (float(a) - float(b) + math.pi / 2.0) % math.pi - math.pi / 2.0


def _source_frame_transverse_spin_norm(case):
    return math.sqrt(sum(float(value) ** 2 for value in case["chi1"][:2]) +
                     sum(float(value) ** 2 for value in case["chi2"][:2]))


def _transport_public_angles(
    public_angles,
    public_geometry_zero,
    public_geometry_phic,
    *,
    convention,
    aligned,
):
    """Transport the public angle triplet to a changed source ``phiRef``.

    The public angle API has no ``phiRef`` argument.  Appendix C/Table IV give
    the coordinate change: alpha receives the source-frame ``alpha0`` delta;
    convention 1 receives the opposite J-azimuth delta in gamma.  At aligned
    J the azimuth is a gauge branch, so the gamma transport is intentionally
    omitted and the aligned projected limit is reported separately.
    """
    zero = public_geometry_zero["tuple"]
    current = public_geometry_phic["tuple"]
    delta_alpha = _angle_difference(current[4], zero[4])
    delta_phi_j = _angle_difference(-current[5], -zero[5])
    delta_gamma = 0.0
    if int(convention) == 1 and not aligned:
        delta_gamma = -delta_phi_j
    shifted = {
        "alpha": public_angles["alpha"] + delta_alpha,
        "gamma": public_angles["gamma"] + delta_gamma,
        "cos_beta": public_angles["cos_beta"],
    }
    return shifted, {
        "delta_alpha_from_public_alpha0": delta_alpha,
        "delta_phi_j_from_public_phiJL0": delta_phi_j,
        "delta_gamma_convention1": delta_gamma,
        "aligned_gamma_transport_omitted": bool(aligned and int(convention) == 1),
        "source": "Appendix C/Table IV coordinate change; no residual fit",
    }


def _geometry_comparison(fresh_geometry, public_geometry, mapped_geometry):
    fresh = tuple(float(value) for value in fresh_geometry)
    public = public_geometry["tuple"]
    return {
        "fresh_vs_public_raw": {
            "alpha0_wrapped": _angle_difference(fresh[0], public[4]),
            "theta_jn": fresh[1] - public[3],
            "zeta_wrapped": _angle_difference(fresh[2], public[6]),
            "phi_j_vs_neg_phiJL0_wrapped": _angle_difference(fresh[3], -public[5]),
        },
        "fresh_vs_mapped_public": {
            "alpha0_wrapped": _angle_difference(fresh[0], mapped_geometry["alpha0"]),
            "theta_jn": fresh[1] - mapped_geometry["theta_jn"],
            "zeta_wrapped_2pi": _angle_difference(fresh[2], mapped_geometry["zeta"]),
            "zeta_wrapped_mod_pi": _polarization_angle_difference(fresh[2], mapped_geometry["zeta"]),
            "phi_j_wrapped": _angle_difference(fresh[3], mapped_geometry["phi_j"]),
        },
    }


def _rotation_z(angle, vector):
    """Apply the Appendix C active R_z(angle) matrix to a 3-vector."""
    c = math.cos(angle)
    s = math.sin(angle)
    x, y, z = vector
    return (c * x - s * y, s * x + c * y, z)


def _rotation_y(angle, vector):
    """Apply the Appendix C active R_y(angle) matrix to a 3-vector."""
    c = math.cos(angle)
    s = math.sin(angle)
    x, y, z = vector
    return (c * x + s * z, y, -s * x + c * z)


def _vector_norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def _vector_difference(first, second):
    return tuple(a - b for a, b in zip(first, second))


def _normalized_spin_sum(case):
    """Return the source-total-mass-normalized spin sum from Eq. (JinLframe)."""
    m1 = float(case["mass1_msun"])
    m2 = float(case["mass2_msun"])
    total = m1 + m2
    n1 = m1 / total
    n2 = m2 / total
    return tuple(
        n1 * n1 * float(case["chi1"][i]) + n2 * n2 * float(case["chi2"][i])
        for i in range(3)
    )


def _clean_orbital_l_candidates(case):
    """Evaluate explicit clean L prescriptions used by the two clean paths."""
    m1 = torch.tensor(case["mass1_msun"], dtype=torch.float64)
    m2 = torch.tensor(case["mass2_msun"], dtype=torch.float64)
    f_ref = torch.tensor(F_REF, dtype=torch.float64)
    s1z = torch.tensor(case["chi1"][2], dtype=torch.float64)
    s2z = torch.tensor(case["chi2"][2], dtype=torch.float64)
    values = {
        "source_frame_102_printed": orbital_angular_momentum(
            m1, m2, f_ref, s1z, s2z, pn_version=102, compat_102=False,
        ),
        "source_frame_102_compatibility": orbital_angular_momentum(
            m1, m2, f_ref, s1z, s2z, pn_version=102, compat_102=True,
        ),
        "source_frame_full_printed": orbital_angular_momentum(
            m1, m2, f_ref, s1z, s2z, pn_version="full", compat_102=False,
        ),
    }
    total = m1 + m2
    eta = m1 * m2 / total.square()
    v_ref = (math.pi * total * 4.925490947641267e-6 * f_ref).pow(1.0 / 3.0)
    values["native_msa_newtonian_reference"] = eta / v_ref
    return {name: _scalar(value) for name, value in values.items()}


def _appendix_c_public_jhat(public_geometry, phic):
    """Invert the documented Appendix C geometry tuple to a J direction.

    The public metadata uses phiJL0 with the opposite azimuth sign from the
    clean source-frame helper, so phi_j=-phiJL0 is applied explicitly. This
    function performs only the rotations in Eqs. (NinLframe)--(rotJtoL); it
    does not infer a public implementation formula.
    """
    values = public_geometry["tuple"]
    theta_jn = float(values[3])
    alpha0 = float(values[4])
    phi_jl0 = float(values[5])
    phi_j = -phi_jl0
    kappa = math.pi - alpha0
    n_l = (
        math.sin(INCLINATION) * math.sin(float(phic)),
        math.sin(INCLINATION) * math.cos(float(phic)),
        math.cos(INCLINATION),
    )
    # Eq. (rotJtoL) gives N_L=Rz(phiJ) Ry(thetaJL0) Rz(kappa) N_J.
    n_j = (math.sin(theta_jn), 0.0, math.cos(theta_jn))
    v = _rotation_z(kappa, n_j)
    u = _rotation_z(-phi_j, n_l)
    numerator = v[2] * u[0] - v[0] * u[2]
    denominator = v[0] * u[0] + v[2] * u[2]
    theta_jl0 = math.atan2(numerator, denominator)
    jhat = (
        math.sin(theta_jl0) * math.cos(phi_j),
        math.sin(theta_jl0) * math.sin(phi_j),
        math.cos(theta_jl0),
    )
    # These residuals independently check the inversion, rather than serving
    # as waveform residuals or fitted adjustments.
    u_reconstructed = _rotation_y(theta_jl0, v)
    n_reconstructed = _rotation_z(phi_j, u_reconstructed)
    return {
        "theta_jl0": theta_jl0,
        "phi_j": phi_j,
        "kappa": kappa,
        "jhat": list(jhat),
        "rotation_y_residual": u[1] - v[1],
        "n_l_reconstruction_l2": _vector_norm(_vector_difference(n_reconstructed, n_l)),
        "jhat_norm_error": abs(_vector_norm(jhat) - 1.0),
    }


def _effective_l_from_jhat(spin_sum, jhat):
    """Solve J=S+L*z for L, conditional on the Appendix C J direction."""
    jx, jy, jz = jhat
    sx, sy, sz = spin_sum
    j_perp_sq = jx * jx + jy * jy
    if j_perp_sq <= 1.0e-28:
        return {
            "status": "aligned_branch_degenerate",
            "l_over_m2": None,
            "j_over_m2": None,
            "transverse_component_residual": None,
        }
    j_magnitude = (sx * jx + sy * jy) / j_perp_sq
    inferred_l = j_magnitude * jz - sz
    residual = max(abs(sx - j_magnitude * jx), abs(sy - j_magnitude * jy))
    j_vector = (j_magnitude * jx, j_magnitude * jy, j_magnitude * jz)
    return {
        "status": "conditional_geometric_inversion",
        "l_over_m2": inferred_l,
        "j_over_m2": list(j_vector),
        "j_magnitude_over_m2": j_magnitude,
        "transverse_component_residual": residual,
    }


def _clean_j_vector(case, l_over_m2):
    spin_sum = _normalized_spin_sum(case)
    j = (spin_sum[0], spin_sum[1], spin_sum[2] + l_over_m2)
    norm = _vector_norm(j)
    return {
        "spin_sum_over_m2": list(spin_sum),
        "l_over_m2": l_over_m2,
        "j_over_m2": list(j),
        "j_magnitude_over_m2": norm,
        "jhat": [value / norm for value in j],
    }


def _geometry_vector_diagnostic(case, zero_by_version, current_by_version, phic):
    """Compare exact Appendix C components and J/L diagnostics by version."""
    clean_l = _clean_orbital_l_candidates(case)
    clean = _clean_j_vector(case, clean_l["source_frame_102_printed"])
    spin_sum = tuple(clean["spin_sum_over_m2"])
    result = {
        "status": "complete",
        "equations": {
            "NinLframe": "N=(sin(iota) sin(phiRef), sin(iota) cos(phiRef), cos(iota))",
            "JinLframe": "J=(Sx,Sy,L+Sz), with S_i=(m_i/M)^2 chi_i",
            "rotJtoL": "N_L=Rz(phiJ) Ry(thetaJL0) Rz(kappa) N_J",
            "alpha0": "alpha0=pi-kappa",
            "phi_mapping": "phiJ=-phiJL0 for the public tuple",
        },
        "clean_reference": clean,
        "clean_l_candidates": clean_l,
        "versions": {},
        "v223_interpretation": (
            "The v223 effective L is conditional on treating its returned "
            "thetaJN/alpha0/phiJL0 as the Appendix C geometric tuple. The "
            "paper identifies v223 as a distinct MSA phi_z,0/zeta_z,0 "
            "prescription; this diagnostic does not promote an undocumented "
            "effective L or fit any correction."
        ),
    }
    control = zero_by_version.get("220")
    control_tuple = control["tuple"] if control is not None else None
    for version in VERSIONS:
        key = str(version)
        zero = zero_by_version.get(key)
        current = current_by_version.get(key)
        entry = {"version": int(version)}
        if zero is None or current is None:
            entry["status"] = "unavailable"
            result["versions"][key] = entry
            continue
        zero_inverse = _appendix_c_public_jhat(zero, 0.0)
        current_inverse = _appendix_c_public_jhat(current, phic)
        zero_l = _effective_l_from_jhat(spin_sum, zero_inverse["jhat"])
        current_l = _effective_l_from_jhat(spin_sum, current_inverse["jhat"])
        entry["status"] = zero_l["status"]
        entry["phic_zero"] = {**zero_inverse, **zero_l}
        entry["phic_current"] = {**current_inverse, **current_l}
        entry["jhat_vs_clean_l2"] = _vector_norm(
            _vector_difference(zero_inverse["jhat"], clean["jhat"])
        )
        entry["jhat_current_vs_zero_l2"] = _vector_norm(
            _vector_difference(current_inverse["jhat"], zero_inverse["jhat"])
        )
        if zero_l["j_over_m2"] is not None:
            clean_j = tuple(clean["j_over_m2"])
            public_j = tuple(zero_l["j_over_m2"])
            j_delta = _vector_difference(public_j, clean_j)
            entry["j_over_m2_delta_vs_clean"] = list(j_delta)
            entry["j_over_m2_delta_norm_vs_clean"] = _vector_norm(j_delta)
            entry["j_magnitude_delta_vs_clean"] = (
                zero_l["j_magnitude_over_m2"] - clean["j_magnitude_over_m2"]
            )
        if zero_l["l_over_m2"] is not None:
            entry["conditional_l_delta_vs_clean"] = (
                zero_l["l_over_m2"] - clean["l_over_m2"]
            )
        if control_tuple is not None:
            version_tuple = zero["tuple"]
            entry["metadata_delta_vs_v220"] = {
                "thetaJN": version_tuple[3] - control_tuple[3],
                "alpha0_wrapped": _angle_difference(version_tuple[4], control_tuple[4]),
                "phiJL0_wrapped": _angle_difference(version_tuple[5], control_tuple[5]),
                "zeta_wrapped": _angle_difference(version_tuple[6], control_tuple[6]),
            }
        result["versions"][key] = entry
    return result


def _rotation_identity_checks():
    """Check Eq. (45) alpha covariance and the Eq. (146) basis sign."""
    f = torch.tensor([20.0, 80.0, 512.0], dtype=torch.float64)
    alpha = torch.tensor([0.13, 0.41, 0.92], dtype=torch.float64)
    gamma = torch.tensor([-0.2, 0.07, -0.31], dtype=torch.float64)
    beta = torch.tensor([0.24, 0.29, 0.33], dtype=torch.float64)
    delta = 0.37
    mode_values = torch.tensor([1.0 + 0.2j, 0.4 - 0.7j, -0.2 + 0.9j], dtype=torch.complex128)
    twisted_modes = {2: mode_values}
    twisted = twist_mode(2, 2, alpha, gamma, beta, twisted_modes)
    twisted_shifted = twist_mode(2, 2, alpha + delta, gamma, beta, twisted_modes)
    alpha_factor = torch.exp(torch.tensor(-2j * delta, dtype=torch.complex128))
    alpha_identity = torch.max(torch.abs(twisted_shifted - alpha_factor * twisted))
    projected_modes = {(2, 2): mode_values}
    projected = project_twisted_positive_modes(projected_modes, alpha, gamma, beta, 0.61, 0.27)
    projected_shifted = project_twisted_positive_modes(projected_modes, alpha, gamma, beta, 0.61, 0.27 + math.pi / 2.0)
    zeta_identity = max(
        torch.max(torch.abs(projected_shifted[0] + projected[0])),
        torch.max(torch.abs(projected_shifted[1] + projected[1])),
    )
    return {
        "alpha_shift_max_abs": float(alpha_identity),
        "zeta_plus_pi_over_2_sign_max_abs": float(zeta_identity),
        "frequency_grid_hz": f.tolist(),
        "status": "pass" if float(max(alpha_identity, zeta_identity)) < 1e-13 else "fail",
    }


def _lal_waveform(
    case, phic, *, convention, final_spin_mod, physical_parameters=None
):
    physical = physical_parameters or _public_reference_parameters(case)
    hp, hc = ls.SimInspiralChooseFDWaveform(
        physical["mass1_si"],
        physical["mass2_si"],
        *physical["spin1"],
        *physical["spin2"],
        DISTANCE * 1.0e6 * lal.PC_SI,
        INCLINATION,
        phic,
        0.0,
        0.0,
        0.0,
        1.0,
        FLOW,
        FHIGH,
        F_REF,
        _dict_options(
            version=PRIMARY_VERSION,
            convention=convention,
            final_spin_mod=final_spin_mod,
        ),
        ls.IMRPhenomXP,
    )
    # The public constructor starts at zero Hz, so the requested grid is the
    # integer-frequency slice [20:513].
    return (
        torch.as_tensor(hp.data.data[20:513].copy(), dtype=torch.complex128),
        torch.as_tensor(hc.data.data[20:513].copy(), dtype=torch.complex128),
    )


def _target_parameter_record(case, physical_parameters):
    """Check and record the exact vectors passed to both public calls."""
    angle_spin1 = tuple(float(value) for value in case["chi1"])
    angle_spin2 = tuple(float(value) for value in case["chi2"])
    angle_call_spin1 = tuple(physical_parameters["spin1"])
    angle_call_spin2 = tuple(physical_parameters["spin2"])
    waveform_call_spin1 = tuple(physical_parameters["spin1"])
    waveform_call_spin2 = tuple(physical_parameters["spin2"])
    matches = (
        angle_spin1 == angle_call_spin1 == waveform_call_spin1
        and angle_spin2 == angle_call_spin2 == waveform_call_spin2
    )
    if not matches:
        raise AssertionError("public waveform target spins differ from angle-input spins")
    return {
        "mass1_si": physical_parameters["mass1_si"],
        "mass2_si": physical_parameters["mass2_si"],
        "angle_input_spin1": list(angle_spin1),
        "angle_input_spin2": list(angle_spin2),
        "angle_call_spin1": list(angle_call_spin1),
        "angle_call_spin2": list(angle_call_spin2),
        "waveform_call_spin1": list(waveform_call_spin1),
        "waveform_call_spin2": list(waveform_call_spin2),
        "waveform_target_spin1": list(waveform_call_spin1),
        "waveform_target_spin2": list(waveform_call_spin2),
        "spin_vectors_exact_match": matches,
    }


def _torchwave_public(case, phic, *, convention, final_spin_mod):
    s1 = torch.as_tensor(case["chi1"], dtype=torch.float64)
    s2 = torch.as_tensor(case["chi2"], dtype=torch.float64)
    return get_fd_waveform(
        "IMRPhenomXP",
        mass1=case["mass1_msun"],
        mass2=case["mass2_msun"],
        spin1x=s1[0],
        spin1y=s1[1],
        spin1z=s1[2],
        spin2x=s2[0],
        spin2y=s2[1],
        spin2z=s2[2],
        distance=DISTANCE,
        inclination=INCLINATION,
        phic=phic,
        tc=0.0,
        f_ref=F_REF,
        sample_frequencies=FREQUENCIES,
        dtype=torch.float64,
        prec_version=PRIMARY_VERSION,
        convention=convention,
        final_spin_mod=final_spin_mod,
    )


def _nnlo_final_spin(case):
    m1 = torch.tensor(case["mass1_msun"], dtype=torch.float64)
    m2 = torch.tensor(case["mass2_msun"], dtype=torch.float64)
    s1 = torch.tensor(case["chi1"], dtype=torch.float64)
    s2 = torch.tensor(case["chi2"], dtype=torch.float64)
    eta = m1 * m2 / (m1 + m2).square()
    chip = effective_chip(m1, m2, s1[0], s1[1], s2[0], s2[1])
    return XASParameterFits.fit_final_spin_nnlo(eta, s1[2], s2[2], chip)


def _clean_geometry(case, phic):
    return source_frame_geometry(
        case["mass1_msun"],
        case["mass2_msun"],
        case["chi1"][0],
        case["chi1"][1],
        case["chi1"][2],
        case["chi2"][0],
        case["chi2"][1],
        case["chi2"][2],
        INCLINATION,
        phic,
        F_REF,
    )


def _native_angles(case):
    return assemble_precession_msa_angles(
        FREQUENCIES,
        case["mass1_msun"],
        case["mass2_msun"],
        case["chi1"],
        case["chi2"],
        f_ref=F_REF,
        order=EXPANSION_ORDER,
        beta_pn=False,
        average_mode="leading",
    )


def _synthetic_aligned_angles():
    zeros = torch.zeros_like(FREQUENCIES)
    return {"alpha": zeros, "gamma": zeros, "cos_beta": torch.ones_like(zeros)}


def _compose(case, phic, angles, geometry, final_spin, *, convention, absolute):
    s1 = case["chi1"]
    s2 = case["chi2"]
    return compose_msa_waveform(
        FREQUENCIES,
        case["mass1_msun"],
        case["mass2_msun"],
        s1[2],
        s2[2],
        angles,
        geometry,
        final_spin,
        distance=DISTANCE,
        tc=0.0,
        f_ref=F_REF,
        convention=convention,
        angles_are_absolute=absolute,
    )


def _raw_metric(got, target):
    got = torch.as_tensor(got, dtype=torch.complex128)
    target = torch.as_tensor(target, dtype=torch.complex128)
    finite = torch.isfinite(got.real) & torch.isfinite(got.imag)
    finite &= torch.isfinite(target.real) & torch.isfinite(target.imag)
    if not bool(torch.all(finite)):
        return {
            "support": int(finite.sum()),
            "nonfinite": int((~finite).sum()),
            "relative_l2": None,
            "max_abs": None,
        }
    target_norm = torch.linalg.vector_norm(target)
    if float(target_norm) == 0.0:
        return {
            "support": int(target.numel()),
            "nonfinite": 0,
            "relative_l2": None,
            "max_abs": float(torch.max(torch.abs(got - target))),
        }
    return {
        "support": int(target.numel()),
        "nonfinite": 0,
        "relative_l2": float(torch.linalg.vector_norm(got - target) / target_norm),
        "max_abs": float(torch.max(torch.abs(got - target))),
    }


def _metrics(pair, target):
    return {
        "hp": _raw_metric(pair[0], target[0]),
        "hc": _raw_metric(pair[1], target[1]),
    }


def _scalar(value):
    return float(torch.as_tensor(value).detach().cpu())


def _angle_summary(public):
    return {
        "alpha_at_f_ref": _scalar(public["alpha"][0]),
        "gamma_at_f_ref": _scalar(public["gamma"][0]),
        "cos_beta_at_f_ref": _scalar(public["cos_beta"][0]),
        "alpha_last": _scalar(public["alpha"][-1]),
        "gamma_last": _scalar(public["gamma"][-1]),
        "cos_beta_last": _scalar(public["cos_beta"][-1]),
    }


def _run_one(case, phic, convention):
    physical_parameters = _public_reference_parameters(case)
    aligned = _source_frame_transverse_spin_norm(case) <= 1.0e-14
    row = {
        "case": case["name"],
        "phic": phic,
        "convention": convention,
        "parameters": {
            "mass1_msun": case["mass1_msun"],
            "mass2_msun": case["mass2_msun"],
            "chi1": list(case["chi1"]),
            "chi2": list(case["chi2"]),
            "distance_mpc": DISTANCE,
            "inclination_rad": INCLINATION,
            "f_ref_hz": F_REF,
            "frequency_hz": [FLOW, FHIGH],
            "delta_f_hz": 1.0,
        },
        "target_parameter_record": _target_parameter_record(case, physical_parameters),
        "public_callable": {
            "angles": "SimIMRPhenomXPMSAAngles",
            "source_frame_geometry": "SimIMRPhenomXPCalculateModelParametersFromSourceFrame",
            "waveform": "SimInspiralChooseFDWaveform(..., IMRPhenomXP)",
            "angles_version": PRIMARY_VERSION,
            "source_frame_geometry_versions": list(VERSIONS),
            "expansion_order": EXPANSION_ORDER,
            "mprime": 2,
            "convention": convention,
            "final_spin_mod": 0,
        },
    }
    try:
        geometry = _clean_geometry(case, phic)
        fs0 = _nnlo_final_spin(case)
        public_geometry_zero_by_version = {
            str(version): _public_source_frame_geometry(
                case, 0.0, version=version, convention=convention, final_spin_mod=0,
                physical_parameters=physical_parameters,
            )
            for version in VERSIONS
        }
        public_geometry_current_by_version = {
            str(version): _public_source_frame_geometry(
                case, phic, version=version, convention=convention, final_spin_mod=0,
                physical_parameters=physical_parameters,
            )
            for version in VERSIONS
        }
        public_geometry_zero = public_geometry_zero_by_version[str(PRIMARY_VERSION)]
        public_geometry_current = public_geometry_current_by_version[str(PRIMARY_VERSION)]
        mapped_geometry_zero = _public_geometry_for_composition(
            public_geometry_zero, convention=convention,
        )
        mapped_geometry = _public_geometry_for_composition(
            public_geometry_current, convention=convention,
        )
        row["geometry"] = {
            "alpha0": _scalar(geometry[0]),
            "theta_jn": _scalar(geometry[1]),
            "zeta": _scalar(geometry[2]),
            "phi_j": _scalar(geometry[3]),
        }
        row["public_source_frame_geometry"] = {
            "phic_zero": public_geometry_zero,
            "phic_current": public_geometry_current,
            "versions_phic_zero": public_geometry_zero_by_version,
            "versions_phic_current": public_geometry_current_by_version,
            "mapped_zero": mapped_geometry_zero,
            "mapped_current": mapped_geometry,
        }
        row["geometry_comparison"] = _geometry_comparison(
            geometry, public_geometry_current, mapped_geometry,
        )
        row["geometry_vector_diagnostic"] = _geometry_vector_diagnostic(
            case,
            public_geometry_zero_by_version,
            public_geometry_current_by_version,
            phic,
        )
        row["final_spin_mod0"] = {"explicit_scalar": _scalar(fs0)}
        public_target0 = _lal_waveform(
            case, phic, convention=convention, final_spin_mod=0,
            physical_parameters=physical_parameters,
        )
        public_target3 = _lal_waveform(
            case, phic, convention=convention, final_spin_mod=3,
            physical_parameters=physical_parameters,
        )
        row["public_lal_target"] = {
            "mod0": {"hp_norm": _scalar(torch.linalg.vector_norm(public_target0[0])),
                     "hc_norm": _scalar(torch.linalg.vector_norm(public_target0[1]))},
            "mod3": {"hp_norm": _scalar(torch.linalg.vector_norm(public_target3[0])),
                     "hc_norm": _scalar(torch.linalg.vector_norm(public_target3[1]))},
        }

        public_angles_raw = _lal_angles(
            case, convention=convention, physical_parameters=physical_parameters
        )
        public_angles, transport = _transport_public_angles(
            public_angles_raw,
            public_geometry_zero,
            public_geometry_current,
            convention=convention,
            aligned=aligned,
        )
        if aligned:
            native_angles = _synthetic_aligned_angles()
            row["angle_status"] = (
                "public angles injected after source-frame transport; clean native angle cubic "
                "is degenerate, so its comparison uses synthetic aligned angles; "
                "the aligned convention-1 gamma branch is gauge-degenerate"
            )
        else:
            native_angles = _native_angles(case)
            row["angle_status"] = "public angles transported from phic=0; clean native angles evaluated"
        row["public_angle_samples"] = {
            "raw": _angle_summary(public_angles_raw),
            "transported": _angle_summary(public_angles),
            "transport": transport,
            "injected_as_absolute": True,
        }
        public_injected = _compose(
            case, phic, public_angles, mapped_geometry, fs0,
            convention=convention, absolute=True,
        )
        public_unshifted = _compose(
            case, phic, public_angles_raw, mapped_geometry, fs0,
            convention=convention, absolute=True,
        )
        public_with_fresh_geometry = _compose(
            case, phic, public_angles, geometry, fs0,
            convention=convention, absolute=True,
        )
        native_same_spin = _compose(
            case, phic, native_angles, geometry, fs0,
            convention=convention, absolute=False,
        )
        row["composition_vs_public_lal_mod0"] = {
            "public_angle_injection_transported_public_geometry": _metrics(public_injected, public_target0),
            "public_angle_injection_unshifted_public_geometry": _metrics(public_unshifted, public_target0),
            "public_angles_transported_fresh_geometry": _metrics(public_with_fresh_geometry, public_target0),
            "native_angles_same_mod0_spin": _metrics(native_same_spin, public_target0),
        }
        try:
            local_target0 = _torchwave_public(
                case, phic, convention=convention, final_spin_mod=0
            )
            row["composition_vs_torchwave_public_mod0"] = {
                "public_angle_injection_transported_public_geometry": _metrics(public_injected, local_target0),
                "public_angle_injection_unshifted_public_geometry": _metrics(public_unshifted, local_target0),
                "public_angles_transported_fresh_geometry": _metrics(public_with_fresh_geometry, local_target0),
                "native_angles_same_mod0_spin": _metrics(native_same_spin, local_target0),
            }
        except Exception as exc:
            row["torchwave_public_mod0_error"] = f"{type(exc).__name__}: {exc}"

        if case["name"] == "aligned_control":
            row["native_msa_final_spin"] = {
                "status": "not evaluated for degenerate aligned MSA roots"
            }
        else:
            native_fs, detail = _native_msa_final_spin(
                torch.tensor(case["mass1_msun"], dtype=torch.float64),
                torch.tensor(case["mass2_msun"], dtype=torch.float64),
                torch.tensor(case["chi1"], dtype=torch.float64),
                torch.tensor(case["chi2"], dtype=torch.float64),
                torch.tensor(F_REF, dtype=torch.float64),
                native_angles,
            )
            a_parallel = detail["aligned_kerr_parameter"]
            mf = detail["remnant_mass_fraction"]
            normalizations = msa_final_spin_normalization_report(a_parallel, mf)
            native_msa = _compose(
                case, phic, native_angles, geometry, native_fs,
                convention=convention, absolute=False,
            )
            paper_fs = native_fs * mf.square()
            native_paper = _compose(
                case, phic, native_angles, geometry, paper_fs,
                convention=convention, absolute=False,
            )
            row["native_msa_final_spin"] = {
                "status": "formula evaluated",
                "remnant_normalized": _scalar(native_fs),
                "paper_total_mass_normalized": _scalar(paper_fs),
                "aligned_kerr_parameter": _scalar(a_parallel),
                "remnant_mass_fraction": _scalar(mf),
                "s_av_squared": _scalar(detail["s_av_squared"]),
                "mean_s1_l": _scalar(detail["mean_s1_l"]),
                "mean_s2_l": _scalar(detail["mean_s2_l"]),
                "initial_s1_l": _scalar(detail["initial_s1_l"]),
                "initial_s2_l": _scalar(detail["initial_s2_l"]),
                "l_fin": _scalar(detail["l_fin"]),
                "l_fin_normalization": detail["l_fin_normalization"],
                "l_fin_spin_projection_source": "initial source-frame aligned spin projections",
                "normalization_aligned_limit": {
                    key: _scalar(value) for key, value in normalizations.items()
                },
                "public_scalar_mapping": "No documented scalar getter found; waveform comparison is black-box.",
            }
            row["composition_vs_public_lal_mod3"] = {
                "native_formula_remnant_normalized": _metrics(native_msa, public_target3),
                "native_formula_paper_total_mass_normalized": _metrics(native_paper, public_target3),
            }
            try:
                local_target3 = _torchwave_public(
                    case, phic, convention=convention, final_spin_mod=3
                )
                row["composition_vs_torchwave_public_mod3"] = {
                    "native_formula_remnant_normalized": _metrics(native_msa, local_target3),
                    "native_formula_paper_total_mass_normalized": _metrics(native_paper, local_target3),
                }
            except Exception as exc:
                row["torchwave_public_mod3_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    else:
        row["status"] = "complete"
    return row


def _version_isolation(case):
    if case["name"] == "aligned_control":
        return {"status": "skipped: aligned MSA cubic is degenerate"}
    rows = {}
    for version in VERSIONS:
        try:
            angles = _lal_angles(case, version=version, convention=0)
            rows[str(version)] = _angle_summary(angles)
        except Exception as exc:
            rows[str(version)] = {"error": f"{type(exc).__name__}: {exc}"}
    return rows


def _sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_provenance(module_names):
    result = {}
    for name in module_names:
        try:
            module = importlib.import_module(name)
            path = getattr(module, "__file__", None)
            result[name] = {"path": path, "sha256": _sha256(path) if path else None}
        except Exception as exc:
            result[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def _clean_source_hashes():
    names = (
        "precession_msa_waveform_clean.py",
        "precession_msa_angles_clean.py",
        "precession_msa_initial_clean.py",
        "precession_msa_pn_clean.py",
        "precession_msa_secular_clean.py",
        "precession_msa_oscillatory_clean.py",
        "precession_source_frame_clean.py",
        "precession_beta_clean.py",
        "precession_twisting_clean.py",
        "precession_nnlo_clean.py",
    )
    root = Path(__file__).resolve().parents[1] / "src/torchwave/approximants"
    return {name: _sha256(root / name) for name in names}


def main():
    rows = []
    for case in CASES:
        for phic in PHASES:
            for convention in (0, 1):
                print(
                    f"running case={case['name']} phic={phic} convention={convention}",
                    flush=True,
                )
                rows.append(_run_one(case, phic, convention))

    root = Path(__file__).resolve().parents[1]
    report_dir = root / "docs/reviews/2026-09-13-parity-review/verification/msa-composition"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "provenance": (
            "Bounded clean-room MSA composition diagnostic. Public angle and "
            "waveform functions are black-box references. The clean carrier "
            "is called with an explicit final-spin override, and fresh "
            "positive-frequency twisting performs the projection. Raw complex "
            "errors contain no phase, time, or norm correction."
        ),
        "paper_equations": {
            "appendix_C_NinLframe": "N_L=(sin(iota) sin(phiRef), sin(iota) cos(phiRef), cos(iota))",
            "appendix_C_JinLframe": "J/M^2=(Sx,Sy,L/M^2+Sz), S_i=(m1/M)^2 chi1_i+(m2/M)^2 chi2_i",
            "appendix_C_rotJtoL": "N_L=Rz(phiJ) Ry(thetaJL0) Rz(kappa) N_J; alpha0=pi-kappa; phiJ=-phiJL0",
            "geometry_l_inversion": (
                "Given Jhat and S, solve J=(Sx/jx) jhat (or Sy/jy) and "
                "L/M^2=|J| jz-Sz; the transverse residual is reported and "
                "the aligned J_perp=0 branch is left degenerate."
            ),
            "MSA_finalSpin": "<S_fin^2> = S_av^2 + L_fin^2 + 2 L_fin (<S1.L> + <S2.L>)",
            "afin_L_fin": "S_parallel + L_fin = M_fin^2 a_parallel",
            "aligned_fit_note": (
                "The text immediately following Eq. (afin_L_fin) states that "
                "the aligned a_parallel fit is constructed as a fit for "
                "L_fin; the native compatibility mode therefore consumes "
                "that scalar in initial-total-M^2 units."
            ),
            "implemented_l_fin": (
                "native compatibility path: the aligned fit is supplied as "
                "L_fin in initial-total-M^2 units, so L_fin/M^2 = "
                "a_parallel_fit - (S1.Lhat_initial/M^2 + "
                "S2.Lhat_initial/M^2); the MSA cross term retains the "
                "precession-averaged projections. The equation helper's "
                "explicit remnant mode separately implements "
                "(M_fin/M)^2*a_parallel."
            ),
            "finalspin_3_printed": "|a_fin| = sqrt(<S_fin^2>)/M^2",
            "normalization_note": (
                "The printed finalspin_3 denominator uses source M while "
                "afin_L_fin defines the Kerr parameter with M_fin. The report "
                "shows both explicit normalizations and their aligned-limit "
                "difference; no fitted bridge is used."
            ),
        },
        "api_parameters": {
            "composition": str(inspect.signature(compose_msa_waveform)),
            "native_wrapper": {
                "signature": "waveform(..., convention={0,1}, final_spin_mod=3, order=5, beta_pn=False)",
                "status": "experimental clean MSA composition; no public prec_version is claimed",
                "beta_prescription": "beta_pn=False: clean Newtonian orbital-angular-momentum cos(beta); beta_pn=True: clean PN beta",
                "final_spin_contract": (
                    "native mod-3 passes initial_s1_l/initial_s2_l to L_fin "
                    "and selects l_fin_normalization='initial'; the aligned "
                    "fit is interpreted in initial-total-M^2 units, while "
                    "mean_s1_l/mean_s2_l remain in the MSA cross term. The "
                    "equation helper retains its explicit remnant mode and "
                    "mean-for-both default when compatibility projections "
                    "are omitted."
                ),
            },
            "reference_version": {
                "status": "diagnostic metadata for black-box public calls only",
                "public_angle_versions": [220, 223, 224],
                "primary_injection_version": 223,
            },
            "geometry_diagnostic": {
                "public_versions": [220, 223, 224],
                "public_geometry_tuple": "(chi1,chi2,chip,thetaJN,alpha0,phiJL0,zeta)",
                "clean_source_l_reference": "source_frame_geometry uses orbital_angular_momentum(pn_version=102, compat_102=False)",
                "clean_l_candidates": [
                    "source_frame_102_printed",
                    "source_frame_102_compatibility",
                    "source_frame_full_printed",
                    "native_msa_newtonian_reference",
                ],
                "v223_effective_l": "conditional Appendix-C inversion only; not a public formula and not used by runtime",
                "aligned_branch": "J_perp=0 leaves L magnitude underdetermined by direction; the control reports this explicitly",
            },
            "public_angles": "SimIMRPhenomXPMSAAngles(frequency_vector, m1_SI, m2_SI, S1x,S1y,S1z,S2x,S2y,S2z, inclination, f_ref, mprime=2, lal.Dict)",
            "public_source_frame_geometry": "SimIMRPhenomXPCalculateModelParametersFromSourceFrame(m1_SI, m2_SI, f_ref, phiRef, incl, S1x,S1y,S1z,S2x,S2y,S2z, lal.Dict) -> (chi1,chi2,chip,thetaJN,alpha0,phiJL0,zeta)",
            "public_waveform": "SimInspiralChooseFDWaveform(..., deltaF=1, f_min=20, f_max=512, f_ref=20, lal.Dict, IMRPhenomXP)",
            "public_options": {
                "prec_version": [220, 223, 224],
                "angle_expansion_order": 5,
                "mprime": 2,
                "convention": [0, 1],
                "final_spin_mod": [0, 3],
            },
            "source_frame_transport": {
                "alpha": "alpha += alpha0(phiRef)-alpha0(0), wrapped",
                "gamma_convention1": "gamma -= phiJ(phiRef)-phiJ(0), wrapped",
                "phi_j": "phiJ=-phiJL0",
                "aligned_branch": "omit convention-1 gamma transport when transverse spins vanish; public phiJL0 is gauge-degenerate",
                "zeta": "map public convention-0 zeta by +pi/2 to the clean basis; public convention-1 metadata already contains that basis shift, so no extra metadata shift is applied",
                "source": "Pratten et al. arXiv:2004.06503v2 Appendix C and Table IV; no numerical reanchoring",
            },
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": getattr(torch, "__version__", None),
            "lal": getattr(lal, "__version__", None),
            "lalsimulation": getattr(ls, "__version__", None),
            "clean_source_sha256": _clean_source_hashes(),
            "dependency_modules": _module_provenance((
                "torch", "lal", "lalsimulation",
                "torchwave.approximants.imrphenom_xas",
            )),
            "diagnostic_script_sha256": _sha256(Path(__file__).resolve()),
            "primary_paper_sha256": _sha256("/tmp/pratten2021/phenomx.tex"),
        },
        "grid": {
            "flow_hz": FLOW,
            "fhigh_hz": FHIGH,
            "delta_f_hz": 1.0,
            "count": int(FREQUENCIES.numel()),
            "distance_mpc": DISTANCE,
            "inclination_rad": INCLINATION,
            "phic_rad": list(PHASES),
        },
        "public_msa_version_samples": {
            case["name"]: _version_isolation(case) for case in CASES
        },
        "rotation_identity_checks": _rotation_identity_checks(),
        "rows": rows,
    }
    out = report_dir / "report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    print(json.dumps({"rows": len(rows), "report": str(out)}, indent=2))


if __name__ == "__main__":
    main()
