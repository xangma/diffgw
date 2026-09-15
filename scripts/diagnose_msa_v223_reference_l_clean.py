"""Bounded clean-room audit of the public v223 reference-frame ``L``.

The public ``SimIMRPhenomXPCalculateModelParametersFromSourceFrame`` call is
used only as a black-box geometry reference.  Its documented Appendix-C
tuple is inverted to a direction ``Jhat``; with the independently supplied
spin sum, the transverse components give a *conditional* effective
``L/M**2``.  That scalar is diagnostic evidence, not a runtime formula.

The clean candidates evaluated are equations printed in Pratten et al.,
arXiv:2004.06503v2 plus the older-spin-variable 3PN Eq. (6.9) of Blanchet,
Buonanno and Faye, arXiv:gr-qc/0605140v4.  No coefficients are fitted to the
public result.
The report intentionally stops if these published alternatives do not close
the v223 geometry.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import platform
import sys
from pathlib import Path

import lal
import lalsimulation as ls
import torch

from torchwave.approximants.precession_msa_reference_l_clean import (
    orbital_angular_momentum_candidates,
)
from torchwave.approximants.precession_source_frame_clean import (
    source_frame_geometry,
)


INCLINATION = 0.7
PHASES = (0.0, 0.2)
VERSIONS = (220, 223, 224)
CONVENTIONS = (0, 1)
EXPANSION_ORDER = 5

CASES = (
    {
        "name": "A",
        "mass1_msun": 35.0,
        "mass2_msun": 25.0,
        "chi1": (0.2, 0.1, 0.2),
        "chi2": (-0.1, 0.15, -0.1),
        "f_refs_hz": (20.0, 40.0, 80.0, 160.0, 320.0),
    },
    {
        "name": "B",
        "mass1_msun": 45.0,
        "mass2_msun": 15.0,
        "chi1": (0.45, -0.2, 0.3),
        "chi2": (0.1, 0.2, -0.25),
        "f_refs_hz": (20.0, 40.0, 80.0, 160.0, 320.0),
    },
    {
        "name": "C",
        "mass1_msun": 30.0,
        "mass2_msun": 10.0,
        "chi1": (0.12, -0.31, 0.44),
        "chi2": (-0.22, 0.18, -0.07),
        "f_refs_hz": (20.0, 40.0, 80.0, 160.0, 320.0),
    },
    {
        "name": "D_new_probe",
        "mass1_msun": 60.0,
        "mass2_msun": 20.0,
        "chi1": (0.31, 0.17, -0.2),
        "chi2": (-0.14, 0.09, 0.13),
        "f_refs_hz": (25.0, 75.0, 180.0),
    },
    {
        "name": "aligned_control",
        "mass1_msun": 35.0,
        "mass2_msun": 25.0,
        "chi1": (0.0, 0.0, 0.2),
        "chi2": (0.0, 0.0, -0.1),
        "f_refs_hz": (20.0, 100.0),
    },
)


def _sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dict_options(*, version, convention, final_spin_mod=0):
    params = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(params, int(version))
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


def _looked_up_options(params):
    return {
        "prec_version": int(ls.SimInspiralWaveformParamsLookupPhenomXPrecVersion(params)),
        "expansion_order": int(ls.SimInspiralWaveformParamsLookupPhenomXPExpansionOrder(params)),
        "convention": int(ls.SimInspiralWaveformParamsLookupPhenomXPConvention(params)),
        "final_spin_mod": int(ls.SimInspiralWaveformParamsLookupPhenomXPFinalSpinMod(params)),
    }


def _public_physical_inputs(case):
    """Construct one SI/spin record shared by every public call."""
    return {
        "mass1_si": float(case["mass1_msun"]) * lal.MSUN_SI,
        "mass2_si": float(case["mass2_msun"]) * lal.MSUN_SI,
        "spin1": tuple(float(value) for value in case["chi1"]),
        "spin2": tuple(float(value) for value in case["chi2"]),
    }


def _public_source_geometry(case, f_ref, phic, *, version, convention):
    physical = _public_physical_inputs(case)
    params = _dict_options(version=version, convention=convention)
    values = ls.SimIMRPhenomXPCalculateModelParametersFromSourceFrame(
        physical["mass1_si"],
        physical["mass2_si"],
        float(f_ref),
        float(phic),
        INCLINATION,
        *physical["spin1"],
        *physical["spin2"],
        params,
    )
    return {
        "tuple": [float(value) for value in values],
        "semantics": [
            "chi1", "chi2", "chip", "thetaJN", "alpha0", "phiJL0", "zeta"
        ],
        "call": {
            "function": (
                "SimIMRPhenomXPCalculateModelParametersFromSourceFrame"
            ),
            "mass1_si": physical["mass1_si"],
            "mass2_si": physical["mass2_si"],
            "f_ref_hz": float(f_ref),
            "phiRef": float(phic),
            "inclination": INCLINATION,
            "spin1": list(physical["spin1"]),
            "spin2": list(physical["spin2"]),
            "options_requested": {
                "prec_version": int(version),
                "expansion_order": EXPANSION_ORDER,
                "convention": int(convention),
                "final_spin_mod": 0,
            },
            "options_looked_up": _looked_up_options(params),
        },
    }


def _rotation_z(angle, vector):
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = vector
    return c * x - s * y, s * x + c * y, z


def _rotation_y(angle, vector):
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = vector
    return c * x + s * z, y, -s * x + c * z


def _vector_norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def _vector_difference(first, second):
    return tuple(a - b for a, b in zip(first, second))


def _wrapped_difference(first, second, period=2.0 * math.pi):
    return (float(first) - float(second) + period / 2.0) % period - period / 2.0


def _normalized_spin_sum(case):
    total = float(case["mass1_msun"]) + float(case["mass2_msun"])
    n1 = float(case["mass1_msun"]) / total
    n2 = float(case["mass2_msun"]) / total
    return tuple(
        n1 * n1 * float(case["chi1"][index])
        + n2 * n2 * float(case["chi2"][index])
        for index in range(3)
    )


def _clean_geometry(case, f_ref, phic):
    values = source_frame_geometry(
        case["mass1_msun"], case["mass2_msun"],
        *case["chi1"], *case["chi2"],
        INCLINATION, float(phic), float(f_ref),
    )
    return tuple(float(value) for value in values)


def _appendix_c_public_jhat(public_geometry, phic):
    """Invert Appendix C's documented tuple to ``Jhat``."""
    values = public_geometry["tuple"]
    theta_jn, alpha0, phi_jl0 = float(values[3]), float(values[4]), float(values[5])
    phi_j = -phi_jl0
    kappa = math.pi - alpha0
    n_l = (
        math.sin(INCLINATION) * math.sin(float(phic)),
        math.sin(INCLINATION) * math.cos(float(phic)),
        math.cos(INCLINATION),
    )
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
    reconstructed = _rotation_z(phi_j, _rotation_y(theta_jl0, v))
    return {
        "theta_jl0": theta_jl0,
        "phi_j": phi_j,
        "kappa": kappa,
        "jhat": list(jhat),
        "rotation_y_residual": u[1] - v[1],
        "n_l_reconstruction_l2": _vector_norm(_vector_difference(reconstructed, n_l)),
        "jhat_norm_error": abs(_vector_norm(jhat) - 1.0),
    }


def _effective_l_from_jhat(spin_sum, jhat):
    """Conditionally solve ``J=S+L*z`` from a non-aligned ``Jhat``."""
    jx, jy, jz = jhat
    sx, sy, sz = spin_sum
    j_perp_sq = jx * jx + jy * jy
    if j_perp_sq <= 1.0e-28:
        return {
            "status": "aligned_branch_degenerate",
            "l_over_m2": None,
            "j_over_m2": None,
            "j_magnitude_over_m2": None,
            "transverse_component_residual": None,
        }
    j_magnitude = (sx * jx + sy * jy) / j_perp_sq
    inferred_l = j_magnitude * jz - sz
    residual = max(abs(sx - j_magnitude * jx), abs(sy - j_magnitude * jy))
    return {
        "status": "conditional_geometric_inversion",
        "l_over_m2": inferred_l,
        "j_over_m2": [j_magnitude * jx, j_magnitude * jy, j_magnitude * jz],
        "j_magnitude_over_m2": j_magnitude,
        "transverse_component_residual": residual,
    }


def _clean_j_vector(spin_sum, l_over_m2):
    vector = (spin_sum[0], spin_sum[1], spin_sum[2] + l_over_m2)
    norm = _vector_norm(vector)
    return {
        "l_over_m2": float(l_over_m2),
        "j_over_m2": list(vector),
        "j_magnitude_over_m2": norm,
        "jhat": [value / norm for value in vector],
    }


def _clean_candidates(case, f_ref):
    candidates = orbital_angular_momentum_candidates(
        case["mass1_msun"], case["mass2_msun"], f_ref,
        case["chi1"][2], case["chi2"][2],
    )
    return {
        name: float(value.detach().cpu()) for name, value in candidates.items()
    }


def _geometry_record(case, f_ref, phic, *, version, convention):
    public = _public_source_geometry(
        case, f_ref, phic, version=version, convention=convention,
    )
    fresh = _clean_geometry(case, f_ref, phic)
    public_inverse = _appendix_c_public_jhat(public, phic)
    spin_sum = _normalized_spin_sum(case)
    effective = _effective_l_from_jhat(spin_sum, public_inverse["jhat"])
    candidates = _clean_candidates(case, f_ref)
    candidate_records = {}
    for name, value in candidates.items():
        clean_j = _clean_j_vector(spin_sum, value)
        entry = {
            "l_over_m2": value,
            "j_over_m2": clean_j["j_over_m2"],
            "jhat": clean_j["jhat"],
            "jhat_l2_vs_public": _vector_norm(
                _vector_difference(public_inverse["jhat"], clean_j["jhat"])
            ),
        }
        if effective["l_over_m2"] is not None:
            entry["l_delta_public_minus_candidate"] = (
                effective["l_over_m2"] - value
            )
        candidate_records[name] = entry

    # Convention 0 public zeta is in the source-frame basis; convention 1's
    # metadata already contains the Appendix-C basis shift.  This mapping is
    # only for a geometry comparison and is never sent back to the runtime.
    public_tuple = public["tuple"]
    zeta_shift = math.pi / 2.0 if int(convention) == 0 else 0.0
    mapped_public = (
        public_tuple[4], public_tuple[3], public_tuple[6] + zeta_shift,
        -public_tuple[5],
    )
    fresh_vs_public = {
        "alpha0_wrapped": _wrapped_difference(fresh[0], public_tuple[4]),
        "theta_jn": fresh[1] - public_tuple[3],
        "zeta_wrapped": _wrapped_difference(fresh[2], public_tuple[6]),
        "phi_j_vs_neg_phiJL0_wrapped": _wrapped_difference(fresh[3], -public_tuple[5]),
    }
    fresh_vs_mapped = {
        "alpha0_wrapped": _wrapped_difference(fresh[0], mapped_public[0]),
        "theta_jn": fresh[1] - mapped_public[1],
        "zeta_wrapped": _wrapped_difference(fresh[2], mapped_public[2]),
        "zeta_wrapped_mod_pi": _wrapped_difference(fresh[2], mapped_public[2], math.pi),
        "phi_j_wrapped": _wrapped_difference(fresh[3], mapped_public[3]),
    }
    return {
        "case": case["name"],
        "f_ref_hz": float(f_ref),
        "phiRef": float(phic),
        "version": int(version),
        "convention": int(convention),
        "physical_spin_record": {
            "mass1_msun": float(case["mass1_msun"]),
            "mass2_msun": float(case["mass2_msun"]),
            "spin1": list(case["chi1"]),
            "spin2": list(case["chi2"]),
            "source_total_mass_spin_sum_over_m2": list(spin_sum),
        },
        "public_geometry": public,
        "fresh_geometry": {
            "alpha0": fresh[0],
            "theta_jn": fresh[1],
            "zeta": fresh[2],
            "phi_j": fresh[3],
        },
        "fresh_vs_public_geometry": fresh_vs_public,
        "fresh_vs_mapped_public_geometry": {
            "public_zeta_basis_shift": zeta_shift,
            "mapped_public": list(mapped_public),
            **fresh_vs_mapped,
        },
        "public_appendix_c_inverse": public_inverse,
        "effective_public_l": effective,
        "clean_candidates": candidate_records,
        "interpretation": (
            "effective_public_l is a conditional inversion of the returned "
            "Appendix-C direction and spin sum; it is not a public formula."
        ),
    }


def _transport_summary(rows):
    grouped = {}
    for row in rows:
        key = (row["case"], row["f_ref_hz"], row["version"], row["convention"])
        grouped.setdefault(key, {})[row["phiRef"]] = row
    summary = []
    for key, values in sorted(grouped.items()):
        zero, current = values.get(0.0), values.get(0.2)
        if zero is None or current is None:
            continue
        zero_j = zero["public_appendix_c_inverse"]["jhat"]
        current_j = current["public_appendix_c_inverse"]["jhat"]
        result = {
            "case": key[0], "f_ref_hz": key[1], "version": key[2],
            "convention": key[3],
            "public_jhat_phic_delta_l2": _vector_norm(
                _vector_difference(current_j, zero_j)
            ),
        }
        l0 = zero["effective_public_l"]["l_over_m2"]
        l1 = current["effective_public_l"]["l_over_m2"]
        result["public_effective_l_delta"] = None if l0 is None or l1 is None else l1 - l0
        result["aligned_branch"] = zero["effective_public_l"]["status"] == "aligned_branch_degenerate"
        summary.append(result)
    return summary


def _frequency_summary(rows):
    summary = []
    for case in CASES:
        for version in VERSIONS:
            selected = [
                row for row in rows
                if row["case"] == case["name"]
                and row["version"] == version
                and row["convention"] == 0
                and row["phiRef"] == 0.0
            ]
            candidate_names = tuple(
                selected[0]["clean_candidates"].keys()
            ) if selected else ()
            max_errors = {}
            for name in candidate_names:
                values = [
                    row["clean_candidates"][name].get("l_delta_public_minus_candidate")
                    for row in selected
                ]
                values = [value for value in values if value is not None]
                max_errors[name] = max((abs(value) for value in values), default=None)
            summary.append({
                "case": case["name"],
                "version": version,
                "phic": 0.0,
                "f_refs_hz": list(case["f_refs_hz"]),
                "public_effective_l": [
                    row["effective_public_l"]["l_over_m2"] for row in selected
                ],
                "conditional_inversion_residual_max": max((
                    row["effective_public_l"]["transverse_component_residual"]
                    for row in selected
                    if row["effective_public_l"]["transverse_component_residual"] is not None
                ), default=None),
                "candidate_max_abs_l_delta": max_errors,
            })
    return summary


def _candidate_delta_summary(rows, name, *, version=223):
    """Summarize one candidate without treating the public inversion as a fit."""
    selected = [
        row for row in rows
        if row["version"] == version
        and row["clean_candidates"].get(name, {}).get(
            "l_delta_public_minus_candidate"
        ) is not None
    ]
    if not selected:
        return {"status": "no_nondegenerate_rows"}
    worst = max(
        selected,
        key=lambda row: abs(
            row["clean_candidates"][name]["l_delta_public_minus_candidate"]
        ),
    )
    fresh = [row for row in selected if row["case"] == "D_new_probe"]
    return {
        "status": "comparison_only",
        "version": version,
        "rows": len(selected),
        "max_abs_l_delta_public_minus_candidate": max(
            abs(row["clean_candidates"][name]["l_delta_public_minus_candidate"])
            for row in selected
        ),
        "fresh_case": "D_new_probe",
        "fresh_case_max_abs_l_delta_public_minus_candidate": max(
            (abs(row["clean_candidates"][name]["l_delta_public_minus_candidate"])
             for row in fresh),
            default=None,
        ),
        "worst_row": {
            "case": worst["case"],
            "f_ref_hz": worst["f_ref_hz"],
            "phiRef": worst["phiRef"],
            "convention": worst["convention"],
            "signed_l_delta_public_minus_candidate": worst["clean_candidates"][name][
                "l_delta_public_minus_candidate"
            ],
        },
    }


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


def main():
    rows = []
    for case in CASES:
        for f_ref in case["f_refs_hz"]:
            for phic in PHASES:
                for version in VERSIONS:
                    for convention in CONVENTIONS:
                        print(
                            f"running case={case['name']} f_ref={f_ref} "
                            f"phic={phic} version={version} convention={convention}",
                            flush=True,
                        )
                        rows.append(_geometry_record(
                            case, f_ref, phic,
                            version=version, convention=convention,
                        ))

    root = Path(__file__).resolve().parents[1]
    report_dir = root / "docs/reviews/2026-09-13-parity-review/verification/msa-composition"
    report_dir.mkdir(parents=True, exist_ok=True)
    paper_paths = (
        "/tmp/pratten2021/phenomx.tex",
        "/tmp/torchwave-clean-room-precession/1809.10113-phenompv3.tex",
        "/tmp/torchwave-clean-room-precession/1703.03967v3-paper.tex",
        "/tmp/msa-paper-cache/0605140.txt",
    )
    nonconserved_summary = _candidate_delta_summary(
        rows, "three_pn_spin_nonconserved",
    )
    report = {
        "provenance": (
            "Bounded clean-room public geometry audit. The LALSimulation "
            "source-frame metadata function is a black-box reference. "
            "Appendix-C rotations and the conditional J=S+L*z inversion "
            "are implemented independently; no coefficients, phase, time, "
            "norm, or residual correction is fitted."
        ),
        "finding": {
            "status": "no_published_v223_reference_l_formula_identified",
            "conclusion": (
                "The rendered primary papers specify 3PN spinning L for the "
                "beta prescription and describe v223 as an MSA-angle variant, "
                "but do not publish a v223-specific source-frame L or phi_z,0 "
                "formula. The older non-conserved-spin 3PN Eq. (6.9) candidate "
                "reduces the v223 scalar mismatch but does not close it on the "
                "full grid, so it remains a named diagnostic candidate rather "
                "than being wired into clean geometry."
            ),
            "unresolved_contract": (
                "A reproducible v223 source-frame reference requires the "
                "undocumented MSA phi_z,0/zeta_z,0 initialization or an "
                "equivalent published J/L prescription. The direct Eq. (6.9) "
                "transcription remains frequency-dependent in the residual "
                "(see nonconserved_candidate_result). Supplying an effective "
                "L inferred from public angles would be an oracle substitute "
                "and is intentionally not implemented."
            ),
            "nonconserved_candidate_result": {
                "equation": "gr-qc/0605140v4 Eq. (6.9)",
                "status": "reduces_but_does_not_resolve",
                **nonconserved_summary,
                "interpretation": (
                    "The direct published transcription is close at the low "
                    "frequency end and diverges increasingly with frequency; "
                    "the remaining convention is not identified by this paper."
                ),
            },
        },
        "paper_equations": {
            "L4PN": (
                "L/M^2=eta/sqrt(x)[L0+L1*x+L2*x^2+L3*x^3+L4*x^4 "
                "+L1.5SO*x^1.5+L2.5SO*x^2.5+L3.5SO*x^3.5]"
            ),
            "LO_S_infinity": (
                "Appendix-G L2^(LO-Sinf) and L3.5^(LO-Sinf) are added inside "
                "the same eta/sqrt(x) bracket."
            ),
            "beta_choice": (
                "Pv3 paper: highest available 3PN |L| including spin-orbit "
                "terms is selected for beta; in-plane spin components are "
                "neglected in this aligned-spin L approximation."
            ),
            "msa_dynamics": (
                "The original MSA paper uses Newtonian L=eta/v for the clean "
                "MSA dynamics; it does not state a public v223 frame gauge."
            ),
            "old_spin_variables": (
                "Blanchet-Buonanno-Faye Eq. (6.9) gives the circular aligned "
                "3PN series in their non-conserved spin variables, with the "
                "2.5PN spin-orbit coefficients -77/8+259*eta/72 and "
                "-33/8+7*eta/4."
            ),
            "appendix_C": (
                "J/M^2=(Sx,Sy,L/M^2+Sz), N_L=Rz(phiJ)Ry(thetaJL0)Rz(kappa)N_J, "
                "alpha0=pi-kappa, phiJ=-phiJL0."
            ),
            "source_locations": {
                "/tmp/pratten2021/phenomx.tex": ["Sec. PN_L", "Appendix G", "Table prec_version"],
                "/tmp/torchwave-clean-room-precession/1809.10113-phenompv3.tex": ["Sec. Upgrading to Pv3", "beta choice"],
                "/tmp/torchwave-clean-room-precession/1703.03967v3-paper.tex": ["Sec. Precession", "Newtonian L=eta/v"],
            },
        },
        "api_contract": {
            "public_source_frame_geometry": (
                "SimIMRPhenomXPCalculateModelParametersFromSourceFrame(" 
                "m1_SI,m2_SI,f_ref,phiRef,inclination,S1x,S1y,S1z," 
                "S2x,S2y,S2z,lal.Dict) -> "
                "(chi1,chi2,chip,thetaJN,alpha0,phiJL0,zeta)"
            ),
            "options": {
                "prec_version": list(VERSIONS),
                "expansion_order": EXPANSION_ORDER,
                "convention": list(CONVENTIONS),
                "final_spin_mod": 0,
            },
            "public_inputs_are_shared": True,
            "clean_candidates": [
                "three_pn_spin", "three_pn_spin_nonconserved", "four_pn_spin",
                "four_pn_spin_losinfty", "native_msa_newtonian",
            ],
            "effective_l_status": (
                "conditional inversion from public Jhat and the exact input "
                "spin vectors; aligned J_perp=0 is explicitly degenerate"
            ),
        },
        "cases": [
            {
                key: value for key, value in case.items()
            } for case in CASES
        ],
        "grid": {
            "inclination_rad": INCLINATION,
            "phiRef_rad": list(PHASES),
            "versions": list(VERSIONS),
            "conventions": list(CONVENTIONS),
            "angle_expansion_order": EXPANSION_ORDER,
        },
        "frequency_summary": _frequency_summary(rows),
        "phiRef_transport_summary": _transport_summary(rows),
        "rows": rows,
        "external_waveforms": [],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": getattr(torch, "__version__", None),
            "lal": getattr(lal, "__version__", None),
            "lalsimulation": getattr(ls, "__version__", None),
            "diagnostic_script_sha256": _sha256(Path(__file__).resolve()),
            "paper_sha256": {path: _sha256(path) for path in paper_paths},
            "dependency_modules": _module_provenance((
                "torch", "lal", "lalsimulation",
                "torchwave.approximants.precession_msa_reference_l_clean",
                "torchwave.approximants.precession_beta_clean",
                "torchwave.approximants.precession_source_frame_clean",
            )),
        },
    }
    output = report_dir / "v223-reference-l-diagnostic.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
