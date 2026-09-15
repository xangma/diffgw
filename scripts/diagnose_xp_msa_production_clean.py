"""Public-binary diagnostic for the clean production IMRPhenomXP MSA route.

The diagnostic first checks that the public IMRPhenomXP binary gives the same
waveform when its precession/remnant choices are omitted as when the audited
v223, convention-1, MSA-remnant tuple is explicit. It then compares the
unchanged frozen rows, their separately derived explicit-tuple copies, and
fresh probes against that public binary. No phase, time, amplitude, or
residual fit is applied. The small in-plane-spin probe remains visible as a
known failure while the helper's pole boundary is under review.
"""

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from parity_harness import (
    PARITY_GATE,
    get_canonical_catalog_cases,
    get_canonical_holdout_cases,
    run_oracle_comparison,
)


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "src" / "torchwave" / "approximants" / "precession_msa_waveform_clean.py"
DEFAULT_OUTPUT = (
    ROOT / "docs" / "reviews" / "2026-09-13-parity-review" / "verification"
    / "xp-msa-production-clean" / "diagnostic.json"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_case(case, case_id=None):
    """Copy case metadata without changing its physical parameters."""
    result = copy.deepcopy(case)
    if case_id is not None:
        result["id"] = case_id
    result["torch_name"] = "IMRPhenomXP"
    result["lal_name"] = "IMRPhenomXP"
    # Keep the repository-wide strict parity gate unchanged.
    result["expected_gate"] = copy.deepcopy(PARITY_GATE)
    return result


def explicit_msa_tuple_case(case, case_id, convention=1):
    """Return a derived copy with the audited production tuple made explicit."""
    result = _copy_case(case, case_id)
    result["params"].update(
        prec_version=223, convention=convention, final_spin_mod=3
    )
    return result


def _without_routing_options(params):
    result = copy.deepcopy(params)
    for name in ("precession_method", "prec_version", "convention",
                 "final_spin_mod", "final_spin_version"):
        result.pop(name, None)
    return result


def _public_lal_xp(case, params):
    """Call only the public LALSimulation IMRPhenomXP binary."""
    import lal
    import lalsimulation as lalsim

    p = copy.deepcopy(params)
    grid = case["grid"]
    flow = float(grid["flow"])
    fhigh = float(grid["fhigh"])
    df = float(grid["df"])
    fref_value = p.get("f_ref", flow)
    fref = flow if fref_value is None else float(fref_value)
    oracle_fhigh = float(case.get("oracle_fhigh", fhigh))

    extra = lal.CreateDict()
    if p.get("prec_version") is not None:
        lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(
            extra, int(p["prec_version"])
        )
    if p.get("final_spin_mod") is not None:
        lalsim.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(
            extra, int(p["final_spin_mod"])
        )
    if p.get("convention") is not None:
        lalsim.SimInspiralWaveformParamsInsertPhenomXPConvention(
            extra, int(p["convention"])
        )

    m1_si = float(p["mass1"]) * lal.MSUN_SI
    m2_si = float(p["mass2"]) * lal.MSUN_SI
    s1x = float(p.get("spin1x", 0.0))
    s1y = float(p.get("spin1y", 0.0))
    s1z = float(p.get("spin1z", p.get("chi1", 0.0)))
    s2x = float(p.get("spin2x", 0.0))
    s2y = float(p.get("spin2y", 0.0))
    s2z = float(p.get("spin2z", p.get("chi2", 0.0)))
    distance_si = float(p.get("distance", 100.0)) * 1e6 * lal.PC_SI
    inclination = float(p.get("inclination", 0.0))
    phic = float(p.get("phic", 0.0))
    ecc = float(p.get("e0", 0.0))
    approximant = lalsim.GetApproximantFromString("IMRPhenomXP")
    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1_si, m2_si, s1x, s1y, s1z, s2x, s2y, s2z, distance_si,
        inclination, phic, 0.0, ecc, 0.0, df, flow, oracle_fhigh, fref,
        extra, approximant,
    )
    return np.asarray(hp.data.data), np.asarray(hc.data.data), df, flow, fhigh


def binary_option_equivalence(case):
    """Compare omitted public options with the explicit audited tuple."""
    default_params = _without_routing_options(case.get("params", {}))
    explicit_params = copy.deepcopy(default_params)
    explicit_params.update(prec_version=223, convention=1, final_spin_mod=3)
    result = {
        "case_id": case.get("id"),
        "default_params": default_params,
        "explicit_params": explicit_params,
        "options_equivalent": False,
    }
    try:
        default_hp, default_hc, df, flow, fhigh = _public_lal_xp(
            case, default_params
        )
        explicit_hp, explicit_hc, _, _, _ = _public_lal_xp(
            case, explicit_params
        )
        freqs = np.arange(default_hp.size, dtype=np.float64) * df
        in_band = (freqs >= flow) & (freqs < fhigh)
        plus_delta = default_hp[in_band] - explicit_hp[in_band]
        cross_delta = default_hc[in_band] - explicit_hc[in_band]
        plus_norm = np.linalg.norm(explicit_hp[in_band])
        cross_norm = np.linalg.norm(explicit_hc[in_band])
        plus_l2 = float(np.linalg.norm(plus_delta) / plus_norm) if plus_norm else 0.0
        cross_l2 = float(np.linalg.norm(cross_delta) / cross_norm) if cross_norm else 0.0
        options_equivalent = bool(
            np.array_equal(default_hp, explicit_hp)
            and np.array_equal(default_hc, explicit_hc)
        )
        result.update(
            status="PASSED" if options_equivalent else "FAILED",
            options_equivalent=options_equivalent,
            plus_complex_relative_l2=plus_l2,
            cross_complex_relative_l2=cross_l2,
            plus_max_abs=float(np.max(np.abs(plus_delta), initial=0.0)),
            cross_max_abs=float(np.max(np.abs(cross_delta), initial=0.0)),
            sample_count=int(default_hp.size),
        )
    except Exception as exc:
        result.update(status="FAILED", reason=f"Public binary generation error: {exc}")
    return result


def fresh_cases():
    common = dict(
        distance=500.0,
        inclination=0.7,
        phic=0.3,
        tc=0.0,
        f_ref=20.0,
    )
    probes = [
        (
            "fresh_generic",
            dict(mass1=36.0, mass2=22.0, spin1x=0.17, spin1y=-0.11,
                 spin1z=0.31, spin2x=-0.08, spin2y=0.14, spin2z=-0.22),
        ),
        (
            "fresh_equal_mass",
            dict(mass1=25.0, mass2=25.0, spin1x=0.20, spin1y=0.10,
                 spin1z=0.0, spin2x=-0.05, spin2y=0.12, spin2z=-0.1),
        ),
        (
            "fresh_aligned",
            dict(mass1=35.0, mass2=25.0, spin1x=0.0, spin1y=0.0,
                 spin1z=0.30, spin2x=0.0, spin2y=0.0, spin2z=-0.20),
        ),
        (
            "fresh_zero_spin",
            dict(mass1=35.0, mass2=25.0, spin1x=0.0, spin1y=0.0,
                 spin1z=0.0, spin2x=0.0, spin2y=0.0, spin2z=0.0),
        ),
        (
            "fresh_small_in_plane_known_failure",
            dict(mass1=35.0, mass2=25.0, spin1x=1.0e-5, spin1y=0.0,
                 spin1z=0.20, spin2x=0.0, spin2y=1.0e-5, spin2z=-0.10),
        ),
    ]
    cases = []
    for label, spins in probes:
        params = dict(common)
        params.update(spins)
        cases.append({
            "id": label,
            "torch_name": "IMRPhenomXP",
            "lal_name": "IMRPhenomXP",
            "params": params,
            "grid": {"flow": 20.0, "fhigh": 512.0, "df": 0.25},
            "expected_gate": copy.deepcopy(PARITY_GATE),
            "oracle_type": "lal",
            "oracle_api": "SimInspiralChooseFDWaveform",
            "oracle_fhigh": 512.0,
        })
    return cases


def _annotate_and_run(cases, role):
    rows = []
    for case in cases:
        row = run_oracle_comparison(case, strict=True)
        row["case_role"] = role
        row["known_small_spin_boundary_probe"] = (
            "small_in_plane_known_failure" in case["id"]
        )
        rows.append(row)
    return rows


def main(output=DEFAULT_OUTPUT):
    before = sha256(HELPER)

    # These are fresh copies supplied by the harness. Keep their parameters
    # unchanged; explicit routing probes are derived copies below.
    frozen_catalog_source = next(
        case for case in get_canonical_catalog_cases()
        if case["torch_name"] == "IMRPhenomXP"
    )
    frozen_holdout_source = next(
        case for case in get_canonical_holdout_cases()
        if case["torch_name"] == "IMRPhenomXP"
    )
    frozen_cases = [
        _copy_case(frozen_catalog_source, "frozen_catalog_xp_unchanged"),
        _copy_case(frozen_holdout_source, "frozen_holdout_xp_unchanged"),
    ]
    frozen_explicit_cases = [
        explicit_msa_tuple_case(frozen_catalog_source,
                                "derived_catalog_xp_explicit_223_conv1_final3"),
        explicit_msa_tuple_case(frozen_holdout_source,
                                "derived_holdout_xp_explicit_223_conv1_final3"),
    ]
    fresh_source_cases = fresh_cases()
    fresh_production_cases = []
    for case in fresh_source_cases:
        # Keep the generic and equal-mass convention-0 probes compact, while
        # exercising both public conventions at the exact aligned/zero-spin
        # chart boundaries.  Every entry remains a derived copy; the source
        # cases above are never mutated.
        conventions = (0, 1) if case["id"] in {
            "fresh_aligned", "fresh_zero_spin"
        } else (0,)
        for convention in conventions:
            fresh_production_cases.append(
                explicit_msa_tuple_case(
                    case,
                    f"{case['id']}_explicit_223_conv{convention}_final3",
                    convention=convention,
                )
            )

    equivalence_inputs = frozen_cases + fresh_source_cases
    equivalence = [binary_option_equivalence(case) for case in equivalence_inputs]

    # Run the unchanged frozen rows first, then keep derived explicit tuples
    # and fresh defaults in separate report sections for auditability.
    frozen_results = _annotate_and_run(frozen_cases, "frozen_unchanged")
    frozen_explicit_results = _annotate_and_run(
        frozen_explicit_cases, "derived_explicit_223_conv1_final3"
    )
    fresh_production_results = []
    for case in fresh_production_cases:
        convention = case["params"].get("convention")
        fresh_production_results.extend(
            _annotate_and_run(
                [case], f"derived_explicit_223_conv{convention}_final3"
            )
        )

    after = sha256(HELPER)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "provenance": (
            "Public LALSimulation binary comparison for production IMRPhenomXP. "
            "Omitted public precession/remnant choices are checked against the "
            "explicit v223, convention-1, final-spin-3 tuple; waveform metrics "
            "are direct complex comparisons with no residual or fitted correction."
        ),
        "python": sys.version,
        "helper": str(HELPER.relative_to(ROOT)),
        "helper_sha256_before": before,
        "helper_sha256_after": after,
        "helper_stable_during_run": before == after,
        "parity_gate": copy.deepcopy(PARITY_GATE),
        "known_small_spin_status": (
            "visible_expected_failure; do not claim full-domain parity"
        ),
        "exact_convention1_boundary_status": (
            "ordinary aligned/zero-spin rows use clean v223; their convention-1 "
            "chart differences remain visible in fresh strict results"
        ),
        "extreme_guard_status": (
            "L+S_parallel<=0 uses the row-local legacy fallback; strict public "
            "parity is not claimed for this unsupported guard"
        ),
        "binary_default_vs_explicit_223_conv1_final3": equivalence,
        "frozen_unchanged_cases": frozen_results,
        "derived_explicit_tuple_cases": frozen_explicit_results,
        "fresh_production_cases": fresh_production_results,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    raise SystemExit(main(destination))
