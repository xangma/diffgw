"""Frozen paper-derived (2,1) residual discriminator; no LAL parity claim."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import mpmath as mp

from .reference.flux import (
    factorized_21_rho_hybrid_amplitude, factorized_21_spin_amplitude,
)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _decimal(value: mp.mpf) -> str:
    return mp.nstr(value, 30)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text())
    if plan["schema"] != "diffgw.seobnrv4.mode-21-rho-hybrid-plan.v1":
        parser.error("incompatible frozen plan")
    audit_path = args.plan.parent / plan["source_audit"]
    audit = json.loads(audit_path.read_text())
    if audit["upstream_tag"] != "lalsimulation-v6.2.1":
        parser.error("incompatible source audit")
    reference_path = Path(__file__).resolve().parent / "reference" / "flux.py"
    threshold = mp.mpf(plan["control_max_absolute_difference"])
    rows = []
    controls_pass = True
    differences_present = True

    with mp.workdps(plan["precision_decimal_digits"]):
        for case in plan["cases"]:
            m1, m2, chi1, chi2 = (mp.mpf(case[key]) for key in
                                   ("mass1", "mass2", "chi1", "chi2"))
            nu = m1*m2/(m1+m2)**2
            for omega_text in plan["omega_M"]:
                omega = mp.mpf(omega_text)
                common = (m1, m2, chi1, chi2, omega, omega,
                          omega**(-mp.mpf(1)/3),
                          mp.sqrt(1-nu*omega**(mp.mpf(2)/3)))
                taylor = factorized_21_spin_amplitude(*common,
                                                       digits=plan["precision_decimal_digits"])
                rho_flux = factorized_21_rho_hybrid_amplitude(
                    *common, auxiliary_spin_order=3,
                    digits=plan["precision_decimal_digits"])
                rho_wave = factorized_21_rho_hybrid_amplitude(
                    *common, auxiliary_spin_order=1,
                    digits=plan["precision_decimal_digits"])
                if not all(mp.isfinite(x) and x >= 0
                           for x in (taylor, rho_flux, rho_wave)):
                    raise ValueError("nonfinite or negative mode modulus")
                if case["name"] == "q4_zero_spin_control":
                    controls_pass &= max(abs(taylor-rho_flux),
                                         abs(rho_flux-rho_wave)) < threshold
                elif case["name"] == "equal_mass_unequal_spin_control":
                    controls_pass &= (abs(taylor-rho_flux) < threshold
                                      and rho_flux > 0)
                else:
                    differences_present &= (abs(taylor-rho_flux) > threshold
                                            and abs(rho_flux-rho_wave) > threshold)
                rows.append({
                    "case": case["name"], "omega_M": omega_text,
                    "taylor_f_flux_amplitude": _decimal(taylor),
                    "rho_hybrid_flux_amplitude": _decimal(rho_flux),
                    "rho_hybrid_complex_mode_amplitude": _decimal(rho_wave),
                    "rho_flux_minus_taylor_relative": _decimal((rho_flux-taylor)/taylor),
                    "rho_flux_minus_wave_relative": _decimal((rho_flux-rho_wave)/rho_wave),
                    "rho_flux_contribution": _decimal(omega**2*rho_flux**2/(8*mp.pi)),
                    "taylor_flux_contribution": _decimal(omega**2*taylor**2/(8*mp.pi)),
                })

    if not controls_pass or not differences_present:
        raise ValueError("frozen (2,1) discriminator failed its control gates")
    report = {
        "schema": "diffgw.seobnrv4.mode-21-rho-hybrid-diagnostic.v1",
        "status": "diagnostic_complete",
        "plan_sha256": _digest(args.plan),
        "source_audit_sha256": _digest(audit_path),
        "scalar_reference_sha256": _digest(reference_path),
        "upstream_tag_source_audited": audit["upstream_tag"],
        "compiled_numerical_comparison_performed": False,
        "historical_parity_established": False,
        "release_qualified": False,
        "controls_pass": bool(controls_pass),
        "differences_present": bool(differences_present),
        "rows": rows,
    }
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
