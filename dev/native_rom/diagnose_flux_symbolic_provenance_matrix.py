"""Run the frozen public-equation symbolic provenance matrix for v4 flux."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import socket
import sys
import time
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-27"
PLAN = EVIDENCE / "flux_symbolic_provenance_matrix_plan.json"
AUDIT = EVIDENCE / "flux_symbolic_provenance_matrix_audit.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mode_equation(ell: int, m: int) -> str:
    letter = chr(ord("a") + ell - m)
    return f"Taracchini 2012 Eq. (A{ell + 6}{letter})"


def _pan_equation(ell: int, m: int) -> str:
    if ell <= 4:
        offset = sum(range(2, ell)) + ell - m
        letter = chr(ord("a") + offset)
        return f"Pan Eqs. (28{letter}), (29{letter})"
    return "Pan Appendices B-C"


def _matrix(plan: dict, audit: dict) -> list[dict]:
    unresolved = {
        (entry["field"], mode): entry["evidence"]
        for entry in audit["unresolved_decisions"]
        for mode in entry["modes"]
    }
    bfh = {
        "2,1": "BFH Eqs. (52), (53a)",
        "2,2": "BFH Eqs. (52), (53b)",
        "3,1": "BFH Eqs. (52), (53c)",
        "3,3": "BFH Eqs. (52), (53d)",
        "4,1": "BFH Eqs. (52), (53e)",
        "4,3": "BFH Eqs. (52), (53f)",
    }
    low_odd = {"2,1", "3,1", "3,3", "4,1", "4,3"}
    rows = []
    for mode in plan["matrix"]["expected_modes"]:
        ell, m = map(int, mode.split(","))
        parity = "even" if (ell + m) % 2 == 0 else "odd"
        decisions = {
            "replacement_decision": (
                "Bohe Eqs. (4)-(7) replace the stated predecessor (2,2) terms"
                if mode == "2,2" else "retain cited mode ingredients"
            ),
            "coexistence_decision": (
                "Bohe prints the assembled v4 (2,2) residual"
                if mode == "2,2" else "no overlapping BFH 2PN-spin mode term"
            ),
            "power_placement_decision": (
                "Taracchini 2012 low-ell odd-m rule: (rho_NS)^ell + f_spin"
                if mode in low_odd else "Taracchini 2014: rho^ell"
            ),
            "truncation_decision": "retain the cited published terms",
        }
        for field in tuple(decisions):
            evidence = unresolved.get((field, mode))
            if evidence:
                decisions[field] = f"UNRESOLVED: {evidence}"
        row = {
            "mode": mode,
            "parity": parity,
            "effective_source": "H_eff" if parity == "even" else "p_phi*v_Omega",
            "baseline_equation": _mode_equation(ell, m),
            "residual_power_rule": (
                "(rho_NS)^ell + f_spin" if mode in low_odd else "rho^ell"
            ),
            "comparable_mass_spin_equation": bfh.get(
                mode, "no nonzero BFH 2PN-spin mode term"
            ),
            "test_particle_spin_equation": _pan_equation(ell, m),
            "v4_override": "Bohe Eqs. (4)-(7)" if mode == "2,2" else "none",
            **decisions,
        }
        row["selector_fields_complete"] = not any(
            value.startswith("UNRESOLVED:") for value in decisions.values()
        )
        rows.append(row)
    return rows


def _rho43_symbolic() -> dict:
    v, q, r2, r4, r6 = sp.symbols("v q r2 r4 r6")
    base = 1 + r2 * v**2 + r4 * v**4 + r6 * v**6
    spin_f = sp.Rational(3, 2) * q**2 * v**4 - sp.Rational(12113, 1540) * q * v**5
    spin_rho = sp.Rational(3, 8) * q**2 * v**4 - sp.Rational(12113, 6160) * q * v**5
    taylor_form = sp.expand(base**4 + spin_f)
    rho_form = sp.expand((base + spin_rho) ** 4)
    difference = sp.series(rho_form - taylor_form, v, 0, 11).removeO().expand()
    coefficients = {
        order: sp.factor(difference.coeff(v, order)) for order in range(11)
    }
    nonzero = [order for order, coefficient in coefficients.items() if coefficient != 0]
    return {
        "equal_through_published_spin_order_v5": all(
            coefficients[order] == 0 for order in range(6)
        ),
        "first_difference_order": nonzero[0],
        "first_difference_coefficient": str(coefficients[nonzero[0]]),
        "difference_through_v10": str(sp.collect(difference, v)),
        "primary_source_selected_form": "nonspinning_rho43_to_fourth_power_plus_spinning_f43",
        "selection_evidence": "Taracchini 2012 Eqs. (A10b), (A15d), and the low-ell odd-m rule adopted explicitly by Taracchini 2014",
    }


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bohe", "taracchini_2012", "taracchini_2014", "pan", "din", "bfh"):
        option = name.replace("_", "-")
        parser.add_argument(f"--{option}-source", required=True, type=Path)
        parser.add_argument(f"--{option}-pdf", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--log-path", required=True, type=Path)
    args = parser.parse_args(argv)

    plan = json.loads(PLAN.read_text())
    audit = json.loads(AUDIT.read_text())
    if plan["schema"] != "diffgw.seobnrv4.flux-symbolic-provenance-matrix-plan.v1":
        parser.error("frozen matrix plan differs")
    if audit["schema"] != "diffgw.seobnrv4.flux-symbolic-provenance-matrix-audit.v1":
        parser.error("frozen matrix audit differs")
    if plan["historical_fidelity_criterion"]["prior_plan_sha256"] != _sha256(
        EVIDENCE / "flux_historical_source_gate_plan.json"
    ):
        parser.error("historical-fidelity criterion drifted")

    actual_hashes = {}
    for name, source in plan["primary_sources"].items():
        for kind in ("source", "pdf"):
            path = getattr(args, f"{name}_{kind}")
            actual = _sha256(path)
            if actual != source[f"{kind}_sha256"]:
                parser.error(f"{name} {kind} SHA-256 mismatch")
            actual_hashes[f"{name}_{kind}"] = actual

    rows = _matrix(plan, audit)
    expected = plan["matrix"]["expected_modes"]
    if [row["mode"] for row in rows] != expected or len(rows) != 35:
        parser.error("matrix does not exactly cover the frozen 35 modes")
    missing_fields = [
        f"{row['mode']}:{field}"
        for row in rows
        for field in plan["matrix"]["required_fields"]
        if field not in row or not str(row[field]).strip()
    ]
    if missing_fields:
        parser.error(f"matrix fields missing: {missing_fields}")

    symbolic = _rho43_symbolic()
    findings = audit["requirement_findings"]
    if not symbolic["equal_through_published_spin_order_v5"]:
        parser.error("Pan (4,3) normal forms do not agree through frozen order")
    if symbolic["first_difference_order"] <= plan["rho43_symbolic_check"]["published_spin_order"]:
        parser.error("Pan (4,3) first-difference order violates frozen plan")
    requirements = {
        name: bool(findings[name]["satisfied"])
        for name in plan["mandatory_requirements"]
    }
    incomplete_modes = [row["mode"] for row in rows if not row["selector_fields_complete"]]
    if requirements["complete_final_mode_selector"] != (not incomplete_modes):
        parser.error("complete-selector verdict does not follow the matrix")

    report = {
        "schema": "diffgw.seobnrv4.flux-symbolic-provenance-matrix.v1",
        "status": "diagnostic_complete",
        "component_id": plan["component_id"],
        "requirements": requirements,
        "resolved_requirements": [name for name, value in requirements.items() if value],
        "blocking_requirements": [name for name, value in requirements.items() if not value],
        "rho43_symbolic_check": symbolic,
        "mode_count": len(rows),
        "selector_complete_mode_count": len(rows) - len(incomplete_modes),
        "selector_incomplete_modes": incomplete_modes,
        "matrix": rows,
        "source_sha256": actual_hashes,
        "plan_sha256": _sha256(PLAN),
        "audit_sha256": _sha256(AUDIT),
        "script_sha256": _sha256(Path(__file__)),
        "process": {
            "host": socket.gethostname(),
            "cwd": str(ROOT),
            "pid": os.getpid(),
            "command": shlex.join([sys.executable, *sys.argv]),
            "log_path": str(args.log_path),
            "python_version": platform.python_version(),
            "sympy_version": sp.__version__,
            "duration_seconds": time.monotonic() - started,
        },
        "clean_room": audit["clean_room"],
        "conclusion": audit["conclusion"],
        "historical_fidelity_claim": False,
        "production_change": False,
        "release_qualified": False,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "requirements": requirements,
        "mode_count": len(rows),
        "selector_incomplete_modes": incomplete_modes,
        "rho43_first_difference_order": symbolic["first_difference_order"],
        "release_qualified": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
