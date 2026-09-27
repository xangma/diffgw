"""Evaluate the frozen public-source completeness gate for the historical v4 Hamiltonian."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "dev/native_rom/evidence/2026-09-27/hamiltonian_historical_source_gate_plan.json"
AUDIT = ROOT / "dev/native_rom/evidence/2026-09-27/hamiltonian_historical_source_audit.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bohe-source", required=True, type=Path)
    parser.add_argument("--bohe-pdf", required=True, type=Path)
    parser.add_argument("--barausse-buonanno-source", required=True, type=Path)
    parser.add_argument("--barausse-buonanno-pdf", required=True, type=Path)
    parser.add_argument("--steinhoff-source", required=True, type=Path)
    parser.add_argument("--steinhoff-pdf", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)

    plan = json.loads(PLAN.read_text())
    audit = json.loads(AUDIT.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.hamiltonian-historical-source-gate-plan.v1"
            or plan["component_id"] != "V4-HAM-1-3"
            or plan["release_qualified"] is not False
            or plan["historical_fidelity_claim"] is not False
            or plan["production_change"] is not False):
        parser.error("frozen source-gate plan differs")
    if (audit["schema"] != "diffgw.seobnrv4.hamiltonian-historical-source-audit.v1"
            or audit["component_id"] != plan["component_id"]
            or audit["source_scope"] != "public_primary_sources_only"
            or audit["release_qualified"] is not False
            or audit["historical_fidelity_claim"] is not False):
        parser.error("source audit differs")

    source_paths = {
        "bohe_source": args.bohe_source,
        "bohe_pdf": args.bohe_pdf,
        "barausse_buonanno_source": args.barausse_buonanno_source,
        "barausse_buonanno_pdf": args.barausse_buonanno_pdf,
        "steinhoff_source": args.steinhoff_source,
        "steinhoff_pdf": args.steinhoff_pdf,
    }
    expected_source_sha256 = {
        "bohe_source": plan["primary_sources"]["bohe_2017"]["source_sha256"],
        "bohe_pdf": plan["primary_sources"]["bohe_2017"]["pdf_sha256"],
        "barausse_buonanno_source": plan["primary_sources"]["barausse_buonanno_2011"]["source_sha256"],
        "barausse_buonanno_pdf": plan["primary_sources"]["barausse_buonanno_2011"]["pdf_sha256"],
        "steinhoff_source": plan["primary_sources"]["steinhoff_2016"]["source_sha256"],
        "steinhoff_pdf": plan["primary_sources"]["steinhoff_2016"]["pdf_sha256"],
    }
    actual_source_sha256 = {name: _sha256(path) for name, path in source_paths.items()}
    if actual_source_sha256 != expected_source_sha256:
        parser.error("primary-source SHA-256 mismatch")

    required = [entry["id"] for entry in plan["mandatory_requirements"]]
    audited = [entry["id"] for entry in audit["requirements"]]
    if len(required) != len(set(required)) or audited != required:
        parser.error("audit requirements do not exactly match the frozen plan")
    if any(not entry["evidence"].strip() for entry in audit["requirements"]):
        parser.error("source-audit requirement lacks evidence")
    if audit["clean_room"] != {
        "lal_source_consulted": False,
        "binary_oracle_queried": False,
        "historical_coefficients_transcribed": False,
        "fit_or_threshold_retuned": False,
    }:
        parser.error("clean-room exclusions were not preserved")

    gates = {entry["id"]: bool(entry["satisfied"])
             for entry in audit["requirements"]}
    source_gate_passed = all(gates.values())
    blocking_requirements = [name for name, passed in gates.items() if not passed]
    report = {
        "schema": "diffgw.seobnrv4.hamiltonian-historical-source-gate.v1",
        "status": "diagnostic_complete",
        "component_id": plan["component_id"],
        "source_gate_passed": source_gate_passed,
        "gates": gates,
        "satisfied_requirement_count": sum(gates.values()),
        "mandatory_requirement_count": len(gates),
        "blocking_requirements": blocking_requirements,
        "conclusion": audit["conclusion"],
        "clean_room": audit["clean_room"],
        "source_sha256": actual_source_sha256,
        "plan_sha256": _sha256(PLAN),
        "audit_sha256": _sha256(AUDIT),
        "script_sha256": _sha256(Path(__file__)),
        "process": {"host": socket.gethostname(), "cwd": str(ROOT)},
        "production_change": False,
        "historical_fidelity_claim": False,
        "release_qualified": False,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "source_gate_passed": source_gate_passed,
        "satisfied": report["satisfied_requirement_count"],
        "required": report["mandatory_requirement_count"],
        "blocking_requirements": blocking_requirements,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
