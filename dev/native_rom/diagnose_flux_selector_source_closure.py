"""Evaluate the frozen nine-row public-source flux-selector closure."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import shlex
import socket
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-27"
PLAN = EVIDENCE / "flux_selector_source_closure_plan.json"
AUDIT = EVIDENCE / "flux_selector_source_closure_audit.json"
PRIOR = EVIDENCE / "flux_symbolic_provenance_matrix.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tex(path: Path) -> str:
    data = path.read_bytes()
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith(".tex")
            ]
            if not members:
                raise ValueError(f"no TeX source in {path}")
            member = max(members, key=lambda item: item.size)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read {member.name} in {path}")
            return extracted.read().decode("utf-8", errors="replace")
    except tarfile.ReadError:
        return gzip.decompress(data).decode("utf-8", errors="replace")


def _require(text: str, fragments: list[str], source: str) -> None:
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise ValueError(f"{source} lacks frozen evidence fragments: {missing}")


def _rows() -> list[dict]:
    rows = []
    overlap_equations = {
        "2,1": ("BFH Eq. (53a)", "Pan Eqs. (28b), (29b)"),
        "3,1": ("BFH Eq. (53c)", "Pan Eqs. (28e), (29e)"),
        "3,3": ("BFH Eq. (53d)", "Pan Eqs. (28c), (29c)"),
        "4,1": ("BFH Eq. (53e)", "Pan Eqs. (28i), (29i)"),
        "4,3": ("BFH Eq. (53f)", "Pan Eqs. (28g), (29g)"),
    }
    for mode, (bfh_equation, pan_equations) in overlap_equations.items():
        rows.append(
            {
                "mode": mode,
                "question": "bfh_pan_coexistence",
                "status": "pass",
                "bfh_equation": bfh_equation,
                "pan_equations": pan_equations,
                "selector": (
                    "Replace Pan test-particle terms at every overlapping PN order "
                    "with the BFH generic-mass-ratio counterpart; retain Pan terms "
                    "only where no generic counterpart is known. Place the combined "
                    "spin residual in f_lm outside (rho_lm_NS)^ell."
                ),
                "evidence": (
                    "Pan Sec. IV replacement rule; BFH Eqs. (52)-(53) and stated "
                    "Pan test-mass limit; Taracchini 2014 mode-source composition "
                    "and low-ell odd-m prescription."
                ),
                "missing_contract": None,
            }
        )
    for mode in ("8,2", "8,4", "8,6", "8,8"):
        rows.append(
            {
                "mode": mode,
                "question": "ell8_even_m_truncation",
                "status": "fail",
                "bfh_equation": None,
                "pan_equations": "Pan Appendix B f_lm and Appendix C rho_lm",
                "selector": None,
                "evidence": (
                    "Pan's generic even-parity equation includes "
                    "rho_lm^spin=-2*q*v^3/3, while its stated complete ell=8 "
                    "f_lm and rho_lm lists omit that term; Taracchini 2014 does "
                    "not select between them."
                ),
                "missing_contract": (
                    "One eligible sentence or equation stating whether this mode's "
                    "rho_lm includes -2*chi_Kerr*v^3/3 (equivalently whether f_lm "
                    "includes -16*chi_Kerr*v^3/3) or intentionally truncates it."
                ),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--log-path", required=True, type=Path)
    args = parser.parse_args(argv)

    plan = json.loads(PLAN.read_text())
    audit = json.loads(AUDIT.read_text())
    prior = json.loads(PRIOR.read_text())
    if plan["schema"] != "diffgw.seobnrv4.flux-selector-source-closure-plan.v1":
        parser.error("frozen closure plan differs")
    if audit["schema"] != "diffgw.seobnrv4.flux-selector-source-closure-audit.v1":
        parser.error("frozen closure audit differs")
    if _sha256(PRIOR) != plan["prior_matrix_sha256"]:
        parser.error("prior 35-mode matrix drifted")
    if prior["selector_incomplete_modes"] != [
        "2,1", "3,1", "3,3", "4,1", "4,3", "8,2", "8,4", "8,6", "8,8"
    ]:
        parser.error("prior incomplete-row set drifted")

    source_hashes = {}
    texts = {}
    for source in audit["sources"]:
        path = args.source_dir / source["file"]
        actual = _sha256(path)
        if actual != source["sha256"]:
            parser.error(f"{source['id']} SHA-256 mismatch")
        source_hashes[source["id"]] = actual
        if path.suffix == ".tar":
            texts[source["id"]] = _tex(path)

    pan_fragments = [
        "replacing all the test-particle terms in",
        "whose generic mass-ratio counterparts are known with their",
        "generic expressions.",
        "f_{88}=1-\\frac{1\\,741}{171}\\,v^2",
        "\\rho_{88}=1-\\frac{1\\,741}{1\\,368} v^2",
        "f_{\\ell m}^{\\rm even} = - 2\\ell\\,q\\,v^3/3",
    ]
    for version in ("pan-v1", "pan-v2"):
        _require(texts[version], pan_fragments, version)
    for version in ("bfh-v1", "bfh-v2"):
        _require(texts[version], ["test-mass limit", "reduce to the 2PN"], version)
    _require(
        texts["taracchini-2014-v1"],
        [
            "include comparable-mass",
            "include all the known spin effects from the test-particle limit",
            "except those with $\\ell \\leq 4$ and",
        ],
        "taracchini-2014-v1",
    )
    for version in ("bohe-v1", "bohe-v2", "bohe-v3"):
        _require(
            texts[version],
            ["adopt the same expressions", "replace the spinning test-particle-limit terms"],
            version,
        )

    rows = _rows()
    expected = [
        *plan["rows"]["bfh_pan_coexistence"]["modes"],
        *plan["rows"]["ell8_even_m_truncation"]["modes"],
    ]
    if [row["mode"] for row in rows] != expected:
        parser.error("closure table does not exactly cover the frozen rows")
    passed = [row["mode"] for row in rows if row["status"] == "pass"]
    failed = [row["mode"] for row in rows if row["status"] == "fail"]
    report = {
        "schema": "diffgw.seobnrv4.flux-selector-source-closure.v1",
        "status": "diagnostic_complete",
        "component_id": plan["component_id"],
        "row_count": len(rows),
        "pass_count": len(passed),
        "fail_count": len(failed),
        "passed_modes": passed,
        "failed_modes": failed,
        "rows": rows,
        "prior_selector_complete_mode_count": prior["selector_complete_mode_count"],
        "refined_selector_complete_mode_count": (
            prior["selector_complete_mode_count"] + len(passed)
        ),
        "complete_final_mode_selector": not failed,
        "source_sha256": source_hashes,
        "plan_sha256": _sha256(PLAN),
        "audit_sha256": _sha256(AUDIT),
        "prior_matrix_sha256": _sha256(PRIOR),
        "script_sha256": _sha256(Path(__file__)),
        "process": {
            "host": socket.gethostname(),
            "cwd": str(ROOT),
            "pid": os.getpid(),
            "command": shlex.join([sys.executable, *sys.argv]),
            "log_path": str(args.log_path),
            "python_version": platform.python_version(),
            "duration_seconds": time.monotonic() - started,
        },
        "clean_room": audit["clean_room"],
        "conclusion": (
            "The explicit Pan replacement rule closes the five BFH/Pan overlap "
            "rows. The four ell=8 rows remain source-underdetermined, so the "
            "35-mode selector and V4-FLUX-4-9 remain blocked."
        ),
        "production_change": False,
        "historical_fidelity_claim": False,
        "release_qualified": False,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "pass_count": len(passed),
                "fail_count": len(failed),
                "failed_modes": failed,
                "complete_final_mode_selector": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
