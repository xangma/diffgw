"""Evaluate the frozen native-SEOBNRv4 historical-fidelity release boundary."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import socket


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "dev/native_rom/evidence/2026-09-27/historical_fidelity_release_plan.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(root: Path, relative: str) -> dict:
    return json.loads((root / relative).read_text())


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise ValueError(f"{name} not found in {path}")


def _approximant_bindings(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "_APPROXIMANTS"
                   for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            raise ValueError(f"_APPROXIMANTS is not a literal dictionary in {path}")
        result = {}
        for key, value in zip(node.value.keys, node.value.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise ValueError(f"nonliteral approximant key in {path}")
            if isinstance(value, ast.Name):
                target = value.id
            elif isinstance(value, ast.Attribute):
                target = value.attr
            else:
                target = ast.dump(value)
            result[key.value] = target
        return result
    raise ValueError(f"_APPROXIMANTS not found in {path}")


def evaluate(plan: dict, root: Path = ROOT) -> dict:
    if plan.get("schema") != "diffgw.seobnrv4.historical-fidelity-release-plan.v1":
        raise ValueError("unexpected release plan schema")
    if plan.get("criterion") != "historical_fidelity":
        raise ValueError("release plan does not select historical fidelity")
    for relative, expected in plan["input_sha256"].items():
        actual = sha256(root / relative)
        if actual != expected:
            raise ValueError(f"pinned release input changed: {relative}: {actual}")

    manifest = _load_json(root, plan["inputs"]["manifest"])
    whole = _load_json(root, plan["inputs"]["whole_waveform"])
    numerical = _load_json(root, plan["inputs"]["numerical"])
    gradient = _load_json(root, plan["inputs"]["gradient"])
    current_nr = _load_json(root, plan["inputs"]["current_nr"])
    public_nr = _load_json(root, plan["inputs"]["public_nr"])
    component_status = {item["id"]: item["status"] for item in manifest["components"]}

    source_ids = plan["required_source_components"]
    missing_source = [name for name in source_ids if component_status.get(name) != "executable"]
    whole_passed = bool(
        whole.get("whole_waveform_parity_validated")
        and whole.get("prerequisite_numerical_checks_passed")
    )
    gradient_checks = gradient.get("checks", {})
    rom_status = component_status.get(plan["required_rom_component"])
    gates = {
        "source_fidelity": {
            "passed": not missing_source,
            "pending_components": missing_source,
        },
        "whole_waveform": {
            "passed": whole_passed,
            "whole_waveform_parity_validated": whole.get("whole_waveform_parity_validated"),
            "prerequisite_numerical_checks_passed": whole.get(
                "prerequisite_numerical_checks_passed"
            ),
        },
        "numerical": {
            "passed": numerical.get("status") == "diagnostic_complete",
            "status": numerical.get("status"),
        },
        "gradient": {
            "passed": bool(gradient_checks) and all(gradient_checks.values()),
            "checks": gradient_checks,
            "eager_jit_relative_l2": gradient.get("metrics", {}).get(
                "eager_jit_relative_l2"
            ),
            "eager_jit_relative_l2_max": plan["gradient_eager_jit_relative_l2_max"],
        },
        "rom_holdout": {
            "passed": (
                rom_status == "executable" and plan.get("rom_holdout_evidence") is not None
            ),
            "component_status": rom_status,
            "holdout_evidence": plan.get("rom_holdout_evidence"),
        },
    }
    release_qualified = all(gate["passed"] for gate in gates.values())

    native_overlap = current_nr["weighted"]["minimum"]["overlap"]
    public_case = public_nr["cases"][plan["nr_disclosure"]["case"]]
    public_overlap = public_case["weighted_overlaps_at_selected_minimum"]["public_vs_nr"][
        "overlap"
    ]
    threshold = plan["nr_disclosure"]["threshold"]
    disclosure = {
        "classification": "physical_accuracy_disclosure_not_historical_fidelity_gate",
        "case": plan["nr_disclosure"]["case"],
        "total_mass_msun": current_nr["weighted"]["minimum"]["total_mass_msun"],
        "threshold": threshold,
        "native_overlap": native_overlap,
        "public_historical_overlap": public_overlap,
        "physical_accuracy_passed": native_overlap >= threshold and public_overlap >= threshold,
        "release_gate_for_selected_criterion": False,
    }
    expected = plan["nr_disclosure"]
    disclosure_preserved = (
        abs(native_overlap - expected["native_overlap"]) <= expected["absolute_tolerance"]
        and abs(public_overlap - expected["public_historical_overlap"])
        <= expected["absolute_tolerance"]
        and not disclosure["physical_accuracy_passed"]
    )

    canonical = _literal_assignment(root / plan["inputs"]["catalog"], "CANONICAL_APPROXIMANTS")
    frontend_bindings = {
        frontend: _approximant_bindings(root / relative)
        for frontend, relative in plan["inputs"]["frontend_dispatch"].items()
    }
    forbidden = set(plan["forbidden_native_catalog_names"])
    canonical_native_absent = forbidden.isdisjoint(name.lower().replace("_", "") for name in canonical)
    frontend_native_absent = all(
        forbidden.isdisjoint(bindings) for bindings in frontend_bindings.values()
    )
    official_keys = set(plan["official_external_rom_dispatch_keys"])
    official_wrapper_preserved = all(
        official_keys.issubset(bindings)
        and all(bindings[key] == "OfficialSEOBNRv4ROM" for key in official_keys)
        for bindings in frontend_bindings.values()
    ) and plan["official_external_rom_catalog_name"] in canonical
    catalog_boundary_enforced = (
        canonical_native_absent and frontend_native_absent and official_wrapper_preserved
    )
    boundary_enforced = catalog_boundary_enforced and disclosure_preserved and not release_qualified

    return {
        "schema": "diffgw.seobnrv4.historical-fidelity-release.v1",
        "status": "boundary_enforced" if boundary_enforced else "boundary_failed",
        "criterion": "historical_fidelity",
        "release_qualified": release_qualified,
        "historical_fidelity_gates": gates,
        "nr_disclosure": disclosure,
        "nr_disclosure_preserved": disclosure_preserved,
        "catalog_boundary": {
            "enforced": catalog_boundary_enforced,
            "native_catalog_names_absent": canonical_native_absent,
            "native_frontend_aliases_absent": frontend_native_absent,
            "official_external_rom_wrapper_preserved": official_wrapper_preserved,
        },
        "pending_manifest_components": sorted(
            name for name, status in component_status.items() if status == "pending"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    report = evaluate(plan)
    report.update(
        plan_sha256=sha256(args.plan),
        script_sha256=sha256(Path(__file__)),
        input_sha256=plan["input_sha256"],
        process={"host": socket.gethostname(), "cwd": str(ROOT)},
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "release_qualified": report["release_qualified"],
        "gates": {name: gate["passed"] for name, gate in report["historical_fidelity_gates"].items()},
    }))
    return 0 if report["status"] == "boundary_enforced" else 1


if __name__ == "__main__":
    raise SystemExit(main())
