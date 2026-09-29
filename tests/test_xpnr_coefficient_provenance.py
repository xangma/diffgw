"""Consistency checks for the provisional XPNR coefficient provenance gate."""

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_PATH = ROOT / "docs/waveforms/research/imrphenomxpnr-qualification.json"
COEFFICIENT_PATHS = {
    "torch": ROOT / "src/torchwave/approximants/pnr_coefficients.py",
    "jax": ROOT / "src/jaxwave/approximants/pnr_coefficients.py",
}


def _qualification():
    return json.loads(QUALIFICATION_PATH.read_text())["calibration_assets"]


def _coefficient_literals(path):
    """Read literal tables without importing either numerical frontend."""
    arrays = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "COEFF_ARRAYS"
            and isinstance(target.slice, ast.Constant)
            and isinstance(target.slice.value, str)
            and isinstance(node.value, ast.Call)
            and node.value.args
        ):
            continue
        arrays[target.slice.value] = ast.literal_eval(node.value.args[0])
    return arrays


def _shape(value):
    if not isinstance(value, list):
        return ()
    child_shapes = {_shape(child) for child in value}
    assert len(child_shapes) == 1
    return (len(value), *child_shapes.pop())


def test_provisional_gate_distinguishes_each_evidence_class():
    assets = _qualification()
    evidence = {item["id"]: item for item in assets["source_to_component"]}

    assert assets["provenance"] == "unresolved_independent_source_and_redistribution_terms"
    assert assets["regeneration_assessment"]["status"] == "not_independently_reproducible_from_published_assets"
    assert assets["regeneration_assessment"]["prohibited_methods_used"] is False
    assert {item["evidence_class"] for item in evidence.values()} == {
        "published_equations",
        "published_partial_generation_procedure",
        "third_party_calibration_input",
        "third_party_fitting_tool",
        "author_release_partial_carrier_workflow",
        "unsupported_inherited_tensors",
    }
    assert evidence["diffgw_inherited_angle_tensors"]["license"] == "unresolved"
    assert assets["tensor_inventory"]["redistribution_terms"] is None
    assert len(assets["regeneration_assessment"]["missing"]) == 5


def test_manifest_identifies_the_unmodified_inherited_payload():
    assets = _qualification()
    inventory = assets["tensor_inventory"]
    expected_names = set(inventory["names"])
    expected_shape = tuple(inventory["shape_by_name"])

    literals = {backend: _coefficient_literals(path) for backend, path in COEFFICIENT_PATHS.items()}
    assert set(literals) == {"torch", "jax"}
    for backend, arrays in literals.items():
        assert set(arrays) == expected_names
        assert all(_shape(value) == expected_shape for value in arrays.values())
        digest = hashlib.sha256(COEFFICIENT_PATHS[backend].read_bytes()).hexdigest()
        assert digest == assets[f"{backend}_source_sha256"]
    assert literals["torch"] == literals["jax"]
