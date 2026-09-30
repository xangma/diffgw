"""Regression gate for the unresolved XPNR spin-at-attachment step."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_PATH = ROOT / "docs/waveforms/research/imrphenomxpnr-qualification.json"
BACKEND_PATHS = {
    "torch": ROOT / "src/torchwave/approximants",
    "jax": ROOT / "src/jaxwave/approximants",
}


def _function(path, name):
    tree = ast.parse(path.read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _assigned_value(function, name):
    matches = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    assert len(matches) == 1
    return matches[0]


def test_qualification_records_paper_only_attachment_boundary():
    qualification = json.loads(QUALIFICATION_PATH.read_text())
    attachment = qualification["spin_at_attachment"]

    assert attachment["status"] == "terminal_state_geometry_pass_stopping_procedure_unresolved"
    assert attachment["scalar_reference"]["function"].endswith("single_spin_map_from_transition_vectors")
    assert attachment["current_diffgw_boundary"]["map_input_epoch"] == "reference_frequency"
    assert attachment["current_diffgw_boundary"]["terminal_state_returned"] is False
    assert attachment["current_diffgw_boundary"]["status"] == "not_paper_fidelity"
    assert len(attachment["unresolved"]) >= 5


def test_landmark_selector_remains_a_provisional_component():
    qualification = json.loads(QUALIFICATION_PATH.read_text())
    joins = qualification["angle_join_components"]
    bounded_pass = qualification["qualification_pass_2026_09_26"]

    assert qualification["qualified_full_waveform"] is False
    assert "lower_connection_selection" in joins["status"]
    assert any("landmarks" in item and "correct-root" in item for item in joins["open"])
    assert "no LAL source or coefficients inspected or copied" in bounded_pass["scope"]
    assert "exact terminal-sample rule" in bounded_pass["terminal_state_blocker"]
    assert "licensed angle-tensor release" in bounded_pass["calibration_blocker"]
    for directory in BACKEND_PATHS.values():
        assert "beta_lower_connection_frequency" not in (
            directory / "precession_pnr.py"
        ).read_text()


def test_backends_still_map_reference_spins_and_do_not_return_terminal_state():
    expected_arguments = ["m1", "m2", "s1x", "s1y", "s1z", "s2x", "s2y", "s2z"]
    expected_return = ["alpha_final", "beta_out", "gamma_final", "theta_jn", "zeta", "carrier_phase"]

    for directory in BACKEND_PATHS.values():
        pnr = _function(directory / "precession_pnr.py", "evaluate_pnr_precession")
        calls = [
            node
            for node in ast.walk(pnr)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "compute_pnr_single_spin"
        ]
        assert len(calls) == 1
        assert [argument.id for argument in calls[0].args] == expected_arguments

        spintaylor = _function(directory / "precession_spintaylor.py", "evaluate_spintaylor_precession")
        final_return = spintaylor.body[-1]
        assert isinstance(final_return, ast.Return)
        assert isinstance(final_return.value, ast.Tuple)
        assert [value.id for value in final_return.value.elts] == expected_return

        # Reference anchoring and output-grid invariance are checked in
        # test_spintaylor_reference_angle.py, independently of node counts.
        transition = _assigned_value(spintaylor, "f_trans")
        assert isinstance(transition, ast.BinOp) and isinstance(transition.op, ast.Mult)
        assert isinstance(transition.left, ast.Constant) and transition.left.value == 0.98
        assert isinstance(transition.right, ast.Name) and transition.right.id == "f_meco"
