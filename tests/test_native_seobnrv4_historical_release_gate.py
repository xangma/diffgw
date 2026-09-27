"""Regression checks for the native-SEOBNRv4 historical-fidelity boundary."""

import json

from dev.native_rom.diagnose_historical_fidelity_release import DEFAULT_PLAN, evaluate


def test_unqualified_candidate_remains_outside_release_catalog():
    report = evaluate(json.loads(DEFAULT_PLAN.read_text()))

    assert report["status"] == "boundary_enforced"
    assert report["release_qualified"] is False
    gates = report["historical_fidelity_gates"]
    assert gates["gradient"]["passed"] is True
    assert all(
        gate["passed"] is False
        for name, gate in gates.items()
        if name != "gradient"
    )
    assert report["catalog_boundary"] == {
        "enforced": True,
        "native_catalog_names_absent": True,
        "native_frontend_aliases_absent": True,
        "official_external_rom_wrapper_preserved": True,
    }


def test_subthreshold_nr_result_is_disclosure_not_historical_gate():
    report = evaluate(json.loads(DEFAULT_PLAN.read_text()))
    disclosure = report["nr_disclosure"]

    assert report["nr_disclosure_preserved"] is True
    assert disclosure["native_overlap"] < disclosure["threshold"]
    assert disclosure["public_historical_overlap"] < disclosure["threshold"]
    assert disclosure["physical_accuracy_passed"] is False
    assert disclosure["release_gate_for_selected_criterion"] is False
    assert disclosure["classification"] == (
        "physical_accuracy_disclosure_not_historical_fidelity_gate"
    )
