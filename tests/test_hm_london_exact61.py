"""Checks for the bounded exact-N=61 HM waveform diagnostic."""

import importlib.util
import hashlib
import inspect
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_hm_london_exact61.py"
REPORT = (
    ROOT
    / "docs"
    / "reviews"
    / "2026-09-13-parity-review"
    / "verification"
    / "hm-london-exact61"
    / "report.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("torchwave_hm_london_exact61", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_declared_controls_and_frequencies_are_bounded():
    module = _module()
    assert module.MODE_LABELS == ((2, 2, 0), (2, 1, 0), (3, 3, 0), (3, 2, 0), (4, 4, 0), (4, 3, 0))
    assert module.VARIANT_NAMES == ("n61_main", "printed_london", "n257_main")
    assert module.JOIN_AMPLITUDE == 0.014
    assert module.JOIN_PHASE == 0.018


def test_n61_runtime_payload_has_fixed_expected_hash():
    from torchwave.approximants import imrphenom_hm_qnm_n61 as n61

    expected = "afcb21b634efacbc1ed0b44f24fda5df5ab08e236f3b7cbfdc7e71d3452d6525"
    actual = hashlib.sha256(repr(dict(n61.N61_COEFFICIENTS)).encode("utf-8")).hexdigest()
    assert n61.N61_COEFFICIENT_PAYLOAD_SHA256 == expected
    assert actual == expected


def test_fit_inputs_use_main_full_precision_coefficients():
    module = _module()
    if not module.EXACT_REPORT_PATH.exists() or not module.N257_REPORT_PATH.exists():
        pytest.skip("optional London fit diagnostic reports are not present")
    exact, exact_meta = module._load_exact_coefficients(module.EXACT_REPORT_PATH)
    n257, n257_meta = module._load_n257_coefficients(module.N257_REPORT_PATH)
    assert len(exact) == len(n257) == 6
    assert all(len(values) in (5, 7) for values in exact.values())
    assert all(len(values) in (5, 7) for values in n257.values())
    assert exact_meta["count"] == 61 and exact_meta["strategy"] == "main"
    assert n257_meta["count"] == 257 and n257_meta["strategy"] == "main"
    assert exact_meta["source_hashes_before"] == exact_meta["source_hashes_after"]


def test_join_values_are_forwarded_when_exposed_by_assembly():
    module = _module()
    from torchwave.approximants.hm_assembly_clean import assemble_mode

    kwargs, metadata = module._assembly_option_kwargs(assemble_mode)
    assert metadata["amplitude_join"] == 0.014
    assert metadata["phase_join"] == 0.018
    assert metadata["base_fr22"] == "per_case_D_carrier"
    assert metadata["base_fd22"] == "per_case_D_carrier"
    parameters = inspect.signature(assemble_mode).parameters
    if "amplitude_join" in parameters:
        assert kwargs["amplitude_join"] == 0.014
    if "phase_join" in parameters:
        assert kwargs["phase_join"] == 0.018


def test_report_has_raw_mode_and_polarization_metrics():
    if not REPORT.exists():
        pytest.skip("diagnostic report has not been generated")
    report = json.loads(REPORT.read_text())
    assert report["status"] == "complete_diagnostic_only"
    assert report["source_hashes_changed_during_run"] == {}
    assert report["choices"]["base_fr22"] == "per_case_D_carrier"
    assert report["choices"]["base_fd22"] == "per_case_D_carrier"
    assert "HM QNM values define the outer frequency map" in report["provenance"]["eq14_carrier_contract"]
    assert "per-case D-carrier QNM contract" in report["provenance"]["note"]
    assert len(report["cases"]) == 4
    for case in report["cases"]:
        remnant = case["remnant"]
        assert "final_mass_solar_mass" in remnant
        assert remnant["final_mass_fraction"] == pytest.approx(
            remnant["final_mass_solar_mass"] / sum(case["inputs"][:2])
        )
        assert case["sample_count"] == len(report["choices"]["frequency_hz"])
        assert set(case["variants"]) == {"n61_main", "printed_london", "n257_main"}
        for variant_name, variant in case["variants"].items():
            assert set(variant["mode_errors"]) == {
                "(2, 2)",
                "(2, 1)",
                "(3, 3)",
                "(3, 2)",
                "(4, 4)",
                "(4, 3)",
            }
            for mode in variant["mode_errors"].values():
                assert "vs_public_hm" in mode
                assert "raw_error_l2" in mode["vs_public_hm"]
                if variant_name != "printed_london":
                    assert "vs_printed_london" in mode
            for pol in ("hp", "hc"):
                assert "raw_error_l2" in variant["polarization_errors"][pol]["vs_public_hm"]
