"""Structural evidence and frontend selector contracts for the spin extension."""

import ast
import re

import pytest

pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import aligned_spin_traceability as trace
from diffgw_reference import taylorf2 as base
from diffgw_reference import taylorf2_aligned_spin as ref
from diffgw_reference import traceability as base_trace
from test_math_reference_traceability import ROOT, resolve_function, source_tree


@pytest.mark.parametrize("selector, expected", [
    (0, ()), (2, ()), (3, ("SO3",)), (4, ("SO3", "SS4")),
    (5, ("SO3", "SS4", "SO5", "SO6", "SO7")),
    (6, ("SO3", "SS4", "SO5", "SO6", "SS6", "SO7")),
    (7, ("SO3", "SS4", "SO5", "SO6", "SS6", "SO7")),
    (-1, ("SO3", "SS4", "SO5", "SO6", "SS6", "SO7")),
    (8, ("SO3", "SS4", "SO5", "SO6", "SS6", "SO7", "SS7", "SSS7")),
])
def test_frontend_selector_metadata(selector, expected):
    assert trace.frontend_terms(selector) == expected
    calibrated = tuple(term for term in expected if term not in ("SS6", "SS7", "SSS7"))
    assert trace.frontend_terms(selector, phenom_d_calibration=True) == calibrated


@pytest.mark.parametrize("bad", [True, 5.0, "5", None])
def test_frontend_selector_metadata_rejects_undocumented_types(bad):
    with pytest.raises(TypeError):
        trace.frontend_terms(bad)
    with pytest.raises(TypeError):
        trace.frontend_terms(5, phenom_d_calibration=1)


def test_named_frontend_profiles_match_explicit_options():
    for profile in trace.FRONTEND_PROFILES.values():
        assert profile.terms == trace.frontend_terms(
            profile.spin_order, phenom_d_calibration=profile.phenom_d_calibration,
        )
        assert profile.scope
    assert trace.FRONTEND_PROFILES["public_default"].spin_order == 7
    assert trace.FRONTEND_PROFILES["phase_default"].spin_order == 5
    assert trace.FRONTEND_PROFILES["full"].terms == trace.TERM_ORDER


def test_aligned_spin_inherited_components_keep_their_versions():
    assert ref.INHERITED_SPEC_VERSION == base.SPEC_VERSION == "taylorf2-nonspinning-3.5pn/v1"
    assert trace.INHERITED_SPEC_VERSION == ref.INHERITED_SPEC_VERSION
    assert ref.INHERITED_COMPONENT_IDS == base.COMPONENT_IDS
    assert trace.INHERITED_COMPONENT_TRACES == base_trace.COMPONENT_TRACES
    assert not (set(trace.COMPONENT_TRACES) & set(trace.INHERITED_COMPONENT_TRACES))
    assert trace.ALL_COMPONENT_TRACES == {**base_trace.COMPONENT_TRACES, **trace.COMPONENT_TRACES}
    assert trace.TERM_ORDER == ref.SPIN_TERMS


def _validate_source(source, convention_version):
    key, *equations = source
    if key == "project-convention":
        assert equations == [convention_version]
        return
    metadata = ref.SOURCES[key]
    assert all(metadata.get(name) for name in ("arxiv", "url", "doi", "equations", "use"))
    assert re.fullmatch(r"\d{4}\.\d{4,5}v[1-9]\d*", metadata["arxiv"])
    assert metadata["url"] == f"https://arxiv.org/pdf/{metadata['arxiv']}"
    assert metadata["doi"].startswith("10.")
    assert equations and set(equations) <= set(metadata["equations"])
    for erratum in metadata.get("errata", ()):
        assert erratum.startswith("10.")


def test_aligned_spin_every_component_has_complete_evidence():
    assert set(trace.COMPONENT_TRACES) == set(ref.COMPONENT_IDS.values())
    assert set(ref.COMPONENT_SOURCES) == set(trace.COMPONENT_TRACES)
    for component_id, item in trace.ALL_COMPONENT_TRACES.items():
        inherited = component_id in trace.INHERITED_COMPONENT_TRACES
        sources = base.COMPONENT_SOURCES if inherited else ref.COMPONENT_SOURCES
        version = base.SPEC_VERSION if inherited else ref.SPEC_VERSION
        assert item.source == sources[component_id]
        _validate_source(item.source, version)
        assert item.selector and item.tests and len(set(item.tests)) == len(item.tests)
        assert item.torch.startswith("torchwave.approximants.taylorf2:")
        assert item.jax.startswith("jaxwave.approximants.taylorf2:")
    assert {key for key, value in trace.ALL_COMPONENT_TRACES.items() if value.reference_only} == {
        base.COMPONENT_IDS["paper_phase"],
    }


def test_aligned_spin_derivation_evidence_is_complete():
    assert set(trace.DERIVATION_REFERENCES) == set(trace.DERIVATION_CHECKS) == set(ref.DERIVATIONS)
    for component_id, derivation in ref.DERIVATIONS.items():
        assert component_id in trace.COMPONENT_TRACES
        assert all(derivation.get(name) for name in
                   ("sources", "expression", "normalization", "balance", "spa", "comparison"))
        assert trace.DERIVATION_REFERENCES[component_id] == (
            f"diffgw_reference.taylorf2_aligned_spin:{derivation['expression']}"
        )
        for source in derivation["sources"]:
            _validate_source(source, ref.SPEC_VERSION)
        assert trace.DERIVATION_CHECKS[component_id]
    assert ref.COMPONENT_SOURCES[ref.COMPONENT_IDS["SS7"]] == ("pratten2020", "A19", "A24")


@pytest.mark.parametrize("qualified", sorted(
    {getattr(item, field) for item in trace.ALL_COMPONENT_TRACES.values()
     for field in ("reference", "torch", "jax")} | set(trace.DERIVATION_REFERENCES.values())
))
def test_aligned_spin_reference_and_frontend_links_resolve(qualified):
    resolve_function(qualified)


@pytest.mark.parametrize("nodeid", sorted(
    {test for item in trace.ALL_COMPONENT_TRACES.values() for test in item.tests}
    | {test for checks in trace.DERIVATION_CHECKS.values() for test in checks}
))
def test_aligned_spin_maintained_check_links_resolve(nodeid):
    filename, name = nodeid.split("::")
    assert filename.startswith("tests/test_") and name.startswith("test_")
    tree = source_tree(ROOT / filename)
    assert any(isinstance(node, ast.FunctionDef) and node.name == name for node in tree.body), nodeid


def test_aligned_spin_coefficient_slots_are_explicit():
    assert trace.PHASE_COEFFICIENT_ORDER == base_trace.PHASE_COEFFICIENT_ORDER
    assert trace.CORRECTION_COEFFICIENTS == {
        "SO3": "phi3", "SS4": "phi4", "SO5": "phi5", "SO5_log": "phi5_log",
        "SO6": "phi6", "SS6": "phi6", "SO7": "phi7", "SS7": "phi7", "SSS7": "phi7",
    }
    assert set(trace.CORRECTION_COEFFICIENTS) == set(ref.spin_corrections())
    for name, coefficient in trace.CORRECTION_COEFFICIENTS.items():
        item = trace.COMPONENT_TRACES[ref.COMPONENT_IDS[name]]
        assert item.selector == name
        assert trace.PHASE_COEFFICIENT_ORDER[item.coefficient_index] == coefficient
    for frontend in ("torchwave", "jaxwave"):
        function = resolve_function(f"{frontend}.approximants.taylorf2:TaylorF2.phase_coefficients")
        returns = [node.value for node in function.body if isinstance(node, ast.Return)]
        assert len(returns) == 1 and isinstance(returns[0], ast.Tuple)
        assert all(isinstance(value, ast.Name) for value in returns[0].elts)
        assert tuple(value.id for value in returns[0].elts) == trace.PHASE_COEFFICIENT_ORDER


@pytest.mark.parametrize("start, end, render", [
    (trace.PROFILES_START, trace.PROFILES_END, trace.profile_table),
    (trace.TABLE_START, trace.TABLE_END, trace.traceability_table),
])
def test_aligned_spin_documented_maps_are_current(start, end, render):
    document = (ROOT / "docs/math/taylorf2-aligned-spin.md").read_text()
    assert document.count(start) == document.count(end) == 1
    assert start + document.split(start)[1].split(end)[0] + end == render()
