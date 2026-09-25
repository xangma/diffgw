"""Ensure every spin-quadrupole component retains its source and executable links."""

import ast
import json

import pytest

pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import spin_quadrupole_artifacts as artifacts
from diffgw_reference import spin_quadrupole_traceability as trace
from diffgw_reference import taylorf2_spin_quadrupole as ref
from test_math_reference_traceability import ROOT, resolve_function, source_tree


def test_spin_quadrupole_evidence_and_inheritance():
    assert trace.INHERITED_SPEC_VERSIONS == ref.INHERITED_SPEC_VERSIONS
    assert set(trace.COMPONENT_TRACES) == set(ref.COMPONENT_IDS.values())
    assert set(ref.COMPONENT_SOURCES) == set(ref.COMPONENT_IDS.values())
    assert len(trace.ALL_COMPONENT_TRACES) == sum(map(len, (
        *trace.INHERITED_COMPONENT_TRACES.values(), trace.COMPONENT_TRACES,
    )))
    for component_id, item in trace.COMPONENT_TRACES.items():
        assert item.source == ref.COMPONENT_SOURCES[component_id]
        source, *equations = item.source
        assert item.selector and item.tests
        if source == "project-convention":
            assert equations == [ref.SPEC_VERSION]
        else:
            assert source in ("nagar2018", "dietrich2019")
            assert set(equations) <= set(ref.SOURCES[source]["equations"])
            assert ref.SOURCES[source]["url"] == f"https://arxiv.org/pdf/{ref.SOURCES[source]['arxiv']}"
        for qualified in (item.reference, item.torch, item.jax):
            assert resolve_function(qualified)
        for check in item.tests:
            path, name = check.split("::")
            assert path == "tests/test_taylorf2_spin_quadrupole_math_reference.py"
            assert any(isinstance(node, ast.FunctionDef) and node.name == name
                       for node in source_tree(ROOT / path).body)
    for name in ref.QUAD_TERMS:
        item = trace.COMPONENT_TRACES[ref.COMPONENT_IDS[name]]
        assert item.coefficient_index == trace.PHASE_COEFFICIENT_ORDER.index(
            trace.TERM_COEFFICIENTS[name])


def test_spin_quadrupole_frontend_selector_metadata():
    expected = {0: (), 4: ("Q4",), 5: ("Q4",), 6: ("Q4", "Q6"),
                7: ("Q4", "Q6"), 8: ref.QUAD_TERMS, -1: ("Q4", "Q6")}
    for selector, terms in expected.items():
        assert trace.frontend_terms(selector) == terms
    assert trace.frontend_terms(8, True) == ("Q4",)
    for bad in (True, 8., "8", None):
        with pytest.raises(TypeError):
            trace.frontend_terms(bad)


def test_spin_quadrupole_artifacts_are_current():
    data = json.loads((ROOT / "tests/data/taylorf2_spin_quadrupole_reference_v1.json").read_text())
    assert data == json.loads(json.dumps(artifacts.fixtures()))
    assert data["spec_version"] == ref.SPEC_VERSION
    assert {case["name"] for case in data["cases"]} == {
        "equal_mass", "unequal_mass", "black_hole_neutron_star", "kerr_limit",
    }
    assert {case["reference_frequency"] for case in data["cases"]} == {False, True}
    assert (ROOT / "docs/math/taylorf2-spin-quadrupole-coefficients.md").read_text() == artifacts.coefficient_table()
