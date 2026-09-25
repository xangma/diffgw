"""Evidence, inheritance, tuple slots, and option-map drift for tidal TaylorF2."""

import ast
import re

import pytest

pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import aligned_spin_traceability as spin_trace
from diffgw_reference import tidal_traceability as trace
from diffgw_reference import taylorf2 as base
from diffgw_reference import taylorf2_aligned_spin as spin
from diffgw_reference import taylorf2_tidal as ref
from diffgw_reference import traceability as base_trace
from test_math_reference_traceability import ROOT, resolve_function, source_tree


@pytest.mark.parametrize("selector, expected", [
    (0, ()), (9, ()), (10, ("T10",)), (11, ("T10",)),
    (12, ("T10", "T12")), (13, ("T10", "T12", "T13")),
    (14, ("T10", "T12", "T13")), (-1, ("T10", "T12", "T13")),
])
def test_tidal_frontend_selector_metadata(selector, expected):
    assert trace.frontend_terms(selector) == expected
    for present1, present2 in ((False, True), (True, False), (False, False)):
        assert trace.frontend_terms(selector, lambda1_present=present1, lambda2_present=present2) == ()


@pytest.mark.parametrize("bad", [True, 12.0, "12", None])
def test_tidal_frontend_selector_rejects_undocumented_types(bad):
    with pytest.raises(TypeError):
        trace.frontend_terms(bad)
    for flags in ({"lambda1_present": 1}, {"lambda2_present": None}):
        with pytest.raises(TypeError):
            trace.frontend_terms(12, **flags)


def test_tidal_named_profiles_match_explicit_options():
    for profile in trace.FRONTEND_PROFILES.values():
        assert profile.terms == trace.frontend_terms(
            profile.tidal_order, lambda1_present=profile.lambda1_present,
            lambda2_present=profile.lambda2_present,
        )
        assert profile.scope
    assert trace.FRONTEND_PROFILES["default"].tidal_order == 12
    assert trace.FRONTEND_PROFILES["default"].terms == ("T10", "T12")
    assert trace.FRONTEND_PROFILES["full"].terms == trace.TERM_ORDER


def test_tidal_inherited_components_keep_their_versions():
    assert ref.INHERITED_SPEC_VERSIONS == trace.INHERITED_SPEC_VERSIONS == {
        "nonspinning": "taylorf2-nonspinning-3.5pn/v1",
        "aligned_spin": "taylorf2-aligned-bh-spin-3.5pn/v1",
    }
    assert base.SPEC_VERSION == ref.INHERITED_SPEC_VERSIONS["nonspinning"]
    assert spin.SPEC_VERSION == ref.INHERITED_SPEC_VERSIONS["aligned_spin"]
    assert ref.INHERITED_COMPONENT_IDS == {
        "nonspinning": base.COMPONENT_IDS, "aligned_spin": spin.COMPONENT_IDS,
    }
    assert trace.INHERITED_COMPONENT_TRACES == {
        "nonspinning": base_trace.COMPONENT_TRACES, "aligned_spin": spin_trace.COMPONENT_TRACES,
    }
    assert trace.ALL_COMPONENT_TRACES == {
        **base_trace.COMPONENT_TRACES, **spin_trace.COMPONENT_TRACES, **trace.COMPONENT_TRACES,
    }
    assert len(trace.ALL_COMPONENT_TRACES) == sum(map(len, (
        base_trace.COMPONENT_TRACES, spin_trace.COMPONENT_TRACES, trace.COMPONENT_TRACES,
    )))
    assert trace.TERM_ORDER == ref.TIDAL_TERMS == ("T10", "T12", "T13")


def _validate_source(source, version):
    key, *equations = source
    if key == "project-convention":
        assert equations == [version]
        return
    metadata = ref.SOURCES[key]
    assert all(metadata.get(name) for name in ("arxiv", "url", "doi", "equations", "use"))
    assert re.fullmatch(r"\d{4}\.\d{4,5}v[1-9]\d*", metadata["arxiv"])
    assert metadata["url"] == f"https://arxiv.org/pdf/{metadata['arxiv']}"
    assert metadata["doi"].startswith("10.")
    assert equations and set(equations) <= set(metadata["equations"])
    for erratum in metadata.get("errata", ()):
        assert erratum.startswith("10.")


def test_tidal_every_component_has_complete_evidence():
    assert set(trace.COMPONENT_TRACES) == set(ref.COMPONENT_IDS.values())
    assert set(ref.COMPONENT_SOURCES) == set(trace.COMPONENT_TRACES)
    for module, traces in ((base, base_trace.COMPONENT_TRACES),
                           (spin, spin_trace.COMPONENT_TRACES), (ref, trace.COMPONENT_TRACES)):
        for component_id, item in traces.items():
            assert item.source == module.COMPONENT_SOURCES[component_id]
            _validate_source(item.source, module.SPEC_VERSION)
            assert item.selector and item.tests and len(set(item.tests)) == len(item.tests)
            assert item.torch.startswith("torchwave.") and item.jax.startswith("jaxwave.")
    assert {key for key, value in trace.ALL_COMPONENT_TRACES.items() if value.reference_only} == {
        base.COMPONENT_IDS["paper_phase"],
    }
    for item in trace.AUXILIARY_TRACES.values():
        assert item.sources and item.selector and item.tests
        for source in item.sources:
            _validate_source(source, ref.SPEC_VERSION)
    assert ref.COMPONENT_SOURCES[ref.COMPONENT_IDS["T13"]] == (
        "damour2012", "6", "B8", "B9", "B10", "B12", "B14",
    )
    rendered_evidence = {(source[0], equation) for item in trace.AUXILIARY_TRACES.values()
                         for source in item.sources for equation in source[1:]}
    rendered_functions = {item.reference for item in
                          (*trace.COMPONENT_TRACES.values(), *trace.AUXILIARY_TRACES.values())}
    assert set(ref.DERIVATIONS) == {"component_to_effective_parameters", ref.COMPONENT_IDS["T13"]}
    for derivation in ref.DERIVATIONS.values():
        for source in derivation["sources"]:
            _validate_source(source, ref.SPEC_VERSION)
            assert {(source[0], equation) for equation in source[1:]} <= rendered_evidence
        for name in derivation.get("expressions", ()):
            assert f"diffgw_reference.taylorf2_tidal:{name}" in rendered_functions
    assert all(ref.DERIVATIONS["component_to_effective_parameters"].get(key)
               for key in ("sources", "expressions", "coordinates", "units"))
    assert all(ref.DERIVATIONS[ref.COMPONENT_IDS["T13"]].get(key)
               for key in ("sources", "normalization", "tail"))


@pytest.mark.parametrize("qualified", sorted(
    {getattr(item, field) for item in trace.ALL_COMPONENT_TRACES.values()
     for field in ("reference", "torch", "jax")}
    | {item.reference for item in trace.AUXILIARY_TRACES.values()}
))
def test_tidal_reference_and_frontend_links_resolve(qualified):
    resolve_function(qualified)


@pytest.mark.parametrize("nodeid", sorted(
    {test for item in (*trace.ALL_COMPONENT_TRACES.values(), *trace.AUXILIARY_TRACES.values())
     for test in item.tests}
))
def test_tidal_maintained_check_links_resolve(nodeid):
    filename, name = nodeid.split("::")
    assert filename.startswith("tests/test_") and name.startswith("test_")
    tree = source_tree(ROOT / filename)
    assert any(isinstance(node, ast.FunctionDef) and node.name == name for node in tree.body), nodeid


def _assert_tuple_return(qualified, expected):
    function = resolve_function(qualified)
    returns = [node.value for node in function.body if isinstance(node, ast.Return)]
    assert len(returns) == 1 and isinstance(returns[0], ast.Tuple)
    assert all(isinstance(value, ast.Name) for value in returns[0].elts)
    assert tuple(value.id for value in returns[0].elts) == expected


def test_tidal_coefficient_and_combination_slots_are_explicit():
    assert trace.PHASE_COEFFICIENT_ORDER == base_trace.PHASE_COEFFICIENT_ORDER
    assert trace.COMBINATION_ORDER == ("lambda_tilde", "delta_lambda_tilde")
    assert trace.CORRECTION_COEFFICIENTS == {
        "T10": "phi10_tidal", "T12": "phi12_tidal", "T13": "phi13_tidal",
    }
    assert set(trace.CORRECTION_COEFFICIENTS) == set(ref.tidal_corrections())
    assert set(trace.COMBINATION_ORDER) == set(ref.tidal_parameters())
    for name, coefficient in trace.CORRECTION_COEFFICIENTS.items():
        item = trace.COMPONENT_TRACES[ref.COMPONENT_IDS[name]]
        assert item.selector == name
        assert trace.PHASE_COEFFICIENT_ORDER[item.coefficient_index] == coefficient
    for frontend in ("torchwave", "jaxwave"):
        _assert_tuple_return(f"{frontend}.approximants.taylorf2:TaylorF2.phase_coefficients",
                             trace.PHASE_COEFFICIENT_ORDER)
        _assert_tuple_return(f"{frontend}.conversions:tidal_combinations", trace.COMBINATION_ORDER)


@pytest.mark.parametrize("start, end, render", [
    (trace.PROFILES_START, trace.PROFILES_END, trace.profile_table),
    (trace.TABLE_START, trace.TABLE_END, trace.traceability_table),
])
def test_tidal_documented_maps_are_current(start, end, render):
    document = (ROOT / "docs/math/taylorf2-tidal.md").read_text()
    assert document.count(start) == document.count(end) == 1
    assert start + document.split(start)[1].split(end)[0] + end == render()
