"""Detect dangling literature, implementation, test, and documentation links.

Source resolution avoids importing either optional frontend. Numerical agreement
is established by the linked tests; this suite checks that the map stays usable.
"""

import ast
from functools import lru_cache
from pathlib import Path
import re

import pytest

pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import taylorf2 as ref
from diffgw_reference import traceability as trace

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def source_tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def resolve_function(qualified):
    module, attribute = qualified.split(":")
    current = source_tree(ROOT / "src" / Path(*module.split(".")).with_suffix(".py"))
    for name in attribute.split("."):
        definitions = {node.name: node for node in current.body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        if name not in definitions:
            # Torch TaylorF2 exposes evaluate = __call__ in the class body.
            aliases = {target.id: node.value.id for node in current.body
                       if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name)
                       for target in node.targets if isinstance(target, ast.Name)}
            name = aliases.get(name, name)
        assert name in definitions, f"Unresolved function: {qualified}"
        current = definitions[name]
    assert isinstance(current, ast.FunctionDef), qualified
    return current


def test_every_component_has_complete_trace_and_evidence():
    assert set(trace.COMPONENT_TRACES) == set(ref.COMPONENT_IDS.values())
    assert set(ref.COMPONENT_SOURCES) == set(ref.COMPONENT_IDS.values())
    for component_id, item in trace.COMPONENT_TRACES.items():
        assert item.source == ref.COMPONENT_SOURCES[component_id]
        source_key, *equations = item.source
        assert item.selector and item.tests and len(set(item.tests)) == len(item.tests)
        assert item.torch.startswith("torchwave.approximants.taylorf2:")
        assert item.jax.startswith("jaxwave.approximants.taylorf2:")
        if source_key == "project-convention":
            assert equations == [ref.SPEC_VERSION]
        else:
            source = ref.SOURCES[source_key]
            assert all(source.get(key) for key in ("arxiv", "url", "doi", "equations", "use"))
            assert re.fullmatch(r"\d{4}\.\d{4,5}v[1-9]\d*", source["arxiv"])
            assert source["url"] == f"https://arxiv.org/pdf/{source['arxiv']}"
            assert source["doi"].startswith("10.")
            assert equations and set(equations) <= set(source["equations"])
    reference_only = {key for key, item in trace.COMPONENT_TRACES.items() if item.reference_only}
    assert reference_only == {ref.COMPONENT_IDS["paper_phase"]}


@pytest.mark.parametrize("qualified", sorted({getattr(item, field)
                                             for item in trace.COMPONENT_TRACES.values()
                                             for field in ("reference", "torch", "jax")}))
def test_reference_and_frontend_functions_resolve(qualified):
    resolve_function(qualified)


@pytest.mark.parametrize("nodeid", sorted({test for item in trace.COMPONENT_TRACES.values()
                                          for test in item.tests}))
def test_maintained_checks_resolve(nodeid):
    filename, name = nodeid.split("::")
    assert filename.startswith("tests/test_") and name.startswith("test_")
    tree = source_tree(ROOT / filename)
    assert any(isinstance(node, ast.FunctionDef) and node.name == name for node in tree.body), nodeid


def test_coefficient_tuple_order_is_explicit_and_matches_both_frontends():
    order = trace.PHASE_COEFFICIENT_ORDER
    assert len(order) == len(set(order)) == 12
    assert set(order[:9]) == set(ref.coefficients())
    assert order[9:] == ("phi10_tidal", "phi12_tidal", "phi13_tidal")
    for name in ref.coefficients():
        item = trace.COMPONENT_TRACES[ref.COMPONENT_IDS[name]]
        assert item.selector == name
        assert order[item.coefficient_index] == name
    for frontend in ("torchwave", "jaxwave"):
        function = resolve_function(f"{frontend}.approximants.taylorf2:TaylorF2.phase_coefficients")
        returns = [node.value for node in function.body if isinstance(node, ast.Return)]
        assert len(returns) == 1 and isinstance(returns[0], ast.Tuple)
        assert all(isinstance(value, ast.Name) for value in returns[0].elts)
        assert tuple(value.id for value in returns[0].elts) == order


def test_documented_component_map_is_current():
    document = (ROOT / "docs/math/taylorf2.md").read_text()
    assert document.count(trace.TABLE_START) == document.count(trace.TABLE_END) == 1
    table = trace.TABLE_START + document.split(trace.TABLE_START)[1].split(trace.TABLE_END)[0] + trace.TABLE_END
    assert table == trace.traceability_table()
