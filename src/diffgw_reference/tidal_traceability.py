"""Tidal evidence and frontend option adapters, independent of production imports.

The reference selects physical terms. This metadata separately describes the
frontends' integer selector and missing-argument behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import aligned_spin_traceability as spin_trace
from . import taylorf2_tidal as ref
from . import traceability as base_trace


TERM_ORDER = ("T10", "T12", "T13")


def frontend_terms(tidal_order: int, *, lambda1_present: bool = True,
                   lambda2_present: bool = True) -> tuple[str, ...]:
    """Map documented frontend options; presence means supplied rather than None.

    Numerical zero is a supplied deformability. Omission of either argument
    disables all tides. Undocumented noninteger selectors are outside this map.
    """
    if isinstance(tidal_order, bool) or not isinstance(tidal_order, int):
        raise TypeError("tidal_order must be an integer")
    if not isinstance(lambda1_present, bool) or not isinstance(lambda2_present, bool):
        raise TypeError("deformability presence flags must be boolean")
    if not lambda1_present or not lambda2_present:
        return ()
    return tuple(term for term, order in (("T10", 10), ("T12", 12), ("T13", 13))
                 if tidal_order >= order or tidal_order < 0)


@dataclass(frozen=True)
class FrontendProfile:
    tidal_order: int
    lambda1_present: bool
    lambda2_present: bool
    terms: tuple[str, ...]
    scope: str


FRONTEND_PROFILES = {
    "default": FrontendProfile(12, True, True, frontend_terms(12), "public waveform and low-level phase"),
    "full": FrontendProfile(13, True, True, frontend_terms(13), "all reference tidal terms"),
    "negative": FrontendProfile(-1, True, True, frontend_terms(-1), "all reference tidal terms"),
    "missing_lambda1": FrontendProfile(13, False, True, (), "all tides disabled"),
    "missing_lambda2": FrontendProfile(13, True, False, (), "all tides disabled"),
}

PROFILES_START = "<!-- BEGIN GENERATED TIDAL PROFILES -->"
PROFILES_END = "<!-- END GENERATED TIDAL PROFILES -->"


def profile_table() -> str:
    lines = [PROFILES_START, "", "| `tidal_order` | Both deformabilities supplied | Either is `None` |",
             "| --- | --- | --- |"]
    for label, selector in (("0 through 9", 0), ("10, 11", 10), ("12 (default)", 12),
                            ("13 or greater", 13), ("negative", -1)):
        terms = ", ".join(f"`{term}`" for term in frontend_terms(selector)) or "None"
        lines.append(f"| {label} | {terms} | None |")
    return "\n".join([*lines, "", PROFILES_END])


PHASE_COEFFICIENT_ORDER = base_trace.PHASE_COEFFICIENT_ORDER
COMBINATION_ORDER = ("lambda_tilde", "delta_lambda_tilde")
CORRECTION_COEFFICIENTS = {"T10": "phi10_tidal", "T12": "phi12_tidal", "T13": "phi13_tidal"}
INHERITED_SPEC_VERSIONS = dict(ref.INHERITED_SPEC_VERSIONS)
INHERITED_COMPONENT_TRACES = {
    "nonspinning": dict(base_trace.COMPONENT_TRACES),
    "aligned_spin": dict(spin_trace.COMPONENT_TRACES),
}
_TEST_FILE = "tests/test_taylorf2_tidal_math_reference.py"


def _checks(*names):
    return tuple(f"{_TEST_FILE}::test_tidal_{name}" for name in names)


_COEFFICIENT_CHECKS = _checks("coefficients_against_reference", "selector_contract",
                             "default_and_missing_arguments", "zero_reduction_and_exchange")
_PHASE_CHECKS = _checks("values_against_reference", "parameter_jacobian", "frequency_derivative",
                       "seeded_parameter_sweep", "batch_and_compilation", "transformed_jacobian",
                       "default_and_missing_arguments", "zero_reduction_and_exchange")
_WAVEFORM_CHECKS = _PHASE_CHECKS + _checks("label_exchange", "reference_domains_and_convergence",
                                          "reference_artifacts_are_current")


def _trace(name, reference_function, selector, frontend_function, tests, **kwargs):
    return base_trace.ComponentTrace(
        source=ref.COMPONENT_SOURCES[ref.COMPONENT_IDS[name]],
        reference=f"diffgw_reference.taylorf2_tidal:{reference_function}",
        selector=selector,
        torch=f"torchwave.{frontend_function}",
        jax=f"jaxwave.{frontend_function}",
        tests=tests,
        **kwargs,
    )


COMPONENT_TRACES = {
    ref.COMPONENT_IDS[name]: _trace(
        name, "tidal_corrections", name, "approximants.taylorf2:TaylorF2.phase_coefficients",
        _COEFFICIENT_CHECKS, coefficient_index=PHASE_COEFFICIENT_ORDER.index(coefficient),
    ) for name, coefficient in CORRECTION_COEFFICIENTS.items()
}
COMPONENT_TRACES.update({
    ref.COMPONENT_IDS[name]: _trace(
        name, "tidal_parameters", f"{name}; frontend tuple[{index}]", "conversions:tidal_combinations",
        _checks("combinations_against_reference", "zero_reduction_and_exchange", "label_exchange",
                "transformed_jacobian"),
    ) for index, name in enumerate(COMBINATION_ORDER)
})
COMPONENT_TRACES.update({
    ref.COMPONENT_IDS["intrinsic_phase"]: _trace(
        "intrinsic_phase", "phase_expr", "inherited phase plus selected tides",
        "approximants.taylorf2:TaylorF2.evaluate_phase", _PHASE_CHECKS,
    ),
    ref.COMPONENT_IDS["coalescence_phase"]: _trace(
        "coalescence_phase", "waveform_expr", "phase; reference_frequency=False",
        "approximants.taylorf2:TaylorF2.evaluate_phase", _PHASE_CHECKS,
    ),
    ref.COMPONENT_IDS["reference_phase"]: _trace(
        "reference_phase", "waveform_expr", "phase; reference_frequency=True",
        "approximants.taylorf2:TaylorF2.evaluate_phase", _WAVEFORM_CHECKS,
    ),
    ref.COMPONENT_IDS["polarizations"]: _trace(
        "polarizations", "waveform_expr", "h_plus, h_cross; both phase conventions",
        "approximants.taylorf2:TaylorF2.evaluate", _WAVEFORM_CHECKS,
    ),
})
ALL_COMPONENT_TRACES = {
    **INHERITED_COMPONENT_TRACES["nonspinning"],
    **INHERITED_COMPONENT_TRACES["aligned_spin"], **COMPONENT_TRACES,
}


@dataclass(frozen=True)
class AuxiliaryTrace:
    sources: tuple[tuple[str, ...], ...]
    reference: str
    selector: str
    tests: tuple[str, ...]


AUXILIARY_TRACES = {
    "Effective-parameter coefficients": AuxiliaryTrace(
        (("wade2014", "4", "5", "6", "A26"), ("damour2012", "B12")),
        "diffgw_reference.taylorf2_tidal:effective_tidal_corrections", "T10, T12, T13",
        _checks("zero_reduction_and_exchange", "coefficients_against_reference"),
    ),
    "Isolated additive tidal phase": AuxiliaryTrace(
        (("wade2014", "A26"), ("damour2012", "6", "B8", "B9", "B10", "B11", "B12", "B14")),
        "diffgw_reference.taylorf2_tidal:tidal_phase_expr", "3/(128 eta) times selected v powers",
        _checks("zero_reduction_and_exchange", "values_against_reference", "frequency_derivative"),
    ),
}

TABLE_START = "<!-- BEGIN GENERATED TIDAL TRACEABILITY -->"
TABLE_END = "<!-- END GENERATED TIDAL TRACEABILITY -->"
_REPOSITORY = "https://github.com/xangma/diffgw/blob/main"


def _evidence(source):
    key, *equations = source
    if key == "project-convention":
        return f"Project convention `{equations[0]}`"
    return f"[{key}]({ref.SOURCES[key]['url']}), Eq. " + ", ".join(equations)


def traceability_table() -> str:
    checks = tuple(dict.fromkeys(
        test for item in (*ALL_COMPONENT_TRACES.values(), *AUXILIARY_TRACES.values()) for test in item.tests
    ))
    check_ids = {test: f"T{index:02}" for index, test in enumerate(checks, start=1)}

    def check_links(tests):
        return ", ".join(f"[{check_ids[test]}](#tidal-trace-{check_ids[test].lower()})" for test in tests)

    def rows(traces):
        lines = ["| Component ID | Evidence | Reference expression | Frontends | Checks |",
                 "| --- | --- | --- | --- | --- |"]
        for component_id, item in traces.items():
            expression = base_trace._function_link(item.reference, item.reference.split(":")[1])
            expression += f"; {item.selector}"
            frontends = []
            for backend in ("torch", "jax"):
                qualified = getattr(item, backend)
                index = "" if item.coefficient_index is None else f"[{item.coefficient_index}]"
                label = f"{backend}.{qualified.split(':')[1].split('.')[-1]}{index}"
                frontends.append(base_trace._function_link(qualified, label))
            frontend_text = "; ".join(frontends)
            if item.reference_only:
                frontend_text += " (related adapters; reference-only expression)"
            lines.append(f"| `{component_id}` | {_evidence(item.source)} | {expression} | "
                         f"{frontend_text} | {check_links(item.tests)} |")
        return lines

    lines = [TABLE_START, "", f"New components: `{ref.SPEC_VERSION}`.", "", *rows(COMPONENT_TRACES)]
    for name, inherited in INHERITED_COMPONENT_TRACES.items():
        lines.extend(["", f"Unchanged inherited components: `{INHERITED_SPEC_VERSIONS[name]}`.",
                      "", *rows(inherited)])
    lines.extend(["", "Additional reference expressions:", "",
                  "| Expression | Evidence | Reference function | Checks |", "| --- | --- | --- | --- |"])
    for name, item in AUXILIARY_TRACES.items():
        evidence = "; ".join(_evidence(source) for source in item.sources)
        expression = base_trace._function_link(item.reference, item.reference.split(":")[1])
        lines.append(f"| {name}: {item.selector} | {evidence} | {expression} | {check_links(item.tests)} |")
    lines.extend(["", "Check labels are local to this table; function names identify maintained tests.",
                  "", "| Check | Maintained test |", "| --- | --- |"])
    for test in checks:
        filename, name = test.split("::")
        label = check_ids[test]
        lines.append(f'| <span id="tidal-trace-{label.lower()}">{label}</span> | '
                     f'[`{name}`]({_REPOSITORY}/{filename}) |')
    return "\n".join([*lines, "", TABLE_END])


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="DiffGW repository root")
    parser.add_argument("--check", action="store_true", help="Fail if the documented maps differ")
    args = parser.parse_args()
    path = args.root / "docs/math/taylorf2-tidal.md"
    current = path.read_text()
    updated = current
    for start, end, render in ((PROFILES_START, PROFILES_END, profile_table),
                               (TABLE_START, TABLE_END, traceability_table)):
        before, remainder = updated.split(start)
        _, after = remainder.split(end)
        updated = before + render() + after
    if args.check:
        if current != updated:
            raise SystemExit(f"Tidal traceability tables need regeneration: {path}")
    else:
        path.write_text(updated)
    print(f"{'Checked' if args.check else 'Wrote'} docs/math/taylorf2-tidal.md tables")


if __name__ == "__main__":
    main()
