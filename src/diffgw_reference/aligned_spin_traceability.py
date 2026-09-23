"""Aligned-spin evidence and frontend option adapters, separate from the kernel.

The mathematical reference selects named physical terms. This module describes
the existing frontend selectors; integer ``spin_order`` is a legacy recipe,
not a uniform physical PN cutoff. No production modules are imported here.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import taylorf2_aligned_spin as ref
from . import traceability as base_trace


TERM_ORDER = ("SO3", "SS4", "SO5", "SO6", "SS6", "SO7", "SS7", "SSS7")


def frontend_terms(spin_order: int, *, phenom_d_calibration: bool = False) -> tuple[str, ...]:
    """Translate documented integer frontend options to explicit physical terms.

    This adapter accepts a Python integer selector and boolean calibration flag;
    it does not define behavior for undocumented tensor or floating selectors.
    """
    if isinstance(spin_order, bool) or not isinstance(spin_order, int):
        raise TypeError("spin_order must be an integer")
    if not isinstance(phenom_d_calibration, bool):
        raise TypeError("phenom_d_calibration must be boolean")
    selected = set()
    if spin_order >= 3 or spin_order < 0:
        selected.add("SO3")
    if spin_order >= 4 or spin_order < 0:
        selected.add("SS4")
    if spin_order >= 5 or spin_order < 0:
        selected.update(("SO5", "SO6", "SO7"))
    if (spin_order >= 6 or spin_order < 0) and not phenom_d_calibration:
        selected.add("SS6")
    if spin_order >= 8 and not phenom_d_calibration:
        selected.update(("SS7", "SSS7"))
    return tuple(term for term in TERM_ORDER if term in selected)


@dataclass(frozen=True)
class FrontendProfile:
    spin_order: int
    phenom_d_calibration: bool
    terms: tuple[str, ...]
    scope: str


FRONTEND_PROFILES = {
    "public_default": FrontendProfile(7, False, frontend_terms(7), "public waveform"),
    "phase_default": FrontendProfile(5, False, frontend_terms(5), "low-level phase"),
    "full": FrontendProfile(8, False, frontend_terms(8), "public waveform; all reference terms"),
    "negative_default": FrontendProfile(-1, False, frontend_terms(-1), "public waveform"),
    "calibration_full": FrontendProfile(8, True, frontend_terms(8, phenom_d_calibration=True),
                                         "coefficient and low-level phase only"),
}

PROFILES_START = "<!-- BEGIN GENERATED ALIGNED SPIN PROFILES -->"
PROFILES_END = "<!-- END GENERATED ALIGNED SPIN PROFILES -->"


def profile_table() -> str:
    """Show every integer selector range and the calibration exception."""
    lines = [PROFILES_START, "", "| `spin_order` | Terms | With `phenom_d_calibration=True` |",
             "| --- | --- | --- |"]
    for label, selector in (("0, 1, 2", 0), ("3", 3), ("4", 4), ("5", 5),
                            ("6, 7", 7), ("8 or greater", 8), ("negative", -1)):
        terms = [", ".join(f"`{name}`" for name in frontend_terms(
            selector, phenom_d_calibration=calibration,
        )) or "None" for calibration in (False, True)]
        lines.append(f"| {label} | {terms[0]} | {terms[1]} |")
    return "\n".join([*lines, "", PROFILES_END])


# The mathematical module has no dependency on this frontend compatibility map.
PHASE_COEFFICIENT_ORDER = base_trace.PHASE_COEFFICIENT_ORDER
INHERITED_SPEC_VERSION = ref.INHERITED_SPEC_VERSION
INHERITED_COMPONENT_TRACES = dict(base_trace.COMPONENT_TRACES)

# These slots contain the sum of all selected terms, not an isolated correction.
CORRECTION_COEFFICIENTS = {
    "SO3": "phi3", "SS4": "phi4", "SO5": "phi5", "SO5_log": "phi5_log",
    "SO6": "phi6", "SS6": "phi6", "SO7": "phi7", "SS7": "phi7", "SSS7": "phi7",
}
_TEST_FILE = "tests/test_taylorf2_aligned_spin_math_reference.py"


def _checks(*names):
    return tuple(f"{_TEST_FILE}::test_aligned_spin_{name}" for name in names)


_COEFFICIENT_CHECKS = _checks("coefficients_against_reference", "corrections_by_spin_parity",
                             "selector_contract", "zero_spin_reduction")
_PHASE_CHECKS = _checks("values_against_reference", "default_profiles", "calibration_phase",
                       "parameter_jacobian", "frequency_derivative", "seeded_parameter_sweep",
                       "batch_and_compilation", "zero_spin_reduction", "zero_spin_backend_values")
_POLARIZATION_CHECKS = _checks("values_against_reference", "parameter_jacobian", "label_exchange",
                              "seeded_parameter_sweep", "batch_and_compilation", "extremal_spin_bounds",
                              "zero_spin_reduction", "zero_spin_backend_values",
                              "reference_precision_convergence", "reference_artifacts_are_current")


def _trace(name, reference_function, selector, frontend_function, tests, **kwargs):
    return base_trace.ComponentTrace(
        source=ref.COMPONENT_SOURCES[ref.COMPONENT_IDS[name]],
        reference=f"diffgw_reference.taylorf2_aligned_spin:{reference_function}",
        selector=selector,
        torch=f"torchwave.approximants.taylorf2:TaylorF2.{frontend_function}",
        jax=f"jaxwave.approximants.taylorf2:TaylorF2.{frontend_function}",
        tests=tests,
        **kwargs,
    )


COMPONENT_TRACES = {
    ref.COMPONENT_IDS[name]: _trace(
        name, "spin_corrections", name, "phase_coefficients", _COEFFICIENT_CHECKS,
        coefficient_index=PHASE_COEFFICIENT_ORDER.index(coefficient),
    )
    for name, coefficient in CORRECTION_COEFFICIENTS.items()
}
COMPONENT_TRACES.update({
    ref.COMPONENT_IDS["intrinsic_phase"]: _trace(
        "intrinsic_phase", "phase_expr", "composed intrinsic phase; explicit terms", "evaluate_phase",
        _PHASE_CHECKS + _checks("reference_domains"),
    ),
    ref.COMPONENT_IDS["coalescence_phase"]: _trace(
        "coalescence_phase", "waveform_expr", "phase; reference_frequency=False", "evaluate_phase",
        _PHASE_CHECKS,
    ),
    ref.COMPONENT_IDS["reference_phase"]: _trace(
        "reference_phase", "waveform_expr", "phase; reference_frequency=True", "evaluate_phase",
        _PHASE_CHECKS + _checks("reference_precision_convergence"),
    ),
    ref.COMPONENT_IDS["polarizations"]: _trace(
        "polarizations", "waveform_expr", "h_plus, h_cross; both phase modes", "evaluate",
        _POLARIZATION_CHECKS,
    ),
})
ALL_COMPONENT_TRACES = {**INHERITED_COMPONENT_TRACES, **COMPONENT_TRACES}
DERIVATION_REFERENCES = {
    ref.COMPONENT_IDS["SS7"]: "diffgw_reference.taylorf2_aligned_spin:ss7_balance_terms",
}
DERIVATION_CHECKS = {
    ref.COMPONENT_IDS["SS7"]: _checks("spa_spin7_balance_factor"),
}

TABLE_START = "<!-- BEGIN GENERATED ALIGNED SPIN TRACEABILITY -->"
TABLE_END = "<!-- END GENERATED ALIGNED SPIN TRACEABILITY -->"
_REPOSITORY = "https://github.com/xangma/diffgw/blob/main"


def _evidence(source):
    source_key, *locations = source
    if source_key == "project-convention":
        return f"Project convention `{locations[0]}`"
    return f"[{source_key}]({ref.SOURCES[source_key]['url']}), Eq. " + ", ".join(locations)


def traceability_table() -> str:
    """Render new and unchanged inherited components with resolvable checks."""
    checks = tuple(dict.fromkeys(
        test for tests in [*(item.tests for item in ALL_COMPONENT_TRACES.values()),
                           *DERIVATION_CHECKS.values()] for test in tests
    ))
    check_ids = {test: f"S{index:02}" for index, test in enumerate(checks, start=1)}

    def check_links(tests):
        return ", ".join(f"[{check_ids[test]}](#spin-trace-{check_ids[test].lower()})" for test in tests)

    def rows(traces):
        lines = ["| Component ID | Evidence | Reference expression | Frontends | Checks |",
                 "| --- | --- | --- | --- | --- |"]
        for component_id, item in traces.items():
            expression = base_trace._function_link(item.reference, item.reference.split(":")[1])
            expression += f"; {item.selector}"
            frontend_links = []
            for backend in ("torch", "jax"):
                qualified = getattr(item, backend)
                index = "" if item.coefficient_index is None else f"[{item.coefficient_index}]"
                frontend_links.append(base_trace._function_link(
                    qualified, f"{backend}.{qualified.split('.')[-1]}{index}",
                ))
            frontends = "; ".join(frontend_links)
            if item.reference_only:
                frontends += " (related adapters; reference-only expression)"
            lines.append(f"| `{component_id}` | {_evidence(item.source)} | {expression} | "
                         f"{frontends} | {check_links(item.tests)} |")
        return lines

    lines = [TABLE_START, "", f"New components: `{ref.SPEC_VERSION}`.", "", *rows(COMPONENT_TRACES),
             "", f"Unchanged inherited components: `{INHERITED_SPEC_VERSION}`.", "",
             *rows(INHERITED_COMPONENT_TRACES), "", "Additional independent derivation:", "",
             "| Component | Sources | Reference expression | Checks |", "| --- | --- | --- | --- |"]
    for component_id, qualified in DERIVATION_REFERENCES.items():
        evidence = "; ".join(_evidence(source) for source in ref.DERIVATIONS[component_id]["sources"])
        expression = base_trace._function_link(qualified, qualified.split(":")[1])
        lines.append(f"| `{component_id}` | {evidence} | {expression} | "
                     f"{check_links(DERIVATION_CHECKS[component_id])} |")
    lines.extend(["", "Check labels are local to this table; function names identify maintained tests.",
                  "", "| Check | Maintained test |", "| --- | --- |"])
    for test in checks:
        filename, name = test.split("::")
        label = check_ids[test]
        lines.append(f'| <span id="spin-trace-{label.lower()}">{label}</span> | '
                     f'[`{name}`]({_REPOSITORY}/{filename}) |')
    return "\n".join([*lines, "", TABLE_END])


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="DiffGW repository root")
    parser.add_argument("--check", action="store_true", help="Fail if the documented maps differ")
    args = parser.parse_args()
    path = args.root / "docs/math/taylorf2-aligned-spin.md"
    current = path.read_text()
    updated = current
    for start, end, render in ((PROFILES_START, PROFILES_END, profile_table),
                               (TABLE_START, TABLE_END, traceability_table)):
        before, remainder = updated.split(start)
        _, after = remainder.split(end)
        updated = before + render() + after
    if args.check:
        if current != updated:
            raise SystemExit(f"Aligned-spin traceability tables need regeneration: {path}")
    else:
        path.write_text(updated)
    print(f"{'Checked' if args.check else 'Wrote'} docs/math/taylorf2-aligned-spin.md tables")


if __name__ == "__main__":
    main()
