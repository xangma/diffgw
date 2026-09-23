"""TaylorF2 component links, kept separate from the mathematical kernel.

Qualified frontend names are metadata, never imports. The maintained tests
resolve these links from source without requiring either optional frontend.
Regenerate the embedded documentation with ``python -m
diffgw_reference.traceability``; ``--check`` reports documentation drift.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from . import taylorf2 as ref


# Public tuple positions in both TaylorF2.phase_coefficients implementations.
# Spell out the complete contract: reference dictionary order is not an API.
PHASE_COEFFICIENT_ORDER = (
    "phi0", "phi2", "phi3", "phi4", "phi5", "phi5_log", "phi6",
    "phi6_log", "phi7", "phi10_tidal", "phi12_tidal", "phi13_tidal",
)

_TEST_FILE = "tests/test_taylorf2_math_reference.py"
_COEFFICIENT_TEST = f"{_TEST_FILE}::test_pn_coefficients_against_exact_reference"
_VALUE_TEST = f"{_TEST_FILE}::test_values_against_reference"
_DERIVATIVE_TEST = f"{_TEST_FILE}::test_parameter_jacobian_against_symbolic_derivatives"
_GAUGE_TEST = f"{_TEST_FILE}::test_exact_equations_and_phase_gauge"
_AMPLITUDE_TEST = f"{_TEST_FILE}::test_newtonian_amplitude_reduction_and_polarization_limits"
_SWEEP_TESTS = tuple(f"{_TEST_FILE}::{name}" for name in (
    "test_seeded_parameter_sweep", "test_single_precision_parameter_sweep",
))
_LIMIT_TEST = f"{_TEST_FILE}::test_low_velocity_and_inclination_limits"
_NEIGHBOR_TEST = f"{_TEST_FILE}::test_reference_frequency_neighbors"
_SYMMETRY_TEST = f"{_TEST_FILE}::test_nonspinning_symmetries"
_COMPILED_TEST = f"{_TEST_FILE}::test_compiled_waveform_against_reference"
_GRID_TESTS = tuple(f"{_TEST_FILE}::{name}" for name in (
    "test_batch_and_jax_jit_against_scalar_reference",
    "test_per_binary_frequency_grid_and_reference_vector",
))


@dataclass(frozen=True)
class ComponentTrace:
    """A component's expression, evidence, frontend relationship and checks.

    ``reference_only`` means the frontend names identify the related adapter,
    not an implementation of the source-gauge expression. ``selector`` records
    the selected dictionary output or waveform mode. Coefficient tuple positions
    are explicit, and tested against both frontend contracts.
    """

    source: tuple[str, ...]
    reference: str
    selector: str
    torch: str
    jax: str
    tests: tuple[str, ...]
    reference_only: bool = False
    coefficient_index: int | None = None


def _trace(name, reference_function, selector, frontend_function, tests, **kwargs):
    return ComponentTrace(
        source=ref.COMPONENT_SOURCES[ref.COMPONENT_IDS[name]],
        reference=f"diffgw_reference.taylorf2:{reference_function}",
        selector=selector,
        torch=f"torchwave.approximants.taylorf2:TaylorF2.{frontend_function}",
        jax=f"jaxwave.approximants.taylorf2:TaylorF2.{frontend_function}",
        tests=tests,
        **kwargs,
    )


COMPONENT_TRACES = {
    ref.COMPONENT_IDS[name]: _trace(
        name, "coefficients", name, "phase_coefficients", (_COEFFICIENT_TEST,),
        coefficient_index=index,
    )
    for index, name in enumerate(PHASE_COEFFICIENT_ORDER)
    if name in ref.COMPONENT_IDS
}
COMPONENT_TRACES.update({
    ref.COMPONENT_IDS["intrinsic_phase"]: _trace(
        "intrinsic_phase", "phase_expr", "log(v) intrinsic phase", "evaluate_phase",
        (_GAUGE_TEST, _VALUE_TEST, _DERIVATIVE_TEST, _LIMIT_TEST, *_SWEEP_TESTS),
    ),
    ref.COMPONENT_IDS["paper_phase"]: _trace(
        "paper_phase", "paper_phase_expr", "paper gauge; related adapters only", "evaluate_phase",
        (_GAUGE_TEST,), reference_only=True,
    ),
    ref.COMPONENT_IDS["amplitude"]: _trace(
        "amplitude", "amplitude_expr", "positive amplitude", "evaluate_amplitude",
        (_AMPLITUDE_TEST, _VALUE_TEST, _DERIVATIVE_TEST, _LIMIT_TEST, *_SWEEP_TESTS),
    ),
    ref.COMPONENT_IDS["polarizations"]: _trace(
        "polarizations", "waveform_expr", "h_plus, h_cross; both phase modes", "evaluate",
        (_AMPLITUDE_TEST, _VALUE_TEST, _DERIVATIVE_TEST, _LIMIT_TEST,
         _SYMMETRY_TEST, _COMPILED_TEST, *_SWEEP_TESTS, *_GRID_TESTS),
    ),
    ref.COMPONENT_IDS["coalescence_phase"]: _trace(
        "coalescence_phase", "waveform_expr", "phase; reference_frequency=False", "evaluate_phase",
        (_VALUE_TEST, _DERIVATIVE_TEST, *_SWEEP_TESTS, *_GRID_TESTS),
    ),
    ref.COMPONENT_IDS["reference_phase"]: _trace(
        "reference_phase", "waveform_expr", "phase; reference_frequency=True", "evaluate_phase",
        (_GAUGE_TEST, _VALUE_TEST, _DERIVATIVE_TEST, _NEIGHBOR_TEST, *_SWEEP_TESTS, *_GRID_TESTS),
    ),
})

TABLE_START = "<!-- BEGIN GENERATED TAYLORF2 TRACEABILITY -->"
TABLE_END = "<!-- END GENERATED TAYLORF2 TRACEABILITY -->"
_REPOSITORY = "https://github.com/xangma/diffgw/blob/main"


def _function_link(qualified, label):
    module, _ = qualified.split(":")
    return f"[`{label}`]({_REPOSITORY}/src/{module.replace('.', '/')}.py)"


def traceability_table() -> str:
    """Render every canonical component, retaining full maintained test names."""
    checks = tuple(dict.fromkeys(test for item in COMPONENT_TRACES.values() for test in item.tests))
    check_ids = {test: f"T{index:02}" for index, test in enumerate(checks, start=1)}
    lines = [TABLE_START, "", "| Component ID | Evidence | Reference expression | Frontends | Checks |",
             "| --- | --- | --- | --- | --- |"]
    for component_id in ref.COMPONENT_IDS.values():
        trace = COMPONENT_TRACES[component_id]
        source_key, *locations = trace.source
        if source_key == "project-convention":
            evidence = f"Project convention `{locations[0]}`"
        else:
            evidence = f"[{source_key}]({ref.SOURCES[source_key]['url']}), Eq. " + ", ".join(locations)
        reference_name = trace.reference.split(":")[1]
        expression = _function_link(trace.reference, reference_name) + f"; {trace.selector}"
        method = trace.torch.split(".")[-1]
        index = "" if trace.coefficient_index is None else f"[{trace.coefficient_index}]"
        frontend_links = "; ".join(
            _function_link(getattr(trace, backend), f"{backend}.{method}{index}")
            for backend in ("torch", "jax")
        )
        if trace.reference_only:
            frontend_links += " (related adapters; reference-only expression)"
        check_links = ", ".join(f"[{check_ids[test]}](#trace-{check_ids[test].lower()})" for test in trace.tests)
        lines.append(f"| `{component_id}` | {evidence} | {expression} | {frontend_links} | {check_links} |")
    lines.extend(["", "Check labels below are local to this table; the function names identify the maintained tests.",
                  "", "| Check | Maintained test |", "| --- | --- |"])
    for test in checks:
        label = check_ids[test]
        filename, name = test.split("::")
        lines.append(f'| <span id="trace-{label.lower()}">{label}</span> | [`{name}`]({_REPOSITORY}/{filename}) |')
    return "\n".join([*lines, "", TABLE_END])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="DiffGW repository root")
    parser.add_argument("--check", action="store_true", help="Fail if the documented map differs")
    args = parser.parse_args()
    path = args.root / "docs/math/taylorf2.md"
    current = path.read_text()
    before, remainder = current.split(TABLE_START)
    _, after = remainder.split(TABLE_END)
    updated = before + traceability_table() + after
    if args.check:
        if current != updated:
            raise SystemExit(f"Traceability table needs regeneration: {path}")
    else:
        path.write_text(updated)
    print(f"{'Checked' if args.check else 'Wrote'} docs/math/taylorf2.md traceability table")


if __name__ == "__main__":
    main()
