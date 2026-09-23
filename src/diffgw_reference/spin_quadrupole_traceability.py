"""Evidence and frontend links for the non-Kerr TaylorF2 spin quadrupole slice."""

from . import taylorf2_spin_quadrupole as ref
from . import traceability as base_trace
from . import aligned_spin_traceability as aligned_trace
from . import tidal_traceability as tidal_trace


INHERITED_SPEC_VERSIONS = dict(ref.INHERITED_SPEC_VERSIONS)
INHERITED_COMPONENT_TRACES = {
    "nonspinning": dict(base_trace.COMPONENT_TRACES),
    "aligned_spin": dict(aligned_trace.COMPONENT_TRACES),
    "tidal": dict(tidal_trace.COMPONENT_TRACES),
}
PHASE_COEFFICIENT_ORDER = base_trace.PHASE_COEFFICIENT_ORDER
TERM_COEFFICIENTS = {"Q4": "phi4", "Q6": "phi6", "Q7_tail": "phi7", "Q7_cubic": "phi7"}
_TEST_FILE = "tests/test_taylorf2_spin_quadrupole_math_reference.py"


def _checks(*names):
    return tuple(f"{_TEST_FILE}::test_quadrupole_{name}" for name in names)


def _trace(name, reference_function, selector, frontend_function, tests,
           coefficient_index=None):
    return base_trace.ComponentTrace(
        source=ref.COMPONENT_SOURCES[ref.COMPONENT_IDS[name]],
        reference=f"diffgw_reference.taylorf2_spin_quadrupole:{reference_function}",
        selector=selector,
        torch=f"torchwave.approximants.taylorf2:TaylorF2.{frontend_function}",
        jax=f"jaxwave.approximants.taylorf2:TaylorF2.{frontend_function}",
        tests=tests,
        coefficient_index=coefficient_index,
    )


COMPONENT_TRACES = {
    ref.COMPONENT_IDS[name]: _trace(
        name, "quadrupole_corrections", name, "phase_coefficients",
        _checks("paper_identities", "selector_and_coefficients", "batch_and_compilation"),
        PHASE_COEFFICIENT_ORDER.index(TERM_COEFFICIENTS[name]),
    ) for name in ref.QUAD_TERMS
}
COMPONENT_TRACES.update({
    ref.COMPONENT_IDS[name]: _trace(
        name, function, selector, frontend,
        _checks("waveform_and_phase", "ad_derivatives", "kerr_zero_and_omitted_defaults"),
    ) for name, function, selector, frontend in (
        ("intrinsic_phase", "phase_expr", "inherited plus selected quadrupole terms", "evaluate_phase"),
        ("coalescence_phase", "waveform_expr", "reference_frequency=False", "evaluate_phase"),
        ("reference_phase", "waveform_expr", "reference_frequency=True", "evaluate_phase"),
        ("polarizations", "waveform_expr", "h_plus, h_cross", "evaluate"),
    )
})
ALL_COMPONENT_TRACES = {
    **INHERITED_COMPONENT_TRACES["nonspinning"],
    **INHERITED_COMPONENT_TRACES["aligned_spin"],
    **INHERITED_COMPONENT_TRACES["tidal"],
    **COMPONENT_TRACES,
}


def frontend_terms(spin_order: int, phenom_d_calibration: bool = False) -> tuple[str, ...]:
    return ref.frontend_terms(spin_order, phenom_d_calibration)
