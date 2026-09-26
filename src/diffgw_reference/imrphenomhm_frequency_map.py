"""Scalar mathematical reference for London's HM frequency maps.

The input and output frequencies are positive, dimensionless ``Mf`` values.
No QNM fit or production waveform helper is imported here: the two QNM
frequencies are explicit inputs.  London et al., arXiv:1708.00404v4,
Eq. (5) gives the generic map; Eq. (8) changes only the final phase branch
in its IMRPhenomD application.
"""

from __future__ import annotations

import mpmath as mp
import sympy as sp


SOURCE = "https://arxiv.org/pdf/1708.00404v4"
COMPONENT_SOURCES = {
    "hm.fmap.eq5.v1": (SOURCE, "Eq. (5), page 2"),
    "hm.fmap.phase.eq8.v1": (SOURCE, "Eq. (8), page 3; Eq. (5) below ringdown"),
}
COMPONENT_BINDINGS = {
    "hm.fmap.eq5.v1": {
        "reference": "eq5_frequency_map",
        "torch": ("torchwave.approximants.imrphenom_hm", "frequency_mapping_amplitude"),
        "jax": ("jaxwave.approximants.imrphenom_hm", "frequency_mapping_amplitude"),
        "tests": ("test_frontend_values_and_joins", "test_frontend_frequency_derivatives"),
    },
    "hm.fmap.phase.eq8.v1": {
        "reference": "eq8_phase_frequency_map",
        "torch": ("torchwave.approximants.imrphenom_hm", "frequency_mapping_phase"),
        "jax": ("jaxwave.approximants.imrphenom_hm", "frequency_mapping_phase"),
        "tests": ("test_frontend_values_and_joins", "test_frontend_frequency_derivatives"),
    },
}

F, F0, FR, F22 = sp.symbols("f f_0 f_RD_lm f_RD_22", positive=True)
M = sp.symbols("m", integer=True, positive=True)


def branch_expressions(*, phase_d_carrier: bool = False) -> tuple[sp.Expr, ...]:
    """Return the three exact affine branches of Eq. (5) or Eq. (8)."""
    low = 2 * F / M
    middle = (F22 - 2 * F0 / M) * (F - F0) / (FR - F0) + 2 * F0 / M
    high = F22 * F / FR if phase_d_carrier else F - (FR - F22)
    return low, middle, high


def symbolic_frequency_map(*, phase_d_carrier: bool = False) -> sp.Piecewise:
    """Use the paper's lower-frequency branch at each join."""
    low, middle, high = branch_expressions(phase_d_carrier=phase_d_carrier)
    return sp.Piecewise((low, F <= F0), (middle, F <= FR), (high, True))


def _checked_inputs(f, f0, f_rd_lm, f_rd_22, m):
    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer")
    f, f0, fr, f22 = (mp.mpf(str(value)) for value in (f, f0, f_rd_lm, f_rd_22))
    if not (f > 0 and f0 > 0 and fr > f0 and f22 > 0):
        raise ValueError("require f>0, 0<f0<f_rd_lm, and f_rd_22>0")
    return f, f0, fr, f22


def eq5_frequency_map(f, f0, f_rd_lm, f_rd_22, m: int) -> mp.mpf:
    """Evaluate Eq. (5) with arbitrary precision and explicit QNM inputs."""
    f, f0, fr, f22 = _checked_inputs(f, f0, f_rd_lm, f_rd_22, m)
    if f <= f0:
        return 2 * f / m
    if f <= fr:
        return 2 * f0 / m + (f22 - 2 * f0 / m) * (f - f0) / (fr - f0)
    return f - (fr - f22)


def eq8_phase_frequency_map(f, f0, f_rd_lm, f_rd_22, m: int) -> mp.mpf:
    """Evaluate Eq. (5) below ringdown and the Eq. (8) phase branch above."""
    f, f0, fr, f22 = _checked_inputs(f, f0, f_rd_lm, f_rd_22, m)
    if f <= fr:
        return eq5_frequency_map(f, f0, fr, f22, m)
    return f22 * f / fr


def branch_slopes(f0, f_rd_lm, f_rd_22, m: int, *, phase_d_carrier=False):
    """Return the analytic one-sided derivatives at either join."""
    _, f0, fr, f22 = _checked_inputs(f0, f0, f_rd_lm, f_rd_22, m)
    return (mp.mpf(2) / m,
            (f22 - 2 * f0 / m) / (fr - f0),
            f22 / fr if phase_d_carrier else mp.mpf(1))
