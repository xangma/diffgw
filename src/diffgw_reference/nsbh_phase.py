"""Conditional scalar PhenomNSBH phase in the explicit PhenomD gauge.

Thompson et al., arXiv:2002.08383v2, Sec. II.2. The remnant/QNM pair and
PhenomD fit rows are declared inputs to the composed PhenomD reference.
No production waveform implementation is imported here.
"""

import sympy as sp

from . import imrphenomd_waveform as phenomd
from . import nrtidalv2_phase as tidal

SPEC_VERSION = "nsbh-phase-explicit-composition/v1"
COMPONENT_ID = "nsbh.phase.explicit-composition.v1"
SOURCES = (
    "https://arxiv.org/html/2002.08383v2",
    "https://arxiv.org/html/1508.07253v2",
    "https://arxiv.org/pdf/1905.06011v2",
    "https://arxiv.org/pdf/1402.5156v1",
)


def phase_expr(spec: phenomd.WaveformSpec, frequency_hz, mass1, mass2,
               lambda2, *, tc=0, phic=0, convention="coalescence",
               f_ref_hz=None):
    """Return the composed phase with an explicitly conditioned PhenomD spec."""
    m1, m2, lam = map(phenomd.amp.exact, (mass1, mass2, lambda2))
    if m1 < m2 or m2 <= 0 or lam < 0:
        raise ValueError("require m1 >= m2 > 0 and Lambda2 >= 0")
    if convention not in ("coalescence", "reference_frequency"):
        raise ValueError("only explicit phase gauges are covered")
    _, _, baseline = spec.expressions(
        frequency_hz, tc=tc, phic=phic, convention=convention,
        f_ref_hz=f_ref_hz)
    if baseline == 0 and spec.m_sec * phenomd.amp.exact(frequency_hz) > sp.Rational(1, 5):
        raise ValueError("frequency is outside the PhenomD phase domain")
    delta = (m1 - m2) / (m1 + m2)
    correction = tidal.phase_expr(
        phenomd.amp.exact(frequency_hz), spec.m_sec, delta, sp.S.Zero, lam)
    return baseline + correction


def evaluate(*args, dps=70, **kwargs):
    """Evaluate the unwrapped phase at decimal precision."""
    return sp.N(phase_expr(*args, **kwargs), dps)
