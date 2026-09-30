"""Scalar London higher-mode phase offsets at supplied carrier and QNM inputs.

London et al., arXiv:1708.00404v4, Eqs. (3), (5)--(8), including the
PhenomD-specific high-frequency mapping in Eq. (8). No frontend,
calibration, or QNM implementation is imported.
"""

from __future__ import annotations

import mpmath as mp

from .imrphenomhm_frequency_map import eq8_phase_frequency_map


SOURCE = "https://arxiv.org/pdf/1708.00404v4"
COMPONENT_SOURCES = {
    # Historical component ID is retained for receipt compatibility; offsets
    # are Eq. (7) in the numbered PDF, not Eq. (9).
    "hm.phase.offsets.eq9.v1": (SOURCE, "Methods, Eqs. (3), (5)--(8)"),
}
COMPONENT_BINDINGS = {
    "hm.phase.offsets.eq9.v1": {
        "reference": "mapped_phase",
        "torch": ("torchwave.approximants.imrphenom_hm", "_production_mapped_phase_c1"),
        "jax": ("jaxwave.approximants.imrphenom_hm", "_production_mapped_phase_c1"),
        "tests": ("test_frontend_phase_values", "test_frontend_phase_derivatives"),
    },
}


def _inputs(f, f0, f_rd_lm, f_rd_22, ell: int, m: int):
    if not isinstance(ell, int) or not isinstance(m, int) or ell < 2 or not 1 <= m <= ell:
        raise ValueError("require ell>=2 and 1<=m<=ell")
    f, f0, fr, f22 = (mp.mpf(str(x)) for x in (f, f0, f_rd_lm, f_rd_22))
    if not (f > 0 and f0 > 0 and fr > f0 and f22 > 2 * f0 / m):
        raise ValueError("require f>0, 0<f0<f_rd_lm, and f_rd_22>2*f0/m")
    return f, f0, fr, f22


def low_frequency_offset(ell: int, m: int):
    """PN/symmetry offset of London's Eq. (7), in radians."""
    if not isinstance(ell, int) or not isinstance(m, int) or ell < 2 or not 1 <= m <= ell:
        raise ValueError("require ell>=2 and 1<=m<=ell")
    return mp.pi * (3 * ell + (ell + m) % 2 - 2) / 2


def phase_coefficients(f0, f_rd_lm, f_rd_22, ell: int, m: int, carrier_phase):
    """Return branch multipliers and continuity offsets for supplied phi_22."""
    _, f0, fr, f22 = _inputs(f0, f0, f_rd_lm, f_rd_22, ell, m)
    slope = (f22 - 2 * f0 / m) / (fr - f0)
    k_low, k_mid, k_high = mp.mpf(m) / 2, 1 / slope, fr / f22
    d_low = low_frequency_offset(ell, m)
    d_mid = d_low + (k_low - k_mid) * carrier_phase(2 * f0 / m)
    d_high = d_mid + (k_mid - k_high) * carrier_phase(f22)
    return ((k_low, k_mid, k_high), (d_low, d_mid, d_high))


def mapped_phase(f, f0, f_rd_lm, f_rd_22, ell: int, m: int, carrier_phase):
    """Evaluate London's continuous three-branch higher-mode phase."""
    f, f0, fr, f22 = _inputs(f, f0, f_rd_lm, f_rd_22, ell, m)
    kappa, delta = phase_coefficients(f0, fr, f22, ell, m, carrier_phase)
    branch = 0 if f <= f0 else 1 if f <= fr else 2
    g = eq8_phase_frequency_map(f, f0, fr, f22, m)
    return kappa[branch] * carrier_phase(g) + delta[branch]
