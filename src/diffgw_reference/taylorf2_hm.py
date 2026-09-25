"""Scalar leading-amplitude TaylorF2 higher-mode carrier reference.

The spin-weighted harmonics are generated from Wigner d matrices rather than
the polynomial branches in either native frontend. The intrinsic phase composes
the independently specified aligned-spin TaylorF2 reference. This describes
the current carrier convention, including its common mode phase adapter; it
does not assert that the truncated carrier is a complete PN mode expansion.
"""

from __future__ import annotations

from functools import lru_cache
from collections.abc import Mapping

import sympy as sp
from sympy.physics.wigner import wigner_d_small

from . import taylorf2 as base
from . import taylorf2_aligned_spin as spin
from .aligned_spin_traceability import frontend_terms

SPEC_VERSION = "taylorf2-hm-leading-carrier/v1"
REFERENCE_VERSION = SPEC_VERSION
INHERITED_SPEC_VERSIONS = (base.SPEC_VERSION, spin.SPEC_VERSION)
MODES = ((2, 2), (2, 1), (3, 1), (3, 2), (3, 3), (4, 2), (4, 3), (4, 4))
DEFAULT_SPIN_TERMS = frontend_terms(7)
PHI0 = sp.symbols("phi0", real=True)
F, F_REF, M_SEC, D_SEC = base.F, base.F_REF, base.M_SEC, base.D_SEC
IOTA, TC, PHIC = base.IOTA, base.TC, base.PHIC
DELTA, CHI1, CHI2 = spin.DELTA, spin.CHI1, spin.CHI2
ETA = (1 - DELTA**2) / 4
SYMBOLS = {**spin.SYMBOLS, PHI0.name: PHI0}
COMPONENT_IDS = {
    **{f"ratio_{l}{m}": f"tf2.hm.ratio_{l}{m}.v1" for l, m in MODES},
    **{f"projection_{l}{m}": f"tf2.hm.projection_{l}{m}.v1" for l, m in MODES},
    "mode_phase": "tf2.hm.mode_phase.v1",
    "polarizations": "tf2.hm.polarizations.v1",
    "cutoff_frequency": "tf2.hm.cutoff_frequency.v1",
}


def amplitude_ratio(l: int, m: int, *, delta=DELTA, v=None) -> sp.Expr:
    """Derive the real carrier ratio from Mezzasoma--Yunes Eqs. (20)--(21i), (30).

    Eq. (21) gives the source's complex time-domain H_lm(x). Eq. (30)
    supplies sqrt(2/m)*x_m**(-7/4), where x_m=(2/m)**(2/3)*v**2.
    Dividing by the m=2 Newtonian amplitude leaves (m/2)**(2/3)*H_lm.
    Fixed unit-complex multipliers convert source mode phases to the native
    real-amplitude carrier convention; Fourier/sign choices are documented in
    the specification. ``v`` uses observed mode frequency, not 2f/m.
    """
    if (l, m) not in MODES:
        raise ValueError(f"unsupported mode {(l, m)}")
    delta = sp.sympify(delta)
    v = sp.symbols("v", positive=True) if v is None else sp.sympify(v)
    eta = (1 - delta**2) / 4
    r = sp.Rational
    x = r(2, m)**r(2, 3) * v**2
    source_h = {
        (2, 2): sp.S.One,
        (2, 1): sp.I * delta * sp.sqrt(x) / 3,
        (3, 1): sp.I * delta * sp.sqrt(x) / (12*sp.sqrt(14)),
        (3, 2): sp.sqrt(r(5, 7)) * (1 - 3*eta) * x / 3,
        (3, 3): -r(3, 4) * sp.I * sp.sqrt(r(15, 14)) * delta * sp.sqrt(x),
        (4, 2): sp.sqrt(5) * (1 - 3*eta) * x / 63,
        (4, 3): -r(9, 4) * sp.I * delta * (1 - 2*eta) * x**r(3, 2) / sp.sqrt(70),
        (4, 4): -r(8, 9) * sp.sqrt(r(5, 7)) * (1 - 3*eta) * x,
    }[(l, m)]
    carrier_factor = {
        (2, 2): 1, (2, 1): -sp.I, (3, 1): -sp.I, (3, 2): 1,
        (3, 3): sp.I, (4, 2): 1, (4, 3): sp.I, (4, 4): -1,
    }[(l, m)]
    return sp.simplify(r(m, 2)**r(2, 3) * carrier_factor * source_h)


@lru_cache(maxsize=None)
def _wigner(l: int) -> sp.Matrix:
    return wigner_d_small(l, IOTA)


def spin_weighted_harmonic(l: int, m: int) -> sp.Expr:
    """s=-2 harmonic in the basis fixed by Y_22=sqrt(5/64pi)(1+c)^2 e^2iφ.

    The (-1)^m factor converts SymPy's Wigner-d indexing to this project basis.
    """
    if (l, abs(m)) not in MODES:
        raise ValueError(f"unsupported harmonic {(l, m)}")
    return (sp.Integer(-1)**m * sp.sqrt((2*l + 1) / (4*sp.pi))
            * _wigner(l)[l-m, l-2] * sp.exp(sp.I*m*PHI0))


def projection(l: int, m: int) -> tuple[sp.Expr, sp.Expr]:
    """Positive-frequency plus/cross projection of the ±m mode pair."""
    yp = spin_weighted_harmonic(l, m)
    yn = spin_weighted_harmonic(l, -m)
    partner = (-1)**l * sp.conjugate(yp)
    return (yn + partner) / 2, sp.I * (yn - partner) / 2


def mode_phase(l: int, m: int, *, reference_frequency: bool = False,
               terms=DEFAULT_SPIN_TERMS) -> sp.Expr:
    """Carrier phase from TaylorF2 evaluated at the orbital harmonic 2f/m."""
    if (l, m) not in MODES:
        raise ValueError(f"unsupported mode {(l, m)}")
    q = sp.Rational(2, m)
    intrinsic = spin.phase_expr((sp.pi*M_SEC*q*F)**sp.Rational(1, 3), terms=terms)
    if reference_frequency:
        intrinsic -= spin.phase_expr((sp.pi*M_SEC*q*F_REF)**sp.Rational(1, 3), terms=terms)
    offset = 3*sp.pi/4 if reference_frequency else -sp.pi/4
    quadrupole_phase = intrinsic + 2*sp.pi*q*F*TC - 2*PHIC + offset
    return sp.Rational(m, 2)*quadrupole_phase + 3*sp.pi/4*(1-sp.Rational(m, 2))


def cutoff_frequency_expr(l: int, m: int) -> sp.Expr:
    """Mode's Schwarzschild ISCO hard-cutoff frequency in Hz."""
    if (l, m) not in MODES:
        raise ValueError(f"unsupported mode {(l, m)}")
    return sp.Rational(m, 2) / (6**sp.Rational(3, 2) * sp.pi * M_SEC)


def mode_expr(l: int, m: int, *, reference_frequency: bool = False,
              terms=DEFAULT_SPIN_TERMS) -> dict[str, sp.Expr]:
    """Unwindowed complex mode and its two polarization contributions."""
    if (l, m) not in MODES:
        raise ValueError(f"unsupported mode {(l, m)}")
    amplitude = (sp.sqrt(16*sp.pi/5) * base.amplitude_expr().subs(base.ETA, ETA)
                 * amplitude_ratio(l, m, v=(sp.pi*M_SEC*F)**sp.Rational(1, 3)))
    phase = mode_phase(l, m, reference_frequency=reference_frequency, terms=terms)
    h_mode = amplitude * sp.exp(-sp.I*phase)
    d_plus, d_cross = projection(l, m)
    return {"amplitude": amplitude, "phase": phase, "h_mode": h_mode,
            "h_plus": d_plus*h_mode, "h_cross": d_cross*h_mode}


def waveform_expr(*, modes=MODES, reference_frequency: bool = False,
                  terms=DEFAULT_SPIN_TERMS) -> dict[str, sp.Expr]:
    """Sum selected unwindowed complex polarizations."""
    chosen = tuple(modes)
    if len(set(chosen)) != len(chosen):
        raise ValueError("duplicate modes")
    pieces = [mode_expr(*mode, reference_frequency=reference_frequency, terms=terms)
              for mode in chosen]
    return {key: sp.Add(*(piece[key] for piece in pieces), evaluate=False)
            for key in ("h_plus", "h_cross")}


def evaluate(expression: sp.Expr, values: Mapping[sp.Symbol | str, object], *, dps: int = 80):
    """Evaluate with the independent TaylorF2 scalar mpmath walker."""
    resolved = {SYMBOLS[key] if isinstance(key, str) else key: value
                for key, value in values.items()}
    return spin.evaluate(expression, resolved, dps=dps)
