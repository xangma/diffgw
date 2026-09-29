"""Exact signed-pair projection, including opposite-frequency leakage.

García-Quirós et al., arXiv:2001.10914v2, Eqs. (2), (4)-(9), supply
the strain, equatorial symmetry, Fourier sign, and one-sided limit. Keeping
both Fourier supports is an independent algebraic extension of those
definitions, not a claim that the paper's one-sided model includes leakage.
No production waveform, NR data, or external implementation is imported.
"""

from __future__ import annotations

from functools import lru_cache

import mpmath as mp
import sympy as sp

from . import imrphenomhm_projection as projection
from . import imrphenomhm_spa_conventions as spa


SPEC_VERSION = "imrphenomhm-signed-projection/v1"
COMPONENT = "hm.conventions.signed-pair-projection.v1"
COMPONENT_SOURCES = {
    COMPONENT: (("https://arxiv.org/abs/2001.10914v2",
                 "Eqs. (2), (4)-(9); two-sided extension derived from "
                 "the strain and Fourier definitions"),),
}
DEPENDENCIES = (projection.SPEC_VERSION, spa.SPEC_VERSION)
MODES = projection.MODES


def symbolic_project_pair(ell: int, m: int, positive, opposite) -> tuple[sp.Expr, sp.Expr]:
    """Return (hplus, hcross) for P(f) and Q(f)=conj(P(-f)).

    P=F_minus[conj(h_lm)], Q=F_minus[h_lm], m>0. Inclination is
    projection.IOTA and physical azimuth is -projection.AZIMUTH (native
    phi0). Equatorial symmetry fixes h_l,-m=(-1)^ell conj(h_lm).
    Setting Q=0 recovers the unchanged one-sided projection reference.
    """
    return tuple(d * positive + sp.conjugate(d) * opposite
                 for d in projection.native_carrier_coefficients(ell, m))


def symbolic_phase_factor(m: int, constant) -> sp.Expr:
    """Conditional 22-preserving q_m=exp[i(m/2-1)K], with real K."""
    spa.primitive_powers(m)
    return sp.exp(sp.I * (sp.Rational(m, 2) - 1) * constant)


def symbolic_transport_pair(m: int, constant, positive, opposite) -> tuple[sp.Expr, sp.Expr]:
    """Transport P and Q consistently: q P, conj(q) Q.

    For real K this corresponds to h_lm -> conj(q) h_lm and its
    equatorial partner -> q h_l,-m. It preserves signed-mode symmetry;
    it is not multiplication of the entire TD strain by a common phase.
    """
    q = symbolic_phase_factor(m, constant)
    return q * positive, sp.conjugate(q) * opposite


@lru_cache(maxsize=None)
def _numeric_coefficients(ell: int, m: int):
    return tuple(sp.lambdify((projection.IOTA, projection.AZIMUTH), d, "mpmath")
                 for d in projection.native_carrier_coefficients(ell, m))


def project_pair(ell: int, m: int, inclination, phi0, positive, opposite):
    """Evaluate the exact pair at the caller's mpmath precision."""
    angle, azimuth = mp.mpf(inclination), mp.mpf(phi0)
    p, q = mp.mpc(positive), mp.mpc(opposite)
    return tuple(d * p + mp.conj(d) * q
                 for d in (function(angle, azimuth)
                           for function in _numeric_coefficients(ell, m)))


@lru_cache(maxsize=None)
def _gaussian_function(m: int):
    f, phi0, omega = sp.symbols("f phi0 omega", real=True)
    k, epsilon, amplitude = sp.symbols("k epsilon amplitude", positive=True)
    coefficient = sp.Symbol("coefficient", complex=True)
    expression = spa.symbolic_gaussian_carrier(
        m, f, phi0, omega, k, epsilon, coefficient, amplitude)
    return sp.lambdify((f, phi0, omega, k, epsilon, coefficient, amplitude),
                      expression, "mpmath")


def gaussian_pair(m: int, f, phi0, omega, chirp_rate, epsilon,
                  coefficient=1, amplitude=1) -> tuple[mp.mpc, mp.mpc]:
    """Exact P(f), Q(f) for the existing damped quadratic TD chirp.

    Frequency may have either sign or be zero. Both supports are evaluated
    from the unchanged symbolic Gaussian transform, at caller precision.
    This deliberately avoids the positive-frequency-only numeric SPA API.
    """
    spa.primitive_powers(m)
    f, phi0, omega, k, epsilon, amplitude = map(
        mp.mpf, (f, phi0, omega, chirp_rate, epsilon, amplitude))
    coefficient = mp.mpc(coefficient)
    if not all(mp.isfinite(x) for x in
               (f, phi0, omega, k, epsilon, amplitude, coefficient)):
        raise ValueError("Gaussian parameters must be finite")
    if k <= 0 or epsilon <= 0 or amplitude <= 0:
        raise ValueError("chirp_rate, epsilon, and amplitude must be positive")
    function = _gaussian_function(m)
    p = function(f, phi0, omega, k, epsilon, coefficient, amplitude)
    q = mp.conj(function(-f, phi0, omega, k, epsilon, coefficient, amplitude))
    return mp.mpc(p), mp.mpc(q)
