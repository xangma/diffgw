"""Independent London Eq. (4) (4,3) ratio with Mishra Eq. (12g)."""

from __future__ import annotations

import mpmath as mp
import sympy as sp

from .imrphenomhm_frequency_map import branch_slopes, eq5_frequency_map


COMPONENT = "hm.amp.beta43.1p5pn.eq4.v1"
COMPONENT_SOURCES = {
    COMPONENT: (
        ("https://arxiv.org/pdf/1708.00404v4", "Eqs. (3)–(5), paragraph after Eq. (4)"),
        ("https://arxiv.org/pdf/1601.05588v1", "Eqs. (7), (11), (12a), (12g)"),
    )
}
COMPONENT_BINDINGS = {
    COMPONENT: {
        "reference": "beta43_1p5pn",
        "torch": ("torchwave.approximants.imrphenom_hm", "IMRPhenomHM"),
        "jax": ("jaxwave.approximants.imrphenom_hm", "IMRPhenomHM"),
        "tests": ("test_public_beta43_values_and_one_sided_slopes",),
    }
}

X, G = sp.symbols("x g", positive=True)
DELTA, ETA = sp.symbols("delta eta", real=True)
CHI_S, CHI_A = sp.symbols("chi_s chi_a", real=True)


def symbolic_p43(v: sp.Expr) -> sp.Expr:
    """The retained brace of Mishra Eq. (12g), without a native import."""
    return DELTA * (1 - 2 * ETA) * v**3 + sp.Rational(5, 2) * ETA * (CHI_A - DELTA * CHI_S) * v**4


def symbolic_eq4_beta43() -> sp.Expr:
    """London Eq. (4) built from full Mishra Eq. (11) SPA envelopes."""
    def v(m, y):
        return (2 * sp.pi * y / m) ** sp.Rational(1, 3)

    def h43(y):
        return -sp.Rational(3, 4) * sp.sqrt(sp.Rational(3, 35)) * v(3, y) ** sp.Rational(-7, 2) * symbolic_p43(v(3, y))

    def h22(y):
        return -v(2, y) ** sp.Rational(-7, 2)

    return h43(G) / h22(G) * h43(X) / h43(2 * X / 3)


def _p43(v, delta, eta, chi_s, chi_a):
    a = delta * (1 - 2 * eta)
    b = mp.mpf(5) / 2 * eta * (chi_a - delta * chi_s)
    return a * v**3 + b * v**4, 3 * a * v**2 + 4 * b * v**3


def _beta43_terms(x, mapped_frequency, mass1, mass2, spin1, spin2):
    x, g, m1, m2, s1, s2 = (mp.mpf(str(z)) for z in (x, mapped_frequency, mass1, mass2, spin1, spin2))
    if x <= 0 or g <= 0 or m2 <= 0 or m1 < m2 or abs(s1) > 1 or abs(s2) > 1:
        raise ValueError("require x,g>0, m1>=m2>0, and |spin_i|<=1")
    delta = (m1 - m2) / (m1 + m2)
    if delta == 0 and s1 == s2:
        raise ValueError("equal-mass equal-spin Eq. (4) ratio is 0/0")
    eta = m1 * m2 / (m1 + m2) ** 2
    chi_s, chi_a = (s1 + s2) / 2, (s1 - s2) / 2
    vg = mp.root(2 * mp.pi * g / 3, 3)
    vx = mp.root(2 * mp.pi * x / 3, 3)
    vd = mp.root(4 * mp.pi * x / 9, 3)
    pg, dpg = _p43(vg, delta, eta, chi_s, chi_a)
    px, dpx = _p43(vx, delta, eta, chi_s, chi_a)
    pd, dpd = _p43(vd, delta, eta, chi_s, chi_a)
    if pd == 0:
        raise ValueError("Eq. (4) denominator vanishes")
    c = mp.mpf(3) / 4 * mp.sqrt(mp.mpf(3) / 35)
    b = c * pg * px / pd
    db_dg = c * dpg * vg / (3 * g) * px / pd
    db_dx = c * pg * (dpx * vx / (3 * x * pd) - px * dpd * vd / (3 * x * pd**2))
    return b, db_dg, db_dx


def beta43_1p5pn(x, mapped_frequency, mass1, mass2, spin1, spin2) -> mp.mpf:
    """Magnitude of London's Eq. (4) with Mishra's (4,3) 1.5PN brace."""
    return abs(_beta43_terms(x, mapped_frequency, mass1, mass2, spin1, spin2)[0])


def beta43_on_eq5(x, f0, f_rd_43, f_rd_22, mass1, mass2, spin1, spin2):
    g = eq5_frequency_map(x, f0, f_rd_43, f_rd_22, 3)
    return beta43_1p5pn(x, g, mass1, mass2, spin1, spin2)


def beta43_eq5_one_sided_slopes(join, f0, f_rd_43, f_rd_22, mass1, mass2, spin1, spin2):
    """Analytic left/right d|beta_43|/d(Mf) at an Eq. (5) join."""
    j, f0, fr = (mp.mpf(str(z)) for z in (join, f0, f_rd_43))
    if j != f0 and j != fr:
        raise ValueError("join must equal f0 or f_rd_43")
    g = eq5_frequency_map(j, f0, fr, f_rd_22, 3)
    b, db_dg, db_dx = _beta43_terms(j, g, mass1, mass2, spin1, spin2)
    if b == 0:
        raise ValueError("magnitude slope undefined at a zero")
    low, middle, high = branch_slopes(f0, fr, f_rd_22, 3)
    left_g, right_g = (low, middle) if j == f0 else (middle, high)
    return tuple(mp.re(mp.conj(b) * (db_dg * slope + db_dx)) / abs(b)
                 for slope in (left_g, right_g))
