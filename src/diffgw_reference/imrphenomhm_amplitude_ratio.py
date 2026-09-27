"""Independent restricted (2,1), (3,3), and (4,4) SPA ratios for London Eq. (4).

London et al., arXiv:1708.00404v4, Eqs. (3)–(4), retain only the
leading-frequency term for ell=|m|. Mishra et al., arXiv:1601.05588v1,
Eqs. (11), (12a), (12b), (12c), (12f), supply the terms and SPA factor.
The amplitude map is a separate reference component with supplied QNM inputs.
"""

from __future__ import annotations

import mpmath as mp
import sympy as sp

from .imrphenomhm_frequency_map import eq5_frequency_map, branch_slopes


LONDON = "https://arxiv.org/html/1708.00404v4"
MISHRA = "https://arxiv.org/html/1601.05588v1"
COMPONENT_SOURCES = {
    "hm.amp.beta33.leading.eq4.v1": (
        (LONDON, "Eqs. (3)–(4), paragraph after Eq. (4)"),
        (MISHRA, "Eqs. (11), (12a), (12c) and V_m definition"),
    ),
    "hm.amp.beta44.leading.eq4.v1": (
        (LONDON, "Eqs. (3)–(4), paragraph after Eq. (4)"),
        (MISHRA, "Eqs. (11), (12a), (12f) and V_m definition"),
    ),
    "hm.amp.beta21.1p5pn.eq4.v1": (
        (LONDON, "Eqs. (3)–(4), paragraph after Eq. (4)"),
        (MISHRA, "Eqs. (7), (11), (12a), (12b) and V_m definition"),
    ),
}
COMPONENT_BINDINGS = {
    "hm.amp.beta33.leading.eq4.v1": {
        "reference": "beta33_leading",
        "torch": ("torchwave.approximants.imrphenom_hm", "IMRPhenomHM"),
        "jax": ("jaxwave.approximants.imrphenom_hm", "IMRPhenomHM"),
        "tests": ("test_public_beta33_values_and_one_sided_slopes",),
    },
    "hm.amp.beta44.leading.eq4.v1": {
        "reference": "beta44_leading",
        "torch": ("torchwave.approximants.imrphenom_hm", "IMRPhenomHM"),
        "jax": ("jaxwave.approximants.imrphenom_hm", "IMRPhenomHM"),
        "tests": ("test_public_beta44_values_and_one_sided_slopes",),
    },
    "hm.amp.beta21.1p5pn.eq4.v1": {
        "reference": "beta21_1p5pn",
        "torch": ("torchwave.approximants.imrphenom_hm", "IMRPhenomHM"),
        "jax": ("jaxwave.approximants.imrphenom_hm", "IMRPhenomHM"),
        "tests": ("test_public_beta21_values_and_one_sided_slopes",),
    },
}

Y, G, DELTA = sp.symbols("y g delta", positive=True)
ETA = sp.symbols("eta", positive=True)
X = sp.symbols("x", positive=True)
CHI_S, CHI_A = sp.symbols("chi_s chi_a", real=True)


def _v(m: int, frequency: sp.Expr) -> sp.Expr:
    return (2 * sp.pi * frequency / m) ** sp.Rational(1, 3)


def symbolic_eq4_beta33() -> sp.Expr:
    """Build Eq. (4) from full leading SPA amplitudes, not its reduced form."""
    c = sp.Rational(3, 4) * sp.sqrt(sp.Rational(5, 7))
    def h33(z):
        return -c * DELTA * _v(3, z) ** sp.Rational(-5, 2)

    def h22(z):
        return -_v(2, z) ** sp.Rational(-7, 2)

    return sp.simplify(h33(G) / h22(G) * h33(Y) / h33(2 * Y / 3))


def symbolic_eq4_beta44() -> sp.Expr:
    """Reduce London's Eq. (4) from Mishra's full leading SPA amplitudes."""
    c = sp.Rational(4, 9) * sp.sqrt(sp.Rational(10, 7))

    def h44(z):
        return -c * (1 - 3 * ETA) * _v(4, z) ** sp.Rational(-3, 2)

    def h22(z):
        return -_v(2, z) ** sp.Rational(-7, 2)

    return sp.simplify(h44(G) / h22(G) * h44(Y) / h44(Y / 2))


def symbolic_p21(v: sp.Expr) -> sp.Expr:
    """Mishra Eq. (12b) brace through V_1^4 for aligned spins."""
    c_a = sp.Rational(4771, 1344) - sp.Rational(11941, 336) * ETA
    c_s = sp.Rational(4771, 1344) - sp.Rational(2549, 336) * ETA
    tail = -sp.pi - sp.I * (sp.Rational(1, 2) + 2 * sp.log(2))
    return (DELTA * v - sp.Rational(3, 2) * (CHI_A + DELTA * CHI_S) * v**2
            + DELTA * (sp.Rational(335, 672) + sp.Rational(117, 56) * ETA) * v**3
            + (CHI_A * c_a + DELTA * CHI_S * c_s + DELTA * tail) * v**4)


def symbolic_eq4_beta21() -> sp.Expr:
    """Reduce London's Eq. (4) using full Mishra Eq. (11) SPA envelopes."""
    def h21(z):
        return -sp.sqrt(2) / 3 * _v(1, z) ** sp.Rational(-7, 2) * symbolic_p21(_v(1, z))

    def h22(z):
        return -_v(2, z) ** sp.Rational(-7, 2)
    return sp.factor(h21(G) / h22(G) * h21(X) / h21(2 * X))


def beta33_leading(mapped_frequency, delta) -> mp.mpf:
    """Positive-mass-ordering Eq. (4) magnitude at supplied mapped Mf."""
    g, d = mp.mpf(str(mapped_frequency)), mp.mpf(str(delta))
    if g <= 0 or d <= 0 or d >= 1:
        raise ValueError("require mapped_frequency>0 and 0<delta<1")
    return mp.mpf(3) / 4 * mp.sqrt(mp.mpf(5) / 7) * d * mp.root(mp.pi * g, 3)


def beta33_on_eq5(x, f0, f_rd_33, f_rd_22, delta) -> mp.mpf:
    """Compose the Eq. (4) factor with London's Eq. (5) amplitude map."""
    return beta33_leading(eq5_frequency_map(x, f0, f_rd_33, f_rd_22, 3), delta)


def beta33_eq5_one_sided_slopes(join, f0, f_rd_33, f_rd_22, delta):
    """Return analytic left/right d beta_33 / d(Mf) at one Eq. (5) join."""
    j, f0, rd = (mp.mpf(str(v)) for v in (join, f0, f_rd_33))
    if j != f0 and j != rd:
        raise ValueError("join must equal f0 or f_rd_33")
    g = eq5_frequency_map(j, f0, rd, f_rd_22, 3)
    beta = beta33_leading(g, delta)
    low, middle, high = branch_slopes(f0, rd, f_rd_22, 3)
    left, right = (low, middle) if j == f0 else (middle, high)
    return beta * left / (3 * g), beta * right / (3 * g)


def beta44_leading(mapped_frequency, eta) -> mp.mpf:
    """Positive leading (4,4) Eq. (4) factor at supplied mapped Mf."""
    g, e = mp.mpf(str(mapped_frequency)), mp.mpf(str(eta))
    if g <= 0 or e <= 0 or e > mp.mpf(1) / 4:
        raise ValueError("require mapped_frequency>0 and 0<eta<=1/4")
    return mp.mpf(4) / 9 * mp.sqrt(mp.mpf(10) / 7) * (1 - 3 * e) * mp.root(mp.pi * g, 3) ** 2


def beta44_on_eq5(x, f0, f_rd_44, f_rd_22, eta) -> mp.mpf:
    """Compose the Eq. (4) factor with London's Eq. (5) amplitude map."""
    return beta44_leading(eq5_frequency_map(x, f0, f_rd_44, f_rd_22, 4), eta)


def beta44_eq5_one_sided_slopes(join, f0, f_rd_44, f_rd_22, eta):
    """Return analytic left/right d beta_44 / d(Mf) at one Eq. (5) join."""
    j, f0, rd = (mp.mpf(str(v)) for v in (join, f0, f_rd_44))
    if j != f0 and j != rd:
        raise ValueError("join must equal f0 or f_rd_44")
    g = eq5_frequency_map(j, f0, rd, f_rd_22, 4)
    beta = beta44_leading(g, eta)
    low, middle, high = branch_slopes(f0, rd, f_rd_22, 4)
    left, right = (low, middle) if j == f0 else (middle, high)
    return 2 * beta * left / (3 * g), 2 * beta * right / (3 * g)


def _p21(v, delta, eta, chi_s, chi_a):
    """Direct mpmath transcription of Mishra Eq. (12b)'s printed brace."""
    c_a = mp.mpf(4771) / 1344 - mp.mpf(11941) / 336 * eta
    c_s = mp.mpf(4771) / 1344 - mp.mpf(2549) / 336 * eta
    tail = -mp.pi - mp.j * (mp.mpf(1) / 2 + 2 * mp.log(2))
    coeff4 = chi_a * c_a + delta * chi_s * c_s + delta * tail
    value = (delta * v - mp.mpf(3) / 2 * (chi_a + delta * chi_s) * v**2
             + delta * (mp.mpf(335) / 672 + mp.mpf(117) / 56 * eta) * v**3
             + coeff4 * v**4)
    derivative = (delta - 3 * (chi_a + delta * chi_s) * v
                  + 3 * delta * (mp.mpf(335) / 672 + mp.mpf(117) / 56 * eta) * v**2
                  + 4 * coeff4 * v**3)
    return value, derivative


def _beta21_terms(x, mapped_frequency, mass1, mass2, spin1, spin2):
    x, g, m1, m2, s1, s2 = (mp.mpf(str(v)) for v in
                             (x, mapped_frequency, mass1, mass2, spin1, spin2))
    if x <= 0 or g <= 0 or m2 <= 0 or m1 < m2 or abs(s1) > 1 or abs(s2) > 1:
        raise ValueError("require x,g>0, m1>=m2>0, and |spin_i|<=1")
    delta = (m1 - m2) / (m1 + m2)
    if delta == 0 and s1 == s2:
        raise ValueError("equal-mass equal-spin Eq. (4) ratio is 0/0")
    eta = m1 * m2 / (m1 + m2) ** 2
    chi_s, chi_a = (s1 + s2) / 2, (s1 - s2) / 2
    vg = mp.root(2 * mp.pi * g, 3)
    vx = mp.root(2 * mp.pi * x, 3)
    v2x = mp.root(4 * mp.pi * x, 3)
    pg, dpg = _p21(vg, delta, eta, chi_s, chi_a)
    px, dpx = _p21(vx, delta, eta, chi_s, chi_a)
    p2x, dp2x = _p21(v2x, delta, eta, chi_s, chi_a)
    if p2x == 0:
        raise ValueError("Eq. (4) denominator vanishes")
    c = mp.sqrt(2) / 3
    b = c * pg * px / p2x
    db_dg = c * dpg * vg / (3 * g) * px / p2x
    db_dx = c * pg * (dpx * vx / (3 * x * p2x)
                      - px * dp2x * v2x / (3 * x * p2x**2))
    return b, db_dg, db_dx


def beta21_1p5pn(x, mapped_frequency, mass1, mass2, spin1, spin2) -> mp.mpf:
    """Magnitude of London's Eq. (4) with Mishra's (2,1) 1.5PN brace."""
    return abs(_beta21_terms(x, mapped_frequency, mass1, mass2, spin1, spin2)[0])


def beta21_on_eq5(x, f0, f_rd_21, f_rd_22, mass1, mass2, spin1, spin2) -> mp.mpf:
    """Compose the 1.5PN (2,1) ratio with London's Eq. (5) amplitude map."""
    g = eq5_frequency_map(x, f0, f_rd_21, f_rd_22, 1)
    return beta21_1p5pn(x, g, mass1, mass2, spin1, spin2)


def beta21_eq5_one_sided_slopes(join, f0, f_rd_21, f_rd_22,
                                mass1, mass2, spin1, spin2):
    """Analytic left/right d|beta_21|/d(Mf) at an Eq. (5) join."""
    j, f0, fr = (mp.mpf(str(v)) for v in (join, f0, f_rd_21))
    if j != f0 and j != fr:
        raise ValueError("join must equal f0 or f_rd_21")
    g = eq5_frequency_map(j, f0, fr, f_rd_22, 1)
    b, db_dg, db_dx = _beta21_terms(j, g, mass1, mass2, spin1, spin2)
    if b == 0:
        raise ValueError("magnitude slope undefined at a zero")
    low, middle, high = branch_slopes(f0, fr, f_rd_22, 1)
    left_g, right_g = (low, middle) if j == f0 else (middle, high)
    left = mp.re(mp.conj(b) * (db_dg * left_g + db_dx)) / abs(b)
    right = mp.re(mp.conj(b) * (db_dg * right_g + db_dx)) / abs(b)
    return left, right
