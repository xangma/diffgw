"""Independent restricted (3,3) and (4,4) SPA ratios for London Eq. (4).

London et al., arXiv:1708.00404v4, Eqs. (3)–(4), retain only the
leading-frequency term for ell=|m|. Mishra et al., arXiv:1601.05588v1,
Eqs. (11), (12a), (12c), (12f), supply the terms and V_m^(-7/2) SPA factor.
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
}

Y, G, DELTA = sp.symbols("y g delta", positive=True)
ETA = sp.symbols("eta", positive=True)


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
