"""Independent restricted (3,3) SPA amplitude ratio for London Eq. (4).

London et al., arXiv:1708.00404v4, Eqs. (3)–(4), retain only the
leading-frequency term for ell=|m|. Mishra et al., arXiv:1601.05588v1,
Eqs. (11), (12a), (12c), supply that term and its V_m^(-7/2) SPA factor.
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
}
COMPONENT_BINDINGS = {
    "hm.amp.beta33.leading.eq4.v1": {
        "reference": "beta33_leading",
        "torch": ("torchwave.approximants.imrphenom_hm", "IMRPhenomHM"),
        "jax": ("jaxwave.approximants.imrphenom_hm", "IMRPhenomHM"),
        "tests": ("test_public_beta33_values_and_one_sided_slopes",),
    },
}

Y, G, DELTA = sp.symbols("y g delta", positive=True)


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
