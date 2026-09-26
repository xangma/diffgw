"""Leading orbit-averaged Peters flux in dimensionless semilatus rectum.

Peters (1964), Phys. Rev. 136 B1224, Eqs. (5.6)-(5.7), gives the
semimajor-axis and eccentricity rates.  The EMRI limit uses central mass M,
secondary mass mu, A=a_orbit/M and p=A(1-e**2).  This module is independent of
the Torch and JAX carriers; it does not describe their complete EMRI model.
"""

from functools import lru_cache

import mpmath as mp
import sympy as sp


SPEC_VERSION = "emri-peters-leading-flux/v1"
COMPONENT_IDS = {"flux": "emri.flux.peters_leading.v1"}
SOURCE = "https://doi.org/10.1103/PhysRev.136.B1224"
# IAU 2015 nominal solar mass parameter GM_sun = 1.3271244e20 m^3/s^2;
# speed of light is exact in SI.  No production constants are imported.
MSUN_SEC = sp.Rational("1.3271244e20") / sp.Integer(299792458)**3

M, MU, A, P, E = sp.symbols("M mu A p e", positive=True)
Q = MU / M**2  # M and mu are geometric seconds; rates use physical seconds.


def semimajor_axis_rate_expr(a=A, e=E):
    """Peters (1964) Eq. (5.6), with a scaled by central mass M."""
    return -sp.Rational(64, 5) * Q / (a**3 * (1-e**2)**sp.Rational(7, 2)) * (
        1 + sp.Rational(73, 24)*e**2 + sp.Rational(37, 96)*e**4
    )


def eccentricity_rate_expr(a=A, e=E):
    """Peters (1964) Eq. (5.7), in the extreme mass-ratio limit."""
    return -sp.Rational(304, 15) * Q * e / (
        a**4 * (1-e**2)**sp.Rational(5, 2)
    ) * (1 + sp.Rational(121, 304)*e**2)


def semilatus_rectum_rate_expr(p=P, e=E):
    """Derived exactly from p=A(1-e**2) and the two source equations."""
    a = p / (1-e**2)
    return sp.factor(
        ((1-e**2)*semimajor_axis_rate_expr(A, e)
         - 2*A*e*eccentricity_rate_expr(A, e)).subs(A, a)
    )


def eccentricity_rate_p_expr(p=P, e=E):
    """Peters Eq. (5.7) after A=p/(1-e**2)."""
    return sp.factor(eccentricity_rate_expr(A, e).subs(A, p/(1-e**2)))


@lru_cache(maxsize=6)
def _evaluator(component, derivative):
    expr = (semilatus_rectum_rate_expr() if component == "p"
            else eccentricity_rate_p_expr())
    if derivative is not None:
        expr = sp.diff(expr, {"p": P, "e": E}[derivative])
    return sp.lambdify((M, MU, P, E), expr, "mpmath")


def evaluate(mass_solar, secondary_solar, p, e, *, component="p",
             derivative=None, dps=80):
    """Evaluate dp/dt or de/dt, optionally differentiated by p or e."""
    if component not in ("p", "e") or derivative not in (None, "p", "e"):
        raise ValueError("component and derivative must be 'p' or 'e'")
    mass, secondary, p_value, eccentricity = (
        mp.mpf(str(value)) for value in (mass_solar, secondary_solar, p, e)
    )
    if mass <= 0 or secondary <= 0 or p_value <= 0 or not 0 <= eccentricity < 1:
        raise ValueError("require M, mu, p > 0 and 0 <= e < 1")
    with mp.workdps(dps):
        return +_evaluator(component, derivative)(
            mass*mp.mpf(str(MSUN_SEC.evalf(dps))),
            secondary*mp.mpf(str(MSUN_SEC.evalf(dps))),
            p_value, eccentricity,
        )
