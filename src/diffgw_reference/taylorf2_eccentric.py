"""Independent symbolic reference for the low-eccentricity TaylorF2 phase.

The polynomial below is transcribed from Moore et al. (2016), Eq. (6.26),
arXiv:1605.00304v2. It uses SymPy rationals and mathematical constants; no
native frontend or LAL coefficients enter this reference.
"""

from functools import lru_cache

import sympy as sp


V, V0, ETA, E0 = sp.symbols("v v0 eta e0", positive=True)
R = sp.Rational


@lru_cache(maxsize=1)
def pn_terms() -> dict[int, sp.Expr]:
    """Return individual relative-PN terms (twice the PN order as key)."""
    e, v, u = ETA, V, V0
    p = sp.pi
    return {
        0: sp.S.One,
        2: (R(299076223, 81976608) + R(18766963, 2927736)*e)*v**2
           + (R(2833, 1008) - R(197, 36)*e)*u**2,
        3: -R(2819123, 282600)*p*v**3 + R(377, 72)*p*u**3,
        4: (R(16237683263, 3330429696) + R(24133060753, 971375328)*e
            + R(1562608261, 69383952)*e**2)*v**4
           + (R(847282939759, 82632420864) - R(718901219, 368894736)*e
              - R(3697091711, 105398496)*e**2)*v**2*u**2
           + (-R(1193251, 3048192) - R(66317, 9072)*e
              + R(18155, 1296)*e**2)*u**4,
        5: (-R(2831492681, 118395270) - R(11552066831, 270617760)*e)*p*v**5
           + (-R(7986575459, 284860800) + R(555367231, 10173600)*e)*p*v**3*u**2
           + (R(112751736071, 5902315776) + R(7075145051, 210796992)*e)*p*v**2*u**3
           + (R(764881, 90720) - R(949457, 22680)*e)*p*u**5,
        6: (
            -R(43603153867072577087, 132658535116800000)
            + R(536803271, 19782000)*sp.EulerGamma
            + R(15722503703, 325555200)*p**2
            + (R(299172861614477, 689135247360)
               - R(15075413, 1446912)*p**2)*e
            + R(3455209264991, 41019955200)*e**2
            + R(50612671711, 878999040)*e**3
            + R(3843505163, 59346000)*sp.log(2)
            - R(1121397129, 17584000)*sp.log(3)
            + R(536803271, 39564000)*sp.log(16*v**2)
        )*v**6
           + (R(46001356684079, 3357073133568)
              + R(253471410141755, 5874877983744)*e
              - R(1693852244423, 23313007872)*e**2
              - R(307833827417, 2497822272)*e**3)*v**4*u**2
           - R(1062809371, 20347200)*p**2*v**3*u**3
           + (-R(356873002170973, 249880440692736)
              - R(260399751935005, 8924301453312)*e
              + R(150484695827, 35413894656)*e**2
              + R(340714213265, 3794345856)*e**3)*v**2*u**4
           + (R(26531900578691, 168991764480)
              - R(3317, 126)*sp.EulerGamma
              + R(122833, 10368)*p**2
              + (R(9155185261, 548674560)
                 - R(3977, 1152)*p**2)*e
              - R(5732473, 1306368)*e**2
              - R(3090307, 139968)*e**3
              + R(87419, 1890)*sp.log(2)
              - R(26001, 560)*sp.log(3)
              - R(3317, 252)*sp.log(16*u**2)
              )*u**6,
    }


def phase_expr(ecc_order: int = 6) -> sp.Expr:
    """Eccentric term in the SPA phase; ``ecc_order`` is twice PN order."""
    if ecc_order not in (0, 2, 3, 4, 5, 6):
        raise ValueError("ecc_order must be one of 0, 2, 3, 4, 5, 6")
    bracket = sp.Add(*(term for order, term in pn_terms().items() if order <= ecc_order))
    return -R(2355, 1462)*R(3, 128)*E0**2*(V0/V)**R(19, 3)*bracket/(ETA*V**5)


@lru_cache(maxsize=6)
def _numerical_phase(ecc_order: int):
    return sp.lambdify((V, V0, ETA, E0), phase_expr(ecc_order), modules="mpmath")


def evaluate(frequency: float, mass1: float, mass2: float, e0: float,
             f0: float, ecc_order: int = 6, *, dps: int = 80):
    """Evaluate the paper's eccentric SPA term at positive frequencies in Hz."""
    import mpmath as mp

    if frequency <= 0 or f0 <= 0 or mass1 <= 0 or mass2 <= 0:
        raise ValueError("frequencies and component masses must be positive")
    # Only the unit conversion is shared with the frontend; coefficients are
    # derived from the cited paper above.
    from diffgw_core.constants import MSUN_SEC

    with mp.workdps(dps):
        m1, m2 = mp.mpf(str(mass1)), mp.mpf(str(mass2))
        m = (m1 + m2)*mp.mpf(str(MSUN_SEC))
        eta = m1*m2/(m1 + m2)**2
        v = mp.root(mp.pi*m*mp.mpf(str(frequency)), 3)
        v0 = mp.root(mp.pi*m*mp.mpf(str(f0)), 3)
        return _numerical_phase(ecc_order)(v, v0, eta, mp.mpf(str(e0)))


def anchored_phase(frequency: float, mass1: float, mass2: float, e0: float,
                   f0: float, f_ref: float, ecc_order: int = 6):
    """Project phase gauge: subtract the paper term at the waveform's f_ref."""
    return evaluate(frequency, mass1, mass2, e0, f0, ecc_order) - evaluate(
        f_ref, mass1, mass2, e0, f0, ecc_order)
