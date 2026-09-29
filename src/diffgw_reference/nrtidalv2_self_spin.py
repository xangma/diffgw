"""Scalar NRTidalv2 self-spin equations from Dietrich et al. (2019), Eqs. 26–29.

The D composition uses only 2PN+3PN (Table II). The 3.5PN component is
returned separately for auditing Eq. (27), but belongs to the Pv2 extension.
This reference is independent of both native frontends and the LAL binary.
"""

import mpmath as mp


def multipoles(lam):
    """Published Eqs. (28)–(29) for positive dimensionless Lambda."""
    lam = mp.mpf(lam)
    if lam <= 0:
        raise ValueError("The published Love-Q logarithm requires Lambda > 0")
    ell = mp.log(lam)
    log_cq = (mp.mpf("0.1940") + mp.mpf("0.09163") * ell
              + mp.mpf("0.04812") * ell**2 - mp.mpf("0.004286") * ell**3
              + mp.mpf("0.00012450") * ell**4)
    log_co = (mp.mpf("0.003131") + mp.mpf("2.071") * log_cq
              - mp.mpf("0.7152") * log_cq**2
              + mp.mpf("0.2458") * log_cq**3
              - mp.mpf("0.03309") * log_cq**4)
    return mp.exp(log_cq), mp.exp(log_co)


def components(x, m1, m2, chi1, chi2, lam1, lam2):
    """Return Eq. (26)'s separate 2PN, 3PN and 3.5PN phase contributions."""
    x, m1, m2 = (mp.mpf(v) for v in (x, m1, m2))
    chi1, chi2, lam1, lam2 = (mp.mpf(v) for v in (chi1, chi2, lam1, lam2))
    if min(x, m1, m2) <= 0:
        raise ValueError("x and component masses must be positive")
    total = m1 + m2
    eta = m1 * m2 / total**2
    a, b = m1 / total, m2 / total
    cq1, co1 = multipoles(lam1)
    cq2, co2 = multipoles(lam2)

    def star(xa, xb, sa, sb, cq, co):
        hat_q, hat_o = cq - 1, co - 1
        q = hat_q * xa**2 * sa**2
        p2 = -50 * q
        p3 = mp.mpf(5) / 84 * (9407 + 8218*xa - 2016*xa**2) * q
        bracket = ((xa**2 + mp.mpf(308)/3*xa)*sa
                   + (xb**2 - mp.mpf(89)/3*xb)*sb - 40*mp.pi)
        p35 = 10*bracket*q - 440*hat_o*xa**3*sa**3
        return p2, p3, p35

    terms1 = star(a, b, chi1, chi2, cq1, co1)
    terms2 = star(b, a, chi2, chi1, cq2, co2)
    scale = 3 / (128 * eta * x**mp.mpf("2.5"))
    return tuple(scale * (u+v) * x**power for u, v, power in zip(
        terms1, terms2, (2, 3, mp.mpf("3.5"))))


def d_phase(x, m1, m2, chi1, chi2, lam1, lam2):
    """Published IMRPhenomD_NRTidalv2 self-spin composition, Table II."""
    p2, p3, _ = components(x, m1, m2, chi1, chi2, lam1, lam2)
    return p2 + p3
