"""Exact-decimal symbolic transcription of Bohé et al., arXiv:1611.03703v3.

Only the cited equations are defined here. No production modules, numerical
compatibility rounding, or coefficients inferred from a LAL run are imported.
"""

from functools import lru_cache

import sympy as sp


nu, chi = sp.symbols("nu chi", real=True)


def D(value: str) -> sp.Rational:
    """Preserve every printed decimal digit as an exact rational."""
    return sp.Rational(value)


@lru_cache(maxsize=1)
def calibration():
    """Eqs. (25), (26), (29)--(32); delta_t is in initial-mass units."""
    k0 = D("267.788247")*nu**3 - D("126.686734")*nu**2 + D("10.257281")*nu + D("1.733598")
    dt0 = D("716.044155")*nu**3 - D("13.087878")*nu**2 - D("45.883834")*nu - D("2.504992")
    k = (k0 - D("59.165806")*chi**3*nu**3 - D("0.426958")*chi**3*nu
         + D("1.436589")*chi**3 + D("31.17459")*chi**2*nu**3
         + D("6.164663")*chi**2*nu**2 - D("1.380863")*chi**2
         - D("27.520106")*chi*nu**3 + D("17.373601")*chi*nu**2
         + D("2.268313")*chi*nu - D("1.62045")*chi)
    d_so = (D("147.481449")*chi**3*nu**2 - D("568.651115")*chi**3*nu
            + D("66.198703")*chi**3 - D("343.313058")*chi**2*nu
            + D("2495.293427")*chi*nu**2 - D("44.532373"))
    d_ss = (D("528.511252")*chi**3*nu**2 - D("41.000256")*chi**3*nu
            + D("1161.780126")*chi**2*nu**3 - D("326.324859")*chi**2*nu**2
            + D("37.196389")*chi*nu + D("706.958312")*nu**3
            - D("36.027203")*nu + D("6.068071"))
    dt = (dt0 - D("0.192775")*chi**3*nu**2 + D("19.053803")*chi**3*nu
          - D("11.543497")*chi**2 + D("40.318332")*chi*nu
          - D("13.006363")*chi)
    return {"K_nonspinning": k0, "peak_lag_nonspinning": dt0,
            "K": k, "d_SO": d_so, "d_SS": d_ss, "peak_lag": dt}


@lru_cache(maxsize=1)
def ringdown_shape():
    """The four NR-calibrated free fits, Eqs. (13), (14), (16), (17)."""
    c1 = (-D("0.0893454")*nu**2 + D("0.0612892")*nu
          + D("0.00146142")*nu*chi - D("0.0136459")*chi**2
          - D("0.0196758")*chi + D("0.0830664"))
    c2 = (-D("1.82173")*nu**2 - D("5.25339")*nu**2*chi
          + D("2.40203")*nu*chi + D("1.39777")*nu
          - D("0.371365")*chi - D("0.623953"))
    d1 = (-D("0.808987")*nu**2 + D("0.263456")*nu
          - D("0.120853")*nu*chi - D("0.0244358")*chi**2
          + D("0.00779176")*chi + D("0.147584"))
    d2 = (D("17.5646")*nu**2 - D("6.99396")*nu
          - D("9.61861")*nu*chi + D("0.581626")*chi**2
          + D("3.13067")*chi + D("2.46654"))
    return {"c1f": c1, "c2f": c2, "d1f": d1, "d2f": d2}


def _endpoint_quadratic(low, high, linear):
    """Appendix A endpoint constraints at nu=1/1000 and nu=1/4."""
    lo, hi = sp.Rational(1, 1000), sp.Rational(1, 4)
    curvature = (high-low-linear*(hi-lo))/(hi**2-lo**2)
    return low + linear*(nu-lo) + curvature*(nu**2-lo**2)


@lru_cache(maxsize=1)
def peak_targets():
    """Appendix A, Eqs. (33)--(44), in initial-mass and paper mode units."""
    amplitude_tpl = (D("1.452857") + D("0.166134")*chi
                     + D("0.027356")*chi**2 - D("0.020073")*chi**3)
    amplitude_eq = (D("1.577458") - D("0.007695")*chi
                    + D("0.021887")*chi**2 + D("0.023268")*chi**3)
    amplitude_linear = (D("-0.034424") - D("1.218066")*chi
                        - D("0.568373")*chi**2 + D("0.401114")*chi**3)
    amplitude = nu*_endpoint_quadratic(amplitude_tpl, amplitude_eq, amplitude_linear)
    shifted = chi-1
    curvature_tpl = (D("0.00239561")*shifted - D("0.00019274")*shifted**2
                     - D("0.00029666")*shifted**3)
    curvature_eq = D("-0.00412651") + D("0.00222400")*chi
    curvature = nu*_endpoint_quadratic(curvature_tpl, curvature_eq,
                                       D("-0.00577654")+D("0.00103086")*chi)
    # Eq. (41) uses a nu=0 endpoint for peak frequency, unlike the other fits.
    a3 = D("10.262073") + 4*(D("10.262073")-D("25.850378"))*(nu-sp.Rational(1, 4))
    a4 = D("7.629922") + 4*(D("7.629922")-D("25.819795"))*(nu-sp.Rational(1, 4))
    frequency = D("0.562679")+(D("-0.087062")+D("0.001743")*chi)*sp.log(a3-a4*chi)
    slope_tpl = (D("-0.011210")+(D("0.004087")+D("0.000633")*chi)
                 *sp.log(D("68.474666")-D("58.301488")*chi))
    slope_eq = D("0.011282")+D("0.000287")*chi
    slope = _endpoint_quadratic(slope_tpl, slope_eq,
                                D("0.015743")+D("0.022442")*chi)
    return {"amplitude": amplitude, "amplitude_derivative": sp.Integer(0),
            "amplitude_curvature": curvature, "frequency": frequency,
            "frequency_derivative": slope}


def mass_spin(m1, m2, chi1, chi2):
    """Eq. (8): symmetric mass ratio and calibrated effective spin.

    The formula uses dimensionless component spins and either mass unit. It is
    symmetric under simultaneous exchange of body labels.
    """
    m1, m2, chi1, chi2 = (sp.Rational(str(value)) for value in (m1, m2, chi1, chi2))
    if min(m1, m2) <= 0 or max(abs(chi1), abs(chi2)) > 1:
        raise ValueError("positive masses and component spins in [-1, 1] required")
    total = m1 + m2
    n = m1*m2/total**2
    c = (m1**2*chi1 + m2**2*chi2)/(m1**2 + m2**2)
    return n, c


def evaluate(expressions, n, c, digits=50):
    """Evaluate named symbolic expressions without binary-float coefficients."""
    substitutions = {nu: sp.Rational(str(n)), chi: sp.Rational(str(c))}
    return {key: value.subs(substitutions).evalf(digits) for key, value in expressions.items()}
