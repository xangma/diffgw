"""Versioned, coefficient-free SymPy/mpmath specification for XPNR components.

Only published algebra is represented here. All frequencies are dimensionless
``Mf``; spin vectors are dimensionless and ``(lx, ly, lz)`` is a unit vector at
the caller-supplied transition. Boundary values, fitted coefficients, carrier
deviations, and asymmetry amplitudes are *inputs*, not inferred assets.

The expressions are independent of the float reference and Torch/JAX kernels.
Their use does not establish a terminal-spin evolution, fitted asset provenance,
attachment landmark construction, or a qualified complete waveform.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import mpmath as mp
import sympy as sp


SPEC_VERSION = "xpnr-published-components-sympy-1.0.0"
SOURCES = {
    "XPNR": "https://arxiv.org/html/2507.02604v1",
    "XO4a": "https://arxiv.org/html/2312.10025v1",
    "FinalTwist": "https://arxiv.org/html/2107.08876v1",
}


def calibration_polynomial(terms: Mapping[tuple[int, int, int], str | int | float]) -> sp.Expr:
    """XO4a Eq. (35), with caller-supplied sparse ``(p,q,r)`` coefficients.

    Decimal strings retain published precision; this function never imports
    inherited backend coefficient tensors. Exponents use the printed 4x4x5 box.
    """
    eta, chi, c = sp.symbols("eta chi c", real=True)
    result = sp.S.Zero
    for (p, q, r), value in terms.items():
        if not (0 <= p < 4 and 0 <= q < 4 and 0 <= r < 5):
            raise ValueError("fit exponent is outside the XO4a (4,4,5) box")
        result += sp.Rational(str(value)) * eta**p * chi**q * c**r
    return result


def _hermite(f: sp.Symbol, f0: sp.Symbol, f1: sp.Symbol, y0: sp.Expr,
             dy0: sp.Expr, y1: sp.Expr, dy1: sp.Expr) -> sp.Expr:
    width = f1 - f0
    t = (f - f0) / width
    return ((2*t**3 - 3*t**2 + 1)*y0 + (t**3 - 2*t**2 + t)*width*dy0
            + (-2*t**3 + 3*t**2)*y1 + (t**3 - t**2)*width*dy1)


@lru_cache(maxsize=1)
def expressions() -> dict[str, sp.Expr]:
    """Return the executable published-form component expressions.

    Piecewise derivatives are interpreted strictly inside a smooth branch.
    At branch boundaries, use one-sided derivatives; no smoothness is inferred.
    """
    def s(names):
        return sp.symbols(names, real=True)
    f, f0, f1, fc = s("f f0 f1 fc")
    y0, y1, dy0, dy1, y, y_c, dy_c, target, dtarget = s(
        "y0 y1 dy0 dy1 y y_c dy_c target dtarget")
    eta, chi, c, q, theta_f = s("eta chi c q theta_f")
    lam_a1, lam_a2, lam_a3, lam_a4 = s("lam_a1 lam_a2 lam_a3 lam_a4")
    lam_b0, lam_b1, lam_b2, lam_b3, lam_b4, lam_b5, lam_bf = s(
        "lam_b0 lam_b1 lam_b2 lam_b3 lam_b4 lam_b5 lam_bf")
    a1, a2, a3, a4 = s("a1 a2 a3 a4")
    b0, b1, b2, b3, b4, b5 = s("b0 b1 b2 b3 b4 b5")
    m1, m2 = s("m1 m2")
    sx1, sy1, sz1, sx2, sy2, sz2, lx, ly, lz = s(
        "sx1 sy1 sz1 sx2 sy2 sz2 lx ly lz")
    l = sp.Matrix([lx, ly, lz])
    spin1, spin2 = sp.Matrix([sx1, sy1, sz1]), sp.Matrix([sx2, sy2, sz2])
    parallel1, parallel2 = spin1.dot(l), spin2.dot(l)
    transverse = m1**2 * (spin1 - parallel1*l) + m2**2 * (spin2 - parallel2*l)
    chi_eff = (m1*parallel1 + m2*parallel2)/(m1+m2)
    chi_parallel = (m1+m2)*chi_eff/m1
    chi_perp = sp.sqrt(transverse.dot(transverse))/m1**2
    chi_mag = sp.sqrt(chi_parallel**2 + chi_perp**2)
    transverse_fit = chi*sp.sqrt(1-c**2)

    cutoff_feature, fmax, finf, fmin, ddmax, dddmax, dinf = s(
        "cutoff_feature fmax finf fmin ddmax dddmax dinf")
    def cutoff(z):
        return sp.Piecewise((z-sp.Rational(3,100), z >= sp.Rational(6,100)),
                            (sp.Rational(3,5)*z, True))
    turnover = fmax + (-ddmax + sp.sqrt(ddmax**2 +
                 2*dddmax*sp.Rational(25,10000)*dinf**2))/dddmax
    lower = sp.Piecewise((cutoff(fmin), finf <= fmax),
                         (cutoff(finf), fmax <= cutoff(finf)),
                         (turnover, True))

    u, u0, u1 = f**-2, f0**-2, f1**-2
    du1 = -f1**3*dy1/2
    alpha_ext = y1 + du1*(u-u1) + (y0-y1-du1*(u0-u1))/(u0-u1)**2*(u-u1)**2
    beta_rd, kappa = s("beta_rd kappa")
    ub, ub0, ub1 = 1/f, 1/f0, 1/f1
    r0 = (y0-beta_rd)*sp.exp(kappa*(f0-f1))*f0**2
    r1 = (y1-beta_rd)*f1**2
    dr1 = -f1**4*(dy1+(y1-beta_rd)*(kappa+2/f1))
    r = r1+dr1*(ub-ub1)+(r0-r1-dr1*(ub0-ub1))/(ub0-ub1)**2*(ub-ub1)**2
    beta_ext = beta_rd+sp.exp(-kappa*(f-f1))*r/f**2

    kc = target/y_c
    dkc = (dtarget-kc*dy_c)/y_c
    k2 = (dkc*fc-(kc-1))/fc**2
    k1 = dkc-2*k2*fc

    v, iota, delta, chi_perp_input, chi_parallel_input = s(
        "v iota delta chi_perp_input chi_parallel_input")
    p = (6+6*delta+5*eta)*chi_perp_input
    z = (1+delta-eta)*chi_parallel_input
    c0 = 84*sp.sin(iota)
    c2 = (110*eta-214)*sp.sin(iota)
    c3 = -7*p*(2*sp.cos(iota)-1)+56*(3*sp.pi-z)*sp.sin(iota)
    d0 = 84*sp.cos(iota/2)
    d2 = (110*eta-214)*sp.cos(iota/2)
    d3 = 14*p*sp.sin(iota/2)+56*sp.cos(iota/2)*(3*sp.pi-z)
    emission = 2*sp.atan((c0+c2*v**2+c3*v**3)/
                          (2*sp.cos(iota/2)*(d0+d2*v**2+d3*v**3)))

    mode, rd22, rdlm = s("mode rd22 rdlm")
    shift = rdlm-rd22
    hm = sp.Piecewise((2*f/mode, f <= f0),
                      (2*f0/mode+(f-f0)*(f1-shift-2*f0/mode)/(f1-f0), f < f1),
                      (f-shift, True))
    lam_k, u_k, omega_lm, omega_22, omega_21, cos_beta_rd = s(
        "lam_k u_k omega_lm omega_22 omega_21 cos_beta_rd")
    a_s, a_a, phi_s, phi_a = s("a_s a_a phi_s phi_a")

    def window(x, lo, width):
        return sp.Piecewise(
            (sp.S.One, x <= lo), (sp.S.Zero, x >= lo+width),
            (sp.cos(sp.pi*(x-lo)/(2*width))**2, True))
    bound_center = sp.pi/2
    bound_offset = y-bound_center
    bound_t = (bound_offset/bound_center**sp.Rational(499,500))**500
    beta_bound = sp.Piecewise(
        (y, sp.Eq(y,bound_center)),
        (bound_center+bound_offset*(sp.atan(bound_t)/bound_t)**sp.Rational(1,500), True))
    return {
        "map.chi_effective": chi_eff,
        "map.chi_parallel": chi_parallel,
        "map.chi_perpendicular": chi_perp,
        "map.chi": chi_mag,
        "map.cos_theta": sp.Piecewise((sp.S.Zero, sp.Eq(chi_mag, 0)),
                                       (chi_parallel/chi_mag, True)),
        "fit.eta_frozen": sp.Max(eta, sp.Rational(9,100)),
        "fit.chi_frozen": sp.Min(chi, sp.Rational(4,5)-sp.Rational(1,5)*
                                sp.exp(-((q-6)/sp.Rational(3,2))**8)),
        "fit.a1": transverse_fit*lam_a1**2,
        "fit.a2": sp.Max(-transverse_fit*lam_a2**2,
                         -sp.pi**2*sp.sqrt(sp.Max(lam_a3**2, sp.Rational(1,100000)))),
        "fit.a3": sp.Max(lam_a3**2, sp.Rational(1,100000)),
        "fit.a4": lam_a4**2,
        "fit.b0": theta_f-transverse_fit*lam_b0,
        "fit.b1": transverse_fit*sp.exp(lam_b1),
        "fit.b2": -transverse_fit*sp.exp(lam_b2),
        "fit.b3": transverse_fit*(sp.exp(lam_b3)-200),
        "fit.b4": sp.Max(lam_b4**2, 175),
        "fit.b5": -lam_b5**2,
        "fit.beta_rd": theta_f-transverse_fit*lam_bf,
        "angles.alpha_mr": -(a1/f+a2*sp.sqrt(a3)/(a3+(f-a4)**2)),
        "angles.beta_mr": b0+(b1+b2*f+b3*f**2)/(1+b4*(f+b5)**2),
        "attach.alpha": _hermite(f,f0,f1,f0*y0,y0+f0*dy0,f1*y1,y1+f1*dy1)/f,
        "attach.beta": _hermite(f,f0,f1,y0,dy0,y1,dy1),
        "attach.beta_cutoff": cutoff(cutoff_feature),
        "attach.beta_turnover": turnover,
        "attach.beta_lower": lower,
        "attach.taper": sp.Piecewise((y1+sp.cos(sp.pi*f/(2*fc))**2*(y0-y1),f <= fc),
                                      (y1,True)),
        "attach.beta_bound": beta_bound,
        "attach.emission_beta": emission,
        "attach.alpha_exterior": alpha_ext,
        "attach.beta_exterior": beta_ext,
        "attach.beta_rescaled": y*(1+k1*f+k2*f**2),
        "carrier.deformation": lam_k+chi*sp.sqrt(1-c**2)*u_k,
        "carrier.effective_ringdown": omega_lm-mode*(1-sp.Abs(cos_beta_rd))*
                                      (omega_22-omega_21),
        "carrier.asymmetric_plus": a_s*sp.exp(sp.I*phi_s)+a_a*sp.exp(sp.I*phi_a),
        "carrier.asymmetric_minus": a_s*sp.exp(-sp.I*phi_s)-a_a*sp.exp(-sp.I*phi_a),
        "higher_mode.frequency_map": hm,
        "window.endpoint_complete": window(q,sp.Rational(17,2),sp.Rational(7,2))*
                                    window(chi,sp.Rational(17,20),sp.Rational(7,20)),
    }


def evaluate(name: str, values: Mapping[str, str | int | float], *,
             derivative: str | None = None, dps: int = 50) -> mp.mpf | mp.mpc:
    """Evaluate an expression or its smooth-branch derivative with mpmath."""
    expr = expressions()[name]
    if derivative is not None:
        expr = sp.diff(expr, sp.Symbol(derivative, real=True))
    symbols = sorted(expr.free_symbols, key=lambda symbol: symbol.name)
    missing = {symbol.name for symbol in symbols} - set(values)
    if missing:
        raise ValueError(f"missing symbolic inputs: {sorted(missing)}")
    with mp.workdps(dps):
        function = sp.lambdify(symbols, expr, modules="mpmath")
        args = [mp.mpf(str(values[symbol.name])) for symbol in symbols]
        return +function(*args)


__all__ = ["SPEC_VERSION", "SOURCES", "calibration_polynomial", "expressions", "evaluate"]
