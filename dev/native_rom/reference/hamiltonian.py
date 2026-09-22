"""Scalar paper reference for the SEOBNRv4 radial potentials and energy map.

This deliberately stops before the spin-coupled effective Hamiltonian.  The
point-mass coefficients are Steinhoff et al. 1608.01907v2, Appendix A;
SEOBNRv4's 4PN increment and energy map are Bohé et al. 1611.03703v3,
Eqs. (3) and (1).  No tensor frontend or implementation data is imported.
"""

import mpmath as mp
import sympy as sp

from .published import calibration, chi, mass_spin, nu


def _mp(value):
    if isinstance(value, sp.Rational):
        return mp.mpf(int(value.p)) / int(value.q)
    return mp.mpf(str(value))


def body_parameters(m1, m2, chi1, chi2):
    """Return (nu, chi_Kerr, K) in initial-total-mass units."""
    n, c = mass_spin(m1, m2, chi1, chi2)
    x1 = sp.Rational(str(m1)) / (sp.Rational(str(m1)) + sp.Rational(str(m2)))
    x2 = 1 - x1
    a = x1**2 * sp.Rational(str(chi1)) + x2**2 * sp.Rational(str(chi2))
    k = calibration()["K"].subs({nu: n, chi: c})
    return _mp(n), _mp(a), _mp(k)


def radial_coefficients(n, a, k):
    """Steinhoff Appendix A (A2a--h), with Bohé Eq. (3) in Delta_5.

    The returned Delta_5 is a function of u because the 4PN term contains
    log(u).  The explicit nu prefactor before the logarithm in A1 resolves
    the omitted prefactor in Bohé's printed Eq. (2).
    """
    n, a, k = map(_mp, (n, a, k))
    m = k*n - 1
    if m == 0:
        raise ValueError("The deformed-horizon factor is singular at K*nu=1")
    d0 = k*(k*n - 2)
    d1 = -2*(d0 + k)*m
    d2 = (d1*(d1 - 4*m) - 2*a**2*d0*m**2)/2
    d3 = -a**2*d1*m**2 - d1**3/3 + d1**2*m + d1*d2 - 2*m*(d2 - m)
    d4 = (
        8*(6*a**2*(d1**2 - 2*d2)*m**2 + 3*d1**4 - 8*m*d1**3
           - 12*d1**2*d2 + 12*d1*(2*d2*m + d3))
        + 48*d2**2 - 64*m*(3*d3 - 47*m) - 123*mp.pi**2*m**2
    )/96
    d5_constant = m**2*(
        -a**2*(d1**3 - 3*d1*d2 + 3*d3)/3
        + (d1**4 - 4*d1**2*d2 + 4*d1*d3 + 2*d2**2 - 4*d4)/(2*m)
        - (d1**5 - 5*d1**3*d2 + 5*d1**2*d3 + 5*d1*d2**2
           - 5*d2*d3 - 5*d4*d1)/(5*m**2)
        + 2275*mp.pi**2/512 + 128*mp.euler/5 - mp.mpf(4237)/60
        + 256*mp.log(2)/5
        + n*(41*mp.pi**2/32 - mp.mpf(221)/6)
    )
    return d0, d1, d2, d3, d4, d5_constant, m**2*mp.mpf(64)/5


def radial_potentials(r, n, a, k, *, digits=60):
    """Return Delta_t, Delta_r, their geometry, and analytic-limit derivative.

    The horizon product is expanded so a=0 is its continuous limit.  Its
    factors are otherwise exactly the r_+ and r_- of Steinhoff (A2a--b).
    Delta_t_prime and Lambda_t_prime are differentiated independently by
    mpmath, rather than transcribing the tensor frontend's derivatives.
    """
    with mp.workdps(digits):
        r, n, a, k = map(_mp, (r, n, a, k))
        if r <= 0 or not 0 <= n <= mp.mpf("0.25") or abs(a) > 1:
            raise ValueError("Require r>0, 0<=nu<=1/4, and |chi_Kerr|<=1")
        d0, d1, d2, d3, d4, d5c, d5log = radial_coefficients(n, a, k)
        m = k*n - 1

        def delta_u(radius):
            u = 1/radius
            poly = 1 + d1*u + d2*u**2 + d3*u**3 + d4*u**4
            poly += (d5c + d5log*mp.log(u))*u**5
            if poly <= 0:
                raise ValueError("The logarithmic radial potential is outside its real domain")
            horizon = a**2*u**2 + 2*u/m + 1/m**2
            return horizon*(1 + n*d0 + n*mp.log(poly))

        def delta_t(radius):
            return radius**2*delta_u(radius)

        def lambda_t(radius):
            return (radius**2 + a**2)**2 - a**2*delta_t(radius)

        dt = delta_t(r)
        d_inverse = 1 + mp.log(1 + 6*n/r**2 + 2*(26-3*n)*n/r**3)
        dr = dt*d_inverse
        lam = lambda_t(r)
        if dt <= 0 or dr <= 0 or lam <= 0:
            raise ValueError("The equatorial metric is outside its exterior real domain")
        return {
            "delta_t": dt,
            "delta_r": dr,
            "delta_t_prime": mp.diff(delta_t, r),
            "lambda_t": lam,
            "lambda_t_prime": mp.diff(lambda_t, r),
            "xi": mp.sqrt(dt*dr)/(r**2 + a**2),
            "lapse_squared": dt*r**2/lam,
            "g_rr": r**2/dr,
        }


def energy_map(n, h_eff, *, digits=60):
    """Bohé Eq. (1): total H_real/M and binding H_EOB/mu."""
    with mp.workdps(digits):
        n, h_eff = map(_mp, (n, h_eff))
        if not 0 < n <= mp.mpf("0.25"):
            raise ValueError("Binding energy per reduced mass requires 0<nu<=1/4")
        total = mp.sqrt(1 + 2*n*(h_eff - 1))
        return total, 2*(h_eff - 1)/(total + 1)


def nonspinning_effective_hamiltonian(r, p_r_star, p_phi, n, k, *, digits=60):
    """Equatorial orbital H_eff/mu, with spins zero and 3PN p_r* quartic.

    Uses the deformed metric, tortoise momentum, and 3PN correction of the
    EOB Hamiltonian cited by Bohé et al. Sec. II A.  This does not test any
    spin-orbit or spin-spin part of the full effective Hamiltonian.
    """
    with mp.workdps(digits):
        r, p_r_star, p_phi, n, k = map(_mp, (r, p_r_star, p_phi, n, k))
        if not 0 <= n <= mp.mpf("0.25"):
            raise ValueError("Require 0<=nu<=1/4")
        metric = radial_potentials(r, n, 0, k, digits=digits)
        radial = (p_r_star/metric["xi"])**2/metric["g_rr"]
        angular = p_phi**2*r**2/metric["lambda_t"]
        quartic = 2*n*(4-3*n)*p_r_star**4/r**2
        return mp.sqrt(metric["lapse_squared"])*mp.sqrt(1+radial+angular+quartic)
