"""Scalar paper reference for selected SEOBNRv4 Hamiltonian components.

This checks the aligned-spin effective Hamiltonian at selected points, while
the full historical generator remains unqualified. The point-mass
coefficients are Steinhoff et al. 1608.01907v2, Appendix A;
SEOBNRv4's 4PN increment and energy map are Bohé et al. 1611.03703v3,
Eqs. (3) and (1). The spin map follows Barausse--Buonanno 1107.2904,
Eqs. (50)--(52), with the v2/v4 choices stated by Taracchini et al.
1311.2544v1. The aligned Hamiltonian follows Barausse--Buonanno
0912.3517v2, Eqs. (161)--(164). No tensor frontend or implementation data
is imported.
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


def effective_spin_mapping(m1, m2, chi1, chi2, r, q_minus_one,
                           radial_momentum_squared, *, digits=60):
    """Aligned S*/M² from BB Eqs. (50)--(52), with v2/v4 zero-gauge choice.

    q_minus_one is Q-1 and radial_momentum_squared is
    (Delta_r/Sigma)*(n.p_hat)**2, in the paper's dimensionless variables.
    The calibrated 4.5PN addition d_SO*nu*S_Kerr/r³ is from Taracchini et al.
    1311.2544v1, effective-one-body model paragraph.
    """
    with mp.workdps(digits):
        n_sym, c = mass_spin(m1, m2, chi1, chi2)
        m1, m2, chi1, chi2, r, q, p = map(
            _mp, (m1, m2, chi1, chi2, r, q_minus_one, radial_momentum_squared))
        if min(m1, m2, r) <= 0 or abs(chi1) > 1 or abs(chi2) > 1 or q < p or p < 0:
            raise ValueError("Require positive masses/r, physical spins, and Q-1 >= radial >= 0")
        n = _mp(n_sym)
        a = (m1**2*chi1 + m2**2*chi2)/(m1+m2)**2
        sigma_star = n*(chi1+chi2)
        d_so = _mp(calibration()["d_SO"].subs({nu: n_sym, chi: c}))
        u = 1/r

        # Set a_0...a_3 and b_0...b_3 to zero in printed Eqs. (51)--(52).
        first_star = n*(7*u/6 + q/3 - 5*p/2)
        first_kerr = n*(-2*u/3 + q/4 - 3*p)
        second_star = (
            n*(353-27*n)*u**2/36 + 5*n**2*p**2
            - n*(23+3*n)*q**2/72 + n*(-103+60*n)*u*q/36
            + n*(16-21*n)*p*q/12 + n*(47-54*n)*u*p/12
        )
        second_kerr = (
            -n*(56+21*n)*u**2/9 + 135*n**2*p**2/24
            - 45*n*q**2/144 + n*(-109+51*n)*u*q/36
            + n*(6-39*n)*p*q/24 - n*(16+147*n)*u*p/24
        )
        return +(sigma_star*(1+first_star+second_star)
                 + a*(first_kerr+second_kerr+d_so*n*u**3))


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


def aligned_effective_hamiltonian(m1, m2, chi1, chi2, r, p_r_star, p_phi,
                                  *, digits=60):
    """Aligned H_eff/mu from Barausse--Buonanno (161)--(164) and v4 choices.

    Uses the paper's equatorial spin couplings, the zero-gauge spin mapping,
    Taracchini's calibrated spin-spin term, and the v4 radial potential. This
    is an independent scalar component check, not a validated time evolution.
    """
    with mp.workdps(digits):
        n_sym, c = mass_spin(m1, m2, chi1, chi2)
        m1, m2, chi1, chi2, r, pr_star, p_phi = map(
            _mp, (m1, m2, chi1, chi2, r, p_r_star, p_phi))
        if min(m1, m2, r) <= 0 or abs(chi1) > 1 or abs(chi2) > 1:
            raise ValueError("Require positive masses/r and physical aligned spins")
        n, a, k = body_parameters(m1, m2, chi1, chi2)
        d_ss = _mp(calibration()["d_SS"].subs({nu: n_sym, chi: c}))
        x1, x2 = m1/(m1+m2), m2/(m1+m2)
        s1, s2 = x1**2*chi1, x2**2*chi2

        geometry = radial_potentials(r, n, a, k, digits=digits)
        dt, dr = geometry["delta_t"], geometry["delta_r"]
        lam, lam_r = geometry["lambda_t"], geometry["lambda_t_prime"]
        b, j = mp.sqrt(dt), mp.sqrt(dr)
        lapse = mp.sqrt(geometry["lapse_squared"])
        p_r = pr_star/geometry["xi"]
        radial = dr*p_r**2/r**2
        q = radial + p_phi**2*r**2/lam
        root_q = mp.sqrt(1+q)

        # Equatorial metric gradients in Barausse--Buonanno (162)--(163).
        nu_r = (geometry["delta_t_prime"]/dt + 2/r - lam_r/lam)/2
        b_r = geometry["delta_t_prime"]/(2*b) - b/j
        omega = 2*a*r/lam
        omega_r = 2*a*(lam-r*lam_r)/lam**2
        mapped_spin = effective_spin_mapping(m1, m2, chi1, chi2, r, q, radial,
                                             digits=digits)

        so = (lapse**2/r
              * (-root_q*j*(b_r-2*b*nu_r) + (r*lapse-b)*(root_q+1)
                 + (b*nu_r-b_r)*j)
              / (b**2*(root_q+1)*root_q))
        ss = (omega + b*omega_r*j/(2*r*lapse)
              + (p_phi**2 - b**2*dr*p_r**2/(r**2*lapse**2))
              * lapse*omega_r*j/(2*r*b*(root_q+1)*root_q))
        quartic = 2*n*(4-3*n)*pr_star**4/r**2
        geodesic = omega*p_phi + lapse*mp.sqrt(1+q+quartic)
        spin = so*p_phi*mapped_spin + ss*mapped_spin
        spin += -mapped_spin**2/(2*r**3) + d_ss*n*(s1**2+s2**2)/r**4
        return +(geodesic+spin)
