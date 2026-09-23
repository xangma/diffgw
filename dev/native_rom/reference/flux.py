"""Scalar SEOBNRv4 radiation-flux components, excluding NQC.

Bohé et al. 1611.03703v3 Eq. (9) sums both signs of m through ell=8.
Reflection symmetry folds that into a positive-m sum with 1/(8*pi).
The dominant factorized modulus is checked below; all other mode amplitudes
remain pending independent reference checks.
"""

import mpmath as mp


def emitted_flux(omega, positive_m_modes, *, digits=60):
    """Positive energy flux from dimensionless pre-NQC R*h_lm/M modes."""
    with mp.workdps(digits):
        w = mp.mpf(str(omega))
        if w <= 0 or not positive_m_modes:
            raise ValueError("Positive orbital frequency and modes are required")
        total = mp.mpf(0)
        for (ell, m), value in positive_m_modes.items():
            if not 2 <= ell <= 8 or not 1 <= m <= ell:
                raise ValueError("Use positive-m modes with 2<=ell<=8")
            z = mp.mpc(value)
            total += (m*w)**2*(z.real**2 + z.imag**2)
        return total/(8*mp.pi)


def newtonian_22_mode(n, omega, phase=0):
    """Pan et al. 1006.0431 Eqs. (3), (30), dominant Newtonian mode."""
    n, w, phase = map(lambda x: mp.mpf(str(x)), (n, omega, phase))
    if not 0 <= n <= mp.mpf("0.25") or w <= 0:
        raise ValueError("Require 0<=nu<=1/4 and positive orbital frequency")
    return -8*mp.sqrt(mp.pi/5)*n*w**(mp.mpf(2)/3)*mp.exp(-2j*phase)


def factorized_22_amplitude(m1, m2, chi1, chi2, omega, omega_zero_radial,
                            h_eff, h_real, *, digits=60):
    """Pre-NQC |R h22/M| from published residual rho22 and tail modulus.

    Uses Taracchini's nonspinning coefficients, Pan/Buonanno spin terms,
    Bohé's replacement at 2.5PN and 3.5PN, and the DIN tail modulus.
    Caller supplies dimensionless orbital frequencies and total h_real/M.
    """
    with mp.workdps(digits):
        m1, m2, s1, s2, w, w0, he, hr = map(
            lambda x: mp.mpf(str(x)),
            (m1, m2, chi1, chi2, omega, omega_zero_radial, h_eff, h_real),
        )
        if min(m1, m2, w, w0, he, hr) <= 0 or max(abs(s1), abs(s2)) > 1:
            raise ValueError("Require physical masses/spins and positive frequency/energy")
        x1, x2 = m1/(m1+m2), m2/(m1+m2)
        n, delta = x1*x2, x1-x2
        cs, ca = (s1+s2)/2, (s1-s2)/2
        sk = (1-2*n)*cs+delta*ca
        c0 = cs+delta*ca
        v, vphi = mp.root(w, 3), w*w0**(-mp.mpf(2)/3)
        el2 = mp.euler+mp.log(4*v)
        rho2 = 55*n/84-mp.mpf(43)/42
        rho3 = -mp.mpf(2)/3*((1-n)*cs+delta*ca)
        rho4 = (19583*n**2/42336-33025*n/21168-mp.mpf(20555)/10584
                +c0**2/2)
        rho5 = ((-mp.mpf(34)/21+49*n/18+209*n**2/126)*cs
                -(mp.mpf(34)/21+19*n/42)*delta*ca)
        rho6 = (10620745*n**3/39118464-6292061*n**2/3259872
                +41*mp.pi**2*n/192-48993925*n/9779616
                -428*el2/105+mp.mpf(1556919113)/122245200+89*sk**2/252)
        rho7 = ((mp.mpf(18733)/15876+74749*n/5292-245717*n**2/63504
                 +50803*n**3/63504)*cs
                +(mp.mpf(18733)/15876+50140*n/3969+97865*n**2/63504)
                *delta*ca)
        rho8 = (9202*el2/2205-mp.mpf(387216563023)/160190110080
                +18353*sk**2/21168-sk**4/8)
        rho10 = 439877*el2/55566-mp.mpf(16094530514677)/533967033600
        rho = (1+rho2*v**2+rho3*v**3+rho4*v**4+rho5*v**5
               +rho6*v**6+rho7*v**7+rho8*v**8+rho10*v**10)
        khat = 2*w*hr
        tail_sq = 4*mp.pi*khat/(-mp.expm1(-4*mp.pi*khat))
        tail_sq *= (1+4*khat**2)*(1+khat**2)
        return +(8*mp.sqrt(mp.pi/5)*n*vphi**2*he*mp.sqrt(tail_sq)*rho**2)
