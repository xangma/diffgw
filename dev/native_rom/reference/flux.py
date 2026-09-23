"""Scalar SEOBNRv4 radiation-flux components, excluding NQC.

Bohé et al. 1611.03703v3 Eq. (9) sums both signs of m through ell=8.
Reflection symmetry folds that into a positive-m sum with 1/(8*pi).
The dominant and three subdominant moduli for the studied rows are checked below;
the other 31 mode amplitudes remain pending independent reference checks.
"""

import mpmath as mp


def _tail_modulus(ell, m, omega, h_real):
    """Damour--Iyer--Nagar factorized tail modulus, Eqs. (19)--(20)."""
    khat = m*omega*h_real
    squared = 4*mp.pi*khat/(-mp.expm1(-4*mp.pi*khat))
    for degree in range(1, ell+1):
        squared *= 1+4*khat**2/degree**2
    return mp.sqrt(squared)


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


def factorized_33_44_amplitudes(m1, m2, chi1, chi2, omega,
                               omega_zero_radial, h_eff, h_real, *, digits=60):
    """Published pre-NQC moduli of two leading subdominant flux modes.

    Newtonian constants: Pan et al. 1006.0431v2 Eqs. (38c), (38f).
    Nonspin rho: Damour--Iyer--Nagar 0811.2069v2 Eqs. (52), (62).
    Generic leading spin and test-particle completion: Pan Eqs. (28c),
    (29f), (41c), (42f). The odd-m spin term is combined before taking
    the modulus, so the equal-mass unequal-spin limit stays finite.
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
        q = (1-2*n)*cs+delta*ca
        v, vphi = mp.root(w, 3), w*w0**(-mp.mpf(2)/3)
        el3 = mp.euler+mp.log(6*v)
        el4 = mp.euler+mp.log(8*v)
        rho33 = (1+(2*n/3-mp.mpf(7)/6)*v**2
                 +(149*n**2/330-1861*n/990-mp.mpf(6719)/3960)*v**4
                 +(mp.mpf(3203101567)/227026800-26*el3/7)*v**6
                 +(13*el3/3-mp.mpf(57566572157)/8562153600)*v**8)
        # Pan (28c): omit its generic-mass spin term at v^3, because
        # (41c) replaces it rather than adding a second copy.
        higher33 = (3*q**2/2*v**4+2*q/3*v**5-7*q**2/4*v**6
                    +(6187*q/330-q**3)*v**7)
        residual33 = (delta*(rho33**3+higher33)
                      -((2-5*n/2)*delta*cs+(2-19*n/2)*ca)*v**3)
        rho44 = (1+(2625*n**2-5870*n+1614)/(1320*(3*n-1))*v**2
                 +(1252563795*n**4-6733146000*n**3-313857376*n**2
                   +2338945704*n-511573572)/(317116800*(1-3*n)**2)*v**4
                 +(mp.mpf(16600939332793)/1098809712000-12568*el4/3465)*v**6
                 -((42*n**2-41*n+10)*cs+(10-39*n)*delta*ca)
                 /(15*(1-3*n))*v**3
                 +q**2/2*v**4-69*q/55*v**5+217*q**2/3960*v**6)

        a33 = 3*mp.sqrt(6*mp.pi/7)*n*vphi**3*he*_tail_modulus(3, 3, w, hr)*abs(residual33)
        a44 = (mp.mpf(64)/9*mp.sqrt(mp.pi/7)*n*vphi**4*he
               *_tail_modulus(4, 4, w, hr)*(1-3*n)*abs(rho44)**4)
        return +a33, +a44


def factorized_32_amplitude(m1, m2, chi1, chi2, omega,
                            omega_zero_radial, pphi, h_real, *, digits=60):
    """Published pre-NQC |R h32/M| with the J-source convention.

    Pan 1006.0431v2 Eqs. (29d), (38d), (42d), Appendix F.1;
    Damour--Iyer--Nagar 0811.2069v2 Eq. (53). The generic-mass 0.5PN
    spin term also requires the compensating quadratic term in rho.
    """
    with mp.workdps(digits):
        m1, m2, s1, s2, w, w0, j, hr = map(
            lambda x: mp.mpf(str(x)),
            (m1, m2, chi1, chi2, omega, omega_zero_radial, pphi, h_real),
        )
        if min(m1, m2, w, w0, j, hr) <= 0 or max(abs(s1), abs(s2)) > 1:
            raise ValueError("Require physical masses/spins and positive frequency/energy")
        x1, x2 = m1/(m1+m2), m2/(m1+m2)
        n, delta = x1*x2, x1-x2
        cs, ca = (s1+s2)/2, (s1-s2)/2
        q = (1-2*n)*cs+delta*ca
        v, vphi = mp.root(w, 3), w*w0**(-mp.mpf(2)/3)
        el2 = mp.euler+mp.log(4*v)
        spin_linear = -4*n*cs/(3*(3*n-1))
        rho = (1+spin_linear*v
               +(320*n**2-1115*n+328)/(270*(3*n-1))*v**2
               -spin_linear**2*v**2
               +2*q/9*v**3
               +(3085640*n**4-20338960*n**3-4725605*n**2
                 +8050045*n-1444528)/(1603800*(1-3*n)**2)*v**4
               +q**2/3*v**4-2788*q/1215*v**5
               +(mp.mpf(5849948554)/940355325-104*el2/63
                 +488*q**2/405)*v**6
               +(17056*el2/8505-mp.mpf(10607269449358)/3072140846775)*v**8)
        return +(mp.mpf(8)/3*mp.sqrt(mp.pi/7)*n*vphi**4*j*v
                 *_tail_modulus(3, 2, w, hr)*(1-3*n)*abs(rho)**3)


def factorized_21_nonspin_amplitude(m1, m2, omega, omega_zero_radial,
                                    pphi, h_real, *, digits=60):
    """Published pre-NQC |R h21/M| for zero spins and the J source.

    Pan 1006.0431v2 Eq. (38b) fixes the Newtonian modulus; Damour--Iyer--
    Nagar 0811.2069v2 Eq. (51) and Damour--Nagar--Bernuzzi 1212.4357v2
    Eq. (D2) give the hybridized nonspinning residual through v**10.
    """
    with mp.workdps(digits):
        m1, m2, w, w0, j, hr = map(
            lambda value: mp.mpf(str(value)),
            (m1, m2, omega, omega_zero_radial, pphi, h_real),
        )
        if min(m1, m2, w, w0, j, hr) <= 0:
            raise ValueError("Require positive masses, frequencies, angular momentum, and energy")
        x1, x2 = m1/(m1+m2), m2/(m1+m2)
        n, delta = x1*x2, x1-x2
        v, vphi = mp.root(w, 3), w*w0**(-mp.mpf(2)/3)
        x = v**2
        el1 = mp.euler+mp.log(2*v)
        rho = (1+(23*n/84-mp.mpf(59)/56)*x
               +(617*n**2/4704-10993*n/14112-mp.mpf(47009)/56448)*x**2
               +(mp.mpf(7613184941)/2607897600-107*el1/105)*x**3
               +(6313*el1/5880-mp.mpf(1168617463883)/911303737344)*x**4
               +(mp.mpf(5029963)/5927040*el1
                 -mp.mpf(63735873771463)/16569158860800)*x**5)
        return +(mp.mpf(8)/3*mp.sqrt(mp.pi/5)*n*vphi**3*j*v
                 *_tail_modulus(2, 1, w, hr)*abs(delta*rho**2))
