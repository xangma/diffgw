"""Scalar SEOBNRv4 radiation-flux components, excluding NQC.

Bohé et al. 1611.03703v3 Eq. (9) sums both signs of m through ell=8.
Reflection symmetry folds that into a positive-m sum with 1/(8*pi).
The dominant and fifteen subdominant moduli have conditional scalar checks;
19 modes have no scalar amplitude reference, and the complete historical
(2,1) spin prescription also remains unresolved.
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


def factorized_55_amplitude(m1, m2, chi1, chi2, omega,
                            omega_zero_radial, h_eff, h_real, *, digits=60):
    """Published pre-NQC |R h55/M| for the hybridized even-parity mode.

    DIN 0811.2069v2 Eqs. (4)--(7), (104) fix the Newtonian factor and
    finite-mass nonspin rho. Pan 1006.0431v2 Eq. (77a) supplies the
    test-particle spin terms. The hybrid choice applies those terms to
    q=(1-2nu)chiS+delta chiA; it is not historical-v4 parity evidence.
    """
    with mp.workdps(digits):
        m1, m2, s1, s2, w, w0, he, hr = map(
            lambda value: mp.mpf(str(value)),
            (m1, m2, chi1, chi2, omega, omega_zero_radial, h_eff, h_real),
        )
        if min(m1, m2, w, w0, he, hr) <= 0 or max(abs(s1), abs(s2)) > 1:
            raise ValueError("Require physical masses/spins and positive frequency/energy")
        x1, x2 = m1/(m1+m2), m2/(m1+m2)
        n, delta = x1*x2, x1-x2
        cs, ca = (s1+s2)/2, (s1-s2)/2
        q = (1-2*n)*cs+delta*ca
        v, vphi = mp.root(w, 3), w*w0**(-mp.mpf(2)/3)
        rho = (1+(512*n**2-1298*n+487)/(390*(2*n-1))*v**2
               -2*q/3*v**3+(q**2/2-mp.mpf(3353747)/2129400)*v**4
               -241*q/195*v**5)
        # DIN (5) with Y_(5,-5)(pi/2,0) gives 125*sqrt(330*pi)/792.
        newtonian = mp.mpf(125)/792*mp.sqrt(330*mp.pi)
        return +(newtonian*n*vphi**5*he*_tail_modulus(5, 5, w, hr)
                 *abs(x2**4-x1**4)*abs(rho)**5)


def factorized_31_amplitude(m1, m2, chi1, chi2, omega,
                            omega_zero_radial, h_eff, h_real, *, digits=60):
    """Conditional pre-NQC |R h31/M| with finite equal-mass spin behavior.

    Pan 1006.0431v2 Eqs. (28e), (38e), (41e) fix the Newtonian factor
    and spin terms; DIN 0811.2069v2 Eq. (54) fixes nonspin rho31.
    The complete historical v4 hybrid is not identified by those papers.
    """
    with mp.workdps(digits):
        m1, m2, s1, s2, w, w0, he, hr = map(
            lambda value: mp.mpf(str(value)),
            (m1, m2, chi1, chi2, omega, omega_zero_radial, h_eff, h_real),
        )
        if min(m1, m2, w, w0, he, hr) <= 0 or max(abs(s1), abs(s2)) > 1:
            raise ValueError("Require physical masses/spins and positive frequency/energy")
        x1, x2 = m1/(m1+m2), m2/(m1+m2)
        n, delta = x1*x2, x1-x2
        cs, ca = (s1+s2)/2, (s1-s2)/2
        q = (1-2*n)*cs+delta*ca
        v, vphi = mp.root(w, 3), w*w0**(-mp.mpf(2)/3)
        el1 = mp.euler+mp.log(2*v)
        rho = (1+(-2*n/9-mp.mpf(13)/18)*v**2
               +(-829*n**2/1782-1685*n/1782+mp.mpf(101)/7128)*v**4
               +(mp.mpf(11706720301)/6129723600-26*el1/63)*v**6
               +(169*el1/567+mp.mpf(2606097992581)/4854741091200)*v**8)
        c = x2**2-x1**2
        higher = (-5*q**2/2*v**4+38*q/9*v**5+43*q**2/12*v**6
                  +(11*q**3/3-2657*q/594)*v**7)
        residual = (c*(rho**3+higher)
                    -((2-13*n/2)*c*cs-(2-11*n/2)*ca)*v**3)
        return +(mp.sqrt(2*mp.pi/35)/3*n*vphi**3*he
                 *_tail_modulus(3, 1, w, hr)*abs(residual))


def factorized_66_64_amplitudes(m1, m2, chi1, chi2, omega,
                                omega_zero_radial, h_eff, h_real, *, digits=60):
    """Conditional pre-NQC (6,6) and (6,4) even-parity moduli.

    DIN 0811.2069v2 Eqs. (4)-(8), (109), (111) give the Newtonian
    factors and finite-mass 1PN residuals. Taracchini 1202.0790v2
    Eqs. (54a), (54c) give the nonspin 2PN terms; Pan 1006.0431v2
    Eqs. (77f), (77g) give the test-particle spin terms. The declared
    hybrid uses q=(1-2nu)chiS+delta chiA, not historical-v4 evidence.
    """
    with mp.workdps(digits):
        m1, m2, s1, s2, w, w0, he, hr = map(
            lambda value: mp.mpf(str(value)),
            (m1, m2, chi1, chi2, omega, omega_zero_radial, h_eff, h_real),
        )
        if min(m1, m2, w, w0, he, hr) <= 0 or max(abs(s1), abs(s2)) > 1:
            raise ValueError("Require physical masses/spins and positive frequency/energy")
        x1, x2 = m1/(m1+m2), m2/(m1+m2)
        n, delta = x1*x2, x1-x2
        cs, ca = (s1+s2)/2, (s1-s2)/2
        q = (1-2*n)*cs+delta*ca
        c6 = x1**5+x2**5
        v, vphi = mp.root(w, 3), w*w0**(-mp.mpf(2)/3)
        rho66 = (1+(273*n**3-861*n**2+602*n-106)/(84*c6)*v**2
                 -2*q/3*v**3+(q**2/2-mp.mpf(1025435)/659736)*v**4)
        rho64 = (1+(133*n**3-581*n**2+462*n-86)/(84*c6)*v**2
                 -2*q/3*v**3+(q**2/2-mp.mpf(476887)/659736)*v**4)
        common = n*c6*vphi**6*he
        a66 = (mp.mpf(432)/3575*mp.sqrt(715*mp.pi)*common
               *_tail_modulus(6, 6, w, hr)*abs(rho66)**6)
        a64 = (mp.mpf(1024)/96525*mp.sqrt(390*mp.pi)*common
               *_tail_modulus(6, 4, w, hr)*abs(rho64)**6)
        return +a66, +a64


def factorized_62_amplitude(m1, m2, chi1, chi2, omega,
                            omega_zero_radial, h_eff, h_real, *, digits=60):
    """Conditional pre-NQC (6,2) even-parity modulus.

    DIN 0811.2069v2 Eqs. (4)-(8), (113), Taracchini 1202.0790v2
    Eq. (54e), and Pan 1006.0431v2 Eq. (77h) supply the terms. The
    finite-mass use of Pan's test-particle spin with q=chiK is a hybrid.
    """
    with mp.workdps(digits):
        m1, m2, s1, s2, w, w0, he, hr = map(
            lambda value: mp.mpf(str(value)),
            (m1, m2, chi1, chi2, omega, omega_zero_radial, h_eff, h_real),
        )
        if min(m1, m2, w, w0, he, hr) <= 0 or max(abs(s1), abs(s2)) > 1:
            raise ValueError("Require physical masses/spins and positive frequency/energy")
        x1, x2 = m1/(m1+m2), m2/(m1+m2)
        n, delta = x1*x2, x1-x2
        cs, ca = (s1+s2)/2, (s1-s2)/2
        q = (1-2*n)*cs+delta*ca
        c6 = x1**5+x2**5
        v, vphi = mp.root(w, 3), w*w0**(-mp.mpf(2)/3)
        rho = (1+(49*n**3-413*n**2+378*n-74)/(84*c6)*v**2
               -2*q/3*v**3+(q**2/2-mp.mpf(817991)/3298680)*v**4)
        return +(mp.mpf(16)/19305*mp.sqrt(13*mp.pi)*n*c6*vphi**6*he
                 *_tail_modulus(6, 2, w, hr)*abs(rho)**6)


def factorized_8_even_amplitudes(m1, m2, chi1, chi2, omega,
                                 omega_zero_radial, h_eff, h_real, *,
                                 spin_truncation, digits=60):
    """Conditional pre-NQC (8,8),(8,6),(8,4),(8,2) moduli.

    DIN 0811.2069v2 Eqs. (4)-(8), (122), (124), (126), (128) provide
    the finite-mass terms; Taracchini 1202.0790v2 Eqs. (56a,c,e,g) and
    Pan 1006.0431v2 Eqs. (33), (77l,m) expose two spin truncations.
    Neither identifies the historical SEOBNRv4 high-ell prescription.
    """
    if spin_truncation not in {"tabulated", "generic-leading-ell8"}:
        raise ValueError("Choose an explicit ell=8 spin truncation")
    with mp.workdps(digits):
        m1, m2, s1, s2, w, w0, he, hr = map(
            lambda value: mp.mpf(str(value)),
            (m1, m2, chi1, chi2, omega, omega_zero_radial, h_eff, h_real),
        )
        if min(m1, m2, w, w0, he, hr) <= 0 or max(abs(s1), abs(s2)) > 1:
            raise ValueError("Require physical masses/spins and positive frequency/energy")
        x1, x2 = m1/(m1+m2), m2/(m1+m2)
        n, delta = x1*x2, x1-x2
        cs, ca = (s1+s2)/2, (s1-s2)/2
        q = (1-2*n)*cs+delta*ca
        c8 = x1**7+x2**7
        v, vphi = mp.root(w, 3), w*w0**(-mp.mpf(2)/3)
        spin = -2*q/3*v**3 if spin_truncation == "generic-leading-ell8" else 0
        polynomials = (
            (8, mp.mpf(131072)/5360355*mp.sqrt(34034*mp.pi), 2736,
             3482-26778*n+64659*n**2-53445*n**3+12243*n**4),
            (6, mp.mpf(1944)/2977975*mp.sqrt(255255*mp.pi), 912,
             1002-7498*n+17269*n**2-13055*n**3+2653*n**4),
            (4, mp.mpf(1024)/3828825*mp.sqrt(1870*mp.pi), 2736,
             2666-19434*n+42627*n**2-28965*n**3+4899*n**4),
            (2, mp.mpf(8)/765765*mp.sqrt(17*mp.pi), 2736,
             2462-17598*n+37119*n**2-22845*n**3+3063*n**4),
        )
        common = n*c8*vphi**8*he
        return tuple(+(constant*common*_tail_modulus(8, m, w, hr)
                       *abs(1-polynomial/(denominator*c8)*v**2+spin)**8)
                     for m, constant, denominator, polynomial in polynomials)


def factorized_42_amplitude(m1, m2, chi1, chi2, omega,
                            omega_zero_radial, h_eff, h_real, *, digits=60):
    """Published pre-NQC |R h42/M| for the hybridized even-parity mode.

    Pan 1006.0431v2 Eq. (38h) fixes the Newtonian factor. DIN
    0811.2069v2 Eq. (102) supplies finite-mass nonspin rho; Pan Eqs.
    (29h), (42h) supply test-particle and generic-mass spin terms.
    This declared hybrid is not historical-v4 parity evidence.
    """
    with mp.workdps(digits):
        m1, m2, s1, s2, w, w0, he, hr = map(
            lambda value: mp.mpf(str(value)),
            (m1, m2, chi1, chi2, omega, omega_zero_radial, h_eff, h_real),
        )
        if min(m1, m2, w, w0, he, hr) <= 0 or max(abs(s1), abs(s2)) > 1:
            raise ValueError("Require physical masses/spins and positive frequency/energy")
        x1, x2 = m1/(m1+m2), m2/(m1+m2)
        n, delta = x1*x2, x1-x2
        cs, ca = (s1+s2)/2, (s1-s2)/2
        q = (1-2*n)*cs+delta*ca
        v, vphi = mp.root(w, 3), w*w0**(-mp.mpf(2)/3)
        el2 = mp.euler+mp.log(4*v)
        rho = (1+(285*n**2-3530*n+1146)/(1320*(3*n-1))*v**2
               -((78*n**2-59*n+10)*cs+(10-21*n)*delta*ca)
               /(15*(1-3*n))*v**3
               +((-379526805*n**4-3047981160*n**3+1204388696*n**2
                  +295834536*n-114859044)/(317116800*(1-3*n)**2)
                 +q**2/2)*v**4
               -7*q/110*v**5
               +(mp.mpf(848238724511)/219761942400-3142*el2/3465
                 +2323*q**2/3960)*v**6)
        return +(mp.mpf(8)/63*mp.sqrt(mp.pi)*n*vphi**4*he
                 *_tail_modulus(4, 2, w, hr)*(1-3*n)*abs(rho)**4)


def factorized_43_amplitude(m1, m2, chi1, chi2, omega,
                            omega_zero_radial, pphi, h_real, *, digits=60):
    """Conditional pre-NQC |R h43/M| with regularized odd-mass spin term.

    Pan 1006.0431v2 Eqs. (29g), (38g), (41g) and DIN 0811.2069v2
    Eq. (101) give the J-source leading mode, nonspin rho, and spin
    additions. Taracchini 1202.0790v2 Appendix A adds the printed
    test-particle v**6 term. No division by mass difference is needed.
    """
    with mp.workdps(digits):
        m1, m2, s1, s2, w, w0, j, hr = map(
            lambda value: mp.mpf(str(value)),
            (m1, m2, chi1, chi2, omega, omega_zero_radial, pphi, h_real),
        )
        if min(m1, m2, w, w0, j, hr) <= 0 or max(abs(s1), abs(s2)) > 1:
            raise ValueError("Require physical masses/spins and positive orbital inputs")
        x1, x2 = m1/(m1+m2), m2/(m1+m2)
        n, delta = x1*x2, x1-x2
        cs, ca = (s1+s2)/2, (s1-s2)/2
        q = (1-2*n)*cs+delta*ca
        v, vphi = mp.root(w, 3), w*w0**(-mp.mpf(2)/3)
        el3 = mp.euler+mp.log(6*v)
        rho = (1+(160*n**2-547*n+222)/(176*(2*n-1))*v**2
               -mp.mpf(6894273)/7047040*v**4
               +(mp.mpf(1664224207351)/195343948800-1571*el3/770)*v**6)
        c = x2**4-x1**4
        residual = (c*(rho**4+3*q**2/2*v**4-12113*q/1540*v**5)
                    -5*n/2*(delta*cs-ca)*v)
        return +(mp.mpf(9)/5*mp.sqrt(2*mp.pi/7)*n*vphi**5*j*v
                 *_tail_modulus(4, 3, w, hr)*abs(residual))


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


def factorized_21_spin_amplitude(m1, m2, chi1, chi2, omega,
                                 omega_zero_radial, pphi, h_real, *, digits=60):
    """Reconstructed pre-NQC |R h21/M| with a regularized spin residual.

    Pan (38b), (28b), (41b) and DIN (51)/DNB (D2) fix the Newtonian,
    Kerr-spin, leading comparable-mass, and nonspin pieces. The 2PN
    comparable-mass spin coefficient follows by matching BFH (52)-(53a)
    after the odd-parity source expansion. No complete historical v4 (2,1)
    polynomial is printed in those papers, so this is a conditional oracle.
    """
    with mp.workdps(digits):
        m1, m2, s1, s2, w, w0, j, hr = map(
            lambda value: mp.mpf(str(value)),
            (m1, m2, chi1, chi2, omega, omega_zero_radial, pphi, h_real),
        )
        if min(m1, m2, w, w0, j, hr) <= 0 or max(abs(s1), abs(s2)) > 1:
            raise ValueError("Require physical masses/spins and positive orbital inputs")
        x1, x2 = m1/(m1+m2), m2/(m1+m2)
        n, delta = x1*x2, x1-x2
        cs, ca = (s1+s2)/2, (s1-s2)/2
        q = (1-2*n)*cs+delta*ca
        v, vphi = mp.root(w, 3), w*w0**(-mp.mpf(2)/3)
        x = v**2
        el1 = mp.euler+mp.log(2*v)
        rho = (1+(23*n/84-mp.mpf(59)/56)*x
               +(617*n**2/4704-10993*n/14112-mp.mpf(47009)/56448)*x**2
               +(mp.mpf(7613184941)/2607897600-107*el1/105)*x**3
               +(6313*el1/5880-mp.mpf(1168617463883)/911303737344)*x**4
               +(mp.mpf(5029963)/5927040*el1
                 -mp.mpf(63735873771463)/16569158860800)*x**5)
        higher_kerr_spin = (-3*q**2*v**4+3*q*(4*q**2-27)*v**5/16
                            +4163*q**2*v**6/252
                            +(-2593*q**3/168+107*q*el1/35
                              -mp.mpf(11847887)*q/1058400)*v**7)
        # Multiply by delta before evaluating: the equal-mass unequal-spin
        # limit is finite even though an unregularized f21 contains chiA/delta.
        residual = (delta*(rho**2+higher_kerr_spin)
                    -mp.mpf(3)/2*(delta*cs+ca)*v
                    +(delta*(mp.mpf(61)/12+79*n/84)*cs
                      +(mp.mpf(61)/12+131*n/84)*ca)*v**3)
        return +(mp.mpf(8)/3*mp.sqrt(mp.pi/5)*n*vphi**3*j*v
                 *_tail_modulus(2, 1, w, hr)*abs(residual))
