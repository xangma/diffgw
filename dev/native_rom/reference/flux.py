"""Scalar SEOBNRv4 radiation-flux components, excluding NQC.

Bohé et al. 1611.03703v3 Eq. (9) sums both signs of m through ell=8.
Reflection symmetry folds that into a positive-m sum with 1/(8*pi).
All 35 pre-NQC moduli have conditional scalar checks. Historical spin and
high-ell truncations, including the complete (2,1) prescription, remain
unresolved.
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
                               omega_zero_radial, h_eff, h_real, *, digits=60,
                               mode_33_higher_spin="taylor_f"):
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
        if mode_33_higher_spin == "lal_v4_rho":
            # LALSimIMRSpinEOBFactorizedWaveform.c, non-HM v4 coefficients:
            # rho33v4, rho33v5, rho33v6, rho33v7 are cubed with rho33.
            rho33 += (q**2/2*v**4-4*q/3*v**5+5*q**2/36*v**6
                      +(5297*q/2970+q**3/3)*v**7)
            higher33 = 0
        elif mode_33_higher_spin != "taylor_f":
            raise ValueError("mode_33_higher_spin must be taylor_f or lal_v4_rho")
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


def factorized_54_52_amplitudes(m1, m2, chi1, chi2, omega,
                               omega_zero_radial, pphi, h_real, *, digits=60):
    """Conditional J-source pre-NQC (5,4) and (5,2) moduli.

    DIN 0811.2069v2 Eqs. (4)-(7), (16), (105), (107) give Newtonian
    factors and finite-mass 1PN residuals. Taracchini 1202.0790v2
    Eqs. (53b), (53d) give 2PN nonspin terms. Pan 1006.0431v2
    Eqs. (77b), (77d) give test-particle spin terms. Applying those
    terms to finite masses through q=chiK is an explicit hybrid choice.
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
        q = (1-2*n)*(s1+s2)/2+delta*(s1-s2)/2
        c6 = x1**5+x2**5
        v, vphi = mp.root(w, 3), w*w0**(-mp.mpf(2)/3)
        polynomials = (
            (4, mp.mpf(256)/1485*mp.sqrt(33*mp.pi),
             33320*n**3-127610*n**2+96019*n-17448,
             mp.mpf(16213384)/15526875),
            (2, mp.mpf(16)/1485*mp.sqrt(11*mp.pi),
             21980*n**3-104930*n**2+84679*n-15828,
             mp.mpf(7187914)/15526875),
        )
        common = n*c6*vphi**6*j*v
        return tuple(+(constant*common*_tail_modulus(5, m, w, hr)
                       *abs(1+polynomial/(13650*c6)*v**2-2*q/15*v**3
                            +(2*q**2/5-nonspin_2pn)*v**4)**5)
                     for m, constant, polynomial, nonspin_2pn in polynomials)


def factorized_7_even_amplitudes(m1, m2, chi1, chi2, omega,
                                 omega_zero_radial, pphi, h_real, *, digits=60):
    """Conditional J-source pre-NQC (7,6), (7,4), (7,2) moduli.

    DIN 0811.2069v2 Eqs. (4)-(7), (16), (116), (118), (120) and
    Taracchini 1202.0790v2 Eqs. (55b), (55d), (55f) give the printed
    1PN residuals. Pan 1006.0431v2 Eqs. (77i)-(77k) have no spin
    term in this selected truncation; that does not prove one vanishes.
    """
    with mp.workdps(digits):
        m1, m2, s1, s2, w, w0, j, hr = map(
            lambda value: mp.mpf(str(value)),
            (m1, m2, chi1, chi2, omega, omega_zero_radial, pphi, h_real),
        )
        if min(m1, m2, w, w0, j, hr) <= 0 or max(abs(s1), abs(s2)) > 1:
            raise ValueError("Require physical masses/spins and positive orbital inputs")
        x1, x2 = m1/(m1+m2), m2/(m1+m2)
        n = x1*x2
        c8 = x1**7+x2**7
        v, vphi = mp.root(w, 3), w*w0**(-mp.mpf(2)/3)
        polynomials = (
            (6, mp.mpf(648)/25025*mp.sqrt(2145*mp.pi), 1666,
             6104*n**4-29351*n**3+37828*n**2-16185*n+2144),
            (4, mp.mpf(1024)/225225*mp.sqrt(330*mp.pi), 14994,
             41076*n**4-217959*n**3+298872*n**2-131805*n+17756),
            (2, mp.mpf(8)/45045*mp.sqrt(15*mp.pi), 14994,
             32760*n**4-190239*n**3+273924*n**2-123489*n+16832),
        )
        common = n*c8*vphi**8*j*v
        return tuple(+(constant*common*_tail_modulus(7, m, w, hr)
                       *abs(1-polynomial/(denominator*c8)*v**2)**7)
                     for m, constant, denominator, polynomial in polynomials)


def factorized_6_odd_amplitudes(m1, m2, chi1, chi2, omega,
                                omega_zero_radial, pphi, h_real, *, digits=60):
    """Conditional J-source pre-NQC (6,5), (6,3), and (6,1) moduli.

    DIN 0811.2069v2 Eqs. (4)-(7), (16), (110), (112), (114) and
    Taracchini 1202.0790v2 Eqs. (54b), (54d), (54f) fix the finite-mass
    1PN terms. Pan 1006.0431v2 Eqs. (77f)-(77h) give test-particle
    spin terms, extended here through q=chiK as a declared hybrid.
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
        q = (1-2*n)*(s1+s2)/2+delta*(s1-s2)/2
        c7 = x2**6-x1**6
        v, vphi = mp.root(w, 3), w*w0**(-mp.mpf(2)/3)
        polynomials = (
            (5, mp.mpf(625)/27027*mp.sqrt(2145*mp.pi),
             220*n**3-910*n**2+838*n-185),
            (3, mp.mpf(81)/5005*mp.sqrt(13*mp.pi),
             156*n**3-750*n**2+742*n-169),
            (1, mp.sqrt(130*mp.pi)/135135,
             124*n**3-670*n**2+694*n-161),
        )
        common = n*abs(c7)*vphi**7*j*v
        return tuple(+(constant*common*_tail_modulus(6, m, w, hr)
                       *abs(1+polynomial/(144*(1-4*n+3*n**2))*v**2
                            -2*q/9*v**3)**6)
                     for m, constant, polynomial in polynomials)


def factorized_8_odd_amplitudes(m1, m2, chi1, chi2, omega,
                                omega_zero_radial, pphi, h_real, *, digits=60):
    """Conditional tabulated J-source pre-NQC (8,7),(8,5),(8,3),(8,1).

    DIN 0811.2069v2 Eqs. (4)-(7), (16), (123), (125), (127), (129),
    Taracchini 1202.0790v2 Eqs. (56b), (56d), (56f), (56h), and Pan
    1006.0431v2 Eqs. (77l)-(77m) give the printed 1PN truncation.
    No spin term is printed there; omitted terms need not vanish.
    """
    with mp.workdps(digits):
        m1, m2, s1, s2, w, w0, j, hr = map(
            lambda value: mp.mpf(str(value)),
            (m1, m2, chi1, chi2, omega, omega_zero_radial, pphi, h_real),
        )
        if min(m1, m2, w, w0, j, hr) <= 0 or max(abs(s1), abs(s2)) > 1:
            raise ValueError("Require physical masses/spins and positive orbital inputs")
        x1, x2 = m1/(m1+m2), m2/(m1+m2)
        n = x1*x2
        c9 = x2**8-x1**8
        v, vphi = mp.root(w, 3), w*w0**(-mp.mpf(2)/3)
        polynomials = (
            (7, mp.mpf(117649)/15752880*mp.sqrt(34034*mp.pi), 18240,
             38920*n**4-207550*n**3+309498*n**2-154099*n+23478),
            (5, mp.mpf(15625)/22054032*mp.sqrt(24310*mp.pi), 3648,
             6056*n**4-34598*n**3+54642*n**2-28055*n+4350),
            (3, mp.mpf(81)/1361360*mp.sqrt(1122*mp.pi), 18240,
             24520*n**4-149950*n**3+249018*n**2-131059*n+20598),
            (1, mp.sqrt(1190*mp.pi)/110270160, 18240,
             21640*n**4-138430*n**3+236922*n**2-126451*n+20022),
        )
        common = n*abs(c9)*vphi**9*j*v
        return tuple(+(constant*common*_tail_modulus(8, m, w, hr)
                       *abs(1-polynomial/(denominator*(1-6*n+10*n**2-4*n**3))
                            *v**2)**8)
                     for m, constant, denominator, polynomial in polynomials)


def factorized_41_amplitude(m1, m2, chi1, chi2, omega,
                            omega_zero_radial, pphi, h_real, *, digits=60):
    """Conditional regularized J-source pre-NQC (4,1) modulus.

    DIN 0811.2069v2 Eqs. (4)-(7), (16), (103), Taracchini
    1202.0790v2 Eq. (52d), and Pan 1006.0431v2 Eqs. (28i), (41i)
    give the selected nonspin and spin terms. The finite-mass leading
    spin term is kept outside c5, so equal-mass unequal-spin values
    stay finite. The full historical v4 hybrid is unverified.
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
        c5 = x2**4-x1**4
        v, vphi = mp.root(w, 3), w*w0**(-mp.mpf(2)/3)
        el1 = mp.euler+mp.log(2*v)
        rho = (1+(288*n**2-1385*n+602)/(528*(2*n-1))*v**2
               -mp.mpf(7775491)/21141120*v**4
               +(mp.mpf(1227423222031)/1758095539200-1571*el1/6930)*v**6)
        residual = (c5*(rho**4+3*q**2/2*v**4
                        -(10*q**3/3+mp.mpf(20033)*q/13860)*v**5)
                    -5*n/2*(delta*cs-ca)*v)
        return +(mp.sqrt(2*mp.pi)/105*n*vphi**5*j*v
                 *_tail_modulus(4, 1, w, hr)*abs(residual))


def factorized_53_51_amplitudes(m1, m2, chi1, chi2, omega,
                                omega_zero_radial, h_eff, h_real, *, digits=60):
    """Conditional H-source pre-NQC (5,3) and (5,1) moduli.

    DIN 0811.2069v2 Eqs. (4)-(7), (106), (108), Taracchini
    1202.0790v2 Eqs. (53c), (53e), and Pan 1006.0431v2 Eqs.
    (77c), (77e) give the selected terms. Finite-mass use of
    test-particle spin through q=chiK is a declared hybrid.
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
        q = (1-2*n)*(s1+s2)/2+delta*(s1-s2)/2
        c5 = x2**4-x1**4
        v, vphi = mp.root(w, 3), w*w0**(-mp.mpf(2)/3)
        polynomials = (
            (3, mp.mpf(9)/440*mp.sqrt(66*mp.pi),
             176*n**2-850*n+375, mp.mpf(410833)/709800, -mp.mpf(103)/325),
            (1, mp.sqrt(77*mp.pi)/13860,
             8*n**2-626*n+319, mp.mpf(31877)/304200, mp.mpf(139)/975),
        )
        common = n*abs(c5)*vphi**5*he
        return tuple(+(constant*common*_tail_modulus(5, m, w, hr)
                       *abs(1+polynomial/(390*(2*n-1))*v**2
                            -2*q/3*v**3+(q**2/2-nonspin_2pn)*v**4
                            +spin_25pn*q*v**5)**5)
                     for m, constant, polynomial, nonspin_2pn, spin_25pn in polynomials)


def factorized_7_odd_amplitudes(m1, m2, chi1, chi2, omega,
                                omega_zero_radial, h_eff, h_real, *, digits=60):
    """Conditional H-source pre-NQC (7,7),(7,5),(7,3),(7,1) moduli.

    DIN 0811.2069v2 Eqs. (4)-(7), (115), (117), (119), (121),
    Taracchini 1202.0790v2 Eqs. (55a), (55c), (55e), (55g), and
    Pan 1006.0431v2 Eqs. (77i)-(77k) fix the selected truncation.
    Finite-mass use of test-particle spin is a declared hybrid.
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
        q = (1-2*n)*(s1+s2)/2+delta*(s1-s2)/2
        c7 = x2**6-x1**6
        v, vphi = mp.root(w, 3), w*w0**(-mp.mpf(2)/3)
        polynomials = (
            (7, mp.mpf(16807)/772200*mp.sqrt(30030*mp.pi),
             1380*n**3-4963*n**2+4246*n-906),
            (5, mp.mpf(3125)/216216*mp.sqrt(330*mp.pi),
             804*n**3-3523*n**2+3382*n-762),
            (3, mp.mpf(243)/200200*mp.sqrt(30*mp.pi),
             420*n**3-2563*n**2+2806*n-666),
            (1, mp.sqrt(10*mp.pi)/1081080,
             228*n**3-2083*n**2+2518*n-618),
        )
        common = n*abs(c7)*vphi**7*he
        return tuple(+(constant*common*_tail_modulus(7, m, w, hr)
                       *abs(1+polynomial/(714*(1-4*n+3*n**2))*v**2
                            -2*q/3*v**3)**7)
                     for m, constant, polynomial in polynomials)


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
    return _factorized_21_spin_amplitude(
        m1, m2, chi1, chi2, omega, omega_zero_radial, pphi, h_real,
        higher_spin="taylor_f", auxiliary_spin_order=3, digits=digits,
    )


def factorized_21_rho_hybrid_amplitude(m1, m2, chi1, chi2, omega,
                                       omega_zero_radial, pphi, h_real, *,
                                       auxiliary_spin_order=3, digits=60):
    """Conditional pre-NQC |R h21/M| with Pan (29b) higher-spin rho.

    Retain the published spin-dependent rho21 coefficients at v**4--v**7,
    with lower-order spin in a regularized auxiliary factor. The latter can
    end at v (complex-waveform path) or v**3 (flux path). The selection is
    source-audited for pinned ordinary v4 but has no compiled parity check.
    """
    if auxiliary_spin_order not in (1, 3):
        raise ValueError("auxiliary_spin_order must be 1 or 3")
    return _factorized_21_spin_amplitude(
        m1, m2, chi1, chi2, omega, omega_zero_radial, pphi, h_real,
        higher_spin="rho", auxiliary_spin_order=auxiliary_spin_order,
        digits=digits,
    )


def _factorized_21_spin_amplitude(m1, m2, chi1, chi2, omega,
                                  omega_zero_radial, pphi, h_real, *,
                                  higher_spin, auxiliary_spin_order, digits):
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
        if higher_spin == "taylor_f":
            # Pan (28b), after the v and v**3 auxiliary replacements.
            higher_kerr_spin = (-3*q**2*v**4+3*q*(4*q**2-27)*v**5/16
                                +4163*q**2*v**6/252
                                +(-2593*q**3/168+107*q*el1/35
                                  -mp.mpf(11847887)*q/1058400)*v**7)
            base = rho**2+higher_kerr_spin
        else:
            # Pan (29b): retain spin-dependent rho21 at v**4--v**7 only.
            higher_kerr_rho = (
                (-865*q**2/1792-405*q**4/2048)*v**4
                +(-mp.mpf(98635)*q/75264+2031*q**3/7168
                  -1701*q**5/8192)*v**5
                +(-mp.mpf(15309)*q**6/65536+3897*q**4/16384
                  +mp.mpf(9032393)*q**2/1806336)*v**6
                +(-mp.mpf(72171)*q**7/262144+18603*q**5/65536
                  -55169*q**3/16384
                  +(107*el1/140-mp.mpf(3859374457)/1159065600)*q)*v**7
            )
            base = (rho+higher_kerr_rho)**2
        # Multiply by delta before evaluating: the equal-mass unequal-spin
        # limit is finite even though an unregularized f21 contains chiA/delta.
        residual = delta*base-mp.mpf(3)/2*(delta*cs+ca)*v
        if auxiliary_spin_order == 3:
            residual += (delta*(mp.mpf(61)/12+79*n/84)*cs
                         +(mp.mpf(61)/12+131*n/84)*ca)*v**3
        return +(mp.mpf(8)/3*mp.sqrt(mp.pi/5)*n*vphi**3*j*v
                 *_tail_modulus(2, 1, w, hr)*abs(residual))
