"""Independent scalar remnant prescriptions cited by SEOBNRv4.

Bohé et al. (2017), Sec. II.C, cite Taracchini et al. (2014), which combines
Barausse et al. (2012), Eq. (18), with the equal-mass Hemberger et al. (2013)
radiated-energy fit. Ossokine et al. (2020), Eq. (21), prints the full
unequal-mass expression and identifies it as the fit used in earlier EOB
models. The printed Hemberger coefficients limit physical accuracy;
arbitrary-precision evaluation only checks the transcription.

The final-spin family is from Hofmann, Barausse and Rezzolla (2016),
Eqs. (2)-(7), Eq. (11), and Table 1. The SEOBNRv4 paper does not specify
which Table 1 fit order to select.
"""

import mpmath as mp


_HOFMANN_TABLE_1 = {
    (1, 2): ("0.41616", (
        (None, "-1.2019", "-1.20764"),
        ("3.79245", "1.18385", "4.90494"),
    )),
    (3, 3): ("0.463926", (
        (None, "2.87025", "-1.53315", "-3.78893"),
        ("32.9127", "-62.9901", "10.0068", "56.1926"),
        ("-136.832", "329.32", "-13.2034", "-252.27"),
        ("210.075", "-545.35", "-3.97509", "368.405"),
    )),
    (3, 4): ("0.474046", (
        (None, "3.39221", "4.48865", "-5.77101", "-13.0459"),
        ("35.1278", "-72.9336", "-86.0036", "93.7371", "200.975"),
        ("-146.822", "387.184", "447.009", "-467.383", "-884.339"),
        ("223.911", "-648.502", "-697.177", "753.738", "1166.89"),
    )),
}


def _parameters(mass1, mass2, chi1, chi2):
    m1, m2, s1, s2 = (mp.mpf(str(value)) for value in
                       (mass1, mass2, chi1, chi2))
    if not all(mp.isfinite(value) for value in (m1, m2, s1, s2)):
        raise ValueError("remnant inputs must be finite")
    if m1 <= 0 or m2 <= 0 or abs(s1) > 1 or abs(s2) > 1:
        raise ValueError("masses must be positive and spins in [-1,1]")
    nu = m1*m2/(m1+m2)**2
    total_spin = (m1**2*s1+m2**2*s2)/(m1+m2)**2
    return m1, m2, s1, s2, nu, total_spin


def _isco(spin):
    z1 = 1+(1-spin*spin)**(mp.mpf(1)/3)*(
        (1+spin)**(mp.mpf(1)/3)+(1-spin)**(mp.mpf(1)/3))
    z2 = mp.sqrt(3*spin*spin+z1*z1)
    radius = 3+z2-mp.sign(spin)*mp.sqrt((3-z1)*(3+z1+2*z2))
    energy = mp.sqrt(1-mp.mpf(2)/(3*radius))
    angular_momentum = mp.mpf(2)/(3*mp.sqrt(3))*(1+2*mp.sqrt(3*radius-2))
    return energy, angular_momentum


def final_mass_fraction(mass1, mass2, chi1, chi2, *, digits=60):
    """Return the aligned-spin remnant mass divided by the initial total mass."""
    with mp.workdps(digits):
        m1, m2, s1, s2, nu, total_spin = _parameters(mass1, mass2, chi1, chi2)
        # Barausse Eq. (17): total spin per squared total mass. Ossokine
        # Eq. (21) extends the Hemberger argument to unequal masses as this
        # mass-squared weighted mean spin.
        weighted_spin = (m1**2*s1+m2**2*s2)/(m1**2+m2**2)
        e_isco, _ = _isco(total_spin)

        b0, b1, b2 = mp.mpf("0.00258"), mp.mpf("-0.07730"), mp.mpf("-1.6939")
        equal_mass_radiation = b0+b1/(b2+weighted_spin)
        radiation = nu*(1-4*nu)*(1-e_isco)+16*nu**2*equal_mass_radiation
        return +(1-radiation)


def final_spin_projection(mass1, mass2, chi1, chi2, *, fit_order, digits=60):
    """Evaluate one specified Hofmann et al. Table 1 parallel-spin fit.

    Bohé et al. cite Hofmann without specifying a Table 1 fit order. The
    caller must select a published variant; this function does not assert
    which variant SEOBNRv4 used.
    """
    if (not isinstance(fit_order, tuple) or len(fit_order) != 2
            or any(type(order) is not int for order in fit_order)
            or fit_order not in _HOFMANN_TABLE_1):
        raise ValueError("fit_order must be (1,2), (3,3), or (3,4)")
    with mp.workdps(digits):
        _, _, s1, s2, nu, total_spin = _parameters(mass1, mass2, chi1, chi2)
        xi_text, printed = _HOFMANN_TABLE_1[fit_order]
        effective_spin = total_spin+mp.mpf(xi_text)*nu*(s1+s2)
        energy, angular_momentum = _isco(effective_spin)

        # Eq. (11) fixes the unprinted k_00 at the equal-mass nonspinning
        # central value. Every other entry is transcribed from Table 1.
        k00 = 16*(mp.mpf("0.68646")-mp.sqrt(3)/2-sum(
            mp.mpf(row[0])/4**(2+i)
            for i, row in enumerate(printed) if i))
        series = mp.mpf(0)
        for i, row in enumerate(printed):
            for j, value in enumerate(row):
                coefficient = k00 if (i, j) == (0, 0) else mp.mpf(value)
                series += coefficient*nu**(2+i)*effective_spin**j
        return +(total_spin+nu*(angular_momentum-2*total_spin*(energy-1))
                 +series)
