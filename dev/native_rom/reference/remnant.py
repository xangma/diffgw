"""Independent scalar final-mass prescription cited by SEOBNRv4.

Bohé et al. (2017), Sec. II.C, cite Taracchini et al. (2014), which combines
Barausse et al. (2012), Eq. (18), with the equal-mass Hemberger et al. (2013)
radiated-energy fit. Ossokine et al. (2020), Eq. (21), prints the full
unequal-mass expression and identifies it as the fit used in earlier EOB
models. The printed Hemberger coefficients limit physical accuracy;
arbitrary-precision evaluation only checks the transcription.
"""

import mpmath as mp


def final_mass_fraction(mass1, mass2, chi1, chi2, *, digits=60):
    """Return the aligned-spin remnant mass divided by the initial total mass."""
    with mp.workdps(digits):
        m1, m2, s1, s2 = (mp.mpf(str(value)) for value in
                           (mass1, mass2, chi1, chi2))
        if not all(mp.isfinite(value) for value in (m1, m2, s1, s2)):
            raise ValueError("remnant inputs must be finite")
        if m1 <= 0 or m2 <= 0 or abs(s1) > 1 or abs(s2) > 1:
            raise ValueError("masses must be positive and spins in [-1,1]")

        nu = m1*m2/(m1+m2)**2
        # Barausse Eq. (17): total spin per squared total mass. Ossokine
        # Eq. (21) extends the Hemberger argument to unequal masses as this
        # mass-squared weighted mean spin.
        weighted_spin = (m1**2*s1+m2**2*s2)/(m1**2+m2**2)
        a = (m1**2*s1+m2**2*s2)/(m1+m2)**2
        z1 = 1+(1-a*a)**(mp.mpf(1)/3)*(
            (1+a)**(mp.mpf(1)/3)+(1-a)**(mp.mpf(1)/3))
        z2 = mp.sqrt(3*a*a+z1*z1)
        r_isco = 3+z2-mp.sign(a)*mp.sqrt((3-z1)*(3+z1+2*z2))
        e_isco = mp.sqrt(1-mp.mpf(2)/(3*r_isco))

        b0, b1, b2 = mp.mpf("0.00258"), mp.mpf("-0.07730"), mp.mpf("-1.6939")
        equal_mass_radiation = b0+b1/(b2+weighted_spin)
        radiation = nu*(1-4*nu)*(1-e_isco)+16*nu**2*equal_mass_radiation
        return +(1-radiation)
