"""Berti et al. (2006) printed 220 fit: a diagnostic, not SEOBNRv4's QNM.

SEOBNRv4 instead interpolates the authors' numerical QNM results. Its exact
historical interpolation grid and rule remain outside this paper reference.
"""

import mpmath as mp


# Author archive l2.tar.gz, n1l2m2.dat: spin, real frequency, positive damping.
# The archive indexes the fundamental as n1. These sparse checks do not bundle
# the archive or define the historical SEOBNRv4 interpolation.
BERTI_AUTHOR_220_SAMPLES = (
    (0.0, 0.3736716844, 0.08896231569),
    (0.8, 0.5860169749, 0.07562955239),
    (0.98, 0.8254294775, 0.03863019715),
)

# Cook (2019) author archive, KerrQNM_00.h5, /n00/m+02/{2,2,0}.
# This later, high-accuracy independent dataset is only a validation oracle;
# none of its rows are used by the native spline generator or waveform runtime.
COOK_AUTHOR_220_SAMPLES = (
    (0.999, 0.9558544196291532, 0.010530589035568865),
)
COOK_AUTHOR_220_NEAR_EXTREMAL = (
    0.99990625, 0.9861231248046504, 0.0033605524664073053,
)


def berti_220_fit(final_spin, digits=60):
    """Return positive (M_f omega_R, M_f/tau) from Eqs. (73)-(74), Table VIII.

    This analytic fit is published for the corotating (l,m,n)=(2,2,0) mode
    over 0 <= final_spin <= 0.99. It is not Berti's numerical table or the
    interpolation selected by SEOBNRv4.
    """
    with mp.workdps(digits):
        spin = mp.mpf(str(final_spin))
        if not (mp.isfinite(spin) and 0 <= spin <= mp.mpf("0.99")):
            raise ValueError("Berti 220 fit requires 0 <= final_spin <= 0.99")
        frequency = mp.mpf("1.5251")-mp.mpf("1.1568")*(1-spin)**mp.mpf("0.1292")
        quality = mp.mpf("0.7000")+mp.mpf("1.4187")*(1-spin)**mp.mpf("-0.4990")
        return +frequency, +(frequency/(2*quality))


def sigma_initial_mass_units(frequency, damping, final_mass_fraction, digits=60):
    """Convert positive final-mass-scaled QNM inputs to Bohé's complex sigma.

    The attachment time is measured in initial total mass M. Since the QNM
    inputs are scaled by remnant mass M_f, both are divided by M_f/M. Bohé's
    exp(-i sigma t) convention needs Im(sigma)=-1/tau.
    """
    with mp.workdps(digits):
        omega, gamma = mp.mpf(str(frequency)), mp.mpf(str(damping))
        fraction = mp.mpf(str(final_mass_fraction))
        if not (mp.isfinite(omega) and mp.isfinite(gamma) and mp.isfinite(fraction)
                and omega > 0 and gamma > 0 and 0 < fraction <= 1):
            raise ValueError("positive finite QNM and 0 < M_f/M <= 1 required")
        return mp.mpc(omega/fraction, -gamma/fraction)
