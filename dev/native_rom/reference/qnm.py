"""Scalar QNM conventions, author-data checks, and the native 220 spline rule.

The native degree-five rule below is independent of the historical SEOBNRv4
107-knot GSL cubic interpolation. Neither those knots nor the native fitted
coefficients are imported into this scalar reference.
"""

from bisect import bisect_right

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

# Cook (2019) /n00/m-02/{2,-2,0}; signed spin is negative for the positive-
# frequency counterrotating 220 branch. Sparse validation only, not fit inputs.
COOK_AUTHOR_COUNTERROTATING_SAMPLES = (
    (-0.99, 0.29210669859710475, 0.08805227615867654),
    (-0.999, 0.29160863510363183, 0.08802849032798563),
    (-0.9995, 0.29158104550836406, 0.08802716407625095),
)


def native_signed_220_spline(final_spin, positive_knots, positive_coefficients,
                             negative_knots, negative_coefficients, digits=60):
    """Evaluate the native signed-spin 220 degree-five spline at one spin.

    Knots and (frequency, signed-imaginary-frequency) control pairs are
    supplied by the caller. The reference contains no generated coefficient
    table. For negative spin, use the positive-spin (2,-2,0) branch. Clamp
    signed spin to +/-0.9999, then interpolate in x=-log(1-|j|). The result is
    positive (M_f omega_R, M_f/tau).

    This specifies the native interpolation arithmetic, not the historical
    SEOBNRv4 GSL cubic rule or the accuracy of its numerical Kerr inputs.
    """
    with mp.workdps(digits):
        spin = mp.mpf(str(final_spin))
        if not mp.isfinite(spin):
            raise ValueError("finite final_spin required")
        limit = mp.mpf("0.9999")
        spin = max(-limit, min(limit, spin))
        knots, coefficients = ((positive_knots, positive_coefficients)
                                if spin >= 0 else (negative_knots, negative_coefficients))
        degree = 5
        count = len(coefficients)
        if count < degree + 1 or len(knots) != count + degree + 1:
            raise ValueError("degree-five knots/control-point lengths required")
        x = -mp.log1p(-abs(spin))
        # The supplied native splines extend to |j|=0.99999, beyond the
        # signed frontend clamp. Guard against accidental extrapolation when
        # this routine is exercised with another coefficient set.
        if not (knots[degree] <= x <= knots[count]):
            raise ValueError("clamped spin lies outside the spline knot domain")
        span = max(degree, min(bisect_right(knots, x) - 1, count - 1))
        work = [[mp.mpf(str(value)) for value in coefficients[span-degree+j]]
                for j in range(degree + 1)]
        for order in range(1, degree + 1):
            for j in range(degree, order - 1, -1):
                left = mp.mpf(str(knots[span-degree+j]))
                right = mp.mpf(str(knots[span+1+j-order]))
                weight = (x-left)/(right-left)
                work[j] = [(1-weight)*a + weight*b
                           for a, b in zip(work[j-1], work[j])]
        frequency, signed_imaginary = work[degree]
        return +frequency, -signed_imaginary


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
