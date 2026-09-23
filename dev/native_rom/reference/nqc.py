"""Scalar NQC matching equations from Bohé et al. 1611.03703v3 Eq. (7).

The paper fixes the correction and five matching conditions, but does not
specify how a numerical trajectory should be differentiated. Callers supply
smooth scalar functions so that mpmath can differentiate them independently.
"""

import mpmath as mp


def match_22(r, pr_star, orbital_frequency, amplitude, gw_frequency,
             match_time, targets, *, digits=60):
    """Return the three amplitude and two phase coefficients at a fixed time.

    ``targets`` is (A, A', A'', omega_22, omega_22'). The base mode convention
    is h22=A exp(-i phi), with positive omega_22=phi'. Derivatives use t/M.
    The supplied functions must be smooth around ``match_time`` and describe
    the pre-NQC trajectory and mode. Peak selection is outside this function.
    """
    with mp.workdps(digits):
        t = mp.mpf(str(match_time))
        target = tuple(mp.mpf(str(value)) for value in targets)
        if len(target) != 5 or target[0] <= 0 or target[3] <= 0:
            raise ValueError("NQC matching needs five physical peak targets")

        def bases(x):
            radius = r(x)
            momentum = pr_star(x)
            frequency = orbital_frequency(x)
            if radius <= 0 or frequency == 0:
                raise ValueError("NQC basis requires positive radius and nonzero frequency")
            z = momentum/(radius*frequency)
            return (z*z, z*z/radius, z*z/radius**mp.mpf("1.5")), (
                z, z*momentum*momentum)

        bases(t)
        products = tuple(lambda x, index=i: amplitude(x)*bases(x)[0][index]
                         for i in range(3))
        phases = tuple(lambda x, index=i: bases(x)[1][index]
                       for i in range(2))
        amp_matrix = mp.matrix([[mp.diff(product, t, order) for product in products]
                                for order in (0, 1, 2)])
        amp_rhs = mp.matrix([target[order]-mp.diff(amplitude, t, order)
                             for order in (0, 1, 2)])
        phase_matrix = mp.matrix([[mp.diff(phase, t, order) for phase in phases]
                                  for order in (1, 2)])
        phase_rhs = mp.matrix([gw_frequency(t)-target[3],
                               mp.diff(gw_frequency, t)-target[4]])
        a = mp.lu_solve(amp_matrix, amp_rhs)
        b = mp.lu_solve(phase_matrix, phase_rhs)
        return tuple(+value for value in a), tuple(+value for value in b)
