"""Scalar SEOBNRv4 radiation-flux normalization, excluding NQC.

Bohé et al. 1611.03703v3 Eq. (9) sums both signs of m through ell=8.
Reflection symmetry folds that into a positive-m sum with 1/(8*pi).
The factorized amplitude prescription for all 35 modes is a separate,
still-pending reference component.
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
