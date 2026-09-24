"""Independent scalar evaluation of Bohé et al. (2017), Eqs. (10)--(17).

The caller must supply a validated remnant QNM and the inspiral amplitude,
positive GW frequency, and phase at the amplitude peak. This module neither
chooses a QNM fit nor performs attachment to an inspiral trajectory.
"""

import mpmath as mp

from .published import evaluate, ringdown_shape


def peak_preserving_shape(*, nu, chi, sigma, digits=60):
    """Paper-fit shape with the C1 peak condition imposed when necessary.

    This follows the ansatz-derived guard in historical SEOBNRv4 source
    (LALSimIMREOBHybridRingdown.c, Eqs. (57)-(58) of LIGO-T1600383), not an
    additional printed free-coefficient calibration. Other fits are retained.
    """
    with mp.workdps(digits):
        n, spin = mp.mpf(str(nu)), mp.mpf(str(chi))
        damping = -mp.mpc(sigma).imag
        if not (0 < n <= mp.mpf("0.25") and abs(spin) <= 1 and damping > 0):
            raise ValueError("peak continuation requires physical fit parameters and damping")
        fit = evaluate(ringdown_shape(), n, spin, digits)
        shape = {key: mp.mpf(str(fit[key])) for key in ("c1f", "c2f", "d1f", "d2f")}
        offset = mp.tanh(shape["c2f"])
        if damping + 2 * shape["c1f"] * offset < 0:
            shape["c1f"] = -damping / (2 * offset)
        return shape


def h22(t, *, nu, chi, sigma, amplitude, frequency, phase, digits=60,
        peak_preserving=False):
    """Peak-attached mode in initial-mass units, h=A exp(-i phase).

    ``sigma = omega_QNM - i gamma_QNM`` with gamma_QNM>0. ``amplitude`` is
    |h22| at the peak, so its time derivative is zero. The printed waveform
    carries a factor nu; this reference converts the supplied physical peak
    amplitude into the paper's rescaled tilde-A coefficient.
    """
    with mp.workdps(digits):
        t, n, a0, w0, phi0 = map(lambda x: mp.mpf(str(x)), (t, nu, amplitude, frequency, phase))
        qnm = mp.mpc(sigma)
        w, gamma = qnm.real, -qnm.imag
        if not (0 < n <= mp.mpf("0.25") and abs(mp.mpf(str(chi))) <= 1):
            raise ValueError("published fit requires 0 < nu <= 1/4 and |chi| <= 1")
        if not (t >= 0 and a0 > 0 and w0 > 0 and w > 0 and gamma > 0):
            raise ValueError("t >= 0 and positive amplitude, frequencies, damping required")
        shape = (peak_preserving_shape(nu=n, chi=chi, sigma=sigma, digits=digits)
                 if peak_preserving else evaluate(ringdown_shape(), n, chi, digits))
        c1, c2, d1, d2 = (mp.mpf(str(shape[key])) for key in ("c1f", "c2f", "d1f", "d2f"))
        if not (c1 > 0 and d1 > 0 and d2 > -1):
            raise ValueError("ringdown ansatz requires c1f,d1f > 0 and d2f > -1")
        # From continuity of |h| and its first derivative at the peak.
        c1c = gamma*a0*mp.cosh(c2)**2/(n*c1)
        c2c = a0/n - c1c*mp.tanh(c2)
        tilde_amplitude = c1c*mp.tanh(c1*t+c2)+c2c
        # Eq. (15) rewritten with log1p to retain its continuous d2f=0
        # limit, including near the fit's zero crossing.
        change = mp.expm1(-d1*t)
        log_argument_change = d2*change/(1+d2)
        log_ratio = mp.log1p(log_argument_change)/log_argument_change if log_argument_change else 1
        tilde_phase = -phi0-(w-w0)*change*log_ratio/d1
        return n*tilde_amplitude*mp.exp(1j*tilde_phase-1j*qnm*t)
