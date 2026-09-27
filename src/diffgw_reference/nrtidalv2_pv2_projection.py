"""Scalar positive-frequency (2,2) twist-up boundary for NRTidalv2 Pv2.

This accepts a supplied aligned-frame mode. It does not generate that mode or
infer how the published absolute tidal amplitude is normalized into it.
"""

import mpmath as mp


SPEC_VERSION = "nrtidalv2-pv2-positive-22-projection/v1"


def _wigner_d(m, mp_index, angle):
    """Pratten et al. (2020), Eq. (53), specialized to ell=2."""
    ell = 2
    first = max(0, m - mp_index)
    last = min(ell + m, ell - mp_index)
    numerator = mp.sqrt(
        mp.factorial(ell + m) * mp.factorial(ell - m)
        * mp.factorial(ell + mp_index) * mp.factorial(ell - mp_index)
    )
    cosine, sine = mp.cos(angle / 2), mp.sin(angle / 2)
    return mp.fsum(
        (-1) ** k * numerator
        / (mp.factorial(k) * mp.factorial(ell + m - k)
           * mp.factorial(ell - mp_index - k)
           * mp.factorial(k - m + mp_index))
        * cosine ** (2 * ell + m - mp_index - 2 * k)
        * sine ** (2 * k - m + mp_index)
        for k in range(first, last + 1)
    )


def project_positive_22(h22, alpha, gamma, beta, theta_jn, zeta, *, dps=80):
    """Pratten Eqs. (11)-(13), (53), inverse basis rotation (93)-(94)."""
    with mp.workdps(dps):
        h = mp.mpc(str(h22.real), str(h22.imag))
        a, g, b, t, z = (mp.mpf(str(value)) for value in
                         (alpha, gamma, beta, theta_jn, zeta))
        plus, cross = mp.mpc(0), mp.mpc(0)
        for m in range(-2, 3):
            harmonic = mp.sqrt(5 / (4 * mp.pi)) * _wigner_d(m, 2, t)
            phase = mp.exp(-mp.j * m * a)
            aminus = phase * _wigner_d(m, -2, b) * harmonic
            aplus = phase * _wigner_d(m, 2, b) * harmonic
            carrier = h * mp.exp(2 * mp.j * g) / 2
            plus += carrier * (aminus + mp.conj(aplus))
            cross += mp.j * carrier * (aminus - mp.conj(aplus))
        c, s = mp.cos(2 * z), mp.sin(2 * z)
        return complex(c * plus + s * cross), complex(c * cross - s * plus)
