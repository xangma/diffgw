"""Printed Kerr QNM fits from London & Fauchon-Jones (arXiv:1810.03550).

This is a clean-room transcription of Appendix equations (24)--(32).  The
returned value is the dimensionless complex angular frequency ``M_f omega``;
its real and imaginary parts are the central frequency and damping rate.
"""
import cmath
import math


_COEFFS = {
    (2, 2): (1.0, ((1.5578, 2.9031), (1.9510, 5.9210), (2.0997, 2.7606),
                    (1.4109, 5.9143), (0.4106, 2.7952))),
    (3, 3): (1.5, ((2.0957, 2.9650), (2.4696, 5.9967), (2.6655, 2.8176),
                    (1.7584, 5.9327), (0.4991, 2.7817))),
    (4, 4): (2.0, ((2.6589, 3.0028), (2.9783, 6.0510), (3.2184, 2.8775),
                    (2.1276, 5.9897), (0.6034, 2.8300))),
    (4, 3): (1.5, ((0.2050, 0.5953), (3.1033, 3.0162), (4.2361, 6.0388),
                    (3.0289, 2.8262), (0.9084, 5.9152))),
}

_GENERAL = {
    (2, 1): ((0.5891, 0.0435), (0.1890, 2.2899), (1.1501, 5.8101),
             (6.0459, 2.7420), (11.1263, 5.8441), (9.3471, 2.6694),
             (3.0384, 5.7915)),
    (3, 2): ((1.0225, 0.0049), (0.2473, 0.6653), (1.7047, 3.1383),
             (0.9460, 0.1632), (1.5319, 5.7036), (2.2805, 2.6852),
             (0.9215, 5.8417)),
}


def kappa(final_spin, ell, m):
    """Eq. (21): map ``-1 <= j_f <= 1`` to the polynomial coordinate."""
    j = float(final_spin)
    if not -1.0 <= j <= 1.0:
        raise ValueError("final_spin must lie in [-1, 1]")
    return math.log(2.0 - j, 3.0) ** (1.0 / (2 + ell - abs(m)))


def dimensionless_qnm(final_spin, ell, m, overtone=0):
    """Return printed Eq. (24)--(32) ``M_f omega_lmn`` for n=0 or n=1."""
    if overtone != 0:
        if overtone == 1 and (ell, m) in ((2, 2), (3, 3)):
            # Eq. (25) and (27) are the only first-overtone fits printed for
            # the modes used by this diagnostic.
            overtone_coeffs = {
                (2, 2): (1.0, ((1.8709, 2.5112), (2.7192, 5.4250), (3.0565, 2.2857), (2.0531, 5.4862), (0.5955, 2.4225))),
                (3, 3): (1.5, ((2.3391, 2.6497), (3.1399, 5.5525), (3.5916, 2.3472), (2.4490, 5.4435), (0.7004, 2.2830))),
            }
            base, coeffs = overtone_coeffs[(ell, m)]
        else:
            raise ValueError("printed fit unavailable for this overtone")
    elif (ell, m) in _COEFFS:
        base, coeffs = _COEFFS[(ell, m)]
    elif (ell, m) in _GENERAL:
        coeffs = _GENERAL[(ell, m)]
        base = 0.0
    else:
        raise ValueError("printed fit unavailable for this mode")
    kap = kappa(final_spin, ell, m)
    value = complex(base, 0.0)
    for power, (radius, angle) in enumerate(coeffs, start=1 if base else 0):
        value += kap ** power * radius * cmath.exp(1j * angle)
    return value
