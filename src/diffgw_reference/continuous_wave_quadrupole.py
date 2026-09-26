"""Independent scalar CW quadrupole amplitude for a principal-axis rotator.

Jaranowski, Królak & Schutz (1998), gr-qc/9804014v1, Eqs. (32)-(33):
    h0 = 16*pi**2*G*Izz*epsilon*f_rot**2/(c**4*d).
The native API takes the GW frequency f0=2*f_rot, so its prefactor is 4*pi**2.
This component does not specify detector timing, antenna response, or precession.
"""

from __future__ import annotations

import mpmath as mp

SPEC_VERSION = "cw.quadrupole_h0/v1"
SOURCE = "https://arxiv.org/html/gr-qc/9804014"
SOURCE_EQUATIONS = ("32", "33")
# SI: exact defining speed of light and 2018 CODATA gravitational constant.
C_SI = "299792458"
G_SI = "6.67430e-11"


def h0_from_rotation_frequency(f_rot, ellipticity, distance_m, i_zz=1.0e38):
    """Paper-literal strain amplitude; f_rot in Hz, d in m, Izz in kg m²."""
    f_rot, eps, dist, inertia = (
        mp.mpf(str(value)) for value in (f_rot, ellipticity, distance_m, i_zz)
    )
    if f_rot < 0 or dist <= 0 or inertia <= 0:
        raise ValueError("frequency must be nonnegative; distance and inertia must be positive")
    return 16 * mp.pi**2 * mp.mpf(G_SI) * inertia * eps * f_rot**2 / (mp.mpf(C_SI)**4 * dist)


def h0_from_gw_frequency(f0, ellipticity, distance_m, i_zz=1.0e38):
    """Translate the native f0=2*f_rot convention to the paper equation."""
    return h0_from_rotation_frequency(mp.mpf(str(f0)) / 2, ellipticity, distance_m, i_zz)


def dh0_df0(f0, ellipticity, distance_m, i_zz=1.0e38):
    """Analytic native-frequency derivative of the independent amplitude."""
    f0 = mp.mpf(str(f0))
    if f0 < 0:
        raise ValueError("frequency must be nonnegative")
    return 8 * mp.pi**2 * mp.mpf(G_SI) * mp.mpf(str(i_zz)) * mp.mpf(str(ellipticity)) * f0 / (
        mp.mpf(C_SI)**4 * mp.mpf(str(distance_m))
    )
