"""Scalar monochromatic H1 direct-class composition, conditional on site survey data.

JKS gr-qc/9804014v1 PDF Eqs. (1)-(6), (14), (18)-(24). The 2015
GPS-to-UTC offset is fixed to 17 s (IERS Bulletin C49); Aoki's 1982 GMST
formula uses UTC as a stated approximation to UT1. This is a selected
implementation check, not an ephemeris or physical timing model.
"""

from __future__ import annotations

import mpmath as mp

from .continuous_wave_quadrupole import h0_from_gw_frequency
from .continuous_wave_phase_response import (
    jks_arm_tensor_beam_patterns, jks_circular_roemer_delay, jks_2f_phase,
    jks_2f_strain,
)

SPEC_VERSION = "cw-monochromatic-h1-direct-class/v1"
COMPONENT_IDS = {
    "amplitude": "cw.direct_h1.amplitude.v1",
    "clock": "cw.direct_h1.clock.v1",
    "delay_phase": "cw.direct_h1.delay_phase.v1",
    "antenna": "cw.direct_h1.antenna.v1",
    "strain": "cw.direct_h1.strain.v1",
}
SOURCE = "https://arxiv.org/pdf/gr-qc/9804014v1"
C_SI = mp.mpf("299792458")
YEAR_DAYS = "365.256363004"


def _m(value):
    return mp.mpf(str(value))


def gmst_utc_2015(t_gps):
    """Aoki 1982 Eqs. (13),(19), with UTC replacing UT1, 2015-07 to 2016-12."""
    gps = _m(t_gps)
    if not (mp.mpf("1119744017") <= gps < mp.mpf("1167264018")):
        raise ValueError("fixed 17-second GPS-UTC offset applies only in 2015-07 to 2016-12")
    utc = gps - 17
    day = mp.floor(utc / 86400)
    ut = utc - 86400 * day
    centuries = (day - mp.mpf("7300.5")) / 36525
    at_midnight = (mp.mpf("24110.54841") + mp.mpf("8640184.812866") * centuries
                   + mp.mpf("0.093104") * centuries**2 - mp.mpf("0.0000062") * centuries**3)
    rate = (mp.mpf("1.002737909350795") + mp.mpf("5.9006e-11") * centuries
            - mp.mpf("5.9e-15") * centuries**2)
    return mp.fmod(at_midnight + ut * rate, 86400) * 2 * mp.pi / 86400


def monochromatic_h1_sample(
    t_gps, *, t0, ra, dec, f0, cos_iota, psi, phi0,
    vertex_ecef, x_arm_ecef, y_arm_ecef, h0=None,
    ellipticity=None, distance=None, i_zz="1e38",
):
    """Paper 2f strain with a declared circular orbit and supplied H1 survey.

    The detector vertex/arms are explicit conditioning inputs; this does not
    independently validate their inherited full-precision site survey digits.
    """
    if h0 is None:
        if ellipticity is None or distance is None:
            raise ValueError("supply h0 or ellipticity and distance")
        h0 = h0_from_gw_frequency(f0, ellipticity, distance, i_zz)
    t, origin = _m(t_gps), _m(t0)
    theta = gmst_utc_2015(t)
    alpha, delta = _m(ra), _m(dec)
    direction = (mp.cos(delta) * mp.cos(alpha), mp.cos(delta) * mp.sin(alpha), mp.sin(delta))
    vertex = tuple(map(_m, vertex_ecef))
    # Rotate the Earth-fixed vertex into the celestial frame.
    ct, st = mp.cos(theta), mp.sin(theta)
    celestial_vertex = (ct * vertex[0] - st * vertex[1],
                        st * vertex[0] + ct * vertex[1], vertex[2])
    spin_delay = sum(v * n for v, n in zip(celestial_vertex, direction)) / C_SI
    orbit_delay = jks_circular_roemer_delay(
        ra, dec, orbit_phase=2 * mp.pi * (t - origin) / (_m(YEAR_DAYS) * 86400),
    )
    phase = jks_2f_phase(t - origin, orbit_delay + spin_delay, f0, phi0=phi0)
    plus, cross = jks_arm_tensor_beam_patterns(ra, dec, psi, theta, x_arm_ecef, y_arm_ecef)
    return jks_2f_strain(h0, cos_iota, phase, plus, cross)
