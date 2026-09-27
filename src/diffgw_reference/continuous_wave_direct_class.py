"""Scalar monochromatic H1 direct-class composition with a surveyed-site option.

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

SPEC_VERSION = "cw-monochromatic-h1-direct-class/v2"
COMPONENT_IDS = {
    "amplitude": "cw.direct_h1.amplitude.v1",
    "clock": "cw.direct_h1.clock.v1",
    "delay_phase": "cw.direct_h1.delay_phase.v1",
    "antenna": "cw.direct_h1.antenna.v1",
    "strain": "cw.direct_h1.strain.v1",
    "site": "cw.direct_h1.site.v1",
}
COMPONENT_SPEC_VERSIONS = {
    **{name: "cw-monochromatic-h1-direct-class/v1" for name in (
        "amplitude", "clock", "delay_phase", "antenna", "strain")},
    "site": "cw-h1-site-survey/v1",
}
SOURCE = "https://arxiv.org/pdf/gr-qc/9804014v1"
C_SI = mp.mpf("299792458")
YEAR_DAYS = "365.256363004"
# LIGO T980044-10 Table 4, Earth-fixed coordinates in m and arm unit vectors.
# Preserve the publication's decimal precision; these are not the inherited
# higher-precision native coordinates.
SURVEY_H1_SITE = {
    "vertex_ecef": ("-2161414.9", "-3834695.2", "4600350.2"),
    "x_arm_ecef": ("-0.223892", "0.799831", "0.556905"),
    "y_arm_ecef": ("-0.913978", "0.0260945", "-0.404923"),
}
# IERS EOP 14 C04 (IAU1980), MJD 57279 at 0h UTC. Diagnostic only.
UT1_MINUS_UTC_2015_09_14_0H = mp.mpf("0.2553711")


def _m(value):
    return mp.mpf(str(value))


def survey_h1_ecef_from_geodetic():
    """Convert LIGO T980044-10 Table 4's printed H1 geodetic vertex.

    Section 2.1 specifies WGS-84 a=6378137 m and b=6356752.314 m.
    The returned arithmetic precision exceeds the source's angular precision.
    """
    latitude = (mp.mpf(46) + mp.mpf(27) / 60 + mp.mpf("18.528") / 3600) * mp.pi / 180
    longitude = -(mp.mpf(119) + mp.mpf(24) / 60 + mp.mpf("27.5657") / 3600) * mp.pi / 180
    height = mp.mpf("142.554")
    a, b = mp.mpf("6378137"), mp.mpf("6356752.314")
    sin_lat = mp.sin(latitude)
    radius = a / mp.sqrt(1 - (1 - (b / a)**2) * sin_lat**2)
    return ((radius + height) * mp.cos(latitude) * mp.cos(longitude),
            (radius + height) * mp.cos(latitude) * mp.sin(longitude),
            (radius * (b / a)**2 + height) * sin_lat)


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


def gmst_ut1_epoch_diagnostic():
    """Aoki GMST at 2015-09-14 0h UTC using measured UT1-UTC.

    This is a single-epoch diagnostic, not an interpolated Earth-orientation
    model or a replacement for the native UTC clock.
    """
    gps_midnight = mp.mpf("1126224017")
    utc_angle = gmst_utc_2015(gps_midnight)
    day = mp.floor((gps_midnight - 17) / 86400)
    centuries = (day - mp.mpf("7300.5")) / 36525
    rate = (mp.mpf("1.002737909350795") + mp.mpf("5.9006e-11") * centuries
            - mp.mpf("5.9e-15") * centuries**2)
    return mp.fmod(utc_angle + UT1_MINUS_UTC_2015_09_14_0H * rate
                   * 2 * mp.pi / 86400, 2 * mp.pi)


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


def survey_h1_sample(t_gps, **kwargs):
    """Whole-class scalar at the independent, publication-precision H1 site."""
    return monochromatic_h1_sample(t_gps, **SURVEY_H1_SITE, **kwargs)
