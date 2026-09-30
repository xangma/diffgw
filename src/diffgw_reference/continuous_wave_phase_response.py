"""Scalar JKS continuous-wave phase and detector response (PDF numbering).

Source: Jaranowski, Królak & Schutz, gr-qc/9804014v1 PDF, Sec. 2,
Eqs. (1)-(6), (10)-(14), (18)-(22). The arXiv HTML conversion of the *same v1*
shifts equation numbering; use the PDF numbers here. Here ``f0`` and
its derivatives are GW frequencies for the triaxial 2f component: the paper's
rotational phase and its frequency derivatives are doubled.

This is the paper's first-order Roemer phase, not a barycentric polynomial
in ``t + tau``. An orbit phase or local sidereal phase must be supplied; the
paper does not tie either to the GPS epoch. No ephemeris or site survey data
are imported from native code.
"""

from __future__ import annotations

import mpmath as mp

SPEC_VERSION = "cw.phase_response.paper/v1"
SOURCE = "https://arxiv.org/pdf/gr-qc/9804014v1"
SOURCE_EQUATIONS = (
    "1", "2", "3", "4", "5", "6", "10", "11", "12", "13", "14",
    "18", "19", "20", "21", "22",
)
C_SI = mp.mpf("299792458")


def _m(value):
    return mp.mpf(str(value))


def jks_circular_roemer_delay(
    ra, dec, *, orbit_phase, local_sidereal=0, latitude=None,
    orbital_radius_m="149597870700", earth_radius_m=0, obliquity="0.4090928042223289",
):
    """Eq. (18) geometric delay, with orbital/sidereal phases supplied.

    ``local_sidereal`` is the detector site's angle to the vernal point;
    the rotational term is omitted when ``latitude`` is None. The epoch and
    angular velocities belong to the caller's specified timing convention.
    """
    ra, dec, op = _m(ra), _m(dec), _m(orbit_phase)
    eps = _m(obliquity)
    orbit = _m(orbital_radius_m) * (
        mp.cos(ra) * mp.cos(dec) * mp.cos(op)
        + (mp.cos(eps) * mp.sin(ra) * mp.cos(dec) + mp.sin(eps) * mp.sin(dec)) * mp.sin(op)
    )
    spin = mp.mpf("0")
    if latitude is not None:
        lat, lst = _m(latitude), _m(local_sidereal)
        spin = _m(earth_radius_m) * (
            mp.sin(lat) * mp.sin(dec) + mp.cos(lat) * mp.cos(dec) * mp.cos(ra - lst)
        )
    return (orbit + spin) / C_SI


def jks_2f_phase(dt, tau, f0, f1=0, f2=0, phi0=0):
    """Twice JKS Eq. (14), mapped to the carrier's GW Hz and phase origin.

    ``dt`` is seconds since the SSB frequency-reference epoch; ``tau`` is
    ``n . r_d / c`` at detector time. ``phi0`` equals twice JKS's Phi_0.
    """
    dt, tau, f0, f1, f2, phi0 = map(_m, (dt, tau, f0, f1, f2, phi0))
    intrinsic = f0 * dt + f1 * dt**2 / 2 + f2 * dt**3 / 6
    instantaneous_frequency = f0 + f1 * dt + f2 * dt**2 / 2
    return phi0 + 2 * mp.pi * (intrinsic + tau * instantaneous_frequency)


def jks_beam_patterns(ra, dec, psi, latitude, gamma, zeta, local_sidereal):
    """JKS Eqs. (10)-(13); gamma runs east to the arm bisector, CCW."""
    ra, dec, psi, lat, gamma, zeta, lst = map(
        _m, (ra, dec, psi, latitude, gamma, zeta, local_sidereal)
    )
    hour = ra - lst
    a = (
        mp.sin(2 * gamma) * (3 - mp.cos(2 * lat)) * (3 - mp.cos(2 * dec)) * mp.cos(2 * hour) / 16
        - mp.cos(2 * gamma) * mp.sin(lat) * (3 - mp.cos(2 * dec)) * mp.sin(2 * hour) / 4
        + mp.sin(2 * gamma) * mp.sin(2 * lat) * mp.sin(2 * dec) * mp.cos(hour) / 4
        - mp.cos(2 * gamma) * mp.cos(lat) * mp.sin(2 * dec) * mp.sin(hour) / 2
        + 3 * mp.sin(2 * gamma) * mp.cos(lat)**2 * mp.cos(dec)**2 / 4
    )
    b = (
        mp.cos(2 * gamma) * mp.sin(lat) * mp.sin(dec) * mp.cos(2 * hour)
        + mp.sin(2 * gamma) * (3 - mp.cos(2 * lat)) * mp.sin(dec) * mp.sin(2 * hour) / 4
        + mp.cos(2 * gamma) * mp.cos(lat) * mp.cos(dec) * mp.cos(hour)
        + mp.sin(2 * gamma) * mp.sin(2 * lat) * mp.cos(dec) * mp.sin(hour) / 2
    )
    return (
        mp.sin(zeta) * (a * mp.cos(2 * psi) + b * mp.sin(2 * psi)),
        mp.sin(zeta) * (b * mp.cos(2 * psi) - a * mp.sin(2 * psi)),
    )


def jks_arm_tensor_beam_patterns(ra, dec, psi, gmst, x_arm_ecef, y_arm_ecef):
    """JKS PDF Eqs. (1)-(6) for supplied Earth-fixed unit arm vectors.

    The wave-frame axes are the first two rows of Eq. (6), rotated from the
    celestial frame into ECEF by the supplied Greenwich sidereal angle. This
    keeps the response algebra separate from a site's survey and clock model.
    """
    ra, dec, psi, gmst = map(_m, (ra, dec, psi, gmst))
    x_arm = tuple(map(_m, x_arm_ecef))
    y_arm = tuple(map(_m, y_arm_ecef))
    if len(x_arm) != 3 or len(y_arm) != 3:
        raise ValueError("each arm must have three ECEF components")

    wave_x = (
        mp.sin(ra) * mp.cos(psi) - mp.cos(ra) * mp.sin(dec) * mp.sin(psi),
        -mp.cos(ra) * mp.cos(psi) - mp.sin(ra) * mp.sin(dec) * mp.sin(psi),
        mp.cos(dec) * mp.sin(psi),
    )
    wave_y = (
        -mp.sin(ra) * mp.sin(psi) - mp.cos(ra) * mp.sin(dec) * mp.cos(psi),
        mp.cos(ra) * mp.sin(psi) - mp.sin(ra) * mp.sin(dec) * mp.cos(psi),
        mp.cos(dec) * mp.cos(psi),
    )
    cg, sg = mp.cos(gmst), mp.sin(gmst)

    def earth_fixed(v):
        return (cg * v[0] + sg * v[1], -sg * v[0] + cg * v[1], v[2])

    def dot(a, b):
        return sum((p * q for p, q in zip(a, b)), mp.mpf("0"))

    wx, wy = earth_fixed(wave_x), earth_fixed(wave_y)
    xx, xy = dot(x_arm, wx), dot(x_arm, wy)
    yx, yy = dot(y_arm, wx), dot(y_arm, wy)
    return ((xx**2 - xy**2 - yx**2 + yy**2) / 2, xx * xy - yx * yy)


def jks_2f_strain(h0, cos_iota, gw_phase, f_plus, f_cross):
    """JKS Eqs. (19)-(22) at theta=pi/2, retaining only the 2f branch."""
    h0, ci, phase, fp, fx = map(_m, (h0, cos_iota, gw_phase, f_plus, f_cross))
    return h0 * ((1 + ci**2) * fp * mp.cos(phase) / 2 + ci * fx * mp.sin(phase))
