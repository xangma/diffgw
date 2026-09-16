"""Clean-room Differentiable Gravitational-Wave Detector Network Module in JAX.

Strict Clean-Room Literature Implementation:
- Anderson et al. (2001), Phys. Rev. D 63, 042003 (arXiv:gr-qc/0008066), Appendix B [Detector geometry and tensors]
- Schutz (2011), Class. Quantum Grav. 28, 125023 (arXiv:1102.5421) [Detector response and network geometry]
- Aoki et al. (1982), Astron. Astrophys. 105, 359 [IAU GMST definition]
"""

from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import jax
import jax.numpy as jnp

from diffgw_core.constants import (
    C_SI,
    PI,
)


# Earth rotation rate in rad/s (7.292115146706979e-5 rad/s = 2*pi / 86164.09053083288 s)
OMEGA_EARTH: float = 7.292115146706979e-5

# Reference GMST at GPS epoch 1126259462.0 (GW150914 event time) in radians
GPS_REF_TIME: float = 1126259462.0
GMST_REF_RAD: float = 2.4565038851564722

# IERS Bulletin C 72: (GPS time of the first UTC midnight after the insertion,
# GPS - UTC).  GPS = TAI - 19 s.  The inserted 23:59:60 uses the old offset.
GPS_LEAP_SECONDS: Sequence[Tuple[float, int]] = (
    (46828801.0, 1),
    (78364802.0, 2),
    (109900803.0, 3),
    (173059204.0, 4),
    (252028805.0, 5),
    (315187206.0, 6),
    (346723207.0, 7),
    (393984008.0, 8),
    (425520009.0, 9),
    (457056010.0, 10),
    (504489611.0, 11),
    (551750412.0, 12),
    (599184013.0, 13),
    (820108814.0, 14),
    (914803215.0, 15),
    (1025136016.0, 16),
    (1119744017.0, 17),
    (1167264018.0, 18),
)


def gps_to_utc_seconds(t_gps: Union[float, jnp.ndarray]) -> jnp.ndarray:
    """Return float64 UTC calendar seconds since 1980-01-06.

    Calendar days have 86400 nominal seconds: 23:59:60 and the following
    00:00:00 share a coordinate.  This is not a continuous elapsed-time scale.
    """
    tg = jnp.asarray(t_gps, dtype=jnp.float64)
    leap = jnp.zeros_like(tg)
    for gps_boundary, cum_leap in GPS_LEAP_SECONDS:
        leap = jnp.where(tg >= gps_boundary, float(cum_leap), leap)
    return tg - leap


# Earth-Centered Earth-Fixed (ECEF) coordinates in meters and rounded arm
# directions.  Survey checks: LIGO T980044-08 Tables 4/18; Anderson et al.
# Appendix B Table I (H1/L1/V1); KAGRA JGW-T1605295-v1 Tables 1/4/5.
DETECTOR_SITE_DATA = {
    "H1": {
        "name": "LIGO Hanford",
        "vertex": (-2161414.926331955, -3834695.17880384, 4600350.226900666),
        "x_arm": (-0.22389272, 0.79983063, 0.55690485),
        "y_arm": (-0.91397814, 0.02609386, -0.40492355),
    },
    "L1": {
        "name": "LIGO Livingston",
        "vertex": (-74276.04470677664, -5496283.719640533, 3224257.017675462),
        "x_arm": (-0.95457413, -0.14158077, -0.26218910),
        "y_arm": (0.29774148, -0.48791035, -0.82054464),
    },
    "V1": {
        "name": "Virgo",
        "vertex": (4546374.098928465, 842989.6976197582, 4378576.962654242),
        "x_arm": (-0.70045821, 0.20848949, 0.68256166),
        "y_arm": (-0.05379254, -0.96908181, 0.24080451),
    },
    "K1": {
        "name": "KAGRA",
        "vertex": (-3777336.022693953, 3484898.4121661168, 3765313.696981537),
        "x_arm": (-0.37590399, -0.83615834, 0.39941885),
        "y_arm": (0.71643788, 0.01114077, 0.69756193),
    },
}


def greenwich_mean_sidereal_time(
    t_gps: Union[float, jnp.ndarray],
) -> jnp.ndarray:
    """Compute Greenwich Mean Sidereal Time (GMST) in radians for given GPS time(s).

    Literature:
    - Aoki et al. (1982), Astron. Astrophys. 105, 359 [IAU 1982 GMST definition]
    - Explanatory Supplement to the Astronomical Almanac.
    Returns float64, including for integer inputs. Precision already lost in
    float32 cannot be recovered. Derivatives are defined away from leap transitions
    and the angle wrap.
    """
    tg = jnp.asarray(t_gps, dtype=jnp.float64)
    t_utc = gps_to_utc_seconds(tg)
    # Split the day before forming Julian centuries.
    days = jnp.floor(t_utc / 86400.0)
    ut_seconds = t_utc - days * 86400.0
    T_0 = (days - 7300.5) / 36525.0
    # Aoki et al. (1982), Eqs. (13) and (19).
    theta0 = 24110.54841 + 8640184.812866 * T_0 + 0.093104 * (T_0**2) - 6.2e-6 * (T_0**3)
    ratio = 1.002737909350795 + 5.9006e-11 * T_0 - 5.9e-15 * (T_0**2)
    theta_sec = (theta0 % 86400.0 + ut_seconds * ratio) % 86400.0
    theta_rad = theta_sec * (2.0 * PI / 86400.0)
    return theta_rad


@jax.tree_util.register_pytree_node_class
class Detector:
    """Clean-room Differentiable Interferometric Gravitational-Wave Detector in JAX."""

    def __init__(self, prefix: str):
        key = prefix.upper()
        if key not in DETECTOR_SITE_DATA:
            raise ValueError(
                f"Detector '{prefix}' not recognized. Supported: {list(DETECTOR_SITE_DATA.keys())}"
            )
        self.prefix = key
        data = DETECTOR_SITE_DATA[key]
        self.name = data["name"]

        # Register vertex and normalized arm vectors as float64 arrays (shape (3,))
        self.vertex = jnp.asarray(data["vertex"], dtype=jnp.float64)
        x_raw = jnp.asarray(data["x_arm"], dtype=jnp.float64)
        y_raw = jnp.asarray(data["y_arm"], dtype=jnp.float64)
        self.x_arm = x_raw / jnp.linalg.norm(x_raw)
        self.y_arm = y_raw / jnp.linalg.norm(y_raw)

        # Detector response tensor D^ij = 0.5 * (u^i u^j - v^i v^j)
        self.response = 0.5 * (jnp.outer(self.x_arm, self.x_arm) - jnp.outer(self.y_arm, self.y_arm))

    def tree_flatten(self):
        children = (self.vertex, self.x_arm, self.y_arm, self.response)
        aux_data = (self.prefix, self.name)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.prefix, obj.name = aux_data
        obj.vertex, obj.x_arm, obj.y_arm, obj.response = children
        return obj

    @property
    def arm_x(self) -> jnp.ndarray:
        return self.x_arm

    @property
    def arm_y(self) -> jnp.ndarray:
        return self.y_arm

    def time_delay_from_earth_center(
        self,
        ra: Union[float, jnp.ndarray],
        dec: Union[float, jnp.ndarray],
        t_gps: Union[float, jnp.ndarray],
    ) -> jnp.ndarray:
        """Compute geometric time delay Delta t = - (r_det . n) / c from Earth center in seconds.

        Args:
            ra: Right ascension in radians
            dec: Declination in radians
            t_gps: Geocentric GPS time in seconds

        Returns:
            Delta t: Time delay in seconds (|Delta t| <= R_Earth / c ~ 0.0213 s).
        """
        ra_t = jnp.asarray(ra, dtype=self.vertex.dtype)
        dec_t = jnp.asarray(dec, dtype=self.vertex.dtype)
        tg_t = jnp.asarray(t_gps, dtype=jnp.float64)

        gmst = greenwich_mean_sidereal_time(tg_t).astype(self.vertex.dtype)
        gha = gmst - ra_t

        cos_gha = jnp.cos(gha)
        sin_gha = jnp.sin(gha)
        cos_dec = jnp.cos(dec_t)
        sin_dec = jnp.sin(dec_t)

        # Source unit vector in ECEF frame (pointing toward source in the sky):
        nx = cos_dec * cos_gha
        ny = -cos_dec * sin_gha
        nz = sin_dec

        vx = self.vertex[0]
        vy = self.vertex[1]
        vz = self.vertex[2]

        proj = vx * nx + vy * ny + vz * nz
        delay = -proj / C_SI
        return delay

    def antenna_patterns(
        self,
        ra: Union[float, jnp.ndarray],
        dec: Union[float, jnp.ndarray],
        psi: Union[float, jnp.ndarray],
        t_gps: Union[float, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute antenna response factors F+ and Fx for given sky position and polarization angle.

        Literature: Anderson et al. (2001), Eqs. (B1)-(B8); Schutz (2011).

        Args:
            ra: Right ascension in radians
            dec: Declination in radians
            psi: Polarization angle in radians
            t_gps: GPS time in seconds

        Returns:
            (F+, Fx): Antenna factors satisfying F+^2 + Fx^2 <= 1.
        """
        ra_t = jnp.asarray(ra, dtype=self.vertex.dtype)
        dec_t = jnp.asarray(dec, dtype=self.vertex.dtype)
        psi_t = jnp.asarray(psi, dtype=self.vertex.dtype)
        tg_t = jnp.asarray(t_gps, dtype=jnp.float64)

        gmst = greenwich_mean_sidereal_time(tg_t).astype(self.vertex.dtype)
        gha = gmst - ra_t

        cos_gha = jnp.cos(gha)
        sin_gha = jnp.sin(gha)
        cos_dec = jnp.cos(dec_t)
        sin_dec = jnp.sin(dec_t)
        cos_psi = jnp.cos(psi_t)
        sin_psi = jnp.sin(psi_t)

        # Wave frame orthonormal vectors in Earth-fixed coordinates (Anderson et al. 2001)
        x0 = -cos_psi * sin_gha - sin_psi * cos_gha * sin_dec
        x1 = -cos_psi * cos_gha + sin_psi * sin_gha * sin_dec
        x2 = sin_psi * cos_dec

        y0 = sin_psi * sin_gha - cos_psi * cos_gha * sin_dec
        y1 = sin_psi * cos_gha + cos_psi * sin_gha * sin_dec
        y2 = cos_psi * cos_dec

        # Contract with detector response tensor D^ij
        D = self.response
        dx0 = D[0, 0] * x0 + D[0, 1] * x1 + D[0, 2] * x2
        dx1 = D[1, 0] * x0 + D[1, 1] * x1 + D[1, 2] * x2
        dx2 = D[2, 0] * x0 + D[2, 1] * x1 + D[2, 2] * x2

        dy0 = D[0, 0] * y0 + D[0, 1] * y1 + D[0, 2] * y2
        dy1 = D[1, 0] * y0 + D[1, 1] * y1 + D[1, 2] * y2
        dy2 = D[2, 0] * y0 + D[2, 1] * y1 + D[2, 2] * y2

        # F+ = x . (D x) - y . (D y)
        f_plus = (x0 * dx0 + x1 * dx1 + x2 * dx2) - (y0 * dy0 + y1 * dy1 + y2 * dy2)
        # Fx = x . (D y) + y . (D x) = 2 * (x . (D y))
        f_cross = (x0 * dy0 + x1 * dy1 + x2 * dy2) + (y0 * dx0 + y1 * dx1 + y2 * dx2)

        return f_plus, f_cross

    def project_fd_strain(
        self,
        f: Union[float, jnp.ndarray],
        h_plus: jnp.ndarray,
        h_cross: jnp.ndarray,
        ra: Union[float, jnp.ndarray],
        dec: Union[float, jnp.ndarray],
        psi: Union[float, jnp.ndarray],
        t_gps: Union[float, jnp.ndarray],
    ) -> jnp.ndarray:
        """Project frequency-domain polarizations into detector strain.

        h_det(f) = (F+ * h+(f) + Fx * hx(f)) * exp(-2*pi*i*f * Delta_t)

        Supports both (f, h_plus, h_cross, ...) and (h_plus, h_cross, f, ...) order.
        """
        if jnp.iscomplexobj(f) and not jnp.iscomplexobj(h_cross):
            hp_actual, hc_actual, f_actual = f, h_plus, h_cross
        else:
            f_actual, hp_actual, hc_actual = f, h_plus, h_cross

        f_actual = jnp.asarray(f_actual)
        hp_actual = jnp.asarray(hp_actual)
        hc_actual = jnp.asarray(hc_actual)

        f_plus, f_cross = self.antenna_patterns(ra, dec, psi, t_gps)
        delta_t = self.time_delay_from_earth_center(ra, dec, t_gps)

        # The final waveform axis is frequency. Sky parameters describe the
        # leading batch axes even when all waveforms share a one-dimensional f.
        if max(f_actual.ndim, hp_actual.ndim, hc_actual.ndim) > 0:
            f_plus = jnp.expand_dims(f_plus, -1)
            f_cross = jnp.expand_dims(f_cross, -1)
            delta_t = jnp.expand_dims(delta_t, -1)

        phase = -2.0 * PI * f_actual * delta_t
        time_shift = (jnp.cos(phase) + 1j * jnp.sin(phase)).astype(hp_actual.dtype)

        h_comb = f_plus.astype(hp_actual.dtype) * hp_actual + f_cross.astype(hc_actual.dtype) * hc_actual
        return h_comb * time_shift

    def project_td_strain(
        self,
        arg1: Union[float, jnp.ndarray],
        arg2: jnp.ndarray,
        arg3: Union[float, jnp.ndarray],
        ra: Union[float, jnp.ndarray],
        dec: Union[float, jnp.ndarray],
        psi: Union[float, jnp.ndarray],
        t_gps: Union[float, jnp.ndarray],
        delta_t: Optional[float] = None,
    ) -> jnp.ndarray:
        """Project time-domain polarizations into detector strain.

        Supports both:
            project_td_strain(hp, hc, delta_t, ra, dec, psi, t_gps)
            project_td_strain(times, hp, hc, ra, dec, psi, t_gps)
        """
        scalar_dt = isinstance(arg3, (float, int)) or (
            isinstance(arg3, (jnp.ndarray, np.ndarray))
            and (arg3.ndim == 0 or (arg3.ndim == 1 and arg3.size == 1 and delta_t is None))
        )
        if scalar_dt:
            h_plus = jnp.asarray(arg1)
            h_cross = jnp.asarray(arg2)
            dt_val = arg3
        else:
            times = jnp.asarray(arg1)
            h_plus = jnp.asarray(arg2)
            h_cross = jnp.asarray(arg3)
            dt_val = delta_t if delta_t is not None else (times[1] - times[0])

        f_plus, f_cross = self.antenna_patterns(ra, dec, psi, t_gps)
        delay = self.time_delay_from_earth_center(ra, dec, t_gps)

        shift_samples = jnp.asarray(jnp.round(delay / dt_val), dtype=jnp.int32)

        hp_shifted = jnp.roll(h_plus, shift=shift_samples, axis=-1)
        hc_shifted = jnp.roll(h_cross, shift=shift_samples, axis=-1)

        if h_plus.ndim >= 1 and f_plus.ndim < h_plus.ndim:
            f_plus = jnp.expand_dims(f_plus, -1)
            f_cross = jnp.expand_dims(f_cross, -1)

        h_det = f_plus * hp_shifted + f_cross * hc_shifted
        return h_det

    # Method aliases for convenience
    project_fd = project_fd_strain
    project_td = project_td_strain


@jax.tree_util.register_pytree_node_class
class DetectorNetwork:
    """Clean-room Differentiable Network of Gravitational-Wave Detectors in JAX."""

    def __init__(self, detectors: Sequence[Union[str, Detector]] = ("H1", "L1", "V1")):
        self.detector_dict: Dict[str, Detector] = {}
        for det in detectors:
            if isinstance(det, str):
                d = Detector(det)
            elif isinstance(det, Detector):
                d = det
            else:
                raise ValueError(f"Invalid detector specification: {det}")
            self.detector_dict[d.prefix] = d

    def tree_flatten(self):
        children = tuple(self.detector_dict.values())
        aux_data = tuple(self.detector_dict.keys())
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.detector_dict = dict(zip(aux_data, children))
        return obj

    def __getitem__(self, key: str) -> Detector:
        return self.detector_dict[key.upper()]

    def antenna_patterns(
        self,
        ra: Union[float, jnp.ndarray],
        dec: Union[float, jnp.ndarray],
        psi: Union[float, jnp.ndarray],
        t_gps: Union[float, jnp.ndarray],
    ) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Compute antenna patterns (F+, Fx) for each detector in the network."""
        return {
            prefix: det.antenna_patterns(ra, dec, psi, t_gps)
            for prefix, det in self.detector_dict.items()
        }

    def time_delays_from_earth_center(
        self,
        ra: Union[float, jnp.ndarray],
        dec: Union[float, jnp.ndarray],
        t_gps: Union[float, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Compute arrival time delays relative to Earth center for each detector in seconds."""
        return {
            prefix: det.time_delay_from_earth_center(ra, dec, t_gps)
            for prefix, det in self.detector_dict.items()
        }

    def project_fd_strain(
        self,
        f: Union[float, jnp.ndarray],
        h_plus: jnp.ndarray,
        h_cross: jnp.ndarray,
        ra: Union[float, jnp.ndarray],
        dec: Union[float, jnp.ndarray],
        psi: Union[float, jnp.ndarray],
        t_gps: Union[float, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Project frequency-domain polarizations onto each detector in the network."""
        if jnp.iscomplexobj(f) and not jnp.iscomplexobj(h_cross):
            hp_actual, hc_actual, f_actual = f, h_plus, h_cross
        else:
            f_actual, hp_actual, hc_actual = f, h_plus, h_cross

        return {
            prefix: det.project_fd_strain(f_actual, hp_actual, hc_actual, ra, dec, psi, t_gps)
            for prefix, det in self.detector_dict.items()
        }

    def project_td_strain(
        self,
        arg1: Union[float, jnp.ndarray],
        arg2: jnp.ndarray,
        arg3: Union[float, jnp.ndarray],
        ra: Union[float, jnp.ndarray],
        dec: Union[float, jnp.ndarray],
        psi: Union[float, jnp.ndarray],
        t_gps: Union[float, jnp.ndarray],
        delta_t: Optional[float] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Project time-domain polarizations onto each detector in the network."""
        return {
            prefix: det.project_td_strain(arg1, arg2, arg3, ra, dec, psi, t_gps, delta_t=delta_t)
            for prefix, det in self.detector_dict.items()
        }

    # Method and property aliases
    project_fd = project_fd_strain
    project_td = project_td_strain

    @property
    def detectors(self) -> Dict[str, Detector]:
        return self.detector_dict

    def __iter__(self):
        return iter(self.detector_dict.values())

    def items(self):
        return self.detector_dict.items()

    def keys(self):
        return self.detector_dict.keys()

    def values(self):
        return self.detector_dict.values()
