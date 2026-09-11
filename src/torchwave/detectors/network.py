"""Clean-room Differentiable Gravitational-Wave Detector Network Module.

Strict Clean-Room Literature Implementation:
- Anderson et al. (2001), Phys. Rev. D 63, 042003 (arXiv:gr-qc/0001044) [Detector geometry and tensors]
- Schutz (2011), Class. Quantum Grav. 28, 125023 (arXiv:1102.5421) [Detector response and network geometry]
- Aoki et al. (1982), Astron. Astrophys. 105, 359 [IAU GMST definition]
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn

from torchwave.constants import (
    C_SI,
    PI,
)
from torchwave.conversions import as_tensor


# Earth rotation rate in rad/s (7.292115146706979e-5 rad/s = 2*pi / 86164.09053083288 s)
OMEGA_EARTH: float = 7.292115146706979e-5

# Reference GMST at GPS epoch 1126259462.0 (GW150914 event time) in radians
GPS_REF_TIME: float = 1126259462.0
GMST_REF_RAD: float = 2.4565038851564722

# Standard GPS leap-seconds table: (GPS timestamp boundary, cumulative leap seconds)
GPS_LEAP_SECONDS: Sequence[Tuple[float, int]] = (
    (46828800.0, 1),
    (78364801.0, 2),
    (109900802.0, 3),
    (173059203.0, 4),
    (252028804.0, 5),
    (315187205.0, 6),
    (346723206.0, 7),
    (393984007.0, 8),
    (425520008.0, 9),
    (457056009.0, 10),
    (504489610.0, 11),
    (551750411.0, 12),
    (599184012.0, 13),
    (820108813.0, 14),
    (914803214.0, 15),
    (1024012815.0, 16),
    (1119744016.0, 17),
    (1167264017.0, 18),
)


def gps_to_utc_seconds(t_gps: torch.Tensor) -> torch.Tensor:
    """Convert continuous GPS seconds to UTC seconds since 1980-01-06 00:00:00 UTC."""
    leap = torch.zeros_like(t_gps)
    for gps_boundary, cum_leap in GPS_LEAP_SECONDS:
        leap = torch.where(t_gps >= gps_boundary, float(cum_leap), leap)
    return t_gps - leap


# Exact Earth-Centered Earth-Fixed (ECEF) coordinates in meters and unit arm vectors
# Sources: Anderson et al. (2001), Phys. Rev. D 63, 042003 Table I; LAL / ECEF Geodesy standards.
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
    t_gps: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Compute Greenwich Mean Sidereal Time (GMST) in radians for given GPS time(s).

    Literature:
    - Aoki et al. (1982), Astron. Astrophys. 105, 359 [IAU 1982 GMST definition]
    - Explanatory Supplement to the Astronomical Almanac.
    Fully differentiable with respect to GPS time.
    """
    is_tensor = isinstance(t_gps, torch.Tensor)
    tg = t_gps if is_tensor else torch.tensor(t_gps, dtype=torch.float64)
    t_utc = gps_to_utc_seconds(tg)
    jd = 2444244.5 + t_utc / 86400.0
    jd_0h = torch.floor(jd - 0.5) + 0.5
    T_0 = (jd_0h - 2451545.0) / 36525.0
    ut_seconds = (jd - jd_0h) * 86400.0
    theta0 = 24110.54841 + 8640184.812866 * T_0 + 0.093104 * (T_0**2) - 6.2e-6 * (T_0**3)
    ratio = 1.002737909350795 + 5.9006e-11 * T_0 - 5.9e-15 * (T_0**2)
    theta_sec = theta0 + ut_seconds * ratio
    theta_rad = theta_sec * (2.0 * PI / 86400.0)
    return theta_rad % (2.0 * PI)


class Detector(nn.Module):
    """Clean-room Differentiable Interferometric Gravitational-Wave Detector."""

    def __init__(self, prefix: str):
        super().__init__()
        key = prefix.upper()
        if key not in DETECTOR_SITE_DATA:
            raise ValueError(
                f"Detector '{prefix}' not recognized. Supported: {list(DETECTOR_SITE_DATA.keys())}"
            )
        self.prefix = key
        data = DETECTOR_SITE_DATA[key]
        self.name = data["name"]

        # Register vertex and arm vectors as buffers (shape (3,))
        self.register_buffer(
            "vertex",
            torch.tensor(data["vertex"], dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "x_arm",
            torch.tensor(data["x_arm"], dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "y_arm",
            torch.tensor(data["y_arm"], dtype=torch.float64),
            persistent=True,
        )

        # Detector response tensor D^ij = 0.5 * (u^i u^j - v^i v^j)
        u = self.x_arm
        # Register response tensor
        response = 0.5 * (torch.outer(self.x_arm, self.x_arm) - torch.outer(self.y_arm, self.y_arm))
        self.register_buffer("response", response.to(dtype=self.x_arm.dtype))

    @property
    def arm_x(self) -> torch.Tensor:
        return self.x_arm

    @property
    def arm_y(self) -> torch.Tensor:
        return self.y_arm

    def time_delay_from_earth_center(
        self,
        ra: Union[float, torch.Tensor],
        dec: Union[float, torch.Tensor],
        t_gps: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Compute geometric time delay Delta t = - (r_det . n) / c from Earth center in seconds.

        Args:
            ra: Right ascension in radians
            dec: Declination in radians
            t_gps: Geocentric GPS time in seconds

        Returns:
            Delta t: Time delay in seconds (|Delta t| <= R_Earth / c ~ 0.0213 s).
        """
        device = self.vertex.device
        dtype = self.vertex.dtype

        ra_t = as_tensor(ra, device=device, dtype=dtype)
        dec_t = as_tensor(dec, device=device, dtype=dtype)
        tg_t = as_tensor(t_gps, device=device, dtype=dtype)

        gmst = greenwich_mean_sidereal_time(tg_t)
        gha = gmst - ra_t

        cos_gha = torch.cos(gha)
        sin_gha = torch.sin(gha)
        cos_dec = torch.cos(dec_t)
        sin_dec = torch.sin(dec_t)

        # Source unit vector in ECEF frame (pointing toward source in the sky):
        # n = [cos(dec)*cos(gha), -cos(dec)*sin(gha), sin(dec)]
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
        ra: Union[float, torch.Tensor],
        dec: Union[float, torch.Tensor],
        psi: Union[float, torch.Tensor],
        t_gps: Union[float, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute antenna response factors F+ and Fx for given sky position and polarization angle.

        Literature: Anderson et al. (2001) Eqs. 20-25; Schutz (2011).

        Args:
            ra: Right ascension in radians
            dec: Declination in radians
            psi: Polarization angle in radians
            t_gps: GPS time in seconds

        Returns:
            (F+, Fx): Antenna factors satisfying F+^2 + Fx^2 <= 1.
        """
        device = self.vertex.device
        dtype = self.vertex.dtype

        ra_t = as_tensor(ra, device=device, dtype=dtype)
        dec_t = as_tensor(dec, device=device, dtype=dtype)
        psi_t = as_tensor(psi, device=device, dtype=dtype)
        tg_t = as_tensor(t_gps, device=device, dtype=dtype)

        gmst = greenwich_mean_sidereal_time(tg_t)
        gha = gmst - ra_t

        cos_gha = torch.cos(gha)
        sin_gha = torch.sin(gha)
        cos_dec = torch.cos(dec_t)
        sin_dec = torch.sin(dec_t)
        cos_psi = torch.cos(psi_t)
        sin_psi = torch.sin(psi_t)

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
        f: torch.Tensor,
        h_plus: torch.Tensor,
        h_cross: torch.Tensor,
        ra: Union[float, torch.Tensor],
        dec: Union[float, torch.Tensor],
        psi: Union[float, torch.Tensor],
        t_gps: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Project frequency-domain polarizations into detector strain.

        h_det(f) = (F+ * h+(f) + Fx * hx(f)) * exp(-2*pi*i*f * Delta_t)

        Args:
            f: Frequency tensor in Hz
            h_plus, h_cross: Complex frequency-domain polarization tensors
            ra: Right ascension in radians
            dec: Declination in radians
            psi: Polarization angle in radians
            t_gps: Geocentric GPS time in seconds

        Returns:
            h_det(f): Complex strain tensor at detector.
        """
        # Support both (f, h_plus, h_cross, ...) and (h_plus, h_cross, f, ...)
        if torch.is_complex(f) and not torch.is_complex(h_cross):
            hp_actual, hc_actual, f_actual = f, h_plus, h_cross
        else:
            f_actual, hp_actual, hc_actual = f, h_plus, h_cross

        f_plus, f_cross = self.antenna_patterns(ra, dec, psi, t_gps)
        delta_t = self.time_delay_from_earth_center(ra, dec, t_gps)

        # Broadcast parameters across frequency dimension
        if f_actual.dim() >= 1 and f_plus.dim() < f_actual.dim():
            f_plus = f_plus.unsqueeze(-1)
            f_cross = f_cross.unsqueeze(-1)
            delta_t = delta_t.unsqueeze(-1)

        # Geometric time delay phase factor: exp(-2*pi*i*f * delta_t)
        phase = -2.0 * PI * f_actual * delta_t
        time_shift = torch.complex(torch.cos(phase), torch.sin(phase))

        h_comb = f_plus.to(hp_actual.dtype) * hp_actual + f_cross.to(hc_actual.dtype) * hc_actual
        return h_comb * time_shift.to(h_comb.dtype)

    def project_td_strain(
        self,
        arg1: Union[float, torch.Tensor],
        arg2: torch.Tensor,
        arg3: Union[float, torch.Tensor],
        ra: Union[float, torch.Tensor],
        dec: Union[float, torch.Tensor],
        psi: Union[float, torch.Tensor],
        t_gps: Union[float, torch.Tensor],
        delta_t: Optional[float] = None,
    ) -> torch.Tensor:
        """Project time-domain polarizations into detector strain.

        Supports both:
            project_td_strain(hp, hc, delta_t, ra, dec, psi, t_gps)
            project_td_strain(times, hp, hc, ra, dec, psi, t_gps)

        Returns:
            h_det(t): Real strain tensor at detector.
        """
        if isinstance(arg3, (float, int)) or (isinstance(arg3, torch.Tensor) and arg3.numel() == 1):
            h_plus = arg1
            h_cross = arg2
            dt_val = float(arg3)
        else:
            times = arg1
            h_plus = arg2
            h_cross = arg3
            dt_val = delta_t if delta_t is not None else float((times[1] - times[0]).item())

        f_plus, f_cross = self.antenna_patterns(ra, dec, psi, t_gps)
        delay = self.time_delay_from_earth_center(ra, dec, t_gps)

        # Shift in discrete samples
        shift_samples = int(torch.round(delay / dt_val).item())

        hp_shifted = torch.roll(h_plus, shifts=shift_samples, dims=-1)
        hc_shifted = torch.roll(h_cross, shifts=shift_samples, dims=-1)

        if h_plus.dim() >= 1 and f_plus.dim() < h_plus.dim():
            f_plus = f_plus.unsqueeze(-1)
            f_cross = f_cross.unsqueeze(-1)

        h_det = f_plus * hp_shifted + f_cross * hc_shifted
        return h_det

    # Method aliases for convenience
    project_fd = project_fd_strain
    project_td = project_td_strain


class DetectorNetwork(nn.Module):
    """Clean-room Differentiable Network of Gravitational-Wave Detectors."""

    def __init__(self, detectors: Sequence[Union[str, Detector]] = ("H1", "L1", "V1")):
        super().__init__()
        self.detector_dict = nn.ModuleDict()
        for det in detectors:
            if isinstance(det, str):
                d = Detector(det)
            elif isinstance(det, Detector):
                d = det
            else:
                raise ValueError(f"Invalid detector specification: {det}")
            self.detector_dict[d.prefix] = d

    def __getitem__(self, key: str) -> Detector:
        return self.detector_dict[key.upper()]

    def antenna_patterns(
        self,
        ra: Union[float, torch.Tensor],
        dec: Union[float, torch.Tensor],
        psi: Union[float, torch.Tensor],
        t_gps: Union[float, torch.Tensor],
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute antenna patterns (F+, Fx) for each detector in the network."""
        return {
            prefix: det.antenna_patterns(ra, dec, psi, t_gps)
            for prefix, det in self.detector_dict.items()
        }

    def time_delays_from_earth_center(
        self,
        ra: Union[float, torch.Tensor],
        dec: Union[float, torch.Tensor],
        t_gps: Union[float, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute arrival time delays relative to Earth center for each detector in seconds."""
        return {
            prefix: det.time_delay_from_earth_center(ra, dec, t_gps)
            for prefix, det in self.detector_dict.items()
        }

    def project_fd_strain(
        self,
        f: torch.Tensor,
        h_plus: torch.Tensor,
        h_cross: torch.Tensor,
        ra: Union[float, torch.Tensor],
        dec: Union[float, torch.Tensor],
        psi: Union[float, torch.Tensor],
        t_gps: Union[float, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Project frequency-domain polarizations onto each detector in the network."""
        # Support both (f, h_plus, h_cross, ...) and (h_plus, h_cross, f, ...)
        if torch.is_complex(f) and not torch.is_complex(h_cross):
            hp_actual, hc_actual, f_actual = f, h_plus, h_cross
        else:
            f_actual, hp_actual, hc_actual = f, h_plus, h_cross

        return {
            prefix: det.project_fd_strain(f_actual, hp_actual, hc_actual, ra, dec, psi, t_gps)
            for prefix, det in self.detector_dict.items()
        }

    def project_td_strain(
        self,
        arg1: Union[float, torch.Tensor],
        arg2: torch.Tensor,
        arg3: Union[float, torch.Tensor],
        ra: Union[float, torch.Tensor],
        dec: Union[float, torch.Tensor],
        psi: Union[float, torch.Tensor],
        t_gps: Union[float, torch.Tensor],
        delta_t: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Project time-domain polarizations onto each detector in the network."""
        return {
            prefix: det.project_td_strain(arg1, arg2, arg3, ra, dec, psi, t_gps, delta_t=delta_t)
            for prefix, det in self.detector_dict.items()
        }

    # Method and property aliases
    project_fd = project_fd_strain
    project_td = project_td_strain

    @property
    def detectors(self) -> nn.ModuleDict:
        return self.detector_dict

    def __iter__(self):
        return iter(self.detector_dict.values())

    def items(self):
        return self.detector_dict.items()

    def keys(self):
        return self.detector_dict.keys()

    def values(self):
        return self.detector_dict.values()
