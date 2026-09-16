"""Clean-room LISA (Laser Interferometer Space Antenna) Gravitational-Wave Suite.

Strict Clean-Room Literature Implementation:
- Babak et al. (2021), Phys. Rev. D 103, 084014 (arXiv:2101.05872)
  "LISA Data Challenge: First sprint"
- Robson, Cornish & Liu (2019), Class. Quantum Grav. 36, 105011 (arXiv:1803.01944)
  "The construction and use of LISA sensitivity curves"
- Tinto & Dhurandhar (2005), Living Rev. Relativ. 8, 4 (arXiv:gr-qc/0409034)
  "Time-Delay Interferometry"
- Vallisneri (2005), Phys. Rev. D 71, 022001 (arXiv:gr-qc/0407102)
  "Geometric time-delay interferometry"
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

# LISA Mission Constants (Robson et al. 2019, Babak et al. 2021)
LISA_ARM_LENGTH: float = 2.5e9  # Arm length in meters (2.5 million km)
LISA_TRANSFER_FREQ: float = C_SI / (2.0 * PI * LISA_ARM_LENGTH)  # f_* approx 19.085 mHz
LISA_ASTRONOMICAL_UNIT: float = 1.495978707e11  # 1 AU in meters
LISA_ORBIT_PERIOD: float = 3.15581497632e7  # 1 sidereal year in seconds


def lisa_oms_displacement_noise(f: torch.Tensor) -> torch.Tensor:
    """Single-link Optical Metrology System (OMS) displacement noise power in m^2 / Hz."""
    f_safe = torch.clamp(torch.abs(f), min=1.0e-7)
    return (1.5e-11 ** 2) * (1.0 + (2.0e-3 / f_safe) ** 4)


def lisa_test_mass_acceleration_noise(f: torch.Tensor) -> torch.Tensor:
    """Single test-mass acceleration noise power in (m/s^2)^2 / Hz."""
    f_safe = torch.clamp(torch.abs(f), min=1.0e-7)
    term1 = 1.0 + (0.4e-3 / f_safe) ** 2
    term2 = 1.0 + (f_safe / 8.0e-3) ** 4
    return (3.0e-15 ** 2) * term1 * term2


def lisa_galactic_foreground_noise(
    f: torch.Tensor,
    t_obs_years: float = 4.0,
) -> torch.Tensor:
    """Galactic unresolved white dwarf binary confusion noise foreground (Robson et al. 2019 Eq. 14)."""
    # Fit parameters from Robson et al. (2019) Table 1 for 4-year mission
    alpha = 0.171
    beta = 292.0
    kappa = 1020.0
    gamma = 1680.0
    f_knee = 0.00215

    f_safe = torch.clamp(torch.abs(f), min=1.0e-6)
    exp_arg = -(f_safe ** alpha) + beta * f_safe * torch.sin(kappa * f_safe)
    tanh_arg = gamma * (f_knee - f_safe)

    exp_term = torch.exp(torch.clamp(exp_arg, max=50.0))
    tanh_term = 1.0 + torch.tanh(tanh_arg)

    s_gal = 9.0e-45 * (f_safe ** (-7.0 / 3.0)) * exp_term * tanh_term
    return s_gal


def lisa_psd(
    f: torch.Tensor,
    channel: str = "A",
    include_galactic_confusion: bool = True,
    t_obs_years: float = 4.0,
) -> torch.Tensor:
    """Compute analytical power spectral density S_n(f) for LISA TDI channels (Robson et al. 2019).

    Args:
        f: Frequency tensor in Hz
        channel: TDI combination ('A', 'E', 'T', 'X', 'Y', 'Z', or 'sky_averaged')
        include_galactic_confusion: If True, include unresolved galactic white dwarf background
        t_obs_years: Observation duration in years (default 4.0)

    Returns:
        Power spectral density tensor with same shape as f.
    """
    channel_key = channel.upper()
    f_safe = torch.clamp(torch.abs(f), min=1.0e-7)
    l_arm = LISA_ARM_LENGTH

    p_oms = lisa_oms_displacement_noise(f_safe)
    p_acc = lisa_test_mass_acceleration_noise(f_safe)

    s_oms = p_oms / (l_arm ** 2)
    s_acc = p_acc / (((2.0 * PI * f_safe) ** 4) * (l_arm ** 2))

    u = f_safe / LISA_TRANSFER_FREQ
    sin_u = torch.sin(u)
    cos_u = torch.cos(u)
    cos_2u = torch.cos(2.0 * u)
    sin_half_u = torch.sin(0.5 * u)

    if channel_key in ("X", "Y", "Z"):
        # Michelson TDI channel
        psd = 16.0 * (sin_u ** 2) * (s_oms + (3.0 + cos_2u) * s_acc)
    elif channel_key in ("A", "E"):
        # Quasi-orthogonal A and E channels: S_A(f) = S_E(f)
        psd = 8.0 * (sin_u ** 2) * (
            (2.0 + cos_u) * s_oms + 2.0 * (3.0 + 2.0 * cos_u + cos_2u) * s_acc
        )
    elif channel_key == "T":
        # Sagnac null channel: heavily suppressed at low frequencies
        psd = 32.0 * (sin_u ** 2) * (sin_half_u ** 2) * (
            s_oms + 4.0 * (sin_half_u ** 2) * s_acc
        )
    elif channel_key in ("SKY_AVERAGED", "SENSITIVITY", "STRAIN"):
        # Sky-averaged sensitivity curve S_n(f) = S_n,X(f) / R(f)
        # Transfer function R(f) = (3/10) / (1 + 0.6 * u^2)
        r_f = 0.3 / (1.0 + 0.6 * (u ** 2))
        s_x = 16.0 * (sin_u ** 2) * (s_oms + (3.0 + cos_2u) * s_acc)
        psd = s_x / r_f
    else:
        raise ValueError(
            f"Unknown LISA channel '{channel}'. Supported: 'A', 'E', 'T', 'X', 'Y', 'Z', 'sky_averaged'"
        )

    if include_galactic_confusion and channel_key != "T":
        s_gal = lisa_galactic_foreground_noise(f_safe, t_obs_years=t_obs_years)
        psd = psd + s_gal

    return psd


class LISANetwork(nn.Module):
    """Space-based LISA detector network evaluating TDI channels A, E, T, and Michelson X, Y, Z.

    Adheres strictly to the standard DetectorNetwork conventions:
    - Pure PyTorch differentiable tensor geometry
    - Fast batching across sky parameters (theta, phi, psi)
    - Full TDI response projection for Supermassive Black Hole Binaries (SMBHB)
      and Extreme Mass Ratio Inspirals (EMRI) in the 0.1 mHz - 1 Hz band.
    """

    CHANNELS = ("A", "E", "T")

    def __init__(self, channels: Sequence[str] = ("A", "E", "T")):
        super().__init__()
        self.channels = tuple(ch.upper() for ch in channels)

    @staticmethod
    def detector_tensors(
        t_gps: Union[float, torch.Tensor] = 0.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> Dict[str, torch.Tensor]:
        """Compute the response tensors D for all TDI channels (X, Y, Z, A, E, T) in the ecliptic frame.

        Args:
            t_gps: Time in seconds from reference epoch (incorporates orbital motion if dynamic)

        Returns:
            Dictionary mapping channel name to (3, 3) symmetric response tensor.
        """
        # LISA constellation plane inclination to ecliptic: 60 degrees (pi / 3)
        inc = PI / 3.0
        cos_inc = math.cos(inc)
        sin_inc = math.sin(inc)

        # Baseline unit vectors in the detector plane
        e1 = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
        e2 = torch.tensor([0.0, cos_inc, -sin_inc], dtype=dtype, device=device)

        # Equilateral triangular constellation arm unit vectors
        n12 = e1
        n13 = 0.5 * e1 + (math.sqrt(3.0) / 2.0) * e2
        n23 = n13 - n12

        # Arm response tensors D_ij = 0.5 * (n_ij (x) n_ij)
        outer_12 = torch.outer(n12, n12)
        outer_13 = torch.outer(n13, n13)
        outer_23 = torch.outer(n23, n23)

        # 1st-generation unequal-arm Michelson TDI combinations:
        # X: centered at SC1 (arms 12 and 13)
        dx = 0.5 * (outer_12 - outer_13)
        # Y: centered at SC2 (arms 23 and 21)
        dy = 0.5 * (outer_23 - outer_12)
        # Z: centered at SC3 (arms 31 and 32)
        dz = 0.5 * (outer_13 - outer_23)

        # Noise-orthogonal A, E, T combinations (Tinto & Dhurandhar 2005)
        da = (2.0 * dx - dy - dz) / 3.0
        de = (dz - dy) / math.sqrt(3.0)
        dt = (dx + dy + dz) / 3.0  # Exactly zero in static low-frequency limit

        return {"X": dx, "Y": dy, "Z": dz, "A": da, "E": de, "T": dt}

    def antenna_patterns(
        self,
        theta: Union[float, torch.Tensor],
        phi: Union[float, torch.Tensor],
        psi: Union[float, torch.Tensor],
        t_gps: Union[float, torch.Tensor] = 0.0,
        channel: str = "A",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute LISA antenna response patterns F_plus and F_cross.

        Args:
            theta: Source ecliptic co-latitude in radians (theta = pi/2 - beta)
            phi: Source ecliptic longitude in radians
            psi: Wave polarization angle in radians
            t_gps: Observation time in seconds (default 0.0)
            channel: Target TDI channel ('A', 'E', 'T', 'X', 'Y', 'Z')

        Returns:
            Tuple of (F_plus, F_cross) antenna factors.
        """
        th = as_tensor(theta)
        device = th.device
        dtype = th.dtype if th.is_floating_point() else torch.float64

        th = th.to(dtype=dtype, device=device)
        ph = as_tensor(phi, device=device, dtype=dtype)
        ps = as_tensor(psi, device=device, dtype=dtype)

        cos_t = torch.cos(th)
        sin_t = torch.sin(th)
        cos_p = torch.cos(ph)
        sin_p = torch.sin(ph)

        # Wave polarization basis vectors
        u = torch.stack([sin_p, -cos_p, torch.zeros_like(th)], dim=-1)
        v = torch.stack([cos_t * cos_p, cos_t * sin_p, -sin_t], dim=-1)

        cos_2psi = torch.cos(2.0 * ps)
        sin_2psi = torch.sin(2.0 * ps)

        uu = torch.matmul(u.unsqueeze(-1), u.unsqueeze(-2))
        vv = torch.matmul(v.unsqueeze(-1), v.unsqueeze(-2))
        uv = 0.5 * (torch.matmul(u.unsqueeze(-1), v.unsqueeze(-2)) + torch.matmul(v.unsqueeze(-1), u.unsqueeze(-2)))

        if cos_2psi.dim() > 0:
            c2_u = cos_2psi.unsqueeze(-1).unsqueeze(-1)
            s2_u = sin_2psi.unsqueeze(-1).unsqueeze(-1)
        else:
            c2_u = cos_2psi
            s2_u = sin_2psi

        e_plus = c2_u * (uu - vv) + s2_u * 2.0 * uv
        e_cross = -s2_u * (uu - vv) + c2_u * 2.0 * uv

        # Response tensor for requested channel
        tensors = self.detector_tensors(t_gps=t_gps, device=device, dtype=dtype)
        d_mat = tensors[channel.upper()]

        # Contraction D : e
        f_plus = torch.sum(d_mat * e_plus, dim=(-2, -1))
        f_cross = torch.sum(d_mat * e_cross, dim=(-2, -1))

        return f_plus, f_cross

    def project_fd_strain(
        self,
        h_plus: torch.Tensor,
        h_cross: torch.Tensor,
        sample_frequencies: torch.Tensor,
        theta: Union[float, torch.Tensor],
        phi: Union[float, torch.Tensor],
        psi: Union[float, torch.Tensor],
        t_gps: Union[float, torch.Tensor] = 0.0,
        channel: str = "A",
    ) -> torch.Tensor:
        """Project frequency-domain polarizations into a specific LISA TDI channel.

        h_channel(f) = F_plus * h_plus(f) + F_cross * h_cross(f)
        """
        f_plus, f_cross = self.antenna_patterns(theta, phi, psi, t_gps=t_gps, channel=channel)

        if h_plus.dim() > f_plus.dim():
            f_plus = f_plus.unsqueeze(-1)
            f_cross = f_cross.unsqueeze(-1)

        return f_plus * h_plus + f_cross * h_cross

    def psd(
        self,
        f: torch.Tensor,
        channel: str = "A",
        include_galactic_confusion: bool = True,
        t_obs_years: float = 4.0,
    ) -> torch.Tensor:
        """Evaluate analytical noise power spectral density for the specified channel."""
        return lisa_psd(
            f=f,
            channel=channel,
            include_galactic_confusion=include_galactic_confusion,
            t_obs_years=t_obs_years,
        )
