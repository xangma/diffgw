"""Clean-room LISA (Laser Interferometer Space Antenna) Gravitational-Wave Suite in JAX.

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

from typing import Dict, Optional, Sequence, Tuple, Union
import math
import jax
import jax.numpy as jnp

from diffgw_core.constants import (
    C_SI,
    PI,
)

# LISA Mission Constants (Robson et al. 2019, Babak et al. 2021)
LISA_ARM_LENGTH: float = 2.5e9  # Arm length in meters (2.5 million km)
LISA_TRANSFER_FREQ: float = C_SI / (2.0 * PI * LISA_ARM_LENGTH)  # f_* approx 19.085 mHz
LISA_ASTRONOMICAL_UNIT: float = 1.495978707e11  # 1 AU in meters
LISA_ORBIT_PERIOD: float = 3.15581497632e7  # 1 sidereal year in seconds


def lisa_oms_displacement_noise(f: Union[float, jnp.ndarray]) -> jnp.ndarray:
    """Single-link Optical Metrology System (OMS) displacement noise power in m^2 / Hz."""
    f_safe = jnp.maximum(jnp.abs(jnp.asarray(f, dtype=jnp.float64)), 1.0e-7)
    return (1.5e-11 ** 2) * (1.0 + (2.0e-3 / f_safe) ** 4)


def lisa_test_mass_acceleration_noise(f: Union[float, jnp.ndarray]) -> jnp.ndarray:
    """Single test-mass acceleration noise power in (m/s^2)^2 / Hz."""
    f_safe = jnp.maximum(jnp.abs(jnp.asarray(f, dtype=jnp.float64)), 1.0e-7)
    term1 = 1.0 + (0.4e-3 / f_safe) ** 2
    term2 = 1.0 + (f_safe / 8.0e-3) ** 4
    return (3.0e-15 ** 2) * term1 * term2


def lisa_galactic_foreground_noise(
    f: Union[float, jnp.ndarray],
    t_obs_years: float = 4.0,
) -> jnp.ndarray:
    """Galactic unresolved white dwarf binary confusion noise foreground (Robson et al. 2019 Eq. 14)."""
    alpha = 0.171
    beta = 292.0
    kappa = 1020.0
    gamma = 1680.0
    f_knee = 0.00215

    f_safe = jnp.maximum(jnp.abs(jnp.asarray(f, dtype=jnp.float64)), 1.0e-6)
    exp_arg = -(f_safe ** alpha) + beta * f_safe * jnp.sin(kappa * f_safe)
    tanh_arg = gamma * (f_knee - f_safe)

    exp_term = jnp.exp(jnp.minimum(exp_arg, 50.0))
    tanh_term = 1.0 + jnp.tanh(tanh_arg)

    s_gal = 9.0e-45 * (f_safe ** (-7.0 / 3.0)) * exp_term * tanh_term
    return s_gal


def lisa_psd(
    f: Union[float, jnp.ndarray],
    channel: str = "A",
    include_galactic_confusion: bool = True,
    t_obs_years: float = 4.0,
) -> jnp.ndarray:
    """Compute analytical power spectral density S_n(f) for LISA TDI channels (Robson et al. 2019) in JAX.

    Args:
        f: Frequency array or scalar in Hz
        channel: TDI combination ('A', 'E', 'T', 'X', 'Y', 'Z', or 'sky_averaged')
        include_galactic_confusion: If True, include unresolved galactic white dwarf background
        t_obs_years: Observation duration in years (default 4.0)

    Returns:
        Power spectral density array with same shape as f.
    """
    channel_key = channel.upper()
    f_arr = jnp.asarray(f, dtype=jnp.float64)
    f_safe = jnp.maximum(jnp.abs(f_arr), 1.0e-7)
    l_arm = LISA_ARM_LENGTH

    p_oms = lisa_oms_displacement_noise(f_safe)
    p_acc = lisa_test_mass_acceleration_noise(f_safe)

    s_oms = p_oms / (l_arm ** 2)
    s_acc = p_acc / (((2.0 * PI * f_safe) ** 4) * (l_arm ** 2))

    u = f_safe / LISA_TRANSFER_FREQ
    sin_u = jnp.sin(u)
    cos_u = jnp.cos(u)
    cos_2u = jnp.cos(2.0 * u)
    sin_half_u = jnp.sin(0.5 * u)

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


@jax.tree_util.register_pytree_node_class
class LISANetwork:
    """Space-based LISA detector network evaluating TDI channels A, E, T, and Michelson X, Y, Z in JAX."""

    CHANNELS = ("A", "E", "T")

    def __init__(self, channels: Sequence[str] = ("A", "E", "T")):
        self.channels = tuple(ch.upper() for ch in channels)

    def tree_flatten(self):
        return (), (self.channels,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.channels = aux_data[0]
        return obj

    @staticmethod
    def detector_tensors(
        t_gps: Union[float, jnp.ndarray] = 0.0,
        dtype=jnp.float64,
    ) -> Dict[str, jnp.ndarray]:
        """Compute the response tensors D for all TDI channels (X, Y, Z, A, E, T) in the ecliptic frame.

        Args:
            t_gps: Time in seconds from reference epoch
            dtype: Floating point dtype (default jnp.float64)

        Returns:
            Dictionary mapping channel name to (3, 3) symmetric response tensor.
        """
        # LISA constellation plane inclination to ecliptic: 60 degrees (pi / 3)
        inc = PI / 3.0
        cos_inc = math.cos(inc)
        sin_inc = math.sin(inc)

        # Baseline unit vectors in the detector plane
        e1 = jnp.array([1.0, 0.0, 0.0], dtype=dtype)
        e2 = jnp.array([0.0, cos_inc, -sin_inc], dtype=dtype)

        # Equilateral triangular constellation arm unit vectors
        n12 = e1
        n13 = 0.5 * e1 + (math.sqrt(3.0) / 2.0) * e2
        n23 = n13 - n12

        # Arm response tensors D_ij = 0.5 * (n_ij (x) n_ij)
        outer_12 = jnp.outer(n12, n12)
        outer_13 = jnp.outer(n13, n13)
        outer_23 = jnp.outer(n23, n23)

        # 1st-generation unequal-arm Michelson TDI combinations:
        dx = 0.5 * (outer_12 - outer_13)
        dy = 0.5 * (outer_23 - outer_12)
        dz = 0.5 * (outer_13 - outer_23)

        # Noise-orthogonal A, E, T combinations (Tinto & Dhurandhar 2005)
        da = (2.0 * dx - dy - dz) / 3.0
        de = (dz - dy) / math.sqrt(3.0)
        dt = (dx + dy + dz) / 3.0

        return {"X": dx, "Y": dy, "Z": dz, "A": da, "E": de, "T": dt}

    def antenna_patterns(
        self,
        theta: Union[float, jnp.ndarray],
        phi: Union[float, jnp.ndarray],
        psi: Union[float, jnp.ndarray],
        t_gps: Union[float, jnp.ndarray] = 0.0,
        channel: str = "A",
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute LISA antenna response patterns F_plus and F_cross in JAX.

        Args:
            theta: Source ecliptic co-latitude in radians (theta = pi/2 - beta)
            phi: Source ecliptic longitude in radians
            psi: Wave polarization angle in radians
            t_gps: Observation time in seconds (default 0.0)
            channel: Target TDI channel ('A', 'E', 'T', 'X', 'Y', 'Z')

        Returns:
            Tuple of (F_plus, F_cross) antenna factors.
        """
        th = jnp.asarray(theta, dtype=jnp.float64)
        ph = jnp.asarray(phi, dtype=jnp.float64)
        ps = jnp.asarray(psi, dtype=jnp.float64)

        cos_t = jnp.cos(th)
        sin_t = jnp.sin(th)
        cos_p = jnp.cos(ph)
        sin_p = jnp.sin(ph)

        # Wave polarization basis vectors
        u = jnp.stack([sin_p, -cos_p, jnp.zeros_like(th)], axis=-1)
        v = jnp.stack([cos_t * cos_p, cos_t * sin_p, -sin_t], axis=-1)

        cos_2psi = jnp.cos(2.0 * ps)
        sin_2psi = jnp.sin(2.0 * ps)

        uu = jnp.matmul(jnp.expand_dims(u, -1), jnp.expand_dims(u, -2))
        vv = jnp.matmul(jnp.expand_dims(v, -1), jnp.expand_dims(v, -2))
        uv = 0.5 * (
            jnp.matmul(jnp.expand_dims(u, -1), jnp.expand_dims(v, -2))
            + jnp.matmul(jnp.expand_dims(v, -1), jnp.expand_dims(u, -2))
        )

        if cos_2psi.ndim > 0:
            c2_u = jnp.expand_dims(jnp.expand_dims(cos_2psi, -1), -1)
            s2_u = jnp.expand_dims(jnp.expand_dims(sin_2psi, -1), -1)
        else:
            c2_u = cos_2psi
            s2_u = sin_2psi

        e_plus = c2_u * (uu - vv) + s2_u * 2.0 * uv
        e_cross = -s2_u * (uu - vv) + c2_u * 2.0 * uv

        # Response tensor for requested channel
        tensors = self.detector_tensors(t_gps=t_gps, dtype=th.dtype)
        d_mat = tensors[channel.upper()]

        # Contraction D : e
        f_plus = jnp.sum(d_mat * e_plus, axis=(-2, -1))
        f_cross = jnp.sum(d_mat * e_cross, axis=(-2, -1))

        return f_plus, f_cross

    def project_fd_strain(
        self,
        h_plus: jnp.ndarray,
        h_cross: jnp.ndarray,
        sample_frequencies: jnp.ndarray,
        theta: Union[float, jnp.ndarray],
        phi: Union[float, jnp.ndarray],
        psi: Union[float, jnp.ndarray],
        t_gps: Union[float, jnp.ndarray] = 0.0,
        channel: str = "A",
    ) -> jnp.ndarray:
        """Project frequency-domain polarizations into a specific LISA TDI channel in JAX.

        h_channel(f) = F_plus * h_plus(f) + F_cross * h_cross(f)
        """
        f_plus, f_cross = self.antenna_patterns(theta, phi, psi, t_gps=t_gps, channel=channel)

        hp = jnp.asarray(h_plus)
        hc = jnp.asarray(h_cross)

        if hp.ndim > f_plus.ndim:
            f_plus = jnp.expand_dims(f_plus, -1)
            f_cross = jnp.expand_dims(f_cross, -1)

        return f_plus.astype(hp.dtype) * hp + f_cross.astype(hc.dtype) * hc

    def psd(
        self,
        f: Union[float, jnp.ndarray],
        channel: str = "A",
        include_galactic_confusion: bool = True,
        t_obs_years: float = 4.0,
    ) -> jnp.ndarray:
        """Evaluate analytical noise power spectral density for the specified channel."""
        return lisa_psd(
            f=f,
            channel=channel,
            include_galactic_confusion=include_galactic_confusion,
            t_obs_years=t_obs_years,
        )
