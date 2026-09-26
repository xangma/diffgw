"""Continuous Gravitational Waves (CW) from Triaxial Spinning Neutron Stars in JAX.

Strict Clean-Room Implementation based exclusively on published peer-reviewed literature:
- Jaranowski, Królak, Schutz (1998), Phys. Rev. D 58, 063001 (arXiv:gr-qc/9804014)
  [Data analysis of gravitational-wave signals from spinning neutron stars. I. The signal and its detection]
- Prix (2011), arXiv:1108.3546
  [The F-statistic and its generalizations in continuous gravitational-wave data analysis]
"""

from typing import Dict, Optional, Sequence, Tuple, Union
import math
import jax
import jax.numpy as jnp

from diffgw_core.constants import (
    C_SI,
    G_SI,
    PI,
)
from jaxwave.conversions import as_array
from jaxwave.detectors.network import (
    Detector,
    DetectorNetwork,
    greenwich_mean_sidereal_time,
)

# Standard astronomical constants for CW Doppler modeling
# Earth orbital radius (1 AU) in meters
AU_SI: float = 1.495978707e11
# Earth orbital frequency: Omega_Y = 2*pi / (1 sidereal year in seconds)
SIDEREAL_YEAR_SEC: float = 365.256363004 * 86400.0
OMEGA_ORBIT: float = 2.0 * math.pi / SIDEREAL_YEAR_SEC
# Obliquity of the ecliptic in radians (23.4392911 deg)
OBLIQUITY_RAD: float = 0.4090928042223289


def continuous_wave_h0(
    frequency: Union[float, jnp.ndarray],
    ellipticity: Union[float, jnp.ndarray],
    distance: Union[float, jnp.ndarray],
    i_zz: float = 1.0e38,
) -> jnp.ndarray:
    """Compute the continuous wave characteristic strain amplitude h_0 in JAX.

    Formula:
        h_0 = (4 * pi^2 * G / c^4) * (I_zz * epsilon * f_0^2 / d)
    Reference:
        Jaranowski, Królak, Schutz (1998) Eqs. (32)-(33), with f_0=2*f_rot

    Args:
        frequency: Continuous GW frequency f_0 in Hz (twice rotation frequency)
        ellipticity: Equatorial ellipticity epsilon = (I_xx - I_yy) / I_zz
        distance: Distance to source in meters
        i_zz: Principal moment of inertia in kg*m^2 (default: 1e38 kg*m^2)

    Returns:
        h_0: Dimensionless strain amplitude
    """
    f0 = as_array(frequency, dtype=jnp.float64)
    eps = as_array(ellipticity, dtype=f0.dtype)
    dist = as_array(distance, dtype=f0.dtype)

    prefactor = (4.0 * (PI ** 2) * G_SI) / (C_SI ** 4)
    h0 = prefactor * (i_zz * eps * (f0 ** 2)) / dist
    return h0


def detector_roemer_delay(
    t: jnp.ndarray,
    ra: Union[float, jnp.ndarray],
    dec: Union[float, jnp.ndarray],
    detector: Optional[Union[str, Detector]] = None,
    t_epoch: float = 0.0,
) -> jnp.ndarray:
    """Compute the total geometric Roemer delay tau = (r_det . n) / c in seconds in JAX.

    Accounts for both Earth orbital motion around the SSB and diurnal Earth rotation.

    Args:
        t: GPS time array in seconds
        ra: Right ascension in radians
        dec: Declination in radians
        detector: Detector object or prefix string (e.g. 'H1', 'L1', 'V1')
        t_epoch: Reference epoch in seconds for orbital phase

    Returns:
        tau: Roemer delay in seconds
    """
    t_t = as_array(t, dtype=jnp.float64)
    ra_t = as_array(ra, dtype=t_t.dtype)
    dec_t = as_array(dec, dtype=t_t.dtype)

    # Unit vector pointing to source in equatorial frame
    cos_dec = jnp.cos(dec_t)
    sin_dec = jnp.sin(dec_t)
    cos_ra = jnp.cos(ra_t)
    sin_ra = jnp.sin(ra_t)

    n_x = cos_dec * cos_ra
    n_y = cos_dec * sin_ra
    n_z = sin_dec

    # 1. Earth orbital motion around SSB
    # Ecliptic plane coordinates projected to equatorial frame
    orbit_phase = OMEGA_ORBIT * (t_t - t_epoch)
    cos_orb = jnp.cos(orbit_phase)
    sin_orb = jnp.sin(orbit_phase)
    cos_eps = math.cos(OBLIQUITY_RAD)
    sin_eps = math.sin(OBLIQUITY_RAD)

    r_orb_x = AU_SI * cos_orb
    r_orb_y = AU_SI * sin_orb * cos_eps
    r_orb_z = AU_SI * sin_orb * sin_eps

    delay_orbital = (r_orb_x * n_x + r_orb_y * n_y + r_orb_z * n_z) / C_SI

    # 2. Diurnal rotation delay from detector vertex
    delay_diurnal = jnp.zeros_like(t_t)
    if detector is not None:
        det_obj = Detector(detector) if isinstance(detector, str) else detector
        # Detector vertex in meters
        v_x, v_y, v_z = det_obj.vertex[0], det_obj.vertex[1], det_obj.vertex[2]
        gmst = greenwich_mean_sidereal_time(t_t)
        cos_g = jnp.cos(gmst)
        sin_g = jnp.sin(gmst)

        # Detector position in equatorial Cartesian coordinates
        r_spin_x = v_x * cos_g - v_y * sin_g
        r_spin_y = v_x * sin_g + v_y * cos_g
        r_spin_z = v_z

        delay_diurnal = (r_spin_x * n_x + r_spin_y * n_y + r_spin_z * n_z) / C_SI

    return delay_orbital + delay_diurnal


def continuous_wave_phase(
    t: jnp.ndarray,
    ra: Union[float, jnp.ndarray],
    dec: Union[float, jnp.ndarray],
    f0: Union[float, jnp.ndarray],
    f1: Union[float, jnp.ndarray] = 0.0,
    f2: Union[float, jnp.ndarray] = 0.0,
    phi0: Union[float, jnp.ndarray] = 0.0,
    t0: float = 0.0,
    detector: Optional[Union[str, Detector]] = None,
) -> jnp.ndarray:
    """Compute Doppler-modulated CW phase evolution phi(t) in JAX.

    Formula:
        phi(t) = phi_0 + 2*pi * sum_{k=0}^s [ f_k / (k+1)! * (t - t_0 + tau(t))^(k+1) ]
    Reference:
        Jaranowski, Królak, Schutz (1998) Eq. (10)
    """
    t_t = as_array(t, dtype=jnp.float64)
    f0_t = as_array(f0, dtype=t_t.dtype)
    f1_t = as_array(f1, dtype=t_t.dtype)
    f2_t = as_array(f2, dtype=t_t.dtype)
    phi0_t = as_array(phi0, dtype=t_t.dtype)

    tau = detector_roemer_delay(t_t, ra=ra, dec=dec, detector=detector, t_epoch=t0)
    t_bary = (t_t - t0) + tau

    phase = phi0_t + 2.0 * PI * (f0_t * t_bary + 0.5 * f1_t * (t_bary ** 2) + (1.0 / 6.0) * f2_t * (t_bary ** 3))
    return phase


def continuous_wave_amplitudes(
    h0: Union[float, jnp.ndarray],
    cos_iota: Union[float, jnp.ndarray],
    psi: Union[float, jnp.ndarray],
    phi0: Union[float, jnp.ndarray],
) -> jnp.ndarray:
    """Compute the four JKS quadrupolar amplitudes A^mu = (A_1, A_2, A_3, A_4) in JAX.

    Formulas:
        A_1 = h_0 * [ 0.5 * (1 + cos^2 iota) * cos(2*psi) * cos(phi_0) - cos(iota) * sin(2*psi) * sin(phi_0) ]
        A_2 = h_0 * [ 0.5 * (1 + cos^2 iota) * sin(2*psi) * cos(phi_0) + cos(iota) * cos(2*psi) * sin(phi_0) ]
        A_3 = -h_0 * [ 0.5 * (1 + cos^2 iota) * cos(2*psi) * sin(phi_0) + cos(iota) * sin(2*psi) * cos(phi_0) ]
        A_4 = -h_0 * [ 0.5 * (1 + cos^2 iota) * sin(2*psi) * sin(phi_0) - cos(iota) * cos(2*psi) * cos(phi_0) ]

    Reference:
        Jaranowski, Królak, Schutz (1998) Eq. (14)-(17); Prix (2011) Eq. (18)

    Returns:
        Array of shape (..., 4)
    """
    h0_t = as_array(h0, dtype=jnp.float64)
    ci_t = as_array(cos_iota, dtype=h0_t.dtype)
    psi_t = as_array(psi, dtype=h0_t.dtype)
    phi0_t = as_array(phi0, dtype=h0_t.dtype)

    plus_factor = 0.5 * (1.0 + ci_t * ci_t)
    cross_factor = ci_t

    cos_2psi = jnp.cos(2.0 * psi_t)
    sin_2psi = jnp.sin(2.0 * psi_t)
    cos_phi0 = jnp.cos(phi0_t)
    sin_phi0 = jnp.sin(phi0_t)

    a1 = h0_t * (plus_factor * cos_2psi * cos_phi0 - cross_factor * sin_2psi * sin_phi0)
    a2 = h0_t * (plus_factor * sin_2psi * cos_phi0 + cross_factor * cos_2psi * sin_phi0)
    a3 = -h0_t * (plus_factor * cos_2psi * sin_phi0 + cross_factor * sin_2psi * cos_phi0)
    a4 = -h0_t * (plus_factor * sin_2psi * sin_phi0 - cross_factor * cos_2psi * cos_phi0)

    return jnp.stack([a1, a2, a3, a4], axis=-1)


def continuous_wave_basis(
    t: jnp.ndarray,
    ra: Union[float, jnp.ndarray],
    dec: Union[float, jnp.ndarray],
    f0: Union[float, jnp.ndarray],
    f1: Union[float, jnp.ndarray] = 0.0,
    f2: Union[float, jnp.ndarray] = 0.0,
    t0: float = 0.0,
    detector: Optional[Union[str, Detector]] = None,
) -> jnp.ndarray:
    """Compute the four time-dependent basis functions h_mu(t) for mu in {1, 2, 3, 4} in JAX.

    Formulas:
        h_1(t) = a(t) * cos(Phi(t))
        h_2(t) = b(t) * cos(Phi(t))
        h_3(t) = a(t) * sin(Phi(t))
        h_4(t) = b(t) * sin(Phi(t))
    where a(t) = F_+(t; psi=0), b(t) = F_x(t; psi=0), and Phi(t) = phi(t) - phi_0.

    Reference:
        Jaranowski, Królak, Schutz (1998) Eq. (13); Prix (2011) Eq. (19)

    Returns:
        h_basis: Array of shape (..., 4, N_samples)
    """
    t_t = as_array(t, dtype=jnp.float64)

    # Antenna patterns at psi = 0
    det_obj = Detector(detector) if isinstance(detector, str) else (detector if detector is not None else Detector("H1"))
    a_t, b_t = det_obj.antenna_patterns(ra=ra, dec=dec, psi=0.0, t_gps=t_t)

    # Phase without initial offset phi_0
    phi_t = continuous_wave_phase(
        t=t_t,
        ra=ra,
        dec=dec,
        f0=f0,
        f1=f1,
        f2=f2,
        phi0=0.0,
        t0=t0,
        detector=detector,
    )

    cos_phi = jnp.cos(phi_t)
    sin_phi = jnp.sin(phi_t)

    h1 = a_t * cos_phi
    h2 = b_t * cos_phi
    h3 = a_t * sin_phi
    h4 = b_t * sin_phi

    return jnp.stack([h1, h2, h3, h4], axis=-2)


def continuous_wave_strain(
    t: jnp.ndarray,
    ra: Union[float, jnp.ndarray],
    dec: Union[float, jnp.ndarray],
    f0: Union[float, jnp.ndarray],
    h0: Optional[Union[float, jnp.ndarray]] = None,
    ellipticity: Optional[Union[float, jnp.ndarray]] = None,
    distance: Optional[Union[float, jnp.ndarray]] = None,
    cos_iota: Union[float, jnp.ndarray] = 1.0,
    psi: Union[float, jnp.ndarray] = 0.0,
    phi0: Union[float, jnp.ndarray] = 0.0,
    f1: Union[float, jnp.ndarray] = 0.0,
    f2: Union[float, jnp.ndarray] = 0.0,
    t0: float = 0.0,
    detector: Optional[Union[str, Detector]] = None,
    i_zz: float = 1.0e38,
) -> jnp.ndarray:
    """Compute the full detector strain h(t) for a continuous gravitational-wave signal in JAX.

    Can be evaluated directly via:
        h(t) = h_0 * [ 0.5 * (1 + cos^2 iota) * F_+(t) * cos(phi(t)) + cos(iota) * F_x(t) * sin(phi(t)) ]
    or decomposed into quadrupolar basis sum:
        h(t) = sum_{mu=1}^4 A^mu h_mu(t)

    Guaranteed property: when ellipticity == 0 (or h0 == 0), h(t) = 0.
    """
    t_t = as_array(t, dtype=jnp.float64)

    if h0 is None:
        if ellipticity is None or distance is None:
            raise ValueError("Must provide either h0 or both (ellipticity, distance).")
        h0_val = continuous_wave_h0(frequency=f0, ellipticity=ellipticity, distance=distance, i_zz=i_zz)
    else:
        h0_val = as_array(h0, dtype=t_t.dtype)

    # Four amplitudes A^mu, shape (..., 4)
    amp_vec = continuous_wave_amplitudes(
        h0=h0_val,
        cos_iota=cos_iota,
        psi=psi,
        phi0=phi0,
    )

    # Four basis functions h_mu(t), shape (..., 4, N)
    basis_vec = continuous_wave_basis(
        t=t_t,
        ra=ra,
        dec=dec,
        f0=f0,
        f1=f1,
        f2=f2,
        t0=t0,
        detector=detector,
    )

    # Contract over mu dimension: h(t) = sum_mu A^mu h_mu(t)
    amp_vec = jnp.expand_dims(amp_vec, -1)
    strain = jnp.sum(amp_vec * basis_vec, axis=-2)
    return strain


@jax.tree_util.register_pytree_node_class
class ContinuousWave:
    """Continuous Gravitational Wave signal generator in JAX."""

    def __init__(
        self,
        detector: Optional[Union[str, Detector]] = None,
        i_zz: float = 1.0e38,
    ):
        if isinstance(detector, str):
            self.detector = Detector(detector)
        else:
            self.detector = detector
        self.i_zz = i_zz

    def tree_flatten(self):
        children = (self.detector,) if self.detector is not None else ()
        aux_data = (self.detector is not None, self.i_zz)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        has_det, i_zz = aux_data
        det = children[0] if has_det else None
        return cls(detector=det, i_zz=i_zz)

    def forward(
        self,
        t: jnp.ndarray,
        ra: Union[float, jnp.ndarray],
        dec: Union[float, jnp.ndarray],
        f0: Union[float, jnp.ndarray],
        h0: Optional[Union[float, jnp.ndarray]] = None,
        ellipticity: Optional[Union[float, jnp.ndarray]] = None,
        distance: Optional[Union[float, jnp.ndarray]] = None,
        cos_iota: Union[float, jnp.ndarray] = 1.0,
        psi: Union[float, jnp.ndarray] = 0.0,
        phi0: Union[float, jnp.ndarray] = 0.0,
        f1: Union[float, jnp.ndarray] = 0.0,
        f2: Union[float, jnp.ndarray] = 0.0,
        t0: float = 0.0,
    ) -> jnp.ndarray:
        return continuous_wave_strain(
            t=t,
            ra=ra,
            dec=dec,
            f0=f0,
            h0=h0,
            ellipticity=ellipticity,
            distance=distance,
            cos_iota=cos_iota,
            psi=psi,
            phi0=phi0,
            f1=f1,
            f2=f2,
            t0=t0,
            detector=self.detector,
            i_zz=self.i_zz,
        )

    def __call__(self, *args, **kwargs) -> jnp.ndarray:
        return self.forward(*args, **kwargs)
