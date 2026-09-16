"""Relativistic EMRI (Extreme Mass Ratio Inspiral) Kludge Waveform Model for LISA in JAX.

Strict Clean-Room Implementation based exclusively on published literature:
- Barack & Cutler (2004), "LISA capture sources: Approximate waveforms, signal-to-noise ratios,
  and parameter estimation accuracy", Phys. Rev. D 69, 082005 (arXiv:gr-qc/0310125)
- Babak, Fang, Gair, Glampedakis, Hughes (2007), "'Kludge' gravitational waveforms for a test-body
  orbiting a Kerr black hole", Phys. Rev. D 75, 024005 (arXiv:gr-qc/0607007)
- Chua, Galley, Vallisneri (2021), "Fast and accurate gravitational-wave waveforms for extreme mass-ratio
  inspirals with deep neural networks", Phys. Rev. Lett. 126, 051102 (arXiv:2008.06071)
- Peters & Mathews (1963), "Gravitational Radiation from Point Masses in a Keplerian Orbit",
  Phys. Rev. 131, 435
"""

from typing import Dict, Optional, Sequence, Tuple, Union
import math
import jax
import jax.numpy as jnp

from diffgw_core.constants import (
    C_SI,
    G_SI,
    MSUN_SEC,
    MPC_SI,
    PI,
)
from jaxwave.conversions import as_array
from jaxwave.detectors.lisa import LISANetwork


def kerr_geodesic_frequencies(
    total_mass: Union[float, jnp.ndarray],
    spin: Union[float, jnp.ndarray],
    p: Union[float, jnp.ndarray],
    e: Union[float, jnp.ndarray] = 0.0,
    inclination: Union[float, jnp.ndarray] = 0.0,
    in_hz: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute Kerr orbital geodesic frequencies and precession rates in JAX.

    Literature: Barack & Cutler (2004) Eqs. 9-11; Babak et al. (2007) Eqs. 16-20.

    Orbital azimuthal frequency:
        Omega_phi = 1 / (M_sec * (p^(3/2) + a))

    Periastron precession rate:
        gamma_dot = Omega_phi - Omega_r = Omega_phi * [ 3/p - 4*a*cos(inc)/p^(3/2) + 9/(2*p^2) ]

    Lense-Thirring nodal precession rate:
        alpha_dot = Omega_phi - Omega_theta = Omega_phi * [ 2*a / p^(3/2) ]

    Args:
        total_mass: Central black hole mass M in solar masses.
        spin: Dimensionless Kerr spin parameter a = J / M^2 (-1 < a < 1).
        p: Dimensionless semi-latus rectum p = r_orb / M.
        e: Orbital eccentricity (0 <= e < 1).
        inclination: Orbital inclination iota in radians (default 0 for equatorial prograde).
        in_hz: If True, frequencies are returned in physical rad/s (or Hz if divided by 2pi).
            If False, returned in dimensionless geometric units (1/M).

    Returns:
        Tuple of (omega_phi, omega_r, omega_theta, gamma_dot, alpha_dot) arrays.
    """
    m = as_array(total_mass, dtype=jnp.float64)
    a = as_array(spin, dtype=m.dtype)
    p_t = as_array(p, dtype=m.dtype)
    e_t = as_array(e, dtype=m.dtype)
    inc = as_array(inclination, dtype=m.dtype)

    # Safe clamps
    p_safe = jnp.maximum(p_t, 1.0)
    e_sq = jnp.minimum(e_t * e_t, 0.999)
    cos_i = jnp.cos(inc)

    # Dimensionless orbital azimuthal frequency: Omega_phi = 1 / (p^(3/2) + a)
    p_32 = jnp.power(p_safe, 1.5)
    denom = p_32 + a
    denom_safe = jnp.where(jnp.abs(denom) < 1e-10, jnp.sign(denom) * 1e-10, denom)
    omega_phi_geom = 1.0 / denom_safe

    # Relativistic periastron advance rate: gamma_dot = Omega_phi - Omega_r
    # Barack & Cutler (2004) Eq. 10
    gamma_dot_factor = (3.0 / p_safe) - (4.0 * a * cos_i / p_32) + (4.5 / (p_safe * p_safe))
    gamma_dot_geom = omega_phi_geom * gamma_dot_factor
    omega_r_geom = omega_phi_geom - gamma_dot_geom

    # Relativistic Lense-Thirring nodal precession rate: alpha_dot = Omega_phi - Omega_theta
    # Barack & Cutler (2004) Eq. 11
    alpha_dot_factor = 2.0 * a / p_32
    alpha_dot_geom = omega_phi_geom * alpha_dot_factor
    omega_theta_geom = omega_phi_geom - alpha_dot_geom

    if isinstance(in_hz, bool):
        if in_hz:
            m_sec = m * MSUN_SEC
            m_sec_safe = jnp.maximum(m_sec, 1e-30)
            return (
                omega_phi_geom / m_sec_safe,
                omega_r_geom / m_sec_safe,
                omega_theta_geom / m_sec_safe,
                gamma_dot_geom / m_sec_safe,
                alpha_dot_geom / m_sec_safe,
            )
        else:
            return (
                omega_phi_geom,
                omega_r_geom,
                omega_theta_geom,
                gamma_dot_geom,
                alpha_dot_geom,
            )
    else:
        m_sec_safe = jnp.maximum(m * MSUN_SEC, 1e-30)
        scale = jnp.where(in_hz, m_sec_safe, 1.0)
        return (
            omega_phi_geom / scale,
            omega_r_geom / scale,
            omega_theta_geom / scale,
            gamma_dot_geom / scale,
            alpha_dot_geom / scale,
        )


def peters_mathews_radiation_reaction(
    total_mass: Union[float, jnp.ndarray],
    secondary_mass: Union[float, jnp.ndarray],
    p: Union[float, jnp.ndarray],
    e: Union[float, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Peters-Mathews gravitational radiation reaction evolution dp/dt and de/dt in JAX.

    Literature:
    - Peters & Mathews (1963), Phys. Rev. 131, 435 Eqs. 5.6 & 5.7
    - Barack & Cutler (2004), Phys. Rev. D 69, 082005 Eqs. 12 & 13

    Formulas:
        dp/dt = -64/5 * (mu / M^2) * p^(-3) * (1 - e^2)^(3/2) * (1 + 7/8 * e^2)
        de/dt = -604/48 * (mu / M^2) * e * p^(-4) * (1 - e^2)^(3/2) * (1 + 121/304 * e^2)

    Args:
        total_mass: Central black hole mass M in solar masses.
        secondary_mass: Compact secondary body mass mu in solar masses.
        p: Dimensionless semi-latus rectum p = r / M.
        e: Orbital eccentricity e in [0, 1).

    Returns:
        Tuple of (dp_dt, de_dt) in units of 1 / seconds.
    """
    m = as_array(total_mass, dtype=jnp.float64)
    mu = as_array(secondary_mass, dtype=m.dtype)
    p_t = as_array(p, dtype=m.dtype)
    e_t = as_array(e, dtype=m.dtype)

    p_safe = jnp.maximum(p_t, 1.0)
    e_safe = jnp.minimum(jnp.maximum(e_t, 0.0), 0.9999)

    e_sq = e_safe * e_safe
    one_minus_e2 = jnp.maximum(1.0 - e_sq, 1e-8)
    e_factor = jnp.power(one_minus_e2, 1.5)

    # Mass ratio scaling in physical seconds: mu / (M^2 * MSUN_SEC)
    mass_factor = mu / (m * m * MSUN_SEC)

    # dp/dt
    p_cubed = p_safe * p_safe * p_safe
    p_term = 1.0 + (7.0 / 8.0) * e_sq
    dp_dt = -(64.0 / 5.0) * mass_factor * (1.0 / p_cubed) * e_factor * p_term

    # de/dt
    p_fourth = p_cubed * p_safe
    e_term = 1.0 + (121.0 / 304.0) * e_sq
    de_dt = -(604.0 / 48.0) * mass_factor * e_safe * (1.0 / p_fourth) * e_factor * e_term

    return dp_dt, de_dt


@jax.tree_util.register_pytree_node_class
class EMRIKludge:
    """Relativistic Extreme Mass Ratio Inspiral (EMRI) Kludge Model for LISA in JAX.

    Strict clean-room literature implementation combining:
    - Kerr relativistic orbital geodesic frequencies (Barack & Cutler 2004)
    - Peters-Mathews radiation reaction dissipation (Peters & Mathews 1963)
    - Multipolar orbital harmonic emission summed across harmonics n in [1, N_harmonics]
    - Direct projection onto LISA TDI channels (A, E, T) via LISANetwork
    """

    def __init__(self, n_harmonics: int = 5):
        self.n_harmonics = int(n_harmonics)
        self.lisa_network = LISANetwork()

    def tree_flatten(self):
        return (self.lisa_network,), (self.n_harmonics,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.n_harmonics = aux_data[0]
        obj.lisa_network = children[0]
        return obj

    def orbital_trajectory(
        self,
        t: jnp.ndarray,
        total_mass: Union[float, jnp.ndarray],
        secondary_mass: Union[float, jnp.ndarray],
        spin: Union[float, jnp.ndarray],
        p0: Union[float, jnp.ndarray],
        e0: Union[float, jnp.ndarray] = 0.0,
        inclination: Union[float, jnp.ndarray] = 0.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Evolve the relativistic orbital trajectory over time array t.

        Computes p(t), e(t), Phi_phi(t), gamma_tilde(t), alpha(t).
        """
        t_t = as_array(t, dtype=jnp.float64)
        m = as_array(total_mass, dtype=t_t.dtype)
        mu = as_array(secondary_mass, dtype=t_t.dtype)
        a = as_array(spin, dtype=t_t.dtype)
        p_init = as_array(p0, dtype=t_t.dtype)
        e_init = as_array(e0, dtype=t_t.dtype)
        inc = as_array(inclination, dtype=t_t.dtype)

        # Initial radiation reaction rates
        dp_dt, de_dt = peters_mathews_radiation_reaction(m, mu, p_init, e_init)

        # Linear expansion for short intervals or Euler integration over t
        # p(t) = p0 + dp/dt * t, e(t) = e0 + de/dt * t
        p_t = jnp.maximum(p_init + dp_dt * t_t, 1.5)
        e_t = jnp.minimum(jnp.maximum(e_init + de_dt * t_t, 0.0), 0.999)

        # Kerr geodesic frequencies
        omega_phi, omega_r, omega_theta, gamma_dot, alpha_dot = kerr_geodesic_frequencies(
            m, a, p_t, e_t, inc, in_hz=True
        )

        # Integrated phase angles: cumulative trapezoid or integral approximation
        if t_t.ndim > 0 and t_t.shape[-1] > 1:
            dt = t_t[1] - t_t[0]
            phi_orb = jnp.cumsum(omega_phi * dt, axis=-1)
            gamma_orb = jnp.cumsum(gamma_dot * dt, axis=-1)
            alpha_orb = jnp.cumsum(alpha_dot * dt, axis=-1)
        else:
            phi_orb = omega_phi * t_t
            gamma_orb = gamma_dot * t_t
            alpha_orb = alpha_dot * t_t

        return p_t, e_t, phi_orb, gamma_orb, alpha_orb

    def polarizations(
        self,
        t: jnp.ndarray,
        total_mass: Union[float, jnp.ndarray] = 1.0e6,
        secondary_mass: Union[float, jnp.ndarray] = 10.0,
        spin: Union[float, jnp.ndarray] = 0.5,
        p0: Union[float, jnp.ndarray] = 8.0,
        e0: Union[float, jnp.ndarray] = 0.1,
        inclination: Union[float, jnp.ndarray] = 0.2,
        distance: Union[float, jnp.ndarray] = 1000.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate time-domain gravitational wave polarizations h_plus(t) and h_cross(t) in JAX.

        Decomposed into orbital harmonics n in [1, N_harmonics].

        Args:
            t: Time array in seconds.
            total_mass: Central SMBH mass M in solar masses (default 1e6).
            secondary_mass: Compact secondary mass mu in solar masses (default 10).
            spin: Dimensionless Kerr spin a in (-1, 1).
            p0: Initial semi-latus rectum (units of M).
            e0: Initial eccentricity.
            inclination: Orbital inclination in radians.
            distance: Luminosity distance in Mpc.

        Returns:
            Tuple of (h_plus, h_cross) arrays matching shape of t.
        """
        t_t = as_array(t, dtype=jnp.float64)
        m = as_array(total_mass, dtype=t_t.dtype)
        mu = as_array(secondary_mass, dtype=t_t.dtype)
        dist_mpc = as_array(distance, dtype=t_t.dtype)
        inc = as_array(inclination, dtype=t_t.dtype)

        # Distance in geometric seconds
        d_sec = (dist_mpc * MPC_SI) / C_SI
        mu_sec = mu * MSUN_SEC

        # Evolve trajectory
        p_t, e_t, phi_orb, gamma_orb, alpha_orb = self.orbital_trajectory(
            t_t, m, mu, spin, p0, e0, inc
        )

        # Characteristic quadrupole strain amplitude
        # h ~ (mu / D) * (M * Omega_phi)^(2/3) ~ (mu / D) / p
        amp0 = (mu_sec / d_sec) / jnp.maximum(p_t, 1.0)

        cos_i = jnp.cos(inc)
        plus_factor = 0.5 * (1.0 + cos_i * cos_i)
        cross_factor = cos_i

        h_plus = jnp.zeros_like(t_t)
        h_cross = jnp.zeros_like(t_t)

        # Multipolar harmonic sum n in [1, N]
        for n in range(1, self.n_harmonics + 1):
            if n == 2:
                # Dominant quadrupolar harmonic
                c_n = 1.0
            elif n == 1:
                c_n = e_t
            elif n == 3:
                c_n = 1.5 * e_t
            elif n == 4:
                c_n = 0.5 * (e_t * e_t)
            else:
                c_n = jnp.power(e_t, float(abs(n - 2)))

            phase_n = float(n) * phi_orb + 2.0 * gamma_orb

            h_plus = h_plus + c_n * amp0 * plus_factor * jnp.cos(phase_n)
            h_cross = h_cross + c_n * amp0 * cross_factor * jnp.sin(phase_n)

        return h_plus, h_cross

    def project_to_lisa(
        self,
        t: jnp.ndarray,
        theta: Union[float, jnp.ndarray],
        phi: Union[float, jnp.ndarray],
        psi: Union[float, jnp.ndarray],
        channels: Sequence[str] = ("A", "E", "T"),
        **emri_kwargs,
    ) -> Dict[str, jnp.ndarray]:
        """Generate EMRI signal projected into LISA TDI channels.

        Args:
            t: Time array in seconds.
            theta: Source ecliptic co-latitude in radians.
            phi: Source ecliptic longitude in radians.
            psi: Polarization angle in radians.
            channels: TDI channels to evaluate ('A', 'E', 'T').
            **emri_kwargs: Parameters passed to polarizations().

        Returns:
            Dictionary mapping channel name to time-domain strain response s(t).
        """
        hp, hc = self.polarizations(t, **emri_kwargs)
        responses: Dict[str, jnp.ndarray] = {}

        for ch in channels:
            fp, fc = self.lisa_network.antenna_patterns(
                theta=theta,
                phi=phi,
                psi=psi,
                channel=ch,
            )
            if hp.ndim > fp.ndim:
                fp_exp = jnp.expand_dims(fp, -1)
                fc_exp = jnp.expand_dims(fc, -1)
            else:
                fp_exp = fp
                fc_exp = fc
            # Response s(t) = F_plus * h_plus(t) + F_cross * h_cross(t)
            s_t = fp_exp * hp + fc_exp * hc
            responses[ch] = s_t

        return responses

    def forward(self, *args, **kwargs):
        if "theta" in kwargs and "phi" in kwargs:
            return self.project_to_lisa(*args, **kwargs)
        return self.polarizations(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
