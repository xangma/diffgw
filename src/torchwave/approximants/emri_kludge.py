"""Relativistic EMRI (Extreme Mass Ratio Inspiral) Kludge Waveform Model for LISA.

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
import torch
import torch.nn as nn

from torchwave.constants import (
    C_SI,
    G_SI,
    MSUN_SEC,
    MPC_SI,
    PI,
)
from torchwave.conversions import as_tensor
from torchwave.detectors.lisa import LISANetwork


def kerr_geodesic_frequencies(
    total_mass: Union[float, torch.Tensor],
    spin: Union[float, torch.Tensor],
    p: Union[float, torch.Tensor],
    e: Union[float, torch.Tensor] = 0.0,
    inclination: Union[float, torch.Tensor] = 0.0,
    in_hz: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Kerr orbital geodesic frequencies and precession rates.

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
        Tuple of (omega_phi, omega_r, omega_theta, gamma_dot, alpha_dot) tensors.
    """
    m = as_tensor(total_mass)
    device = m.device
    dtype = m.dtype

    a = as_tensor(spin, device=device, dtype=dtype)
    p_t = as_tensor(p, device=device, dtype=dtype)
    e_t = as_tensor(e, device=device, dtype=dtype)
    inc = as_tensor(inclination, device=device, dtype=dtype)

    # Safe clamps
    p_safe = torch.clamp(p_t, min=1.0)
    e_sq = torch.clamp(e_t * e_t, max=0.999)
    cos_i = torch.cos(inc)

    # Dimensionless orbital azimuthal frequency: Omega_phi = 1 / (p^(3/2) + a)
    p_32 = torch.pow(p_safe, 1.5)
    denom = p_32 + a
    denom_safe = torch.where(torch.abs(denom) < 1e-10, torch.sign(denom) * 1e-10, denom)
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

    if in_hz:
        m_sec = m * MSUN_SEC
        m_sec_safe = torch.clamp(m_sec, min=1e-30)
        omega_phi = omega_phi_geom / m_sec_safe
        omega_r = omega_r_geom / m_sec_safe
        omega_theta = omega_theta_geom / m_sec_safe
        gamma_dot = gamma_dot_geom / m_sec_safe
        alpha_dot = alpha_dot_geom / m_sec_safe
    else:
        omega_phi = omega_phi_geom
        omega_r = omega_r_geom
        omega_theta = omega_theta_geom
        gamma_dot = gamma_dot_geom
        alpha_dot = alpha_dot_geom

    return omega_phi, omega_r, omega_theta, gamma_dot, alpha_dot


def peters_mathews_radiation_reaction(
    total_mass: Union[float, torch.Tensor],
    secondary_mass: Union[float, torch.Tensor],
    p: Union[float, torch.Tensor],
    e: Union[float, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Peters-Mathews gravitational radiation reaction evolution dp/dt and de/dt.

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
    m = as_tensor(total_mass)
    device = m.device
    dtype = m.dtype

    mu = as_tensor(secondary_mass, device=device, dtype=dtype)
    p_t = as_tensor(p, device=device, dtype=dtype)
    e_t = as_tensor(e, device=device, dtype=dtype)

    p_safe = torch.clamp(p_t, min=1.0)
    e_safe = torch.clamp(e_t, min=0.0, max=0.9999)

    e_sq = e_safe * e_safe
    one_minus_e2 = torch.clamp(1.0 - e_sq, min=1e-8)
    e_factor = torch.pow(one_minus_e2, 1.5)

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


class EMRIKludge(nn.Module):
    """Relativistic Extreme Mass Ratio Inspiral (EMRI) Kludge Model for LISA.

    Strict clean-room literature implementation combining:
    - Kerr relativistic orbital geodesic frequencies (Barack & Cutler 2004)
    - Peters-Mathews radiation reaction dissipation (Peters & Mathews 1963)
    - Multipolar orbital harmonic emission summed across harmonics n in [-N, N]
    - Direct projection onto LISA TDI channels (A, E, T) via LISANetwork
    """

    def __init__(self, n_harmonics: int = 5):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.lisa_network = LISANetwork()

    def orbital_trajectory(
        self,
        t: torch.Tensor,
        total_mass: Union[float, torch.Tensor],
        secondary_mass: Union[float, torch.Tensor],
        spin: Union[float, torch.Tensor],
        p0: Union[float, torch.Tensor],
        e0: Union[float, torch.Tensor] = 0.0,
        inclination: Union[float, torch.Tensor] = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evolve the relativistic orbital trajectory over time array t.

        Computes p(t), e(t), Phi_phi(t), gamma_tilde(t), alpha(t).
        """
        device = t.device
        dtype = t.dtype if t.is_floating_point() else torch.float64

        m = as_tensor(total_mass, device=device, dtype=dtype)
        mu = as_tensor(secondary_mass, device=device, dtype=dtype)
        a = as_tensor(spin, device=device, dtype=dtype)
        p_init = as_tensor(p0, device=device, dtype=dtype)
        e_init = as_tensor(e0, device=device, dtype=dtype)
        inc = as_tensor(inclination, device=device, dtype=dtype)

        # Initial radiation reaction rates
        dp_dt, de_dt = peters_mathews_radiation_reaction(m, mu, p_init, e_init)

        # Linear expansion for short intervals or Euler integration over t
        # p(t) = p0 + dp/dt * t, e(t) = e0 + de/dt * t
        p_t = torch.clamp(p_init + dp_dt * t, min=1.5)
        e_t = torch.clamp(e_init + de_dt * t, min=0.0, max=0.999)

        # Kerr geodesic frequencies
        omega_phi, omega_r, omega_theta, gamma_dot, alpha_dot = kerr_geodesic_frequencies(
            m, a, p_t, e_t, inc, in_hz=True
        )

        # Integrated phase angles: cumulative trapezoid or integral approximation
        if t.dim() > 0 and len(t) > 1:
            dt = t[1] - t[0]
            phi_orb = torch.cumsum(omega_phi * dt, dim=-1)
            gamma_orb = torch.cumsum(gamma_dot * dt, dim=-1)
            alpha_orb = torch.cumsum(alpha_dot * dt, dim=-1)
        else:
            phi_orb = omega_phi * t
            gamma_orb = gamma_dot * t
            alpha_orb = alpha_dot * t

        return p_t, e_t, phi_orb, gamma_orb, alpha_orb

    def polarizations(
        self,
        t: torch.Tensor,
        total_mass: Union[float, torch.Tensor] = 1.0e6,
        secondary_mass: Union[float, torch.Tensor] = 10.0,
        spin: Union[float, torch.Tensor] = 0.5,
        p0: Union[float, torch.Tensor] = 8.0,
        e0: Union[float, torch.Tensor] = 0.1,
        inclination: Union[float, torch.Tensor] = 0.2,
        distance: Union[float, torch.Tensor] = 1000.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate time-domain gravitational wave polarizations h_plus(t) and h_cross(t).

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
            Tuple of (h_plus, h_cross) tensors matching shape of t.
        """
        device = t.device
        dtype = t.dtype if t.is_floating_point() else torch.float64

        m = as_tensor(total_mass, device=device, dtype=dtype)
        mu = as_tensor(secondary_mass, device=device, dtype=dtype)
        dist_mpc = as_tensor(distance, device=device, dtype=dtype)
        inc = as_tensor(inclination, device=device, dtype=dtype)

        # Distance in geometric seconds
        d_sec = (dist_mpc * MPC_SI) / C_SI
        mu_sec = mu * MSUN_SEC
        m_sec = m * MSUN_SEC

        # Evolve trajectory
        p_t, e_t, phi_orb, gamma_orb, alpha_orb = self.orbital_trajectory(
            t, m, mu, spin, p0, e0, inc
        )

        # Characteristic quadrupole strain amplitude
        # h ~ (mu / D) * (M * Omega_phi)^(2/3) ~ (mu / D) / p
        amp0 = (mu_sec / d_sec) / torch.clamp(p_t, min=1.0)

        cos_i = torch.cos(inc)
        sin_i = torch.sin(inc)
        plus_factor = 0.5 * (1.0 + cos_i * cos_i)
        cross_factor = cos_i

        h_plus = torch.zeros_like(t)
        h_cross = torch.zeros_like(t)

        # Multipolar harmonic sum n in [1, N]
        # Peters & Mathews (1963) harmonic amplitudes:
        # Dominant quadrupole n=2, eccentric sidebands n=1, 3, ...
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
                c_n = torch.pow(e_t, float(abs(n - 2)))

            phase_n = float(n) * phi_orb + 2.0 * gamma_orb

            h_plus = h_plus + c_n * amp0 * plus_factor * torch.cos(phase_n)
            h_cross = h_cross + c_n * amp0 * cross_factor * torch.sin(phase_n)

        return h_plus, h_cross

    def project_to_lisa(
        self,
        t: torch.Tensor,
        theta: Union[float, torch.Tensor],
        phi: Union[float, torch.Tensor],
        psi: Union[float, torch.Tensor],
        channels: Sequence[str] = ("A", "E", "T"),
        **emri_kwargs,
    ) -> Dict[str, torch.Tensor]:
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
        responses: Dict[str, torch.Tensor] = {}

        for ch in channels:
            fp, fc = self.lisa_network.antenna_patterns(
                theta=theta,
                phi=phi,
                psi=psi,
                channel=ch,
            )
            # Response s(t) = F_plus * h_plus(t) + F_cross * h_cross(t)
            s_t = fp * hp + fc * hc
            responses[ch] = s_t

        return responses

    def forward(self, *args, **kwargs):
        if "theta" in kwargs and "phi" in kwargs:
            return self.project_to_lisa(*args, **kwargs)
        return self.polarizations(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
