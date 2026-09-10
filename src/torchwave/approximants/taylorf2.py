"""TaylorF2 Frequency-Domain Gravitational Waveform Approximant.

Strict Clean-Room Implementation based exclusively on published literature:
- Blanchet (2014), Living Rev. Relativity 17, 2 (arXiv:1310.1528)
- Buonanno, Iyer, Ochsner, Pan, Sathyaprakash (2009), Phys. Rev. D 80, 084043 (arXiv:0907.0700)
- Arun, Buonanno, Faye, Ochsner (2009), Phys. Rev. D 79, 104023 (arXiv:0810.5336)
- Bohé, Marsat, Blanchet (2013), Class. Quantum Grav. 30, 135009 (arXiv:1303.7412)
- Flanagan & Hinderer (2008), Phys. Rev. D 77, 021502 (arXiv:0709.1915)
- Vines, Flanagan, Hinderer (2011), Phys. Rev. D 83, 084051 (arXiv:1101.1673)
- Wade et al. (2014), Phys. Rev. D 89, 103012 (arXiv:1402.5156)
- Bohé et al. (2015), Z. Angew. Math. Phys. 66, 1729 (arXiv:1501.01529)
- Marsat (2015), Class. Quantum Grav. 32, 085008 (arXiv:1411.4118)
- Pratten et al. (2020), Phys. Rev. D 102, 064001 (arXiv:2001.11412)
"""

from typing import Optional, Tuple, Union
import torch
import math

from torchwave.constants import (
    C_SI,
    MSUN_SEC,
    MPC_SI,
    EULER_GAMMA,
    PI,
)
from torchwave.conversions import (
    as_tensor,
    component_masses_to_eta,
    component_masses_to_delta,
    component_masses_to_chirp_mass,
    spin_combinations,
    tidal_combinations,
    kerr_isco_frequency,
)


class TaylorF2:
    """Post-Newtonian Stationary Phase Approximation (SPA) frequency-domain model."""

    @staticmethod
    def phase_coefficients(
        mass1: torch.Tensor,
        mass2: torch.Tensor,
        chi1: torch.Tensor,
        chi2: torch.Tensor,
        lambda1: Optional[torch.Tensor] = None,
        lambda2: Optional[torch.Tensor] = None,
        quad_monopole1: Optional[torch.Tensor] = None,
        quad_monopole2: Optional[torch.Tensor] = None,
        spin_order: int = 7,
    ) -> Tuple[torch.Tensor, ...]:
        """Compute the 3.5PN point-particle, spin, and tidal phase coefficients.

        Returns:
            Tuple of coefficients (phi0, phi2, phi3, phi4, phi5, phi5_log, phi6, phi6_log, phi7, phi10_tidal, phi12_tidal)
        """
        eta = component_masses_to_eta(mass1, mass2)
        delta = component_masses_to_delta(mass1, mass2)
        chi_s, chi_a, chi_pn, _ = spin_combinations(mass1, mass2, chi1, chi2)

        total_mass = mass1 + mass2
        m1_ratio = mass1 / total_mass
        m2_ratio = mass2 / total_mass

        eta2 = eta * eta
        eta3 = eta2 * eta

        # 0PN
        phi0 = torch.ones_like(eta)

        # 1PN
        phi2 = (3715.0 / 756.0) + (55.0 / 9.0) * eta

        # 1.5PN: Point-particle + Spin-Orbit (Khan et al. 2016 Eq. B2)
        phi3_pp = -16.0 * PI
        phi3_so = (113.0 / 3.0) * delta * chi_a + ((113.0 / 3.0) - (76.0 * eta / 3.0)) * chi_s
        phi3 = phi3_pp + (phi3_so if (spin_order >= 3 or spin_order < 0) else torch.zeros_like(eta))

        # 2PN: Point-particle + Spin-Spin (Khan et al. 2016 Eq. B3, Poisson 1998)
        phi4_pp = (15293365.0 / 508032.0) + (27145.0 / 504.0) * eta + (3085.0 / 72.0) * eta2
        phi4_ss = (
            (-405.0 / 8.0 + 200.0 * eta) * (chi_a * chi_a)
            - (405.0 / 4.0) * delta * chi_a * chi_s
            + (-405.0 / 8.0 + 5.0 * eta / 2.0) * (chi_s * chi_s)
        )
        # Optional quadrupole-monopole deformation for neutron stars (C_Q != 1)
        cq1 = 1.0 if quad_monopole1 is None else quad_monopole1
        cq2 = 1.0 if quad_monopole2 is None else quad_monopole2
        if cq1 != 1.0 or cq2 != 1.0:
            phi4_qm = -(50.0 / eta) * (
                (m1_ratio * m1_ratio) * (cq1 - 1.0) * (chi1 * chi1)
                + (m2_ratio * m2_ratio) * (cq2 - 1.0) * (chi2 * chi2)
            )
        else:
            phi4_qm = torch.zeros_like(eta)

        phi4 = phi4_pp
        if spin_order >= 4 or spin_order < 0:
            phi4 = phi4 + phi4_ss + phi4_qm

        # 2.5PN: Point-particle + Spin-Orbit (Khan et al. 2016 Eq. B4)
        # [1 + 3 ln(v)] * phi5_bracket
        phi5_pp = (38645.0 * PI / 756.0) - (65.0 * PI * eta / 9.0)
        phi5_so = (
            delta * (-(732985.0 / 2268.0) - (140.0 * eta / 9.0)) * chi_a
            + (-(732985.0 / 2268.0) + (24260.0 * eta / 81.0) + (340.0 * eta2 / 9.0)) * chi_s
        )
        phi5_bracket = phi5_pp + (phi5_so if (spin_order >= 5 or spin_order < 0) else torch.zeros_like(eta))
        phi5 = phi5_bracket
        phi5_log = 3.0 * phi5_bracket

        # 3PN: Point-particle + Spin-Orbit (Khan et al. 2016 Eq. B5) + Spin-Spin (Bohé et al. 2015 Eq. 4.13, Pratten et al. 2020 Eq. A13)
        phi6_pp_const = (
            (11583231236531.0 / 4694215680.0)
            - (6848.0 / 21.0) * EULER_GAMMA
            - (640.0 / 3.0) * (PI * PI)
            + ((-15737765635.0 / 3048192.0) + (2255.0 / 12.0) * (PI * PI)) * eta
            + (76055.0 / 1728.0) * eta2
            - (127825.0 / 1296.0) * eta3
        )
        phi6_so = (2270.0 * PI / 3.0) * delta * chi_a + ((2270.0 * PI / 3.0) - 520.0 * PI * eta) * chi_s
        phi6_ss = (
            chi_s * chi_a * (75515.0 * delta / 144.0 - 8225.0 * delta * eta / 18.0)
            + (chi_a * chi_a) * (-480.0 * eta2 - 263245.0 * eta / 252.0 + 75515.0 / 288.0)
            + (chi_s * chi_s) * (1255.0 * eta2 / 9.0 - 232415.0 * eta / 504.0 + 75515.0 / 288.0)
        )
        phi6 = phi6_pp_const
        if spin_order >= 6 or spin_order < 0:
            phi6 = phi6 + phi6_so + phi6_ss
        phi6_log = -(6848.0 / 21.0) * torch.ones_like(eta)  # multiplier for ln(4v)

        # 3.5PN: Point-particle + Spin-Orbit (Khan et al. 2016 Eq. B6) + Cubic-Spin / SSO (Marsat 2015 Eq. 6.22, Pratten et al. 2020 Eq. A14)
        phi7_pp = (
            (77096675.0 * PI / 254016.0)
            + (378515.0 * PI * eta / 1512.0)
            - (74045.0 * PI * eta2 / 756.0)
        )
        phi7_so = (
            delta * (-(25150083775.0 / 3048192.0) + (26804935.0 * eta / 6048.0) - (1985.0 * eta2 / 48.0)) * chi_a
            + (-(25150083775.0 / 3048192.0) + (10566655595.0 * eta / 762048.0) - (1042165.0 * eta2 / 3024.0) + (5345.0 * eta3 / 36.0)) * chi_s
        )
        phi7_sss = (
            chi_s * (-1140.0 * PI * delta * chi_a + (80.0 * eta2 - 7270.0 * eta + 14585.0 / 8.0) * (chi_a * chi_a))
            + (chi_a * chi_a * chi_a) * (14585.0 * delta / 24.0 - 2380.0 * delta * eta)
            + (chi_s * chi_s) * (chi_a * (14585.0 * delta / 8.0 - 215.0 * delta * eta / 2.0) + 40.0 * PI * eta - 570.0 * PI)
            + (chi_s * chi_s * chi_s) * (100.0 * eta2 / 3.0 - 475.0 * eta / 6.0 + 14585.0 / 24.0)
            + (2240.0 * PI * eta - 570.0 * PI) * (chi_a * chi_a)
        )
        phi7 = phi7_pp
        if spin_order >= 7 or spin_order < 0:
            phi7 = phi7 + phi7_so + phi7_sss

        # Tidal contributions (5PN and 6PN) - Wade et al. (2014, arXiv:1402.5156, Eq. 110)
        if lambda1 is not None and lambda2 is not None:
            lambda_tilde, delta_lambda_tilde = tidal_combinations(mass1, mass2, lambda1, lambda2)
            phi10_tidal = -(39.0 / 2.0) * lambda_tilde
            phi12_tidal = -(3115.0 / 64.0) * lambda_tilde + (6595.0 / 364.0) * delta * delta_lambda_tilde
        else:
            phi10_tidal = torch.zeros_like(eta)
            phi12_tidal = torch.zeros_like(eta)

        return phi0, phi2, phi3, phi4, phi5, phi5_log, phi6, phi6_log, phi7, phi10_tidal, phi12_tidal

    @staticmethod
    def evaluate_phase(
        f: torch.Tensor,
        mass1: torch.Tensor,
        mass2: torch.Tensor,
        chi1: torch.Tensor,
        chi2: torch.Tensor,
        lambda1: Optional[torch.Tensor] = None,
        lambda2: Optional[torch.Tensor] = None,
        tc: torch.Tensor = 0.0,
        phic: torch.Tensor = 0.0,
        quad_monopole1: Optional[torch.Tensor] = None,
        quad_monopole2: Optional[torch.Tensor] = None,
        spin_order: int = 7,
    ) -> torch.Tensor:
        """Compute the full SPA phase Psi(f).

        Args:
            f: Frequency tensor in Hz, shape (N_f,) or (B, N_f)
            mass1, mass2: Component masses in solar masses, shape (B,) or ()
            chi1, chi2: Aligned dimensionless spins, shape (B,) or ()
            lambda1, lambda2: Dimensionless tidal deformabilities, shape (B,) or ()
            tc: Coalescence time in seconds
            phic: Coalescence phase in radians
            quad_monopole1, quad_monopole2: Quadrupole-monopole parameters
            spin_order: Highest PN spin order (default 7 for 3.5PN spin, -1 for all)

        Returns:
            Phase tensor Psi(f) with broadcast shape (B, N_f)
        """
        total_mass = mass1 + mass2
        eta = component_masses_to_eta(mass1, mass2)

        # Broadcast parameters to allow batching
        if f.dim() == 1 and total_mass.dim() >= 1:
            f = f.unsqueeze(0)  # (1, N_f)
        for p in [total_mass, eta]:
            if p.dim() >= 1 and f.dim() > p.dim():
                pass

        # Total mass in seconds
        m_sec = total_mass * MSUN_SEC
        if m_sec.dim() >= 1 and f.dim() > m_sec.dim():
            m_sec = m_sec.unsqueeze(-1)
            eta = eta.unsqueeze(-1)

        # PN expansion parameter v = (pi * M * f)^(1/3)
        v = (PI * m_sec * f) ** (1.0 / 3.0)
        v2 = v * v
        v3 = v2 * v
        v4 = v2 * v2
        v5 = v4 * v
        v6 = v3 * v3
        v7 = v6 * v

        (
            phi0, phi2, phi3, phi4, phi5, phi5_log,
            phi6, phi6_log, phi7, phi10_t, phi12_t
        ) = TaylorF2.phase_coefficients(
            mass1=mass1,
            mass2=mass2,
            chi1=chi1,
            chi2=chi2,
            lambda1=lambda1,
            lambda2=lambda2,
            quad_monopole1=quad_monopole1,
            quad_monopole2=quad_monopole2,
            spin_order=spin_order,
        )

        def unsqueeze_if_needed(t: torch.Tensor) -> torch.Tensor:
            if t.dim() >= 1 and f.dim() > t.dim():
                return t.unsqueeze(-1)
            return t

        phi0 = unsqueeze_if_needed(phi0)
        phi2 = unsqueeze_if_needed(phi2)
        phi3 = unsqueeze_if_needed(phi3)
        phi4 = unsqueeze_if_needed(phi4)
        phi5 = unsqueeze_if_needed(phi5)
        phi5_log = unsqueeze_if_needed(phi5_log)
        phi6 = unsqueeze_if_needed(phi6)
        phi6_log = unsqueeze_if_needed(phi6_log)
        phi7 = unsqueeze_if_needed(phi7)
        phi10_t = unsqueeze_if_needed(phi10_t)
        phi12_t = unsqueeze_if_needed(phi12_t)

        ln_v = torch.log(v)
        ln_4v = torch.log(4.0 * v)

        # Point-particle & spin polynomial sum
        pn_sum = (
            phi0
            + phi2 * v2
            + phi3 * v3
            + phi4 * v4
            + (phi5 + phi5_log * ln_v) * v5
            + (phi6 + phi6_log * ln_4v) * v6
            + phi7 * v7
            + phi10_t * (v5 * v5)  # 5PN tidal (v^10)
            + phi12_t * (v6 * v6)  # 6PN tidal (v^12)
        )

        # Standard SPA phasing: Psi(f) = 2*pi*f*t_c - phi_c - pi/4 + 3/(128*eta*v^5) * pn_sum
        psi = (
            2.0 * PI * f * unsqueeze_if_needed(as_tensor(tc, device=f.device, dtype=f.dtype))
            - unsqueeze_if_needed(as_tensor(phic, device=f.device, dtype=f.dtype))
            - (PI / 4.0)
            + (3.0 / (128.0 * eta * v5)) * pn_sum
        )
        return psi

    @staticmethod
    def evaluate_amplitude(
        f: torch.Tensor,
        mass1: torch.Tensor,
        mass2: torch.Tensor,
        distance: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Newtonian leading-order amplitude A(f) = A0 * f^(-7/6).

        A0 = sqrt(5 * eta / 24) * M^(5/6) / (pi^(2/3) * D_L)
        """
        total_mass = mass1 + mass2
        eta = component_masses_to_eta(mass1, mass2)

        m_sec = total_mass * MSUN_SEC
        d_sec = (distance * MPC_SI) / C_SI

        is_batched = total_mass.dim() >= 1
        if f.dim() == 1 and is_batched:
            f_eval = f.unsqueeze(0)
            m_sec = m_sec.unsqueeze(-1)
            eta = eta.unsqueeze(-1)
            d_sec = d_sec.unsqueeze(-1)
        else:
            f_eval = f
            if m_sec.dim() >= 1 and f_eval.dim() > m_sec.dim():
                m_sec = m_sec.unsqueeze(-1)
                eta = eta.unsqueeze(-1)
                d_sec = d_sec.unsqueeze(-1)

        # Leading Newtonian amplitude
        prefactor = torch.sqrt(5.0 * eta / 24.0) / (PI ** (2.0 / 3.0) * d_sec)
        a0 = prefactor * (m_sec ** (5.0 / 6.0))
        amp = a0 * (f_eval ** (-7.0 / 6.0))
        return amp

    @staticmethod
    def __call__(
        f: torch.Tensor,
        mass1: Union[float, torch.Tensor],
        mass2: Union[float, torch.Tensor],
        chi1: Union[float, torch.Tensor] = 0.0,
        chi2: Union[float, torch.Tensor] = 0.0,
        distance: Union[float, torch.Tensor] = 100.0,
        inclination: Union[float, torch.Tensor] = 0.0,
        phic: Union[float, torch.Tensor] = 0.0,
        tc: Union[float, torch.Tensor] = 0.0,
        lambda1: Optional[Union[float, torch.Tensor]] = None,
        lambda2: Optional[Union[float, torch.Tensor]] = None,
        f_isco_cutoff: bool = True,
        isco_type: str = "schwarzschild",
        spin_order: int = 7,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate frequency-domain polarizations (h_plus, h_cross) using TaylorF2.

        Args:
            f: Frequency tensor in Hz
            mass1: Primary mass in solar masses (m1 >= m2)
            mass2: Secondary mass in solar masses
            chi1: Primary aligned spin
            chi2: Secondary aligned spin
            distance: Luminosity distance in Mpc
            inclination: Orbital inclination in radians
            phic: Coalescence phase in radians
            tc: Coalescence time in seconds
            lambda1: Primary tidal deformability
            lambda2: Secondary tidal deformability
            f_isco_cutoff: If True, set signal to zero above ISCO
            isco_type: ISCO formulation ("schwarzschild" or "kerr")
            spin_order: Highest PN spin order (default 7 for 3.5PN spin, -1 for all)

        Returns:
            (h_plus, h_cross) complex tensors
        """
        device = f.device
        dtype = f.dtype if f.is_floating_point() else torch.float64

        m1 = as_tensor(mass1, device=device, dtype=dtype)
        m2 = as_tensor(mass2, device=device, dtype=dtype)
        s1 = as_tensor(chi1, device=device, dtype=dtype)
        s2 = as_tensor(chi2, device=device, dtype=dtype)
        dist = as_tensor(distance, device=device, dtype=dtype)
        inc = as_tensor(inclination, device=device, dtype=dtype)

        l1 = as_tensor(lambda1, device=device, dtype=dtype) if lambda1 is not None else None
        l2 = as_tensor(lambda2, device=device, dtype=dtype) if lambda2 is not None else None

        # Calculate phase and amplitude
        psi = TaylorF2.evaluate_phase(
            f=f,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            lambda1=l1,
            lambda2=l2,
            tc=tc,
            phic=phic,
            spin_order=spin_order,
        )
        amp = TaylorF2.evaluate_amplitude(
            f=f,
            mass1=m1,
            mass2=m2,
            distance=dist,
        )

        # Cutoff above ISCO:
        # Schwarzschild: f_ISCO = 1 / (6^(3/2) * pi * M_sec)
        # Kerr: f_ISCO = 1 / (pi * M_sec * (r_ISCO^(3/2) + chi_eff)) (Bardeen et al. 1972)
        if f_isco_cutoff:
            total_m = m1 + m2
            m_sec = total_m * MSUN_SEC
            if m_sec.dim() >= 1 and amp.dim() > m_sec.dim():
                m_sec = m_sec.unsqueeze(-1)
                total_m_for_cutoff = total_m.unsqueeze(-1)
                chi_eff = ((m1 * s1 + m2 * s2) / total_m).unsqueeze(-1)
            else:
                total_m_for_cutoff = total_m
                chi_eff = (m1 * s1 + m2 * s2) / total_m

            if isco_type.lower() == "kerr":
                f_isco = kerr_isco_frequency(total_m_for_cutoff, chi_eff)
            else:
                f_isco = 1.0 / (6.0 ** 1.5 * PI * m_sec)

            f_eval = f.unsqueeze(0) if (m1.dim() >= 1 and f.dim() == 1) else f
            mask = (f_eval <= f_isco) & (f_eval > 0.0)
            amp = torch.where(mask, amp, torch.zeros_like(amp))

        # Inclination projections:
        # h_plus = 0.5 * (1 + cos^2(iota)) * A(f) * exp(-i * Psi(f))
        # h_cross = -i * cos(iota) * A(f) * exp(-i * Psi(f))
        cos_inc = torch.cos(inc)
        if cos_inc.dim() >= 1 and amp.dim() > cos_inc.dim():
            cos_inc = cos_inc.unsqueeze(-1)

        plus_factor = 0.5 * (1.0 + cos_inc * cos_inc)
        cross_factor = cos_inc

        # Complex exponential exp(-i * psi)
        exp_minus_i_psi = torch.complex(torch.cos(psi), -torch.sin(psi))

        h_plus = (plus_factor * amp).to(exp_minus_i_psi.dtype) * exp_minus_i_psi
        h_cross = torch.complex(
            -cross_factor * amp * torch.sin(psi),
            -cross_factor * amp * torch.cos(psi),
        )

        return h_plus, h_cross
