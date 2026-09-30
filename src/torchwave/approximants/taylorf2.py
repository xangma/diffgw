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
    spin_combinations,
    tidal_combinations,
    kerr_isco_frequency,
)
from diffgw_core.windowing import torch_planck_taper


class TaylorF2:
    """Post-Newtonian Stationary Phase Approximation (SPA) frequency-domain model."""

    definition_id = "TaylorF2"

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
        tidal_order: int = 12,
        phenom_d_calibration: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Compute the 3.5PN point-particle, spin, and tidal phase coefficients.

        ``spin_order`` selects historical term groups, not a strict PN cutoff:
        3 adds 1.5PN spin-orbit, 4 adds 2PN spin-spin, 5 adds all remaining
        spin-orbit terms through 3.5PN, 6 adds 3PN spin-spin, and 8 adds
        3.5PN spin-spin tails and cubic spin. Negative values select the same
        spin terms as 7. ``phenom_d_calibration`` omits the additions at 6 and 8.
        See docs/math/taylorf2-aligned-spin.md for the explicit term map.
        ``quad_monopole1/2`` are full spin-induced quadrupole coefficients
        (Kerr=1). Their 2PN, 3PN, and 3.5PN corrections follow
        docs/math/taylorf2-spin-quadrupole.md; omitted inputs default to 1.

        ``tidal_order`` includes electric quadrupolar phase terms at powers
        10, 12 (default), and 13; negative values include all three. Both lambdas
        must be supplied; use explicit zero for a black-hole companion.
        See docs/math/taylorf2-tidal.md for equations and option semantics.

        Returns:
            Tuple of coefficients (phi0, phi2, phi3, phi4, phi5, phi5_log, phi6, phi6_log, phi7, phi10_tidal, phi12_tidal, phi13_tidal)
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
        # Spin-induced quadrupole corrections relative to the Kerr value C_Q=1.
        # Nagar et al. (2018), Eqs. (44), (46), (47); Xi=mi/(m1+m2).
        cq1 = 1.0 if quad_monopole1 is None else quad_monopole1
        cq2 = 1.0 if quad_monopole2 is None else quad_monopole2
        q1 = (m1_ratio * chi1) ** 2 * (cq1 - 1.0)
        q2 = (m2_ratio * chi2) ** 2 * (cq2 - 1.0)
        phi4_qm = -50.0 * (q1 + q2)

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

        # 3PN: Point-particle + Spin-Orbit (Khan et al. 2016 Eq. B5)
        # + Spin-Spin (Pratten et al. 2020, arXiv:2001.11412v1 Eq. A18).
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
        if spin_order >= 5 or spin_order < 0:
            phi6 = phi6 + phi6_so
        if not phenom_d_calibration and (spin_order >= 6 or spin_order < 0):
            phi6_qm = (15635.0 / 21.0 + 120.0 * eta) * (q1 + q2) + (2215.0 / 12.0) * delta * (q1 - q2)
            phi6 = phi6 + phi6_ss + phi6_qm
        phi6_log = -(6848.0 / 21.0) * torch.ones_like(eta)  # multiplier for ln(4v)

        # 3.5PN: Point-particle + Spin-Orbit (Khan et al. 2016 Eq. B6)
        # + Spin-Spin tails and Cubic-Spin (Pratten et al. 2020,
        # arXiv:2001.11412v1 Eqs. A19 + A24, including the explicit SS tail).
        phi7_pp = (
            (77096675.0 * PI / 254016.0)
            + (378515.0 * PI * eta / 1512.0)
            - (74045.0 * PI * eta2 / 756.0)
        )
        phi7_so = (
            delta * (-(25150083775.0 / 3048192.0) + (26804935.0 * eta / 6048.0) - (1985.0 * eta2 / 48.0)) * chi_a
            + (-(25150083775.0 / 3048192.0) + (10566655595.0 * eta / 762048.0) - (1042165.0 * eta2 / 3024.0) + (5345.0 * eta3 / 36.0)) * chi_s
        )
        phi7_ss = PI * (
            (30.0 * eta - 815.0 / 2.0) * (chi_s * chi_s)
            + (1600.0 * eta - 815.0 / 2.0) * (chi_a * chi_a)
            - 815.0 * delta * chi_s * chi_a
        )
        phi7_sss = (
            chi_s * (80.0 * eta2 - 7270.0 * eta + 14585.0 / 8.0) * (chi_a * chi_a)
            + (chi_a * chi_a * chi_a) * (14585.0 * delta / 24.0 - 2380.0 * delta * eta)
            + (chi_s * chi_s) * chi_a * (14585.0 * delta / 8.0 - 215.0 * delta * eta / 2.0)
            + (chi_s * chi_s * chi_s) * (100.0 * eta2 / 3.0 - 475.0 * eta / 6.0 + 14585.0 / 24.0)
        )
        phi7 = phi7_pp
        if spin_order >= 5 or spin_order < 0:
            phi7 = phi7 + phi7_so
        if not phenom_d_calibration and spin_order >= 8:
            # Dietrich et al. (2019), Eq. (27): C_Q-dependent cubic spin
            # alongside Nagar et al. (2018), Eq. (52), quadratic-spin tail.
            phi7_qm = -400.0 * PI * (q1 + q2) + 10.0 * (
                q1 * ((m1_ratio * m1_ratio + 308.0 * m1_ratio / 3.0) * chi1
                      + (m2_ratio * m2_ratio - 89.0 * m2_ratio / 3.0) * chi2)
                + q2 * ((m2_ratio * m2_ratio + 308.0 * m2_ratio / 3.0) * chi2
                        + (m1_ratio * m1_ratio - 89.0 * m1_ratio / 3.0) * chi1)
            )
            phi7 = phi7 + phi7_ss + phi7_sss + phi7_qm

        # Electric quadrupolar tides: Wade1402.5156v1 Eq. (4), and
        # Damour-Nagar-Villain1203.4352v1 Eqs. (B8)-(B12) for the 6.5PN tail.
        if lambda1 is not None and lambda2 is not None and (tidal_order >= 10 or tidal_order < 0):
            lambda_tilde, delta_lambda_tilde = tidal_combinations(mass1, mass2, lambda1, lambda2)
            phi10_tidal = -(39.0 / 2.0) * lambda_tilde
            if tidal_order >= 12 or tidal_order < 0:
                phi12_tidal = -(3115.0 / 64.0) * lambda_tilde + (6595.0 / 364.0) * delta * delta_lambda_tilde
            else:
                phi12_tidal = torch.zeros_like(eta)
            if tidal_order >= 13 or tidal_order < 0:
                phi13_tidal = (39.0 / 2.0) * PI * lambda_tilde
            else:
                phi13_tidal = torch.zeros_like(eta)
        else:
            phi10_tidal = torch.zeros_like(eta)
            phi12_tidal = torch.zeros_like(eta)
            phi13_tidal = torch.zeros_like(eta)

        return phi0, phi2, phi3, phi4, phi5, phi5_log, phi6, phi6_log, phi7, phi10_tidal, phi12_tidal, phi13_tidal

    @staticmethod
    def evaluate_phase_derivative(
        f: torch.Tensor,
        mass1: torch.Tensor,
        mass2: torch.Tensor,
        chi1: torch.Tensor,
        chi2: torch.Tensor,
        lambda1: Optional[torch.Tensor] = None,
        lambda2: Optional[torch.Tensor] = None,
        tc: torch.Tensor = 0.0,
        f_ref: Optional[Union[float, torch.Tensor]] = None,
        quad_monopole1: Optional[torch.Tensor] = None,
        quad_monopole2: Optional[torch.Tensor] = None,
        spin_order: int = 7,
        tidal_order: int = 12,
        phenom_d_calibration: bool = False,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute the analytical phase derivative dPsi/df in seconds with optional f_ref conditioning.

        dPsi/df = 2*pi*tc + (3 / (128 * eta)) * (d_pn_sum_dv) * (dv/df)
        """
        total_mass = mass1 + mass2
        eta = component_masses_to_eta(mass1, mass2)

        if f.dim() == 1 and total_mass.dim() >= 1:
            f = f.unsqueeze(0)

        f_safe = torch.clamp(f, min=1e-8)

        m_sec = total_mass * MSUN_SEC
        if m_sec.dim() >= 1 and f.dim() > m_sec.dim():
            m_sec = m_sec.unsqueeze(-1)
            eta = eta.unsqueeze(-1)

        v = (PI * m_sec * f_safe) ** (1.0 / 3.0)
        dv_df = (PI * m_sec / 3.0) * (PI * m_sec * f_safe) ** (-2.0 / 3.0)

        (
            phi0, phi2, phi3, phi4, phi5, phi5_log,
            phi6, phi6_log, phi7, phi10_t, phi12_t, phi13_t
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
            tidal_order=tidal_order,
            phenom_d_calibration=phenom_d_calibration,
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
        phi13_t = unsqueeze_if_needed(phi13_t)

        ln_4v = torch.log(4.0 * v)

        # d/dv [ pn_sum / v^5 ]
        d_pn_sum_dv = (
            -5.0 * phi0 * (v ** -6)
            - 3.0 * phi2 * (v ** -4)
            - 2.0 * phi3 * (v ** -3)
            - phi4 * (v ** -2)
            + phi5_log * (v ** -1)
            + (phi6 + phi6_log * ln_4v + phi6_log)
            + 2.0 * phi7 * v
            + 5.0 * phi10_t * (v ** 4)
            + 7.0 * phi12_t * (v ** 6)
            + 8.0 * phi13_t * (v ** 7)
        )

        dpsi_raw = (3.0 / (128.0 * eta)) * d_pn_sum_dv * dv_df
        t_c_t = unsqueeze_if_needed(as_tensor(tc, device=f.device, dtype=f.dtype))
        dpsi_df = 2.0 * PI * t_c_t + dpsi_raw
        return dpsi_df

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
        f_ref: Optional[Union[float, torch.Tensor]] = None,
        quad_monopole1: Optional[torch.Tensor] = None,
        quad_monopole2: Optional[torch.Tensor] = None,
        spin_order: int = 5,
        tidal_order: int = 12,
        phenom_d_calibration: bool = False,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute the full SPA phase Psi(f) with configurable phasing convention.

        Args:
            f: Shared frequency grid in Hz, shape (N_f,), or explicit (B, N_f)
                grids. Use (B, 1) for one frequency per binary.
            mass1, mass2: Component masses in solar masses, shape (B,) or ()
            chi1, chi2: Aligned dimensionless spins, shape (B,) or ()
            lambda1, lambda2: Dimensionless tidal deformabilities, shape (B,) or ()
            tc: Coalescence time in seconds
            phic: Coalescence phase in radians
            f_ref: Reference frequency in Hz
            quad_monopole1, quad_monopole2: Dimensionless spin-induced quadrupoles
                (Kerr=1; omitted inputs default to 1)
            spin_order: Historical term group; see phase_coefficients. Default 5
                omits 3PN spin-spin, unlike the public waveform's default 7.
            tidal_order: Highest PN tidal order (default 12 for 6PN, 13 for 6.5PN tail)
            phenom_d_calibration: If True, use Khan et al. 2016 Appendix B terms only
            phasing_convention: 'coalescence' or 'reference_frequency'
            reference_frame: Alias for phasing_convention

        Returns:
            Phase tensor Psi(f) with broadcast shape (B, N_f)
        """
        total_mass = mass1 + mass2
        eta = component_masses_to_eta(mass1, mass2)

        # A 1D grid is shared, even when N_f == B. Per-binary frequencies
        # must carry an explicit frequency axis, including the (B, 1) case.
        if f.dim() == 1 and total_mass.dim() >= 1:
            f = f.unsqueeze(0)

        # Total mass in seconds
        m_sec = total_mass * MSUN_SEC
        if m_sec.dim() >= 1 and f.dim() > m_sec.dim():
            m_sec = m_sec.unsqueeze(-1)
            eta = eta.unsqueeze(-1)

        (
            phi0, phi2, phi3, phi4, phi5, phi5_log,
            phi6, phi6_log, phi7, phi10_t, phi12_t, phi13_t
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
            tidal_order=tidal_order,
            phenom_d_calibration=phenom_d_calibration,
        )

        def unsqueeze_to_m_sec(t: torch.Tensor) -> torch.Tensor:
            if t.dim() >= 1 and m_sec.dim() > t.dim():
                return t.unsqueeze(-1)
            return t

        phi0 = unsqueeze_to_m_sec(phi0)
        phi2 = unsqueeze_to_m_sec(phi2)
        phi3 = unsqueeze_to_m_sec(phi3)
        phi4 = unsqueeze_to_m_sec(phi4)
        phi5 = unsqueeze_to_m_sec(phi5)
        phi5_log = unsqueeze_to_m_sec(phi5_log)
        phi6 = unsqueeze_to_m_sec(phi6)
        phi6_log = unsqueeze_to_m_sec(phi6_log)
        phi7 = unsqueeze_to_m_sec(phi7)
        phi10_t = unsqueeze_to_m_sec(phi10_t)
        phi12_t = unsqueeze_to_m_sec(phi12_t)
        phi13_t = unsqueeze_to_m_sec(phi13_t)
        LN_4 = 1.3862943611198906

        def _calc_raw(f_in: torch.Tensor) -> torch.Tensor:
            f_s = torch.clamp(f_in, min=1e-8)
            v_val = (PI * m_sec * f_s) ** (1.0 / 3.0)
            v2_v = v_val * v_val
            v3_v = v2_v * v_val
            v4_v = v2_v * v2_v
            v5_v = v4_v * v_val
            v6_v = v3_v * v3_v
            v7_v = v6_v * v_val
            ln_v_v = torch.log(v_val)
            pn_sum_v = (
                phi0
                + phi2 * v2_v
                + phi3 * v3_v
                + phi4 * v4_v
                + (phi5 + phi5_log * ln_v_v) * v5_v
                + (phi6 + phi6_log * (LN_4 + ln_v_v)) * v6_v
                + phi7 * v7_v
                + phi10_t * (v5_v * v5_v)
                + phi12_t * (v6_v * v6_v)
                + phi13_t * (v6_v * v7_v)
            )
            return -(PI / 4.0) + (3.0 / (128.0 * eta * v5_v)) * pn_sum_v

        # Raw SPA phasing on target grid
        psi_raw = _calc_raw(f)

        t_c_t = unsqueeze_to_m_sec(as_tensor(tc, device=f.device, dtype=f.dtype))
        phi_c_t = unsqueeze_to_m_sec(as_tensor(phic, device=f.device, dtype=f.dtype))

        conv = reference_frame if reference_frame is not None else (
            phasing_convention if phasing_convention is not None else ("reference_frequency" if f_ref is not None else "coalescence")
        )
        conv = conv.lower()

        if conv in ("reference_frequency", "reffreq", "ref"):
            ref_f = f_ref if f_ref is not None else (f.reshape(-1)[0] if f.numel() > 0 else 20.0)
            f_ref_t = as_tensor(ref_f, device=f.device, dtype=f.dtype)
            if f_ref_t.dim() == 0:
                f_ref_eval = f_ref_t.reshape(1, 1) if m_sec.dim() == 2 else f_ref_t.reshape(1)
            elif total_mass.dim() >= 1 and f_ref_t.dim() == 1 and f_ref_t.shape[0] == total_mass.shape[0]:
                f_ref_eval = f_ref_t.reshape(-1, 1)
            else:
                f_ref_eval = f_ref_t

            psi_ref = _calc_raw(f_ref_eval)

            # Transformed SPA phase: Psi(f) -> (Psi_raw(f) - Psi_raw(f_ref)) + 3*pi/4 + 2*pi*f*t_c - 2*phi_c
            psi = (psi_raw - psi_ref) + (3.0 * PI / 4.0) + 2.0 * PI * f * t_c_t - 2.0 * phi_c_t
        else:
            # Default 'coalescence': Psi(f) = 2*pi*f*tc - 2*phic + Psi_raw(f)
            psi = psi_raw + 2.0 * PI * f * t_c_t - 2.0 * phi_c_t

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
        f_eval_safe = torch.clamp(f_eval, min=1e-8)
        prefactor = torch.sqrt(5.0 * eta / 24.0) / (PI ** (2.0 / 3.0) * d_sec)
        a0 = prefactor * (m_sec ** (5.0 / 6.0))
        amp = a0 * (f_eval_safe ** (-7.0 / 6.0))
        amp = torch.where(f_eval > 0.0, amp, torch.zeros_like(amp))
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
        f_ref: Optional[Union[float, torch.Tensor]] = None,
        lambda1: Optional[Union[float, torch.Tensor]] = None,
        lambda2: Optional[Union[float, torch.Tensor]] = None,
        f_isco_cutoff: bool = True,
        isco_type: str = "schwarzschild",
        smooth_cutoff: bool = False,
        roll_off_delta: float = 0.1,
        spin_order: int = 7,
        tidal_order: int = 12,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        quad_monopole1: Optional[Union[float, torch.Tensor]] = None,
        quad_monopole2: Optional[Union[float, torch.Tensor]] = None,
        window: Optional[str] = None,
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
            f_ref: Reference frequency in Hz for physical reference phasing
            lambda1: Primary tidal deformability
            lambda2: Secondary tidal deformability
            quad_monopole1, quad_monopole2: Spin-induced quadrupole coefficients C_Qi (Kerr=1)
            f_isco_cutoff: If True, set signal to zero above ISCO
            isco_type: ISCO formulation ("schwarzschild" or "kerr"); the Kerr
                cutoff uses mass-weighted aligned spin (m1*chi1 + m2*chi2)/(m1+m2)
            smooth_cutoff: If True, smoothly roll off amplitude to zero at ISCO
            roll_off_delta: Fractional width of smooth roll-off window below ISCO
            window: Smooth window type ('planck' or 'cosine'). If 'planck', applies
                an infinitely differentiable Planck taper over [(1 - roll_off_delta) * f_isco, f_isco].
                If None (default), behavior is controlled by smooth_cutoff.
            spin_order: Historical term group; see phase_coefficients. Default 7
                includes spin-orbit through 3.5PN and spin-spin through 3PN;
                use 8 to also include 3.5PN spin-spin tails and cubic spin.
            tidal_order: Highest PN tidal order (default 12 for 6PN, 13 for 6.5PN tail)
            phasing_convention: Phasing convention ('coalescence' or 'reference_frequency')
            reference_frame: Alias for phasing_convention

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
        cq1 = as_tensor(quad_monopole1, device=device, dtype=dtype) if quad_monopole1 is not None else None
        cq2 = as_tensor(quad_monopole2, device=device, dtype=dtype) if quad_monopole2 is not None else None

        # Calculate phase and amplitude
        psi = TaylorF2.evaluate_phase(
            f=f,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            lambda1=l1,
            lambda2=l2,
            quad_monopole1=cq1,
            quad_monopole2=cq2,
            tc=tc,
            phic=phic,
            f_ref=f_ref,
            spin_order=spin_order,
            tidal_order=tidal_order,
            phasing_convention=phasing_convention,
            reference_frame=reference_frame,
        )
        amp = TaylorF2.evaluate_amplitude(
            f=f,
            mass1=m1,
            mass2=m2,
            distance=dist,
        )

        f_eval = f.unsqueeze(0) if (m1.dim() >= 1 and f.dim() == 1) else f

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

            use_planck = window is not None and str(window).lower() == "planck"
            use_smooth = smooth_cutoff or (window is not None)

            if not use_smooth:
                mask = (f_eval <= f_isco) & (f_eval > 0.0)
                amp = torch.where(mask, amp, torch.zeros_like(amp))
            elif use_planck:
                f_start = (1.0 - roll_off_delta) * f_isco
                w = torch_planck_taper(f_eval, f_start, f_isco)
                w = torch.where(f_eval > 0.0, w, torch.zeros_like(w))
                amp = amp * w
            else:
                f_start = (1.0 - roll_off_delta) * f_isco
                x = torch.clamp((f_eval - f_start) / (f_isco - f_start + 1e-30), 0.0, 1.0)
                w = 0.5 * (1.0 + torch.cos(PI * x))
                w = torch.where(f_eval <= f_start, torch.ones_like(w), w)
                w = torch.where(f_eval >= f_isco, torch.zeros_like(w), w)
                w = torch.where(f_eval > 0.0, w, torch.zeros_like(w))
                amp = amp * w

        # Inclination projections:
        # h_plus = 0.5 * (1 + cos^2(iota)) * A(f) * exp(-i * Psi(f))
        # h_cross = -i * cos(iota) * A(f) * exp(-i * Psi(f))
        cos_inc = torch.cos(inc)
        if cos_inc.dim() >= 1 and amp.dim() > cos_inc.dim():
            cos_inc = cos_inc.unsqueeze(-1)

        plus_factor = 0.5 * (1.0 + cos_inc * cos_inc)
        cross_factor = cos_inc

        # Real Cartesian polarizations
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin(psi)

        h_plus_re = plus_factor * amp * cos_psi
        h_plus_im = -plus_factor * amp * sin_psi

        h_cross_re = -cross_factor * amp * sin_psi
        h_cross_im = -cross_factor * amp * cos_psi

        # Enforce exact finite zeros for DC and non-positive frequency bins directly on real floats
        pos_mask = f_eval > 0.0
        zero = torch.zeros_like(h_plus_re)
        h_plus_re = torch.where(pos_mask, h_plus_re, zero)
        h_plus_im = torch.where(pos_mask, h_plus_im, zero)
        h_cross_re = torch.where(pos_mask, h_cross_re, zero)
        h_cross_im = torch.where(pos_mask, h_cross_im, zero)

        return (
            torch.view_as_complex(torch.stack([h_plus_re, h_plus_im], dim=-1)),
            torch.view_as_complex(torch.stack([h_cross_re, h_cross_im], dim=-1)),
        )

    evaluate = __call__
