"""IMRPhenomP Frequency-Domain Gravitational Waveform Approximant.

Mathematical references for the inherited model and audited corrections:
- Hannam et al. (2014), Phys. Rev. Lett. 113, 151101 (arXiv:1308.3271)
- Schmidt et al. (2015), Phys. Rev. D 91, 024043 (arXiv:1408.1810)
- Pratten et al. (2021), Phys. Rev. D 103, 104056 (arXiv:2004.06503), Appendix B & D
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import math

from torchwave.constants import (
    C_SI,
    MSUN_SEC,
    MPC_SI,
    PI,
)
from torchwave.conversions import (
    as_tensor,
    component_masses_to_eta,
    component_masses_to_delta,
)
from torchwave.approximants.imrphenomd import IMRPhenomD


class IMRPhenomP:
    """IMRPhenomP precessing frequency-domain phenomenological waveform model.

    Constructs precessing waveforms by 'twisting up' the underlying aligned-spin
    IMRPhenomD carrier waveform with next-to-next-to-leading order (NNLO) Euler
    angles and a Wigner d-matrix rotation into the observer frame.
    """

    def __init__(self) -> None:
        self.phenom_d = IMRPhenomD()

    @staticmethod
    def nnlo_angle_coefficients(
        eta: torch.Tensor,
        delta: torch.Tensor,
        m1_norm: torch.Tensor,
        chip: torch.Tensor,
        chi_l: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Compute NNLO Euler angle polynomial coefficients.

        Equations (G7)–(G9) in Appendix G of Pratten et al. (2021), Phys. Rev. D 103, 104056 [arXiv:2004.06503].
        """
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta2 * eta2

        delta2 = delta * delta
        delta3 = delta2 * delta

        m1_2 = m1_norm * m1_norm
        m1_3 = m1_2 * m1_norm
        m1_4 = m1_2 * m1_2
        m1_5 = m1_4 * m1_norm
        m1_6 = m1_4 * m1_2
        m1_7 = m1_6 * m1_norm
        m1_8 = m1_4 * m1_4

        chip2 = chip * chip
        chip4 = chip2 * chip2

        chil2 = chi_l * chi_l

        # alpha coefficients
        a_minus3 = -(5.0 * delta / (64.0 * m1_norm)) - (35.0 / 192.0)
        a_minus2 = -(5.0 * m1_norm * chi_l * (3.0 * delta + 7.0 * m1_norm)) / (128.0 * eta)
        a_minus1 = (
            -(5.0 * (824.0 * eta + 1103.0) / 3072.0)
            - (15.0 * delta2 * eta / (256.0 * m1_2))
            - (15.0 * delta * m1_3 * chip2 / (128.0 * eta2))
            - (5.0 * delta * (980.0 * eta + 911.0) / (7168.0 * m1_norm))
            - (35.0 * m1_4 * chip2 / (128.0 * eta2))
        )
        a_1 = (
            5.0 * (36.0 * eta * (85568.0 * eta + 23817.0) + 8024297.0) / 9289728.0
            + 5.0 * m1_2 * (3.0 * delta2 * chip2 + 75.0 * delta2 * chil2 - 112.0 * PI * chi_l) / (256.0 * eta)
            - 15.0 * delta * m1_7 * (chip4 - 4.0 * chip2 * chil2) / (512.0 * eta4)
            + 5.0 * delta * m1_3 * ((812.0 * eta - 97.0) * chip2 + 20328.0 * eta * chil2) / (14336.0 * eta2)
            - 15.0 * PI * delta * m1_norm * chi_l / (16.0 * eta)
            - 35.0 * m1_8 * (chip4 - 4.0 * chip2 * chil2) / (512.0 * eta4)
            + m1_4 * (25.0 * (92.0 * eta + 19.0) * chip2 + 52640.0 * eta * chil2) / (6144.0 * eta2)
            + 15.0 * delta3 * eta2 / (1024.0 * m1_3)
            + 5.0 * delta2 * eta * (784.0 * eta + 323.0) / (28672.0 * m1_2)
            + 5.0 * delta * (504.0 * eta * (7630.0 * eta - 159.0) + 5579177.0) / (21676032.0 * m1_norm)
        )
        a_log = (
            -(5.0 / 48.0) * (7.0 * PI - 3.0 * delta2 * chi_l)
            - 5.0 * delta * m1_5 * chip2 * chi_l / (128.0 * eta3)
            + 5.0 * delta * (7168.0 * eta + 407.0) * m1_norm * chi_l / (21504.0 * eta)
            - 35.0 * m1_6 * chip2 * chi_l / (384.0 * eta3)
            + 5.0 * (4072.0 * eta + 599.0) * m1_2 * chi_l / (9216.0 * eta)
            - 5.0 * PI * delta / (16.0 * m1_norm)
        )

        # epsilon coefficients
        e_minus3 = a_minus3
        e_minus2 = a_minus2
        e_minus1 = (
            -(5.0 * (824.0 * eta + 1103.0) / 3072.0)
            - (15.0 * delta2 * eta / (256.0 * m1_2))
            - (5.0 * delta * (980.0 * eta + 911.0) / (7168.0 * m1_norm))
        )
        e_1 = (
            5.0 * (36.0 * eta * (85568.0 * eta + 23817.0) + 8024297.0) / 9289728.0
            + 5.0 * m1_2 * chi_l * (75.0 * delta2 * chi_l - 112.0 * PI) / (256.0 * eta)
            + 1815.0 * delta * m1_3 * chil2 / (256.0 * eta)
            - 15.0 * PI * delta * m1_norm * chi_l / (16.0 * eta)
            + 1645.0 * m1_4 * chil2 / (192.0 * eta)
            + 15.0 * delta3 * eta2 / (1024.0 * m1_3)
            + 5.0 * delta2 * eta * (784.0 * eta + 323.0) / (28672.0 * m1_2)
            + 5.0 * delta * (504.0 * eta * (7630.0 * eta - 159.0) + 5579177.0) / (21676032.0 * m1_norm)
        )
        e_log = (
            -(5.0 / 48.0) * (7.0 * PI - 3.0 * delta2 * chi_l)
            + 5.0 * delta * (7168.0 * eta + 407.0) * m1_norm * chi_l / (21504.0 * eta)
            + 5.0 * (4072.0 * eta + 599.0) * m1_2 * chi_l / (9216.0 * eta)
            - 5.0 * PI * delta / (16.0 * m1_norm)
        )

        return a_minus3, a_minus2, a_minus1, a_1, a_log, e_minus3, e_minus2, e_minus1, e_1, e_log

    @staticmethod
    def evaluate_nnlo_angles(
        v: torch.Tensor,
        v_ref: torch.Tensor,
        a_minus3: torch.Tensor,
        a_minus2: torch.Tensor,
        a_minus1: torch.Tensor,
        a_1: torch.Tensor,
        a_log: torch.Tensor,
        e_minus3: torch.Tensor,
        e_minus2: torch.Tensor,
        e_minus1: torch.Tensor,
        e_1: torch.Tensor,
        e_log: torch.Tensor,
        anchor_reference: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate NNLO primitives, optionally subtracting their reference values."""
        v_safe = torch.clamp(v if isinstance(v, torch.Tensor) else torch.tensor(v), min=1e-8)
        v_ref_safe = torch.clamp(v_ref if isinstance(v_ref, torch.Tensor) else torch.tensor(v_ref, device=v_safe.device, dtype=v_safe.dtype), min=1e-8)

        v_inv = 1.0 / v_safe
        v_inv2 = v_inv * v_inv
        v_inv3 = v_inv2 * v_inv
        log_v3 = torch.log(v_safe * v_safe * v_safe)

        v_ref_inv = 1.0 / v_ref_safe
        v_ref_inv2 = v_ref_inv * v_ref_inv
        v_ref_inv3 = v_ref_inv2 * v_ref_inv
        log_vref3 = torch.log(v_ref_safe * v_ref_safe * v_ref_safe)

        alpha = a_minus3 * v_inv3 + a_minus2 * v_inv2 + a_minus1 * v_inv + a_1 * v_safe + a_log * log_v3
        alpha_ref = a_minus3 * v_ref_inv3 + a_minus2 * v_ref_inv2 + a_minus1 * v_ref_inv + a_1 * v_ref_safe + a_log * log_vref3

        eps = e_minus3 * v_inv3 + e_minus2 * v_inv2 + e_minus1 * v_inv + e_1 * v_safe + e_log * log_v3
        eps_ref = e_minus3 * v_ref_inv3 + e_minus2 * v_ref_inv2 + e_minus1 * v_ref_inv + e_1 * v_ref_safe + e_log * log_vref3

        if anchor_reference:
            return alpha - alpha_ref, eps - eps_ref
        return alpha, eps

    @staticmethod
    def orbital_angular_momentum(v: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        """2PN non-spinning orbital angular momentum (Kidder 1995, Pratten et al. 2021)."""
        v_safe = torch.clamp(v, min=1e-8)
        x = v_safe * v_safe
        l1 = 1.5 + eta / 6.0
        l2 = (27.0 / 8.0) - (19.0 * eta / 8.0) + (eta * eta / 24.0)
        return (eta / v_safe) * (1.0 + l1 * x + l2 * (x * x))

    @staticmethod
    def pv2_opening_angle(
        orbital_angular_momentum: torch.Tensor,
        chip: torch.Tensor,
        m1_norm: torch.Tensor,
        spin_parallel: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Pv2 cone angle from the published physical spins.

        Pratten et al. (2021), Eqs. (4.3)--(4.4), use
        ``S_perp = m1**2 * chi_p`` and ``beta = atan2(S_perp, L+S_parallel)``.
        ``A1`` belongs only to the definition of ``chi_p``.
        """
        s_perp = chip * m1_norm.square()
        return torch.atan2(s_perp, orbital_angular_momentum + spin_parallel)

    @staticmethod
    def _safe_in_plane_norm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the exact Euclidean norm with a finite derivative at zero.

        The direction of a zero in-plane spin is undefined, but its magnitude
        is exactly zero.  Selecting a nonzero square-root branch at the origin
        avoids the ``0/0`` derivative of ``sqrt(x*x + y*y)`` without adding a
        physical floor to the cone angle.
        """
        r2 = x * x + y * y
        nonzero = r2 > 0
        r2_safe = torch.where(nonzero, r2, torch.ones_like(r2))
        return torch.where(nonzero, torch.sqrt(r2_safe), torch.zeros_like(r2))

    def __call__(
        self,
        f: torch.Tensor,
        mass1: Union[float, torch.Tensor],
        mass2: Union[float, torch.Tensor],
        spin1x: Union[float, torch.Tensor] = 0.0,
        spin1y: Union[float, torch.Tensor] = 0.0,
        spin1z: Union[float, torch.Tensor] = 0.0,
        spin2x: Union[float, torch.Tensor] = 0.0,
        spin2y: Union[float, torch.Tensor] = 0.0,
        spin2z: Union[float, torch.Tensor] = 0.0,
        distance: Union[float, torch.Tensor] = 100.0,
        inclination: Union[float, torch.Tensor] = 0.0,
        phic: Union[float, torch.Tensor] = 0.0,
        tc: Union[float, torch.Tensor] = 0.0,
        f_ref: Union[float, torch.Tensor] = 20.0,
        precession_version: str = "v2",
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate frequency-domain polarizations (h_plus, h_cross) using IMRPhenomP.

        Args:
            f: Frequency tensor in Hz, shape (N_f,) or (B, N_f)
            mass1: Primary mass in solar masses (m1 >= m2)
            mass2: Secondary mass in solar masses
            spin1x, spin1y, spin1z: Dimensionless spin vector of primary
            spin2x, spin2y, spin2z: Dimensionless spin vector of secondary
            distance: Luminosity distance in Mpc (default 100.0)
            inclination: Observer inclination in radians (default 0.0)
            phic: Coalescence phase in radians (default 0.0)
            tc: Coalescence time in seconds (default 0.0)
            f_ref: Reference frequency in Hz (default 20.0)
            precession_version: Legacy label; this implementation uses NNLO angles and 2PN L
            phasing_convention: Phasing convention ('coalescence' or 'reference_frequency')
            reference_frame: Alias for phasing_convention

        Returns:
            (h_plus, h_cross) complex tensors
        """
        device = f.device
        dtype = f.dtype if f.is_floating_point() else torch.float64

        m1 = as_tensor(mass1, device=device, dtype=dtype)
        m2 = as_tensor(mass2, device=device, dtype=dtype)
        s1x = as_tensor(spin1x, device=device, dtype=dtype)
        s1y = as_tensor(spin1y, device=device, dtype=dtype)
        s1z = as_tensor(spin1z, device=device, dtype=dtype)
        s2x = as_tensor(spin2x, device=device, dtype=dtype)
        s2y = as_tensor(spin2y, device=device, dtype=dtype)
        s2z = as_tensor(spin2z, device=device, dtype=dtype)

        dist = as_tensor(distance, device=device, dtype=dtype)
        inc = as_tensor(inclination, device=device, dtype=dtype)
        phi_c = as_tensor(phic, device=device, dtype=dtype)
        t_c = as_tensor(tc, device=device, dtype=dtype)
        fref = as_tensor(f_ref, device=device, dtype=dtype)

        # NNLO coefficients and chi_p identify body 1 as the heavier body.
        # Broadcast scalar masses with vector spin/extrinsic inputs first.
        (m1, m2, s1x, s1y, s1z, s2x, s2y, s2z,
         dist, inc, phi_c, t_c, fref) = torch.broadcast_tensors(
            m1, m2, s1x, s1y, s1z, s2x, s2y, s2z,
            dist, inc, phi_c, t_c, fref,
        )
        swap = m1 < m2
        m1, m2 = torch.where(swap, m2, m1), torch.where(swap, m1, m2)
        s1x, s2x = torch.where(swap, s2x, s1x), torch.where(swap, s1x, s2x)
        s1y, s2y = torch.where(swap, s2y, s1y), torch.where(swap, s1y, s2y)
        s1z, s2z = torch.where(swap, s2z, s1z), torch.where(swap, s1z, s2z)

        total_mass = m1 + m2
        m_sec = total_mass * MSUN_SEC
        eta = component_masses_to_eta(m1, m2)
        delta = component_masses_to_delta(m1, m2)

        m1_norm = 0.5 * (1.0 + delta)
        m2_norm = 0.5 * (1.0 - delta)

        q = m1 / m2
        a1_coeff = 2.0 + 1.5 / q
        a2_coeff = 2.0 + 1.5 * q

        # Effective precession parameter chi_p (Schmidt et al. 2015, Pratten et al. 2021)
        # The published definition uses the Euclidean in-plane norms.  Keep
        # the exact zero so the aligned-spin limit is an exact zero-cone case.
        s1_perp = (m1 * m1) * self._safe_in_plane_norm(s1x, s1y)
        s2_perp = (m2 * m2) * self._safe_in_plane_norm(s2x, s2y)
        s_p = torch.maximum(a1_coeff * s1_perp, a2_coeff * s2_perp)
        chip = s_p / (a1_coeff * m1 * m1)

        # In-plane spin initial orientation angle at f_ref: alpha(f_ref) = phi_s1
        s1_perp_sq = s1x * s1x + s1y * s1y
        s1_has_direction = s1_perp_sq > 1e-12
        s2_has_direction = s2x * s2x + s2y * s2y > 1e-12
        phi_s1_primary = torch.atan2(
            torch.where(s1_has_direction, s1y, torch.zeros_like(s1y)),
            torch.where(s1_has_direction, s1x, torch.ones_like(s1x)),
        )
        phi_s1_secondary = torch.atan2(
            torch.where(s2_has_direction, s2y, torch.zeros_like(s2y)),
            torch.where(s2_has_direction, s2x, torch.ones_like(s2x)),
        )
        phi_s1 = torch.where(s1_has_direction, phi_s1_primary, phi_s1_secondary)

        # Aligned spin components in coprecessing frame
        chi_l = m1_norm * s1z + m2_norm * s2z

        # Generate face-on co-precessing frame waveform using IMRPhenomD conditioned at f_ref
        h22, _ = self.phenom_d(
            f=f,
            mass1=m1,
            mass2=m2,
            chi1=s1z,
            chi2=s2z,
            distance=dist,
            inclination=torch.zeros_like(inc),
            phic=phi_c,
            tc=t_c,
            f_ref=fref,
            phasing_convention=phasing_convention,
            reference_frame=reference_frame,
        )

        # Broadcasting setup
        is_batched = total_mass.dim() >= 1
        if f.dim() == 1 and is_batched:
            f_eval = f.unsqueeze(0)
        else:
            f_eval = f

        def unsqueeze_param(t: torch.Tensor) -> torch.Tensor:
            if is_batched and f_eval.dim() > t.dim():
                return t.unsqueeze(-1)
            return t

        m_sec_u = unsqueeze_param(m_sec)
        eta_u = unsqueeze_param(eta)
        delta_u = unsqueeze_param(delta)
        m1n_u = unsqueeze_param(m1_norm)
        m2n_u = unsqueeze_param(m2_norm)
        chip_u = unsqueeze_param(chip)
        chil_u = unsqueeze_param(chi_l)
        s1z_u = unsqueeze_param(s1z)
        s2z_u = unsqueeze_param(s2z)
        inc_u = unsqueeze_param(inc)
        fref_u = unsqueeze_param(fref)
        phi_s1_u = unsqueeze_param(phi_s1)

        # Precession angle polynomial coefficients
        (
            a_m3, a_m2, a_m1, a_1, a_log,
            e_m3, e_m2, e_m1, e_1, e_log
        ) = IMRPhenomP.nnlo_angle_coefficients(
            eta=eta_u,
            delta=delta_u,
            m1_norm=m1n_u,
            chip=chip_u,
            chi_l=chil_u,
        )

        # Expansion parameter v = (pi * M_sec * f)^(1/3)
        v = (PI * m_sec_u * f_eval) ** (1.0 / 3.0)
        v_ref = (PI * m_sec_u * fref_u) ** (1.0 / 3.0)

        alpha_dyn, epsilon = IMRPhenomP.evaluate_nnlo_angles(
            v=v,
            v_ref=v_ref,
            a_minus3=a_m3, a_minus2=a_m2, a_minus1=a_m1, a_1=a_1, a_log=a_log,
            e_minus3=e_m3, e_minus2=e_m2, e_minus1=e_m1, e_1=e_1, e_log=e_log,
        )
        alpha = alpha_dyn - phi_s1_u

        # Orbital angular momentum L(v) & opening angle beta(v)
        ang_l = IMRPhenomP.orbital_angular_momentum(v, eta_u)
        # Pratten et al. (2021), Eqs. (4.3)--(4.4), define the physical
        # in-plane spin entering beta as S_perp = m1^2 * chi_p.  A1 is
        # already present in the definition of chi_p above and must not be
        # multiplied into this physical spin a second time.
        sp_geom = chip_u * (m1n_u * m1n_u)
        sl_geom = (m1n_u * m1n_u) * s1z_u + (m2n_u * m2n_u) * s2z_u

        l_plus_sl = ang_l + sl_geom
        # This is the same signed angle as Eq. (4.4), while preserving the
        # exact zero-cone limit and the anti-aligned geometric branch.
        beta = IMRPhenomP.pv2_opening_angle(ang_l, chip_u, m1n_u, sl_geom)

        # Wigner d^2 matrix elements d^2_{m, m'}(beta)
        cb = torch.cos(0.5 * beta)
        sb = torch.sin(0.5 * beta)
        cb2 = cb * cb
        sb2 = sb * sb
        sqrt6 = math.sqrt(6.0)

        # d^2_{m, -2}(beta)
        d_2_m2 = sb2 * sb2
        d_1_m2 = 2.0 * cb * (sb2 * sb)
        d_0_m2 = sqrt6 * cb2 * sb2
        d_m1_m2 = 2.0 * (cb2 * cb) * sb
        d_m2_m2 = cb2 * cb2

        # d^2_{m, 2}(beta)
        d_2_2 = cb2 * cb2
        d_1_2 = -2.0 * (cb2 * cb) * sb
        d_0_2 = sqrt6 * cb2 * sb2
        d_m1_2 = -2.0 * cb * (sb2 * sb)
        d_m2_2 = sb2 * sb2

        # Basis functions at inclination iota
        ci = torch.cos(0.5 * inc_u)
        si = torch.sin(0.5 * inc_u)
        ci2 = ci * ci
        si2 = si * si

        y_2 = si2 * si2
        y_1 = 2.0 * ci * (si2 * si)
        y_0 = sqrt6 * ci2 * si2
        y_m1 = 2.0 * (ci2 * ci) * si
        y_m2 = ci2 * ci2

        m_indices = [2, 1, 0, -1, -2]
        d_m2_list = [d_2_m2, d_1_m2, d_0_m2, d_m1_m2, d_m2_m2]
        d_p2_list = [d_2_2, d_1_2, d_0_2, d_m1_2, d_m2_2]
        y_list = [y_2, y_1, y_0, y_m1, y_m2]

        term_plus = torch.zeros_like(h22)
        term_cross = torch.zeros_like(h22)

        for m_val, dm2, dp2, y in zip(m_indices, d_m2_list, d_p2_list, y_list):
            m_alpha = m_val * alpha
            exp_minus_ima = torch.complex(torch.cos(-m_alpha), torch.sin(-m_alpha))
            exp_plus_ima = torch.complex(torch.cos(m_alpha), torch.sin(m_alpha))

            a_m_minus2 = exp_minus_ima * dm2 * y
            a_m_plus2_star = exp_plus_ima * dp2 * y

            term_plus = term_plus + (a_m_minus2 + a_m_plus2_star)
            term_cross = term_cross + (a_m_minus2 - a_m_plus2_star)

        exp_minus_2ie = torch.complex(torch.cos(-2.0 * epsilon), torch.sin(-2.0 * epsilon))

        # Polarizations
        h_plus = exp_minus_2ie * h22 * term_plus
        h_cross = -1j * exp_minus_2ie * h22 * term_cross

        return h_plus, h_cross
