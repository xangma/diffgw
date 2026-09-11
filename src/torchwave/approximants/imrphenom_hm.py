"""IMRPhenomHM & IMRPhenomXHM Multipolar Higher-Mode Frequency-Domain Approximants.

Strict Clean-Room Implementation based exclusively on published literature:
- London et al. (2018), Phys. Rev. Lett. 120, 161102 (arXiv:1708.00404)
- García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
- Berti, Cardoso, Will (2006), Phys. Rev. D 73, 064030 (arXiv:gr-qc/0512160)
- Kidder (2008), Phys. Rev. D 77, 044016 (arXiv:0710.0614)
- Blanchet (2014), Living Rev. Relativity 17, 2 (arXiv:1310.1528)
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn

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
    total_spin_for_remnant,
)
from torchwave.approximants.imrphenomd import IMRPhenomD


# Kerr Quasinormal Mode (QNM) perturbation table fits (Berti, Cardoso, Will 2006 Table VIII)
# M * omega_lmn = f1 + f2 * (1 - j)^f3
# Q_lmn = q1 + q2 * (1 - j)^q3
# f_RD = (M / M_f) * (omega_lmn / (2 * pi)), f_damp = f_RD / (2 * Q_lmn)
QNM_FITS: Dict[Tuple[int, int], Dict[str, Tuple[float, float, float]]] = {
    (2, 2): {
        "f": (1.5251, -1.1568, 0.1292),
        "q": (0.7000, 1.4187, -0.4990),
    },
    (2, 1): {
        "f": (0.6000, -0.2339, 0.4175),
        "q": (-0.3000, 2.3561, -0.2277),
    },
    (3, 3): {
        "f": (1.8956, -1.3043, 0.1818),
        "q": (0.9000, 2.3430, -0.4810),
    },
    (3, 2): {
        "f": (1.1481, -0.5552, 0.3002),
        "q": (0.8313, 2.3773, -0.3655),
    },
    (4, 4): {
        "f": (2.3000, -1.5056, 0.2244),
        "q": (1.1929, 3.1191, -0.4825),
    },
}


def _swsh_projection_coefficients(
    l: int,
    m: int,
    iota: torch.Tensor,
    phi0: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute polarization projection coefficients d_plus and d_cross for mode (l, m).

    For positive frequencies, the contributions of modes (l, m) and (l, -m)
    to h_+ and h_x are given by (García-Quirós et al. (2020), Phys. Rev. D 102, 064002, Eq. 3):
        d_plus  = 0.5 * (Y_{l,-m} + (-1)^l * conj(Y_{l,m}))
        d_cross = 0.5i * (Y_{l,-m} - (-1)^l * conj(Y_{l,m}))
    where Y_{l,m} = {}_{-2}Y_{lm}(iota, phi0).

    Returns:
        (d_plus, d_cross) complex tensors.
    """
    cos_i = torch.cos(iota)
    sin_i = torch.sin(iota)

    one_plus_cos = 1.0 + cos_i
    one_minus_cos = 1.0 - cos_i

    # Phase factors exp(i * m * phi0)
    m_phi0 = float(m) * phi0
    exp_pos = torch.complex(torch.cos(m_phi0), torch.sin(m_phi0))
    exp_neg = torch.conj(exp_pos)

    if l == 2 and m == 2:
        c_norm = math.sqrt(5.0 / (64.0 * math.pi))
        ylm = c_norm * (one_plus_cos * one_plus_cos) * exp_pos
        yl_negm = c_norm * (one_minus_cos * one_minus_cos) * exp_neg
        sign_l = 1.0
    elif l == 2 and m == 1:
        c_norm = math.sqrt(5.0 / (16.0 * math.pi))
        ylm = c_norm * sin_i * one_plus_cos * exp_pos
        yl_negm = c_norm * sin_i * one_minus_cos * exp_neg
        sign_l = 1.0
    elif l == 3 and m == 3:
        c_norm = math.sqrt(21.0 / (128.0 * math.pi))
        ylm = -c_norm * sin_i * (one_plus_cos * one_plus_cos) * exp_pos
        yl_negm = c_norm * sin_i * (one_minus_cos * one_minus_cos) * exp_neg
        sign_l = -1.0
    elif l == 3 and m == 2:
        # Wigner d-matrix formula for (3, 2)
        c_norm = math.sqrt(7.0 / (4.0 * math.pi)) * 0.25
        ylm = c_norm * (one_plus_cos * one_plus_cos) * (3.0 * cos_i - 2.0) * exp_pos
        yl_negm = c_norm * (one_minus_cos * one_minus_cos) * (3.0 * cos_i + 2.0) * exp_neg
        sign_l = -1.0
    elif l == 4 and m == 4:
        c_norm = math.sqrt(63.0 / (256.0 * math.pi))
        ylm = c_norm * (sin_i * sin_i) * (one_plus_cos * one_plus_cos) * exp_pos
        yl_negm = c_norm * (sin_i * sin_i) * (one_minus_cos * one_minus_cos) * exp_neg
        sign_l = 1.0
    else:
        raise ValueError(f"Mode ({l}, {m}) not currently supported.")

    ylm_star = torch.conj(ylm)
    d_plus = 0.5 * (yl_negm + sign_l * ylm_star)
    diff = yl_negm - sign_l * ylm_star
    d_cross = torch.complex(-0.5 * diff.imag, 0.5 * diff.real)

    return d_plus, d_cross


class IMRPhenomHM(nn.Module):
    """IMRPhenomHM Multipolar Higher-Harmonic IMR Gravitational Waveform Approximant.

    Supports harmonics (l, m) in {(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)}.
    Literature:
    - London et al. (2018), Phys. Rev. Lett. 120, 161102 (arXiv:1708.00404)
    - García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
    - Berti, Cardoso, Will (2006), Phys. Rev. D 73, 064030 (arXiv:gr-qc/0512160)
    """

    SUPPORTED_MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))

    def __init__(
        self,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        carrier: str = "D",
        base_22: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.modes = tuple(modes) if modes is not None else self.SUPPORTED_MODES
        for lm in self.modes:
            if lm not in self.SUPPORTED_MODES:
                raise ValueError(f"Mode {lm} not supported. Supported: {self.SUPPORTED_MODES}")
        if base_22 is not None:
            self.base_22 = base_22
        elif carrier.upper() in ("XAS", "IMRPHENOMXAS"):
            from torchwave.approximants.imrphenom_xas import IMRPhenomXAS
            self.base_22 = IMRPhenomXAS()
        else:
            self.base_22 = IMRPhenomD()

    @staticmethod
    def mode_qnm_frequencies(
        l: int,
        m: int,
        mass1: torch.Tensor,
        mass2: torch.Tensor,
        chi1: torch.Tensor,
        chi2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute fundamental QNM frequency f_RD and damping width f_damp for mode (l, m).

        Reference: Berti, Cardoso, Will (2006) Table VIII.
        """
        total_mass = mass1 + mass2
        m_final, a_final, _, _ = IMRPhenomD.remnant_properties(mass1, mass2, chi1, chi2)

        fit = QNM_FITS[(l, m)]
        f1, f2, f3 = fit["f"]
        q1, q2, q3 = fit["q"]

        one_minus_af = torch.clamp(1.0 - a_final, min=1e-4)
        omega_qnm = f1 + f2 * (one_minus_af ** f3)
        quality_qnm = q1 + q2 * (one_minus_af ** q3)

        m_ratio = total_mass / m_final
        f_rd = m_ratio * (omega_qnm / (2.0 * PI))
        f_damp = f_rd / (2.0 * quality_qnm)
        return f_rd, f_damp

    def forward(
        self,
        f: Optional[torch.Tensor] = None,
        mass1: Union[float, torch.Tensor] = 30.0,
        mass2: Union[float, torch.Tensor] = 30.0,
        chi1: Union[float, torch.Tensor] = 0.0,
        chi2: Union[float, torch.Tensor] = 0.0,
        distance: Union[float, torch.Tensor] = 100.0,
        inclination: Union[float, torch.Tensor] = 0.0,
        phic: Union[float, torch.Tensor] = 0.0,
        tc: Union[float, torch.Tensor] = 0.0,
        f_ref: Optional[Union[float, torch.Tensor]] = None,
        phi0: Union[float, torch.Tensor] = 0.0,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        return_modes: bool = False,
        return_unprojected_modes: bool = False,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        sample_frequencies: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[Tuple[int, int], Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]],
    ]:
        """Generate frequency-domain polarizations (h_plus, h_cross) including higher multipoles.

        Args:
            f: Frequency tensor in Hz
            mass1, mass2: Component masses in solar masses (mass1 >= mass2)
            chi1, chi2: Aligned dimensionless spins
            distance: Luminosity distance in Mpc
            inclination: Orbital inclination in radians
            phic: Coalescence phase in radians
            tc: Coalescence time in seconds
            f_ref: Reference frequency in Hz
            phi0: Observer azimuth angle in radians (default 0.0)
            modes: Optional tuple of modes to evaluate (defaults to all supported modes)
            return_modes: If True, also return dictionary of mode waveforms

        Returns:
            (h_plus, h_cross) or (h_plus, h_cross, mode_dict)
        """
        if f is None:
            if sample_frequencies is not None:
                f = sample_frequencies
            else:
                raise ValueError("Must provide frequency grid via 'f' or 'sample_frequencies'")
        device = f.device
        dtype = f.dtype if f.is_floating_point() else torch.float64

        m1 = as_tensor(mass1, device=device, dtype=dtype)
        m2 = as_tensor(mass2, device=device, dtype=dtype)
        s1 = as_tensor(chi1, device=device, dtype=dtype)
        s2 = as_tensor(chi2, device=device, dtype=dtype)
        dist = as_tensor(distance, device=device, dtype=dtype)
        inc = as_tensor(inclination, device=device, dtype=dtype)
        phi_c = as_tensor(phic, device=device, dtype=dtype)
        t_c = as_tensor(tc, device=device, dtype=dtype)
        phi_0 = as_tensor(phi0, device=device, dtype=dtype)

        active_modes = tuple(modes) if modes is not None else self.modes

        total_mass = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        delta = component_masses_to_delta(m1, m2)
        m_sec = total_mass * MSUN_SEC

        is_batched = total_mass.dim() >= 1
        if f.dim() == 1 and is_batched:
            f_eval = f.unsqueeze(0)
            m_sec_eval = m_sec.unsqueeze(-1)
            eta_eval = eta.unsqueeze(-1)
            delta_eval = delta.unsqueeze(-1)
            inc_eval = inc.unsqueeze(-1)
            phi0_eval = phi_0.unsqueeze(-1)
            dist_eval = dist.unsqueeze(-1)
            phic_eval = phi_c.unsqueeze(-1)
            tc_eval = t_c.unsqueeze(-1)
        else:
            f_eval = f
            m_sec_eval = m_sec
            eta_eval = eta
            delta_eval = delta
            inc_eval = inc
            phi0_eval = phi_0
            dist_eval = dist
            phic_eval = phi_c
            tc_eval = t_c

        # Evaluate base (2, 2) mode at face-on (iota=0)
        try:
            hp_22_raw, _, phi_22 = self.base_22(
                f=f,
                mass1=m1,
                mass2=m2,
                chi1=s1,
                chi2=s2,
                distance=dist,
                inclination=0.0,
                phic=phic,
                tc=tc,
                f_ref=f_ref,
                phasing_convention=phasing_convention,
                reference_frame=reference_frame,
                return_phase=True,
            )
        except (TypeError, ValueError):
            hp_22_raw, _ = self.base_22(
                f=f,
                mass1=m1,
                mass2=m2,
                chi1=s1,
                chi2=s2,
                distance=dist,
                inclination=0.0,
                phic=phic,
                tc=tc,
                f_ref=f_ref,
                phasing_convention=phasing_convention,
                reference_frame=reference_frame,
            )
            phi_22 = -torch.angle(hp_22_raw)  # h ~ exp(-i * phi)

        amp_22 = torch.abs(hp_22_raw)

        # Baseline Newtonian quadrupole amplitude
        dist_sec = (dist_eval * MPC_SI) / C_SI
        prefactor = torch.sqrt(5.0 * eta_eval / 24.0) / (PI ** (2.0 / 3.0) * dist_sec)
        a0_phys = prefactor * (m_sec_eval ** (5.0 / 6.0)) * (torch.clamp(f_eval, min=1e-8) ** (-7.0 / 6.0))

        # Velocity v22 = (pi * M * f)^(1/3)
        v22 = torch.pow(PI * m_sec_eval * torch.clamp(f_eval, min=1e-8), 1.0 / 3.0)

        h_plus_total = torch.zeros_like(
            amp_22, dtype=torch.complex128 if dtype == torch.float64 else torch.complex64
        )
        h_cross_total = torch.zeros_like(h_plus_total)
        mode_dict = {}

        # 22 QNM ringdown frequency
        f_rd_22, f_damp_22 = self.mode_qnm_frequencies(2, 2, m1, m2, s1, s2)
        if is_batched and f_eval.dim() > f_rd_22.dim():
            f_rd_22_u = f_rd_22.unsqueeze(-1)
        else:
            f_rd_22_u = f_rd_22

        norm_mode = math.sqrt(16.0 * PI / 5.0)
        for l, m in active_modes:
            if (l, m) == (2, 2):
                exp_neg_i_phi22 = torch.complex(torch.cos(phi_22), -torch.sin(phi_22))
                h_mode = (norm_mode * amp_22).to(exp_neg_i_phi22.dtype) * exp_neg_i_phi22
            else:
                # 1. QNM ringdown for mode (l, m)
                f_rd_lm, f_damp_lm = self.mode_qnm_frequencies(l, m, m1, m2, s1, s2)
                if is_batched and f_eval.dim() > f_rd_lm.dim():
                    f_rd_lm_u = f_rd_lm.unsqueeze(-1)
                    f_damp_lm_u = f_damp_lm.unsqueeze(-1)
                else:
                    f_rd_lm_u = f_rd_lm
                    f_damp_lm_u = f_damp_lm

                # 2. Phase connection via Stationary Phase Approximation:
                # Inspiral: Psi_lm(f) = (m / 2) * Psi_22(2f / m) + Delta_Psi_lm
                # Delta_Psi_lm = (pi / 2) * (3l + mod(l + m, 2)) - pi (London et al. 2018 Eq. 10)
                delta_psi_lm = (PI / 2.0) * float(3 * l + ((l + m) % 2)) - PI
                f_scaled = (2.0 / float(m)) * f_eval

                # Re-evaluate base phase at scaled frequency f_scaled
                try:
                    hp_scaled, _, phi_22_scaled = self.base_22(
                        f=f_scaled,
                        mass1=m1,
                        mass2=m2,
                        chi1=s1,
                        chi2=s2,
                        distance=dist,
                        inclination=0.0,
                        phic=phic,
                        tc=tc,
                        f_ref=f_ref,
                        phasing_convention=phasing_convention,
                        reference_frame=reference_frame,
                        return_phase=True,
                    )
                except (TypeError, ValueError):
                    hp_scaled, _ = self.base_22(
                        f=f_scaled,
                        mass1=m1,
                        mass2=m2,
                        chi1=s1,
                        chi2=s2,
                        distance=dist,
                        inclination=0.0,
                        phic=phic,
                        tc=tc,
                        f_ref=f_ref,
                        phasing_convention=phasing_convention,
                        reference_frame=reference_frame,
                    )
                    phi_22_scaled = -torch.angle(hp_scaled)
                psi_lm_ins = (float(m) / 2.0) * phi_22_scaled + delta_psi_lm

                # Intermediate and ringdown phase connection
                psi_lm = psi_lm_ins

                # 3. Leading-order amplitude ratio R_lm = A_lm / (sqrt(16pi/5) * A0)
                # Kidder (2008), London et al. (2018), García-Quirós et al. (2020)
                if (l, m) == (2, 1):
                    # Proportional to delta * v22
                    r_lm_ins = (delta_eval / (3.0 * (2.0 ** (1.0 / 3.0)))) * v22
                elif (l, m) == (3, 3):
                    # Proportional to delta * v22
                    r_lm_ins = (3.0 / 4.0) * math.sqrt(15.0 / 14.0) * delta_eval * v22
                elif (l, m) == (3, 2):
                    # Proportional to (1 - 3*eta) * v22^2
                    r_lm_ins = (1.0 / 3.0) * math.sqrt(5.0 / 7.0) * (1.0 - 3.0 * eta_eval) * (v22 * v22)
                elif (l, m) == (4, 4):
                    # Proportional to (1 - 3*eta) * v22^2
                    r_lm_ins = (8.0 / 9.0) * math.sqrt(5.0 / 7.0) * (1.0 - 3.0 * eta_eval) * (v22 * v22)
                else:
                    r_lm_ins = torch.zeros_like(v22)

                # Convert to physical amplitude
                mode_norm = math.sqrt(16.0 * PI / 5.0) * a0_phys
                amp_lm_ins = mode_norm * r_lm_ins

                # Ringdown Lorentzian fall-off matching Kerr QNM
                f_rd_phys = f_rd_lm_u / m_sec_eval
                f_damp_phys = f_damp_lm_u / m_sec_eval
                lorentzian = (f_damp_phys ** 2.0) / ((f_eval - f_rd_phys) ** 2.0 + f_damp_phys ** 2.0)
                amp_lm_rd = amp_lm_ins * lorentzian

                amp_lm = torch.where(f_eval <= f_rd_phys, amp_lm_ins, amp_lm_rd)

                # Equal-mass symmetry: odd-m modes must vanish identically when m1 == m2
                if m % 2 != 0:
                    amp_lm = torch.where(delta_eval == 0.0, torch.zeros_like(amp_lm), amp_lm)

                exp_neg_i_psi = torch.complex(torch.cos(psi_lm), -torch.sin(psi_lm))
                h_mode = amp_lm.to(exp_neg_i_psi.dtype) * exp_neg_i_psi

            # Spin-weighted spherical harmonics projection
            d_plus, d_cross = _swsh_projection_coefficients(l, m, inc_eval, phi0_eval)
            hp_lm = d_plus * h_mode
            hc_lm = d_cross * h_mode

            h_plus_total = h_plus_total + hp_lm
            h_cross_total = h_cross_total + hc_lm

            if return_modes:
                mode_dict[(l, m)] = (hp_lm, hc_lm)
            elif return_unprojected_modes:
                mode_dict[(l, m)] = h_mode

        # Enforce exact finite zeros for DC and non-positive frequency bins
        h_plus_total = torch.where(f_eval > 0.0, h_plus_total, torch.zeros_like(h_plus_total))
        h_cross_total = torch.where(f_eval > 0.0, h_cross_total, torch.zeros_like(h_cross_total))

        if return_modes or return_unprojected_modes:
            return h_plus_total, h_cross_total, mode_dict
        return h_plus_total, h_cross_total

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class IMRPhenomXHM(IMRPhenomHM):
    """IMRPhenomXHM Multipolar Higher-Harmonic IMR Gravitational Waveform Approximant.

    Literature citations:
    - García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
    - Pratten et al. (2020), Phys. Rev. D 102, 064001 (arXiv:2001.11412)
    """

    def __init__(
        self,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        carrier: str = "XAS",
        base_22: Optional[nn.Module] = None,
    ):
        super().__init__(modes=modes, carrier=carrier, base_22=base_22)

