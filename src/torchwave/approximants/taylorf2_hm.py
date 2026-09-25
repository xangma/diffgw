"""TaylorF2_HM Multipolar Frequency-Domain Gravitational Waveform Approximant.

Strict Clean-Room Implementation based exclusively on published literature:
- Kidder (2008), Phys. Rev. D 77, 044016 (arXiv:0710.0614)
- Arun, Buonanno, Faye, Ochsner (2009), Phys. Rev. D 79, 104023 (arXiv:0810.5336)
- Blanchet (2014), Living Rev. Relativity 17, 2 (arXiv:1310.1528)
- García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
- Mezzasoma & Yunes (2022), Phys. Rev. D 106, 024026 (arXiv:2203.15934)
- Bardeen, Press, Teukolsky (1972), Astrophys. J. 178, 347
"""

from typing import Dict, Optional, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn

from torchwave.constants import MSUN_SEC, PI
from torchwave.conversions import (
    as_tensor,
    component_masses_to_eta,
    component_masses_to_delta,
    kerr_isco_frequency,
)
from torchwave.approximants.taylorf2 import TaylorF2


def _swsh_projection_coefficients(
    l: int,
    m: int,
    iota: torch.Tensor,
    phi0: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the polarization projection coefficients d_plus and d_cross.

    For positive frequencies, the contribution of modes (l, m) and (l, -m)
    to h_+ and h_x is given by García-Quirós et al. (2020),
    Phys. Rev. D 102, 064002, Eqs. (8)--(9):
        d_plus  = 0.5 * (Y_{l,-m} + (-1)^l * conj(Y_{l,m}))
        d_cross = 0.5i * (Y_{l,-m} - (-1)^l * conj(Y_{l,m}))
    where Y_{l,m} = {}_{-2}Y_{lm}(\\iota, \\phi_0).

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
        # {}_{-2}Y_{3,2} = sqrt(7/(64*pi)) (1+cos(iota))^2
        #                         * (3*cos(iota)-2) exp(2 i phi0).
        c_norm = math.sqrt(7.0 / (64.0 * math.pi))
        ylm = c_norm * (one_plus_cos * one_plus_cos) * (3.0 * cos_i - 2.0) * exp_pos
        yl_negm = c_norm * (one_minus_cos * one_minus_cos) * (3.0 * cos_i + 2.0) * exp_neg
        sign_l = -1.0
    elif l == 3 and m == 1:
        # {}_{-2}Y_{3,1} = sqrt(70/(256*pi)) sin(iota) (1+cos(iota))
        #                         * (3*cos(iota)-1) exp(i phi0).
        # The negative-m partner follows from the same standard Wigner-d
        # convention used by the lower multipoles above.
        c_norm = math.sqrt(70.0 / (256.0 * math.pi))
        ylm = c_norm * sin_i * one_plus_cos * (3.0 * cos_i - 1.0) * exp_pos
        yl_negm = c_norm * sin_i * one_minus_cos * (1.0 + 3.0 * cos_i) * exp_neg
        sign_l = -1.0
    elif l == 4 and m == 4:
        c_norm = math.sqrt(63.0 / (256.0 * math.pi))
        ylm = c_norm * (sin_i * sin_i) * (one_plus_cos * one_plus_cos) * exp_pos
        yl_negm = c_norm * (sin_i * sin_i) * (one_minus_cos * one_minus_cos) * exp_neg
        sign_l = 1.0
    elif l == 4 and m == 2:
        # {}_{-2}Y_{4,2} = 3/(8*sqrt(pi)) (1+cos(iota))^2
        #                         * (7*cos(iota)^2 - 7*cos(iota) + 1) exp(2 i phi0)
        # and its negative-m partner.  These are the standard s=-2
        # spin-weighted spherical harmonics in the same convention as the
        # lower multipoles above.
        c_norm = 3.0 / (8.0 * math.sqrt(math.pi))
        ylm = c_norm * (one_plus_cos * one_plus_cos) * (
            7.0 * cos_i * cos_i - 7.0 * cos_i + 1.0
        ) * exp_pos
        yl_negm = c_norm * (one_minus_cos * one_minus_cos) * (
            7.0 * cos_i * cos_i + 7.0 * cos_i + 1.0
        ) * exp_neg
        sign_l = 1.0
    elif l == 4 and m == 3:
        # {}_{-2}Y_{4,3} = 3*sqrt(14/(256*pi)) sin(iota) (1+cos(iota))^2
        #                         * (1-2*cos(iota)) exp(3 i phi0).
        c_norm = 3.0 * math.sqrt(14.0 / (256.0 * math.pi))
        ylm = c_norm * sin_i * (one_plus_cos * one_plus_cos) * (1.0 - 2.0 * cos_i) * exp_pos
        yl_negm = c_norm * sin_i * (one_minus_cos * one_minus_cos) * (1.0 + 2.0 * cos_i) * exp_neg
        sign_l = 1.0
    else:
        raise ValueError(f"Mode ({l}, {m}) not currently supported in TaylorF2_HM")

    ylm_star = torch.conj(ylm)
    d_plus = 0.5 * (yl_negm + sign_l * ylm_star)
    # d_cross = 0.5 * 1j * (yl_negm - sign_l * ylm_star)
    diff = yl_negm - sign_l * ylm_star
    d_cross = torch.complex(-0.5 * diff.imag, 0.5 * diff.real)

    return d_plus, d_cross


class TaylorF2_HM(nn.Module):
    """Multipolar Post-Newtonian Stationary Phase Approximation (SPA) frequency-domain model.

    Supports harmonics (l, m) in {(2, 2), (2, 1), (3, 1), (3, 2), (3, 3),
    (4, 2), (4, 3), (4, 4)}.
    """

    definition_id = "TaylorF2_HM"
    DEFAULT_MODES = ((2, 2), (2, 1), (3, 1), (3, 2), (3, 3), (4, 2), (4, 3), (4, 4))

    def __init__(self, modes: Sequence[Tuple[int, int]] = DEFAULT_MODES):
        super().__init__()
        normalized_modes = tuple((l, m) for l, m in modes)
        for l, m in normalized_modes:
            if (l, m) not in self.DEFAULT_MODES:
                raise ValueError(
                    f"Mode ({l}, {m}) is not supported. Supported modes: {self.DEFAULT_MODES}"
                )
        self.modes = normalized_modes

    def forward(
        self,
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
        phi0: Union[float, torch.Tensor] = 0.0,
        lambda1: Optional[Union[float, torch.Tensor]] = None,
        lambda2: Optional[Union[float, torch.Tensor]] = None,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        f_isco_cutoff: bool = True,
        isco_type: str = "schwarzschild",
        smooth_cutoff: bool = False,
        roll_off_delta: float = 0.1,
        spin_order: int = 7,
        return_modes: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        """Generate frequency-domain polarizations (h_plus, h_cross) including higher multipoles.

        Args:
            f: Frequency tensor in Hz
            mass1: Primary mass in solar masses (m1 >= m2)
            mass2: Secondary mass in solar masses
            chi1: Primary aligned dimensionless spin
            chi2: Secondary aligned dimensionless spin
            distance: Luminosity distance in Mpc
            inclination: Orbital inclination in radians
            phic: Coalescence phase in radians
            tc: Coalescence time in seconds
            phi0: Observer azimuthal orientation angle in radians (default 0.0)
            lambda1: Primary tidal deformability
            lambda2: Secondary tidal deformability
            modes: Optional sequence of (l, m) modes to compute
            f_isco_cutoff: If True, set signal to zero above harmonic ISCO frequency
            isco_type: ISCO formulation ("schwarzschild" or "kerr")
            spin_order: Highest PN spin order (default 7 for 3.5PN spin)
            return_modes: If True, also return dictionary of per-mode (h_plus, h_cross)

        Returns:
            (h_plus, h_cross) or (h_plus, h_cross, mode_dict)
        """
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

        l1 = as_tensor(lambda1, device=device, dtype=dtype) if lambda1 is not None else None
        l2 = as_tensor(lambda2, device=device, dtype=dtype) if lambda2 is not None else None

        active_modes = (
            tuple((l, m) for l, m in modes) if modes is not None else self.modes
        )
        for mode in active_modes:
            if mode not in self.DEFAULT_MODES:
                raise ValueError(
                    f"Mode {mode} is not supported. Supported modes: {self.DEFAULT_MODES}"
                )

        # Dimensions & broadcasting
        is_batched = m1.dim() >= 1
        if is_batched:
            f_eval = f.unsqueeze(0) if f.dim() == 1 else f
            inc_eval = inc.unsqueeze(-1) if inc.dim() < f_eval.dim() else inc
            phi0_eval = phi_0.unsqueeze(-1) if phi_0.dim() < f_eval.dim() else phi_0
        else:
            f_eval = f
            inc_eval = inc
            phi0_eval = phi_0

        total_m = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        delta = component_masses_to_delta(m1, m2)

        # Cutoff frequency at ISCO (quadrupole f_ISCO)
        if f_isco_cutoff:
            m_sec = total_m * MSUN_SEC
            if is_batched and f_eval.dim() > m_sec.dim():
                m_sec_cutoff = m_sec.unsqueeze(-1)
                total_m_cutoff = total_m.unsqueeze(-1)
                chi_eff_cutoff = ((m1 * s1 + m2 * s2) / total_m).unsqueeze(-1)
            else:
                m_sec_cutoff = m_sec
                total_m_cutoff = total_m
                chi_eff_cutoff = (m1 * s1 + m2 * s2) / total_m

            if isco_type.lower() == "kerr":
                f_isco_22 = kerr_isco_frequency(total_m_cutoff, chi_eff_cutoff)
            else:
                f_isco_22 = 1.0 / (6.0 ** 1.5 * PI * m_sec_cutoff)

        # Baseline Newtonian quadrupole amplitude A0(f)
        amp0 = TaylorF2.evaluate_amplitude(f=f, mass1=m1, mass2=m2, distance=dist)

        # Orbital velocity v22 = (pi * M * f)^(1/3)
        m_sec_b = total_m * MSUN_SEC
        if is_batched and f_eval.dim() > m_sec_b.dim():
            m_sec_b = m_sec_b.unsqueeze(-1)
            eta_b = eta.unsqueeze(-1)
            delta_b = delta.unsqueeze(-1)
        else:
            eta_b = eta
            delta_b = delta

        # Safe frequency for powers
        safe_f = torch.clamp(f_eval, min=1e-8)
        v22 = torch.pow(PI * m_sec_b * safe_f, 1.0 / 3.0)

        # Factor converting A0 into physical h_lm mode normalisation
        # sqrt(16 * pi / 5) * A0 corresponds to leading 22 mode
        mode_norm = math.sqrt(16.0 * math.pi / 5.0) * amp0

        h_plus_total = torch.zeros_like(amp0, dtype=torch.complex128 if dtype == torch.float64 else torch.complex64)
        h_cross_total = torch.zeros_like(h_plus_total)
        mode_dict = {}

        for l, m in active_modes:
            # Frequency scaling for SPA phase: F_22 = (2 / m) * f
            f_scaled = (2.0 / float(m)) * f_eval
            f_ref_scaled = (2.0 / float(m)) * as_tensor(f_ref, device=device, dtype=dtype) if f_ref is not None else None

            # Base TaylorF2 phase evaluated at f_scaled
            psi_base = TaylorF2.evaluate_phase(
                f=f_scaled,
                mass1=m1,
                mass2=m2,
                chi1=s1,
                chi2=s2,
                lambda1=l1,
                lambda2=l2,
                tc=t_c,
                phic=phi_c,
                f_ref=f_ref_scaled,
                spin_order=spin_order,
            )

            # Phase connection: Phi_lm(f) = (m / 2) * Phi_22(2f/m) + Delta_Phi_lm.
            # García-Quirós et al. (2020), Eqs. (29), (34), give the
            # 2f/m scaling and Delta_Phi_lm = 3*pi/4*(1-m/2).
            delta_phi = (3.0 * PI / 4.0) * (1.0 - float(m) / 2.0)
            psi_lm = (float(m) / 2.0) * psi_base + delta_phi

            # Mode amplitude ratio R_lm = A_lm / (sqrt(16*pi/5) * A0)
            # Leading ratios are derived in docs/math/taylorf2-hm.md from
            # Mezzasoma--Yunes (2022), Eqs. (20)--(21i), (23)--(30), with
            # Mishra et al. (2016), Eqs. (9)--(12h), as a leading-PN check.
            if (l, m) == (2, 2):
                r_lm = torch.ones_like(v22)
            elif (l, m) == (2, 1):
                # (delta / (3 * 2^(1/3))) * v22
                factor_21 = delta_b / (3.0 * (2.0 ** (1.0 / 3.0)))
                r_lm = factor_21 * v22
            elif (l, m) == (3, 1):
                # García-Quirós et al. (2020), Appendix FPN, Eq. (31amp):
                # H_31 = i*delta*v_1/(12*sqrt(7)).  Converting from the
                # m=1 SPA velocity v_1 to v22 and mode_norm gives
                # delta*v22 / (12*sqrt(7)*2^(5/6)).
                factor_31 = delta_b / (12.0 * math.sqrt(7.0) * (2.0 ** (5.0 / 6.0)))
                r_lm = factor_31 * v22
            elif (l, m) == (3, 3):
                # García-Quirós et al. (2020), Appendix E, Eq. (74), after
                # converting v_3=(2/3)^(1/3)*v22 and the common v^(-7/2)
                # SPA factor to the 22-mode normalization:
                # (3/4) * sqrt(5/7) * (3/2)^(5/6) * delta * v22.
                factor_33 = (
                    (3.0 / 4.0)
                    * math.sqrt(5.0 / 7.0)
                    * (3.0 / 2.0) ** (5.0 / 6.0)
                    * delta_b
                )
                r_lm = factor_33 * v22
            elif (l, m) == (3, 2):
                # Leading nonspinning term from García-Quirós et al. (2020),
                # Appendix E, Eq. (75): (1/3) sqrt(5/7) * (1 - 3*eta) * x,
                # with x=v22^2.  The paper's Eq. (71) normalization is the
                # same common SPA normalization used by mode_norm above.
                factor_32 = (1.0 / 3.0) * math.sqrt(5.0 / 7.0) * (1.0 - 3.0 * eta_b)
                r_lm = factor_32 * (v22 * v22)
            elif (l, m) == (4, 4):
                # (8 / 9) * sqrt(5 / 7) * (1 - 3*eta) * v22^2
                factor_44 = (8.0 / 9.0) * math.sqrt(5.0 / 7.0) * (1.0 - 3.0 * eta_b)
                r_lm = factor_44 * (v22 * v22)
            elif (l, m) == (4, 2):
                # Mezzasoma & Yunes (2022), Phys. Rev. D 106, 024026,
                # Eq. (2.21g): H_42 = sqrt(5)/63 * (1 - 3*eta) * x.
                # Here m=2, so x = v22^2, and the mode is in phase with
                # the quadrupole at this leading PN order.
                factor_42 = (math.sqrt(5.0) / 63.0) * (1.0 - 3.0 * eta_b)
                r_lm = factor_42 * (v22 * v22)
            elif (l, m) == (4, 3):
                # García-Quirós et al. (2020), Appendix FPN, Eq. (43amp):
                # H_43 = (3/4)i*sqrt(3/35)*delta*(2*eta-1)*v_3^3.
                # The existing real-amplitude phase convention factors out
                # the common -i phase, leaving the equivalent positive
                # delta*(1-2*eta) coefficient below.
                factor_43 = (
                    (3.0 / 4.0)
                    * math.sqrt(3.0 / 35.0)
                    * (3.0 / 2.0) ** (1.0 / 6.0)
                    * delta_b
                    * (1.0 - 2.0 * eta_b)
                )
                r_lm = factor_43 * (v22 * v22 * v22)

            amp_lm = mode_norm * r_lm

            # Apply mode-specific ISCO cutoff: f_cut,m = (m / 2) * f_ISCO_22
            if f_isco_cutoff:
                f_cut_m = (float(m) / 2.0) * f_isco_22
                if not smooth_cutoff:
                    mask = (f_eval <= f_cut_m) & (f_eval > 0.0)
                    amp_lm = torch.where(mask, amp_lm, torch.zeros_like(amp_lm))
                else:
                    f_start_m = (1.0 - roll_off_delta) * f_cut_m
                    x = torch.clamp((f_eval - f_start_m) / (f_cut_m - f_start_m + 1e-30), 0.0, 1.0)
                    w = 0.5 * (1.0 + torch.cos(PI * x))
                    w = torch.where(f_eval <= f_start_m, torch.ones_like(w), w)
                    w = torch.where(f_eval >= f_cut_m, torch.zeros_like(w), w)
                    w = torch.where(f_eval > 0.0, w, torch.zeros_like(w))
                    amp_lm = amp_lm * w

            # Complex mode waveform h_{l, -m}(f) = amp_lm * exp(-i * psi_lm)
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

        if return_modes:
            return h_plus_total, h_cross_total, mode_dict
        return h_plus_total, h_cross_total

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
