"""IMRPhenomHM & IMRPhenomXHM Multipolar Higher-Mode Frequency-Domain Approximants.

Strict Clean-Room Implementation based exclusively on published literature:
- London et al. (2018), Phys. Rev. Lett. 120, 161102 (arXiv:1708.00404)
- Mishra et al. (2016), Phys. Rev. D 94, 084053 (arXiv:1609.00727)
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
    (4, 3): {
        "f": (1.6869, -0.8862, 0.2822),
        "q": (1.4812, 2.8096, -0.4271),
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
        c_norm = math.sqrt(7.0 / (4.0 * math.pi)) * 0.25
        ylm = c_norm * (one_plus_cos * one_plus_cos) * (3.0 * cos_i - 2.0) * exp_pos
        yl_negm = c_norm * (one_minus_cos * one_minus_cos) * (3.0 * cos_i + 2.0) * exp_neg
        sign_l = -1.0
    elif l == 4 and m == 4:
        c_norm = math.sqrt(63.0 / (256.0 * math.pi))
        ylm = c_norm * (sin_i * sin_i) * (one_plus_cos * one_plus_cos) * exp_pos
        yl_negm = c_norm * (sin_i * sin_i) * (one_minus_cos * one_minus_cos) * exp_neg
        sign_l = 1.0
    elif l == 4 and m == 3:
        c_norm = 3.0 * math.sqrt(14.0) / (16.0 * math.sqrt(math.pi))
        ylm = c_norm * sin_i * (one_plus_cos * one_plus_cos) * (1.0 - 2.0 * cos_i) * exp_pos
        yl_negm = c_norm * sin_i * (one_minus_cos * one_minus_cos) * (1.0 + 2.0 * cos_i) * exp_neg
        sign_l = 1.0
    else:
        raise ValueError(f"Mode ({l}, {m}) not currently supported.")

    ylm_star = torch.conj(ylm)
    d_plus = 0.5 * (yl_negm + sign_l * ylm_star)
    diff = yl_negm - sign_l * ylm_star
    d_cross = torch.complex(-0.5 * diff.imag, 0.5 * diff.real)

    return d_plus, d_cross


def frequency_mapping_amplitude(
    f: torch.Tensor,
    f0_amp: Union[float, torch.Tensor],
    f_rd_lm: Union[float, torch.Tensor],
    f_rd_22: Union[float, torch.Tensor],
    m: int,
) -> torch.Tensor:
    """London et al. (2018) piecewise amplitude frequency mapping f_22^A(f).

    Args:
        f: Frequency tensor
        f0_amp: Transition frequency f_0^A
        f_rd_lm: Ringdown frequency of (l, m) mode
        f_rd_22: Ringdown frequency of (2, 2) mode
        m: Azimuthal mode number
    """
    s2 = (f_rd_22 - (2.0 / float(m)) * f0_amp) / (f_rd_lm - f0_amp)
    reg1 = (2.0 / float(m)) * f
    reg2 = s2 * (f - f0_amp) + (2.0 / float(m)) * f0_amp
    reg3 = f - (f_rd_lm - f_rd_22)
    return torch.where(f <= f0_amp, reg1, torch.where(f <= f_rd_lm, reg2, reg3))


def frequency_mapping_phase(
    f: torch.Tensor,
    f0_phi: Union[float, torch.Tensor],
    f_rd_lm: Union[float, torch.Tensor],
    f_rd_22: Union[float, torch.Tensor],
    m: int,
) -> torch.Tensor:
    """London et al. (2018) piecewise phase frequency mapping f_22^phi(f).

    Args:
        f: Frequency tensor
        f0_phi: Transition frequency f_0^phi
        f_rd_lm: Ringdown frequency of (l, m) mode
        f_rd_22: Ringdown frequency of (2, 2) mode
        m: Azimuthal mode number
    """
    s2 = (f_rd_22 - (2.0 / float(m)) * f0_phi) / (f_rd_lm - f0_phi)
    s3 = f_rd_22 / f_rd_lm
    reg1 = (2.0 / float(m)) * f
    reg2 = s2 * (f - f0_phi) + (2.0 / float(m)) * f0_phi
    reg3 = s3 * f
    return torch.where(f <= f0_phi, reg1, torch.where(f <= f_rd_lm, reg2, reg3))


def h_hat_multipole(
    l: int,
    m: int,
    v_m: torch.Tensor,
    delta: torch.Tensor,
    eta: torch.Tensor,
    chi_s: torch.Tensor,
    chi_a: torch.Tensor,
) -> torch.Tensor:
    """Evaluate leading PN multipole amplitude ratio H_hat_lm(f).

    References:
    - Mishra et al. (2016), Phys. Rev. D 94, 084053
    - London et al. (2018), Phys. Rev. Lett. 120, 161102
    """
    if (l, m) == (2, 2):
        return -torch.ones_like(v_m)
    elif (l, m) == (3, 3):
        c = -0.75 * math.sqrt(5.0 / 7.0)
        return c * delta * v_m
    elif (l, m) == (4, 4):
        c = -(4.0 / 9.0) * math.sqrt(10.0 / 7.0)
        return c * (1.0 - 3.0 * eta) * (v_m ** 2.0)
    elif (l, m) == (2, 1):
        c = -math.sqrt(2.0) / 3.0
        return c * (delta * v_m - 1.5 * (chi_a + delta * chi_s) * (v_m ** 2.0))
    elif (l, m) == (3, 2):
        c = -(1.0 / 3.0) * math.sqrt(5.0 / 7.0)
        return c * ((1.0 - 3.0 * eta) * (v_m ** 2.0) + 4.0 * eta * chi_s * (v_m ** 3.0))
    elif (l, m) == (4, 3):
        c = -0.75 * math.sqrt(3.0 / 35.0)
        return c * (delta * (1.0 - 2.0 * eta) * (v_m ** 3.0) + 2.5 * eta * (chi_a - delta * chi_s) * (v_m ** 4.0))
    else:
        raise ValueError(f"Mode ({l}, {m}) not supported.")


class IMRPhenomHM(nn.Module):
    """IMRPhenomHM Multipolar Higher-Harmonic IMR Gravitational Waveform Approximant.

    Supports harmonics (l, m) in {(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)}.
    Literature:
    - London et al. (2018), Phys. Rev. Lett. 120, 161102 (arXiv:1708.00404)
    - Mishra et al. (2016), Phys. Rev. D 94, 084053 (arXiv:1609.00727)
    - Berti, Cardoso, Will (2006), Phys. Rev. D 73, 064030 (arXiv:gr-qc/0512160)
    """

    SUPPORTED_MODES: Tuple[Tuple[int, int], ...] = (
        (2, 2),
        (2, 1),
        (3, 3),
        (3, 2),
        (4, 4),
        (4, 3),
    )

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
        Returns:
            f_rd, f_damp in dimensionless geometric units (M * f).
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
            return_unprojected_modes: If True, also return dictionary of unprojected h_mode waveforms

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
        chi_s = 0.5 * (s1 + s2)
        chi_a = 0.5 * (s1 - s2)
        m_sec = total_mass * MSUN_SEC

        is_batched = total_mass.dim() >= 1
        if is_batched:
            if f.dim() == 1:
                f_eval = f.unsqueeze(0)
            else:
                f_eval = f

            def _unsqueeze(t: torch.Tensor) -> torch.Tensor:
                if t.dim() < f_eval.dim():
                    return t.unsqueeze(-1)
                return t

            m_sec_eval = _unsqueeze(m_sec)
            eta_eval = _unsqueeze(eta)
            delta_eval = _unsqueeze(delta)
            chi_s_eval = _unsqueeze(chi_s)
            chi_a_eval = _unsqueeze(chi_a)
            inc_eval = _unsqueeze(inc)
            phi0_eval = _unsqueeze(phi_0)
            dist_eval = _unsqueeze(dist)
            phic_eval = _unsqueeze(phi_c)
            tc_eval = _unsqueeze(t_c)
        else:
            f_eval = f
            m_sec_eval = m_sec
            eta_eval = eta
            delta_eval = delta
            chi_s_eval = chi_s
            chi_a_eval = chi_a
            inc_eval = inc
            phi0_eval = phi_0
            dist_eval = dist
            phic_eval = phi_c
            tc_eval = t_c

        # Evaluate base (2, 2) mode at face-on (iota=0)
        try:
            hp_22_raw, _, phi_22_raw = self.base_22(
                f=f,
                mass1=m1,
                mass2=m2,
                chi1=s1,
                chi2=s2,
                distance=dist,
                inclination=0.0,
                phic=0.0,
                tc=0.0,
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
                phic=0.0,
                tc=0.0,
                f_ref=f_ref,
                phasing_convention=phasing_convention,
                reference_frame=reference_frame,
            )
            phi_22_raw = -torch.angle(hp_22_raw)

        amp_22 = torch.abs(hp_22_raw)
        phi_22 = phi_22_raw + 2.0 * PI * f_eval * tc_eval - 2.0 * phic_eval

        h_plus_total = torch.zeros_like(
            amp_22, dtype=torch.complex128 if dtype == torch.float64 else torch.complex64
        )
        h_cross_total = torch.zeros_like(h_plus_total)
        mode_dict = {}

        # 22 QNM ringdown frequency
        f_rd_22_geom, _ = self.mode_qnm_frequencies(2, 2, m1, m2, s1, s2)
        if is_batched:
            f_rd_22_u = _unsqueeze(f_rd_22_geom)
        else:
            f_rd_22_u = f_rd_22_geom
        f_rd_22_phys = f_rd_22_u / m_sec_eval

        norm_mode = math.sqrt(16.0 * PI / 5.0)

        for l, m in active_modes:
            if (l, m) == (2, 2):
                exp_neg_i_phi22 = torch.complex(torch.cos(phi_22), -torch.sin(phi_22))
                h_mode = (norm_mode * amp_22).to(exp_neg_i_phi22.dtype) * exp_neg_i_phi22
            else:
                # 1. QNM ringdown for mode (l, m)
                f_rd_lm_geom, _ = self.mode_qnm_frequencies(l, m, m1, m2, s1, s2)
                if is_batched:
                    f_rd_lm_u = _unsqueeze(f_rd_lm_geom)
                else:
                    f_rd_lm_u = f_rd_lm_geom
                f_rd_lm_phys = f_rd_lm_u / m_sec_eval

                # Transition frequencies in physical Hz
                f0_amp = (0.018 / m_sec_eval) * (f_rd_lm_phys / f_rd_22_phys)
                f0_phi = (0.014 / m_sec_eval) * (f_rd_lm_phys / f_rd_22_phys)

                # 2. London et al. (2018) Frequency Mappings
                f_map_amp = frequency_mapping_amplitude(f_eval, f0_amp, f_rd_lm_phys, f_rd_22_phys, m)
                f_map_phi = frequency_mapping_phase(f_eval, f0_phi, f_rd_lm_phys, f_rd_22_phys, m)

                # 3. Base 22 amplitude at mapped frequency f_22^A
                f_map_amp_safe = torch.clamp(f_map_amp, min=1e-8)
                hp_amp_eval, _ = self.base_22(
                    f=f_map_amp_safe,
                    mass1=m1,
                    mass2=m2,
                    chi1=s1,
                    chi2=s2,
                    distance=dist,
                    inclination=0.0,
                    phic=0.0,
                    tc=0.0,
                    f_ref=f_ref,
                    phasing_convention=phasing_convention,
                    reference_frame=reference_frame,
                )
                amp_22_mapped = torch.abs(hp_amp_eval)

                # 4. Base 22 phase at mapped frequency f_22^phi
                f_map_phi_safe = torch.clamp(f_map_phi, min=1e-8)
                try:
                    _, _, phi_22_mapped = self.base_22(
                        f=f_map_phi_safe,
                        mass1=m1,
                        mass2=m2,
                        chi1=s1,
                        chi2=s2,
                        distance=dist,
                        inclination=0.0,
                        phic=0.0,
                        tc=0.0,
                        f_ref=f_ref,
                        phasing_convention=phasing_convention,
                        reference_frame=reference_frame,
                        return_phase=True,
                    )
                except (TypeError, ValueError):
                    hp_phi_eval, _ = self.base_22(
                        f=f_map_phi_safe,
                        mass1=m1,
                        mass2=m2,
                        chi1=s1,
                        chi2=s2,
                        distance=dist,
                        inclination=0.0,
                        phic=0.0,
                        tc=0.0,
                        f_ref=f_ref,
                        phasing_convention=phasing_convention,
                        reference_frame=reference_frame,
                    )
                    phi_22_mapped = -torch.angle(hp_phi_eval)

                # 5. Base 22 phase at boundaries f_b0 = 2*f0_phi/m and f_bRD = f_rd_22_phys
                f_b0 = (2.0 / float(m)) * f0_phi
                f_bRD = f_rd_22_phys
                if is_batched:
                    f_boundaries = torch.cat([f_b0, f_bRD], dim=-1)
                else:
                    f_boundaries = torch.stack([f_b0, f_bRD])
                f_boundaries_safe = torch.clamp(f_boundaries, min=1e-8)
                try:
                    _, _, phi_bounds = self.base_22(
                        f=f_boundaries_safe,
                        mass1=m1,
                        mass2=m2,
                        chi1=s1,
                        chi2=s2,
                        distance=dist,
                        inclination=0.0,
                        phic=0.0,
                        tc=0.0,
                        f_ref=f_ref,
                        phasing_convention=phasing_convention,
                        reference_frame=reference_frame,
                        return_phase=True,
                    )
                except (TypeError, ValueError):
                    hp_bounds, _ = self.base_22(
                        f=f_boundaries_safe,
                        mass1=m1,
                        mass2=m2,
                        chi1=s1,
                        chi2=s2,
                        distance=dist,
                        inclination=0.0,
                        phic=0.0,
                        tc=0.0,
                        f_ref=f_ref,
                        phasing_convention=phasing_convention,
                        reference_frame=reference_frame,
                    )
                    phi_bounds = -torch.angle(hp_bounds)

                if is_batched:
                    phi_22_f0 = phi_bounds[..., 0:1]
                    phi_22_frd = phi_bounds[..., 1:2]
                else:
                    phi_22_f0 = phi_bounds[0]
                    phi_22_frd = phi_bounds[1]

                # 6. Phase Connection
                slope_2 = (f_rd_22_phys - (2.0 / float(m)) * f0_phi) / (f_rd_lm_phys - f0_phi)
                slope_3 = f_rd_22_phys / f_rd_lm_phys
                kappa_1 = float(m) / 2.0
                kappa_2 = 1.0 / slope_2
                kappa_3 = 1.0 / slope_3

                delta_1 = (PI / 2.0) * float(3 * l + ((l + m) % 2)) - PI
                delta_2 = (kappa_1 - kappa_2) * phi_22_f0 + delta_1
                delta_3 = (kappa_2 - kappa_3) * phi_22_frd + delta_2

                kappa_f = torch.where(f_eval <= f0_phi, kappa_1, torch.where(f_eval <= f_rd_lm_phys, kappa_2, kappa_3))
                delta_f = torch.where(f_eval <= f0_phi, delta_1, torch.where(f_eval <= f_rd_lm_phys, delta_2, delta_3))

                phi_lm_int = kappa_f * phi_22_mapped + delta_f
                psi_lm = phi_lm_int + 2.0 * PI * f_eval * tc_eval - float(m) * phic_eval

                # 7. Amplitude Rescaling: A_lm(f) = |beta_lm(f)| * A_22[f_22^A(f)]
                v_m = torch.pow(PI * m_sec_eval * torch.clamp((2.0 / float(m)) * f_eval, min=1e-8), 1.0 / 3.0)
                h_hat = h_hat_multipole(l, m, v_m, delta_eval, eta_eval, chi_s_eval, chi_a_eval)
                beta_lm = -h_hat
                amp_lm = torch.abs(beta_lm) * amp_22_mapped

                # Complex mode strain
                exp_neg_i_psi = torch.complex(torch.cos(psi_lm), -torch.sin(psi_lm))
                h_mode = (norm_mode * torch.sign(beta_lm) * amp_lm).to(exp_neg_i_psi.dtype) * exp_neg_i_psi
                h_mode = torch.where(f_eval > 0.0, h_mode, torch.zeros_like(h_mode))

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



# Re-export clean-room calibrated IMRPhenomXHM
from torchwave.approximants.imrphenom_xhm import IMRPhenomXHM

__all__ = [
    "IMRPhenomHM",
    "IMRPhenomXHM",
    "frequency_mapping_amplitude",
    "frequency_mapping_phase",
    "h_hat_multipole",
    "_swsh_projection_coefficients",
    "QNM_FITS",
]
