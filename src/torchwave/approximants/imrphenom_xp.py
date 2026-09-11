"""IMRPhenomXP Precessing Dominant-Harmonic Frequency-Domain Approximant.

Strict Clean-Room Implementation based exclusively on published literature:
- Pratten et al. (2021), Phys. Rev. D 103, 104056 (arXiv:2004.06503), Sec. IV & Appendix B
- Pratten et al. (2020), Phys. Rev. D 102, 064001 (arXiv:2001.11412)
"""

from typing import Dict, Optional, Sequence, Tuple, Union
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
)
from torchwave.approximants.imrphenom_xas import IMRPhenomXAS
from torchwave.approximants.precession_msa import evaluate_msa_precession
from torchwave.approximants.imrphenom_xphm import wigner_small_d_matrix


class IMRPhenomXP(nn.Module):
    """IMRPhenomXP Precessing Dominant-Harmonic (l=2, |m|=2) Waveform Approximant.

    Twists up underlying coprecessing dominant (2, 2) mode from IMRPhenomXAS
    using Multiple Scale Analysis (MSA variant 223) or NNLO precession angles
    with Wigner small-d matrix frame rotations and polarization basis alignment.
    """

    def __init__(self):
        super().__init__()
        self.xas_model = IMRPhenomXAS()

    def forward(
        self,
        f: Optional[torch.Tensor] = None,
        mass1: Union[float, torch.Tensor] = 30.0,
        mass2: Union[float, torch.Tensor] = 30.0,
        spin1x: Union[float, torch.Tensor] = 0.0,
        spin1y: Union[float, torch.Tensor] = 0.0,
        spin1z: Optional[Union[float, torch.Tensor]] = None,
        spin2x: Union[float, torch.Tensor] = 0.0,
        spin2y: Union[float, torch.Tensor] = 0.0,
        spin2z: Optional[Union[float, torch.Tensor]] = None,
        chi1: Optional[Union[float, torch.Tensor]] = None,
        chi2: Optional[Union[float, torch.Tensor]] = None,
        distance: Union[float, torch.Tensor] = 100.0,
        inclination: Union[float, torch.Tensor] = 0.0,
        phic: Union[float, torch.Tensor] = 0.0,
        tc: Union[float, torch.Tensor] = 0.0,
        f_ref: Optional[Union[float, torch.Tensor]] = None,
        phi0: Union[float, torch.Tensor] = 0.0,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        sample_frequencies: Optional[torch.Tensor] = None,
        precession_method: str = "msa",
        prec_version: int = 223,
        convention: int = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate frequency-domain precessing dominant-mode polarizations (h_plus, h_cross)."""
        freqs = f if f is not None else sample_frequencies
        if freqs is None:
            raise ValueError("Must provide frequency grid via 'f' or 'sample_frequencies'")

        device = freqs.device if isinstance(freqs, torch.Tensor) else None
        dtype = freqs.dtype if isinstance(freqs, torch.Tensor) and freqs.is_floating_point() else torch.float64
        complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64

        m1 = as_tensor(mass1, device=device, dtype=dtype)
        m2 = as_tensor(mass2, device=device, dtype=dtype)
        s1x = as_tensor(spin1x, device=device, dtype=dtype)
        s1y = as_tensor(spin1y, device=device, dtype=dtype)
        s2x = as_tensor(spin2x, device=device, dtype=dtype)
        s2y = as_tensor(spin2y, device=device, dtype=dtype)

        s1z_val = spin1z if spin1z is not None else (chi1 if chi1 is not None else 0.0)
        s2z_val = spin2z if spin2z is not None else (chi2 if chi2 is not None else 0.0)
        s1z = as_tensor(s1z_val, device=device, dtype=dtype)
        s2z = as_tensor(s2z_val, device=device, dtype=dtype)

        dist = as_tensor(distance, device=device, dtype=dtype)
        inc = as_tensor(inclination, device=device, dtype=dtype)
        phi_c = as_tensor(phic, device=device, dtype=dtype)
        t_c = as_tensor(tc, device=device, dtype=dtype)
        fref = as_tensor(f_ref, device=device, dtype=dtype) if f_ref is not None else None
        phi_0 = as_tensor(phi0, device=device, dtype=dtype)

        # Canonical body ordering: primary mass m1 >= m2
        swap_mask = m1 < m2
        if torch.any(swap_mask):
            m1_eff = torch.where(swap_mask, m2, m1)
            m2_eff = torch.where(swap_mask, m1, m2)
            s1x_eff = torch.where(swap_mask, s2x, s1x)
            s1y_eff = torch.where(swap_mask, s2y, s1y)
            s1z_eff = torch.where(swap_mask, s2z, s1z)
            s2x_eff = torch.where(swap_mask, s1x, s2x)
            s2y_eff = torch.where(swap_mask, s1y, s2y)
            s2z_eff = torch.where(swap_mask, s1z, s2z)
            m1, m2 = m1_eff, m2_eff
            s1x, s1y, s1z = s1x_eff, s1y_eff, s1z_eff
            s2x, s2y, s2z = s2x_eff, s2y_eff, s2z_eff

        # Check aligned-spin limit
        in_plane_sq = s1x * s1x + s1y * s1y + s2x * s2x + s2y * s2y
        is_aligned = in_plane_sq < 1e-12

        # Fast path: if purely aligned scalar, directly return IMRPhenomXAS
        if isinstance(is_aligned, torch.Tensor) and is_aligned.dim() == 0 and bool(is_aligned):
            return self.xas_model(
                f=freqs,
                mass1=m1,
                mass2=m2,
                chi1=s1z,
                chi2=s2z,
                distance=dist,
                inclination=inc,
                phi_c=phi_c,
                t_c=t_c,
                f_ref=fref,
                phasing_convention=phasing_convention,
                reference_frame=reference_frame,
            )

        total_mass = m1 + m2
        m_sec = total_mass * MSUN_SEC
        is_batched = total_mass.dim() >= 1

        f_eval = freqs if (not is_batched or freqs.dim() > 1) else freqs.unsqueeze(0)

        def unsqueeze_param(t: torch.Tensor) -> torch.Tensor:
            if is_batched and f_eval.dim() > t.dim():
                return t.unsqueeze(-1)
            return t

        fref_eval = fref if fref is not None else torch.tensor(20.0, device=device, dtype=dtype)
        v = (PI * unsqueeze_param(m_sec) * f_eval) ** (1.0 / 3.0)
        v0 = (PI * unsqueeze_param(m_sec) * unsqueeze_param(fref_eval)) ** (1.0 / 3.0)

        # Precession method selection
        if precession_method.lower() == "nnlo":
            p_ver = 102 if prec_version in (220, 223) else prec_version
        else:
            p_ver = prec_version

        # Evaluate precession angles
        alpha, beta, gamma, theta_jn, zeta = evaluate_msa_precession(
            v=v,
            v0=v0,
            m1=unsqueeze_param(m1),
            m2=unsqueeze_param(m2),
            s1x=unsqueeze_param(s1x),
            s1y=unsqueeze_param(s1y),
            s1z=unsqueeze_param(s1z),
            s2x=unsqueeze_param(s2x),
            s2y=unsqueeze_param(s2y),
            s2z=unsqueeze_param(s2z),
            iota=unsqueeze_param(inc),
            phi_ref=unsqueeze_param(phi_0),
            prec_version=p_ver,
            convention=convention,
        )

        # Evaluate carrier (2, 2) mode at face-on (iota=0)
        hp_carrier, _ = self.xas_model(
            f=f_eval,
            mass1=m1,
            mass2=m2,
            chi1=s1z,
            chi2=s2z,
            distance=dist,
            inclination=torch.zeros_like(inc),
            phi_c=phi_c,
            t_c=t_c,
            f_ref=fref,
            phasing_convention=phasing_convention,
            reference_frame=reference_frame,
        )

        # Twist-up for (l=2, mp=2)
        norm_mode = math.sqrt(16.0 * PI / 5.0)
        scale_2 = 1.0 / norm_mode
        d_beta = wigner_small_d_matrix(2, beta)
        d_jn = wigner_small_d_matrix(2, unsqueeze_param(theta_jn))

        term_plus = torch.zeros(f_eval.shape, dtype=complex_dtype, device=device)
        term_cross = torch.zeros(f_eval.shape, dtype=complex_dtype, device=device)

        phi0_u = unsqueeze_param(phi_0)

        for m_val in range(-2, 3):
            m_idx = m_val + 2
            dm_minus = d_beta[..., m_idx, -2 + 2]
            dm_plus = d_beta[..., m_idx, 2 + 2]
            y_lm = d_jn[..., m_idx, -2 + 2]

            m_phase = float(m_val) * (phi0_u - alpha)
            cos_m = torch.cos(m_phase)
            sin_m = torch.sin(m_phase)

            exp_minus_ima = torch.complex(cos_m, sin_m)
            exp_plus_ima = torch.complex(cos_m, -sin_m)

            a_m_minus = exp_minus_ima * dm_minus * y_lm
            a_m_plus_star = exp_plus_ima * dm_plus * y_lm

            term_plus = term_plus + (a_m_minus + a_m_plus_star)
            term_cross = term_cross + (a_m_minus - a_m_plus_star)

        # Coprecessing angle factor: exp(2i * gamma) = exp(-2i * epsilon)
        two_gamma = 2.0 * gamma
        exp_two_gamma = torch.complex(torch.cos(two_gamma), torch.sin(two_gamma))

        hp_raw = scale_2 * exp_two_gamma * hp_carrier * term_plus
        hc_raw = -1j * scale_2 * exp_two_gamma * hp_carrier * term_cross

        # Polarization basis rotation by exp(2i * zeta)
        two_zeta = 2.0 * unsqueeze_param(zeta)
        cos_2z = torch.cos(two_zeta)
        sin_2z = torch.sin(two_zeta)

        hp_precess = cos_2z * hp_raw + sin_2z * hc_raw
        hc_precess = -sin_2z * hp_raw + cos_2z * hc_raw

        # If aligned-spin limit, evaluate exact non-precessing waveform
        if is_aligned.any():
            hp_al, hc_al = self.xas_model(
                f=f_eval,
                mass1=m1,
                mass2=m2,
                chi1=s1z,
                chi2=s2z,
                distance=dist,
                inclination=inc,
                phi_c=phi_c,
                t_c=t_c,
                f_ref=fref,
                phasing_convention=phasing_convention,
                reference_frame=reference_frame,
            )
            is_al_u = unsqueeze_param(is_aligned)
            hp_final = torch.where(is_al_u, hp_al, hp_precess)
            hc_final = torch.where(is_al_u, hc_al, hc_precess)
        else:
            hp_final = hp_precess
            hc_final = hc_precess

        return hp_final, hc_final

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
