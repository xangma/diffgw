"""IMRPhenomNSBH Frequency-Domain Gravitational Waveform Approximant.

Strict Clean-Room Implementation based exclusively on published literature:
- Pannarale et al. (2015), Phys. Rev. D 92, 084032 (arXiv:1509.06209)
- Matas et al. (2020), Phys. Rev. D 102, 043023 (arXiv:2004.10001)
- Yagi & Yunes (2013), Science 341, 365; PRD 88, 023009 [Universal compactness relations]
- Dietrich et al. (2019), Phys. Rev. D 100, 044003 [NRTidalv2 phasing]
"""

from typing import Optional, Tuple, Union
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
    kerr_isco_frequency,
)
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.imrphenomd_nrtidal import IMRPhenomD_NRTidalv2, _planck_taper_window


def ns_compactness_from_lambda(lambda_ns: torch.Tensor) -> torch.Tensor:
    """Compute neutron star compactness C = M_NS / R_NS from tidal deformability Lambda.

    Universal I-Love-Q / C-Love relation (Yagi & Yunes 2013, Maselli et al. 2013):
        C = 0.371 - 0.0391 * ln(Lambda) + 0.001056 * ln(Lambda)^2
    """
    lam_safe = torch.clamp(lambda_ns, min=1.0)
    ln_lam = torch.log(lam_safe)
    c = 0.371 - 0.0391 * ln_lam + 0.001056 * (ln_lam ** 2.0)
    # Physical NS compactness is bounded between ~0.08 and 0.30
    c_clamped = torch.clamp(c, min=0.08, max=0.35)
    # For black holes (Lambda -> 0), compactness is 0.5
    return torch.where(lambda_ns > 1e-3, c_clamped, torch.full_like(lambda_ns, 0.5))


def nsbh_frequencies(
    mass1: torch.Tensor,
    mass2: torch.Tensor,
    chi1: torch.Tensor,
    lambda2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute tidal disruption frequency f_tide, Kerr ISCO frequency f_isco, ringdown f_rd, and compactness C.

    Literature: Pannarale et al. (2015), Matas et al. (2020).

    Returns:
        (f_tide, f_isco, f_rd, compactness) in Hz
    """
    # Mass ratio Q = m1 / m2 >= 1
    q = mass1 / torch.clamp(mass2, min=1e-6)
    m1_sec = mass1 * MSUN_SEC
    total_mass = mass1 + mass2
    m_sec = total_mass * MSUN_SEC

    # 1. NS compactness
    c = ns_compactness_from_lambda(lambda2)

    # 2. Dimensionless tidal disruption radius r_tide / M_BH (Pannarale et al. 2015 Eq. 2)
    # r_tide / M_BH = (3 / Q^2)^(1/3) / C
    r_tide_dimless = torch.pow(3.0 / (q * q), 1.0 / 3.0) / torch.clamp(c, min=1e-4)

    # GW frequency at tidal disruption for equatorial Kerr orbit
    # f_GW = 1 / (pi * M_BH_sec * (r^(3/2) + chi_1))
    f_tide = 1.0 / (PI * m1_sec * (torch.pow(torch.clamp(r_tide_dimless, min=1.0), 1.5) + chi1))

    # 3. Kerr ISCO frequency
    f_isco = kerr_isco_frequency(mass1, chi1)

    # 4. Ringdown frequency of remnant BH
    m_final, a_final, f_rd_norm, _ = IMRPhenomD.remnant_properties(mass1, mass2, chi1, torch.zeros_like(chi1))
    f_rd = f_rd_norm / m_sec

    return f_tide, f_isco, f_rd, c


class IMRPhenomNSBH(nn.Module):
    """IMRPhenomNSBH Aligned-Spin Frequency-Domain Model for NS-BH Mergers."""

    def __init__(self):
        super().__init__()
        self.nrtidal_model = IMRPhenomD_NRTidalv2()

    def forward(
        self,
        f: torch.Tensor,
        mass1: Union[float, torch.Tensor],
        mass2: Union[float, torch.Tensor],
        chi1: Union[float, torch.Tensor] = 0.0,
        chi2: Union[float, torch.Tensor] = 0.0,
        lambda2: Union[float, torch.Tensor] = 0.0,
        distance: Union[float, torch.Tensor] = 100.0,
        inclination: Union[float, torch.Tensor] = 0.0,
        phic: Union[float, torch.Tensor] = 0.0,
        tc: Union[float, torch.Tensor] = 0.0,
        f_ref: Optional[Union[float, torch.Tensor]] = None,
        sigma_disrupt: float = 6.0,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate frequency-domain polarizations (h_plus, h_cross) for NSBH coalescences.

        Args:
            f: Frequency tensor in Hz
            mass1: Primary mass (BH) in solar masses
            mass2: Secondary mass (NS) in solar masses
            chi1: Dimensionless aligned spin of BH
            chi2: Dimensionless aligned spin of NS (typically ~0)
            lambda2: Dimensionless tidal deformability of NS
            distance: Luminosity distance in Mpc
            inclination: Orbital inclination in radians
            phic: Coalescence phase in radians
            tc: Coalescence time in seconds
            f_ref: Reference frequency in Hz
            sigma_disrupt: Steepness parameter for exponential disruption cutoff
        """
        device = f.device
        dtype = f.dtype if f.is_floating_point() else torch.float64

        m1 = as_tensor(mass1, device=device, dtype=dtype)
        m2 = as_tensor(mass2, device=device, dtype=dtype)
        s1 = as_tensor(chi1, device=device, dtype=dtype)
        s2 = as_tensor(chi2, device=device, dtype=dtype)
        lam2 = as_tensor(lambda2, device=device, dtype=dtype)

        # 1. Base NRTidalv2 waveform with matter effects in phase
        hp_base, hc_base = self.nrtidal_model(
            f=f,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            distance=distance,
            inclination=inclination,
            phic=phic,
            tc=tc,
            lambda1=torch.zeros_like(lam2),
            lambda2=lam2,
            f_ref=f_ref,
            phasing_convention=phasing_convention,
            reference_frame=reference_frame,
            apply_merger_taper=False,
        )

        # 2. Compute key physical transition frequencies
        f_tide, f_isco, f_rd, _ = nsbh_frequencies(m1, m2, s1, lam2)

        is_batched = m1.dim() >= 1
        if is_batched and f.dim() == 1:
            f_eval = f.unsqueeze(0)
        else:
            f_eval = f

        def unsqueeze_param(tensor: torch.Tensor) -> torch.Tensor:
            if is_batched and f_eval.dim() > tensor.dim():
                return tensor.unsqueeze(-1)
            return tensor

        ft_eval = unsqueeze_param(f_tide)
        fisco_eval = unsqueeze_param(f_isco)
        frd_eval = unsqueeze_param(f_rd)

        # 3. Disruption classification & amplitude window
        # Case A: Non-disrupting (f_tide > f_rd): Full BH-like ringdown
        # Case B: Mildly disrupting (f_isco < f_tide <= f_rd): Partial ringdown suppression
        # Case C: Completely disrupting (f_tide <= f_isco): Ringdown extinguished above f_tide

        # Exponential disruption taper above f_tide
        df_disrupt = ft_eval / sigma_disrupt
        exp_cutoff = torch.exp(-torch.clamp((f_eval - ft_eval) / df_disrupt, min=0.0, max=50.0))
        w_complete = torch.where(f_eval <= ft_eval, torch.ones_like(f_eval), exp_cutoff)

        # Mild disruption: partial suppression connecting f_tide to f_rd
        df_mild = torch.clamp(frd_eval - fisco_eval, min=1.0)
        exp_mild = torch.exp(-torch.clamp((f_eval - ft_eval) / (0.5 * df_mild), min=0.0, max=50.0))
        w_mild = torch.where(f_eval <= ft_eval, torch.ones_like(f_eval), exp_mild)

        # Assemble outcome window
        w_disrupt = torch.where(
            ft_eval <= fisco_eval,
            w_complete,
            torch.where(
                ft_eval <= frd_eval,
                w_mild,
                torch.ones_like(f_eval),
            ),
        )

        h_plus = hp_base * w_disrupt.to(hp_base.dtype)
        h_cross = hc_base * w_disrupt.to(hc_base.dtype)

        return h_plus, h_cross

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
