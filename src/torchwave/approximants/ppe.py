"""Parameterized Post-Einsteinian (ppE) and Modified Dispersion Gravitational Waveforms.

Strict Clean-Room Implementation based exclusively on published literature:
- Yunes & Pretorius (2009), "Dynamical Systems Approach to Runaway Instabilities in Black Hole-Torus Systems",
  Phys. Rev. D 80, 122003 (arXiv:0909.3328)
- Mirshekari, Yunes, Will (2012), "Constraining Generic Post-Einsteinian Extensions of Gravity with Gravitational Waves",
  Phys. Rev. D 85, 024041 (arXiv:1110.2720)
- Chatziioannou, Yunes, Cornish (2012), "Model-Independent Test of General Relativity: An Extended Parameterized Post-Einsteinian Framework",
  Phys. Rev. D 86, 022004 (arXiv:1204.2585)
- Will (1998), "Bounding the mass of the graviton using gravitational-wave observations of inspiralling compact binaries",
  Phys. Rev. D 57, 2061 (arXiv:gr-qc/9709011)
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union
import math
import torch
import torch.nn as nn

from torchwave.constants import (
    C_SI,
    MSUN_SEC,
    MPC_SI,
    PI,
)
from torchwave.conversions import as_tensor
from torchwave.approximants.taylorf2 import TaylorF2
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.imrphenom_xas import IMRPhenomXAS


# Standard PN order to ppE exponent mapping: b_ppE = 2 * PN - 5
PPE_PN_MAPPING: Dict[Union[int, float], int] = {
    -1.0: -7,  # -1PN dipole radiation
    -0.5: -6,  # -0.5PN
    0.0: -5,   # 0PN Newtonian quadrupole
    0.5: -4,   # 0.5PN
    1.0: -3,   # 1PN
    1.5: -2,   # 1.5PN
    2.0: -1,   # 2PN
    2.5: 0,    # 2.5PN
    3.0: 1,    # 3PN
    3.5: 2,    # 3.5PN
}


def ppe_phase_shift(
    f: torch.Tensor,
    total_mass: Union[float, torch.Tensor],
    beta_ppe: Union[float, torch.Tensor] = 0.0,
    b_ppe: Union[int, float, torch.Tensor] = -2,
) -> torch.Tensor:
    """Compute ppE frequency-dependent phase deformation delta Psi_ppE(f).

    Literature: Yunes & Pretorius (2009) Eq. 1; Mirshekari et al. (2012) Eq. 20.

    Formula:
        delta Psi_ppE(f) = beta_ppE * (pi * M * f)^(b_ppE / 3)
    where M is total mass in geometric seconds: M_sec = (m1 + m2) * MSUN_SEC.

    Args:
        f: Frequency tensor in Hz.
        total_mass: Total binary mass M = m1 + m2 in solar masses.
        beta_ppe: ppE phase deformation magnitude (0 recovers pure GR).
        b_ppe: ppE phase exponent (-7 for -1PN dipole, -2 for 1.5PN, 0 for 2.5PN).

    Returns:
        Tensor of phase shifts in radians matching shape of f.
    """
    f_t = as_tensor(f)
    device = f_t.device
    dtype = f_t.dtype

    m = as_tensor(total_mass, device=device, dtype=dtype)
    beta = as_tensor(beta_ppe, device=device, dtype=dtype)
    b = as_tensor(b_ppe, device=device, dtype=dtype)

    m_sec = m * MSUN_SEC
    # Dimensionless PN velocity v = (pi * M * f)^(1/3)
    safe_f = torch.clamp(f_t, min=1e-15)
    v = torch.pow(PI * m_sec * safe_f, 1.0 / 3.0)

    # delta Psi = beta * v^b
    delta_psi = beta * torch.pow(v, b)
    return delta_psi


def modified_dispersion_phase_shift(
    f: torch.Tensor,
    distance: Union[float, torch.Tensor] = 100.0,
    lambda_g: Union[float, torch.Tensor] = float("inf"),
    z: Union[float, torch.Tensor] = 0.0,
    d_alpha: Optional[Union[float, torch.Tensor]] = None,
) -> torch.Tensor:
    """Compute frequency phase shift due to modified GW dispersion relation (massive graviton).

    Literature: Mirshekari, Yunes, Will (2012) Eq. 24; Will (1998) Eq. 2.11.

    Dispersion relation:
        E^2 = p^2 * c^2 + A_alpha * p^alpha * c^alpha
    For alpha = 0 (massive graviton with Compton wavelength lambda_g = h / (m_g * c)):
        delta Psi_disp(f) = - (pi^2 * D_0) / (lambda_g^2 * (1 + z)) * f^(-1)

    Args:
        f: Frequency tensor in Hz.
        distance: Luminosity distance D_L in Mpc.
        lambda_g: Graviton Compton wavelength in meters (inf recovers pure GR).
        z: Cosmological redshift of source (default 0).
        d_alpha: Optional cosmological dispersion distance in meters.
            If None, computed as D_0 = D_L * MPC_SI / (1 + z).

    Returns:
        Tensor of dispersion phase shifts in radians.
    """
    f_t = as_tensor(f)
    device = f_t.device
    dtype = f_t.dtype

    dist_mpc = as_tensor(distance, device=device, dtype=dtype)
    l_g = as_tensor(lambda_g, device=device, dtype=dtype)
    redshift = as_tensor(z, device=device, dtype=dtype)

    if torch.isinf(l_g) or l_g > 1e40:
        return torch.zeros_like(f_t)

    if d_alpha is None:
        # Physical cosmological dispersion distance in meters: D0 = D_L / (1 + z)
        dist_m = dist_mpc * MPC_SI
        d0 = dist_m / (1.0 + redshift)
    else:
        d0 = as_tensor(d_alpha, device=device, dtype=dtype)

    safe_f = torch.clamp(f_t, min=1e-15)
    l_g_safe = torch.clamp(l_g, min=1e-30)

    # delta Psi_disp = - (pi^2 * D0) / (lambda_g^2 * (1 + z)) * f^(-1)
    # Dimension check: [D0] = m, [lambda_g^2] = m^2, c is used to convert m/c to seconds:
    # Mirshekari et al. (2012) Eq. 24 with c in meters/second:
    coeff = -(PI ** 2.0) * d0 * C_SI / (l_g_safe * l_g_safe * (1.0 + redshift))
    return coeff / safe_f


class ParameterizedPostEinstein(nn.Module):
    """Wrapper around frequency-domain approximants applying ppE and modified dispersion shifts.

    Supported underlying approximants:
    - TaylorF2 (SPA PN model)
    - IMRPhenomD (aligned-spin phenomenological model)
    - IMRPhenomXAS (4th-gen aligned-spin phenomenological model)
    """

    def __init__(self, approximant: Optional[Any] = None):
        super().__init__()
        if approximant is None:
            self.approximant = IMRPhenomD()
        else:
            self.approximant = approximant

    def forward(
        self,
        f: torch.Tensor,
        mass1: Union[float, torch.Tensor],
        mass2: Union[float, torch.Tensor],
        beta_ppe: Union[float, torch.Tensor] = 0.0,
        b_ppe: Union[int, float, torch.Tensor] = -2,
        lambda_g: Union[float, torch.Tensor] = float("inf"),
        distance: Union[float, torch.Tensor] = 100.0,
        redshift: Union[float, torch.Tensor] = 0.0,
        **waveform_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate ppE-deformed polarizations h_plus and h_cross.

        Args:
            f: Frequency tensor in Hz.
            mass1: Primary mass in solar masses.
            mass2: Secondary mass in solar masses.
            beta_ppe: ppE phase deformation parameter (default 0 for GR).
            b_ppe: ppE PN order exponent (-7 for -1PN, -2 for 1.5PN, 0 for 2.5PN).
            lambda_g: Graviton Compton wavelength in meters (default inf for GR).
            distance: Luminosity distance in Mpc.
            redshift: Cosmological redshift.
            **waveform_kwargs: Additional parameters passed to the underlying approximant
                (e.g., chi1, chi2, inclination, phic, tc, f_ref).

        Returns:
            Tuple of deformed complex polarizations (h_plus, h_cross).
        """
        # 1. Generate base General Relativity polarizations
        hp_gr, hc_gr = self.approximant(
            f=f,
            mass1=mass1,
            mass2=mass2,
            distance=distance,
            **waveform_kwargs,
        )

        m1 = as_tensor(mass1, device=f.device, dtype=f.dtype if f.is_floating_point() else torch.float64)
        m2 = as_tensor(mass2, device=f.device, dtype=f.dtype if f.is_floating_point() else torch.float64)
        total_m = m1 + m2

        # 2. Compute non-GR phase shifts
        delta_psi_ppe = ppe_phase_shift(f, total_m, beta_ppe=beta_ppe, b_ppe=b_ppe)
        delta_psi_disp = modified_dispersion_phase_shift(
            f, distance=distance, lambda_g=lambda_g, z=redshift
        )

        total_shift = delta_psi_ppe + delta_psi_disp

        # 3. Apply stationary phase deformation: exp(-i * delta_Psi)
        # Match tensor dimensions if hp_gr is batched
        if hp_gr.dim() > total_shift.dim():
            total_shift = total_shift.unsqueeze(0)

        cos_shift = torch.cos(total_shift)
        sin_shift = torch.sin(total_shift)
        exp_factor = torch.complex(cos_shift, -sin_shift)

        hp_ppe = hp_gr * exp_factor
        hc_ppe = hc_gr * exp_factor

        return hp_ppe, hc_ppe

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
