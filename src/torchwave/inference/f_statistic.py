"""Clean-room Differentiable F-Statistic for Continuous Gravitational-Wave Searches.

Strict Clean-Room Implementation based exclusively on published peer-reviewed literature:
- Jaranowski, Królak, Schutz (1998), Phys. Rev. D 58, 063001 (arXiv:gr-qc/9804014)
  [Data analysis of gravitational-wave signals from spinning neutron stars. I. The signal and its detection]
- Prix (2011), arXiv:1108.3546
  [The F-statistic and its generalizations in continuous gravitational-wave data analysis]
- Cutler & Schutz (2005), Phys. Rev. D 72, 063006 (arXiv:gr-qc/0504011)
  [Generalized F-statistic for gravitational wave searches]
"""

from typing import Dict, Optional, Tuple, Union
import math
import torch
import torch.nn as nn

from torchwave.conversions import as_tensor
from torchwave.detectors.network import Detector
from torchwave.approximants.continuous_wave import continuous_wave_basis


def compute_f_statistic(
    data: torch.Tensor,
    t: torch.Tensor,
    ra: Union[float, torch.Tensor],
    dec: Union[float, torch.Tensor],
    f0: Union[float, torch.Tensor],
    f1: Union[float, torch.Tensor] = 0.0,
    f2: Union[float, torch.Tensor] = 0.0,
    t0: float = 0.0,
    detector: Optional[Union[str, Detector]] = None,
    psd: Optional[Union[float, torch.Tensor]] = None,
    sigma: Optional[Union[float, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the maximum-likelihood F-statistic for continuous gravitational waves.

    Formulas:
        x_mu = <d | h_mu>
        M_munu = <h_mu | h_nu>
        2F = sum_{mu, nu=1}^4 x_mu * (M^-1)^munu * x_nu = x^T M^-1 x
        hat{A}^mu = (M^-1)^munu * x_nu

    Theoretical expectation:
        E[2F] = 4 + rho^2
    where rho^2 = <h | h> is the optimal signal-to-noise ratio squared.

    Autograd-differentiable with respect to sky coordinates (ra, dec) and frequencies (f0, f1).

    Args:
        data: Detector time series data d(t), shape (..., N_samples)
        t: Time stamps in seconds, shape (N_samples,)
        ra: Right ascension in radians
        dec: Declination in radians
        f0: Gravitational-wave frequency in Hz
        f1: First spindown derivative df/dt in Hz/s
        f2: Second spindown derivative d^2f/dt^2 in Hz/s^2
        t0: Reference epoch in seconds
        detector: Detector object or string prefix ('H1', 'L1', 'V1', etc.)
        psd: Single-sided noise power spectral density S_n in 1/Hz
        sigma: Noise standard deviation per time sample (where sigma^2 = S_n / (2*dt))

    Returns:
        two_f: 2 * F-statistic value (scalar or tensor)
        a_hat: Maximum-likelihood amplitude vector hat{A}^mu, shape (..., 4)
        m_mat: Fisher / metric matrix M_munu, shape (..., 4, 4)
    """
    device = data.device
    dtype = data.dtype if data.is_floating_point() else torch.float64

    d_tensor = as_tensor(data, device=device, dtype=dtype)
    t_tensor = as_tensor(t, device=device, dtype=dtype)

    dt = float((t_tensor[1] - t_tensor[0]).item()) if t_tensor.numel() > 1 else 1.0

    # Determine noise weighting factor
    if sigma is not None:
        sigma_t = as_tensor(sigma, device=device, dtype=dtype)
        weight = 1.0 / (sigma_t * sigma_t)
    elif psd is not None:
        psd_t = as_tensor(psd, device=device, dtype=dtype)
        # In discrete time: <a | b> = (2 / S_n) * sum a_n b_n * dt
        weight = (2.0 * dt) / psd_t
    else:
        # Unweighted / unit variance inner product
        weight = 1.0

    # Compute four quadrupolar basis functions h_mu(t), shape (4, N_samples) or (..., 4, N_samples)
    h_basis = continuous_wave_basis(
        t=t_tensor,
        ra=ra,
        dec=dec,
        f0=f0,
        f1=f1,
        f2=f2,
        t0=t0,
        detector=detector,
    )

    # 1. Correlate data with each basis template: x_mu = <d | h_mu>
    # data: (..., N), h_basis: (..., 4, N)
    if d_tensor.dim() == 1 and h_basis.dim() == 2:
        # h_basis: (4, N)
        x_mu = torch.sum(h_basis * d_tensor.unsqueeze(0), dim=-1) * weight  # (4,)
        # 2. Metric matrix M_munu = <h_mu | h_nu>, shape (4, 4)
        # Outer product contracted over time
        m_mat = torch.matmul(h_basis, h_basis.transpose(0, 1)) * weight  # (4, 4)
        # Regularize adaptively for numerical stability
        scale = torch.clamp(torch.diagonal(m_mat, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1), min=1e-12)
        m_reg = m_mat + 1e-6 * scale * torch.eye(4, device=device, dtype=dtype)
        a_hat = torch.linalg.lstsq(m_reg, x_mu.unsqueeze(-1)).solution.squeeze(-1)
        two_f = torch.dot(x_mu, a_hat)
    else:
        # Batched support
        # Broadcast data and basis shapes
        if d_tensor.dim() > 1 and h_basis.dim() == 2:
            h_basis = h_basis.unsqueeze(0).expand(d_tensor.shape[0], -1, -1)
        elif d_tensor.dim() == 1 and h_basis.dim() > 2:
            d_tensor = d_tensor.expand(h_basis.shape[0], -1)

        x_mu = torch.sum(h_basis * d_tensor.unsqueeze(-2), dim=-1) * weight  # (..., 4)
        m_mat = torch.matmul(h_basis, h_basis.transpose(-2, -1)) * weight  # (..., 4, 4)
        scale = torch.clamp(torch.diagonal(m_mat, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1), min=1e-12)
        eye = torch.eye(4, device=device, dtype=dtype).expand_as(m_mat)
        m_reg = m_mat + 1e-6 * scale * eye
        a_hat = torch.linalg.lstsq(m_reg, x_mu.unsqueeze(-1)).solution.squeeze(-1)
        two_f = torch.sum(x_mu * a_hat, dim=-1)


    return two_f, a_hat, m_mat


class FStatistic(nn.Module):
    """PyTorch Module for Continuous Wave F-statistic computation."""

    def __init__(
        self,
        t: torch.Tensor,
        detector: Optional[Union[str, Detector]] = None,
        psd: Optional[Union[float, torch.Tensor]] = None,
        sigma: Optional[Union[float, torch.Tensor]] = None,
        t0: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("t", as_tensor(t))
        self.detector = detector
        self.psd = psd
        self.sigma = sigma
        self.t0 = t0

    def forward(
        self,
        data: torch.Tensor,
        ra: Union[float, torch.Tensor],
        dec: Union[float, torch.Tensor],
        f0: Union[float, torch.Tensor],
        f1: Union[float, torch.Tensor] = 0.0,
        f2: Union[float, torch.Tensor] = 0.0,
    ) -> torch.Tensor:
        """Compute 2F statistic value."""
        two_f, _, _ = compute_f_statistic(
            data=data,
            t=self.t,
            ra=ra,
            dec=dec,
            f0=f0,
            f1=f1,
            f2=f2,
            t0=self.t0,
            detector=self.detector,
            psd=self.psd,
            sigma=self.sigma,
        )
        return two_f
