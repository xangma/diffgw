"""Pulsar Timing Array (PTA) Network and Hellings-Downs Likelihood.

Strict Clean-Room Implementation based exclusively on published literature:
- Hellings & Downs (1983), "Upper limits on the isotropic gravitational radiation background from pulsar timing analysis",
  Astrophys. J. Lett. 265, L39
- Arzoumanian et al. (2020), "The NANOGrav 12.5 yr Data Set: Search for an Isotropic Stochastic Gravitational-wave Background",
  Astrophys. J. Lett. 905, L34 (arXiv:2009.04496)
- Allen (2023), "Variance of the Hellings-Downs correlation",
  Phys. Rev. D 107, 043018 (arXiv:2205.05637)
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn

from torchwave.constants import PI
from torchwave.conversions import as_tensor

# 1 Julian year in seconds: 365.25 * 86400 s
SEC_PER_YEAR: float = 3.15576e7
F_YEAR: float = 1.0 / SEC_PER_YEAR  # ~ 3.16875e-8 Hz


def pulsar_line_of_sight(
    ra: Union[float, torch.Tensor],
    dec: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Compute Cartesian unit line-of-sight vector pointing from Earth to pulsar.

    p_hat = (cos(dec) * cos(ra), cos(dec) * sin(ra), sin(dec))

    Args:
        ra: Right ascension in radians.
        dec: Declination in radians.

    Returns:
        Tensor of shape (..., 3) with unit-length vectors.
    """
    alpha = as_tensor(ra)
    delta = as_tensor(dec, device=alpha.device, dtype=alpha.dtype)

    cos_d = torch.cos(delta)
    sin_d = torch.sin(delta)
    cos_a = torch.cos(alpha)
    sin_a = torch.sin(alpha)

    p_x = cos_d * cos_a
    p_y = cos_d * sin_a
    p_z = sin_d

    return torch.stack([p_x, p_y, p_z], dim=-1)


def angular_separation(
    p1: torch.Tensor,
    p2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute angular separation zeta and cos(zeta) between pairs of pulsars.

    Args:
        p1: Unit line-of-sight vector(s) of shape (..., 3).
        p2: Unit line-of-sight vector(s) of shape (..., 3).

    Returns:
        Tuple of (zeta, cos_zeta).
    """
    cos_zeta = torch.sum(p1 * p2, dim=-1)
    cos_zeta_clamped = torch.clamp(cos_zeta, -1.0, 1.0)
    zeta = torch.acos(cos_zeta_clamped)
    return zeta, cos_zeta_clamped


def hellings_downs_correlation(
    zeta: torch.Tensor,
    is_auto: Union[bool, torch.Tensor] = False,
    include_pulsar_term: bool = True,
) -> torch.Tensor:
    """Exact Hellings-Downs spatial correlation function for isotropic unpolarized GWB.

    Literature: Hellings & Downs (1983) Eq. 5; Arzoumanian et al. (2020) Sec. 2.

    Formula:
        x = (1 - cos(zeta)) / 2
        Gamma(zeta) = 1/2 - 1/4 * x + 3/2 * x * ln(x) + 1/2 * delta_{ab}

    Properties:
        - Cross-pulsar limit as zeta -> 0: Gamma(0) = 1/2 = 0.5 (since x * ln(x) -> 0).
        - Auto-correlation (a = b): Gamma(0) + 1/2 = 1.0.
        - Antipodal separation (zeta = pi, x = 1): Gamma(pi) = 1/2 - 1/4 = 0.25.
        - The characteristic dip at zeta ~ 90 deg has Gamma(pi/2) ~ -0.1448.

    Args:
        zeta: Angular separation in radians (scalar or tensor).
        is_auto: Boolean or boolean tensor indicating identical pulsars (delta_{ab} = 1).
        include_pulsar_term: If True, include the 1/2 * delta_{ab} auto-correlation pulsar term.

    Returns:
        Tensor of spatial correlation factors with same shape as zeta.
    """
    z = as_tensor(zeta)
    cos_z = torch.clamp(torch.cos(z), -1.0, 1.0)

    # Normalized separation variable x in [0, 1]
    x = 0.5 * (1.0 - cos_z)
    x_clamped = torch.clamp(x, min=1e-15, max=1.0)

    # Stable x * ln(x) with lim_{x->0} x * ln(x) = 0
    x_ln_x = torch.where(x < 1e-12, torch.zeros_like(x), x_clamped * torch.log(x_clamped))

    gamma = 0.5 - 0.25 * x + 1.5 * x_ln_x

    if include_pulsar_term:
        if isinstance(is_auto, bool):
            if is_auto:
                gamma = gamma + 0.5
        else:
            auto_t = as_tensor(is_auto, device=z.device, dtype=z.dtype)
            gamma = gamma + 0.5 * auto_t

    return gamma


def stochastic_gwb_spectrum(
    f: torch.Tensor,
    a_gwb: Union[float, torch.Tensor],
    gamma: Union[float, torch.Tensor] = 13.0 / 3.0,
    f_ref: float = F_YEAR,
) -> torch.Tensor:
    """Compute stochastic gravitational-wave background strain spectral density S_h(f).

    Literature: Arzoumanian et al. (2020) Eq. 2:
        S_h(f) = (A_GWB^2 / (12 * pi^2)) * (f / f_yr)^(2 - gamma) * f_yr^(-3)

    Args:
        f: Frequency tensor in Hz.
        a_gwb: Dimensionless GWB amplitude at reference frequency f_ref (e.g. ~ 1e-15).
        gamma: Power-law spectral slope index (default 13/3 for SMBHB inspirals).
        f_ref: Reference frequency in Hz (default 1/yr = 3.16875e-8 Hz).

    Returns:
        Tensor of strain power spectral density S_h(f) in 1 / Hz.
    """
    f_t = as_tensor(f)
    device = f_t.device
    dtype = f_t.dtype

    amp = as_tensor(a_gwb, device=device, dtype=dtype)
    gam = as_tensor(gamma, device=device, dtype=dtype)

    safe_f = torch.clamp(f_t, min=1e-15)
    f_ratio = safe_f / f_ref
    spectral_power = torch.pow(f_ratio, 2.0 - gam)

    prefactor = (amp * amp) / (12.0 * (PI ** 2.0) * (f_ref ** 3.0))
    return prefactor * spectral_power


class PTANetwork(nn.Module):
    """Pulsar Timing Array (PTA) Network for Nanohertz Gravitational-Wave Background Detection."""

    def __init__(
        self,
        pulsar_names: Sequence[str],
        ra: Union[Sequence[float], torch.Tensor],
        dec: Union[Sequence[float], torch.Tensor],
        noise_std: Optional[Union[Sequence[float], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.pulsar_names = list(pulsar_names)
        self.n_pulsars = len(self.pulsar_names)

        ra_t = as_tensor(ra, device=device, dtype=dtype)
        dec_t = as_tensor(dec, device=device, dtype=dtype)

        self.register_buffer("ra", ra_t, persistent=True)
        self.register_buffer("dec", dec_t, persistent=True)

        # Precompute (N_p, 3) line-of-sight unit vectors
        pos = pulsar_line_of_sight(ra_t, dec_t)
        self.register_buffer("positions", pos, persistent=True)

        # Precompute (N_p, N_p) pairwise angular separations and Hellings-Downs correlation matrix
        pos_i = pos.unsqueeze(1)  # (N_p, 1, 3)
        pos_j = pos.unsqueeze(0)  # (1, N_p, 3)
        _, cos_zeta = angular_separation(pos_i, pos_j)
        zeta_matrix = torch.acos(torch.clamp(cos_zeta, -1.0, 1.0))
        self.register_buffer("zeta_matrix", zeta_matrix, persistent=True)

        identity = torch.eye(self.n_pulsars, device=device, dtype=dtype)
        hd_matrix = hellings_downs_correlation(zeta_matrix, is_auto=identity, include_pulsar_term=True)
        self.register_buffer("hd_matrix", hd_matrix, persistent=True)

        if noise_std is None:
            noise_t = torch.ones(self.n_pulsars, device=device, dtype=dtype) * 1e-7
        else:
            noise_t = as_tensor(noise_std, device=device, dtype=dtype)
        self.register_buffer("noise_std", noise_t, persistent=True)

    def covariance_matrix(
        self,
        a_gwb: Union[float, torch.Tensor],
        gamma: Union[float, torch.Tensor] = 13.0 / 3.0,
        t_obs: float = 10.0 * SEC_PER_YEAR,
    ) -> torch.Tensor:
        """Compute the total timing residual covariance matrix C = N + S(A_GWB, gamma).

        Args:
            a_gwb: GWB characteristic amplitude.
            gamma: Spectral slope (default 13/3).
            t_obs: Total observation duration in seconds (default 10 years).

        Returns:
            (N_p, N_p) symmetric positive-definite covariance matrix.
        """
        device = self.positions.device
        dtype = self.positions.dtype

        amp = as_tensor(a_gwb, device=device, dtype=dtype)
        gam = as_tensor(gamma, device=device, dtype=dtype)

        # White noise diagonal matrix N
        white_noise_var = self.noise_std * self.noise_std
        noise_matrix = torch.diag(white_noise_var)

        # GWB variance scaling in timing residuals (sigma_GWB^2 ~ S_h * f_0 * T_obs)
        # At fundamental frequency 1 / T_obs
        f0 = 1.0 / t_obs
        s_h0 = stochastic_gwb_spectrum(torch.tensor(f0, device=device, dtype=dtype), amp, gam)
        # Timing residual variance: sigma_t^2 = S_h / (12 * pi^2 * f0) or proportional scaling
        sigma_gwb_sq = s_h0 * f0

        # GWB spatial covariance S = Gamma_ab * sigma_gwb^2
        signal_matrix = self.hd_matrix * sigma_gwb_sq

        total_cov = noise_matrix + signal_matrix
        return total_cov

    def log_likelihood(
        self,
        residuals: torch.Tensor,
        a_gwb: Union[float, torch.Tensor],
        gamma: Union[float, torch.Tensor] = 13.0 / 3.0,
        t_obs: float = 10.0 * SEC_PER_YEAR,
    ) -> torch.Tensor:
        """Compute Gaussian marginal log-likelihood of timing residuals given GWB parameters.

        ln L = -1/2 * [ r^T * C^(-1) * r + ln det(C) + N_p * ln(2*pi) ]

        Args:
            residuals: Timing residual vector r of shape (N_p,) or batched (B, N_p).
            a_gwb: GWB amplitude parameter.
            gamma: GWB spectral slope index.
            t_obs: Observation duration in seconds.

        Returns:
            Log-likelihood tensor (scalar or batched).
        """
        r = as_tensor(residuals, device=self.positions.device, dtype=self.positions.dtype)
        cov = self.covariance_matrix(a_gwb, gamma=gamma, t_obs=t_obs)

        # Cholesky decomposition for positive definiteness and autograd stability
        # C = L * L^T
        chol = torch.linalg.cholesky(cov)

        # Solve L * y = r  =>  r^T * C^(-1) * r = ||y||^2
        if r.dim() == 1:
            y = torch.linalg.solve_triangular(chol, r.unsqueeze(-1), upper=False).squeeze(-1)
            quad_form = torch.sum(y * y)
        else:
            # Batched (B, N_p)
            y = torch.linalg.solve_triangular(chol.unsqueeze(0), r.unsqueeze(-1), upper=False).squeeze(-1)
            quad_form = torch.sum(y * y, dim=-1)

        # Log-determinant: ln det(C) = 2 * sum(ln(diag(L)))
        log_det = 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)))

        n_p = float(self.n_pulsars)
        const_term = n_p * math.log(2.0 * PI)

        ln_l = -0.5 * (quad_form + log_det + const_term)
        return ln_l

    def forward(self, *args, **kwargs):
        return self.log_likelihood(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
