"""Non-Gaussian Glitch Model & Differentiable Wavelet Likelihood (BayesWave-style).

Strict Clean-Room Implementation based exclusively on published peer-reviewed literature:
- Cornish & Littenberg (2015), Class. Quantum Grav. 32, 135012 (arXiv:1410.3838)
  [BayesWave: Bayesian inference for gravitational wave bursts and instrument glitches]
- Chatziioannou et al. (2021), Phys. Rev. D 104, 044015 (arXiv:2105.01521)
  [Modeling transient noise in gravitational wave detectors]
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn

from torchwave.constants import PI
from torchwave.conversions import as_tensor
from torchwave.inference.likelihood import noise_weighted_inner_product


def sine_gaussian_wavelet_td(
    t: torch.Tensor,
    t0: Union[float, torch.Tensor],
    f0: Union[float, torch.Tensor],
    q: Union[float, torch.Tensor],
    amp: Union[float, torch.Tensor],
    phi0: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """Compute time-domain Morlet / Sine-Gaussian wavelet Psi(t).

    Formula:
        tau = Q / (2 * pi * f_0)
        Psi(t) = A * exp( -(t - t_0)^2 / tau^2 ) * cos( 2 * pi * f_0 * (t - t_0) + phi_0 )

    References:
        Cornish & Littenberg (2015) Eq. (7); Chatziioannou et al. (2021) Eq. (1)

    Args:
        t: Time evaluation grid in seconds, shape (..., N_samples)
        t0: Central time in seconds
        f0: Central frequency in Hz
        q: Dimensionless quality factor Q
        amp: Peak strain amplitude A
        phi0: Reference phase in radians

    Returns:
        psi_td: Time-domain wavelet strain
    """
    device = t.device
    dtype = t.dtype if t.is_floating_point() else torch.float64

    t0_t = as_tensor(t0, device=device, dtype=dtype)
    f0_t = as_tensor(f0, device=device, dtype=dtype)
    q_t = as_tensor(q, device=device, dtype=dtype)
    amp_t = as_tensor(amp, device=device, dtype=dtype)
    phi0_t = as_tensor(phi0, device=device, dtype=dtype)

    # Characteristic duration tau = Q / (2 * pi * f_0)
    tau = q_t / (2.0 * PI * f0_t)

    # Align broadcasting dimensions
    if t.dim() > t0_t.dim():
        t0_t = t0_t.unsqueeze(-1)
        f0_t = f0_t.unsqueeze(-1)
        tau = tau.unsqueeze(-1)
        amp_t = amp_t.unsqueeze(-1)
        phi0_t = phi0_t.unsqueeze(-1)

    dt = t - t0_t
    envelope = torch.exp(- (dt * dt) / (tau * tau))
    oscillation = torch.cos(2.0 * PI * f0_t * dt + phi0_t)
    return amp_t * envelope * oscillation


def sine_gaussian_wavelet_fd(
    f: torch.Tensor,
    t0: Union[float, torch.Tensor],
    f0: Union[float, torch.Tensor],
    q: Union[float, torch.Tensor],
    amp: Union[float, torch.Tensor],
    phi0: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """Compute analytical Fourier transform of sine-Gaussian wavelet tilde{Psi}(f).

    Formula:
        tau = Q / (2 * pi * f_0)
        tilde{Psi}(f) = (A * tau * sqrt(pi) / 2) *
            [ exp(i * phi_0) * exp( -pi^2 * tau^2 * (f - f_0)^2 ) +
              exp(-i * phi_0) * exp( -pi^2 * tau^2 * (f + f_0)^2 ) ] *
            exp( -2 * pi * i * f * t_0 )

    References:
        Cornish & Littenberg (2015) Eq. (8); Chatziioannou et al. (2021) Eq. (2)

    Args:
        f: Frequency evaluation grid in Hz, shape (..., N_freqs)
        t0: Central time in seconds
        f0: Central frequency in Hz
        q: Dimensionless quality factor Q
        amp: Peak strain amplitude A
        phi0: Reference phase in radians

    Returns:
        psi_fd: Complex frequency-domain wavelet
    """
    device = f.device
    dtype = f.dtype if f.is_floating_point() else torch.float64

    t0_t = as_tensor(t0, device=device, dtype=dtype)
    f0_t = as_tensor(f0, device=device, dtype=dtype)
    q_t = as_tensor(q, device=device, dtype=dtype)
    amp_t = as_tensor(amp, device=device, dtype=dtype)
    phi0_t = as_tensor(phi0, device=device, dtype=dtype)

    tau = q_t / (2.0 * PI * f0_t)

    if f.dim() > t0_t.dim():
        t0_t = t0_t.unsqueeze(-1)
        f0_t = f0_t.unsqueeze(-1)
        tau = tau.unsqueeze(-1)
        amp_t = amp_t.unsqueeze(-1)
        phi0_t = phi0_t.unsqueeze(-1)

    norm = 0.5 * amp_t * tau * math.sqrt(PI)

    pi_tau = PI * tau
    pi2_tau2 = pi_tau * pi_tau

    df_pos = f - f0_t
    df_neg = f + f0_t

    gauss_pos = torch.exp(-pi2_tau2 * (df_pos * df_pos))
    gauss_neg = torch.exp(-pi2_tau2 * (df_neg * df_neg))

    phase_pos = torch.complex(torch.cos(phi0_t), torch.sin(phi0_t))
    phase_neg = torch.complex(torch.cos(phi0_t), -torch.sin(phi0_t))

    bracket = phase_pos * gauss_pos + phase_neg * gauss_neg

    t0_phase = -2.0 * PI * f * t0_t
    t0_shift = torch.complex(torch.cos(t0_phase), torch.sin(t0_phase))

    psi_fd = norm.to(bracket.dtype) * bracket * t0_shift
    return psi_fd


class SineGaussianWavelet(nn.Module):
    """Single learnable Sine-Gaussian / Morlet wavelet with log-scale amplitude parameterization."""

    def __init__(
        self,
        t0: float = 0.0,
        f0: float = 100.0,
        q: float = 8.0,
        amp: float = 1e-21,
        phi0: float = 0.0,
    ):
        super().__init__()
        self.t0 = nn.Parameter(torch.tensor(t0, dtype=torch.float64))
        self.f0 = nn.Parameter(torch.tensor(f0, dtype=torch.float64))
        self.q = nn.Parameter(torch.tensor(q, dtype=torch.float64))
        safe_amp = max(abs(amp), 1e-50)
        self.log_amp = nn.Parameter(torch.tensor(math.log(safe_amp), dtype=torch.float64))
        self.phi0 = nn.Parameter(torch.tensor(phi0, dtype=torch.float64))

    @property
    def amp(self) -> torch.Tensor:
        return torch.exp(self.log_amp)

    @amp.setter
    def amp(self, val: Union[float, torch.Tensor]) -> None:
        val_t = as_tensor(val, device=self.log_amp.device, dtype=self.log_amp.dtype)
        self.log_amp.data = torch.log(torch.clamp(val_t.abs(), min=1e-50))

    def time_domain(self, t: torch.Tensor) -> torch.Tensor:
        return sine_gaussian_wavelet_td(
            t=t,
            t0=self.t0,
            f0=torch.clamp(self.f0.abs(), min=5.0),
            q=torch.clamp(self.q.abs(), min=2.0),
            amp=self.amp,
            phi0=self.phi0,
        )

    def frequency_domain(self, f: torch.Tensor) -> torch.Tensor:
        return sine_gaussian_wavelet_fd(
            f=f,
            t0=self.t0,
            f0=torch.clamp(self.f0.abs(), min=5.0),
            q=torch.clamp(self.q.abs(), min=2.0),
            amp=self.amp,
            phi0=self.phi0,
        )


class WaveletGlitchModel(nn.Module):
    """Multi-wavelet Non-Gaussian Glitch Model g(f) = sum_k tilde{Psi}_k(f)."""

    def __init__(self, num_wavelets: int = 1):
        super().__init__()
        self.num_wavelets = num_wavelets
        self.wavelets = nn.ModuleList([SineGaussianWavelet() for _ in range(num_wavelets)])

    def forward_fd(self, f: torch.Tensor) -> torch.Tensor:
        """Compute total glitch spectrum g(f) in frequency domain."""
        glitch = torch.zeros(f.shape, dtype=torch.complex128, device=f.device)
        for w in self.wavelets:
            glitch = glitch + w.frequency_domain(f)
        return glitch

    def forward_td(self, t: torch.Tensor) -> torch.Tensor:
        """Compute total glitch time series g(t) in time domain."""
        glitch = torch.zeros(t.shape, dtype=t.dtype, device=t.device)
        for w in self.wavelets:
            glitch = glitch + w.time_domain(t)
        return glitch

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.forward_fd(f)


class WaveletGlitchLikelihood(nn.Module):
    """Differentiable Joint Residual Likelihood separating signal h(f) and glitch g_I(f).

    Likelihood formula:
        ln L = -1/2 * sum_I <d_I - h_I - g_I | d_I - h_I - g_I>_{S_n^(I)}
    """

    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        psds: Dict[str, torch.Tensor],
        frequencies: torch.Tensor,
        df: Optional[float] = None,
    ):
        super().__init__()
        self.detectors = sorted(list(data.keys()))
        self.register_buffer("frequencies", as_tensor(frequencies))
        self.df = df if df is not None else float((frequencies[1] - frequencies[0]).item())

        for det in self.detectors:
            self.register_buffer(f"data_{det}", as_tensor(data[det]))
            self.register_buffer(f"psd_{det}", as_tensor(psds[det]))

    def log_likelihood(
        self,
        h_coherent: Dict[str, torch.Tensor],
        glitches: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute the joint residual log-likelihood ln L."""
        total_chi2 = torch.tensor(0.0, device=self.frequencies.device, dtype=torch.float64)

        for det in self.detectors:
            d = getattr(self, f"data_{det}")
            psd = getattr(self, f"psd_{det}")
            h = h_coherent.get(det, torch.zeros_like(d))
            g = glitches.get(det, torch.zeros_like(d)) if glitches is not None else torch.zeros_like(d)

            residual = d - h - g
            # <r | r>_Sn = 4 * sum Re(r * r* / Sn) * df
            inn = noise_weighted_inner_product(residual, residual, psd=psd, df=self.df)
            total_chi2 = total_chi2 + inn

        return -0.5 * total_chi2

    def snr_diagnostics(
        self,
        h_coherent: Dict[str, torch.Tensor],
        glitches: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """Compute signal, glitch, and residual SNR diagnostics."""
        results = {}
        for det in self.detectors:
            d = getattr(self, f"data_{det}")
            psd = getattr(self, f"psd_{det}")
            h = h_coherent.get(det, torch.zeros_like(d))
            g = glitches.get(det, torch.zeros_like(d)) if glitches is not None else torch.zeros_like(d)

            h_snr = math.sqrt(max(0.0, noise_weighted_inner_product(h, h, psd=psd, df=self.df).item()))
            g_snr = math.sqrt(max(0.0, noise_weighted_inner_product(g, g, psd=psd, df=self.df).item()))
            res = d - h - g
            res_snr = math.sqrt(max(0.0, noise_weighted_inner_product(res, res, psd=psd, df=self.df).item()))

            results[f"{det}_signal_snr"] = h_snr
            results[f"{det}_glitch_snr"] = g_snr
            results[f"{det}_residual_snr"] = res_snr
        return results


def fit_glitch_to_residual(
    residual: torch.Tensor,
    psd: torch.Tensor,
    frequencies: torch.Tensor,
    t0_init: float = 0.0,
    f0_init: float = 100.0,
    q_init: float = 8.0,
    amp_init: Optional[float] = None,
    phi0_init: float = 0.0,
    num_steps: int = 150,
    lr: float = 1e-2,
) -> Tuple[SineGaussianWavelet, List[float]]:
    """Perform gradient-based optimization to fit a Sine-Gaussian wavelet glitch to residual data.

    Utilizes analytical matched-filter amplitude initialization and log-scale amplitude parameterization.

    Returns:
        (fitted_wavelet, loss_history)
    """
    device = residual.device
    df = float((frequencies[1] - frequencies[0]).item())

    # Analytical matched-filter amplitude estimation for initialization
    if amp_init is None:
        unit_wave = sine_gaussian_wavelet_fd(frequencies, t0=t0_init, f0=f0_init, q=q_init, amp=1.0, phi0=phi0_init)
        top = noise_weighted_inner_product(residual, unit_wave, psd=psd, df=df)
        bot = noise_weighted_inner_product(unit_wave, unit_wave, psd=psd, df=df)
        amp_val = max(float(torch.abs(top / torch.clamp(bot, min=1e-50)).item()), 1e-25)
    else:
        amp_val = amp_init

    wavelet = SineGaussianWavelet(
        t0=t0_init,
        f0=f0_init,
        q=q_init,
        amp=amp_val,
        phi0=phi0_init,
    ).to(device)

    # Parameter groups for stable multi-scale optimization
    optimizer = torch.optim.Adam([
        {"params": [wavelet.t0], "lr": lr * 0.01},
        {"params": [wavelet.f0], "lr": lr * 2.0},
        {"params": [wavelet.q], "lr": lr},
        {"params": [wavelet.log_amp], "lr": lr},
        {"params": [wavelet.phi0], "lr": lr},
    ])
    losses = []

    for _ in range(num_steps):
        optimizer.zero_grad()
        g = wavelet.frequency_domain(frequencies)
        diff = residual - g
        loss = noise_weighted_inner_product(diff, diff, psd=psd, df=df)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return wavelet, losses

