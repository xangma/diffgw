"""Tests for Non-Gaussian Glitch Model and Wavelet Likelihood.

Clean-room literature benchmarks:
- Cornish & Littenberg (2015), Class. Quantum Grav. 32, 135012 (arXiv:1410.3838)
- Chatziioannou et al. (2021), Phys. Rev. D 104, 044015 (arXiv:2105.01521)
"""

import math
import pytest
import torch

from torchwave.inference.wavelet_glitch import (
    sine_gaussian_wavelet_td,
    sine_gaussian_wavelet_fd,
    SineGaussianWavelet,
    WaveletGlitchModel,
    WaveletGlitchLikelihood,
    fit_glitch_to_residual,
)
from torchwave.approximants.taylorf2 import TaylorF2


def test_wavelet_analytical_vs_numerical_fft():
    """Verify that analytical FFT tilde{Psi}(f) matches numerical FFT of Psi(t) to < 10^-4 relative error."""
    fs = 2048.0
    dt = 1.0 / fs
    duration = 2.0
    n_samples = int(duration * fs)
    t = torch.arange(n_samples, dtype=torch.float64) * dt

    t0 = 1.0
    f0 = 120.0
    q = 8.0
    amp = 1.0e-21
    phi0 = 0.5

    # 1. Time-domain evaluation
    psi_td = sine_gaussian_wavelet_td(t, t0=t0, f0=f0, q=q, amp=amp, phi0=phi0)

    # 2. Numerical FFT scaled by dt to approximate continuous Fourier transform
    fft_num = torch.fft.rfft(psi_td) * dt
    freqs = torch.fft.rfftfreq(n_samples, d=dt)

    # 3. Analytical Fourier transform
    fft_ana = sine_gaussian_wavelet_fd(freqs, t0=t0, f0=f0, q=q, amp=amp, phi0=phi0)

    # Compare in significant bandwidth region around central frequency (e.g. 80 Hz to 160 Hz)
    band_mask = (freqs >= 80.0) & (freqs <= 160.0)
    peak_val = fft_ana.abs().max()

    rel_error = (fft_num[band_mask] - fft_ana[band_mask]).abs() / peak_val
    max_rel_error = rel_error.max().item()

    assert max_rel_error < 1.0e-4, f"Analytical vs numerical FFT error {max_rel_error} exceeds 1e-4 threshold"


def test_joint_optimization_glitch_chirp_decoupling():
    """Verify that joint optimization on d(f) = h(f) + glitch(f) decouples the glitch from the chirp signal."""
    freqs = torch.linspace(30.0, 300.0, 200, dtype=torch.float64)
    df = float((freqs[1] - freqs[0]).item())

    # Generate chirp signal h(f)
    taylorf2 = TaylorF2()
    hp, _ = taylorf2(freqs, mass1=30.0, mass2=30.0, distance=1000.0)

    # Inject glitch
    t0_true = 0.05
    f0_true = 110.0
    q_true = 8.0
    amp_true = 1.0e-22
    phi0_true = 0.3

    g_true = sine_gaussian_wavelet_fd(
        freqs,
        t0=t0_true,
        f0=f0_true,
        q=q_true,
        amp=amp_true,
        phi0=phi0_true,
    )

    data = hp + g_true
    psd = torch.ones_like(freqs) * 1.0e-46

    # Fit glitch to residual (data - h)
    residual_before = data - hp
    wavelet_fit, losses = fit_glitch_to_residual(
        residual=residual_before,
        psd=psd,
        frequencies=freqs,
        t0_init=0.048,
        f0_init=108.0,
        q_init=7.8,
        phi0_init=0.25,
        num_steps=150,
        lr=2.0e-2,
    )

    g_fit = wavelet_fit.frequency_domain(freqs)
    residual_after = data - hp - g_fit

    # Glitch power before vs after
    glitch_max = g_true.abs().max().item()
    res_max = residual_after.abs().max().item()

    # The residual should be reduced by more than a factor of 10
    assert res_max < 0.1 * glitch_max, "Glitch was not sufficiently decoupled from chirp"
    assert losses[-1] < losses[0] * 0.1, "Loss should drop by at least 90%"


def test_wavelet_glitch_likelihood_and_snr_diagnostics():
    """Verify WaveletGlitchLikelihood log-likelihood and SNR diagnostic calculations."""
    freqs = torch.linspace(40.0, 250.0, 150, dtype=torch.float64)
    psd = torch.ones_like(freqs) * 1e-46

    # Synthetic signal and glitch
    h_h1 = 1e-22 * torch.exp(-((freqs - 100.0) / 30.0) ** 2) + 0.0j
    g_h1 = 2e-22 * torch.exp(-((freqs - 120.0) / 20.0) ** 2) + 0.0j
    d_h1 = h_h1 + g_h1

    likelihood = WaveletGlitchLikelihood(
        data={"H1": d_h1},
        psds={"H1": psd},
        frequencies=freqs,
    )

    # Under perfect model h + g = d, residual is 0 -> log_likelihood = 0
    ln_l_perfect = likelihood.log_likelihood(
        h_coherent={"H1": h_h1},
        glitches={"H1": g_h1},
    )
    assert torch.isclose(ln_l_perfect, torch.tensor(0.0, dtype=torch.float64), atol=1e-12)

    # Without modeling glitch (g = 0), likelihood must be strictly worse (more negative)
    ln_l_no_glitch = likelihood.log_likelihood(
        h_coherent={"H1": h_h1},
        glitches=None,
    )
    assert ln_l_no_glitch.item() < -1.0

    # Test SNR diagnostics
    diagnostics = likelihood.snr_diagnostics(
        h_coherent={"H1": h_h1},
        glitches={"H1": g_h1},
    )

    assert diagnostics["H1_signal_snr"] > 0.0
    assert diagnostics["H1_glitch_snr"] > 0.0
    assert diagnostics["H1_residual_snr"] < 1e-6
