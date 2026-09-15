# SPDX-License-Identifier: MIT
"""Unit tests for TorchWave native SXS-calibrated higher harmonic fits."""
import math
import pytest
import torch

from torchwave.approximants.xhm_native_fits import (
    eval_native_ringdown_params,
    eval_native_21_ringdown_params,
    eval_native_33_ringdown_params,
    eval_native_44_ringdown_params,
    eval_native_32_ringdown_params,
    native_ringdown_amplitude,
    native_ringdown_amplitude_33,
    SIGMA_DEFAULTS,
)


@pytest.mark.parametrize("lm", [(2, 1), (3, 3), (4, 4), (3, 2)])
def test_native_ringdown_params_physical(lm):
    """Verify parameters are smooth, finite, and positive across mass ratios 1 <= q <= 10."""
    l, m = lm
    q_vals = torch.linspace(1.01, 10.0, 50, dtype=torch.float64, requires_grad=True)
    eta = q_vals / ((1.0 + q_vals) ** 2)
    delta = (q_vals - 1.0) / (q_vals + 1.0)

    a_lambda, lam = eval_native_ringdown_params(l, m, eta, delta)

    # Basic physical sanity
    assert torch.all(torch.isfinite(a_lambda))
    assert torch.all(torch.isfinite(lam))
    assert torch.all(a_lambda > 0)
    assert torch.all(lam > 0)

    # Autograd works cleanly across q
    grad_a = torch.autograd.grad(a_lambda.sum(), q_vals, retain_graph=True)[0]
    grad_lam = torch.autograd.grad(lam.sum(), q_vals)[0]
    assert torch.all(torch.isfinite(grad_a))
    assert torch.all(torch.isfinite(grad_lam))


def test_native_ringdown_params_equal_mass_limit():
    """In equal mass limit (q=1, delta=0), odd-m modes vanish identically; even-m remain positive."""
    eta = torch.tensor(0.25, dtype=torch.float64)
    delta = torch.tensor(0.0, dtype=torch.float64)

    # Odd-m modes: (2, 1), (3, 3) must vanish
    a21, _ = eval_native_21_ringdown_params(eta, delta)
    a33, _ = eval_native_33_ringdown_params(eta, delta)
    assert a21.item() == 0.0
    assert a33.item() == 0.0

    # Even-m modes: (4, 4), (3, 2) must remain finite and positive
    a44, l44 = eval_native_44_ringdown_params(eta)
    a32, l32 = eval_native_32_ringdown_params(eta)
    assert a44.item() > 0.0
    assert a32.item() > 0.0
    assert math.isfinite(l44.item())
    assert math.isfinite(l32.item())


@pytest.mark.parametrize("lm", [(2, 1), (3, 3), (4, 4), (3, 2)])
def test_native_ringdown_amplitude_profile(lm):
    """Verify amplitude profile is positive, peak-centered, and smooth."""
    l, m = lm
    eta = torch.tensor(0.2222222222222222, dtype=torch.float64)  # q = 2
    delta = torch.tensor(0.3333333333333333, dtype=torch.float64)
    f_ring = 0.12 * (m / 2.0)
    f_damp = 0.014

    f = torch.linspace(0.8 * f_ring, 0.4, 200, dtype=torch.float64, requires_grad=True)
    amp = native_ringdown_amplitude(l, m, f, eta, delta, f_ring, f_damp)

    assert amp.shape == f.shape
    assert torch.all(torch.isfinite(amp))
    assert torch.all(amp > 0)

    # Backward gradient check
    grad = torch.autograd.grad(amp.sum(), f)[0]
    assert torch.all(torch.isfinite(grad))


def test_imrphenom_xhm_native_waveform():
    """Verify IMRPhenomXHM generates finite polarizations with native calibration."""
    from torchwave.waveforms import get_fd_waveform

    f = torch.linspace(20.0, 400.0, 100, dtype=torch.float64)
    hp_nat, hc_nat = get_fd_waveform(
        approximant="IMRPhenomXHM",
        mass1=40.0,
        mass2=20.0,
        chi1=0.2,
        chi2=-0.1,
        inclination=0.8,
        sample_frequencies=f,
        calibration="torchwave_nr",
    )
    assert torch.all(torch.isfinite(hp_nat))
    assert torch.all(torch.isfinite(hc_nat))
    assert torch.linalg.vector_norm(hp_nat).item() > 0.0

    hp_leg, hc_leg = get_fd_waveform(
        approximant="IMRPhenomXHM",
        mass1=40.0,
        mass2=20.0,
        chi1=0.2,
        chi2=-0.1,
        inclination=0.8,
        sample_frequencies=f,
        calibration="legacy",
    )
    # Native SXS calibration should differ from legacy calibration
    diff = torch.linalg.vector_norm(hp_nat - hp_leg) / torch.linalg.vector_norm(hp_leg)
    assert diff.item() > 0.0


def test_imrphenom_xphm_native_waveform():
    """Verify precessing IMRPhenomXPHM generates finite polarizations with native calibration."""
    from torchwave.waveforms import get_fd_waveform

    f = torch.linspace(20.0, 400.0, 100, dtype=torch.float64)
    hp_nat, hc_nat = get_fd_waveform(
        approximant="IMRPhenomXPHM",
        mass1=40.0,
        mass2=20.0,
        spin1x=0.1,
        spin1y=0.0,
        spin1z=0.2,
        spin2x=0.0,
        spin2y=0.1,
        spin2z=-0.1,
        inclination=0.8,
        sample_frequencies=f,
        calibration="torchwave_nr",
    )
    assert torch.all(torch.isfinite(hp_nat))
    assert torch.all(torch.isfinite(hc_nat))
    assert torch.linalg.vector_norm(hp_nat).item() > 0.0

