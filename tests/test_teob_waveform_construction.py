"""Independent checks of TEOB mode projection and Fourier construction."""

import math

import pytest
import torch

from torchwave.approximants.teobresums import (
    TEOBResumS, TEOBResumSHamiltonian, _piecewise_linear_fourier,
)
from torchwave.constants import C_SI, MSUN_SEC, MPC_SI
from torchwave.approximants.eob.radiation_reaction import EOBRadiationReaction


def test_circular_polarization_quadrature():
    """The azimuth-zero spin -2 quadrature has the stated inclination factors."""
    n = 128
    phase = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64, requires_grad=True)
    amp = torch.ones_like(phase)
    cos_i = torch.tensor(0.37, dtype=torch.float64)
    h22 = torch.complex(amp * torch.cos(phase), -amp * torch.sin(phase))
    hp = math.sqrt(5 / (16 * math.pi)) * (1 + cos_i.square()) * h22.real
    hc = -math.sqrt(5 / (4 * math.pi)) * cos_i * h22.imag
    assert torch.allclose(hp, math.sqrt(5 / (16 * math.pi)) * (1 + cos_i.square()) * torch.cos(phase))
    assert torch.allclose(hc, math.sqrt(5 / (4 * math.pi)) * cos_i * torch.sin(phase))
    (hp.square().mean() + hc.square().mean()).backward()
    assert phase.grad is not None and torch.isfinite(phase.grad).all()


def test_rfft_frequency_sign_and_dt_scaling():
    dt = 0.01
    n = 256
    f0 = 7.0
    t = torch.arange(n, dtype=torch.float64) * dt
    x = torch.cos(2 * math.pi * f0 * t)
    spec = torch.fft.rfft(x) * dt
    freqs = torch.fft.rfftfreq(n, dt)
    k = torch.argmin((spec.abs() - spec.abs().max()).abs())
    assert abs(freqs[k].item() - f0) <= freqs[1].item()
    # A positive time delay multiplies this convention by exp(-2 pi i f tc).
    tc = 0.013
    shifted = spec * torch.exp(torch.complex(torch.tensor(0.0), -2 * math.pi * freqs * tc))
    assert torch.allclose(shifted.abs(), spec.abs(), atol=1e-12)


def test_mode_phase_and_batch_gradients():
    f = torch.linspace(30.0, 80.0, 12, dtype=torch.float64)
    masses = torch.tensor([1.3, 1.4], dtype=torch.float64, requires_grad=True)
    hp, hc = TEOBResumS()(f, masses, masses + 0.1, inclination=torch.tensor([0.0, 0.8], dtype=torch.float64),
                          phic=torch.tensor([0.1, 0.2], dtype=torch.float64), num_steps=12)
    assert hp.shape == (2, f.numel()) and hc.shape == hp.shape
    signal_scale = torch.maximum(hp.detach().abs().max(), hc.detach().abs().max())
    loss = (hp.abs().square().sum() + hc.abs().square().sum()) / signal_scale.square()
    loss.backward()
    assert masses.grad is not None and torch.isfinite(masses.grad).all()
    assert masses.grad.norm() / masses.detach().norm() > 1e-8


def test_hamiltonian_frequency_is_used():
    eta = torch.tensor(0.24, dtype=torch.float64)
    ham = TEOBResumSHamiltonian(eta, torch.tensor(0.3, dtype=torch.float64), torch.tensor(-0.2, dtype=torch.float64),
                                torch.tensor(0.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    rr = EOBRadiationReaction(eta, ham.chi1, ham.chi2)
    r = torch.tensor(5.0, dtype=torch.float64)
    pr = torch.tensor(0.02, dtype=torch.float64)
    pphi = torch.tensor(3.0, dtype=torch.float64)
    _, omega, _, _ = ham.equations_of_motion(r, pr, pphi, rr)
    assert torch.isfinite(omega)
    assert not torch.allclose(omega, pphi / r.square())
    # The emitted frequency must be the Hamiltonian's angular velocity,
    # including the spin-dependent centrifugal-radius terms.
    assert torch.isfinite(omega) and abs(omega - pphi / r.square()) > 1e-6


def test_production_waveform_uses_hamiltonian_omega(monkeypatch):
    """A controlled DC run exposes the production frequency path directly."""
    eta = torch.tensor(1.4 * 1.3 / (1.4 + 1.3) ** 2, dtype=torch.float64)
    chosen_omega = torch.tensor(0.37, dtype=torch.float64)

    def controlled_eom(self, r, pr, pphi, rr):
        return torch.zeros_like(r), chosen_omega.expand_as(r), torch.zeros_like(r), torch.zeros_like(r)

    monkeypatch.setattr(TEOBResumSHamiltonian, "equations_of_motion", controlled_eom)
    hp, hc = TEOBResumS()(torch.tensor([0.0], dtype=torch.float64), 1.4, 1.3,
                          inclination=0.0, distance=100.0, f_lower=30.0, num_steps=0)
    m_sec = (1.4 + 1.3) * MSUN_SEC
    d_sec = (100.0 * MPC_SI) / C_SI
    expected = 4.0 * eta * chosen_omega.pow(2.0 / 3.0) * (m_sec / d_sec) * (0.5 * m_sec)
    expected = torch.complex(expected, torch.zeros_like(expected))
    assert torch.allclose(hp, expected.reshape_as(hp), rtol=0.0, atol=1e-34)
    assert torch.allclose(hc, torch.zeros_like(hc), rtol=0.0, atol=1e-34)


def test_piecewise_linear_fourier_exact_dc_and_shortened_endpoint():
    t = torch.arange(6, dtype=torch.float64)
    dc = _piecewise_linear_fourier(torch.ones_like(t), torch.tensor(4.3),
                                   torch.tensor(1.0), torch.tensor([0.0]))
    linear = _piecewise_linear_fourier(t, torch.tensor(4.3), torch.tensor(1.0),
                                       torch.tensor([0.0]))
    assert torch.allclose(dc.real, torch.tensor([4.3], dtype=torch.float64), atol=1e-13)
    assert torch.allclose(linear.real, torch.tensor([4.3**2 / 2], dtype=torch.float64), atol=1e-13)


def test_piecewise_linear_fourier_sinusoid_is_second_order():
    errors = []
    freq = 0.37
    end = 9.37
    w = 2 * math.pi * freq
    exact = end / (2j) + (1 - torch.exp(torch.tensor(-2j * w * end))) / (4 * w)
    for step in (0.2, 0.1, 0.05):
        t = torch.arange(0, 10 + step / 2, step, dtype=torch.float64)
        signal = torch.sin(w * t)
        result = _piecewise_linear_fourier(signal, torch.tensor(end),
                                           torch.tensor(step), torch.tensor([freq]))[0]
        errors.append((result - exact).abs())
    assert errors[1] / errors[0] < 0.30
    assert errors[2] / errors[1] < 0.30


def test_piecewise_linear_fourier_nonzero_constant_and_linear_integrals():
    step = 0.125
    end = 4.375
    freq = 0.23
    w = 2 * math.pi * freq
    t = torch.arange(0, 40, dtype=torch.float64)
    f = torch.tensor([freq], dtype=torch.float64)
    constant = _piecewise_linear_fourier(torch.ones_like(t), torch.tensor(end),
                                         torch.tensor(step), f)[0]
    linear = _piecewise_linear_fourier(t * step, torch.tensor(end), torch.tensor(step), f)[0]
    phase = torch.exp(torch.complex(torch.tensor(0.0, dtype=torch.float64),
                                    torch.tensor(-w * end, dtype=torch.float64)))
    exact_constant = (1 - phase) / (1j * w)
    exact_linear = (phase * (1 + 1j * w * end) - 1) / w**2
    assert torch.allclose(constant, exact_constant, rtol=0.0, atol=1e-12)
    assert torch.allclose(linear, exact_linear, rtol=0.0, atol=1e-12)


def test_piecewise_linear_fourier_batched_event_edges_and_gradient():
    t = torch.arange(8, dtype=torch.float64)
    signal = torch.stack((t, 2 * t + 1), dim=1).requires_grad_()
    events = torch.tensor([1.4, 3.0], dtype=torch.float64)
    result = _piecewise_linear_fourier(signal, events, torch.ones(2, dtype=torch.float64),
                                       torch.tensor([0.0, 0.3], dtype=torch.float64))
    assert torch.allclose(result[:, 0].real, torch.tensor([1.4**2 / 2, 12.0], dtype=torch.float64), atol=1e-13)
    assert torch.isfinite(result).all()
    result.real.sum().backward()
    assert signal.grad is not None and torch.isfinite(signal.grad).all()
    assert signal.grad.norm() / signal.detach().norm() > 1e-3


def test_piecewise_linear_fourier_mass_time_scaling():
    t = torch.arange(8, dtype=torch.float64)
    signal = torch.cos(0.4 * t)
    f = torch.tensor([0.0, 0.13, 0.4], dtype=torch.float64)
    first = _piecewise_linear_fourier(signal, torch.tensor(5.4), torch.tensor(2.0), f)
    second = _piecewise_linear_fourier(signal, torch.tensor(10.8), torch.tensor(4.0), f / 2)
    assert torch.allclose(second, 2 * first, atol=1e-12, rtol=1e-12)


def test_physical_event_budget_is_explicit():
    f = torch.tensor([40.0, 60.0], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="exhausted its budget"):
        TEOBResumS()(f, 1.4, 1.3, num_steps=2, integration_mode="physical_event")


def test_physical_event_resolves_short_high_mass_run():
    f = torch.linspace(20.0, 100.0, 6, dtype=torch.float64)
    hp, hc = TEOBResumS()(f, 100.0, 100.0, dt_geom=5.0, num_steps=300,
                          integration_mode="physical_event")
    assert hp.shape == f.shape and torch.isfinite(hp).all() and torch.isfinite(hc).all()


def test_batched_physical_events_resolve_independently():
    f = torch.linspace(20.0, 100.0, 6, dtype=torch.float64)
    hp, hc = TEOBResumS()(f, torch.tensor([80.0, 100.0], dtype=torch.float64),
                          torch.tensor([80.0, 100.0], dtype=torch.float64),
                          dt_geom=5.0, num_steps=300,
                          integration_mode="physical_event")
    assert hp.shape == (2, f.numel())
    assert torch.isfinite(hp).all() and torch.isfinite(hc).all()
    for row, mass in enumerate((80.0, 100.0)):
        hp_single, hc_single = TEOBResumS()(f, mass, mass, dt_geom=5.0, num_steps=300,
                                            integration_mode="physical_event")
        assert (hp[row] - hp_single).norm() / hp_single.norm() < 1e-8
        assert (hc[row] - hc_single).norm() / hc_single.norm() < 1e-8


def test_actual_teob_time_delay_and_mode_phase_rotation():
    f = torch.linspace(30.0, 90.0, 9, dtype=torch.float64)
    hp0, hc0 = TEOBResumS()(f, 1.4, 1.3, inclination=0.0, phic=0.0, tc=0.0, num_steps=12)
    hp1, hc1 = TEOBResumS()(f, 1.4, 1.3, inclination=0.0, phic=0.0, tc=0.003, num_steps=12)
    delay = torch.exp(torch.complex(torch.zeros_like(f), -2 * math.pi * f * 0.003))
    assert hp0.abs().max() > 0 and hc0.abs().max() > 0
    assert (hp1 - hp0 * delay).norm() / hp0.norm() < 1e-10
    assert (hc1 - hc0 * delay).norm() / hc0.norm() < 1e-10

    delta = 0.2
    hp_phase, hc_phase = TEOBResumS()(f, 1.4, 1.3, inclination=0.0, phic=delta, tc=0.0, num_steps=12)
    assert (hp_phase - (math.cos(2 * delta) * hp0 - math.sin(2 * delta) * hc0)).norm() / hp0.norm() < 1e-10
    assert (hc_phase - (math.sin(2 * delta) * hp0 + math.cos(2 * delta) * hc0)).norm() / hc0.norm() < 1e-10
