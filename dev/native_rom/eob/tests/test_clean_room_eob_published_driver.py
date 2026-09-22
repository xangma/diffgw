"""Continuous-frequency derivatives and physical units of the public driver."""

from types import SimpleNamespace
import math

import pytest
import torch

from torchwave.constants import C_SI, MPC_SI, MSUN_SEC
from dev.native_rom.eob.torch import published_waveform as driver
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.eob.seobnrv4_mode22 import seobnrv4_published_delta22, seobnrv4_published_mode22
from dev.native_rom.eob.dispatch import torch_get_fd_waveform as get_fd_waveform


PHASE = dict(spin_v6='bohe_total_energy', include_pan_v8=False, include_positive_chi_540=False)
CHOICES = dict(**PHASE, spin_truncation='tabulated', final_spin_fit_order=(1, 2))


def setup_state():
    nu = torch.tensor([[.24], [.2]], dtype=torch.float64)
    ham = SEOBNRv4Hamiltonian(nu, nu*0+.1, nu*0-.2)
    state = torch.tensor([[8., .2, -.02, 3.5], [7.9, .24, -.021, 3.49]], dtype=torch.float64)
    return state.expand(2, 3, 2, 4), ham, SEOBNRv4PublishedFlux(spin_truncation='tabulated')


def direct_mode(state, ham, rr):
    quantities = rr.from_hamiltonian(state, ham)
    residual = seobnrv4_published_delta22(
        **{key: quantities[key] for key in ('omega', 'hreal_over_M', 'x1', 'x2', 'chi1', 'chi2')}, **PHASE)
    return seobnrv4_published_mode22(phi=state[..., 1], residual_phase=residual, **quantities)


def test_frequency_and_mode_retain_state_and_spin_gradients():
    nu = torch.tensor(.24, dtype=torch.float64)
    state = torch.tensor([[8., .2, -.02, 3.5]], dtype=torch.float64, requires_grad=True)
    spin = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    rr = SEOBNRv4PublishedFlux(spin_truncation='tabulated')
    def evaluate(s, a):
        ham = SEOBNRv4Hamiltonian(nu, a, a*0-.1)
        mode, frequency = driver.published_mode22_and_frequency(s, ham, rr, **PHASE)
        return torch.cat((torch.view_as_real(mode).flatten(), frequency.flatten()))
    assert torch.autograd.gradcheck(evaluate, (state, spin), atol=3e-6, rtol=3e-5)


def test_public_fd_driver_projection_units_time_origin_and_start_frequency(monkeypatch):
    calls = []
    dt, count, peak, match = .25, 128, 3.4, 10.7
    time = torch.arange(count, dtype=torch.float64)*dt
    h22 = torch.exp(-.008*time)*torch.exp(-.6j*time)
    def fixed_reconstruction(omega_start, mass1, mass2, *spins, **kwargs):
        calls.append((omega_start, kwargs))
        return SimpleNamespace(
            time=time, mode=h22,
            trajectory=SimpleNamespace(peak=SimpleNamespace(time=torch.tensor(peak, dtype=time.dtype))),
            attachment=SimpleNamespace(match_time=torch.tensor(match, dtype=time.dtype)))
    monkeypatch.setattr(driver, 'reconstruct_seobnrv4_22', fixed_reconstruction)
    total_mass, distance, inclination, phase, delay = 50., 230., .8, .23, .002
    dt_sec = dt*total_mass*MSUN_SEC
    f = torch.arange(1, 8, dtype=time.dtype)/(512*dt_sec)
    hp, hc = get_fd_waveform('SEOBNRv4Published', mass1=30., mass2=20., sample_frequencies=f,
        f_lower=22., distance=distance, inclination=inclination, phic=phase, tc=delay,
        dt_geom=dt, **CHOICES)
    # Direct finite Fourier sum in seconds, independently of the FFT helper.
    physical_mode = h22*torch.exp(torch.tensor(-2j*phase))*total_mass*MSUN_SEC*C_SI/(distance*MPC_SI)
    ci = math.cos(inclination)
    plus = math.sqrt(5/(16*math.pi))*(1+ci*ci)*physical_mode.real
    cross = -math.sqrt(5/(4*math.pi))*ci*physical_mode.imag
    absolute_time = (time-peak)*total_mass*MSUN_SEC+delay
    kernel = torch.exp(-2j*math.pi*f[:, None]*absolute_time)
    torch.testing.assert_close(hp, (kernel*plus).sum(-1)*dt_sec, rtol=1e-7, atol=1e-30)
    torch.testing.assert_close(hc, (kernel*cross).sum(-1)*dt_sec, rtol=1e-7, atol=1e-30)
    torch.testing.assert_close(calls[0][0], time.new_tensor(math.pi*total_mass*MSUN_SEC*22))
    for name, value in CHOICES.items():
        assert calls[0][1][name] == value


def _geometric_sum(q, count):
    return (1 - q**count) / (1 - q)


def _geometric_first_moment(q, count):
    return q * (1 - count*q**(count - 1) + (count - 1)*q**count) / (1 - q)**2


def test_public_fd_driver_propagates_richardson_choice(monkeypatch):
    seen = {}
    time = torch.arange(16, dtype=torch.float64)*.25
    def fixed(*args, **kwargs):
        seen['value'] = kwargs['richardson_extrapolation']
        time = torch.arange(16, dtype=torch.float64)*.25
        return SimpleNamespace(time=time, mode=torch.ones(16, dtype=torch.complex128),
                               trajectory=SimpleNamespace(peak=SimpleNamespace(time=torch.tensor(1.))),
                               attachment=SimpleNamespace(match_time=torch.tensor(2.)))
    monkeypatch.setattr(driver, 'reconstruct_seobnrv4_22', fixed)
    driver.SEOBNRv4Published()(torch.tensor([20.], dtype=torch.float64), 30., 20.,
        f_lower=22., dt_geom=.25, richardson_extrapolation=False, **CHOICES)
    assert seen['value'] is False


def test_public_driver_requires_model_choices_and_rejects_inplane_spin():
    f = torch.tensor([20., 40.], dtype=torch.float64)
    with pytest.raises(TypeError, match='spin_truncation'):
        get_fd_waveform('SEOBNRv4Published', mass1=30., mass2=20., sample_frequencies=f)
    with pytest.raises(ValueError, match='aligned spins'):
        get_fd_waveform('SEOBNRv4Published', mass1=30., mass2=20., spin1x=.1,
                         sample_frequencies=f, **CHOICES)


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf")])
def test_public_driver_rejects_invalid_explicit_start_frequency(invalid, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid f_lower must be rejected before reconstruction")

    monkeypatch.setattr(driver, "reconstruct_seobnrv4_22", unexpected)
    with pytest.raises(ValueError, match="f_lower must be finite and positive"):
        driver.SEOBNRv4Published()(f=torch.tensor([20.0], dtype=torch.float64),
                                   mass1=30.0, mass2=20.0, f_lower=invalid,
                                   **CHOICES)


@pytest.mark.parametrize('as_list', [False, True])
def test_public_frequency_mask_preserves_frequency_batch_dimensions(monkeypatch, as_list):
    time = torch.arange(128, dtype=torch.float64)*.25
    mode = torch.exp(-.008*time-.6j*time)
    def fixed_reconstruction(omega_start, *args, **kwargs):
        return SimpleNamespace(mode=mode.expand(omega_start.shape+(128,)),
            time=time,
            trajectory=SimpleNamespace(peak=SimpleNamespace(time=torch.zeros_like(omega_start)+3.4)),
            attachment=SimpleNamespace(match_time=torch.zeros_like(omega_start)+10.7))
    monkeypatch.setattr(driver, 'reconstruct_seobnrv4_22', fixed_reconstruction)
    f = torch.tensor([[0., 20., 100., 200.], [0., 40., 150., 300.],
                      [0., 60., 200., 400.]], dtype=time.dtype)
    params = dict(mass1=torch.tensor([[30.], [40.]], dtype=time.dtype),
                  mass2=torch.tensor([[15., 20., 25.]], dtype=time.dtype),
                  f_lower=30., inclination=.8, dt_geom=.25, **CHOICES)
    direct = driver.SEOBNRv4Published()(f=f, **params)
    public = get_fd_waveform('SEOBNRv4Published', sample_frequencies=f.tolist() if as_list else f,
                             f_final=300., **params)
    mask = (f < 30) | (f >= 300)
    for actual, reference in zip(public, direct):
        assert actual.shape == (2, 3, 4)
        torch.testing.assert_close(actual, torch.where(mask, 0, reference))


@pytest.mark.parametrize('change', [dict(spin_v6='unknown'), dict(include_pan_v8=1),
                                  dict(final_spin_fit_order=(2, 2))])
def test_invalid_choices_fail_before_trajectory(change, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('initial conditions must not run')
    monkeypatch.setattr(driver, 'find_quasicircular_initial_conditions', unexpected)
    with pytest.raises(ValueError):
        driver.reconstruct_seobnrv4_22(torch.tensor(.03, dtype=torch.float64), 30., 20.,
                                       **(CHOICES | change))
