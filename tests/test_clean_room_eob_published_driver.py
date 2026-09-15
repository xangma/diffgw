"""Continuous-frequency derivatives and physical units of the public driver."""

from types import SimpleNamespace
import math
import numpy as np

import pytest
import torch

from torchwave.constants import C_SI, MPC_SI, MSUN_SEC
from torchwave.approximants.eob import published_waveform as driver
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.eob.seobnrv4_mode22 import seobnrv4_published_delta22, seobnrv4_published_mode22
from torchwave.waveforms import get_fd_waveform
from scripts.verify_eob_published_waveforms import _invert_22, _project_22, _shift_to_peak_origin


PHASE = dict(spin_v6='bohe_total_energy', include_pan_v8=False, include_positive_chi_540=False)
CHOICES = dict(**PHASE, spin_truncation='tabulated', final_spin_fit_order=(1, 2))


def test_paired_22_projection_inversion_keeps_complex_mode_components():
    time = torch.linspace(0., 2., 17, dtype=torch.float64).numpy()
    mode = (0.4 + 0.1*time) * np.exp(1j*(.3 + .7*time))
    plus, cross = _project_22(mode, .8)
    recovered = _invert_22(plus, cross, .8)
    np.testing.assert_allclose(recovered, mode, rtol=1e-13, atol=1e-13)


def test_peak_chart_transform_uses_one_event_origin():
    old = np.array([-2., 0., 3.])
    np.testing.assert_allclose(_shift_to_peak_origin(old, 1.5), [-3.5, -1.5, 1.5])


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


def test_continuous_frequency_against_complex_flow_derivative_and_batch_rows():
    state, ham, rr = setup_state()
    mode, frequency = driver.published_mode22_and_frequency(state, ham, rr, **PHASE)
    field = EOBTrajectoryIntegrator(ham, rr).vector_field(state)
    eps = 1e-4
    hdot = (direct_mode(state+eps*field, ham, rr)-direct_mode(state-eps*field, ham, rr))/(2*eps)
    torch.testing.assert_close(frequency, -(hdot/mode).imag, rtol=2e-9, atol=2e-11)
    assert bool((frequency-2*field[..., 1]).abs().max() > 1e-5)
    for i in range(2):
        single_ham = SEOBNRv4Hamiltonian(ham.nu[i, 0], ham.chi1[i, 0], ham.chi2[i, 0])
        single = driver.published_mode22_and_frequency(state[i, 1], single_ham, rr, **PHASE)
        torch.testing.assert_close(mode[i, 1], single[0])
        torch.testing.assert_close(frequency[i, 1], single[1])


@pytest.mark.parametrize('context', [torch.no_grad, torch.inference_mode])
def test_mode_frequency_with_locally_created_inference_tensors(context):
    state, ham, rr = setup_state()
    expected = driver.published_mode22_and_frequency(state, ham, rr, **PHASE)
    with context():
        state, ham, rr = setup_state()
        actual = driver.published_mode22_and_frequency(state, ham, rr, **PHASE)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)
        assert not value.requires_grad


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


def test_published_eob22_to_fd_analytic_mode_absolute_phase_and_origins():
    """An independently constructed mode fixes the physical Fourier chart."""
    dt, count = .25, 96
    # Keep the local sum origin distinct from the physical chart origin.  The
    # helper must account for this result.time[0] explicitly because
    # _quadrupole_fd starts its finite integral at index zero.
    time = 1.5 + torch.arange(count, dtype=torch.float64)*dt
    mode = (.8 + .002*time) * torch.exp(-.012*time) * torch.exp(-.47j*time)
    peak, match = 2.75, 8.25
    result = SimpleNamespace(
        time=time, mode=mode,
        trajectory=SimpleNamespace(peak=SimpleNamespace(time=torch.tensor(peak, dtype=time.dtype))),
        attachment=SimpleNamespace(match_time=torch.tensor(match, dtype=time.dtype)))
    total_mass, distance, inclination = 50., 230., .8
    phic, tc = .23, .002
    dt_sec = dt*total_mass*MSUN_SEC
    f = torch.arange(1, 8, dtype=time.dtype)/(512*dt_sec)

    hp, hc = driver.published_eob22_to_fd(
        result, f, 30., 20., distance=distance, inclination=inclination,
        phic=phic, tc=tc)
    physical_mode = mode*torch.exp(torch.tensor(-2j*phic))*total_mass*MSUN_SEC*C_SI/(distance*MPC_SI)
    ci = math.cos(inclination)
    plus = math.sqrt(5/(16*math.pi))*(1+ci*ci)*physical_mode.real
    cross = -math.sqrt(5/(4*math.pi))*ci*physical_mode.imag
    absolute_time = (time-peak)*total_mass*MSUN_SEC+tc
    kernel = torch.exp(-2j*math.pi*f[:, None]*absolute_time)
    hp_expected = (kernel*plus).sum(-1)*dt_sec
    hc_expected = (kernel*cross).sum(-1)*dt_sec
    torch.testing.assert_close(hp, hp_expected, rtol=1e-7, atol=1e-30)
    torch.testing.assert_close(hc, hc_expected, rtol=1e-7, atol=1e-30)

    # tc is an absolute physical translation of the same finite integral.
    tc_delta = .0017
    hp_tc, hc_tc = driver.published_eob22_to_fd(
        result, f, 30., 20., distance=distance, inclination=inclination,
        phic=phic, tc=tc+tc_delta)
    tc_factor = torch.exp(-2j*math.pi*f*tc_delta)
    torch.testing.assert_close(hp_tc, hp*tc_factor, rtol=1e-7, atol=1e-30)
    torch.testing.assert_close(hc_tc, hc*tc_factor, rtol=1e-7, atol=1e-30)

    # phic is applied to the mode before independent real-polarization projection.
    phase_delta = .19
    hp_phase, hc_phase = driver.published_eob22_to_fd(
        result, f, 30., 20., distance=distance, inclination=inclination,
        phic=phic+phase_delta, tc=tc)
    mode_phase = mode*torch.exp(torch.tensor(-2j*(phic+phase_delta))) * total_mass*MSUN_SEC*C_SI/(distance*MPC_SI)
    plus_phase = math.sqrt(5/(16*math.pi))*(1+ci*ci)*mode_phase.real
    cross_phase = -math.sqrt(5/(4*math.pi))*ci*mode_phase.imag
    torch.testing.assert_close(hp_phase, (kernel*plus_phase).sum(-1)*dt_sec, rtol=1e-7, atol=1e-30)
    torch.testing.assert_close(hc_phase, (kernel*cross_phase).sum(-1)*dt_sec, rtol=1e-7, atol=1e-30)

    # The explicit legacy origin changes only the absolute Fourier chart.
    hp_match, hc_match = driver.published_eob22_to_fd(
        result, f, 30., 20., distance=distance, inclination=inclination,
        phic=phic, tc=tc, time_origin='nqc_match')
    origin_factor = torch.exp(2j*math.pi*f*(peak-match)*total_mass*MSUN_SEC)
    torch.testing.assert_close(hp, hp_match*origin_factor, rtol=1e-7, atol=1e-30)
    torch.testing.assert_close(hc, hc_match*origin_factor, rtol=1e-7, atol=1e-30)

    hp_far, hc_far = driver.published_eob22_to_fd(
        result, f, 30., 20., distance=2*distance, inclination=inclination,
        phic=phic, tc=tc)
    torch.testing.assert_close(hp_far, hp/2, rtol=1e-7, atol=1e-30)
    torch.testing.assert_close(hc_far, hc/2, rtol=1e-7, atol=1e-30)



def _geometric_sum(q, count):
    return (1 - q**count) / (1 - q)


def _geometric_first_moment(q, count):
    return q * (1 - count*q**(count - 1) + (count - 1)*q**count) / (1 - q)**2


def test_published_eob22_to_fd_off_fft_node_sinusoid_matches_geometric_sum():
    """The direct runtime agrees with a closed finite geometric sum off FFT nodes."""
    count, dt_geom, start, omega = 37, .37, 1.25, .43
    time = start + torch.arange(count, dtype=torch.float64)*dt_geom
    mode = torch.complex(torch.sin(omega*time), torch.zeros_like(time))
    peak, match, tc = 2.1, 4.7, .0007
    result = SimpleNamespace(
        time=time, mode=mode,
        trajectory=SimpleNamespace(peak=SimpleNamespace(time=torch.tensor(peak, dtype=time.dtype))),
        attachment=SimpleNamespace(match_time=torch.tensor(match, dtype=time.dtype)))
    total_mass, distance, inclination = 50., 17., .63
    msec = total_mass*MSUN_SEC
    dt_sec = dt_geom*msec
    fft_len = 2**math.ceil(math.log2(count*4))
    fft_bin = 1/(fft_len*dt_sec)
    frequency = (17.3 + .37)*fft_bin
    assert abs(frequency/fft_bin - round(frequency/fft_bin)) > .1
    f = torch.tensor([frequency], dtype=torch.float64)

    hp, hc = driver.published_eob22_to_fd(
        result, f, 30., 20., distance=distance, inclination=inclination, tc=tc)
    t0_absolute = tc + (start - peak)*msec
    z = torch.exp(-2j*math.pi*f[0]*dt_sec)
    q_plus = z*torch.exp(torch.tensor(1j*omega*dt_geom, dtype=torch.complex128))
    q_minus = z*torch.exp(torch.tensor(-1j*omega*dt_geom, dtype=torch.complex128))
    sinusoid_sum = torch.exp(-2j*math.pi*f[0]*t0_absolute) * (
        torch.exp(torch.tensor(1j*omega*start, dtype=torch.complex128))*_geometric_sum(q_plus, count)
        - torch.exp(torch.tensor(-1j*omega*start, dtype=torch.complex128))*_geometric_sum(q_minus, count)
    )/(2j)
    scale = msec*C_SI/(distance*MPC_SI)
    projection = math.sqrt(5/(16*math.pi))*(1+math.cos(inclination)**2)
    hp_expected = projection*scale*dt_sec*sinusoid_sum
    torch.testing.assert_close(hp, hp_expected.unsqueeze(0), rtol=2e-7, atol=3e-30)
    torch.testing.assert_close(hc, torch.zeros_like(hc), rtol=0, atol=1e-30)


def test_published_eob22_to_fd_off_fft_node_polynomial_matches_geometric_moments():
    count, dt_geom, start = 29, .41, .73
    time = start + torch.arange(count, dtype=torch.float64)*dt_geom
    a, b = .2, -.013
    mode = torch.complex(a + b*time, torch.zeros_like(time))
    peak, match, tc = 1.8, 3.2, -.0004
    result = SimpleNamespace(
        time=time, mode=mode,
        trajectory=SimpleNamespace(peak=SimpleNamespace(time=torch.tensor(peak, dtype=time.dtype))),
        attachment=SimpleNamespace(match_time=torch.tensor(match, dtype=time.dtype)))
    total_mass, distance, inclination = 50., 41., .91
    msec = total_mass*MSUN_SEC
    dt_sec = dt_geom*msec
    fft_len = 2**math.ceil(math.log2(count*4))
    fft_bin = 1/(fft_len*dt_sec)
    frequency = (8.2 + .27)*fft_bin
    f = torch.tensor([frequency], dtype=torch.float64)
    hp, hc = driver.published_eob22_to_fd(
        result, f, 30., 20., distance=distance, inclination=inclination, tc=tc)
    t0_absolute = tc + (start - peak)*msec
    q = torch.exp(-2j*math.pi*f[0]*dt_sec)
    sum_poly = (a + b*start)*_geometric_sum(q, count) + b*dt_geom*_geometric_first_moment(q, count)
    scale = msec*C_SI/(distance*MPC_SI)
    projection = math.sqrt(5/(16*math.pi))*(1+math.cos(inclination)**2)
    hp_expected = projection*scale*dt_sec*torch.exp(-2j*math.pi*f[0]*t0_absolute)*sum_poly
    torch.testing.assert_close(hp, hp_expected.unsqueeze(0), rtol=2e-12, atol=1e-30)
    torch.testing.assert_close(hc, torch.zeros_like(hc), rtol=0, atol=1e-30)


def test_published_eob22_to_fd_frequency_gradient_matches_polynomial_closed_form():
    count, dt_geom, start = 23, .29, .61
    time = start + torch.arange(count, dtype=torch.float64)*dt_geom
    a, b = .17, .021
    mode = torch.complex(a + b*time, torch.zeros_like(time))
    peak, tc = 1.4, .00023
    result = SimpleNamespace(
        time=time, mode=mode,
        trajectory=SimpleNamespace(peak=SimpleNamespace(time=torch.tensor(peak, dtype=time.dtype))),
        attachment=SimpleNamespace(match_time=torch.tensor(2.6, dtype=time.dtype)))
    total_mass, distance, inclination = 50., 73., .72
    msec = total_mass*MSUN_SEC
    dt_sec = dt_geom*msec
    fft_len = 2**math.ceil(math.log2(count*4))
    frequency_value = (11.4 + .19)/(fft_len*dt_sec)
    frequency = torch.tensor([frequency_value], dtype=torch.float64, requires_grad=True)
    hp, _ = driver.published_eob22_to_fd(
        result, frequency, 30., 20., distance=distance, inclination=inclination, tc=tc)
    scale = msec*C_SI/(distance*MPC_SI)
    actual_gradient = torch.autograd.grad((hp.real/scale).sum(), frequency)[0]

    def closed_form(freq):
        q = torch.exp(-2j*math.pi*freq*dt_sec)
        s0 = _geometric_sum(q, count)
        s1 = _geometric_first_moment(q, count)
        t0_absolute = tc + (start - peak)*msec
        projection = math.sqrt(5/(16*math.pi))*(1+math.cos(inclination)**2)
        return projection*dt_sec*torch.exp(-2j*math.pi*freq*t0_absolute)*(
            (a + b*start)*s0 + b*dt_geom*s1)

    closed_frequency = torch.tensor([frequency_value], dtype=torch.float64, requires_grad=True)
    closed_gradient = torch.autograd.grad(closed_form(closed_frequency).real.sum(), closed_frequency)[0]
    torch.testing.assert_close(actual_gradient, closed_gradient, rtol=2e-10, atol=2e-12)

def test_published_eob22_to_fd_rejects_invalid_physical_and_mode_inputs():
    time = 1.5 + torch.arange(16, dtype=torch.float64)*.25
    mode = torch.exp(-.01*time-.4j*time)
    result = SimpleNamespace(
        time=time, mode=mode,
        trajectory=SimpleNamespace(peak=SimpleNamespace(time=torch.tensor(2.))),
        attachment=SimpleNamespace(match_time=torch.tensor(3.)))
    f = torch.tensor([20., 40.], dtype=torch.float64)

    for mass1, mass2 in ((0., 20.), (-1., 20.), (30., 0.), (30., -1.)):
        with pytest.raises(ValueError, match='masses'):
            driver.published_eob22_to_fd(result, f, mass1, mass2)

    bad_origin = SimpleNamespace(
        time=time, mode=mode,
        trajectory=SimpleNamespace(peak=SimpleNamespace(time=torch.tensor(float('nan')))),
        attachment=result.attachment)
    with pytest.raises(ValueError, match='time origin'):
        driver.published_eob22_to_fd(bad_origin, f, 30., 20.)

    bad_mode_value = mode.clone()
    bad_mode_value[4] = torch.complex(torch.tensor(float('nan')), torch.tensor(0.))
    bad_mode = SimpleNamespace(
        time=time, mode=bad_mode_value,
        trajectory=result.trajectory, attachment=result.attachment)
    with pytest.raises(ValueError, match='result.mode'):
        driver.published_eob22_to_fd(bad_mode, f, 30., 20.)

    with pytest.raises(ValueError, match='nonempty'):
        driver.published_eob22_to_fd(result, torch.empty(0, dtype=torch.float64), 30., 20.)


def test_published_eob22_to_fd_broadcasts_frequency_and_parameter_batches():
    time = 1.5 + torch.arange(16, dtype=torch.float64)*.25
    mode = torch.exp(-.01*time-.4j*time)
    result = SimpleNamespace(
        time=time, mode=mode,
        trajectory=SimpleNamespace(peak=SimpleNamespace(time=torch.tensor(2.))),
        attachment=SimpleNamespace(match_time=torch.tensor(3.)))
    frequencies = torch.tensor([[20., 30., 40., 50.], [25., 35., 45., 55.],
                                [30., 40., 50., 60.]], dtype=torch.float64)
    masses1 = torch.tensor([[30.], [35.]], dtype=torch.float64)
    masses2 = torch.tensor([[15., 20., 25.]], dtype=torch.float64)
    actual = driver.published_eob22_to_fd(result, frequencies, masses1, masses2)
    assert actual[0].shape == (2, 3, 4)
    assert actual[1].shape == (2, 3, 4)
    for i in range(2):
        for j in range(3):
            expected = driver.published_eob22_to_fd(
                result, frequencies[j], masses1[i, 0], masses2[0, j])
            torch.testing.assert_close(actual[0][i, j], expected[0])
            torch.testing.assert_close(actual[1][i, j], expected[1])


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
