"""Independent transform and causal-support checks for real ringdown strain."""

import numpy as np
import pytest
import torch
from scipy.integrate import simpson

from torchwave.approximants.kerr_ringdown import KerrRingdown, evaluate_qnm, get_ringdown_waveform
from torchwave.constants import MSUN_SEC


@pytest.mark.parametrize("angular_basis", ["spherical", "spheroidal"])
def test_ringdown_fourier_transform_matches_time_quadrature(angular_basis):
    # Simpson integration of the public TD function is independent of the
    # frequency-domain rational expression, and includes the onset endpoint.
    t0 = 0.013
    times = torch.linspace(t0, t0 + 0.5, 65537, dtype=torch.float64)
    frequencies = torch.tensor([-1200., -300., -17., 0., 31., 270., 913.], dtype=torch.float64)
    params = dict(final_mass=63., final_spin=0.72, inclination=0.9, phi=0.37, angular_basis=angular_basis,
                  t0=t0, amplitudes={(2, 2, 0): 0.8, (3, 3, 1): 0.23, (3, 2, 2): 0.11},
                  phases={(2, 2, 0): 0.41, (3, 3, 1): -0.81, (3, 2, 2): 1.27})
    hp_td, hc_td = KerrRingdown.time_domain(times, **params)
    hp_fd, hc_fd = KerrRingdown.frequency_domain(frequencies, **params)
    kernel = np.exp(-2j*np.pi*frequencies.numpy()[:, None]*times.numpy())
    for td, fd in [(hp_td, hp_fd), (hc_td, hc_fd)]:
        numerical = simpson(td.numpy()[None, :]*kernel, x=times.numpy(), axis=-1)
        np.testing.assert_allclose(fd.numpy(), numerical, rtol=2e-7, atol=1e-12)


def test_ringdown_real_signal_hermitian_symmetry_and_time_shift():
    f = torch.tensor([0., 41., 273., 890.], dtype=torch.float64)
    params = dict(final_mass=61., final_spin=0.81, inclination=1.1, phi=-0.2,
                  phases={(2, 2, 0): 0.8})
    positive = KerrRingdown.frequency_domain(f, **params)
    negative = KerrRingdown.frequency_domain(-f, **params)
    shifted = KerrRingdown.frequency_domain(f, t0=0.03, **params)
    for hp, hn, hs in zip(positive, negative, shifted):
        torch.testing.assert_close(hn, hp.conj(), rtol=1e-13, atol=1e-15)
        torch.testing.assert_close(hs, hp*torch.exp(-2j*torch.pi*f*0.03), rtol=1e-13, atol=1e-15)


def test_ringdown_far_before_onset_is_finite_with_zero_gradients():
    times = torch.tensor([-1e10, -1e5, -100.], dtype=torch.float64)
    mass = torch.tensor(60., dtype=torch.float64, requires_grad=True)
    spin = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    hp, hc = KerrRingdown.time_domain(times, mass, spin)
    assert torch.equal(hp, torch.zeros_like(hp))
    assert torch.equal(hc, torch.zeros_like(hc))
    grads = torch.autograd.grad((hp+hc).sum(), (mass, spin))
    assert all(torch.isfinite(g) and g == 0 for g in grads)


@pytest.mark.parametrize("angular_basis", ["spherical", "spheroidal"])
def test_ringdown_fd_gradients(angular_basis):
    f = torch.tensor([-130., 0., 370.], dtype=torch.float64)
    inputs = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True)
                   for v in [65., 0.6, 0.8, 0.002])
    def waveform(mass, spin, inclination, t0):
        return KerrRingdown.frequency_domain(f, mass, spin, inclination=inclination,
                                            t0=t0, phases={(2, 2, 0): 0.6}, angular_basis=angular_basis)
    assert torch.autograd.gradcheck(waveform, inputs, eps=1e-7, atol=1e-6, rtol=1e-4)


def test_default_independent_frequencies_have_physical_mass_scaling():
    mass = torch.tensor([30., 60.], dtype=torch.float64)
    omega, tau, quality = evaluate_qnm(2, 2, 0, mass, 0.)
    # Independent numerical Schwarzschild eigenvalue, Cook author data.
    torch.testing.assert_close(omega * mass * MSUN_SEC, omega.new_full((2,), .3736716844180418), rtol=0, atol=2e-15)
    torch.testing.assert_close(mass * MSUN_SEC / tau, tau.new_full((2,), .0889623156889357), rtol=0, atol=2e-15)
    torch.testing.assert_close(quality, omega * tau / 2, rtol=1e-15, atol=0)


def test_qnm_backend_reaches_both_ringdown_domains():
    for coordinate in ("times", "frequencies"):
        grid = torch.linspace(0, .02, 31, dtype=torch.float64) if coordinate == "times" else torch.linspace(20, 500, 31, dtype=torch.float64)
        outputs = {}
        direct = KerrRingdown.time_domain if coordinate == "times" else KerrRingdown.frequency_domain
        for backend in ("independent", "author", "berti"):
            outputs[backend] = get_ringdown_waveform(65., .68, **{coordinate: grid}, qnm_backend=backend)
            expected = direct(grid, 65., .68, qnm_backend=backend)
            for result, target in zip(outputs[backend], expected):
                torch.testing.assert_close(result, target, rtol=0, atol=0)
        assert torch.linalg.vector_norm(outputs["independent"][0] - outputs["berti"][0]) > 1e-7


def test_spheroidal_waveform_recovers_spherical_limit():
    params = dict(final_mass=60., final_spin=0., inclination=.93, phi=.27,
                  amplitudes={(2, 2, 0): .8, (2, 1, 1): .2, (3, 3, 0): .1, (3, 2, 2): .04, (4, 4, 0): .03})
    for generate, grid in [(KerrRingdown.time_domain, torch.linspace(0, .02, 31, dtype=torch.float64)),
                           (KerrRingdown.frequency_domain, torch.linspace(-500, 500, 31, dtype=torch.float64))]:
        for sphere, spheroid in zip(generate(grid, **params), generate(grid, **params, angular_basis="spheroidal")):
            torch.testing.assert_close(sphere, spheroid, rtol=2e-13, atol=1e-15)


@pytest.mark.parametrize("angular_basis", ["spherical", "spheroidal"])
def test_ringdown_multidimensional_parameter_batches(angular_basis):
    spin = torch.tensor([[.2], [.7]], dtype=torch.float64)
    inc = torch.tensor([[.4, .8, 1.2]], dtype=torch.float64)
    amp = torch.tensor([[.2, .4, .6]], dtype=torch.float64)
    for generate, grid in [(KerrRingdown.time_domain, torch.linspace(0, .02, 11, dtype=torch.float64)),
                           (KerrRingdown.frequency_domain, torch.linspace(-500, 500, 11, dtype=torch.float64))]:
        batched = generate(grid, 60., spin, inclination=inc, amplitudes={(3, 2, 1): amp}, angular_basis=angular_basis)
        assert batched[0].shape == (2, 3, 11)
        for i in range(2):
            for j in range(3):
                scalar = generate(grid, 60., spin[i, 0], inclination=inc[0, j], amplitudes={(3, 2, 1): amp[0, j]}, angular_basis=angular_basis)
                for result, expected in zip(batched, scalar):
                    torch.testing.assert_close(result[i, j], expected, rtol=2e-13, atol=1e-15)
