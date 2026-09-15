"""Analytic quadrupole, Fourier scaling and normalized energy-balance checks."""
import math

import pytest
import torch

from torchwave.approximants.eob.waveform import _interp1d, _quadrupole_fd
from torchwave.approximants.eob.radiation_reaction import EOBRadiationReaction
from torchwave.waveforms import get_fd_waveform


def test_scalar_interpolation_preserves_table_batches_and_query_derivative():
    grid = torch.tensor([0., 1., 3.], dtype=torch.float64)
    values = torch.tensor([[0., 2., 4.], [1., 4., 12.]], dtype=torch.float64)
    query = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    actual = _interp1d(query, grid, values)
    torch.testing.assert_close(actual, torch.tensor([3., 8.], dtype=torch.float64))
    torch.testing.assert_close(torch.autograd.functional.jacobian(
        lambda q: _interp1d(q, grid, values), query),
        torch.tensor([1., 4.], dtype=torch.float64))
    assert _interp1d(query, grid, values[0]).shape == ()
    assert torch.count_nonzero(_interp1d(query + 2, grid, values)) == 0


@pytest.mark.parametrize("inclination", [0.0, 0.7, math.pi / 2])
def test_periodic_quadrupole_has_correct_positive_frequency_normalization(inclination):
    # Eight complete cycles: exact discrete Fourier orthogonality, no SPA.
    n, dt, k, amplitude = 1024, 1 / 1024, 8, 2.3
    time = torch.arange(n, dtype=torch.float64) * dt
    h22 = amplitude * torch.exp(-2j * math.pi * k * time)
    f = torch.tensor([k], dtype=torch.float64)
    inc = torch.tensor(inclination, dtype=torch.float64)
    hp, hc = _quadrupole_fd(f, h22, torch.tensor(dt, dtype=torch.float64), inc)
    c = math.cos(inclination)
    expected_p = math.sqrt(5 / (16 * math.pi)) * (1 + c*c) * amplitude / 2
    expected_c = -1j * math.sqrt(5 / (4 * math.pi)) * c * amplitude / 2
    torch.testing.assert_close(hp, torch.full_like(hp, expected_p), rtol=2e-13, atol=1e-14)
    torch.testing.assert_close(hc, torch.full_like(hc, expected_c), rtol=2e-13, atol=1e-14)


@pytest.mark.parametrize(
    "inclination,phase_offset",
    [(0.13, 0.0), (0.7, 0.37), (1.41, -1.2)],
)
def test_quadrupole_fd_matches_finite_fourier_sum_for_general_mode_phase(
    inclination, phase_offset
):
    """The published 22 projection and the negative-frequency FFT agree analytically."""
    n, dt, k, amplitude = 1024, 1 / 1024, 8, 2.3
    time = torch.arange(n, dtype=torch.float64) * dt
    h22 = amplitude * torch.exp(
        torch.tensor(1j * phase_offset, dtype=torch.complex128)
        - 2j * math.pi * k * time
    )
    f = torch.tensor([k / (n * dt)], dtype=torch.float64)
    inc = torch.tensor(inclination, dtype=torch.float64)
    actual_p, actual_c = _quadrupole_fd(f, h22, torch.tensor(dt), inc)

    c = math.cos(inclination)
    projected_p = math.sqrt(5 / (16 * math.pi)) * (1 + c * c) * h22.real
    projected_c = -math.sqrt(5 / (4 * math.pi)) * c * h22.imag
    expected_p = (projected_p * torch.exp(-2j * math.pi * f[0] * time)).sum() * dt
    expected_c = (projected_c * torch.exp(-2j * math.pi * f[0] * time)).sum() * dt
    torch.testing.assert_close(actual_p[0], expected_p, rtol=2e-13, atol=1e-14)
    torch.testing.assert_close(actual_c[0], expected_c, rtol=2e-13, atol=1e-14)


def test_fourier_mass_scale_preserves_gradient_and_batched_sampling():
    time = torch.arange(64, dtype=torch.float64)
    signal = torch.exp(-time / 20) * torch.exp(-0.5j * time)
    f = torch.tensor([.071, .139], dtype=torch.float64)
    scale = torch.tensor(.9, dtype=torch.float64, requires_grad=True)
    def evaluate(s):
        return _quadrupole_fd(f, signal * s, s * .5, s * .2)[0]
    assert torch.autograd.gradcheck(evaluate, (scale,), atol=1e-5, rtol=1e-4)
    scales = torch.tensor([.7, 1.3], dtype=torch.float64)
    batched = _quadrupole_fd(f, signal[:, None] * scales, scales * .5, scales * .2)
    for i, s in enumerate(scales):
        single = _quadrupole_fd(f, signal * s, s * .5, s * .2)
        for b, one in zip(batched, single):
            torch.testing.assert_close(b[i], one, atol=1e-14, rtol=1e-13)


def test_normalized_torque_obeys_energy_balance_and_zero_momentum_is_finite():
    nu = torch.tensor(.19, dtype=torch.float64)
    zero = torch.zeros_like(nu)
    rr = EOBRadiationReaction(nu, zero, zero)
    omega = torch.tensor(.008, dtype=torch.float64)
    radial, torque = rr.radiation_reaction_forces(omega, zero, zero)
    torch.testing.assert_close(nu * omega * torque, -rr.energy_flux(omega))
    assert torch.isfinite(radial) and radial == 0


@pytest.mark.parametrize("samples", [2, 3])
def test_normalized_torque_batches_parameters_before_frequency(samples):
    nu = torch.tensor([.17, .24], dtype=torch.float64)
    rr = EOBRadiationReaction(nu, torch.zeros_like(nu), torch.zeros_like(nu))
    omega = torch.full((2, samples), .012, dtype=torch.float64)
    _, torque = rr.radiation_reaction_forces(omega, torch.zeros_like(omega), torch.ones_like(omega))
    torch.testing.assert_close(nu[:, None] * omega * torque, -rr.energy_flux(omega))


@pytest.mark.parametrize("conservative_model", ["prototype", "seobnrv4_published"])
def test_eob_mass_scaling_includes_ringdown_seconds_and_fourier_measure(conservative_model):
    f = torch.linspace(30, 110, 15, dtype=torch.float64)
    args = dict(distance=200., inclination=.8, num_steps=8, dt_geom=1., conservative_model=conservative_model)
    first = get_fd_waveform("SEOBNR_ODE", mass1=30., mass2=20.,
                            sample_frequencies=f, f_lower=30., **args)
    second = get_fd_waveform("SEOBNR_ODE", mass1=60., mass2=40.,
                             sample_frequencies=f/2, f_lower=15., **args)
    for a, b in zip(first, second):
        # h(t) scales as M/D and dt scales as M, hence h(f) scales as M^2/D.
        torch.testing.assert_close(b, 4*a, rtol=2e-11, atol=1e-33)


def test_eob_finite_signal_orbital_rotation_and_time_delay():
    f = torch.tensor([30., 40., 50.], dtype=torch.float64)
    args = dict(mass1=30., mass2=20., sample_frequencies=f, f_lower=30.,
                inclination=.8, num_steps=8, dt_geom=1.)
    hp, hc = get_fd_waveform("SEOBNR_ODE", **args)
    phase, delay = .37, .003
    rotated = get_fd_waveform("SEOBNR_ODE", phic=phase, tc=delay, **args)
    c = math.cos(args['inclination'])
    ratio = (1+c*c)/(2*c)
    shift = torch.exp(-2j*math.pi*f*delay)
    expected = ((math.cos(2*phase)*hp - ratio*math.sin(2*phase)*hc)*shift,
                (math.cos(2*phase)*hc + math.sin(2*phase)*hp/ratio)*shift)
    for actual, reference in zip(rotated, expected):
        torch.testing.assert_close(actual, reference, rtol=2e-11, atol=1e-33)


def test_eob_broadcast_parameter_and_frequency_grids_match_scalar_calls():
    m1 = torch.tensor([[30.], [35.]], dtype=torch.float64)
    m2 = torch.tensor([[20., 22., 24.]], dtype=torch.float64)
    f = torch.tensor([30., 40., 50., 60.], dtype=torch.float64).expand(2, 3, 4)
    args = dict(f_lower=30., inclination=.8, num_steps=3, dt_geom=1., tc=.002)
    batched = get_fd_waveform("SEOBNR_ODE", mass1=m1, mass2=m2, sample_frequencies=f, **args)
    assert batched[0].shape == (2, 3, 4)
    for i in range(2):
        for j in range(3):
            single = get_fd_waveform("SEOBNR_ODE", mass1=m1[i, 0], mass2=m2[0, j], sample_frequencies=f[i,j], **args)
            for b, s in zip(batched, single):
                torch.testing.assert_close(b[i,j], s, rtol=2e-10, atol=1e-33)


def test_eob_requires_a_positive_start_frequency():
    with pytest.raises(ValueError, match="f_lower is required"):
        get_fd_waveform(
            "SEOBNR_ODE",
            mass1=30.,
            mass2=20.,
            sample_frequencies=torch.zeros(3, dtype=torch.float64),
            num_steps=3,
        )


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf")])
def test_eob_rejects_invalid_explicit_start_frequency(invalid, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid f_lower must be rejected before initial data")

    monkeypatch.setattr(
        "torchwave.approximants.eob.waveform.find_quasicircular_initial_conditions",
        unexpected,
    )
    with pytest.raises(ValueError, match="f_lower must be finite and positive"):
        get_fd_waveform(
            "SEOBNR_ODE",
            mass1=30.0,
            mass2=20.0,
            sample_frequencies=torch.tensor([30.0], dtype=torch.float64),
            f_lower=invalid,
            num_steps=3,
        )
