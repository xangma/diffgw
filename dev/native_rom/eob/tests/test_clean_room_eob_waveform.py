"""Analytic quadrupole, Fourier scaling and normalized energy-balance checks."""
import math

import pytest
import torch

from dev.native_rom.eob.dispatch import torch_get_fd_waveform as get_fd_waveform


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
        "dev.native_rom.eob.torch.waveform.find_quasicircular_initial_conditions",
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
