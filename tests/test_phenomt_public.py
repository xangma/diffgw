"""Public dispatch, sampling and validation for native IMRPhenomT."""

import numpy as np
import pytest
import torch

import torchwave
import jaxwave


@pytest.mark.parametrize("backend", [torchwave, jaxwave])
def test_native_public_sampling_and_low_level_equivalence(backend):
    hp, hc, times = backend.get_td_waveform(
        "IMR_Phenom-T",
        mass1=30.0,
        mass2=25.0,
        spin1z=0.2,
        spin2z=-0.1,
        duration=0.5,
        delta_t=1 / 1024,
        f_lower=20.0,
        f_ref=30.0,
        phic=0.3,
        tc=0.01,
        return_times=True,
    )
    assert hp.shape == hc.shape == times.shape == (512,)
    np.testing.assert_allclose(np.asarray(times)[[0, -1]], [-0.3, 0.2 - 1 / 1024])
    native = backend.IMRPhenomT().time_domain(
        times,
        30.0,
        25.0,
        chi1=0.2,
        chi2=-0.1,
        f_ref=30.0,
        phic=0.3,
        tc=0.01,
    )
    for actual, expected in zip((hp, hc), native):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12, atol=1e-30)

    model = backend.IMRPhenomT()
    for actual, expected in zip(model(times, 30.0, 25.0), model.time_domain(times, 30.0, 25.0)):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("backend", [torchwave, jaxwave])
def test_reference_zero_means_lower_and_explicit_time_origin(backend):
    kwargs = dict(
        mass1=30.0, mass2=25.0, duration=0.125, delta_t=1 / 1024, center_coalescence=False, tc=0.08, return_times=True
    )
    hp, hc, times = backend.get_td_waveform("IMRPhenomT", f_ref=0, **kwargs)
    hp2, hc2, _ = backend.get_td_waveform("IMRPhenomT", f_ref=20, **kwargs)
    assert float(times[0]) == 0.0
    np.testing.assert_allclose(np.asarray(hp), np.asarray(hp2), rtol=1e-13, atol=1e-30)
    np.testing.assert_allclose(np.asarray(hc), np.asarray(hc2), rtol=1e-13, atol=1e-30)


@pytest.mark.parametrize("backend", [torchwave, jaxwave])
def test_auto_duration_and_lower_frequency_mask(backend):
    hp, hc, times = backend.get_td_waveform(
        "IMRPhenomT",
        mass1=30.0,
        mass2=25.0,
        f_lower=40.0,
        delta_t=1 / 1024,
        return_times=True,
    )
    start = float(backend.IMRPhenomT().time_at_frequency(40.0, 30.0, 25.0))
    before = np.asarray(times) < start
    assert before.any()
    assert np.all(np.asarray(hp)[before] == 0.0)
    assert np.all(np.asarray(hc)[before] == 0.0)
    assert np.max(np.abs(np.asarray(hp)[~before])) > 0


@pytest.mark.parametrize("backend", [torchwave, jaxwave])
@pytest.mark.parametrize(
    "option",
    [
        {"spin1x": 0.1},
        {"lambda1": 10.0},
        {"f_final": 512.0},
        {"duration": 0.0},
        {"delta_t": -1.0},
        {"f_lower": 0.0},
        {"f_lower": 10000.0},
        {"distance": -10.0},
        {"mass1": 0.0},
        {"chi1": 1.1},
        {"f_ref": -1.0},
        {"t_pad": -0.1},
        {"chi1": 0.2, "spin1z": 0.3},
    ],
)
def test_reject_unsupported_or_invalid_options(backend, option):
    kwargs = dict(mass1=30.0, mass2=25.0, duration=0.25)
    kwargs.update(option)
    with pytest.raises(ValueError):
        backend.get_td_waveform("IMRPhenomT", **kwargs)


@pytest.mark.parametrize("backend", [torchwave, jaxwave])
def test_reject_frequency_domain_and_unknown_options(backend):
    with pytest.raises(ValueError, match="time-domain"):
        backend.get_fd_waveform("IMRPhenomT", mass1=30.0, mass2=25.0)
    with pytest.raises(TypeError, match="Unsupported"):
        backend.get_td_waveform("IMRPhenomT", mass1=30.0, mass2=25.0, typo=1)


def test_torch_catalogue_and_aliases():
    assert "IMRPhenomT" in torchwave.list_approximants("time_domain", inventory="callable")
    assert "IMRPhenomT" not in torchwave.list_approximants("frequency_domain", inventory="callable")
    hp, hc = torchwave.get_td_waveform("IMRPhenomT", m1=30.0, m2=25.0, duration=0.25)
    assert hp.shape == hc.shape == (512,)


def test_batched_public_gradients():
    m1 = torch.tensor([30.0, 35.0], dtype=torch.float64, requires_grad=True)
    distance = torch.tensor([100.0, 200.0], dtype=torch.float64)
    hp, hc = torchwave.get_td_waveform(
        "IMRPhenomT",
        mass1=m1,
        mass2=25.0,
        distance=distance,
        f_ref=30.0,
        duration=0.5,
        delta_t=1 / 1024,
    )
    assert hp.shape == hc.shape == (2, 512)
    (grad,) = torch.autograd.grad((hp.square() + hc.square()).sum() * 1e40, m1)
    assert torch.isfinite(grad).all() and torch.all(grad != 0)


def test_jax_public_fixed_duration_jit():
    import jax

    def wave(mass):
        return jaxwave.get_td_waveform("IMRPhenomT", mass1=mass, mass2=25.0, duration=0.25, delta_t=1 / 1024)

    actual = jax.jit(wave)(30.0)
    expected = wave(30.0)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=2e-10, atol=1e-30)
