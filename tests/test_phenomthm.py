"""Public contract checks for native aligned-spin IMRPhenomTHM."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
import jax.numpy as jnp

from jaxwave.approximants.imrphenomthm import IMRPhenomTHM as JaxTHM
from torchwave.approximants.imrphenomt import IMRPhenomT as TorchT
from torchwave.approximants.imrphenomthm import IMRPhenomTHM as TorchTHM


jax.config.update("jax_enable_x64", True)
_MSUN_SEC = 4.925490947641267e-6


@pytest.mark.parametrize("backend", (TorchTHM, JaxTHM))
def test_scalar_batch_and_call_contract(backend):
    model = backend()
    times = np.linspace(-200.0, 20.0, 9) * 55.0 * _MSUN_SEC
    if backend is TorchTHM:
        times = torch.as_tensor(times, dtype=torch.float64)
        batch = model.modes(times, torch.tensor([30.0, 35.0]), 25.0, chi1=torch.tensor([0.2, 0.3]), chi2=-0.1, mode_array=[(2, 2)])
        called = model(times, 30.0, 25.0, mode_array=[(2, 2)])
    else:
        times = jnp.asarray(times)
        batch = model.modes(times, jnp.array([30.0, 35.0]), 25.0, chi1=jnp.array([0.2, 0.3]), chi2=-0.1, mode_array=[(2, 2)])
        called = model(times, 30.0, 25.0, mode_array=[(2, 2)])
    assert batch[(2, 2)].shape == (2, times.shape[0])
    direct = model.time_domain(times, 30.0, 25.0, mode_array=[(2, 2)])
    for actual, expected in zip(called, direct):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("backend", (TorchTHM, JaxTHM))
def test_all_mode_batch_matches_scalar(backend):
    """Every supported positive mode must preserve the leading parameter batch."""
    model = backend()
    mode_array = list(model.supported_modes)
    raw_times = np.linspace(-120.0, 5.0, 7) * 55.0 * _MSUN_SEC
    if backend is TorchTHM:
        times = torch.as_tensor(raw_times, dtype=torch.float64)
        kwargs = dict(mass1=torch.tensor([30.0, 35.0]), mass2=torch.tensor([25.0, 20.0]),
                      chi1=torch.tensor([0.2, -0.3]), chi2=torch.tensor([-0.1, 0.4]))
    else:
        times = jnp.asarray(raw_times)
        kwargs = dict(mass1=jnp.array([30.0, 35.0]), mass2=jnp.array([25.0, 20.0]),
                      chi1=jnp.array([0.2, -0.3]), chi2=jnp.array([-0.1, 0.4]))
    batched = model.modes(times, mode_array=mode_array, **kwargs)
    for mode in mode_array:
        for j in range(2):
            scalar_kwargs = {name: value[j] for name, value in kwargs.items()}
            scalar = model.modes(times, mode_array=[mode], **scalar_kwargs)[mode]
            np.testing.assert_allclose(np.asarray(batched[mode][j]), np.asarray(scalar), rtol=1e-12, atol=1e-30)


@pytest.mark.parametrize("backend", (TorchTHM, JaxTHM))
def test_mass_spin_swap_is_canonical(backend):
    model = backend()
    times = np.linspace(-150.0, 10.0, 7) * 55.0 * _MSUN_SEC
    if backend is TorchTHM:
        times = torch.as_tensor(times, dtype=torch.float64)
    else:
        times = jnp.asarray(times)
    a = model.modes(times, 30.0, 25.0, chi1=0.2, chi2=-0.1)
    b = model.modes(times, 25.0, 30.0, chi1=-0.1, chi2=0.2)
    for mode in a:
        np.testing.assert_allclose(np.asarray(a[mode]), np.asarray(b[mode]), rtol=1e-12, atol=1e-30)


def test_22_only_matches_phenomt_torch():
    times = torch.linspace(-1.0e-3, 1.0e-3, 11, dtype=torch.float64)
    kwargs = dict(mass1=30.0, mass2=25.0, chi1=0.2, chi2=-0.1, inclination=0.7, phic=0.2, f_ref=20.0)
    hm = TorchTHM().time_domain(times, mode_array=[(2, 2)], **kwargs)
    t = TorchT().time_domain(times, **kwargs)
    for actual, expected in zip(hm, t):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-30)


@pytest.mark.parametrize("backend", (TorchTHM, JaxTHM))
def test_negative_pair_selection_and_invalid_modes(backend):
    model = backend()
    times = torch.tensor([0.0], dtype=torch.float64) if backend is TorchTHM else jnp.array([0.0])
    selected = model.modes(times, 30.0, 25.0, mode_array=[(2, -1)])
    assert set(selected) == {(2, 1), (2, -1)}
    for bad in ([], [(3, 2)]):
        with pytest.raises(ValueError):
            model.modes(times, 30.0, 25.0, mode_array=bad)


def test_equal_mass_equal_spin_odd_mode_is_zero_and_differentiable():
    times = torch.linspace(-1.0e-3, 1.0e-3, 7, dtype=torch.float64)
    mass = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)
    spin = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    value = TorchTHM().modes(times, mass, mass, spin, spin, mode_array=[(2, 1)])[(2, 1)]
    assert torch.equal(value, torch.zeros_like(value))
    objective = value.real.square().sum() + value.imag.square().sum()
    grad = torch.autograd.grad(objective, (mass, spin), allow_unused=False)
    assert all(torch.isfinite(g) for g in grad)


def test_jax_equal_mass_equal_spin_odd_mode_is_zero_and_differentiable():
    times = jnp.linspace(-1.0e-3, 1.0e-3, 7)
    model = JaxTHM()

    def waveform(mass):
        spin = jnp.asarray(0.2, dtype=mass.dtype)
        return model.modes(times, mass, mass, spin, spin, mode_array=[(2, 1)])[(2, 1)]

    value = waveform(jnp.asarray(30.0))
    assert np.array_equal(np.asarray(value), np.zeros_like(np.asarray(value)))
    def objective(mass):
        value = waveform(mass)
        return jnp.sum(jnp.real(value) ** 2 + jnp.imag(value) ** 2)

    assert np.isfinite(np.asarray(jax.grad(objective)(jnp.asarray(30.0))))


@pytest.mark.parametrize("backend", (TorchTHM, JaxTHM))
def test_final_spin_override_only_changes_ringdown(backend):
    """An augmented remnant spin must leave every HM merger sample unchanged."""
    model = backend()
    mode_array = list(model.supported_modes)
    # Times are total-mass units, with negative samples safely before the
    # frequency ringdown join and positive samples in the active ringdown.
    u = np.asarray((-0.05, -0.01, 0.02, 0.05))
    times = u * 55.0 * _MSUN_SEC
    if backend is TorchTHM:
        times = torch.as_tensor(times, dtype=torch.float64)
    else:
        times = jnp.asarray(times)
    kwargs = dict(mass1=40.0, mass2=15.0, chi1=0.3, chi2=-0.2,
                  distance=100.0, mode_array=mode_array)
    aligned = model.modes(times, final_spin=None, **kwargs)
    augmented = model.modes(times, final_spin=0.5, **kwargs)
    for mode in mode_array:
        a, b = np.asarray(aligned[mode]), np.asarray(augmented[mode])
        np.testing.assert_allclose(a[:2], b[:2], rtol=1e-12, atol=1e-30)
    late_delta = max(
        np.max(np.abs(np.asarray(aligned[mode][-2:]) - np.asarray(augmented[mode][-2:])))
        / np.max(np.abs(np.asarray(aligned[mode][-2:])))
        for mode in mode_array
    )
    assert late_delta > 1e-8
