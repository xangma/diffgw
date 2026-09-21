"""The precessing carrier must use one remnant in evaluation and inversion."""

import importlib

import numpy as np
import pytest


@pytest.mark.parametrize("backend", ["torchwave", "jaxwave"])
def test_remnant_override_frequency_inverse_and_reference_phase(backend):
    model = importlib.import_module(backend).IMRPhenomT()
    af = 0.83
    frequency = 270.0  # Above the amplitude peak for this binary.
    root = model.time_at_frequency(frequency, 40.0, 15.0, 0.3, -0.2, _final_spin_override=af)
    t = np.asarray([float(root)])
    actual = model.frequency(t, 40.0, 15.0, 0.3, -0.2, _final_spin_override=af)
    np.testing.assert_allclose(np.asarray(actual), frequency, rtol=2e-13)
    hp, hc = model.time_domain(t, 40.0, 15.0, 0.3, -0.2, inclination=0.7,
                               f_ref=frequency, phic=0.0, _final_spin_override=af)
    assert np.isfinite(np.asarray(hp)).all()
    assert np.max(np.abs(np.asarray(hc))) < 1e-12 * np.max(np.abs(np.asarray(hp)))


def test_remnant_inverse_first_derivative_matches_finite_difference():
    import torch
    from torchwave import IMRPhenomT

    model = IMRPhenomT()
    af = torch.tensor(0.83, dtype=torch.float64, requires_grad=True)
    root = model.time_at_frequency(270.0, 40.0, 15.0, 0.3, -0.2, _final_spin_override=af)
    derivative, = torch.autograd.grad(root, af)
    step = 1e-5
    plus = model.time_at_frequency(270.0, 40.0, 15.0, 0.3, -0.2, _final_spin_override=0.83 + step)
    minus = model.time_at_frequency(270.0, 40.0, 15.0, 0.3, -0.2, _final_spin_override=0.83 - step)
    np.testing.assert_allclose(float(derivative), float((plus - minus) / (2 * step)), rtol=2e-7)


def test_remnant_inverse_jax_jit_and_gradient():
    import jax
    import jax.numpy as jnp
    from jaxwave import IMRPhenomT

    def root(af):
        return IMRPhenomT().time_at_frequency(270.0, 40.0, 15.0, 0.3, -0.2,
                                             _final_spin_override=af)

    af = jnp.asarray(0.83)
    np.testing.assert_allclose(np.asarray(jax.jit(root)(af)), np.asarray(root(af)), rtol=1e-11)
    analytic = jax.grad(root)(af)
    finite_difference = (root(af + 1e-5) - root(af - 1e-5)) / 2e-5
    np.testing.assert_allclose(np.asarray(analytic), np.asarray(finite_difference), rtol=2e-7)


@pytest.mark.parametrize("backend", ["torchwave", "jaxwave"])
def test_remnant_override_preserves_merger_and_ringdown_join(backend):
    model = importlib.import_module(backend).IMRPhenomT()
    times = np.array([-0.1, -0.01, -0.001, 0.0, 0.003, 0.01])
    kwargs = dict(mass1=60.0, mass2=6.0, chi1=0.8, chi2=0.1,
                  inclination=0.7, f_ref=20.0)
    aligned = model.time_domain(times, **kwargs)
    modified = model.time_domain(times, **kwargs, _final_spin_override=0.9)
    for old, new in zip(aligned, modified):
        old, new = np.asarray(old), np.asarray(new)
        np.testing.assert_allclose(new[:4], old[:4], rtol=1e-12, atol=0)
        assert np.linalg.norm(new[4:] - old[4:]) > 1e-3 * np.linalg.norm(old[4:])


@pytest.mark.parametrize("backend", ["torchwave", "jaxwave"])
def test_remnant_override_sets_the_frequency_domain_limit(backend):
    model = importlib.import_module(backend).IMRPhenomT()
    args = (40.0, 15.0, 0.3, -0.2)
    original_limit = float(np.asarray(model.frequency(np.array([1.0]), *args))[0])
    new_limit = float(np.asarray(model.frequency(np.array([1.0]), *args,
                                                _final_spin_override=0.9))[0])
    assert new_limit > original_limit
    target = (original_limit + new_limit) / 2
    root = model.time_at_frequency(target, *args, _final_spin_override=0.9)
    actual = model.frequency(np.array([float(root)]), *args, _final_spin_override=0.9)
    np.testing.assert_allclose(np.asarray(actual), target, rtol=2e-13)
    with pytest.raises(ValueError, match="above the supported"):
        model.time_at_frequency(new_limit * 1.01, *args, _final_spin_override=0.9)
