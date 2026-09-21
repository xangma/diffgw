"""Public API contracts for the higher-mode and precessing T family."""

import importlib

import numpy as np
import pytest


FAMILY = ("IMRPhenomTHM", "IMRPhenomTP", "IMRPhenomTPHM")


@pytest.mark.parametrize("backend", ["torchwave", "jaxwave"])
@pytest.mark.parametrize("name", FAMILY)
def test_native_dispatch_matches_model_and_unified_dispatch(backend, name):
    import diffgw

    module = importlib.import_module(backend)
    options = dict(mass1=40.0, mass2=15.0, chi1=0.3, chi2=-0.2,
                   distance=120.0, inclination=0.8, phic=0.3, tc=0.01,
                   f_ref=30.0)
    if name != "IMRPhenomTHM":
        options.update(spin1x=0.2, spin2y=0.1, precession_method="nnlo")
    sample = dict(duration=0.25, delta_t=1 / 1024, return_times=True)
    hp, hc, times = module.get_td_waveform(name, **options, **sample)
    expected = getattr(module, name)().time_domain(times, **options)
    for actual, reference in zip((hp, hc), expected):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(reference), rtol=1e-11, atol=1e-30)
        assert np.isfinite(np.asarray(actual)).all()
    unified = diffgw.get_td_waveform(name, backend="torch" if backend == "torchwave" else "jax",
                                   **options, **sample)
    np.testing.assert_array_equal(np.asarray(unified[0]), np.asarray(hp))
    assert hp.shape == hc.shape == times.shape == (256,)


@pytest.mark.parametrize("backend", ["torchwave", "jaxwave"])
@pytest.mark.parametrize("name", FAMILY)
def test_family_validation(backend, name):
    module = importlib.import_module(backend)
    kwargs = dict(mass1=40.0, mass2=15.0, duration=0.125)
    with pytest.raises(ValueError, match="time-domain"):
        module.get_fd_waveform(name, mass1=40.0, mass2=15.0)
    with pytest.raises(TypeError, match="Unsupported"):
        module.get_td_waveform(name, **kwargs, typo=1)
    with pytest.raises(ValueError, match="tidal"):
        module.get_td_waveform(name, **kwargs, lambda1=100)
    with pytest.raises(ValueError):
        module.get_td_waveform(name, **kwargs, spin1x=0.8, spin1z=0.8)
    with pytest.raises(ValueError):
        module.get_td_waveform(name, **kwargs, modes=[(3, 2)])


@pytest.mark.parametrize("backend", ["torchwave", "jaxwave"])
@pytest.mark.parametrize("name", ["IMRPhenomTP", "IMRPhenomTPHM"])
def test_unsupported_precession_is_explicit(backend, name):
    module = importlib.import_module(backend)
    with pytest.raises(ValueError, match="nnlo"):
        module.get_td_waveform(name, mass1=40.0, mass2=15.0, duration=0.125,
                               precession_method="unknown")


def test_family_catalogue():
    import torchwave

    for name in FAMILY:
        assert name in torchwave.list_approximants("time_domain")
        assert name not in torchwave.list_approximants("frequency_domain")
    assert "IMRPhenomTHM" in torchwave.list_approximants("higher_modes")
    assert "IMRPhenomTPHM" in torchwave.list_approximants("higher_modes")
    assert "IMRPhenomTP" in torchwave.list_approximants("precessing")


@pytest.mark.parametrize("backend", ["torchwave", "jaxwave"])
@pytest.mark.parametrize("name", ["IMRPhenomTP", "IMRPhenomTPHM"])
@pytest.mark.parametrize("method", ["nnlo", "msa", "numerical"])
def test_precessing_start_uses_same_remnant_as_waveform(backend, name, method):
    module = importlib.import_module(backend)
    precession = importlib.import_module(f"{backend}.approximants._phenomt_dispatch")
    if backend == "torchwave":
        import torch
        def array(x):
            return torch.as_tensor(x, dtype=torch.float64)
    else:
        import jax.numpy as jnp
        def array(x):
            return jnp.asarray(x, dtype=jnp.float64)

    mass1, mass2 = map(array, (40.0, 15.0))
    sx1, sy1, sz1, sx2, sy2, sz2 = map(array, (0.6, 0.1, 0.3, 0.1, 0.2, -0.2))
    af, prepared = precession.prepare_precession(method, mass1, mass2, sx1, sy1, sz1, sx2, sy2, sz2, array(20.0))
    start = module.IMRPhenomT().time_at_frequency(
        250.0, mass1, mass2, sz1, sz2, _final_spin_override=af,
    )
    kwargs = dict(mass1=mass1, mass2=mass2, chi1=sz1, chi2=sz2,
                  spin1x=sx1, spin1y=sy1, spin2x=sx2, spin2y=sy2,
                  inclination=0.8, f_ref=20.0, precession_method=method)
    hp, hc, times = module.get_td_waveform(
        name, **kwargs, f_lower=250.0, delta_t=1 / 8192, duration=0.03,
        t_pad=0.02, return_times=True,
    )
    raw = getattr(module, name)().time_domain(times, **kwargs)
    mask = np.asarray(times) >= np.asarray(start)
    assert mask.any() and (~mask).any()
    for actual, reference in zip((hp, hc), raw):
        np.testing.assert_allclose(np.asarray(actual), np.where(mask, np.asarray(reference), 0.0),
                                   rtol=1e-11, atol=1e-30)
