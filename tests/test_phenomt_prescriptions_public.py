"""End-to-end selection, masking and parity for T-family precession."""

import importlib

import numpy as np
import pytest

from diffgw_core.phenomt_options import resolve_precession


@pytest.mark.parametrize("method,version,expected", [
    ("nnlo", None, 10210), ("NNLO", 102, 10210),
    ("msa", None, 22310), ("msa", 223, 22310),
    ("nnlo", 10221, 10221), ("msa", 22311, 22311),
    ("numerical", None, 300),
])
def test_version_selection(method, version, expected):
    assert resolve_precession(method, version) == (method.lower(), expected)


@pytest.mark.parametrize("method,version", [
    ("msa", 10210), ("nnlo", 22310), ("numerical", 223),
    ("nnlo", 10210.0), ("nnlo", True), ("msa", 22410),
])
def test_incompatible_versions_rejected(method, version):
    with pytest.raises(ValueError, match="precession_version"):
        resolve_precession(method, version)


@pytest.mark.parametrize("backend", ["torchwave", "jaxwave"])
@pytest.mark.parametrize("name", ["IMRPhenomTP", "IMRPhenomTPHM"])
@pytest.mark.parametrize("method,version", [("nnlo", 10211), ("msa", 22310), ("numerical", 300)])
def test_prescription_public_sampling_and_direct_evaluation(backend, name, method, version):
    module = importlib.import_module(backend)
    kwargs = dict(mass1=40.0, mass2=15.0, chi1=0.3, chi2=-0.2,
                  spin1x=0.2, spin1y=0.1, spin2y=0.1,
                  inclination=0.8, phic=0.3, f_ref=30.0,
                  precession_method=method, precession_version=version)
    hp, hc, times = module.get_td_waveform(
        name, **kwargs, duration=0.125, delta_t=1/1024, return_times=True,
    )
    direct = getattr(module, name)().time_domain(times, **kwargs)
    for actual, expected in zip((hp, hc), direct):
        assert np.isfinite(np.asarray(actual)).all()
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-11, atol=1e-30)


@pytest.mark.parametrize("method,version", [("nnlo", 10221), ("msa", 22310), ("numerical", 300)])
def test_prescription_batched_waveform_backend_parity(method, version):
    import torch
    import jax.numpy as jnp
    import torchwave
    import jaxwave

    times = np.array([-0.1, -0.075, -0.02, -0.005, 0.0, 0.01])
    values = dict(mass1=[40.0, 30.0], mass2=[15.0, 25.0],
                  chi1=[0.3, 0.2], chi2=[-0.2, -0.1],
                  spin1x=[0.2, 0.0], spin1y=[0.1, 0.0], spin2y=[0.1, 0.0],
                  inclination=[0.8, 0.5], phic=[0.3, 0.2], f_ref=[30.0, 20.0])
    if method == "numerical":
        # Include precessing equal masses and a strongly spinning 10:1 binary.
        extra = dict(mass1=[30.0, 60.0], mass2=[30.0, 6.0],
                     chi1=[0.4, 0.8], chi2=[-0.3, 0.1],
                     spin1x=[0.2, 0.2], spin1y=[-0.1, 0.1], spin2y=[0.08, 0.0],
                     inclination=[0.9, 1.2], phic=[0.27, 0.1], f_ref=[20.0, 20.0])
        values = {key: value + extra[key] for key, value in values.items()}
        values["spin2x"] = [0.0, 0.0, -0.12, 0.1]
    results = []
    for module, array in ((torchwave, lambda x: torch.as_tensor(x, dtype=torch.float64)),
                          (jaxwave, lambda x: jnp.asarray(x, dtype=jnp.float64))):
        hp, hc = module.IMRPhenomTPHM().time_domain(
            array(times), **{k: array(v) for k, v in values.items()},
            precession_method=method, precession_version=version,
        )
        actual = np.asarray(hp) - 1j * np.asarray(hc)
        assert actual.shape == (len(values["mass1"]), len(times))
        assert np.isfinite(actual).all()
        results.append(actual)
    for actual, reference in zip(*results):
        assert np.linalg.norm(actual-reference)/np.linalg.norm(reference) <= 1e-11
        overlap = abs(np.vdot(actual, reference))/(np.linalg.norm(actual)*np.linalg.norm(reference))
        assert 1-overlap <= 1e-14


@pytest.mark.parametrize("method,version", [
    ("nnlo", 10200), ("nnlo", 10210), ("nnlo", 10211), ("nnlo", 10221),
    ("msa", 22300), ("msa", 22310), ("msa", 22311), ("msa", 22321),
    ("numerical", 300),
])
def test_prescription_query_grid_and_time_translation_invariance(method, version):
    import torch
    from torchwave import IMRPhenomTP

    model = IMRPhenomTP()
    times = torch.tensor([-0.12, -0.02, 0.005], dtype=torch.float64)
    extended = torch.tensor([-0.2, -0.12, -0.05, -0.02, 0.0, 0.005, 0.02], dtype=torch.float64)
    kwargs = dict(mass1=40.0, mass2=15.0, chi1=0.3, chi2=-0.2,
                  spin1x=0.2, spin1y=0.1, spin2y=0.1, f_ref=30.0,
                  inclination=0.8, phic=0.3,
                  precession_method=method, precession_version=version)
    expected = model.time_domain(times, **kwargs)
    enlarged = model.time_domain(extended, **kwargs)
    shifted = model.time_domain(times + 0.25, tc=0.25, **kwargs)
    for reference, actual, translated in zip(expected, enlarged, shifted):
        torch.testing.assert_close(actual[[1, 3, 5]], reference, rtol=1e-11, atol=1e-30)
        torch.testing.assert_close(translated, reference, rtol=1e-11, atol=1e-30)


@pytest.mark.parametrize("backend", ["torchwave", "jaxwave"])
def test_numerical_public_frequency_domain_validation(backend):
    module = importlib.import_module(backend)
    with pytest.raises(ValueError, match="f_lower >= 1 Hz"):
        module.get_td_waveform("IMRPhenomTP", mass1=40.0, mass2=15.0,
                               precession_method="numerical", f_lower=0.5,
                               duration=0.1, delta_t=1/1024)
    with pytest.raises(ValueError, match="reference|f_ref"):
        getattr(module, "IMRPhenomTP")().time_domain(
            np.array([-0.01, 0.0]), 40.0, 15.0, f_ref=10000.0,
            precession_method="numerical",
        )
