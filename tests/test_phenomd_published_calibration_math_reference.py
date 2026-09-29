"""Public contract for the explicitly selected paper-precision D/HM variant."""

import importlib
import math

import numpy as np
import pytest


PARAMETERS = dict(mass1=35., mass2=20., chi1=.2, chi2=-.1,
                  distance=400., inclination=.7, phic=.23, tc=.003, f_ref=40.)
FREQUENCIES = (30., 70., 160., 300.)
HM_OPTIONS = dict(modes=((2, 2), (3, 3)), hm_mapping_transitions="london_2018",
                  hm_21_pn_variant="mishra_eq12b",
                  hm_32_pn_variant="mishra_eq12d",
                  hm_43_pn_variant="mishra_eq12g")


@pytest.fixture(params=("torch", "jax"))
def backend(request):
    name = request.param
    library = pytest.importorskip(name)
    if name == "jax":
        previous = library.config.x64_enabled
        library.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        def array(value):
            return jnp.asarray(value, dtype=jnp.float64)
    else:
        def array(value):
            return library.as_tensor(value, dtype=library.float64)
    d = importlib.import_module(f"{name}wave.approximants.imrphenomd").IMRPhenomD
    hm = importlib.import_module(f"{name}wave.approximants.imrphenom_hm").IMRPhenomHM
    try:
        yield name, array, d, hm
    finally:
        if name == "jax":
            library.config.update("jax_enable_x64", previous)


def _numpy(value):
    return np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)


def _call(backend, model_name, **options):
    _, array, d, hm = backend
    parameters = {name: array(value) for name, value in PARAMETERS.items()}
    parameters.update(options)
    if model_name == "IMRPhenomHM":
        parameters = {**HM_OPTIONS, **parameters}
    return (d() if model_name == "IMRPhenomD" else hm())(
        f=array(FREQUENCIES), **parameters)


def test_published_data_matches_independent_printed_transcription():
    """Compare every printed digit; do not obtain expected rows from runtime."""
    pytest.importorskip("sympy")
    from diffgw_core import phenomd_published_fits as data
    from diffgw_reference import imrphenomd_amplitude as amplitude
    from diffgw_reference import imrphenomd_phase as phase
    from diffgw_reference import imrphenomd_remnant as remnant

    rows = {**amplitude.PUBLISHED_ROWS, **phase.PUBLISHED_ROWS}
    assert data.TABLE_V_COEFFS == {
        name: tuple(float(value) for value in row) for name, row in rows.items()}
    assert data.FINAL_SPIN_ETA_POLY == (
        2 * math.sqrt(3), *map(float, remnant.SPIN_ETA_PRINTED[1:]))
    assert data.FINAL_SPIN_S_COEFFS == tuple(zip(
        map(float, remnant.SPIN_S_ETA_PRINTED),
        map(float, remnant.SPIN_S_ETA2_PRINTED)))
    assert data.ERAD_ETA_NUMERATOR == tuple(map(float, remnant.ENERGY_NS_PRINTED))
    assert data.ERAD_S_NUMERATOR == tuple(map(float, remnant.ENERGY_NUM_PRINTED))
    assert data.ERAD_S_DENOMINATOR == tuple(map(float, remnant.ENERGY_DEN_PRINTED))


@pytest.mark.parametrize("model_name", ("IMRPhenomD", "IMRPhenomHM"))
def test_calibration_default_is_inherited_and_published_is_distinct(backend, model_name):
    default = _call(backend, model_name)
    inherited = _call(backend, model_name, calibration="inherited")
    published = _call(backend, model_name, calibration="published")
    for baseline, explicit, printed in zip(default, inherited, published):
        np.testing.assert_array_equal(_numpy(baseline), _numpy(explicit))
        assert np.isfinite(_numpy(printed)).all()
        assert not np.array_equal(_numpy(printed), _numpy(explicit))


@pytest.mark.parametrize("model_name", ("IMRPhenomD", "IMRPhenomHM"))
def test_dispatch_forwards_published_calibration(backend, model_name):
    from diffgw import get_fd_waveform

    name, array, _, _ = backend
    options = HM_OPTIONS if model_name == "IMRPhenomHM" else {}
    parameters = {key: array(value) for key, value in PARAMETERS.items()}
    parameters["spin1z"] = parameters.pop("chi1")
    parameters["spin2z"] = parameters.pop("chi2")
    dispatched = get_fd_waveform(
        model_name, backend=name, sample_frequencies=array(FREQUENCIES),
        calibration="published", **parameters, **options)
    direct = _call(backend, model_name, calibration="published")
    for actual, expected in zip(dispatched, direct):
        np.testing.assert_allclose(_numpy(actual), _numpy(expected), rtol=2e-12, atol=1e-32)


def test_invalid_calibration_and_remnant_override_rejected(backend):
    _, array, d, hm = backend
    parameters = tuple(array(PARAMETERS[name]) for name in ("mass1", "mass2", "chi1", "chi2"))
    invalid_calls = (
        lambda: _call(backend, "IMRPhenomD", calibration="unrecognized"),
        lambda: _call(backend, "IMRPhenomHM", calibration="unrecognized"),
        lambda: d.remnant_properties(*parameters, calibration="unrecognized"),
        lambda: d.compute_slope_peak(*parameters, calibration="unrecognized"),
        lambda: d.eval_table_v("alpha4", array(.2), array(.1), calibration="unrecognized"),
        lambda: hm.mode_qnm_frequencies(3, 3, *parameters, calibration="unrecognized"),
    )
    for call in invalid_calls:
        with pytest.raises(ValueError, match="calibration"):
            call()
    def forbidden_provider(*_):
        pytest.fail("published calibration called the forbidden remnant override")
    with pytest.raises(ValueError, match="remnant_provider"):
        _call(backend, "IMRPhenomD", calibration="published", remnant_provider=forbidden_provider)
    with pytest.raises(ValueError, match="remnant_provider"):
        d.compute_slope_peak(*parameters, calibration="published", remnant_provider=forbidden_provider)


@pytest.mark.parametrize("carrier", ("xas", "custom"))
def test_published_hm_rejects_other_carriers(backend, carrier):
    name, array, _, hm = backend
    if carrier == "xas":
        xas = importlib.import_module(f"{name}wave.approximants.imrphenom_xas").IMRPhenomXAS
        base = xas() if name == "torch" else xas
    elif name == "torch":
        import torch

        class UnrelatedCarrier(torch.nn.Module):
            def forward(self, *args, **kwargs):
                pytest.fail("published calibration evaluated an unsupported carrier")
        base = UnrelatedCarrier()
    else:
        class UnrelatedCarrier:
            @staticmethod
            def evaluate(*args, **kwargs):
                pytest.fail("published calibration evaluated an unsupported carrier")
        base = UnrelatedCarrier
    with pytest.raises(ValueError, match="published"):
        hm(base_22=base)(f=array(FREQUENCIES), calibration="published", **PARAMETERS)


def test_published_hm_batch_matches_scalar_and_jax_jit(backend):
    """Exercise a non-22 map and the calibration choice inside native transforms."""
    name, array, _, _ = backend
    masses = array((35., 45.))
    batched = _call(backend, "IMRPhenomHM", mass1=masses, calibration="published")
    for index, mass in enumerate((35., 45.)):
        scalar = _call(backend, "IMRPhenomHM", mass1=array(mass), calibration="published")
        for actual, expected in zip(batched, scalar):
            np.testing.assert_allclose(_numpy(actual)[index], _numpy(expected), rtol=2e-10, atol=1e-31)
    if name == "jax":
        import jax

        compiled = jax.jit(lambda values: _call(
            backend, "IMRPhenomHM", mass1=values, calibration="published"))(masses)
        for actual, expected in zip(compiled, batched):
            np.testing.assert_allclose(_numpy(actual), _numpy(expected), rtol=2e-10, atol=1e-31)
