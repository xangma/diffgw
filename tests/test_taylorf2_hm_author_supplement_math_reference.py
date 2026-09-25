"""All-mode amplitude gate from the Mishra et al. author supplement.

The frozen values are generated from the checksum-pinned Mathematica file
published with arXiv:1601.05588v1.  Only leading amplitude magnitudes are in
scope: the supplement is not an external executable for diffGW's composed
complex waveform convention.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import taylorf2_hm as ref


DATA = Path(__file__).parent / "data" / "taylorf2_hm_mishra2016_samples.json"
MODES = ((2, 2), (2, 1), (3, 1), (3, 2), (3, 3), (4, 2), (4, 3), (4, 4))
INCLINATION = 0.91
PHI0 = 0.37


@pytest.fixture(scope="module")
def fixture():
    return json.loads(DATA.read_text())


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device == "mps":
        pytest.fail("The author-supplement gate requires FP64, unsupported on MPS.")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.taylorf2_hm import TaylorF2_HM

        if reference_device == "cuda" and not torch.cuda.is_available():
            pytest.fail("Requested CUDA qualification, but Torch CUDA is unavailable.")
        yield (
            "torch",
            TaylorF2_HM,
            lambda value: torch.as_tensor(
                value, dtype=torch.float64, device=reference_device
            ),
        )
        return

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jaxwave.approximants.taylorf2_hm import TaylorF2_HM

    platform = "gpu" if reference_device == "cuda" else "cpu"
    try:
        devices = jax.devices(platform)
    except RuntimeError as exc:
        pytest.fail(f"Requested {reference_device} qualification is unavailable: {exc}")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        with jax.default_device(devices[0]):
            yield "jax", TaylorF2_HM, lambda value: jnp.asarray(
                value, dtype=jnp.float64
            )
    finally:
        jax.config.update("jax_enable_x64", previous)


@lru_cache(maxsize=None)
def _paper_projection(mode):
    values = {
        ref.IOTA: sp.Rational(str(INCLINATION)),
        ref.PHI0: sp.Rational(str(PHI0)),
    }
    return tuple(
        complex(ref.evaluate(expression, values))
        for expression in ref.projection(*mode)
    )


def _carrier(mode_data, mode):
    """Invert the published polarization map without native private helpers."""
    d_plus, d_cross = _paper_projection(mode)
    h_plus, h_cross = (np.asarray(value) for value in mode_data[mode])
    norm = abs(d_plus) ** 2 + abs(d_cross) ** 2
    return (d_plus.conjugate() * h_plus + d_cross.conjugate() * h_cross) / norm


def _native_modes(backend, sample):
    name, model, array = backend
    options = dict(
        f=array([float(value) for value in sample["frequencies_hz"]]),
        mass1=array(float(sample["mass1_msun"])),
        mass2=array(float(sample["mass2_msun"])),
        chi1=array(0.0),
        chi2=array(0.0),
        distance=array(410.0),
        inclination=array(INCLINATION),
        phic=array(0.23),
        tc=array(0.0041),
        f_ref=array(43.0),
        phi0=array(PHI0),
        modes=MODES,
        f_isco_cutoff=False,
        return_modes=True,
    )
    return (model()(**options) if name == "torch" else model.evaluate(**options))[2]


def test_author_fixture_provenance_and_completeness(fixture):
    assert fixture["format"] == "taylorf2-hm-mishra2016-leading-amplitude-v1"
    assert fixture["source"]["bundle_sha256"] == (
        "f0b6c6810da95f45141dbb9f0c5cccec0ba0742f60b836dba7698c90a25d5154"
    )
    assert fixture["source"]["member_sha256"] == (
        "501ea53c29b77e061585540b9ccfd69a4cd0283e805ea1e9c24c38800e627a70"
    )
    assert tuple(fixture["leading_terms"]) == tuple(f"{l}{m}" for l, m in MODES)
    assert len(fixture["samples"]) == 3


def test_all_modes_match_author_supplement_leading_amplitudes(backend, fixture):
    for sample in fixture["samples"]:
        mode_data = _native_modes(backend, sample)
        quadrupole = np.abs(_carrier(mode_data, (2, 2)))
        for mode in MODES:
            actual = np.abs(_carrier(mode_data, mode)) / quadrupole
            expected = np.asarray(
                sample["expected_abs_hlm_over_h22"][f"{mode[0]}{mode[1]}"],
                dtype=np.float64,
            )
            np.testing.assert_allclose(actual, expected, rtol=8e-13, atol=2e-15)
