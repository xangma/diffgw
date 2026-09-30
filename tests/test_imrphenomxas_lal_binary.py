"""Frozen whole-waveform XAS comparison with a pinned public LAL binary.

This checks the carrier convention, including its supplied calibration, rather
than proving that every carrier choice is the literal printed paper equation.
No LAL source or coefficients are used.
"""

from importlib.metadata import version
import importlib
import os

import numpy as np
import pytest
from diffgw_core.constants import MSUN_SEC


LALSUITE_VERSION = "7.26.15"
RELATIVE_COMPLEX_L2_MAX = 1e-10
# Existing frozen catalog case and two off-catalog holdouts in parity_harness.
# (name, m1, m2, chi1, chi2, f_ref, inclination, phic, distance, df)
CASES = (
    ("catalog", 35., 30., .2, -.3, 20., .6, .4, 500., .125),
    ("unequal_positive", 45., 20., .35, .6, 35., .9, .71, 680., .125),
    ("unequal_negative", 90., 30., -.4, -.6, 27., 1.2, .71, 680., .125),
)


def _lal_modules():
    try:
        import lal
        import lalsimulation as lalsim
    except ImportError as exc:
        if os.environ.get("DIFFGW_REQUIRE_LAL_BINARY") == "1":
            pytest.fail(f"required pinned LALSuite binary unavailable: {exc}")
        pytest.skip(f"optional LALSuite binary unavailable: {exc}")
    assert version("lalsuite") == LALSUITE_VERSION
    return lal, lalsim


def _backend_module(name):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        if os.environ.get("DIFFGW_REQUIRE_LAL_BINARY") == "1":
            pytest.fail(f"required {name} backend unavailable: {exc}")
        pytest.skip(f"optional {name} backend unavailable: {exc}")


def _public_binary(frequencies, case):
    lal, lalsim = _lal_modules()
    _, m1, m2, chi1, chi2, f_ref, inclination, phic, distance, _ = case
    grid = lal.CreateREAL8Vector(len(frequencies))
    grid.data[:] = frequencies
    hp, hc = lalsim.SimInspiralChooseFDWaveformSequence(
        phic, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
        0., 0., chi1, 0., 0., chi2, f_ref,
        distance * 1e6 * lal.PC_SI, inclination,
        lal.CreateDict(), lalsim.IMRPhenomXAS, grid,
    )
    return np.asarray(hp.data.data), np.asarray(hc.data.data)


def _native(backend, frequencies, case):
    _, m1, m2, chi1, chi2, f_ref, inclination, phic, distance, _ = case
    kwargs = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2,
                  f_ref=f_ref, inclination=inclination, phi_c=phic,
                  distance=distance)
    if backend == "torch":
        torch = _backend_module("torch")
        from torchwave.approximants.imrphenom_xas import IMRPhenomXAS
        hp, hc = IMRPhenomXAS()(
            sample_frequencies=torch.as_tensor(frequencies, dtype=torch.float64),
            **kwargs,
        )
        return hp.detach().cpu().numpy(), hc.detach().cpu().numpy()
    jax = _backend_module("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxwave.approximants.imrphenom_xas import IMRPhenomXAS
    hp, hc = IMRPhenomXAS.evaluate(
        sample_frequencies=jnp.asarray(frequencies, dtype=jnp.float64), **kwargs,
    )
    return np.asarray(hp), np.asarray(hc)


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_whole_polarizations_against_public_binary(backend, case):
    _, m1, m2, _, _, f_ref, _, _, _, df = case
    frequencies = np.arange(20., 512., df, dtype=np.float64)
    assert f_ref in frequencies
    reference = _public_binary(frequencies, case)
    actual = _native(backend, frequencies, case)
    # Pratten et al. (2020), Sec. V, Eqs. (64)-(65), identify the public
    # waveform's arbitrary Mf=0.3 upper limit. The native carrier has no
    # wrapper cutoff, so compare the entire nonzero public-binary support.
    support = frequencies * (m1 + m2) * MSUN_SEC < 0.3
    assert np.any(support)
    for polarization, got, expected in zip(("plus", "cross"), actual, reference):
        assert got.shape == expected.shape == frequencies.shape
        assert got.dtype == expected.dtype == np.complex128
        assert np.all(np.isfinite(got)) and np.all(np.isfinite(expected))
        assert np.all(expected[~support] == 0)
        assert np.all(expected[support] != 0)
        norm = np.linalg.norm(expected[support])
        assert norm > 0
        error = float(np.linalg.norm(got[support] - expected[support]) / norm)
        print(f"{backend} {case[0]} {polarization} relative_complex_l2={error:.9g}")
        assert error <= RELATIVE_COMPLEX_L2_MAX
