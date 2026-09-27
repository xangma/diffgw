"""Frozen, unaligned HM plus/cross comparison with the public LAL binary.

The input systems and bounds were committed before any holdout evaluation.
Only the public waveform API is used; the common time shift is applied by its
stated Fourier factor, with no fitted phase, time, or amplitude correction.
"""

from importlib.metadata import version
import os

import numpy as np
import pytest


LALSUITE_VERSION = "7.26.15"
FREQUENCIES = np.arange(20.0, 401.0, 1.0, dtype=np.float64)
RELATIVE_COMPLEX_L2_MAX = 1.0e-3
POINTWISE_COMPLEX_RELATIVE_MAX = 1.0e-2
STRONG_SUPPORT_FRACTION = 1.0e-2
# Frozen holdouts, unseen by the public-binary polarization probe:
# name, m1, m2 [Msun], chi1, chi2, f_ref [Hz], distance [Mpc],
# inclination, observer azimuth phi0, orbital phase phic [rad], tc [s].
CASES = (
    ("mixed_generic", 42.0, 23.0, 0.31, -0.18, 31.0, 610.0,
     0.73, 0.21, -0.08, 0.011),
    ("asymmetric_generic", 55.0, 17.0, -0.32, 0.41, 27.0, 930.0,
     1.29, -0.28, 0.37, -0.007),
    ("positive_spin_backside", 78.0, 31.0, 0.52, 0.07, 37.0, 740.0,
     2.15, 0.43, -0.19, 0.019),
)


def _require_module(name):
    try:
        return __import__(name)
    except ImportError as exc:
        if os.environ.get("DIFFGW_REQUIRE_LAL_BINARY") == "1":
            pytest.fail(f"required binary-oracle dependency {name} unavailable: {exc}")
        pytest.skip(f"optional binary-oracle dependency {name} unavailable: {exc}")


def _public_binary(case):
    lal = _require_module("lal")
    lalsim = _require_module("lalsimulation")
    assert version("lalsuite") == LALSUITE_VERSION
    (_, m1, m2, chi1, chi2, f_ref, distance,
     inclination, phi0, phic, tc) = case
    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
        0.0, 0.0, chi1, 0.0, 0.0, chi2,
        distance * 1.0e6 * lal.PC_SI, inclination,
        phi0 + phic, 0.0, 0.0, 0.0,
        1.0, FREQUENCIES[0], 512.0, f_ref,
        lal.CreateDict(), lalsim.IMRPhenomHM,
    )
    expected = []
    shift = np.exp(-2j * np.pi * FREQUENCIES * tc)
    for series in (hp, hc):
        assert series.deltaF == 1.0
        indices = (FREQUENCIES - series.f0) / series.deltaF
        assert np.array_equal(indices, np.rint(indices))
        expected.append(np.asarray(series.data.data)[indices.astype(int)] * shift)
    return tuple(expected)


def _native(backend, case):
    (_, m1, m2, chi1, chi2, f_ref, distance,
     inclination, phi0, phic, tc) = case
    kwargs = dict(
        mass1=m1, mass2=m2, chi1=chi1, chi2=chi2,
        distance=distance, inclination=inclination,
        phi0=phi0, phic=phic, tc=tc, f_ref=f_ref,
    )
    if backend == "torch":
        torch = _require_module("torch")
        from torchwave.approximants.imrphenom_hm import IMRPhenomHM

        hp, hc = IMRPhenomHM()(
            f=torch.as_tensor(FREQUENCIES, dtype=torch.float64, device="cpu"),
            **kwargs,
        )
        return hp.detach().cpu().numpy(), hc.detach().cpu().numpy()

    jax = _require_module("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxwave.approximants.imrphenom_hm import IMRPhenomHM

    with jax.default_device(jax.devices("cpu")[0]):
        hp, hc = IMRPhenomHM.evaluate(
            f=jnp.asarray(FREQUENCIES, dtype=jnp.float64), **kwargs,
        )
        return np.asarray(hp), np.asarray(hc)


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_complex_plus_cross_against_pinned_binary(backend, case):
    expected = _public_binary(case)
    measured = _native(backend, case)
    for polarization, actual, reference in zip(("plus", "cross"), measured, expected):
        assert actual.shape == reference.shape == FREQUENCIES.shape
        assert actual.dtype == reference.dtype == np.complex128
        assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
        difference = actual - reference
        relative_l2 = np.linalg.norm(difference) / np.linalg.norm(reference)
        strong = np.abs(reference) >= STRONG_SUPPORT_FRACTION * np.max(np.abs(reference))
        assert np.any(strong)
        pointwise = np.max(np.abs(difference[strong]) / np.abs(reference[strong]))
        print(
            f"{backend} {case[0]} {polarization}: "
            f"complex_L2={relative_l2:.9g} "
            f"strong_pointwise={pointwise:.9g} "
            f"strong_bins={np.count_nonzero(strong)}"
        )
        assert relative_l2 <= RELATIVE_COMPLEX_L2_MAX
        assert pointwise <= POINTWISE_COMPLEX_RELATIVE_MAX
