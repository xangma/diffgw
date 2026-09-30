"""Frozen six-mode IMRPhenomHM comparison with the public LAL binary.

Only the binary mode API is used. The fixed Fourier/orbital-phase adapter was
identified on a separate exploratory system; none of these holdout systems is
used to determine a phase, time, or amplitude correction.
"""

from importlib.metadata import version
import os

import numpy as np
import pytest


LALSUITE_VERSION = "7.26.15"
MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3))
FREQUENCIES = np.arange(20.0, 401.0, 1.0, dtype=np.float64)
DELTA_F = 1.0
F_MAX = 512.0
# Frozen before any of these physical systems was run against the binary.
# (name, m1/Msun, m2/Msun, chi1, chi2, f_ref/Hz, distance/Mpc)
CASES = (
    ("moderate_q_mixed_spin", 38.0, 27.0, 0.15, -0.25, 29.0, 530.0),
    ("high_q_negative_spin", 58.0, 14.0, -0.45, 0.27, 33.0, 760.0),
    ("positive_spin", 70.0, 35.0, 0.61, 0.11, 23.0, 850.0),
)
RELATIVE_COMPLEX_L2_MAX = 1.0e-3
POINTWISE_COMPLEX_RELATIVE_MAX = 1.0e-2
STRONG_SUPPORT_FRACTION = 1.0e-2


def _require_module(name):
    try:
        return __import__(name)
    except ImportError as exc:
        if os.environ.get("DIFFGW_REQUIRE_LAL_BINARY") == "1":
            pytest.fail(f"required binary-oracle dependency {name} unavailable: {exc}")
        pytest.skip(f"optional binary-oracle dependency {name} unavailable: {exc}")


def _public_binary_modes(case):
    lal = _require_module("lal")
    lalsim = _require_module("lalsimulation")
    assert version("lalsuite") == LALSUITE_VERSION
    _, m1, m2, chi1, chi2, f_ref, distance = case
    head = lalsim.SimInspiralChooseFDModes(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
        0.0, 0.0, chi1, 0.0, 0.0, chi2,
        DELTA_F, FREQUENCIES[0], F_MAX, f_ref, 0.0,
        distance * 1.0e6 * lal.PC_SI, 0.0,
        lal.CreateDict(), lalsim.IMRPhenomHM,
    )
    observed = set()
    node = head
    while node is not None:
        observed.add((node.l, node.m))
        node = node.next
    assert observed == set(MODES) | {(ell, -m) for ell, m in MODES}

    expected = {}
    for ell, m in MODES:
        series = lalsim.SphHarmFrequencySeriesGetMode(head, ell, m)
        assert series is not None
        assert series.deltaF == DELTA_F
        negative_index = (-FREQUENCIES - series.f0) / series.deltaF
        positive_index = (FREQUENCIES - series.f0) / series.deltaF
        assert np.array_equal(negative_index, np.rint(negative_index))
        assert np.array_equal(positive_index, np.rint(positive_index))
        negative = np.asarray(series.data.data)[negative_index.astype(int)]
        positive = np.asarray(series.data.data)[positive_index.astype(int)]
        assert np.count_nonzero(negative) == len(FREQUENCIES)
        assert np.count_nonzero(positive) == 0
        # The public LAL positive-m mode occupies negative Fourier frequency.
        # The native positive-frequency carrier differs by one fixed orbital
        # phase origin: exp(-i*m*pi/2). No case-specific alignment is applied.
        expected[ell, m] = (-1j) ** m * np.conj(negative)
    return expected


def _native_modes(backend, case):
    _, m1, m2, chi1, chi2, f_ref, distance = case
    kwargs = dict(
        mass1=m1, mass2=m2, chi1=chi1, chi2=chi2,
        distance=distance, inclination=0.0, phic=0.0, tc=0.0,
        f_ref=f_ref, return_unprojected_modes=True,
    )
    if backend == "torch":
        torch = _require_module("torch")
        from torchwave.approximants.imrphenom_hm import IMRPhenomHM

        modes = IMRPhenomHM()(
            f=torch.as_tensor(FREQUENCIES, dtype=torch.float64), **kwargs,
        )[2]
        return {mode: value.detach().cpu().numpy() for mode, value in modes.items()}

    jax = _require_module("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxwave.approximants.imrphenom_hm import IMRPhenomHM

    with jax.default_device(jax.devices("cpu")[0]):
        modes = IMRPhenomHM.evaluate(
            f=jnp.asarray(FREQUENCIES, dtype=jnp.float64), **kwargs,
        )[2]
        return {mode: np.asarray(value) for mode, value in modes.items()}


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_six_complex_modes_against_pinned_binary(backend, case):
    expected = _public_binary_modes(case)
    actual = _native_modes(backend, case)
    assert tuple(actual) == MODES
    for mode in MODES:
        reference = expected[mode]
        measured = actual[mode]
        assert measured.shape == reference.shape == FREQUENCIES.shape
        assert np.all(np.isfinite(measured))
        difference = measured - reference
        relative_l2 = np.linalg.norm(difference) / np.linalg.norm(reference)
        strong = np.abs(reference) >= STRONG_SUPPORT_FRACTION * np.max(np.abs(reference))
        assert np.any(strong)
        pointwise = np.max(np.abs(difference[strong]) / np.abs(reference[strong]))
        missing_tail = (measured == 0) & (reference != 0)
        first_missing_hz = (
            FREQUENCIES[np.flatnonzero(missing_tail)[0]] if np.any(missing_tail) else None
        )
        print(
            f"{backend} {case[0]} {mode}: complex_L2={relative_l2:.9g} "
            f"strong_pointwise={pointwise:.9g} strong_bins={np.count_nonzero(strong)} "
            f"native_zero_bins={np.count_nonzero(missing_tail)} "
            f"first_native_zero_hz={first_missing_hz}"
        )
        assert relative_l2 <= RELATIVE_COMPLEX_L2_MAX
        assert pointwise <= POINTWISE_COMPLEX_RELATIVE_MAX
