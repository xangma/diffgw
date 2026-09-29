"""Supplied-input THM (2,2) ringdown amplitude gate against public carriers."""

import importlib
import math

import mpmath as mp
import numpy as np
import pytest

from diffgw_core.constants import C_SI, MPC_SI, MSUN_SEC
from diffgw_reference import phenomt_ringdown_amplitude as reference


# The first two cases take printed Eq. (27d); the last activates the native
# bounded-d2 convention and is only an ansatz/carrier diagnostic.
CASES = (
    ((30.0, 30.0, 0.0, 0.0), "paper"),
    ((35.0, 25.0, 0.7, -0.5), "paper"),
    ((60.0, 6.0, 0.9, 0.1), "bounded"),
)
TIMES = (0.0, 0.5, 2.0, 8.0, 25.0, 80.0, 160.0)
SLOPE_TIMES = (0.0, 2.0, 25.0, 80.0)
DISTANCE = 100.0
Y22 = math.sqrt(5 / (4 * math.pi))


def _native(backend):
    if backend == "torchwave":
        import torch

        def array(value):
            return torch.as_tensor(value, dtype=torch.float64)

    else:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        def array(value):
            return jnp.asarray(value, dtype=jnp.float64)

    return importlib.import_module(backend).IMRPhenomT(), array


def _scalar(value):
    return float(np.asarray(value))


@pytest.mark.parametrize(
    "supplied",
    (("0.085", "0.25", "0.39", "-0.32"), ("0.05", "0.18", "0.12", "0.45")),
)
def test_scalar_peak_slope_and_finite_difference(supplied):
    """Peak and derivative identities independently constrain the formula."""
    with mp.workdps(100):
        args = tuple(mp.mpf(x) for x in supplied)
        a1, a2, peak, shape = args
        d1, d2, d3, d4 = reference.coefficients(*args, dps=100)
        assert d2 == (a2 - a1) / 2
        assert mp.almosteq(d1 * d2 / mp.cosh(d3) ** 2, peak * a1, rel_eps=mp.mpf("1e-90"))
        assert mp.almosteq(d1 * mp.tanh(d3) + d4, peak, rel_eps=mp.mpf("1e-90"))
        assert mp.almosteq(reference.amplitude(0, *args, dps=100), peak, rel_eps=mp.mpf("1e-90"))
        assert abs(reference.derivative(0, *args, dps=100)) < mp.mpf("1e-90")
        step = mp.mpf("1e-25")
        for time in (mp.mpf("2"), mp.mpf("25")):
            finite_slope = (
                reference.amplitude(time + step, *args, dps=100)
                - reference.amplitude(time - step, *args, dps=100)
            ) / (2 * step)
            assert mp.almosteq(
                reference.derivative(time, *args, dps=100), finite_slope,
                rel_eps=mp.mpf("1e-35"), abs_eps=mp.mpf("1e-60"),
            )


@pytest.mark.parametrize("backend", ("torchwave", "jaxwave"))
@pytest.mark.parametrize("case,branch", CASES, ids=("equal", "moderate", "high-q-bound"))
def test_public_carrier_amplitude_derivative_and_peak(backend, case, branch):
    model, array = _native(backend)
    state = model._coeff(*map(array, case))
    amp_state = state[8]
    alpha1 = _scalar(amp_state[8])
    alpha2 = 2 * math.pi * _scalar(state[19])
    peak = _scalar(state[9])
    shape = _scalar(amp_state[6])
    actual_d2 = _scalar(amp_state[5])
    paper_d2 = (alpha2 - alpha1) / 2
    assert alpha1 > 0 and alpha2 > alpha1 and peak > 0 and actual_d2 > 0
    if branch == "paper":
        np.testing.assert_allclose(actual_d2, paper_d2, rtol=3e-14, atol=1e-15)
        override = None
    else:
        assert not np.isclose(actual_d2, paper_d2, rtol=1e-6)
        np.testing.assert_allclose(actual_d2, -0.5 * alpha1 / math.tanh(shape), rtol=3e-14)
        override = actual_d2
    supplied = (alpha1, alpha2, peak, shape)
    d1, _, _, d4 = reference.coefficients(*supplied, d2=override)
    np.testing.assert_allclose(_scalar(amp_state[4]), float(d1), rtol=3e-12, atol=1e-13)
    np.testing.assert_allclose(_scalar(amp_state[7]), float(d4), rtol=3e-12, atol=1e-13)

    mass_seconds = sum(case[:2]) * MSUN_SEC
    scale = mass_seconds / (DISTANCE * MPC_SI / C_SI) * Y22

    def public_amplitude(u):
        hp, hc = model.time_domain(
            array(u * mass_seconds).reshape(1), *case, distance=DISTANCE, inclination=0.0, phic=0.37
        )
        if backend == "torchwave":
            import torch

            return torch.sqrt(hp[0] ** 2 + hc[0] ** 2) / scale
        import jax.numpy as jnp

        return jnp.sqrt(hp[0] ** 2 + hc[0] ** 2) / scale

    hp, hc = model.time_domain(
        array([u * mass_seconds for u in TIMES]),
        *case, distance=DISTANCE, inclination=0.0, phic=0.37,
    )
    observed = np.hypot(np.asarray(hp), np.asarray(hc)) / scale
    expected = [float(reference.amplitude(u, *supplied, d2=override)) for u in TIMES]
    np.testing.assert_allclose(observed, expected, rtol=3e-11, atol=2e-13)
    np.testing.assert_allclose(observed[0], peak, rtol=3e-12, atol=1e-13)
    assert observed[0] > observed[1] > observed[2]

    if backend == "torchwave":
        import torch

        def slope(u):
            time = array(u).requires_grad_(True)
            return _scalar(torch.autograd.grad(public_amplitude(time), time)[0])
    else:
        import jax

        def slope(u):
            return _scalar(jax.grad(public_amplitude)(array(u)))

    observed_slope = [slope(u) for u in SLOPE_TIMES]
    expected_slope = [float(reference.derivative(u, *supplied, d2=override)) for u in SLOPE_TIMES]
    np.testing.assert_allclose(observed_slope, expected_slope, rtol=5e-10, atol=2e-12)
    np.testing.assert_allclose(observed_slope[0], 0.0, atol=2e-12)
