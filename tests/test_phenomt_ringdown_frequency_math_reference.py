"""Supplied-input dominant PhenomT ringdown gate against scalar paper algebra."""

import importlib

import mpmath as mp
import numpy as np
import pytest

from diffgw_reference import phenomt_ringdown_frequency as reference


CASES = (
    (30.0, 30.0, 0.0, 0.0),
    (35.0, 25.0, 0.7, -0.5),
    (60.0, 6.0, 0.9, 0.1),
)
TIMES = (0.0, 2.0, 8.0, 25.0, 80.0, 160.0)


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


def test_scalar_peak_integral_and_slope():
    """Independent identities guard the reference itself against sign drift."""
    supplied = (0.57, 0.36, 0.055, 1.3)
    with mp.workdps(80):
        args = tuple(mp.mpf(str(value)) for value in supplied)
        wr, wp, c2, c3 = args
        assert mp.almosteq(reference.omega(0, *args), wp)
        assert mp.almosteq(reference.phase_increment(0, *args), 0)
        for time in (mp.mpf("2"), mp.mpf("25"), mp.mpf("160")):
            assert wp < reference.omega(time, *args) < wr
            assert mp.almosteq(
                reference.phase_increment(time, *args),
                mp.quad(lambda u: reference.omega(u, *args), [0, time]),
                rel_eps=mp.mpf("1e-70"),
            )
            step = mp.mpf("1e-20")
            finite_slope = (reference.omega(time + step, *args) - reference.omega(time - step, *args)) / (2 * step)
            assert mp.almosteq(reference.domega(time, *args), finite_slope, rel_eps=mp.mpf("1e-35"))


@pytest.mark.parametrize("backend", ("torchwave", "jaxwave"))
@pytest.mark.parametrize("case", CASES, ids=lambda x: "-".join(map(str, x)))
def test_production_ringdown_frequency_phase_and_slope(backend, case):
    model, array = _native(backend)
    state = model._coeff(*map(array, case))
    omega_peak, c2, c3, omega_rd, c1 = map(
        _scalar, (state[13], state[14], state[15], state[30], state[31])
    )
    assert omega_rd > omega_peak and c2 > 0 and c3 > 0
    supplied = (omega_rd, omega_peak, c2, c3)
    np.testing.assert_allclose(c1, float(reference.c1_from_peak(*supplied)), rtol=3e-12, atol=1e-13)

    u = array(TIMES)
    native_omega, native_phase, _ = model._evaluate(u, state)
    native_slope = model._domega_full(u, state)
    expected_omega = [float(reference.omega(t, *supplied)) for t in TIMES]
    expected_phase = [float(reference.phase_increment(t, *supplied)) for t in TIMES]
    expected_slope = [float(reference.domega(t, *supplied)) for t in TIMES]
    np.testing.assert_allclose(np.asarray(native_omega), expected_omega, rtol=3e-12, atol=1e-13)
    np.testing.assert_allclose(
        np.asarray(native_phase) - np.asarray(native_phase)[0], expected_phase, rtol=3e-12, atol=2e-12
    )
    np.testing.assert_allclose(np.asarray(native_slope), expected_slope, rtol=3e-12, atol=1e-13)
    np.testing.assert_allclose(np.asarray(native_omega)[0], omega_peak, rtol=3e-12, atol=1e-13)
