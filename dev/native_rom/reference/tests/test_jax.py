"""Independent paper reference against the JAX component implementation."""

import jax
import jax.numpy as jnp
import pytest

from dev.native_rom.reference.published import calibration, evaluate, mass_spin, peak_targets, ringdown_shape
from dev.native_rom.reference.ringdown import h22
from jaxwave.approximants.eob.merger_ringdown import merger_ringdown_22, seobnrv4_merger_shape
from jaxwave.approximants.eob.nqc import seobnrv4_nqc_parameters, seobnrv4_nqc_targets, seobnrv4_nqc_time_offset
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

jax.config.update("jax_enable_x64", True)


def scalar(x):
    return jnp.asarray(x, dtype=jnp.float64)


@pytest.mark.parametrize("m1,m2,s1,s2", [(30., 20., 0., 0.), (45., 5., .8, -.3), (20., 20., -.9, .7)])
def test_fits_against_reference(m1, m2, s1, s2):
    n, c = mass_spin(m1, m2, s1, s2)
    params = seobnrv4_nqc_parameters(*map(scalar, (m1, m2, s1, s2)))
    assert abs(float(params[0])-n) < 1e-15
    assert abs(float(params[1])-c) < 1e-15
    expected = evaluate(calibration(), n, c)
    hamiltonian = SEOBNRv4Hamiltonian(scalar(n), scalar(s1), scalar(s2))
    k, d_so, d_ss = hamiltonian.calibration_parameters(scalar(10.))
    for actual, key in zip((k, d_so, d_ss, seobnrv4_nqc_time_offset(*params)),
                           ("K", "d_SO", "d_SS", "peak_lag")):
        assert float(actual) == pytest.approx(float(expected[key]), rel=2e-14, abs=1e-12)
    targets = seobnrv4_nqc_targets(*params)
    for actual, key in zip(targets, peak_targets()):
        expected_value = evaluate(peak_targets(), n, c)[key]
        assert float(actual) == pytest.approx(float(expected_value), rel=2e-14, abs=1e-12)


def test_ringdown_against_reference():
    n, c = 0.24, 0.15
    times = scalar([0., .01, .5, 2., 10.])
    sigma = scalar(.55) + 1j*scalar(-.08)
    shape = seobnrv4_merger_shape(scalar(n), scalar(c))
    printed = evaluate(ringdown_shape(), n, c)
    for actual, key in zip(shape, ("c1f", "c2f", "d1f", "d2f")):
        assert float(actual) == pytest.approx(float(printed[key]), rel=2e-14, abs=1e-14)
    actual = merger_ringdown_22(times, sigma, scalar(.35), scalar(0.), scalar(.32), scalar(.4), shape)
    for t, sample in zip(times.tolist(), actual.tolist()):
        expected = complex(h22(t, nu=n, chi=c, sigma=complex(.55, -.08),
                               amplitude=.35, frequency=.32, phase=.4))
        assert sample == pytest.approx(expected, rel=3e-13, abs=3e-14)
