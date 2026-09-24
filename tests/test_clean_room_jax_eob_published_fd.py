"""Independent finite-sum reference for the JAX EOB frequency projection."""

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from diffgw_core.constants import C_SI, MPC_SI, MSUN_SEC
from jaxwave.approximants.eob.published_waveform import published_eob22_to_fd

jax.config.update("jax_enable_x64", True)


def test_off_grid_polynomial_mode_projection_and_frequency_derivative():
    count, spacing, start = 23, .29, .61
    time = start + jnp.arange(count, dtype=jnp.float64)*spacing
    a, b = .17, .021
    mode = (a + b*time).astype(jnp.complex128)
    peak, tc = 1.4, .00023
    result = SimpleNamespace(
        time=time, mode=mode,
        trajectory=SimpleNamespace(peak=SimpleNamespace(time=jnp.asarray(peak))),
        attachment=SimpleNamespace(match_time=jnp.asarray(2.6)),
    )
    mass, distance, inclination = 50., 73., .72
    msec = mass*MSUN_SEC
    dt_seconds = spacing*msec
    frequency = (11.4+.19)/(128*dt_seconds)
    scale = msec*C_SI/(distance*MPC_SI)
    projection = math.sqrt(5/(16*math.pi))*(1+math.cos(inclination)**2)

    def actual(f):
        plus, cross = published_eob22_to_fd(
            result, jnp.atleast_1d(f), 30., 20., distance=distance,
            inclination=inclination, tc=tc,
        )
        return plus[0]/scale, cross[0]/scale

    def closed(f):
        q = jnp.exp(-2j*math.pi*f*dt_seconds)
        s0 = (1-q**count)/(1-q)
        s1 = q*(1-count*q**(count-1)+(count-1)*q**count)/(1-q)**2
        first_time = tc+(start-peak)*msec
        return (projection*dt_seconds*jnp.exp(-2j*math.pi*f*first_time)
                *((a+b*start)*s0+b*spacing*s1))

    plus, cross = jax.jit(actual)(frequency)
    np.testing.assert_allclose(plus, closed(frequency), rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(cross, 0., atol=1e-14)
    derivative = jax.grad(lambda f: actual(f)[0].real)(frequency)
    reference = jax.grad(lambda f: closed(f).real)(frequency)
    np.testing.assert_allclose(derivative, reference, rtol=2e-10, atol=2e-12)
