"""Frontend parity for the ansatz-derived SEOBNRv4 peak continuation."""

import jax
import jax.numpy as jnp
import numpy as np
import torch

from jaxwave.approximants.eob.merger_ringdown import (
    merger_ringdown_22 as jax_ringdown,
    seobnrv4_merger_shape as jax_shape,
    seobnrv4_peak_preserving_shape as jax_peak_shape,
)
from jaxwave.approximants.eob.attachment import attach_seobnrv4_22
from jaxwave.approximants.eob.nqc import seobnrv4_nqc_time_offset
from torchwave.approximants.eob.merger_ringdown import (
    merger_ringdown_22 as torch_ringdown,
    seobnrv4_merger_shape as torch_shape,
    seobnrv4_peak_preserving_shape as torch_peak_shape,
)


def test_peak_projection_batches_jit_gradients_and_frontend_parity():
    jax.config.update("jax_enable_x64", True)
    nu = np.asarray([8/81, .2])
    chi = np.asarray([.8, .3])
    damping = np.asarray([.04, .12])
    sigma = np.asarray([.52-.04j, .52-.12j])
    time = np.linspace(0, 4, 21)

    jnu, jchi = jnp.asarray(nu), jnp.asarray(chi)
    jsigma, jt = jnp.asarray(sigma), jnp.asarray(time)
    raw = jax_shape(jnu, jchi)
    projected = jax.jit(jax_peak_shape)(jsigma, raw)
    direct = jax_peak_shape(jsigma, raw)
    for compiled, eager in zip(projected, direct):
        np.testing.assert_allclose(compiled, eager, rtol=0, atol=0)
    np.testing.assert_allclose(projected.amplitude_rate[1], raw.amplitude_rate[1], rtol=0, atol=0)
    np.testing.assert_allclose(
        damping + 2*np.asarray(projected.amplitude_rate)*np.tanh(np.asarray(projected.amplitude_offset)),
        [0., damping[1] + 2*np.asarray(raw.amplitude_rate[1])*np.tanh(np.asarray(raw.amplitude_offset[1]))],
        rtol=0, atol=2e-17,
    )

    # On the projected branch, differentiating the analytic boundary
    # c1=-gamma/(2*tanh(c2)) gives dc1/dgamma=-1/(2*tanh(c2)).
    def rate(gamma):
        return jax_peak_shape(jnp.asarray(.52)-1j*gamma, jax_shape(jnu[0], jchi[0])).amplitude_rate
    derivative = jax.jit(jax.grad(rate))(jnp.asarray(.04))
    np.testing.assert_allclose(derivative, -1/(2*np.tanh(np.asarray(raw.amplitude_offset[0]))), rtol=2e-15)

    jh = jax.jit(jax_ringdown)(jt, jsigma, jnp.asarray(.36), jnp.asarray(0.),
                                jnp.asarray(.31), jnp.asarray(.7), projected)
    tshape = torch_peak_shape(
        torch.as_tensor(sigma, dtype=torch.complex128),
        torch_shape(torch.as_tensor(nu, dtype=torch.float64), torch.as_tensor(chi, dtype=torch.float64)),
    )
    th = torch_ringdown(torch.as_tensor(time, dtype=torch.float64),
                        torch.as_tensor(sigma, dtype=torch.complex128),
                        torch.tensor(.36, dtype=torch.float64), torch.tensor(0., dtype=torch.float64),
                        torch.tensor(.31, dtype=torch.float64), torch.tensor(.7, dtype=torch.float64), tshape)
    np.testing.assert_allclose(np.asarray(projected.amplitude_rate), tshape.amplitude_rate.numpy(),
                               rtol=2e-15, atol=0)
    np.testing.assert_allclose(np.asarray(jh), th.numpy(), rtol=2e-14, atol=3e-16)
    assert np.all(np.abs(np.asarray(jh)[0, 1:]) < np.abs(np.asarray(jh)[0, :-1]))


def test_attachment_has_no_postpeak_rebound():
    jax.config.update("jax_enable_x64", True)
    t = jnp.arange(0, 32.05, .1)
    r = 8-.1*t-.003*t**2
    pr = -.01-.003*t-.0007*t**2
    omega = .14-.00005*(t-25)**2
    state = jnp.stack((r, t*.1, pr, t*0+3), axis=-1)
    amplitude = .12+.004*t+.00002*t**2
    frequency = .2+.005*t+.00004*t**2
    phase = .3+.2*t+.0025*t**2+(.00004/3)*t**3
    mode = amplitude*jnp.exp(-1j*phase)
    nu, chi = jnp.asarray(.22), jnp.asarray(.1)
    sigma = jnp.asarray(.5-.04j)
    raw = jax_shape(nu, chi)
    assert .04+2*raw.amplitude_rate*jnp.tanh(raw.amplitude_offset) < 0
    match = 25+seobnrv4_nqc_time_offset(nu, chi)
    query = match+jnp.asarray([0., .001, .01, .1, 1.])
    attached = attach_seobnrv4_22(
        t, state, omega, mode, frequency, nu, chi, sigma, query)
    values = np.abs(np.asarray(attached.mode))
    assert np.all(values[1:] < values[0])
