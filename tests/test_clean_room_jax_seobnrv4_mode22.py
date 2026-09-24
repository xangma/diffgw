"""Independent special-function checks and Torch/JAX pre-NQC component parity."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from scipy.special import digamma, loggamma

from jaxwave.approximants.eob.seobnrv4_mode22 import (
    seobnrv4_published_delta22 as jax_delta,
    seobnrv4_published_mode22 as jax_mode,
    seobnrv4_published_phase22 as jax_phase,
    tail22_phase as jax_tail,
)
from jaxwave.approximants.eob.published_waveform import (
    published_mode22_and_frequency as jax_mode_frequency,
)
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as JaxFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as JaxHam
from torchwave.approximants.eob.published_waveform import (
    published_mode22_and_frequency as torch_mode_frequency,
)
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as TorchFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as TorchHam
from torchwave.approximants.eob.seobnrv4_mode22 import (
    seobnrv4_published_delta22 as torch_delta,
    seobnrv4_published_mode22 as torch_mode,
    tail22_phase as torch_tail,
)

jax.config.update("jax_enable_x64", True)


def t(value):
    return torch.as_tensor(value, dtype=torch.float64)


def source_choices(**kwargs):
    choices = dict(hreal_over_M=.98, x1=.7, x2=.3, chi1=.6, chi2=-.2,
                   spin_v6="bohe_total_energy", include_pan_v8=True,
                   include_positive_chi_540=False)
    choices.update(kwargs)
    return choices


def test_tail_continuous_phase_and_derivative_against_scipy():
    omega = np.geomspace(1e-9, 2, 41)
    energy = np.array([.8, 1., 1.2])
    w, e = omega[:, None], energy[None, :]
    expected = np.imag(loggamma(3-4j*w*e)
                       +4j*w*e*np.log(8*w/math.sqrt(math.e)))
    actual = jax_tail(jnp.asarray(w), hreal_over_M=jnp.asarray(e))
    np.testing.assert_allclose(actual, expected, rtol=4e-14, atol=4e-14)
    np.testing.assert_allclose(jax.jit(jax_tail)(jnp.asarray(w), hreal_over_M=jnp.asarray(e)),
                               actual, rtol=4e-14, atol=4e-14)
    assert float(actual[-1].max()) > math.pi
    w0, e0 = .09, .97
    dw = jax.grad(jax_tail, argnums=0)(w0, hreal_over_M=e0)
    de = jax.grad(lambda x: jax_tail(w0, hreal_over_M=x))(e0)
    psi = np.real(digamma(3-4j*w0*e0))
    np.testing.assert_allclose(dw, 4*e0*(np.log(8*w0/math.sqrt(math.e))+1-psi), rtol=3e-13)
    np.testing.assert_allclose(de, 4*w0*(np.log(8*w0/math.sqrt(math.e))-psi), rtol=3e-13)


@pytest.mark.parametrize("spin_v6", ["none", "bohe_total_energy", "pan_test_particle",
                                      "bohe_plus_pan"])
@pytest.mark.parametrize("include_pan_v8,include_positive_chi_540",
                         [(False, False), (True, False), (False, True), (True, True)])
def test_all_explicit_phase_choices_match_torch(spin_v6, include_pan_v8,
                                                  include_positive_chi_540):
    omega = np.array([.003, .02, .08])
    choices = source_choices(spin_v6=spin_v6, include_pan_v8=include_pan_v8,
                             include_positive_chi_540=include_positive_chi_540)
    expected = torch_delta(t(omega), **choices).numpy()
    actual = jax_delta(jnp.asarray(omega), **choices)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(jax.jit(jax_delta, static_argnames=(
        "spin_v6", "include_pan_v8", "include_positive_chi_540"))(
            jnp.asarray(omega), **choices), actual, rtol=2e-14, atol=2e-14)


def test_complex_mode_batch_dtype_and_gradient_parity():
    omega = np.array([[.01, .02, .06]])
    x1 = np.array([[.5], [.8]])
    chi1, chi2 = np.array([[.2], [-.6]]), np.array([[-.1], [.3]])
    choices = source_choices(x1=x1, x2=1-x1, chi1=chi1, chi2=chi2)
    amplitude = dict(omega_zero_radial=omega*.97, heff_over_mu=.91,
                     hreal_over_M=.98, x1=x1, x2=1-x1, chi1=chi1, chi2=chi2)
    phi = np.array([[.3], [.8]])
    expected_delta = torch_delta(t(omega), **{k: t(v) if isinstance(v, np.ndarray)
                                                else v for k, v in choices.items()})
    expected = torch_mode(t(omega), t(phi), residual_phase=expected_delta,
                          **{k: t(v) if isinstance(v, np.ndarray) else v
                             for k, v in amplitude.items()}).numpy()
    delta = jax_delta(jnp.asarray(omega), **choices)
    actual = jax_mode(jnp.asarray(omega), jnp.asarray(phi), residual_phase=delta,
                      **amplitude)
    assert actual.shape == (2, 3) and actual.dtype == jnp.complex128
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(jax.jit(jax_mode)(
        jnp.asarray(omega), jnp.asarray(phi), residual_phase=delta, **amplitude),
        actual, rtol=3e-14, atol=3e-14)

    def phase_at_frequency(w):
        residual = jax_delta(w, **source_choices())
        return jax_phase(w, .4, hreal_over_M=.98, residual_phase=residual)

    derivative = jax.jit(jax.grad(phase_at_frequency))(.028)
    step = 1e-6
    finite = (phase_at_frequency(.028+step)-phase_at_frequency(.028-step))/(2*step)
    np.testing.assert_allclose(derivative, finite, rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(jax_tail(jnp.asarray(omega), hreal_over_M=.98),
                               torch_tail(t(omega), hreal_over_M=.98).numpy(),
                               rtol=3e-14, atol=3e-14)


def test_invalid_choices_and_domains_fail_closed():
    with pytest.raises(ValueError):
        jax_delta(jnp.asarray(.02), **source_choices(spin_v6="default"))
    with pytest.raises(ValueError):
        jax_delta(jnp.asarray(.02), **source_choices(include_pan_v8=1))
    with pytest.raises(ValueError):
        jax_delta(jnp.asarray(.02), **source_choices(x1=.8, x2=.3))
    with pytest.raises(ValueError):
        jax_tail(jnp.asarray(0.), hreal_over_M=1.)
    assert np.isnan(jax.jit(jax_tail)(jnp.asarray(0.), hreal_over_M=1.))


@pytest.mark.parametrize("delta,chi1,chi2,r,pr_star,pphi", [
    (.2, 0., 0., 12., -.02, 3.8),
    (.6, .7, -.3, 10., -.015, 3.4),
])
def test_full_vector_field_mode_and_frequency_match_torch(
    delta, chi1, chi2, r, pr_star, pphi,
):
    nu = (1-delta**2)/4
    jham = JaxHam(nu, chi1, chi2, delta=delta)
    tham = TorchHam(t(nu), t(chi1), t(chi2), delta=t(delta))
    jflux = JaxFlux(spin_truncation="tabulated")
    tflux = TorchFlux(spin_truncation="tabulated")
    state = [r, .3, pr_star, pphi]
    choices = dict(spin_v6="bohe_total_energy", include_pan_v8=False,
                   include_positive_chi_540=False)
    jm, jf = jax_mode_frequency(jnp.asarray(state), jham, jflux, **choices)
    tm, tf = torch_mode_frequency(t(state), tham, tflux, **choices)
    np.testing.assert_allclose(jm, tm.detach().numpy(), rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(jf, tf.detach().numpy(), rtol=3e-11, atol=3e-11)
    assert float(jf) > 0
    if delta == .2:
        compiled = jax.jit(lambda value: jax_mode_frequency(value, jham, jflux, **choices))
        cm, cf = compiled(jnp.asarray(state))
        np.testing.assert_allclose(cm, jm, rtol=3e-13, atol=3e-13)
        np.testing.assert_allclose(cf, jf, rtol=3e-13, atol=3e-13)


def test_same_state_mode_and_frequency_keep_jax_derivatives():
    ham = JaxHam(.24, .3, -.1, delta=.2)
    flux = JaxFlux(spin_truncation="tabulated")
    choices = dict(spin_v6="bohe_total_energy", include_pan_v8=False,
                   include_positive_chi_540=False)

    def outputs(radius):
        state = jnp.stack((radius, radius*0+.3, radius*0-.02, radius*0+3.8))
        mode, frequency = jax_mode_frequency(state, ham, flux, **choices)
        return jnp.stack((mode.real, mode.imag, frequency))

    radius = jnp.asarray(12.)
    derivative = np.asarray(jax.jacfwd(outputs)(radius))
    step = 1e-4
    finite = np.asarray((outputs(radius+step)-outputs(radius-step))/(2*step))
    assert np.all(np.isfinite(derivative))
    np.testing.assert_allclose(derivative, finite, rtol=2e-7, atol=2e-9)
