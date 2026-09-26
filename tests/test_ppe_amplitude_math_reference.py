"""Source-to-scalar-to-Torch/JAX ppE inspiral amplitude gates."""

import importlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
import torch

from diffgw_reference import ppe as ref

t_ppe = importlib.import_module("torchwave.approximants.ppe")
j_ppe = importlib.import_module("jaxwave.approximants.ppe")


@pytest.mark.parametrize("m1,m2,f,alpha,a", [
    (30.0, 30.0, 40.0, 0.2, 1.0),
    (35.0, 20.0, 80.0, -0.01, -0.5),
    (12.0, 8.0, 160.0, -2.0, 0.0),
    (30.0, 25.0, 90.0, 0.0, 2.0),
])
def test_paper_amplitude_values_and_gradients(m1, m2, f, alpha, a):
    with mp.workdps(70):
        mc_sec = ref.chirp_mass_seconds(m1, m2)
        mc = mc_sec * mp.mpf(ref._C)**3 / (mp.mpf(ref._G) * mp.mpf(ref._MSUN))
        u = mp.pi * mc_sec * mp.mpf(f)
        term = mp.mpf(alpha) * u ** mp.mpf(a)
        expected = float(ref.paper_ppe_amplitude_factor(f, m1, m2, alpha, a))
        gradients = [float(term * a / f), float(u ** mp.mpf(a)),
                     float(term * a / mc), float(term * mp.log(u))]

    ft = torch.tensor(f, dtype=torch.float64, requires_grad=True)
    at = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)
    mt = torch.tensor(float(mc), dtype=torch.float64, requires_grad=True)
    et = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    tv = t_ppe.ppe_amplitude_factor(ft, mt, at, et)
    tg = torch.autograd.grad(tv, (ft, at, mt, et))

    def jf(freq, coeff, mass, exponent):
        return j_ppe.ppe_amplitude_factor(freq, mass, coeff, exponent)

    jv, jg = jax.jit(jax.value_and_grad(jf, (0, 1, 2, 3)))(
        jnp.asarray(f), jnp.asarray(alpha), jnp.asarray(float(mc)), jnp.asarray(a)
    )
    np.testing.assert_allclose([tv.item(), float(jv)], expected, rtol=3e-13, atol=3e-15)
    np.testing.assert_allclose([g.item() for g in tg], gradients, rtol=3e-13, atol=3e-15)
    np.testing.assert_allclose([float(g) for g in jg], gradients, rtol=3e-13, atol=3e-15)
    if a != 0 and alpha != 0:
        # The old total-mass substitution does not pass a source-derived value.
        total_u = mp.pi * mp.mpf(ref._G) * mp.mpf(ref._MSUN) * (m1 + m2) * f / mp.mpf(ref._C)**3
        assert not mp.almosteq(mp.mpf(expected), 1 + mp.mpf(alpha) * total_u ** a)


def test_signed_factor_and_gr_reduction():
    with mp.workdps(70):
        assert ref.paper_ppe_amplitude_factor(40, 30, 30, -2, 0) == -1
        assert ref.paper_ppe_amplitude_factor(40, 30, 30, 0, -2) == 1
    f = np.array([40.0, 80.0, 160.0])
    for alpha, expected in [(-2.0, -np.ones(3)), (0.0, np.ones(3))]:
        tv = t_ppe.ppe_amplitude_factor(torch.tensor(f), 25.0, alpha, 0.0)
        jv = jax.jit(j_ppe.ppe_amplitude_factor)(jnp.asarray(f), 25.0, alpha, 0.0)
        np.testing.assert_array_equal(tv.numpy(), expected)
        np.testing.assert_array_equal(np.asarray(jv), expected)


def test_both_polarizations_paper_amplitude_and_carrier_sign():
    class TorchCarrier:
        def __call__(self, **kw):
            shape = (2, len(kw["f"]))
            return (torch.full(shape, 2 + 1j, dtype=torch.complex128),
                    torch.full(shape, -1 + 3j, dtype=torch.complex128))

    class JaxCarrier:
        def __call__(self, **kw):
            shape = (2, len(kw["f"]))
            return (jnp.full(shape, 2 + 1j, dtype=jnp.complex128),
                    jnp.full(shape, -1 + 3j, dtype=jnp.complex128))

    f = np.array([40.0, 90.0, 180.0])
    kwargs = dict(mass1=30.0, mass2=30.0, alpha_ppe=0.4, a_ppe=0.5,
                  beta_ppe=0.001, b_ppe=-2)
    expected = np.array([complex(ref.carrier_amplitude_multiplier(
        ff, 30, 30, 0.4, 0.5, 0.001, -2, 0, mp.inf, 0)) for ff in f])
    thp, thc = t_ppe.ParameterizedPostEinstein(TorchCarrier())(torch.tensor(f), **kwargs)
    jhp, jhc = jax.jit(j_ppe.ParameterizedPostEinstein(JaxCarrier()))(jnp.asarray(f), **kwargs)
    for actual, base in ((thp.detach().numpy(), 2 + 1j), (thc.detach().numpy(), -1 + 3j),
                         (np.asarray(jhp), 2 + 1j), (np.asarray(jhc), -1 + 3j)):
        np.testing.assert_allclose(actual / base, np.broadcast_to(expected, (2, 3)),
                                   rtol=5e-13, atol=5e-13)
    phase = np.array([float(ref.paper_ppe_phase(ff, 30, 30, 0.001, mp.mpf(-2) / 3))
                      for ff in f])
    amplitude = np.array([float(ref.paper_ppe_amplitude_factor(ff, 30, 30, 0.4, 0.5))
                          for ff in f])
    np.testing.assert_allclose(expected, amplitude * np.exp(-1j * phase), rtol=5e-13)
    assert not np.allclose(expected, amplitude * np.exp(1j * phase), rtol=1e-4)

    alpha_t = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    hp_t, _ = t_ppe.ParameterizedPostEinstein(TorchCarrier())(
        torch.tensor(f), **(kwargs | {"alpha_ppe": alpha_t})
    )
    grad_t = torch.autograd.grad(hp_t[0, 0].real, alpha_t)[0]

    model_j = j_ppe.ParameterizedPostEinstein(JaxCarrier())
    def real_plus(alpha):
        hp, _ = model_j(jnp.asarray(f), **(kwargs | {"alpha_ppe": alpha}))
        return hp[0, 0].real

    grad_j = jax.jit(jax.grad(real_plus))(jnp.asarray(0.4))
    with mp.workdps(70):
        u = mp.pi * ref.chirp_mass_seconds(30, 30) * f[0]
        expected_grad = complex((2 + 1j) * u ** mp.mpf("0.5") *
                                ref.carrier_multiplier(f[0], 30, 30, 0.001,
                                                       -2, 0, mp.inf, 0)).real
    np.testing.assert_allclose([grad_t.item(), float(grad_j)], expected_grad,
                               rtol=5e-13, atol=5e-13)
