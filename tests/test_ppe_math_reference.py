"""Scoped ppE paper-to-scalar-to-Torch/JAX phase and dispersion gates."""

import importlib
import math

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
import torch

jax.config.update("jax_enable_x64", True)

from diffgw_reference import ppe as ref

t_ppe = importlib.import_module("torchwave.approximants.ppe")
j_ppe = importlib.import_module("jaxwave.approximants.ppe")


@pytest.mark.parametrize("z", [0.1, 0.7, 2.0])
def test_direct_paper_cosmology_with_h0_72(z):
    with mp.workdps(70):
        d0 = ref.cosmological_d0(z, h0_km_s_mpc=72)
        luminosity_distance = ref.cosmological_luminosity_distance(z, h0_km_s_mpc=72)
        from_supplied_distance = ref.d0_from_luminosity_distance(
            luminosity_distance / mp.mpf(ref._MPC), z
        )
        assert mp.almosteq(d0, from_supplied_distance)
        expected = float(ref.paper_dispersion_phase(70, d0, 1e17, z))
    distance_mpc = float(luminosity_distance / mp.mpf(ref._MPC))
    tv = t_ppe.modified_dispersion_phase_shift(torch.tensor(70.0, dtype=torch.float64), distance_mpc, 1e17, z)
    jv = j_ppe.modified_dispersion_phase_shift(jnp.asarray(70.0), distance_mpc, 1e17, z)
    np.testing.assert_allclose([tv.item(), float(jv)], expected, rtol=4e-13, atol=0)


@pytest.mark.parametrize("m1,m2,beta,b,f", [
    (30.0, 30.0, 0.003, -7, 35.0),
    (35.0, 20.0, -0.02, -2, 90.0),
    (12.0, 8.0, 0.1, 1, 160.0),
])
def test_paper_chirp_mass_phase_and_frontend_gradients(m1, m2, beta, b, f):
    with mp.workdps(70):
        mc = ref.chirp_mass_seconds(m1, m2) * mp.mpf("299792458")**3 / (
            mp.mpf("6.67430e-11") * mp.mpf("1.9884098706980507e30")
        )
        target = float(ref.paper_ppe_phase(f, m1, m2, beta, mp.mpf(b) / 3))
        d_beta = target / beta
        d_f = target * b / (3 * f)
        # A total-mass variable would be detectably different for nonzero b.
        if b != 0:
            wrong_mass = beta * (math.pi * (m1 + m2) * 4.925490947e-6 * f) ** (b / 3)
            assert not math.isclose(target, wrong_mass, rel_tol=1e-2)

    ft = torch.tensor(f, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(beta, dtype=torch.float64, requires_grad=True)
    tv = t_ppe.ppe_phase_shift(ft, float(mc), bt, b)
    tf, tb = torch.autograd.grad(tv, (ft, bt))
    def jf(freq, amp):
        return j_ppe.ppe_phase_shift(freq, float(mc), amp, b)

    jv, (jdf, jdb) = jax.jit(jax.value_and_grad(jf, (0, 1)))(
        jnp.asarray(f), jnp.asarray(beta)
    )
    np.testing.assert_allclose([tv.item(), float(jv)], target, rtol=3e-13, atol=0)
    np.testing.assert_allclose([tf.item(), float(jdf)], d_f, rtol=3e-13, atol=0)
    np.testing.assert_allclose([tb.item(), float(jdb)], d_beta, rtol=3e-13, atol=0)


@pytest.mark.parametrize("z", [0.0, 0.1, 0.7, 2.0])
def test_paper_cosmological_d0_and_dispersion_frontends(z):
    distance = 300.0
    wavelength = 1e17
    frequency = 70.0
    with mp.workdps(70):
        d0 = ref.d0_from_luminosity_distance(distance, z)
        expected = float(ref.paper_dispersion_phase(frequency, d0, wavelength, z))
        if z > 0:
            # Direct H0 form and D_L-eliminated form must be identical.
            h0 = ((1 + mp.mpf(z)) * mp.mpf("299792458") *
                  ref._cosmology_integral(z, 0, mp.mpf("0.3"), mp.mpf("0.7")) /
                  (mp.mpf(1000) * mp.mpf(distance)))
            assert mp.almosteq(ref.cosmological_d0(z, h0), d0)
            local_substitute = distance * 3.085677581491367e22 / (1 + z)
            assert not math.isclose(float(d0), local_substitute, rel_tol=1e-4)

    t = t_ppe.modified_dispersion_phase_shift(torch.tensor(frequency, dtype=torch.float64),
                                                distance, wavelength, z)
    j = jax.jit(j_ppe.modified_dispersion_phase_shift, static_argnames=("d_alpha",))(
        jnp.asarray(frequency), distance, wavelength, z
    )
    np.testing.assert_allclose([t.item(), float(j)], expected, rtol=4e-13, atol=0)
    t_explicit = t_ppe.modified_dispersion_phase_shift(torch.tensor(frequency, dtype=torch.float64),
                                                         distance, wavelength, z, float(d0))
    j_explicit = j_ppe.modified_dispersion_phase_shift(jnp.asarray(frequency), distance, wavelength, z, float(d0))
    np.testing.assert_allclose([t_explicit.item(), float(j_explicit)], expected, rtol=3e-13, atol=0)


def test_infinite_wavelength_batched_and_phase_derivatives():
    freq = np.array([30.0, 60.0, 120.0])
    tg = t_ppe.modified_dispersion_phase_shift(torch.tensor(freq), lambda_g=torch.tensor(float("inf")))
    jg = jax.jit(j_ppe.modified_dispersion_phase_shift)(jnp.asarray(freq), lambda_g=jnp.asarray(float("inf")))
    assert np.array_equal(tg.numpy(), np.zeros(3))
    assert np.array_equal(np.asarray(jg), np.zeros(3))
    wavelengths = np.array([np.inf, 1e17, 2e17])
    with mp.workdps(70):
        d0_02 = ref.d0_from_luminosity_distance(300, 0.2)
        target = np.array([0.0] + [float(ref.paper_dispersion_phase(ff, d0_02, ll, 0.2))
                                   for ff, ll in zip(freq[1:], wavelengths[1:])])
    tb = t_ppe.modified_dispersion_phase_shift(torch.tensor(freq), distance=300,
                                                lambda_g=torch.tensor(wavelengths), z=0.2)
    jb = jax.jit(j_ppe.modified_dispersion_phase_shift)(jnp.asarray(freq), distance=300,
                                                         lambda_g=jnp.asarray(wavelengths), z=0.2)
    np.testing.assert_allclose([*tb.numpy(), *np.asarray(jb)], [*target, *target], rtol=4e-13, atol=0)

    # Eq. (28): d(delta Psi)/df = -delta Psi/f at fixed D0.
    d0 = float(ref.d0_from_luminosity_distance(300, 0.5))
    ft = torch.tensor(80.0, dtype=torch.float64, requires_grad=True)
    tv = t_ppe.modified_dispersion_phase_shift(ft, lambda_g=1e17, z=0.5, d_alpha=d0)
    td = torch.autograd.grad(tv, ft)[0]
    def jf(ff):
        return j_ppe.modified_dispersion_phase_shift(ff, lambda_g=1e17, z=0.5, d_alpha=d0)

    jv, jd = jax.jit(jax.value_and_grad(jf))(jnp.asarray(80.0))
    np.testing.assert_allclose([td.item(), float(jd)], -tv.item() / 80, rtol=2e-14)
    np.testing.assert_allclose(float(jv), tv.item(), rtol=3e-14)


def test_redshift_and_wavelength_derivatives_against_scalar():
    with mp.workdps(70):
        def scalar(zz, ll):
            return ref.paper_dispersion_phase(80, ref.d0_from_luminosity_distance(300, zz), ll, zz)

        z_target = float(mp.diff(lambda zz: scalar(zz, mp.mpf("1e17")), mp.mpf("0.5")))
        l_target = float(mp.diff(lambda ll: scalar(mp.mpf("0.5"), ll), mp.mpf("1e17")))
    zt = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    lt = torch.tensor(1e17, dtype=torch.float64, requires_grad=True)
    tv = t_ppe.modified_dispersion_phase_shift(torch.tensor(80.0, dtype=torch.float64), 300.0, lt, zt)
    tz, tl = torch.autograd.grad(tv, (zt, lt))
    def jf(zz, ll):
        return j_ppe.modified_dispersion_phase_shift(jnp.asarray(80.0), 300.0, ll, zz)

    _, (jz, jl) = jax.jit(jax.value_and_grad(jf, (0, 1)))(jnp.asarray(0.5), jnp.asarray(1e17))
    np.testing.assert_allclose([tz.item(), float(jz)], z_target, rtol=5e-13, atol=0)
    np.testing.assert_allclose([tl.item(), float(jl)], l_target, rtol=5e-13, atol=0)


def test_carrier_adapter_sign_both_polarizations_and_d0_override():
    class TorchCarrier:
        def __call__(self, **kw):
            return torch.ones_like(kw["f"], dtype=torch.complex128) * (2 + 1j), torch.ones_like(kw["f"], dtype=torch.complex128) * (-1 + 3j)

    class JaxCarrier:
        def __call__(self, **kw):
            return jnp.ones_like(kw["f"], dtype=jnp.complex128) * (2 + 1j), jnp.ones_like(kw["f"], dtype=jnp.complex128) * (-1 + 3j)

    frequencies = np.array([40.0, 90.0, 180.0])
    d0 = float(ref.d0_from_luminosity_distance(300, 0.5))
    expected = np.array([complex(ref.carrier_multiplier(f, 30, 30, 0.001, -2, d0, 1e17, 0.5))
                         for f in frequencies])
    kwargs = dict(mass1=30.0, mass2=30.0, beta_ppe=0.001, b_ppe=-2,
                  lambda_g=1e17, distance=300.0, redshift=0.5, d_alpha=d0)
    thp, thc = t_ppe.ParameterizedPostEinstein(TorchCarrier())(torch.tensor(frequencies), **kwargs)
    jhp, jhc = j_ppe.ParameterizedPostEinstein(JaxCarrier())(jnp.asarray(frequencies), **kwargs)
    for actual, base in ((thp.numpy(), 2 + 1j), (thc.numpy(), -1 + 3j),
                         (np.asarray(jhp), 2 + 1j), (np.asarray(jhc), -1 + 3j)):
        np.testing.assert_allclose(actual / base, expected, rtol=5e-13, atol=5e-13)
