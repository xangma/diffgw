"""Independent ppE equation-map and native carrier-composition checks."""

import importlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
import sympy as sp
import torch

from diffgw_reference import ppe_equations as eq


torch_ppe = importlib.import_module("torchwave.approximants.ppe")
jax_ppe = importlib.import_module("jaxwave.approximants.ppe")


def _integrals(z):
    z = mp.mpf(str(z))

    def e(x):
        return mp.sqrt(mp.mpf("0.3") * (1 + x)**3 + mp.mpf("0.7"))

    if z == 0:
        return mp.mpf(1), mp.mpf(1)  # The ratio has this continuous limit.
    return (mp.quad(lambda x: (1 + x)**-2 / e(x), [0, z]),
            mp.quad(lambda x: 1 / e(x), [0, z]))


_multiplier = sp.lambdify(
    (eq.f, eq.m1, eq.m2, eq.alpha, eq.a, eq.beta, eq.n, eq.d0,
     eq.wavelength, eq.z), eq.FULL_MULTIPLIER, modules="mpmath"
)


def _oracle(f, m1, m2, alpha, a, beta, n, distance, wavelength, z, d0=None):
    with mp.workdps(70):
        if d0 is None:
            i_minus2, i_zero = _integrals(z)
            d0 = mp.mpf(str(distance)) * mp.mpf(str(eq.MPC)) * i_minus2 / i_zero
        values = (f, m1, m2, alpha, a, beta, n, d0, wavelength, z)
        return complex(_multiplier(*(mp.mpf(str(v)) for v in values)))


def test_versioned_source_to_equations_map():
    assert eq.SPEC_VERSION == "ppe-supported-equations/v2"
    assert set(eq.SUPPORTED_CARRIERS) == {"TaylorF2", "IMRPhenomD", "IMRPhenomXAS"}
    assert all(row.source for row in eq.EQUATIONS.values())
    assert sp.simplify(eq.D0_DIRECT / eq.DL_DIRECT - eq.i_minus2 / eq.i_zero) == 0
    assert sp.simplify(eq.FULL_MULTIPLIER / eq.PHASE_ONLY - eq.AMPLITUDE) == 0
    assert sp.simplify(sp.diff(eq.AMPLITUDE, eq.alpha) - eq.U**eq.a) == 0
    assert sp.simplify(sp.diff(eq.PAPER_PHASE, eq.beta) - eq.U**(eq.n / 3)) == 0
    assert sp.simplify(sp.diff(eq.DISPERSION, eq.f) + eq.DISPERSION / eq.f) == 0
    assert sp.limit(eq.DISPERSION, eq.wavelength, sp.oo) == 0
    for pn, native in [(-1, -7), (0, -5), (1.5, -2), (2.5, 0), (3.5, 2)]:
        assert eq.PN_TO_NATIVE_EXPONENT.subs(eq.pn_order, sp.Rational(str(pn))) == native
        assert torch_ppe.PPE_PN_MAPPING[pn] == native
        assert jax_ppe.PPE_PN_MAPPING[pn] == native


@pytest.mark.parametrize("carrier_name", eq.SUPPORTED_CARRIERS)
@pytest.mark.parametrize("supplied_d0", [False, True])
def test_real_carrier_selector_ratio_and_gr_limit(carrier_name, supplied_d0):
    t_carriers = {
        "TaylorF2": "torchwave.approximants.taylorf2",
        "IMRPhenomD": "torchwave.approximants.imrphenomd",
        "IMRPhenomXAS": "torchwave.approximants.imrphenom_xas",
    }
    j_carriers = {key: value.replace("torchwave", "jaxwave") for key, value in t_carriers.items()}
    t_carrier = getattr(importlib.import_module(t_carriers[carrier_name]), carrier_name)()
    j_carrier = getattr(importlib.import_module(j_carriers[carrier_name]), carrier_name)()
    t_model = torch_ppe.ParameterizedPostEinstein(t_carrier)
    j_model = jax_ppe.ParameterizedPostEinstein(j_carrier)
    frequencies = np.array([30., 40., 55., 70.])
    ft, fj = torch.tensor(frequencies, dtype=torch.float64), jnp.asarray(frequencies)
    base_args = dict(mass1=25., mass2=25., distance=200., inclination=0.4)
    options = dict(alpha_ppe=0.4, a_ppe=0.5, beta_ppe=0.001, b_ppe=-2,
                   lambda_g=1e17, redshift=0.2)
    with mp.workdps(70):
        d0 = mp.mpf(200) * mp.mpf(str(eq.MPC)) * _integrals("0.2")[0] / _integrals("0.2")[1]
    if supplied_d0:
        options["d_alpha"] = float(d0)
    expected = np.array([_oracle(f, 25, 25, 0.4, 0.5, 0.001, -2,
                                 200, 1e17, 0.2, d0 if supplied_d0 else None)
                         for f in frequencies])
    t_base = t_carrier(f=ft, **base_args)
    # Match the wrapper's execution mode so carrier JIT roundoff cancels.
    j_base = jax.jit(lambda: j_carrier(f=fj, **base_args))()
    assert all(np.all(np.abs(np.asarray(h)) > 0) for h in (*t_base, *j_base))
    t_wave = t_model(ft, **base_args, **options)
    j_wave = jax.jit(lambda: j_model(fj, **base_args, **options))()
    for wave, base in (*zip(t_wave, t_base), *zip(j_wave, j_base)):
        np.testing.assert_allclose(np.asarray(wave / base), expected, rtol=5e-12, atol=5e-13)
    t_gr = t_model(ft, **base_args)
    j_gr = jax.jit(lambda: j_model(fj, **base_args))()
    for wave, base in (*zip(t_gr, t_base), *zip(j_gr, j_base)):
        np.testing.assert_allclose(np.asarray(wave / base), 1, rtol=0, atol=5e-13)


@pytest.mark.parametrize("frequencies", [[40., 55.], [30., 40., 55.]])
def test_batched_taylorf2_matches_individual_oracle(frequencies):
    from torchwave.approximants.taylorf2 import TaylorF2 as TorchTaylorF2
    from jaxwave.approximants.taylorf2 import TaylorF2 as JaxTaylorF2

    # The first case has B == N, so a mistaken pairwise broadcast can pass a
    # shape check while changing the physics.
    m1, m2 = np.array([25., 30.]), np.array([25., 20.])
    distances, redshifts = np.array([200., 250.]), np.array([0.2, 0.5])
    alphas, exponents = np.array([0.4, -0.15]), np.array([0.5, -0.5])
    betas, powers = np.array([0.001, -0.002]), np.array([-2., 1.])
    wavelengths = np.array([1e17, 2e17])
    expected = np.array([[_oracle(f, m1[b], m2[b], alphas[b], exponents[b],
                                  betas[b], powers[b], distances[b], wavelengths[b],
                                  redshifts[b]) for f in frequencies] for b in range(2)])
    ft, fj = torch.tensor(frequencies, dtype=torch.float64), jnp.asarray(frequencies)
    t_base_args = dict(mass1=torch.tensor(m1), mass2=torch.tensor(m2),
                       distance=torch.tensor(distances), inclination=0.4)
    j_base_args = dict(mass1=jnp.asarray(m1), mass2=jnp.asarray(m2),
                       distance=jnp.asarray(distances), inclination=0.4)
    t_options = dict(alpha_ppe=torch.tensor(alphas), a_ppe=torch.tensor(exponents),
                     beta_ppe=torch.tensor(betas), b_ppe=torch.tensor(powers),
                     lambda_g=torch.tensor(wavelengths), redshift=torch.tensor(redshifts))
    j_options = {key: jnp.asarray(value.numpy()) for key, value in t_options.items()}
    t_carrier, j_carrier = TorchTaylorF2(), JaxTaylorF2()
    t_base, j_base = t_carrier(f=ft, **t_base_args), j_carrier(f=fj, **j_base_args)
    t_wave = torch_ppe.ParameterizedPostEinstein(t_carrier)(ft, **t_base_args, **t_options)
    j_wave = jax.jit(jax_ppe.ParameterizedPostEinstein(j_carrier))(
        fj, **j_base_args, **j_options
    )
    for wave, base in (*zip(t_wave, t_base), *zip(j_wave, j_base)):
        assert wave.shape == expected.shape
        np.testing.assert_allclose(np.asarray(wave / base), expected, rtol=5e-12, atol=5e-13)


def test_full_multiplier_parameter_derivatives_and_transforms():
    class TorchCarrier:
        def __call__(self, **kw):
            return (torch.ones_like(kw["f"], dtype=torch.complex128) * (2 + 1j),
                    torch.ones_like(kw["f"], dtype=torch.complex128) * (-1 + 3j))

    class JaxCarrier:
        def __call__(self, **kw):
            return (jnp.ones_like(kw["f"], dtype=jnp.complex128) * (2 + 1j),
                    jnp.ones_like(kw["f"], dtype=jnp.complex128) * (-1 + 3j))

    symbols = (eq.m1, eq.m2, eq.alpha, eq.a, eq.beta, eq.n, eq.d0, eq.wavelength)
    values = (30., 20., 0.4, 0.5, 0.001, -2., 1e24, 1e17)
    with mp.workdps(70):
        expected = complex(_multiplier(55, *values[:6], values[6], values[7], 0.2))
        derivative = sp.lambdify((eq.f, *symbols, eq.z),
                                 [sp.diff(eq.FULL_MULTIPLIER, symbol) for symbol in symbols],
                                 modules="mpmath")
        expected_derivatives = np.array([complex(x) for x in derivative(
            mp.mpf(55), *(mp.mpf(str(v)) for v in values), mp.mpf("0.2"))])

    def t_ratio(*parameters):
        m1, m2, alpha, a, beta, n, d0, wavelength = parameters
        hp, _ = torch_ppe.ParameterizedPostEinstein(TorchCarrier())(
            torch.tensor([55.], dtype=torch.float64), mass1=m1, mass2=m2,
            alpha_ppe=alpha, a_ppe=a, beta_ppe=beta, b_ppe=n,
            d_alpha=d0, lambda_g=wavelength, redshift=0.2
        )
        return hp[0] / (2 + 1j)

    tv = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in values)
    tq = t_ratio(*tv)
    t_re = torch.autograd.grad(tq.real, tv, retain_graph=True)
    t_im = torch.autograd.grad(tq.imag, tv)
    t_derivatives = np.array([complex(re.item(), im.item()) for re, im in zip(t_re, t_im)])

    def j_ratio(*parameters):
        m1, m2, alpha, a, beta, n, d0, wavelength = parameters
        hp, _ = jax_ppe.ParameterizedPostEinstein(JaxCarrier())(
            jnp.asarray([55.]), mass1=m1, mass2=m2,
            alpha_ppe=alpha, a_ppe=a, beta_ppe=beta, b_ppe=n,
            d_alpha=d0, lambda_g=wavelength, redshift=0.2
        )
        q = hp[0] / (2 + 1j)
        return jnp.array([q.real, q.imag])

    jv = tuple(jnp.asarray(v) for v in values)
    jq, jac = jax.jit(jax.value_and_grad(
        lambda *args: j_ratio(*args)[0], argnums=tuple(range(8))
    ))(*jv)
    jim = jax.jit(jax.grad(lambda *args: j_ratio(*args)[1], argnums=tuple(range(8))))(*jv)
    j_derivatives = np.array([complex(float(re), float(im)) for re, im in zip(jac, jim)])
    np.testing.assert_allclose([complex(tq.item()), complex(float(jq), float(j_ratio(*jv)[1]))],
                               expected, rtol=5e-12, atol=5e-13)
    np.testing.assert_allclose(t_derivatives, expected_derivatives, rtol=5e-11, atol=0)
    np.testing.assert_allclose(j_derivatives, expected_derivatives, rtol=5e-11, atol=0)
    # Compilation is checked on the adapter without invoking carrier-specific
    # dynamical compilation paths.
    compiled = torch.compile(t_ratio, backend="eager")
    np.testing.assert_allclose(compiled(*tv).detach().numpy(), expected, rtol=5e-12, atol=5e-13)
