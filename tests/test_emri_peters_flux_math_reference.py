"""Strict CPU FP64 checks for the Peters leading-flux component only."""

import math

import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("mpmath")
torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from diffgw_reference import emri_peters_flux as ref
from jaxwave.approximants.emri_kludge import peters_mathews_radiation_reaction as jax_flux
from torchwave.approximants.emri_kludge import peters_mathews_radiation_reaction as torch_flux


def test_peters_change_of_orbital_variable():
    """The independent A-to-p chain rule gives both rational coefficients."""
    q, p, e = ref.Q, ref.P, ref.E
    expected_p = -sp.Rational(64, 5)*q*p**-3*(1-e**2)**sp.Rational(3, 2)*(
        1 + sp.Rational(7, 8)*e**2)
    expected_e = -sp.Rational(304, 15)*q*e*p**-4*(1-e**2)**sp.Rational(3, 2)*(
        1 + sp.Rational(121, 304)*e**2)
    assert sp.simplify(ref.semilatus_rectum_rate_expr() - expected_p) == 0
    assert sp.simplify(ref.eccentricity_rate_p_expr() - expected_e) == 0
    assert ref.evaluate(1e6, 10, 8, 0, component="e") == 0


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("p,e", ((8.0, 0.0), (8.0, 0.3), (14.0, 0.7)))
def test_native_flux_value_and_p_derivative(backend, p, e, reference_device):
    assert reference_device == "cpu", "EMRI Peters flux qualified on CPU only"
    m, mu = 1e6, 10.0
    for index, component in enumerate(("p", "e")):
        expected = float(ref.evaluate(m, mu, p, e, component=component))
        expected_derivative = float(ref.evaluate(
            m, mu, p, e, component=component, derivative="p"))
        if backend == "torch":
            parameter = torch.tensor(p, dtype=torch.float64, requires_grad=True)
            actual_tensor = torch_flux(m, mu, parameter, e)[index]
            actual = float(actual_tensor.detach())
            derivative = float(torch.autograd.grad(actual_tensor, parameter)[0])
        else:
            parameter = jnp.asarray(p, dtype=jnp.float64)
            def fn(value):
                return jax_flux(m, mu, value, e)[index]

            actual = float(fn(parameter))
            derivative = float(jax.grad(fn)(parameter))
        assert math.isclose(actual, expected, rel_tol=2e-13, abs_tol=1e-28)
        assert math.isclose(derivative, expected_derivative,
                            rel_tol=3e-13, abs_tol=1e-28)


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_native_eccentricity_derivative(backend, reference_device):
    assert reference_device == "cpu", "EMRI Peters flux qualified on CPU only"
    p, e = 11.0, 0.2
    expected = float(ref.evaluate(1e6, 10, p, e, component="e", derivative="e"))
    if backend == "torch":
        parameter = torch.tensor(e, dtype=torch.float64, requires_grad=True)
        actual = float(torch.autograd.grad(
            torch_flux(1e6, 10, p, parameter)[1], parameter)[0])
    else:
        actual = float(jax.grad(lambda value: jax_flux(1e6, 10, p, value)[1])(
            jnp.asarray(e, dtype=jnp.float64)))
    assert math.isclose(actual, expected, rel_tol=3e-13, abs_tol=1e-28)
