"""Frozen CPU FP64 220 QNM and angular-input checks from a scalar MP solve.

The gates were declared in docs/math/kerr-qnm-scalar-qualification.md before
native-output comparison: |delta omega| <= 1e-8, |delta A| <= 1e-10,
|delta d omega/da| <= 2e-5. See that document for truncation gates.
"""

import pytest

mp = pytest.importorskip("mpmath")
pytest.importorskip("sympy")

from diffgw_reference.kerr_qnm import ScalarQNM220


# From scalar n=0 continued-fraction roots, depth 400, angular ell_max=10,
# 32-point Gauss--Legendre quadrature, 40 MP decimal digits.
ROOTS = {
    0.0: (0.3736716844180418 - 0.0889623156889357j, 4.0 + 0j),
    0.35: (0.4292636831773215 - 0.0873453099742386j,
           3.583266782196927 + 0.08839683825996633j),
    0.417: (0.4436541891564515 - 0.0867032854255785j,
            3.4821195962788583 + 0.10646506742509051j),
    0.7: (0.532600243551018 - 0.08079287315500766j,
          2.903166981873936 + 0.18315825271747033j),
    0.733: (0.5481090795307211 - 0.07941895882864333j,
            2.8089227767475298 + 0.1912267808558153j),
}
# Centered scalar-root differences with h=1e-4, same depth/basis/precision.
SPIN_DERIVATIVES = {
    0.35: 0.20273475343863015 + 0.008420932385907139j,
    0.7: 0.4448991263254646 + 0.037942904159581646j,
}


@pytest.fixture(scope="module")
def scalar():
    return ScalarQNM220(ell_max=10, order=32, dps=40)


@pytest.mark.parametrize("spin", ROOTS)
def test_scalar_root_reproduces_frozen_values_and_depth(scalar, spin):
    expected_w, expected_a = ROOTS[spin]
    w200, _ = scalar.root(spin, depth=200)
    w400, angular = scalar.root(spin, depth=400)
    assert abs(complex(w400) - expected_w) <= 2e-14
    assert abs(complex(angular) - expected_a) <= 2e-13
    assert abs(w400 - w200) <= mp.mpf("2e-12")
    assert abs(scalar.radial_residual(spin, w400, depth=400)) <= mp.mpf("1e-22")


@pytest.mark.parametrize("spin", (0.35, 0.7))
def test_scalar_angular_basis_converges(spin):
    coarse = ScalarQNM220(ell_max=8, order=32, dps=40)
    fine = ScalarQNM220(ell_max=10, order=32, dps=40)
    _, ac = coarse.root(spin, depth=400)
    _, af = fine.root(spin, depth=400)
    assert abs(ac - af) <= mp.mpf("2e-11")


@pytest.mark.parametrize("spin", (0.35, 0.7))
def test_scalar_centered_derivative_reproduces_frozen_values(scalar, spin):
    h = 1e-4
    wp, _ = scalar.root(spin + h, depth=400)
    wm, _ = scalar.root(spin - h, depth=400)
    derivative = complex((wp - wm) / (2 * h))
    assert abs(derivative - SPIN_DERIVATIVES[spin]) <= 2e-12


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("spin", ROOTS)
def test_native_frequency_and_angular_input(backend, spin, reference_device):
    if reference_device != "cpu":
        pytest.fail("Scalar QNM qualification is CPU FP64 only")
    expected_w, expected_a = ROOTS[spin]
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm
        from torchwave.approximants.kerr_angular import spheroidal_coefficients
        value = torch.tensor(spin, dtype=torch.float64)
        real, decay = dimensionless_qnm(2, 2, 0, value)
        omega = complex(float(real), -float(decay))
        angular, _ = spheroidal_coefficients(
            2, 2, torch.tensor(spin * omega, dtype=torch.complex128))
        actual_a = complex(angular.item())
    else:
        jax = pytest.importorskip("jax")
        previous = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)
        try:
            import jax.numpy as jnp
            from jaxwave.approximants import kerr_qnm_interp
            from jaxwave.approximants.kerr_ringdown import spheroidal_coefficients
            assert kerr_qnm_interp._KNOTS_220.dtype == jnp.float64, (
                "Kerr QNM spline data were imported before JAX FP64 was enabled"
            )
            with jax.default_device(jax.devices("cpu")[0]):
                real, decay = kerr_qnm_interp.evaluate_kerr_qnm_220(
                    jnp.asarray(spin, dtype=jnp.float64))
                omega = complex(float(real), -float(decay))
                angular, _ = spheroidal_coefficients(
                    2, 2, jnp.asarray(spin * omega, dtype=jnp.complex128))
                actual_a = complex(angular)
        finally:
            jax.config.update("jax_enable_x64", previous)
    assert abs(omega - expected_w) <= 1e-8
    assert abs(actual_a - expected_a) <= 1e-10


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("spin", SPIN_DERIVATIVES)
def test_native_spin_derivative_matches_scalar_root(backend, spin, reference_device):
    if reference_device != "cpu":
        pytest.fail("Scalar QNM qualification is CPU FP64 only")
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm
        value = torch.tensor(spin, dtype=torch.float64, requires_grad=True)
        real, decay = dimensionless_qnm(2, 2, 0, value)
        dre = torch.autograd.grad(real, value, retain_graph=True)[0]
        ddecay = torch.autograd.grad(decay, value)[0]
        actual = complex(float(dre), -float(ddecay))
    else:
        jax = pytest.importorskip("jax")
        previous = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)
        try:
            import jax.numpy as jnp
            from jaxwave.approximants import kerr_qnm_interp
            assert kerr_qnm_interp._KNOTS_220.dtype == jnp.float64, (
                "Kerr QNM spline data were imported before JAX FP64 was enabled"
            )
            with jax.default_device(jax.devices("cpu")[0]):
                value = jnp.asarray(spin, dtype=jnp.float64)
                dre = jax.grad(lambda a: kerr_qnm_interp.evaluate_kerr_qnm_220(a)[0])(value)
                ddecay = jax.grad(lambda a: kerr_qnm_interp.evaluate_kerr_qnm_220(a)[1])(value)
                actual = complex(float(dre), -float(ddecay))
        finally:
            jax.config.update("jax_enable_x64", previous)
    assert abs(actual - SPIN_DERIVATIVES[spin]) <= 2e-5
