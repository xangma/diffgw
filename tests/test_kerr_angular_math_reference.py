"""Independent SymPy/mpmath angular projection checks for native Kerr modes.

Fixed CPU FP64 gates: spherical values 3e-13, angle derivatives 2e-12;
spheroidal values 3e-10, angle derivatives 2e-9, Torch c derivative 3e-6.
These thresholds were declared before evaluating the new reference checks.
"""

import pytest

mp = pytest.importorskip("mpmath")
pytest.importorskip("sympy")

from diffgw_reference import kerr_angular as ref


SPHERICAL_MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3))
SPHEROIDAL_CASES = ((2, 2, .42 - .10j), (3, 2, .58 - .12j))


def _complex(value):
    return complex(value)


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device != "cpu":
        pytest.fail("This scalar angular reference is qualified on CPU FP64 only")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.kerr_ringdown import _ringdown_projection
        from torchwave.approximants.kerr_angular import spheroidal_coefficients

        def array(value):
            return torch.tensor(value, dtype=torch.complex128 if isinstance(value, complex)
                                else torch.float64)

        def derivative(fun, theta, phi, wrt):
            angle = array(theta).requires_grad_(True)
            azimuth = array(phi).requires_grad_(True)
            values = fun(angle, azimuth)
            variable = (angle, azimuth)[wrt]
            return tuple(complex(
                torch.autograd.grad(value.real, variable, retain_graph=True)[0],
                torch.autograd.grad(value.imag, variable, retain_graph=True)[0],
            ) for value in values)

        yield request.param, array, _ringdown_projection, derivative, spheroidal_coefficients
    else:
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp
        from jaxwave.approximants.kerr_ringdown import (
            _ringdown_projection, spheroidal_coefficients,
        )
        previous = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)
        try:
            with jax.default_device(jax.devices("cpu")[0]):
                def array(value):
                    return jnp.asarray(value, dtype=jnp.complex128 if isinstance(value, complex)
                                       else jnp.float64)

                def derivative(fun, theta, phi, wrt):
                    tangent = ((1., 0.) if wrt == 0 else (0., 1.))
                    return tuple(map(complex, jax.jvp(
                        fun, (array(theta), array(phi)), tuple(map(array, tangent))
                    )[1]))

                yield request.param, array, _ringdown_projection, derivative, spheroidal_coefficients
        finally:
            jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("ell,m", SPHERICAL_MODES)
@pytest.mark.parametrize("theta,phi", ((.43, -.37), (1.19, .28), (2.24, .61)))
def test_spherical_projection_values_and_angle_derivatives(
    backend, ell, m, theta, phi,
):
    _, array, projection, derivative, _ = backend
    with mp.workdps(55):
        expected = ref.spherical_projection(ell, m, mp.mpf(theta), mp.mpf(phi))
        d_theta = tuple(mp.diff(lambda t: ref.spherical_projection(
            ell, m, t, mp.mpf(phi))[k], mp.mpf(theta)) for k in range(2))
        d_phi = tuple(mp.diff(lambda p: ref.spherical_projection(
            ell, m, mp.mpf(theta), p)[k], mp.mpf(phi)) for k in range(2))
    actual = projection(ell, m, array(0j), array(theta), array(phi), "spherical")
    for observed, target in zip(actual, expected):
        assert abs(_complex(observed) - complex(target)) < 3e-13
    fun = lambda angle, azimuth: projection(
        ell, m, array(0j), angle, azimuth, "spherical")
    for observed, target in zip(derivative(fun, theta, phi, 0), d_theta):
        assert abs(observed - complex(target)) < 2e-12
    for observed, target in zip(derivative(fun, theta, phi, 1), d_phi):
        assert abs(observed - complex(target)) < 2e-12


@pytest.mark.parametrize("ell,m,c", SPHEROIDAL_CASES)
def test_spheroidal_projection_and_angle_derivatives(backend, ell, m, c):
    _, array, projection, derivative, native_coefficients = backend
    theta, phi = 1.07, -.31
    with mp.workdps(50):
        eigenvalue, coefficients = ref.spheroidal_coefficients(
            ell, m, mp.mpc(c), ell_max=10, order=32)
        expected = ref.projection_from_coefficients(
            ell, m, coefficients, mp.mpf(theta), mp.mpf(phi))
        d_theta = tuple(mp.diff(lambda t: ref.projection_from_coefficients(
            ell, m, coefficients, t, mp.mpf(phi))[k], mp.mpf(theta)) for k in range(2))
        d_phi = tuple(mp.diff(lambda p: ref.projection_from_coefficients(
            ell, m, coefficients, mp.mpf(theta), p)[k], mp.mpf(phi)) for k in range(2))
    native_eigenvalue, _ = native_coefficients(ell, m, array(c))
    assert abs(_complex(native_eigenvalue) - complex(eigenvalue)) < 3e-10
    fun = lambda angle, azimuth: projection(
        ell, m, array(c), angle, azimuth, "spheroidal")
    actual = fun(array(theta), array(phi))
    for observed, target in zip(actual, expected):
        assert abs(_complex(observed) - complex(target)) < 3e-10
    for observed, target in zip(derivative(fun, theta, phi, 0), d_theta):
        assert abs(observed - complex(target)) < 2e-9
    for observed, target in zip(derivative(fun, theta, phi, 1), d_phi):
        assert abs(observed - complex(target)) < 2e-9


@pytest.mark.parametrize("ell,m,c", SPHEROIDAL_CASES)
def test_torch_spheroidal_real_oblateness_derivative(ell, m, c, reference_device):
    if reference_device != "cpu":
        pytest.fail("This scalar angular reference is qualified on CPU FP64 only")
    torch = pytest.importorskip("torch")
    from torchwave.approximants.kerr_ringdown import _ringdown_projection

    theta, phi = .89, .23
    h = mp.mpf("0.00001")
    with mp.workdps(50):
        plus = ref.spheroidal_projection(
            ell, m, mp.mpc(c) + h, mp.mpf(theta), mp.mpf(phi), ell_max=10, order=32)
        minus = ref.spheroidal_projection(
            ell, m, mp.mpc(c) - h, mp.mpf(theta), mp.mpf(phi), ell_max=10, order=32)
        expected = [(p - q) / (2 * h) for p, q in zip(plus, minus)]
    real_c = torch.tensor(c.real, dtype=torch.float64, requires_grad=True)
    native_c = torch.complex(real_c, torch.tensor(c.imag, dtype=torch.float64))
    actual = _ringdown_projection(
        ell, m, native_c, torch.tensor(theta, dtype=torch.float64),
        torch.tensor(phi, dtype=torch.float64), "spheroidal")
    for observed, target in zip(actual, expected):
        gradient = complex(
            torch.autograd.grad(observed.real, real_c, retain_graph=True)[0],
            torch.autograd.grad(observed.imag, real_c, retain_graph=True)[0],
        )
        assert abs(gradient - complex(target)) < 3e-6
