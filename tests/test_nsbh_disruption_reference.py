"""Frozen independent paper gate for Thompson's NSBH disruption equations."""

from itertools import product

import numpy as np
import pytest

from diffgw_reference.nsbh_disruption import compactness, disruption, largest_tidal_root, tidal_polynomial


MASS_RATIOS = (1, 2, 5, 10, 15)
SPINS = (-0.5, 0.0, 0.5)
DEFORMABILITIES = (0, 1, 10, 300, 800, 2600, 5000)
CASES = tuple(product(MASS_RATIOS, SPINS, DEFORMABILITIES))
FREQUENCY_CASES = tuple(case for case in CASES if case[2] >= 300)
COMPACTNESS_ATOL = 2e-13
ROOT_RTOL = 2e-11
RESIDUAL_ATOL = 2e-12
FREQUENCY_RTOL = 2e-10


def _reference_values(cases):
    return [disruption(*case) for case in cases]


def _relative_error(actual, expected):
    return np.abs(np.asarray(actual) - np.asarray(expected)) / np.abs(np.asarray(expected))


def _normalized_residual(x, q, chi, c, residual):
    mu = q * c
    terms = (
        x**5,
        -3 * mu * x**4,
        2 * chi * np.sqrt(mu**3 * x**7),
        -3 * q * x**2,
        6 * q * mu * x,
        -3 * q * mu**2 * chi**2,
    )
    polynomial = sum(terms)
    scale = 1 + sum(np.abs(term) for term in terms)
    np.testing.assert_allclose(residual / scale, polynomial / scale, atol=RESIDUAL_ATOL, rtol=0)
    return np.abs(polynomial) / scale


def test_scalar_root_is_largest_positive_polynomial_root():
    """Companion roots independently audit rightmost-crossing selection."""
    for q, chi, lam in CASES:
        c = float(compactness(lam))
        mu = q * c
        coefficients = [1, 0, -3 * mu, 2 * chi * mu**1.5, 0, 0, -3 * q, 0, 6 * q * mu, 0, -3 * q * mu**2 * chi**2]
        roots = np.roots(coefficients)
        positive = [root.real**2 for root in roots if root.real > 0 and abs(root.imag) < 1e-7]
        assert positive, (q, chi, lam)
        selected = float(largest_tidal_root(q, chi, compactness(lam)))
        assert abs(selected - max(positive)) / selected < 1e-8, (q, chi, lam)
        assert abs(float(tidal_polynomial(selected, q, chi, c))) < 1e-8


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_native_compactness_and_root_against_paper_reference(backend):
    expected = _reference_values(CASES)
    q, chi, lam = (np.asarray(values, dtype=np.float64) for values in zip(*CASES))
    if backend == "torch":
        import torch
        from torchwave.approximants.imrphenom_nsbh import ns_compactness_from_lambda, solve_xi_tide

        def tensor(values):
            return torch.as_tensor(values, dtype=torch.float64)

        c_tensor = ns_compactness_from_lambda(tensor(lam))
        x_tensor, residual_tensor = solve_xi_tide(tensor(q), tensor(chi), c_tensor)
        c = c_tensor.detach().numpy()
        x = x_tensor.detach().numpy()
        residual = residual_tensor.detach().numpy()
    else:
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenom_nsbh import ns_compactness_from_lambda, solve_xi_tide

        c_tensor = ns_compactness_from_lambda(jnp.asarray(lam))
        x_tensor, residual_tensor = solve_xi_tide(jnp.asarray(q), jnp.asarray(chi), c_tensor)
        c, x, residual = (np.asarray(value) for value in (c_tensor, x_tensor, residual_tensor))

    assert np.all(np.isfinite(x)) and np.all(x > 0)
    compactness_error = np.max(np.abs(c - [float(item.compactness) for item in expected]))
    root_error = np.max(_relative_error(x, [float(item.xi) for item in expected]))
    residual_error = np.max(_normalized_residual(x, q, chi, c, residual))
    assert compactness_error <= COMPACTNESS_ATOL, (backend, compactness_error)
    assert root_error <= ROOT_RTOL, (backend, root_error)
    assert residual_error <= RESIDUAL_ATOL, (backend, residual_error)
    frequency_positions = lam >= 300
    solar_mass_seconds = 6.67430e-11 * 1.9884098706980507e30 / 299792458.0**3
    radius = (
        x[frequency_positions] * 1.4 * solar_mass_seconds * (1 - 2 * c[frequency_positions]) / c[frequency_positions]
    )
    radius_error = np.max(
        _relative_error(radius, [float(item.radius_sec) for item, keep in zip(expected, frequency_positions) if keep])
    )
    assert radius_error <= FREQUENCY_RTOL, (backend, radius_error)
    print(
        f"{backend}: compactness_abs={compactness_error:.3e} root_rel={root_error:.3e} residual={residual_error:.3e} radius_rel={radius_error:.3e}"
    )


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_native_tidal_frequency_against_paper_reference(backend):
    expected = _reference_values(FREQUENCY_CASES)
    assert all(float(item.frequency_hz) > 0 and np.isfinite(float(item.frequency_hz)) for item in expected)
    q, chi, lam = (np.asarray(values, dtype=np.float64) for values in zip(*FREQUENCY_CASES))
    m2 = np.full_like(q, 1.4)
    m1 = q * m2
    if backend == "torch":
        import torch
        from torchwave.approximants.imrphenom_nsbh import nsbh_frequencies

        def tensor(values):
            return torch.as_tensor(values, dtype=torch.float64)

        frequency = np.asarray(nsbh_frequencies(tensor(m1), tensor(m2), tensor(chi), tensor(lam))[0])
    else:
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenom_nsbh import nsbh_frequencies

        frequency = np.asarray(
            nsbh_frequencies(jnp.asarray(m1), jnp.asarray(m2), jnp.asarray(chi), jnp.asarray(lam))[0]
        )
    error = np.max(_relative_error(frequency, [float(item.frequency_hz) for item in expected]))
    assert error <= FREQUENCY_RTOL, (backend, error)
    print(f"{backend}: frequency_rel={error:.3e}")
