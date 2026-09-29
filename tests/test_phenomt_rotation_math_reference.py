"""Paper-level rotation reference shared by the Torch and JAX backends."""

import importlib
import math

import numpy as np
import pytest

from diffgw_reference import phenomt_rotation as reference


def _backend(name):
    precession = importlib.import_module(f"{name}wave.approximants._phenomt_precession")
    if name == "torch":
        import torch

        return (
            precession,
            lambda value: torch.as_tensor(value, dtype=torch.float64),
            lambda value: torch.as_tensor(value, dtype=torch.complex128),
            lambda value: np.asarray(value.detach().cpu()),
        )
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    return (
        precession,
        lambda value: jnp.asarray(value, dtype=jnp.float64),
        lambda value: jnp.asarray(value, dtype=jnp.complex128),
        np.asarray,
    )


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("ell", [2, 3, 4, 5])
def test_generator_exponential_wigner_matrix(backend, ell):
    """The native finite sum equals an independent generator exponential."""
    precession, real, _, array = _backend(backend)
    beta = np.array([0.0, 0.43, 1.27, 2.4, math.pi], dtype=np.float64)
    reference_matrices = [reference.small_d_matrix(ell, float(value)) for value in beta]
    for m in range(-ell, ell + 1):
        for mp in range(-ell, ell + 1):
            actual = array(precession.wigner_d(ell, m, mp, real(beta)))
            expected = np.array([float(matrix[m + ell, mp + ell].real)
                                 for matrix in reference_matrices])
            np.testing.assert_allclose(actual, expected, rtol=0, atol=8e-13)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_signed_modes_and_polarization_projection(backend):
    """Check the complete Eq. (4)/(9) rotation and observer-basis phase."""
    precession, real, complex_array, array = _backend(backend)
    positive = {
        (2, 2): 1.2 + 0.3j,
        (3, 2): -0.17 + 0.31j,
        (4, 4): 0.21 - 0.08j,
        (5, 5): -0.014 + 0.027j,
    }
    angles = [
        (0.18, 0.43, -0.35, 0.9, 0.14),
        (-1.1, 1.27, 0.4, 1.7, -0.2),
    ]
    expected = np.array([reference.project_modes(positive, *row) for row in angles])
    columns = [real([row[i] for row in angles]) for i in range(5)]
    native_modes = {key: complex_array(np.full(len(angles), value))
                    for key, value in positive.items()}
    hp, hc = precession.project_modes(native_modes, *columns)
    np.testing.assert_allclose(array(hp - 1j * hc), expected, rtol=0, atol=2e-13)

    # Explicit negative-m data must take precedence over symmetry completion.
    signed = positive.copy()
    signed[(3, -2)] = 0.047 - 0.019j
    signed_modes = {key: complex_array(np.full(len(angles), value))
                    for key, value in signed.items()}
    hp, hc = precession.project_modes(signed_modes, *columns)
    expected_signed = np.array([reference.project_modes(signed, *row) for row in angles])
    np.testing.assert_allclose(array(hp - 1j * hc), expected_signed, rtol=0, atol=2e-13)


def test_reference_rotation_inverse_at_initial_frame():
    """Eq. (8)'s fixed inverse maps J modes back to the reference L frame."""
    for ell in (2, 3, 4, 5):
        alpha, beta, gamma = 0.28, 1.21, -0.63
        forward = reference.rotation_matrix(ell, alpha, beta, gamma)
        inverse = reference.rotation_matrix(ell, -gamma, -beta, -alpha)
        identity = inverse * forward
        assert max(abs(complex(identity[m, n] - int(m == n)))
                   for m in range(2 * ell + 1)
                   for n in range(2 * ell + 1)) < 1e-48
