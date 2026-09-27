"""Source-first TaylorT3 coefficient and primitive gate for both T carriers."""

import importlib

import mpmath as mp
import numpy as np
import pytest

from diffgw_reference import phenomt_taylor_t3 as scalar


CASES = (
    (30.0, 30.0, 0.0, 0.0),
    (30.0, 30.0, 0.6, 0.6),
    (35.0, 25.0, 0.7, -0.5),
    (60.0, 6.0, 0.9, 0.1),
    (80.0, 10.0, -0.8, 0.95),
)


def _native(backend):
    implementation = importlib.import_module(f"{backend}.approximants.imrphenomt")
    if backend == "torchwave":
        torch = pytest.importorskip("torch")

        def array(value):
            return torch.as_tensor(value, dtype=torch.float64)
    else:
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        jnp = pytest.importorskip("jax.numpy")

        def array(value):
            return jnp.asarray(value, dtype=jnp.float64)
    return implementation, array


@pytest.mark.parametrize("backend", ("torchwave", "jaxwave"))
@pytest.mark.parametrize("m1,m2,chi1,chi2", CASES)
def test_taylor_t3_coefficients_frequency_and_phase(backend, m1, m2, chi1, chi2):
    implementation, array = _native(backend)
    eta = m1 * m2 / (m1 + m2) ** 2
    delta = (m1 - m2) / (m1 + m2)
    expected_coefficients = np.asarray([
        float(x) for x in scalar.coefficients(eta, delta, chi1, chi2)
    ])
    actual_coefficients = np.asarray(implementation._pn_coefficients(
        array(eta), array(chi1), array(chi2), array(delta)
    ))
    np.testing.assert_allclose(actual_coefficients, expected_coefficients,
                               rtol=2e-12, atol=2e-13)

    for theta in (0.33, 0.55, 0.82):
        arguments = (array(theta), array(eta), array(chi1), array(chi2), array(delta))
        actual_frequency = np.asarray(implementation._poly_pn_frequency(*arguments)).item()
        expected_frequency = float(scalar.frequency(theta, eta, delta, chi1, chi2))
        np.testing.assert_allclose(actual_frequency, expected_frequency,
                                   rtol=2e-12, atol=2e-13)

        actual_phase = np.asarray(implementation._pn_phase(
            array(theta), array(eta), array(delta), array(chi1), array(chi2)
        )).item()
        expected_phase = float(scalar.phase(theta, eta, delta, chi1, chi2))
        np.testing.assert_allclose(actual_phase, expected_phase,
                                   rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("m1,m2,chi1,chi2", CASES)
def test_source_variant_and_phase_derivative(m1, m2, chi1, chi2):
    with mp.workdps(80):
        eta = mp.mpf(str(m1 * m2 / (m1 + m2) ** 2))
        delta = mp.mpf(str((m1 - m2) / (m1 + m2)))
        carrier = scalar.coefficients(eta, delta, chi1, chi2)
        printed = scalar.coefficients(eta, delta, chi1, chi2, variant="paper_literal")
        assert all(mp.almosteq(a, b, rel_eps=mp.mpf("1e-75"))
                   for a, b in zip(carrier[:4] + carrier[5:],
                                   printed[:4] + printed[5:]))
        difference = mp.mpf(235925) * eta**3 / 1769472
        assert mp.almosteq(carrier[4] - printed[4], difference,
                           rel_eps=mp.mpf("1e-75"))

        theta = mp.mpf("0.55")
        h = mp.mpf("1e-20")
        phase_derivative = (
            scalar.phase(theta + h, eta, delta, chi1, chi2)
            - scalar.phase(theta - h, eta, delta, chi1, chi2)
        ) / (2 * h)
        recovered_frequency = eta * theta**9 / 40 * phase_derivative
        assert mp.almosteq(recovered_frequency,
                           scalar.frequency(theta, eta, delta, chi1, chi2),
                           rel_eps=mp.mpf("1e-35"))


def test_reject_invalid_reference_variant():
    with pytest.raises(ValueError, match="variant"):
        scalar.coefficients(0.25, 0, 0, 0, variant="unspecified")
