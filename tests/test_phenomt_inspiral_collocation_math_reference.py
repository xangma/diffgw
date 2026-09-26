"""Independent scalar check of THM Eqs. (8)--(11) in both T carriers."""

import importlib

import numpy as np
import pytest

from diffgw_core import phenomt_fits as fits
from diffgw_reference.phenomt_inspiral_collocation import THETA_NODES, frequency, solve


@pytest.mark.parametrize("backend", ["torchwave", "jaxwave"])
@pytest.mark.parametrize("m1,m2,chi1,chi2", [
    (30.0, 30.0, 0.0, 0.0),
    (35.0, 25.0, 0.7, -0.5),
    (60.0, 6.0, 0.9, 0.1),
])
def test_thm_frequency_collocation_against_scalar_solve(backend, m1, m2, chi1, chi2):
    if backend == "torchwave":
        import torch

        def array(value):
            return torch.as_tensor(value, dtype=torch.float64)
    else:
        import jax.numpy as jnp

        def array(value):
            return jnp.asarray(value, dtype=jnp.float64)

    implementation = importlib.import_module(f"{backend}.approximants.imrphenomt")
    model = implementation.IMRPhenomT()
    eta = m1 * m2 / (m1 + m2) ** 2
    delta = (m1 - m2) / (m1 + m2)
    spin = (chi1 * (m1 / (m1 + m2)) ** 2 + chi2 * (m2 / (m1 + m2)) ** 2) / (1 - 2 * eta)
    dchi = chi1 - chi2
    t0 = fits._IMRPhenomT_Inspiral_TaylorT3_t0(eta, spin, dchi, delta)
    fitted = [fits._IMRPhenomT_Inspiral_Freq_CP(eta, spin, dchi, delta, i)
              for i in range(1, 6)]
    pn = np.asarray(implementation._pn_coefficients(
        array(eta), array(chi1), array(chi2), array(delta)
    ))
    scalar_coefficients, targets = solve(eta, t0, fitted, pn)
    native = model._coeff(*map(array, (m1, m2, chi1, chi2)))
    native_coefficients = np.asarray(native[7])
    np.testing.assert_allclose(native_coefficients, [float(x) for x in scalar_coefficients],
                               rtol=2e-8, atol=1e-8)

    # Verify the fitted nodes and off-node interpolation in total-mass time.
    for theta, target in zip(THETA_NODES, targets):
        t = -5 / (eta * float(theta) ** 8)
        actual = np.asarray(model._omega_insp(array([t]), array(eta), array(delta),
                                              array(chi1), array(chi2), native[7])).item()
        np.testing.assert_allclose(actual, float(target), rtol=2e-11, atol=1e-12)
    for theta in ("0.39", "0.6", "0.79"):
        t = -5 / (eta * float(theta) ** 8)
        actual = np.asarray(model._omega_insp(array([t]), array(eta), array(delta),
                                              array(chi1), array(chi2), native[7])).item()
        expected = float(frequency(theta, pn, scalar_coefficients))
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-12)
