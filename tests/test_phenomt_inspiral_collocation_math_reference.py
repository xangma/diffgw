"""Independent scalar check of THM Eqs. (8)--(11) in both T carriers."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from diffgw_reference.phenomt_inspiral_collocation import THETA_NODES, frequency, solve
from diffgw_reference.phenomt_taylor_t3 import coefficients as t3_coefficients


FIT_ROWS = {
    tuple(row["inputs"]): row
    for row in json.loads(
        (Path(__file__).parent / "data/phenomt_author_inspiral_fits_4121.json").read_text()
    )["cases"]
}


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
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        def array(value):
            return jnp.asarray(value, dtype=jnp.float64)

    implementation = importlib.import_module(f"{backend}.approximants.imrphenomt")
    model = implementation.IMRPhenomT()
    eta = m1 * m2 / (m1 + m2) ** 2
    delta = (m1 - m2) / (m1 + m2)
    source_row = FIT_ROWS[(m1, m2, chi1, chi2)]
    t0 = source_row["t0"]
    fitted = source_row["frequencies"]
    pn = np.asarray([float(value) for value in t3_coefficients(eta, delta, chi1, chi2)])
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
