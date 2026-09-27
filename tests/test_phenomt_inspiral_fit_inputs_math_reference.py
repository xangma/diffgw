"""Frozen MIT author fit inputs versus the shared PhenomT fit evaluator."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from diffgw_core import phenomt_fits as fits
from diffgw_reference.phenomt_author_fit_inputs import SOURCE_REVISION, SOURCE_SHA256


FIXTURE = json.loads(
    (Path(__file__).parent / "data/phenomt_author_inspiral_fits_4121.json").read_text()
)
assert FIXTURE["source_revision"] == SOURCE_REVISION
assert FIXTURE["source_sha256"] == SOURCE_SHA256


@pytest.mark.parametrize("backend", ("python", "torchwave", "jaxwave"))
@pytest.mark.parametrize("row", FIXTURE["cases"], ids=lambda row: "-".join(map(str, row["inputs"])))
def test_pinned_author_inspiral_fit_inputs(backend, row):
    m1, m2, chi1, chi2 = row["inputs"]
    M = m1 + m2
    eta = m1 * m2 / M**2
    delta = (m1 - m2) / M
    spin = (m1**2 * chi1 + m2**2 * chi2) / (m1**2 + m2**2)
    dchi = chi1 - chi2
    if backend == "python":
        arguments = (eta, spin, dchi, delta)
    else:
        implementation = importlib.import_module(f"{backend}.approximants.imrphenomt")
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

        _, native_eta, native_delta, native_spin, native_dchi, _, _ = (
            implementation.IMRPhenomT._state(*map(array, row["inputs"]))
        )
        np.testing.assert_allclose(
            np.asarray((native_eta, native_delta, native_spin, native_dchi)),
            (eta, delta, spin, dchi), rtol=2e-12, atol=2e-13,
        )
        arguments = (native_eta, native_spin, native_dchi, native_delta)

    actual_t0 = fits._IMRPhenomT_Inspiral_TaylorT3_t0(*arguments)
    actual_frequencies = [
        fits._IMRPhenomT_Inspiral_Freq_CP(*arguments, index)
        for index in range(1, 6)
    ]
    np.testing.assert_allclose(np.asarray(actual_t0), row["t0"], rtol=2e-11, atol=5e-10)
    np.testing.assert_allclose(
        np.asarray(actual_frequencies), row["frequencies"], rtol=2e-11, atol=2e-12,
    )
