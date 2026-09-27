"""Frozen scalar gate for Thompson's NSBH branch-amplitude fit factors."""

from itertools import product

import mpmath
import numpy as np
import pytest

from diffgw_reference.nsbh_branch_amplitude_fits import branch_amplitude_fits
from diffgw_reference.nsbh_disruption import disruption
from diffgw_reference.nsbh_torus import torus


FIELDS = (
    "f_rd_tilde",
    "x_nd",
    "x_prime_nd",
    "x_d",
    "x_prime_d",
    "epsilon_tide",
    "epsilon_ins",
    "sigma_tide",
)
ABS_LIMIT = 2e-12
REL_FREQUENCY_LIMIT = 2e-13
SCALAR_STABILITY = mpmath.mpf("1e-30")


@pytest.fixture(scope="module")
def cases():
    synthetic = [
        (2, 0.2, 0.17, 1000, f_rd, t, lam)
        for f_rd, t, lam in product((800, 1000, 1200), (0, 0.05), (0, 0.5, 1, 2, 800))
    ]
    physical = []
    for q, chi, lam in product((1, 2, 5, 10, 15), (-0.5, 0.0, 0.5), (300, 800, 2600, 5000)):
        result = disruption(q, chi, lam)
        t = torus(q, chi, lam).carrier_ratio
        for ratio in (0.8, 1.0, 1.2):
            physical.append((q, chi, result.compactness, result.frequency_hz, result.frequency_hz * ratio, t, lam))
    assert len(synthetic) == 30 and len(physical) == 180
    return synthetic + physical


def test_reference_precision_and_frozen_coverage(cases):
    ctx = mpmath.mp.clone()
    ctx.dps = 100
    classes = set()
    for case in cases:
        first = branch_amplitude_fits(*case, dps=70)
        second = branch_amplitude_fits(*case, dps=90)
        for name in FIELDS:
            assert abs(ctx.mpf(getattr(first, name)) - ctx.mpf(getattr(second, name))) <= SCALAR_STABILITY
        classes.add((case[3] >= case[4], case[5] > 0))
    assert classes == {(True, False), (True, True), (False, False), (False, True)}
    assert abs(ctx.mpf(branch_amplitude_fits(2, 0.2, 0.17, 1000, 800, 0, 1).f_rd_tilde) - ctx.mpf("776.16")) < ctx.mpf(
        "1e-60"
    )


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_native_branch_amplitude_fits_against_scalar(backend, cases):
    columns = (np.asarray(values, dtype=np.float64) for values in zip(*cases, strict=True))
    q, chi, c, f_tide, f_rd, t, lam = columns
    eta = q / (1.0 + q) ** 2
    if backend == "torch":
        import torch
        from torchwave.approximants.imrphenom_nsbh import nsbh_branch_amplitude_fits, nsbh_branch_predicates

        def tensor(values):
            return torch.as_tensor(values, dtype=torch.float64)

        high, has_torus = nsbh_branch_predicates(tensor(f_tide), tensor(f_rd), tensor(t))
        native = nsbh_branch_amplitude_fits(
            *(tensor(values) for values in (chi, eta, c, f_tide, f_rd, t, lam)), high, has_torus
        )
        actual = {name: np.asarray(getattr(native, name).detach()) for name in FIELDS}
    else:
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenom_nsbh import nsbh_branch_amplitude_fits, nsbh_branch_predicates

        def tensor(values):
            return jnp.asarray(values, dtype=jnp.float64)

        high, has_torus = nsbh_branch_predicates(tensor(f_tide), tensor(f_rd), tensor(t))
        native = nsbh_branch_amplitude_fits(
            *(tensor(values) for values in (chi, eta, c, f_tide, f_rd, t, lam)), high, has_torus
        )
        actual = {name: np.asarray(getattr(native, name)) for name in FIELDS}

    expected = [branch_amplitude_fits(*case) for case in cases]
    errors = {}
    for name in FIELDS:
        truth = np.asarray([float(getattr(value, name)) for value in expected])
        if name == "f_rd_tilde":
            error = np.max(np.abs(actual[name] - truth) / truth)
            assert error <= REL_FREQUENCY_LIMIT, (backend, name, error)
        else:
            error = np.max(np.abs(actual[name] - truth))
            assert error <= ABS_LIMIT, (backend, name, error)
        errors[name] = error
    print(
        f"{backend}: max_relative_frd_tilde={errors['f_rd_tilde']:.3e} max_absolute_fit={max(errors[name] for name in FIELDS[1:]):.3e}"
    )
