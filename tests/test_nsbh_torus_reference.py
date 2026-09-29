"""Frozen scalar gate for Thompson's NSBH torus fit and Kerr ISCO radius."""

from itertools import product
import math

import mpmath
import numpy as np
import pytest

from diffgw_reference.nsbh_torus import isco_radius, torus


MASS_RATIOS = (1, 2, 5, 10, 15)
SPINS = (-0.5, 0.0, 0.5)
DEFORMABILITIES = (0, 1, 10, 300, 800, 2600, 5000)
CASES = tuple(product(MASS_RATIOS, SPINS, DEFORMABILITIES))
ISCO_ATOL = 2e-13
TORUS_ATOL = 2e-12
SCALAR_STABILITY = mpmath.mpf("1e-30")


@pytest.fixture(scope="module")
def reference():
    return [torus(*case) for case in CASES]


def test_scalar_precision_and_limit():
    context = mpmath.mp.clone()
    context.dps = 100
    for case in CASES:
        a, b = torus(*case, dps=70), torus(*case, dps=90)
        for name in ("isco_radius", "raw_ratio", "carrier_ratio"):
            assert abs(context.mpf(getattr(a, name)) - context.mpf(getattr(b, name))) <= SCALAR_STABILITY
        assert b.carrier_ratio == max(0, b.raw_ratio)
    assert isco_radius(0) == 6


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_native_torus_and_isco_against_paper_reference(backend, reference):
    q, spin, deformability = (np.asarray(values, dtype=np.float64) for values in zip(*CASES))
    if backend == "torch":
        import torch
        from torchwave.approximants.imrphenom_nsbh import (
            ns_compactness_from_lambda,
            nsbh_torus_mass_ratio,
            solve_xi_tide,
        )
        from torchwave.conversions import kerr_isco_radius

        def tensor(values):
            return torch.as_tensor(values, dtype=torch.float64)

        compactness = ns_compactness_from_lambda(tensor(deformability))
        root, _ = solve_xi_tide(tensor(q), tensor(spin), compactness)
        radius = kerr_isco_radius(tensor(spin))
        carrier = nsbh_torus_mass_ratio(tensor(q), tensor(spin), compactness, root)
        c, xi, r, clipped = (np.asarray(value.detach()) for value in (compactness, root, radius, carrier))
    else:
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenom_nsbh import (
            ns_compactness_from_lambda,
            nsbh_torus_mass_ratio,
            solve_xi_tide,
        )
        from jaxwave.conversions import kerr_isco_radius

        compactness = ns_compactness_from_lambda(jnp.asarray(deformability))
        root, _ = solve_xi_tide(jnp.asarray(q), jnp.asarray(spin), compactness)
        radius = kerr_isco_radius(jnp.asarray(spin))
        carrier = nsbh_torus_mass_ratio(jnp.asarray(q), jnp.asarray(spin), compactness, root)
        c, xi, r, clipped = (np.asarray(value) for value in (compactness, root, radius, carrier))

    raw = 0.296 * xi * (1 - 2 * c) - 0.171 * q * c * r
    expected_radius = np.asarray([float(item.isco_radius) for item in reference])
    expected_raw = np.asarray([float(item.raw_ratio) for item in reference])
    expected_clipped = np.asarray([float(item.carrier_ratio) for item in reference])
    radius_error = np.max(np.abs(r - expected_radius))
    raw_error = np.max(np.abs(raw - expected_raw))
    clipped_error = np.max(np.abs(clipped - expected_clipped))
    assert np.any(expected_raw < 0) and np.any(expected_raw > 0)
    assert np.array_equal(clipped > 0, expected_raw > 0)
    assert radius_error <= ISCO_ATOL, (backend, radius_error)
    assert raw_error <= TORUS_ATOL, (backend, raw_error)
    assert clipped_error <= TORUS_ATOL, (backend, clipped_error)
    print(f"{backend}: isco_abs={radius_error:.3e} torus_raw_abs={raw_error:.3e} torus_clipped_abs={clipped_error:.3e}")


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_native_isco_zero_spin_slope(backend):
    expected = -4 * math.sqrt(2 / 3)
    if backend == "torch":
        import torch
        from torchwave.conversions import kerr_isco_radius

        spin = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        measured = torch.autograd.grad(kerr_isco_radius(spin), spin)[0].item()
    else:
        import jax
        import jax.numpy as jnp
        from jaxwave.conversions import kerr_isco_radius

        measured = float(jax.grad(kerr_isco_radius)(jnp.asarray(0.0, dtype=jnp.float64)))
    assert abs(measured - expected) <= ISCO_ATOL
