"""Frozen Thompson Eq. (29)/(28) low-tidal branch gate."""

from dataclasses import astuple
from itertools import product

import mpmath
import numpy as np
import pytest

from diffgw_reference.nsbh_low_lambda_width import low_lambda_ringdown_width


ROWS = tuple(product(
    (1e-4, 0.5, 0.81248, 2.0),
    (0.0, 2**-20, 0.125, 0.5, 0.875, 1.0 - 2**-20, 1.0),
    (100.0, 600.0),
    (2.0, 8.0),
))
FIELDS = (
    "delta2_prime", "sigma", "d_delta2_prime_d_lambda",
    "d_delta2_prime_d_delta2", "d_sigma_d_lambda",
    "d_sigma_d_delta2", "d_sigma_d_f_rd", "d_sigma_d_q",
)
SCALAR_LIMIT = mpmath.mpf("1e-30")
VALUE_LIMIT = 2e-13
PARTIAL_LIMIT = 2e-12


def _native(backend, inputs):
    if backend == "torch":
        import torch

        from torchwave.approximants.nsbh_clean_amplitude import delta2_prime_small_lambda

        args = [torch.tensor(inputs[:, i], dtype=torch.float64, requires_grad=True) for i in range(4)]
        delta, lam, frequency, quality = args
        prime = delta2_prime_small_lambda(delta, lam)
        sigma = prime * frequency / quality
        prime_grad = torch.autograd.grad(prime.sum(), (delta, lam), retain_graph=True)
        sigma_grad = torch.autograd.grad(sigma.sum(), args)
        result = (
            prime, sigma, prime_grad[1], prime_grad[0],
            sigma_grad[1], sigma_grad[0], sigma_grad[2], sigma_grad[3],
        )
        return tuple(value.detach().numpy() for value in result)

    import jax
    import jax.numpy as jnp

    from jaxwave.approximants.imrphenom_nsbh import delta2_prime_small_lambda

    assert jax.default_backend() == "cpu"
    assert jax.config.jax_enable_x64
    args = tuple(jnp.asarray(inputs[:, i], dtype=jnp.float64) for i in range(4))

    def prime(delta, lam, _frequency, _quality):
        return delta2_prime_small_lambda(delta, lam)

    def width(delta, lam, frequency, quality):
        return prime(delta, lam, frequency, quality) * frequency / quality

    prime_values = jax.vmap(prime)(*args)
    sigma_values = jax.vmap(width)(*args)
    prime_grad = jax.vmap(jax.grad(prime, argnums=(0, 1)))(*args)
    sigma_grad = jax.vmap(jax.grad(width, argnums=(0, 1, 2, 3)))(*args)
    return tuple(np.asarray(value) for value in (
        prime_values, sigma_values, prime_grad[1], prime_grad[0],
        sigma_grad[1], sigma_grad[0], sigma_grad[2], sigma_grad[3],
    ))


def test_scalar_stability_and_endpoints():
    ctx = mpmath.mp.clone()
    ctx.dps = 100
    for row in ROWS:
        low = low_lambda_ringdown_width(*row, dps=70)
        high = low_lambda_ringdown_width(*row, dps=90)
        for low_field, high_field in zip(astuple(low), astuple(high), strict=True):
            assert abs(ctx.mpf(low_field) - ctx.mpf(high_field)) <= SCALAR_LIMIT
        delta, lam, frequency, quality = row
        if lam == 0:
            assert abs(low.delta2_prime - ctx.mpf(str(delta))) <= SCALAR_LIMIT
        if lam == 1:
            assert abs(low.delta2_prime - ctx.mpf("0.81248")) <= SCALAR_LIMIT
            assert low.d_delta2_prime_d_lambda == 0
            assert low.d_delta2_prime_d_delta2 == 0
            assert abs(low.sigma - ctx.mpf("0.81248") * frequency / quality) <= SCALAR_LIMIT


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_native_values_and_partials(backend):
    inputs = np.asarray(ROWS, dtype=np.float64)
    expected = tuple(
        np.asarray([float(getattr(low_lambda_ringdown_width(*row), field)) for row in ROWS])
        for field in FIELDS
    )
    actual = _native(backend, inputs)
    assert all(value.shape == (len(ROWS),) for value in actual)
    errors = tuple(float(np.max(np.abs(value - target))) for value, target in zip(actual, expected, strict=True))
    assert max(errors[:2]) <= VALUE_LIMIT, (backend, dict(zip(FIELDS, errors, strict=True)))
    assert max(errors[2:]) <= PARTIAL_LIMIT, (backend, dict(zip(FIELDS, errors, strict=True)))
    print(f"{backend}: max_absolute_value={max(errors[:2]):.3e} max_absolute_partial={max(errors[2:]):.3e}")


@pytest.mark.parametrize("bad_row", (
    (0.0, 0.5, 100.0, 2.0),
    (0.5, -0.01, 100.0, 2.0),
    (0.5, 1.01, 100.0, 2.0),
    (0.5, 0.5, 0.0, 2.0),
    (0.5, 0.5, 100.0, 0.0),
))
def test_reference_rejects_out_of_domain_inputs(bad_row):
    with pytest.raises(ValueError):
        low_lambda_ringdown_width(*bad_row)
