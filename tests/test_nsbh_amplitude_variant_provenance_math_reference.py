"""Frozen published-versus-compatible NSBH amplitude diagnostic."""

import mpmath
import numpy as np
import pytest

from diffgw_reference.nsbh_amplitude_variants import (
    delta2_large_lambda,
    gamma1_small_lambda,
    low_no_torus_pn_center,
)


REL_LIMIT = 2e-13
SCALAR_STABILITY = mpmath.mpf("1e-30")
X3 = mpmath.mpf("0.0188092")
D3 = mpmath.mpf("0.338737")


def variants(*, dps=70):
    return {
        "pn_center": low_no_torus_pn_center(3, "0.09", "0.06", "0.8", dps=dps),
        "gamma1": gamma1_small_lambda(1, dps=dps),
        "delta2": delta2_large_lambda(1 + X3 + D3 / 4, 1, dps=dps),
    }


def test_scalar_variants_are_stable_and_discriminating():
    ctx = mpmath.mp.clone()
    ctx.dps = 100
    low, high = variants(dps=70), variants(dps=90)
    for name in low:
        for side in ("paper", "compatibility"):
            assert abs(ctx.mpf(getattr(low[name], side)) - ctx.mpf(getattr(high[name], side))) <= SCALAR_STABILITY

    assert abs(low["pn_center"].compatibility / low["pn_center"].paper - 1) > ctx.mpf("1e-3")
    for name in ("gamma1", "delta2"):
        assert abs(low[name].compatibility - low[name].paper) > ctx.mpf("1e-3")


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_native_compatibility_variant_without_paper_claim(backend):
    if backend == "torch":
        import torch
        from torchwave.approximants.imrphenom_nsbh import nsbh_joining_frequencies
        from torchwave.approximants.nsbh_clean_amplitude import delta2_prime_plus, gamma1_prime_small_lambda

        def tensor(value):
            return torch.as_tensor(value, dtype=torch.bool if isinstance(value, bool) else torch.float64)

        def scalar(value):
            return float(value.detach())

    else:
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenom_nsbh import (
            delta2_prime_plus,
            gamma1_prime_small_lambda,
            nsbh_joining_frequencies,
        )

        def tensor(value):
            return jnp.asarray(value, dtype=jnp.bool_ if isinstance(value, bool) else jnp.float64)

        def scalar(value):
            return float(np.asarray(value))

    joins = nsbh_joining_frequencies(
        tensor(3), tensor(0.06), tensor(0.09), tensor(0.8), tensor(False), tensor(False)
    )
    actual = {
        "pn_center": scalar(joins.f0),
        "gamma1": scalar(gamma1_prime_small_lambda(tensor(1))),
        "delta2": scalar(delta2_prime_plus(tensor(float(1 + X3 + D3 / 4)), tensor(1))),
    }
    reference = variants()
    errors = {
        name: abs(value - float(reference[name].compatibility)) / abs(float(reference[name].compatibility))
        for name, value in actual.items()
    }
    assert max(errors.values()) <= REL_LIMIT, (backend, errors)
    print(
        f"{backend}: max_relative_compatibility={max(errors.values()):.3e}; "
        + ", ".join(
            f"{name}_paper={float(pair.paper):.8g},native={actual[name]:.8g}"
            for name, pair in reference.items()
        )
    )
