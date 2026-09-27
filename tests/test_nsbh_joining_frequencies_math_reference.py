"""Frozen Table I joining-frequency gate and carrier-adapter diagnostic."""

from itertools import product

import mpmath
import numpy as np
import pytest

from diffgw_reference.nsbh_branch_amplitude_fits import branch_amplitude_fits
from diffgw_reference.nsbh_disruption import disruption
from diffgw_reference.nsbh_joining_frequencies import joining_frequencies
from diffgw_reference.nsbh_torus import torus


REL_LIMIT = 2e-13
SCALAR_STABILITY = mpmath.mpf("1e-30")


@pytest.fixture(scope="module")
def cases():
    synthetic = [
        (2, 0.2, 0.17, 1000, f_rd, t, lam)
        for f_rd, t, lam in product((800, 1000, 1200), (0, 0.05), (0, 0.5, 1, 2, 800))
    ]
    physical = []
    for q, chi, lam in product((1, 2, 5, 10, 15), (-0.5, 0.0, 0.5), (300, 800, 2600, 5000)):
        root = disruption(q, chi, lam)
        t = torus(q, chi, lam).carrier_ratio
        for ratio in (0.8, 1.0, 1.2):
            physical.append((q, chi, root.compactness, root.frequency_hz, root.frequency_hz * ratio, t, lam))
    assert len(synthetic) == 30 and len(physical) == 180
    return synthetic + physical


def paper_join(case, *, dps=70):
    q, chi, c, ft, fr, t, lam = case
    fits = branch_amplitude_fits(q, chi, c, ft, fr, t, lam, dps=dps)
    return joining_frequencies(q, ft, fr, t, fits.f_rd_tilde, fits.epsilon_ins, dps=dps)


def test_scalar_precision_and_coverage(cases):
    ctx = mpmath.mp.clone()
    ctx.dps = 100
    classes = set()
    for case in cases:
        low = paper_join(case, dps=70)
        high = paper_join(case, dps=90)
        for name in ("f0", "f1", "f2"):
            a, b = getattr(low, name), getattr(high, name)
            if a is None:
                assert b is None
            else:
                assert abs(ctx.mpf(a) - ctx.mpf(b)) <= SCALAR_STABILITY
        classes.add((case[3] >= case[4], case[5] > 0))
    assert classes == {(True, False), (True, True), (False, False), (False, True)}


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_native_joining_frequencies_against_paper(backend, cases):
    q, chi, c, ft, fr, t, lam = (np.asarray(values, dtype=np.float64) for values in zip(*cases, strict=True))
    eta = q / (1.0 + q) ** 2
    if backend == "torch":
        import torch
        from torchwave.approximants.imrphenom_nsbh import (
            nsbh_branch_amplitude_fits,
            nsbh_branch_predicates,
            nsbh_joining_frequencies,
        )

        def tensor(values):
            return torch.as_tensor(values, dtype=torch.float64)

        high, has_torus = nsbh_branch_predicates(tensor(ft), tensor(fr), tensor(t))
        fits = nsbh_branch_amplitude_fits(
            *(tensor(values) for values in (chi, eta, c, ft, fr, t, lam)), high, has_torus
        )
        native = nsbh_joining_frequencies(tensor(q), tensor(ft), fits.f_rd_tilde, fits.epsilon_ins, high, has_torus)
        actual = {name: np.asarray(getattr(native, name).detach()) for name in ("f0", "f1", "f2")}
    else:
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenom_nsbh import (
            nsbh_branch_amplitude_fits,
            nsbh_branch_predicates,
            nsbh_joining_frequencies,
        )

        def tensor(values):
            return jnp.asarray(values, dtype=jnp.float64)

        high, has_torus = nsbh_branch_predicates(tensor(ft), tensor(fr), tensor(t))
        fits = nsbh_branch_amplitude_fits(
            *(tensor(values) for values in (chi, eta, c, ft, fr, t, lam)), high, has_torus
        )
        native = nsbh_joining_frequencies(tensor(q), tensor(ft), fits.f_rd_tilde, fits.epsilon_ins, high, has_torus)
        actual = {name: np.asarray(getattr(native, name)) for name in ("f0", "f1", "f2")}

    expected = [paper_join(case) for case in cases]
    high_mask = ft >= fr
    low_no_torus = ~high_mask & (t == 0)
    masks = {"f0": ~low_no_torus, "f1": np.ones(len(cases), dtype=bool), "f2": high_mask}
    errors = {}
    for name, mask in masks.items():
        truth = np.asarray(
            [float(getattr(value, name)) for value, selected in zip(expected, mask, strict=True) if selected]
        )
        error = np.max(np.abs(actual[name][mask] - truth) / np.abs(truth))
        assert error <= REL_LIMIT, (backend, name, error)
        errors[name] = error

    # The paper's low/no-torus f0 is deliberately *not* a passing row.
    truth = np.asarray([float(value.f0) for value, selected in zip(expected, low_no_torus, strict=True) if selected])
    mismatch = np.abs(actual["f0"][low_no_torus] - truth) / np.abs(truth)
    assert mismatch.size > 0 and np.max(mismatch) > 1e-3
    print(
        f"{backend}: max_relative_qualified={max(errors.values()):.3e} "
        f"low_no_torus_f0_max_relative_mismatch={np.max(mismatch):.3e}"
    )
