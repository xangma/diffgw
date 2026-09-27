"""Frozen Thompson Table I merger-branch gate; no waveform-binary oracle."""

from itertools import product

import numpy as np
import pytest

from diffgw_reference.nsbh_branch import classify_branch
from diffgw_reference.nsbh_disruption import disruption
from diffgw_reference.nsbh_torus import torus


def _native_branch(backend, tide, ringdown, torus_ratio):
    if backend == "torch":
        import torch
        from torchwave.approximants.imrphenom_nsbh import nsbh_branch_predicates

        args = (torch.as_tensor(value, dtype=torch.float64) for value in (tide, ringdown, torus_ratio))
        results = nsbh_branch_predicates(*args)
        return tuple(np.asarray(value.detach()) for value in results)
    import jax.numpy as jnp
    from jaxwave.approximants.imrphenom_nsbh import nsbh_branch_predicates

    results = nsbh_branch_predicates(
        *(jnp.asarray(value, dtype=jnp.float64) for value in (tide, ringdown, torus_ratio))
    )
    return tuple(np.asarray(value) for value in results)


def test_reference_exact_boundaries_and_all_four_classes():
    tide = 1000.0
    ringdowns = (np.nextafter(tide, -np.inf), tide, np.nextafter(tide, np.inf))
    cases = tuple(product(ringdowns, (-0.25, 0.0, 0.25)))
    results = [classify_branch(tide, ringdown, ratio) for ringdown, ratio in cases]
    assert {result.name for result in results} == {
        "non_disruptive",
        "mildly_disruptive_torus",
        "mildly_disruptive_no_torus",
        "disruptive",
    }
    assert [classify_branch(tide, tide, ratio).name for ratio in (-0.25, 0, 0.25)] == [
        "non_disruptive",
        "non_disruptive",
        "mildly_disruptive_torus",
    ]
    assert all(
        classify_branch(tide, ringdown, ratio, dps=90) == expected
        for (ringdown, ratio), expected in zip(cases, results, strict=True)
    )


@pytest.mark.parametrize("bad", (0, -1, np.nan, np.inf))
def test_reference_rejects_invalid_frequencies(bad):
    with pytest.raises(ValueError):
        classify_branch(bad, 1000, 0)
    with pytest.raises(ValueError):
        classify_branch(1000, bad, 0)


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_native_exact_boundaries_against_scalar_reference(backend):
    tide = np.full(9, 1000.0)
    ringdown, ratio = (
        np.asarray(values, dtype=np.float64)
        for values in zip(
            *product((np.nextafter(1000.0, -np.inf), 1000.0, np.nextafter(1000.0, np.inf)), (-0.25, 0.0, 0.25)),
            strict=True,
        )
    )
    expected = [classify_branch(t, rd, r) for t, rd, r in zip(tide, ringdown, ratio, strict=True)]
    high, has_torus = _native_branch(backend, tide, ringdown, ratio)
    assert np.array_equal(high, [item.tide_at_or_above_ringdown for item in expected])
    assert np.array_equal(has_torus, [item.has_torus for item in expected])


@pytest.fixture(scope="module")
def physical_reference():
    cases = tuple(product((1, 2, 5, 10, 15), (-0.5, 0.0, 0.5), (300, 800, 2600, 5000)))
    return cases, [disruption(*case).frequency_hz for case in cases], [torus(*case) for case in cases]


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_native_torus_to_branch_composition(backend, physical_reference):
    cases, frequencies, tori = physical_reference
    q, spin, lam = (np.asarray(values, dtype=np.float64) for values in zip(*cases, strict=True))
    raw = np.asarray([float(value.raw_ratio) for value in tori])
    assert np.min(np.abs(raw)) > 2e-12  # Existing torus gate bounds its numerical error.
    if backend == "torch":
        import torch
        from torchwave.approximants.imrphenom_nsbh import (
            ns_compactness_from_lambda,
            nsbh_torus_mass_ratio,
            solve_xi_tide,
        )

        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float64)

        c = ns_compactness_from_lambda(tensor(lam))
        root, _ = solve_xi_tide(tensor(q), tensor(spin), c)
        native_torus = np.asarray(nsbh_torus_mass_ratio(tensor(q), tensor(spin), c, root).detach())
    else:
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenom_nsbh import ns_compactness_from_lambda, nsbh_torus_mass_ratio, solve_xi_tide

        c = ns_compactness_from_lambda(jnp.asarray(lam))
        root, _ = solve_xi_tide(jnp.asarray(q), jnp.asarray(spin), c)
        native_torus = np.asarray(nsbh_torus_mass_ratio(jnp.asarray(q), jnp.asarray(spin), c, root))
    tide = np.repeat(np.asarray([float(value) for value in frequencies]), 3)
    ringdown = tide * np.tile((0.8, 1.0, 1.2), len(cases))
    ratio = np.repeat(native_torus, 3)
    high, has_torus = _native_branch(backend, tide, ringdown, ratio)
    expected = [classify_branch(ft, fr, tr) for ft, fr, tr in zip(tide, ringdown, np.repeat(raw, 3), strict=True)]
    assert np.array_equal(high, [item.tide_at_or_above_ringdown for item in expected])
    assert np.array_equal(has_torus, [item.has_torus for item in expected])
    assert set(zip(high.tolist(), has_torus.tolist(), strict=True)) == {
        (True, False),
        (True, True),
        (False, False),
        (False, True),
    }
