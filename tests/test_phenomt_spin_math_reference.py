"""Independent scalar gates for the PhenomT numerical-precession RHS."""

import numpy as np
import pytest

mpmath = pytest.importorskip("mpmath")

from diffgw_reference.phenomt_spin_evolution import (
    SOURCE_PHENOMTP,
    SOURCE_SPIN_EVOLUTION,
    evaluate_spin_derivatives,
)


LHAT = (0.0, 0.0, 1.0)
SPIN1 = (0.21, 0.08, 0.13)
SPIN2 = (-0.12, 0.17, 0.09)
MASS1 = 0.62
MASS2 = 0.38
KAPPA1 = 1.3
KAPPA2 = 0.4


def _scalar(v, include=True):
    return evaluate_spin_derivatives(
        v, LHAT, SPIN1, SPIN2, MASS1, MASS2,
        quadrupole_kappa1=KAPPA1,
        quadrupole_kappa2=KAPPA2,
        include_n3lo_ss=include,
    )


def _backend(name):
    if name == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants._phenomt_spin_evolution import spin_derivatives

        def evaluate(v, include=True):
            values = spin_derivatives(
                torch.as_tensor(v, dtype=torch.float64),
                torch.as_tensor(LHAT, dtype=torch.float64),
                torch.as_tensor(SPIN1, dtype=torch.float64),
                torch.as_tensor(SPIN2, dtype=torch.float64),
                torch.as_tensor(MASS1, dtype=torch.float64),
                torch.as_tensor(MASS2, dtype=torch.float64),
                quadrupole_kappa1=KAPPA1,
                quadrupole_kappa2=KAPPA2,
                include_n3lo_ss=include,
            )
            return tuple(value.detach().numpy() for value in values)

        return evaluate

    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    from jaxwave.approximants._phenomt_spin_evolution import spin_derivatives

    def evaluate(v, include=True):
        values = spin_derivatives(
            jnp.asarray(v, dtype=jnp.float64),
            jnp.asarray(LHAT, dtype=jnp.float64),
            jnp.asarray(SPIN1, dtype=jnp.float64),
            jnp.asarray(SPIN2, dtype=jnp.float64),
            jnp.asarray(MASS1, dtype=jnp.float64),
            jnp.asarray(MASS2, dtype=jnp.float64),
            quadrupole_kappa1=KAPPA1,
            quadrupole_kappa2=KAPPA2,
            include_n3lo_ss=include,
        )
        return tuple(np.asarray(value) for value in values)

    return evaluate


def test_scalar_reference_is_isolated_and_source_identified():
    original_dps = mpmath.mp.dps
    result = _scalar(0.22)
    assert tuple(result.powers) == ("v5", "v6", "v7", "v8", "v9")
    assert SOURCE_PHENOMTP.endswith("2105.05872v2")
    assert SOURCE_SPIN_EVOLUTION.endswith("T1500554/023/dLdS.pdf")
    assert mpmath.mp.dps == original_dps


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("include", [True, False])
def test_scalar_total_rhs_matches_each_native_backend(backend, include):
    reference = _scalar(0.22, include=include)
    expected = tuple(np.asarray(tuple(map(float, value))) for value in (
        reference.dlhat, reference.ds1, reference.ds2,
    ))
    actual = _backend(backend)(0.22, include=include)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=3e-14, atol=3e-17)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_each_pn_power_is_independently_localized(backend):
    """Recover v5...v9 coefficients without importing backend internals."""
    velocities = np.asarray((0.15, 0.25, 0.35, 0.45, 0.55))
    evaluate = _backend(backend)
    values = np.asarray([evaluate(v)[1:] for v in velocities])
    # ds/v^5 is a fourth-degree polynomial in v.  Solving its Vandermonde
    # system isolates the source-defined v5...v9 pieces independently of
    # how the production function groups its expressions.
    matrix = np.vander(velocities, N=5, increasing=True)
    scaled = values / velocities[:, None, None] ** 5
    recovered = np.linalg.solve(matrix, scaled.reshape(5, -1)).reshape(5, 2, 3)

    reference = _scalar(0.22)
    for index, power in enumerate(range(5, 10)):
        expected = np.asarray([
            [float(value / mpmath.mpf("0.22") ** power) for value in spin]
            for spin in reference.powers[f"v{power}"]
        ])
        np.testing.assert_allclose(recovered[index], expected, rtol=3e-10, atol=3e-12)


def test_production_selection_removes_only_published_v8_piece():
    full = _scalar(0.22, include=True)
    selected = _scalar(0.22, include=False)
    zeros = np.zeros(3)
    np.testing.assert_allclose(np.asarray(tuple(map(float, selected.powers["v8"][0]))), zeros, atol=0)
    np.testing.assert_allclose(np.asarray(tuple(map(float, selected.powers["v8"][1]))), zeros, atol=0)
    for full_spin, selected_spin, v8_spin in zip(
        (full.ds1, full.ds2), (selected.ds1, selected.ds2), full.powers["v8"],
    ):
        np.testing.assert_allclose(
            np.asarray(tuple(map(float, full_spin))),
            np.asarray(tuple(map(float, selected_spin))) + np.asarray(tuple(map(float, v8_spin))),
            rtol=2e-15,
            atol=2e-18,
        )
