"""Contracts for the native PhenomT-driven numerical precession trajectory."""

import math

import numpy as np
import pytest
import torch

from torchwave.approximants._phenomt_numerical import (
    numerical_geometry as torch_geometry,
    prepare_numerical_trajectory as torch_prepare,
)
from torchwave.approximants.imrphenomt import IMRPhenomT
from torchwave.approximants.precession_source_frame_clean import source_frame_geometry


PARAMS = (35.0, 30.0, 0.4, 0.2, 0.5, -0.3, 0.1, 0.2)
TIMES = torch.tensor([-0.8, -0.5, -0.1, 0.0, 0.003], dtype=torch.float64)


def _state_invariants(states):
    l, s1, s2, e = states.split(3, dim=-1)
    return (
        torch.linalg.vector_norm(l, dim=-1),
        torch.linalg.vector_norm(s1, dim=-1),
        torch.linalg.vector_norm(s2, dim=-1),
        (l * e).sum(dim=-1),
    )


def _geometry(prepared, times=TIMES):
    return torch_geometry(
        times, None, *PARAMS, 0.7, 0.2, 20.0, prepared=prepared
    )


def test_torch_trajectory_invariants_and_reference_anchor():
    prepared = torch_prepare(*PARAMS, 20.0, n_backward=128, n_forward=64)
    lnorm, s1norm, s2norm, orth = _state_invariants(prepared["states"])
    torch.testing.assert_close(lnorm, torch.ones_like(lnorm), atol=3e-13, rtol=0)
    torch.testing.assert_close(s1norm, s1norm[0].expand_as(s1norm), atol=3e-13, rtol=0)
    torch.testing.assert_close(s2norm, s2norm[0].expand_as(s2norm), atol=3e-13, rtol=0)
    torch.testing.assert_close(orth, torch.zeros_like(orth), atol=3e-13, rtol=0)

    tref = IMRPhenomT().time_at_frequency(
        torch.tensor(20.0, dtype=torch.float64), torch.tensor(PARAMS[0], dtype=torch.float64),
        torch.tensor(PARAMS[1], dtype=torch.float64), torch.tensor(PARAMS[4], dtype=torch.float64),
        torch.tensor(PARAMS[7], dtype=torch.float64),
    )
    got = _geometry(prepared, tref.reshape(1))
    expected = source_frame_geometry(*PARAMS, 0.7, 0.2, 20.0)
    # The source-frame chart is fixed at f_ref: alpha=a0 and gamma=pi-phi.
    torch.testing.assert_close(got[0][0], expected[0], atol=3e-12, rtol=0)
    torch.testing.assert_close(got[2][0], math.pi - expected[3], atol=3e-12, rtol=0)


def test_torch_resolution_convergence():
    low = torch_prepare(*PARAMS, 20.0, n_backward=1024, n_forward=512)
    high = torch_prepare(*PARAMS, 20.0, n_backward=2048, n_forward=1024)
    low_angles = torch.stack(_geometry(low)[:3])
    high_angles = torch.stack(_geometry(high)[:3])
    relative = torch.linalg.vector_norm(low_angles - high_angles) / torch.linalg.vector_norm(high_angles)
    assert relative < 3e-8


def test_torch_generic_gradient_is_finite():
    x = torch.tensor(PARAMS, dtype=torch.float64, requires_grad=True)
    prepared = torch_prepare(*x, 20.0, n_backward=128, n_forward=64)
    angles = torch_geometry(TIMES[:4], None, *x, 0.7, 0.2, 20.0, prepared=prepared)[:3]
    gradient = torch.autograd.grad(sum(a.sum() for a in angles), x)[0]
    assert torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient) > 0


@pytest.mark.parametrize("resolution", [(128, 64)])
def test_jax_trajectory_invariants_and_jit(resolution):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jaxwave.approximants._phenomt_numerical import numerical_geometry, prepare_numerical_trajectory

    nb, nf = resolution
    prepared = prepare_numerical_trajectory(*PARAMS, 20.0, n_backward=nb, n_forward=nf)
    states = prepared["states"]
    l, s1, s2, e = jnp.split(states, 4, axis=-1)
    np.testing.assert_allclose(jnp.linalg.norm(l, axis=-1), 1.0, atol=3e-12, rtol=0)
    np.testing.assert_allclose(jnp.linalg.norm(s1, axis=-1), jnp.linalg.norm(s1[0]), atol=3e-12, rtol=0)
    np.testing.assert_allclose(jnp.linalg.norm(s2, axis=-1), jnp.linalg.norm(s2[0]), atol=3e-12, rtol=0)
    np.testing.assert_allclose(jnp.sum(l * e, axis=-1), 0.0, atol=3e-12, rtol=0)

    times = jnp.asarray(TIMES.numpy())
    eager = numerical_geometry(times, None, *PARAMS, 0.7, 0.2, 20.0, prepared=prepared)[0]
    fn = jax.jit(lambda tt: numerical_geometry(
        tt, None, *PARAMS, 0.7, 0.2, 20.0, prepared=prepared)[0]
    )
    np.testing.assert_allclose(fn(times), eager, atol=2e-13, rtol=0)

    # Compilation must also cover preparation, rather than only dense output.
    def solve(tt, x):
        p = prepare_numerical_trajectory(*x, 20.0, n_backward=8, n_forward=4)
        return numerical_geometry(tt, None, *x, 0.7, 0.2, 20.0, prepared=p)[0]

    compiled_solve = jax.jit(solve)
    # FP64 eager/XLA ordering changes the accumulated RK4 angle by 2.2e-13
    # (JAX 0.11.1) to 4.2e-13 rad (JAX 0.11.2) on this fixture. The 1e-12 rad
    # envelope remains over nine orders below the 8/4-to-16/8 grid change
    # (at least 7e-3 rad); it tests compiled preparation without requiring
    # identical floating-point operation order.
    np.testing.assert_allclose(compiled_solve(times, jnp.asarray(PARAMS)),
                               solve(times, jnp.asarray(PARAMS)), atol=1e-12, rtol=0)


def test_jax_generic_gradient_is_finite():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jaxwave.approximants._phenomt_numerical import numerical_geometry, prepare_numerical_trajectory

    times = jnp.asarray(TIMES[:4].numpy())

    def objective(x):
        prepared = prepare_numerical_trajectory(*x, 20.0, n_backward=32, n_forward=16)
        angles = numerical_geometry(times, None, *x, 0.7, 0.2, 20.0, prepared=prepared)[:3]
        return sum(a.sum() for a in angles)

    gradient = jax.grad(objective)(jnp.asarray(PARAMS))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0

    # The two native backends use the same low-resolution trajectory and
    # should differentiate the same objective to numerical precision.
    x_torch = torch.tensor(PARAMS, dtype=torch.float64, requires_grad=True)
    p_torch = torch_prepare(*x_torch, 20.0, n_backward=32, n_forward=16)
    a_torch = torch_geometry(TIMES[:4], None, *x_torch, 0.7, 0.2, 20.0, prepared=p_torch)[:3]
    grad_torch = torch.autograd.grad(sum(a.sum() for a in a_torch), x_torch)[0]
    np.testing.assert_allclose(np.asarray(gradient), grad_torch.detach().numpy(), rtol=2e-8, atol=2e-9)
