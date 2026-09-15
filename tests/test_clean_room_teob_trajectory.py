"""Independent checks for conservative TEOB trajectory diagnostics."""

import pytest
import torch

from torchwave.approximants.eob.teob_trajectory import (
    conservative_trajectory_quantities,
)
from torchwave.approximants.teobresums import TEOBResumSHamiltonian


DTYPE = torch.float64


def _ham(nu):
    zero = torch.zeros_like(nu)
    return TEOBResumSHamiltonian(nu, zero, zero, 0.0 * nu, 0.0 * nu)


def test_canonical_pr_star_and_radial_acceleration_follow_conservative_flow():
    ham = _ham(torch.tensor(0.21, dtype=DTYPE))
    state = torch.tensor([8.0, 0.3, -0.07, 3.4], dtype=DTYPE)
    got = conservative_trajectory_quantities(ham, state)
    assert torch.equal(got.pr_star, state[2])

    r, pr, pphi = state[0], state[2], state[3]
    eps = 1e-5
    dr, _, dpr = ham.equations_of_motion_conservative(r, pr, pphi)
    plus = ham.equations_of_motion_conservative(r + eps * dr, pr + eps * dpr, pphi)[0]
    minus = ham.equations_of_motion_conservative(r - eps * dr, pr - eps * dpr, pphi)[0]
    finite_difference = (plus - minus) / (2.0 * eps)
    torch.testing.assert_close(got.radial_acceleration, finite_difference,
                               rtol=2e-8, atol=2e-10)


def test_trajectory_quantities_broadcast_parameter_batches_and_gradients():
    nu = torch.tensor([0.12, 0.22], dtype=DTYPE, requires_grad=True)
    ham = _ham(nu)
    state = torch.tensor([9.0, 0.0, -0.04, 3.2], dtype=DTYPE, requires_grad=True)
    got = conservative_trajectory_quantities(ham, state)
    assert got.pr_star.shape == nu.shape
    assert got.radial_acceleration.shape == nu.shape
    loss = (got.radial_acceleration.square() + got.pr_star.square()).sum()
    grad_state, grad_nu = torch.autograd.grad(loss, (state, nu))
    assert torch.isfinite(grad_state).all()
    assert torch.isfinite(grad_nu).all()


def test_trajectory_quantities_support_sample_axes_and_dtype():
    nu = torch.tensor([0.16, 0.20], dtype=DTYPE)
    ham = _ham(nu)
    state = torch.tensor(
        [[[10.0, 0.0, -0.02, 3.1], [8.0, 0.2, -0.05, 3.3]],
         [[12.0, 0.1, -0.01, 3.4], [7.0, 0.3, -0.08, 3.5]]],
        dtype=DTYPE,
    )
    got = conservative_trajectory_quantities(ham, state)
    assert got.pr_star.shape == (2, 2)
    assert got.radial_acceleration.shape == (2, 2)
    assert got.radial_acceleration.dtype == DTYPE
    assert torch.isfinite(got.radial_acceleration).all()


def test_trajectory_quantities_preserve_leading_parameter_batch_axis():
    """Parameter batches lead sample axes, matching the Hamiltonian contract."""
    nu = torch.tensor([0.14, 0.20], dtype=DTYPE)
    ham = _ham(nu)
    state = torch.tensor(
        [[[10.0, 0.0, -0.02, 3.1], [9.0, 0.1, -0.03, 3.2], [8.0, 0.2, -0.04, 3.3]],
         [[12.0, 0.1, -0.01, 3.4], [11.0, 0.2, -0.02, 3.5], [7.0, 0.3, -0.08, 3.6]]],
        dtype=DTYPE,
    )
    got = conservative_trajectory_quantities(ham, state)
    assert got.radial_velocity.shape == (2, 3)
    assert got.radial_acceleration.shape == (2, 3)
    assert torch.isfinite(got.radial_acceleration).all()


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_trajectory_quantities_detach_local_autograd_graph_during_inference(context):
    ham = _ham(torch.tensor(0.18, dtype=DTYPE, requires_grad=True))
    state = torch.tensor([8.0, 0.0, -0.04, 3.2], dtype=DTYPE, requires_grad=True)
    with context():
        got = conservative_trajectory_quantities(ham, state)
    assert all(not value.requires_grad for value in got)
    assert all(torch.isfinite(value).all() for value in got)
