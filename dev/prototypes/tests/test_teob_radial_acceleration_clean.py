"""Independent checks for the opt in clean TEOB radial branches."""

import pytest
import torch

from torchwave.approximants.eob.teob_radial_acceleration_clean import (
    TEOBRadialAccelerationError,
    approximate_conservative_radial_acceleration,
    frozen_mobility_radial_acceleration,
    radial_acceleration_approximations,
    strict_leading_p_radial_acceleration,
)
from torchwave.approximants.eob.teob_trajectory import (
    conservative_trajectory_quantities,
)
from torchwave.approximants.teobresums import TEOBResumSHamiltonian


DTYPE = torch.float64


def _ham(nu=0.21, *, chi1=0.0, chi2=0.0):
    dtype = DTYPE
    return TEOBResumSHamiltonian(
        torch.as_tensor(nu, dtype=dtype),
        torch.as_tensor(chi1, dtype=dtype),
        torch.as_tensor(chi2, dtype=dtype),
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(0.0, dtype=dtype),
    )


def test_strict_branch_matches_the_zero_momentum_hamiltonian_derivation():
    ham = _ham()
    radius = torch.tensor(8.0, dtype=DTYPE)
    p_phi = torch.tensor(3.4, dtype=DTYPE)
    got = strict_leading_p_radial_acceleration(ham, radius, p_phi)

    a, _, xi = ham.metric_potentials(radius)
    a_prime = ham.metric_potential_derivative(radius)
    k = 1.0 + p_phi.square() / radius.square()
    h_eff0 = torch.sqrt(a * k)
    h_eff0_r = (a_prime * k - 2.0 * a * p_phi.square() / radius.pow(3)) / (2.0 * h_eff0)
    h_real0 = ham.real_hamiltonian_from_eff(h_eff0)
    expected = -xi.square() * h_eff0_r / (h_real0.square() * h_eff0)
    torch.testing.assert_close(got, expected, rtol=2e-12, atol=2e-14)


def test_strict_branch_matches_exact_chain_rule_at_p_zero():
    ham = _ham()
    state = torch.tensor([8.0, 0.3, 0.0, 3.4], dtype=DTYPE)
    exact = conservative_trajectory_quantities(ham, state).radial_acceleration
    terms = radial_acceleration_approximations(
        ham, state[0], state[2], state[3],
    )
    torch.testing.assert_close(terms.strict_p0, exact, rtol=2e-12, atol=2e-14)
    torch.testing.assert_close(terms.p0_radial_velocity, torch.zeros_like(exact))
    torch.testing.assert_close(terms.p0_radial_velocity_radius_partial, torch.zeros_like(exact))
    torch.testing.assert_close(terms.p0_radial_velocity_pr_star_partial,
                               terms.p0_mobility, rtol=2e-12, atol=2e-14)


def test_strict_branch_is_the_finite_difference_leading_p_limit():
    ham = _ham()
    radius = torch.tensor(8.0, dtype=DTYPE)
    p_phi = torch.tensor(3.4, dtype=DTYPE)
    terms = radial_acceleration_approximations(
        ham, radius, torch.tensor(0.0, dtype=DTYPE), p_phi,
    )
    epsilon = 1.0e-5
    displacement = epsilon * terms.p0_pr_star_rate.detach()
    plus = ham.equations_of_motion_conservative(
        radius, displacement, p_phi,
    )[0]
    minus = ham.equations_of_motion_conservative(
        radius, -displacement, p_phi,
    )[0]
    finite_difference = (plus - minus) / (2.0 * epsilon)
    torch.testing.assert_close(terms.strict_p0, finite_difference,
                               rtol=2e-9, atol=2e-12)


def test_frozen_mobility_uses_current_state_and_collapses_to_strict_at_p_zero():
    ham = _ham()
    radius = torch.tensor([8.0, 7.5], dtype=DTYPE)
    p_phi = torch.tensor([3.4, 3.35], dtype=DTYPE)
    zero = torch.zeros_like(radius)
    zero_terms = radial_acceleration_approximations(ham, radius, zero, p_phi)
    torch.testing.assert_close(zero_terms.frozen_mobility, zero_terms.strict_p0,
                               rtol=2e-12, atol=2e-14)

    p = torch.tensor([-0.30, -0.25], dtype=DTYPE)
    terms = radial_acceleration_approximations(ham, radius, p, p_phi)
    direct = terms.current_mobility * terms.current_pr_star_rate
    torch.testing.assert_close(terms.frozen_mobility, direct)
    assert bool(torch.any(torch.abs(terms.frozen_mobility - terms.strict_p0) > 1e-7))
    assert bool(torch.any(torch.abs(terms.mixed_current_p0 - terms.mixed_p0_current) > 1e-8))


def test_exact_chain_rule_is_distinct_from_frozen_mobility_away_from_p_zero():
    ham = _ham()
    state = torch.tensor([8.0, 0.3, -0.30, 3.4], dtype=DTYPE)
    exact = conservative_trajectory_quantities(ham, state).radial_acceleration
    frozen = frozen_mobility_radial_acceleration(ham, state[0], state[2], state[3])
    assert bool(torch.isfinite(exact) & torch.isfinite(frozen))
    assert abs(float((exact - frozen).detach())) > 1e-6


def test_named_dispatch_variants_are_explicit_and_aliases_agree():
    ham = _ham()
    args = (torch.tensor(8.0, dtype=DTYPE), torch.tensor(-0.08, dtype=DTYPE),
            torch.tensor(3.4, dtype=DTYPE))
    terms = radial_acceleration_approximations(ham, *args)
    for name in ("strict_p0", "mixed_current_p0", "mixed_p0_current", "frozen_mobility"):
        got = approximate_conservative_radial_acceleration(ham, *args, variant=name)
        torch.testing.assert_close(got, getattr(terms, name))
    torch.testing.assert_close(terms.strict, terms.strict_leading_p)
    torch.testing.assert_close(terms.frozen, terms.frozen_mobility)
    with pytest.raises(ValueError, match="variant"):
        approximate_conservative_radial_acceleration(ham, *args, variant="guess")


def test_profile_axes_vectorize_without_intrinsic_batch_broadcast():
    ham = _ham()
    radius = torch.tensor([[8.0], [7.5]], dtype=DTYPE)
    p = torch.tensor([[-0.1, -0.08, -0.06]], dtype=DTYPE)
    p_phi = torch.tensor(3.4, dtype=DTYPE)
    terms = radial_acceleration_approximations(ham, radius, p, p_phi)
    assert terms.strict_p0.shape == (2, 3)
    assert torch.isfinite(terms.frozen_mobility).all()

    batched = _ham(torch.tensor([0.18, 0.22], dtype=DTYPE))
    with pytest.raises(ValueError, match="scalar intrinsic"):
        radial_acceleration_approximations(batched, torch.tensor(8.0, dtype=DTYPE),
                                           torch.tensor(-0.1, dtype=DTYPE), p_phi)


def test_acceleration_outputs_are_differentiable_in_state_and_intrinsic_ratio():
    nu = torch.tensor(0.21, dtype=DTYPE, requires_grad=True)
    ham = _ham(nu)
    radius = torch.tensor(8.0, dtype=DTYPE, requires_grad=True)
    p = torch.tensor(-0.08, dtype=DTYPE, requires_grad=True)
    p_phi = torch.tensor(3.4, dtype=DTYPE, requires_grad=True)
    terms = radial_acceleration_approximations(ham, radius, p, p_phi)
    loss = terms.strict_p0.square() + terms.frozen_mobility.square()
    grad_r, grad_p, grad_j, grad_nu = torch.autograd.grad(loss, (radius, p, p_phi, nu))
    assert torch.isfinite(torch.stack((grad_r, grad_p, grad_j, grad_nu))).all()


def test_invalid_domain_and_spin_fail_explicitly():
    ham = _ham()
    with pytest.raises(TEOBRadialAccelerationError, match="radius"):
        radial_acceleration_approximations(ham, torch.tensor(0.0, dtype=DTYPE),
                                           torch.tensor(0.0, dtype=DTYPE),
                                           torch.tensor(3.4, dtype=DTYPE))
    with pytest.raises(TEOBRadialAccelerationError, match="finite"):
        radial_acceleration_approximations(ham, torch.tensor(float("nan"), dtype=DTYPE),
                                           torch.tensor(0.0, dtype=DTYPE),
                                           torch.tensor(3.4, dtype=DTYPE))
    with pytest.raises(NotImplementedError, match="zero-spin"):
        radial_acceleration_approximations(_ham(chi1=0.1), torch.tensor(8.0, dtype=DTYPE),
                                           torch.tensor(0.0, dtype=DTYPE),
                                           torch.tensor(3.4, dtype=DTYPE))
