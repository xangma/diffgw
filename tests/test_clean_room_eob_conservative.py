"""Independent equation and derivative checks for the uncalibrated EOB core."""

import pytest
import torch

from torchwave.approximants.eob.hamiltonian import Nonspinning3PNHamiltonian
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian


DTYPE = torch.float64


def _nonspinning(nu):
    return Nonspinning3PNHamiltonian(nu)


class _ConstantTorque:
    """Prescribed normalized torque isolates the conservative initial-data solve."""

    def radiation_reaction_forces(self, omega, pr_star, pphi, **kwargs):
        return torch.zeros_like(omega), torch.full_like(omega, -1e-6)


def test_schwarzschild_nonzero_tortoise_momentum_and_equations():
    """Geodesic mass shell and Hamilton equations, DN Eqs. (7)-(11), nu=0."""
    ham = _nonspinning(torch.tensor(0.0, dtype=DTYPE))
    r = torch.tensor([4.0, 6.0, 15.0], dtype=DTYPE, requires_grad=True)
    pr = torch.tensor([-0.04, 0.2, -0.1], dtype=DTYPE)
    # Shared incoming graph must not turn coordinate partials into totals.
    angular_momentum = r.sqrt() + 1.0
    a = 1.0 - 2.0 / r
    heff = (pr.square() + a * (1.0 + angular_momentum.square() / r.square())).sqrt()
    expected = (
        a * pr / heff,
        a * angular_momentum / (r.square() * heff),
        -a / (2.0 * heff) * (
            2.0 / r.square() * (1.0 + angular_momentum.square() / r.square())
            - 2.0 * a * angular_momentum.square() / r.pow(3)
        ),
    )
    torch.testing.assert_close(ham.effective_hamiltonian(r, pr, angular_momentum), heff)
    for actual, exact in zip(ham.equations_of_motion_conservative(r, pr, angular_momentum), expected):
        torch.testing.assert_close(actual, exact, rtol=2e-14, atol=2e-15)
    actual_a, actual_b, xi = ham.metric_potentials(r)
    torch.testing.assert_close(actual_a, a, rtol=0, atol=2e-16)
    torch.testing.assert_close(actual_b, 1.0 / a, rtol=2e-15, atol=0)
    torch.testing.assert_close(xi, a, rtol=0, atol=2e-16)


def test_pade_potentials_recover_3pn_series_and_effective_horizon():
    """PN truncation errors have their specified orders; A has a simple zero."""
    nu = torch.tensor(0.25, dtype=DTYPE)
    ham = _nonspinning(nu)
    r = torch.tensor([100.0, 200.0], dtype=DTYPE)
    u = 1.0 / r
    a, b, xi = ham.metric_potentials(r)
    a4 = 94.0 / 3.0 - 41.0 * torch.pi**2 / 32.0
    a_series = 1.0 - 2.0 * u + 2.0 * nu * u**3 + a4 * nu * u**4
    d_series = 1.0 - 6.0 * nu * u**2 + 2.0 * (3.0 * nu - 26.0) * nu * u**3
    # Doubling r reduces the first omitted 4PN/3PN+ terms by 2^5 / 2^4.
    torch.testing.assert_close((a - a_series)[0] / (a - a_series)[1], torch.tensor(32.0, dtype=DTYPE), rtol=0.02, atol=0)
    torch.testing.assert_close((a * b - d_series)[0] / (a * b - d_series)[1], torch.tensor(16.0, dtype=DTYPE), rtol=0.10, atol=0)
    torch.testing.assert_close(xi.square(), a / b)
    # A crosses through zero instead of replacing the horizon by a floor.
    horizon_r = 2.0 - nu * (a4 + 4.0) / (8.0 - 2.0 * nu)
    near_horizon = torch.stack((horizon_r - 1e-5, horizon_r + 1e-5))
    horizon_a, _, _ = ham.metric_potentials(near_horizon)
    assert horizon_a[0] < 0 < horizon_a[1]


def test_conservative_vector_field_preserves_parameter_and_state_gradients():
    r = torch.tensor([8.0, 14.0], dtype=DTYPE, requires_grad=True)
    pr = torch.tensor([-0.03, 0.08], dtype=DTYPE, requires_grad=True)
    angular_momentum = torch.tensor([3.5, 4.5], dtype=DTYPE, requires_grad=True)
    nu = torch.tensor([0.12, 0.21], dtype=DTYPE, requires_grad=True)

    def vector_field(radius, radial, angular, eta):
        return torch.stack(_nonspinning(eta).equations_of_motion_conservative(radius, radial, angular))

    assert torch.autograd.gradcheck(vector_field, (r, pr, angular_momentum, nu), atol=2e-7, rtol=2e-5)


def test_3pn_quartic_radial_term():
    nu = torch.tensor([0.1, 0.25], dtype=DTYPE)
    ham = _nonspinning(nu)
    r = torch.tensor([5.0, 9.0], dtype=DTYPE)
    pr = torch.tensor([0.4, -0.3], dtype=DTYPE)
    angular = torch.tensor([3.5, 4.0], dtype=DTYPE)
    a, _, _ = ham.metric_potentials(r)
    mass_shell = ham.effective_hamiltonian(r, pr, angular).square()
    geodesic_part = pr.square() + a * (1.0 + angular.square() / r.square())
    # Damour--Nagar Eq. (7), with z3=2 nu (4-3 nu).
    expected = a * 2.0 * nu * (4.0 - 3.0 * nu) * pr.pow(4) / r.square()
    torch.testing.assert_close(mass_shell - geodesic_part, expected, rtol=2e-12, atol=2e-16)


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize("create_inside", [False, True])
def test_inference_discards_local_partial_derivative_graphs(context, create_inside):
    def inputs():
        return (
            torch.tensor(0.017, dtype=DTYPE, requires_grad=True),
            _nonspinning(torch.tensor(0.17, dtype=DTYPE, requires_grad=True)),
        )

    omega, ham = inputs()
    expected = find_quasicircular_initial_conditions(ham, _ConstantTorque(), omega)
    with context():
        if create_inside:
            omega, ham = inputs()
        state = find_quasicircular_initial_conditions(ham, _ConstantTorque(), omega)
        derivatives = ham.equations_of_motion_conservative(*state)
    assert all(not value.requires_grad for value in state + derivatives)
    assert all(torch.isfinite(value).all() for value in state + derivatives)
    for actual, reference in zip(state, expected):
        torch.testing.assert_close(actual, reference, rtol=2e-13, atol=2e-14)


def test_schwarzschild_circular_initial_data_and_adiabatic_momentum():
    """Use exact circular geodesics and BD Eq. (4.13), transformed to p_r*."""
    omega = torch.tensor([0.009, 0.02, 0.04], dtype=DTYPE, requires_grad=True)
    ham = _nonspinning(torch.tensor(0.0, dtype=DTYPE))
    radius, radial, angular = find_quasicircular_initial_conditions(ham, _ConstantTorque(), omega, tol=1e-13)
    exact_r = omega.pow(-2.0 / 3.0)
    exact_j = exact_r / (exact_r - 3.0).sqrt()
    j_prime = (exact_r - 6.0) / (2.0 * (exact_r - 3.0).pow(1.5))
    heff = (exact_r - 2.0) / (exact_r * (exact_r - 3.0)).sqrt()
    exact_radial = -1e-6 * heff / ((1.0 - 2.0 / exact_r) * j_prime)
    for actual, expected in zip((radius, radial, angular), (exact_r, exact_radial, exact_j)):
        torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-14)
    dr_domega = torch.autograd.grad(radius.sum(), omega)[0]
    torch.testing.assert_close(dr_domega, -2.0 / 3.0 * omega.pow(-5.0 / 3.0), rtol=3e-13, atol=0)


@pytest.mark.parametrize(
    "nu, chi1, chi2, omega",
    [(0.24, 0.2, -0.1, 0.015), (0.16, 0.7, 0.3, 0.02),
     (0.10, -0.4, 0.5, 0.025)],
)
def test_seobnrv4_initial_momentum_satisfies_buonanno_radial_balance(
    nu, chi1, chi2, omega,
):
    """SEOBNRv4 states obey Buonanno et al. (2006) Eqs. (76)-(77).

    The test recomputes the circular-branch slope and the linearized radial
    Hamilton equation from the returned state.  Their product with p_r* must
    equal the supplied torque; this is the source-level initial-data
    invariant and does not depend on a waveform oracle.
    """
    ham = SEOBNRv4Hamiltonian(
        torch.tensor(nu, dtype=DTYPE), torch.tensor(chi1, dtype=DTYPE),
        torch.tensor(chi2, dtype=DTYPE),
    )
    torque = _ConstantTorque()
    radius, radial, angular = find_quasicircular_initial_conditions(
        ham, torque, torch.tensor(omega, dtype=DTYPE), max_iter=35, tol=1e-12,
    )

    r = radius.detach().requires_grad_(True)
    pphi = angular.detach().requires_grad_(True)
    pr0 = torch.zeros_like(r, requires_grad=True)
    h_eff = ham.effective_hamiltonian(r, pr0, pphi)
    h_real = ham.real_hamiltonian_from_eff(h_eff)
    h_r, h_pr, _ = torch.autograd.grad(
        h_eff, (r, pr0, pphi), create_graph=True,
    )
    h_rr, h_rj = torch.autograd.grad(h_r, (r, pphi), create_graph=True)
    h_prpr = torch.autograd.grad(h_pr, pr0)[0]
    _, f_phi = torque.radiation_reaction_forces(
        torch.tensor(omega, dtype=DTYPE), torch.zeros_like(r), pphi,
    )
    _, _, xi = ham.metric_potentials(r)
    dp_phi_dr = -h_rr / h_rj
    radial_balance = (xi * h_prpr / h_real) * radial * dp_phi_dr
    torch.testing.assert_close(radial_balance, f_phi, rtol=2e-11, atol=2e-13)


def test_initial_state_gradients_and_batched_scalar_frequency():
    omega = torch.tensor(0.017, dtype=DTYPE, requires_grad=True)
    nu = torch.tensor(0.17, dtype=DTYPE, requires_grad=True)

    def initial_state(frequency, eta):
        return torch.stack(find_quasicircular_initial_conditions(
            _nonspinning(eta), _ConstantTorque(), frequency, max_iter=12, tol=1e-12,
        ))

    assert torch.autograd.gradcheck(initial_state, (omega, nu), atol=3e-7, rtol=3e-5)
    nus = torch.tensor([0.0, 0.12, 0.23], dtype=DTYPE)
    batched = initial_state(omega, nus)
    scalar = torch.stack([initial_state(omega, eta) for eta in nus], dim=-1)
    torch.testing.assert_close(batched, scalar, rtol=2e-13, atol=2e-14)
