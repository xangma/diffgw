"""Independent metric/tortoise checks; these do not validate TEOB calibration.

Damour--Nagar 0704.3550v2 Eqs. (3), (6)-(11), and Nagar et al.
1806.01772v2 Eqs. (6), (10), (23). No target-library imports or outputs.
"""

import pytest
import torch

import torchwave.approximants.teobresums as teob_module
from torchwave.approximants.eob.hamiltonian import EOBHamiltonian
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.radiation_reaction import EOBRadiationReaction
from torchwave.approximants.teobresums import (
    TEOBResumSRadiationReaction,
    TEOBResumSHamiltonian,
)
from torchwave.constants import MSUN_SEC, PI
from torchwave.waveforms import get_fd_waveform


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def hamiltonian(nu=.23, lambda1=380., lambda2=420.):
    nu, lambda1, lambda2 = map(tensor, (nu, lambda1, lambda2))
    return TEOBResumSHamiltonian(
        nu, torch.zeros_like(nu), torch.zeros_like(nu), lambda1, lambda2)


def test_nonzero_tides_preserve_d_and_tortoise_mass_shell():
    """AB is the printed D; canonical radial momentum gives the mass shell."""
    nu = tensor([.12, .24])
    ham = hamiltonian(nu, tensor([500., 380.]), tensor([250., 420.]))
    r = tensor([[6., 10., 20.], [7., 12., 24.]])
    a, b, xi = ham.metric_potentials(r)
    a0, _, _ = ham.point_mass_metric_potentials(r)
    # The explicit P^0_3[D] in the footnote following Eq. (3).
    d = 1 / (1 + 6*nu[:, None]/r**2
             + 2*nu[:, None]*(26 - 3*nu[:, None])/r**3)
    assert torch.all(a < a0) and torch.all(a > 0)
    torch.testing.assert_close(a*b, d, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(xi**2, a/b, rtol=2e-15, atol=2e-15)

    pr_star, j = tensor([-.2, .3, -.4]), tensor([3.1, 3.6, 4.2])
    pr = pr_star / xi  # p_r dr = p_r* dr*, Eq. (6).
    geodesic_mass_shell = a*(1 + pr**2/b + j**2/r**2)
    z3 = 2*nu[:, None]*(4 - 3*nu[:, None])
    # Eq. (7) uses the stated 3PN truncation for the quartic term.
    quartic = a*z3*pr_star**4/r**2
    actual = ham.effective_hamiltonian(r, pr_star, j).square()
    torch.testing.assert_close(actual, geodesic_mass_shell + quartic,
                               rtol=2e-15, atol=2e-15)


def test_nonzero_tidal_conservative_equations_with_numerical_radial_slope():
    """Check all three published Hamilton equations with nonzero p_r*."""
    ham = hamiltonian()
    r, p, j = tensor([7., 10., 16.]), tensor([-.13, .09, -.03]), tensor([2.9, 3.5, 4.2])
    a, _, xi = ham.metric_potentials(r)
    step = 1e-4
    da = (ham.metric_potentials(r + step)[0]
          - ham.metric_potentials(r - step)[0]) / (2*step)
    z3 = 2*ham.nu*(4 - 3*ham.nu)
    h = torch.sqrt(p**2 + a*(1 + (j**2 + z3*p**4)/r**2))
    energy = torch.sqrt(1 + 2*ham.nu*(h - 1))
    expected = (
        xi*(p + 2*a*z3*p**3/r**2)/(energy*h),
        a*j/(r**2*energy*h),
        -xi/(2*energy*h)*(da*(1 + (j**2 + z3*p**4)/r**2)
                          - 2*a*(j**2 + z3*p**4)/r**3),
    )
    actual = ham.equations_of_motion_conservative(r, p, j)
    for got, wanted in zip(actual, expected):
        torch.testing.assert_close(got, wanted, rtol=5e-8, atol=1e-10)


def test_teob_quasicircular_radiation_has_only_azimuthal_force():
    """TEOBResumS sets F_r*=0 while retaining the flux torque."""
    nu = tensor(.23)
    rr = TEOBResumSRadiationReaction(nu, tensor(0.), tensor(0.))
    omega = tensor([.01, .025])
    pr_star = tensor([-.2, -.05])
    pphi = tensor([3.2, 2.9])

    radial, torque = rr.radiation_reaction_forces(omega, pr_star, pphi)
    torch.testing.assert_close(radial, torch.zeros_like(radial))
    torch.testing.assert_close(nu * omega * torque, -rr.energy_flux(omega),
                               rtol=2e-14, atol=2e-14)


def test_teob_prototype_flux_has_newtonian_weak_field_limit():
    """The scalar prototype tends to the published quadrupole luminosity."""
    nu = tensor(.23)
    rr = TEOBResumSRadiationReaction(nu, tensor(0.), tensor(0.))
    omega = tensor([1e-8, 2e-8])
    newtonian = (32. / 5.) * nu.square() * omega.pow(10. / 3.)
    torch.testing.assert_close(rr.energy_flux(omega) / newtonian,
                               torch.ones_like(omega), rtol=5e-5, atol=1e-8)


def test_tidal_conservative_state_and_deformability_jacobian():
    def field(values):
        r, p, j, lam = values.unbind()
        return torch.stack(hamiltonian(lambda1=lam).equations_of_motion_conservative(r, p, j))

    point = tensor([10., -.05, 3.5, 380.]).requires_grad_(True)
    actual = torch.autograd.functional.jacobian(field, point)
    columns = []
    for index, step in enumerate((1e-4, 1e-5, 1e-4, 1e-2)):
        displacement = torch.zeros_like(point)
        displacement[index] = step
        columns.append((field(point.detach() + displacement)
                        - field(point.detach() - displacement))/(2*step))
    expected = torch.stack(columns, dim=-1)
    assert torch.all(actual[:, -1].abs() > 0)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-10)


def test_tidal_only_requested_gradient_is_preserved():
    lam = tensor(380.).requires_grad_(True)
    values = hamiltonian(lambda1=lam).equations_of_motion_conservative(
        tensor(10.), tensor(-.05), tensor(3.5))
    for value in values:
        derivative = torch.autograd.grad(value, lam, retain_graph=True)[0]
        assert torch.isfinite(derivative) and derivative != 0


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize("nus", [[.12, .24], [.23, .23]])
def test_tidal_scalar_coordinates_with_parameter_batch_during_inference(context, nus):
    r, p, j = tensor(10.), tensor(-.05), tensor(3.5)
    expected = torch.stack([
        torch.stack(hamiltonian(nu, lam, 420.).equations_of_motion_conservative(r, p, j))
        for nu, lam in zip(nus, (500., 380.))
    ], dim=-1)
    with context():
        # Include parameters and coordinates constructed inside inference mode.
        # The second case introduces the batch through lambda alone.
        nu = tensor(nus) if nus[0] != nus[1] else tensor(nus[0])
        ham = hamiltonian(nu, tensor([500., 380.]), tensor(420.))
        actual = ham.equations_of_motion_conservative(tensor(10.), tensor(-.05), tensor(3.5))
        assert all(not value.requires_grad for value in actual)
        torch.testing.assert_close(torch.stack(actual), expected, rtol=2e-14, atol=2e-14)


def test_zero_tides_use_calibrated_teob_point_mass_dynamics():
    """Zero tides retain the calibrated TEOBResumS orbital potential."""
    ham = hamiltonian(nu=.23, lambda1=0., lambda2=0.)
    r = tensor([6., 10., 20.])
    a, b, xi = ham.metric_potentials(r)
    a_pm, b_pm, xi_pm = ham.point_mass_metric_potentials(r)
    torch.testing.assert_close(a, a_pm, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(b, b_pm, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(xi, xi_pm, rtol=2e-15, atol=2e-15)


def test_teob_initial_data_uses_requested_lower_frequency(monkeypatch):
    captured = {}

    def fake_initial_conditions(hamiltonian, radiation_reaction, omega_start):
        captured["omega_start"] = omega_start
        return (torch.full_like(omega_start, 10.0),
                torch.zeros_like(omega_start),
                torch.full_like(omega_start, 3.0))

    monkeypatch.setattr(teob_module, "find_quasicircular_initial_conditions",
                        fake_initial_conditions)
    frequencies = tensor([0.0, 30.0, 40.0])
    get_fd_waveform(
        approximant="TEOBResumS",
        mass1=1.4,
        mass2=1.4,
        sample_frequencies=frequencies,
        f_lower=35.0,
        num_steps=0,
    )
    expected = PI * (2.8 * MSUN_SEC) * 35.0
    torch.testing.assert_close(captured["omega_start"], tensor(expected))


def initial_conditions_for_extra_parameter(parameter, kind):
    ham = TEOBResumSHamiltonian(
        tensor(.23), tensor(.2), tensor(.1),
        parameter if kind == "lambda" else tensor(380.), tensor(420.),
        cq1=parameter if kind == "cq" else None)
    rr = EOBRadiationReaction(ham.nu, ham.chi1, ham.chi2)
    return torch.stack(find_quasicircular_initial_conditions(ham, rr, tensor(.025)), dim=-1)


@pytest.mark.parametrize("kind", ["lambda", "cq"])
@pytest.mark.parametrize("batched", [False, True])
def test_complete_initial_solve_extra_parameter_gradient(kind, batched):
    point = tensor([350., 450.] if kind == "lambda" else [3., 5.])
    if not batched:
        point = point[0]
    point.requires_grad_(True)
    field = lambda value: initial_conditions_for_extra_parameter(value, kind)
    assert field(point).requires_grad
    actual = torch.autograd.functional.jacobian(field, point)
    step = .05 if kind == "lambda" else .001
    columns = []
    for index in range(point.numel()):
        displacement = torch.zeros_like(point)
        displacement.reshape(-1)[index] = step
        columns.append((field(point.detach() + displacement)
                        - field(point.detach() - displacement))/(2*step))
    expected = torch.stack(columns, dim=-1).reshape(actual.shape)
    # Each returned coordinate depends on its own requested intrinsic input.
    assert torch.all(actual.reshape(-1, point.numel()).abs().sum(dim=-1) > 0)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-10)


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_complete_tidal_initial_solve_remains_detached_during_inference(context):
    expected = initial_conditions_for_extra_parameter(tensor([350., 450.]), "lambda")
    with context():
        parameter = tensor([350., 450.]).requires_grad_(True)
        actual = initial_conditions_for_extra_parameter(parameter, "lambda")
        assert not actual.requires_grad
        torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)
