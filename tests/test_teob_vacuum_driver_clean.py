"""Focused checks for the bounded clean TEOB vacuum RR integration."""

import torch
import pytest

from torchwave.approximants.eob.teob_radiation_clean import TEOBRadiationClean
from torchwave.approximants.eob.radiation_reaction import EOBRadiationReaction
from torchwave.approximants.eob.teob_vacuum_driver_clean import (
    TEOBVacuumRadiationDriver,
)
from torchwave.approximants.teobresums import (
    TEOBResumSHamiltonian,
    TEOBResumSRadiationReaction,
)


def _hamiltonian(*, dtype=torch.float64, batch=False):
    if batch:
        nu = torch.tensor([0.20, 0.24], dtype=dtype)
        zeros = torch.zeros(2, dtype=dtype)
    else:
        nu = torch.tensor(0.22, dtype=dtype)
        zeros = torch.tensor(0.0, dtype=dtype)
    return TEOBResumSHamiltonian(nu, zeros, zeros, zeros, zeros)


def _state(hamiltonian, *, r=10.0, pr=-0.003, pphi=3.4):
    dtype = hamiltonian.nu.dtype
    device = hamiltonian.nu.device
    return (
        torch.as_tensor(r, dtype=dtype, device=device),
        torch.as_tensor(pr, dtype=dtype, device=device),
        torch.as_tensor(pphi, dtype=dtype, device=device),
    )


def test_clean_rr_uses_current_pphi_hcirc_and_zero_radial_force():
    ham = _hamiltonian()
    rr = TEOBResumSRadiationReaction(ham.nu, ham.chi1, ham.chi2)
    r, pr_star, pphi = _state(ham)
    _, omega, _ = ham.equations_of_motion_conservative(r, pr_star, pphi)

    h_eff, h_real, h_circ, a_prime = ham.clean_radiation_inputs(
        r, pr_star, pphi,
    )
    expected_circ = ham.real_hamiltonian(r, torch.zeros_like(pr_star), pphi)
    assert torch.allclose(h_circ, expected_circ, rtol=1e-13, atol=0)
    assert torch.allclose(h_real, ham.real_hamiltonian(r, pr_star, pphi),
                          rtol=1e-13, atol=0)
    assert torch.isfinite(a_prime)

    clean = TEOBRadiationClean(8, residual_phase="zero-unpublished")
    expected_flux = clean.flux(
        r, omega, torch.zeros_like(r), pr_star, pphi, h_eff, h_real,
        h_circ, a_prime, ham.m1, ham.m2, include_horizon=True,
    )
    radial, torque = rr.radiation_reaction_forces(
        omega, pr_star, pphi, r=r, hamiltonian=ham,
    )
    assert torch.equal(radial, torch.zeros_like(radial))
    assert torch.allclose(torque, -expected_flux / ham.nu, rtol=1e-13, atol=0)

    # The circular energy retains the current angular momentum; a circular
    # solve at the same radius would generally give a different value.
    circular_pphi = torch.sqrt(r / torch.clamp(1.0 - 3.0 / r, min=0.1))
    wrong_hcirc = ham.real_hamiltonian(r, torch.zeros_like(pr_star), circular_pphi)
    assert not torch.allclose(h_circ, wrong_hcirc, rtol=1e-8, atol=1e-10)


def test_horizon_policy_and_zero_spin_zero_tide_dispatch():
    ham = _hamiltonian()
    rr = TEOBResumSRadiationReaction(
        ham.nu, ham.chi1, ham.chi2, include_horizon_absorption=True,
    )
    r, pr_star, pphi = _state(ham)
    omega = ham.equations_of_motion_conservative(r, pr_star, pphi)[1]
    _, with_horizon = rr.radiation_reaction_forces(
        omega, pr_star, pphi, r=r, hamiltonian=ham,
    )
    _, infinity_only = rr.radiation_reaction_forces(
        omega, pr_star, pphi, r=r, hamiltonian=ham,
        include_horizon_absorption=False,
    )
    assert torch.isfinite(with_horizon) and torch.isfinite(infinity_only)
    assert torch.abs(with_horizon - infinity_only) > 0

    spin_ham = TEOBResumSHamiltonian(
        ham.nu, torch.tensor(0.1, dtype=ham.nu.dtype), ham.chi2,
        ham.lambda1, ham.lambda2,
    )
    tide_ham = TEOBResumSHamiltonian(
        ham.nu, ham.chi1, ham.chi2,
        torch.tensor(10.0, dtype=ham.nu.dtype), ham.lambda2,
    )
    assert not rr._is_nonspinning_vacuum(spin_ham)
    assert not rr._is_nonspinning_vacuum(tide_ham)


def test_nonvacuum_routes_keep_their_existing_force_sources():
    """The clean source is isolated from tidal and spinning fallbacks."""
    base = _hamiltonian()
    r, pr_star, pphi = _state(base)

    tide_ham = TEOBResumSHamiltonian(
        base.nu, base.chi1, base.chi2,
        torch.tensor(10.0, dtype=base.nu.dtype), base.lambda2,
    )
    tide_rr = TEOBResumSRadiationReaction(
        tide_ham.nu, tide_ham.chi1, tide_ham.chi2,
    )
    tide_omega = tide_ham.equations_of_motion_conservative(r, pr_star, pphi)[1]
    tide_radial, tide_torque = tide_rr.radiation_reaction_forces(
        tide_omega, pr_star, pphi, r=r, hamiltonian=tide_ham,
    )
    tide_flux = tide_rr._factorized.energy_flux(
        tide_omega, pr_star, pphi, r=r, hamiltonian=tide_ham,
    )
    assert torch.equal(tide_radial, torch.zeros_like(tide_radial))
    assert torch.allclose(
        tide_torque, -tide_flux / (tide_ham.nu * tide_omega),
        rtol=1e-13, atol=0,
    )

    spin_ham = TEOBResumSHamiltonian(
        base.nu, torch.tensor(0.1, dtype=base.nu.dtype), base.chi2,
        base.lambda1, base.lambda2,
    )
    spin_rr = TEOBResumSRadiationReaction(
        spin_ham.nu, spin_ham.chi1, spin_ham.chi2,
    )
    spin_omega = spin_ham.equations_of_motion_conservative(r, pr_star, pphi)[1]
    spin_radial, spin_torque = spin_rr.radiation_reaction_forces(
        spin_omega, pr_star, pphi, r=r, hamiltonian=spin_ham,
    )
    generic_torque = EOBRadiationReaction(
        spin_ham.nu, spin_ham.chi1, spin_ham.chi2,
    ).radiation_reaction_forces(spin_omega, pr_star, pphi)[1]
    assert torch.equal(spin_radial, torch.zeros_like(spin_radial))
    assert torch.allclose(spin_torque, generic_torque, rtol=1e-13, atol=0)


def test_scalar_batch_gradients_and_inference_are_finite():
    ham = _hamiltonian()
    rr = TEOBResumSRadiationReaction(ham.nu, ham.chi1, ham.chi2)
    r = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    pr_star = torch.tensor(-0.003, dtype=torch.float64)
    pphi = torch.tensor(3.4, dtype=torch.float64, requires_grad=True)
    omega = ham.equations_of_motion_conservative(r, pr_star, pphi)[1]
    torque = rr.radiation_reaction_forces(
        omega, pr_star, pphi, r=r, hamiltonian=ham,
    )[1]
    grad_r, grad_pphi = torch.autograd.grad(torque, (r, pphi))
    assert torch.isfinite(torque) and torch.isfinite(grad_r) and torch.isfinite(grad_pphi)

    batch_ham = _hamiltonian(batch=True)
    batch_rr = TEOBResumSRadiationReaction(
        batch_ham.nu, batch_ham.chi1, batch_ham.chi2,
    )
    batch_r, batch_pr, batch_pphi = _state(
        batch_ham, r=torch.tensor([10.0, 11.0]),
        pr=torch.tensor([-0.002, -0.003]), pphi=torch.tensor([3.2, 3.5]),
    )
    batch_omega = batch_ham.equations_of_motion_conservative(
        batch_r, batch_pr, batch_pphi,
    )[1]
    with torch.no_grad():
        radial, batch_torque = batch_rr.radiation_reaction_forces(
            batch_omega, batch_pr, batch_pphi,
            r=batch_r, hamiltonian=batch_ham,
        )
    assert radial.shape == batch_torque.shape == batch_omega.shape
    assert torch.isfinite(batch_torque).all()

    driver = TEOBVacuumRadiationDriver(ham)
    state = driver.initial_state(torch.tensor(0.025, dtype=torch.float64))
    with torch.inference_mode():
        inferred = driver.vector_field(state)
    assert not inferred.requires_grad
    assert torch.isfinite(inferred).all()


@pytest.mark.parametrize("phase", [float("nan"), float("inf"), float("-inf")])
def test_initial_state_rejects_nonfinite_phase_before_solver(monkeypatch, phase):
    driver = TEOBVacuumRadiationDriver(_hamiltonian())

    def unexpected_solver(*args, **kwargs):
        raise AssertionError("invalid phase reached the circular solver")

    monkeypatch.setattr(
        "torchwave.approximants.eob.teob_vacuum_driver_clean.find_quasicircular_initial_conditions",
        unexpected_solver,
    )
    with pytest.raises(ValueError, match="phase must be finite"):
        driver.initial_state(torch.tensor(0.025, dtype=torch.float64), phase=phase)


def test_initial_state_rejects_complex_inputs_before_solver(monkeypatch):
    driver = TEOBVacuumRadiationDriver(_hamiltonian())

    def unexpected_solver(*args, **kwargs):
        raise AssertionError("invalid input reached the circular solver")

    monkeypatch.setattr(
        "torchwave.approximants.eob.teob_vacuum_driver_clean.find_quasicircular_initial_conditions",
        unexpected_solver,
    )
    with pytest.raises(TypeError, match="omega_start must be real-valued"):
        driver.initial_state(torch.tensor(0.025 + 1.0e-3j, dtype=torch.complex128))
    with pytest.raises(TypeError, match="phase must be real-valued"):
        driver.initial_state(torch.tensor(0.025, dtype=torch.float64), phase=torch.tensor(1.0j))


@pytest.mark.parametrize("ham_dtype, rejects", [
    (torch.float64, False),
    (torch.float32, True),
])
def test_initial_state_phase_cast_uses_hamiltonian_dtype(
    monkeypatch, ham_dtype, rejects,
):
    driver = TEOBVacuumRadiationDriver(_hamiltonian(dtype=ham_dtype))
    seen = {}

    def valid_solver(hamiltonian, radiation_reaction, omega_start):
        seen["omega"] = omega_start
        value = torch.ones_like(omega_start)
        return value, value, value

    monkeypatch.setattr(
        "torchwave.approximants.eob.teob_vacuum_driver_clean.find_quasicircular_initial_conditions",
        valid_solver,
    )
    if rejects:
        with pytest.raises(ValueError, match="phase must be finite"):
            driver.initial_state(torch.tensor(0.025, dtype=ham_dtype), phase=1e40)
        assert "omega" not in seen
    else:
        state = driver.initial_state(torch.tensor(0.025, dtype=ham_dtype), phase=1e40)
        assert state[1].item() == 1e40
        assert state.dtype == ham_dtype
        assert seen["omega"].dtype == ham_dtype


def test_short_trajectory_dt_refinement():
    ham = _hamiltonian(dtype=torch.float64)
    driver = TEOBVacuumRadiationDriver(ham, include_horizon_absorption=True)
    initial = driver.initial_state(torch.tensor(0.025, dtype=torch.float64))
    coarse_time, coarse = driver.integrate(initial, 0.5, 8)
    fine_time, fine = driver.integrate(initial, 0.25, 16)
    assert coarse_time[-1] == fine_time[-1] == torch.tensor(4.0, dtype=torch.float64)
    assert torch.isfinite(fine).all()
    # This short, weak-field interval should be deep in the RK4 convergence
    # regime; the check is a diagnostic guard rather than a parity target.
    assert torch.max(torch.abs(coarse[-1] - fine[-1])) < 1e-8
