"""Production-path checks for the odd-m amplitude sign convention."""

import torch

from torchwave.approximants import imrphenom_xhm as xhm
from torchwave.approximants.imrphenom_xhm import IMRPhenomXHM
from torchwave.approximants.xhm_fits import XHMParameterFits


DTYPE = torch.float64


def _equal_mass_odd_fit_values():
    mass = torch.tensor(35.0, dtype=DTYPE)
    chi1 = torch.tensor(0.6, dtype=DTYPE)
    chi2 = torch.tensor(-0.3, dtype=DTYPE)
    eta = mass * mass / (2.0 * mass).square()
    _, _, _, spin = XHMParameterFits.compute_spin_variables(eta, chi1, chi2)
    chi_eff = (chi1 + chi2) / 2.0
    chi_pn_hat = (
        chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)
    ) / (1.0 - (76.0 / 113.0) * eta)
    inspiral = torch.stack([
        getattr(XHMParameterFits, f"amp_insp{i}_21")(eta, chi1, chi2, chi_pn_hat)
        for i in (3, 2, 1)
    ])
    intermediate = torch.stack([
        getattr(XHMParameterFits, f"amp_int{i}_21")(eta, chi1, chi2, spin)
        for i in (1, 2)
    ])
    return mass, chi1, chi2, inspiral, intermediate


def test_odd_inspiral_targets_are_abs_before_production_collocation(monkeypatch):
    """The actual inspiral collocation target uses |raw odd-m fit|."""
    mass, chi1, chi2, raw_inspiral, _ = _equal_mass_odd_fit_values()
    assert (raw_inspiral < 0).all()

    target = []
    pn_values = []
    original_matmul = xhm.torch.matmul
    original_pn = xhm._pn_amplitude_ratio

    def capture_matmul(left, right, *args, **kwargs):
        if tuple(left.shape) == (3,) and tuple(right.shape[-2:]) == (3, 3):
            target.append(left.detach().clone())
        return original_matmul(left, right, *args, **kwargs)

    def capture_pn(l, m, frequency, *args, **kwargs):
        value = original_pn(l, m, frequency, *args, **kwargs)
        if (l, m) == (2, 1) and len(pn_values) < 3:
            pn_values.append(value.detach().clone())
        return value

    monkeypatch.setattr(xhm.torch, "matmul", capture_matmul)
    monkeypatch.setattr(xhm, "_pn_amplitude_ratio", capture_pn)
    IMRPhenomXHM(modes=((2, 1),))(
        torch.tensor([20.0, 30.0, 40.0], dtype=DTYPE),
        mass1=mass, mass2=mass, chi1=chi1, chi2=chi2,
        distance=100.0, return_unprojected_modes=True,
    )

    assert len(target) == 1
    assert len(pn_values) == 3
    expected = raw_inspiral.abs() - torch.stack(pn_values)
    torch.testing.assert_close(target[0], expected, rtol=5e-12, atol=0)


def test_odd_intermediate_targets_are_abs_and_gradients_are_finite(monkeypatch):
    """The six-condition production solve receives 1/|raw odd-m fit|."""
    mass, chi1, chi2, _, raw_intermediate = _equal_mass_odd_fit_values()
    assert (raw_intermediate < 0).all()
    rhs_targets = []
    original_solve = torch.linalg.solve

    def capture_solve(matrix, rhs, *args, **kwargs):
        if tuple(matrix.shape[-2:]) == (6, 6):
            rhs_targets.append(rhs.detach().clone())
        return original_solve(matrix, rhs, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "solve", capture_solve)
    frequency = torch.tensor([20.0, 30.0, 40.0], dtype=DTYPE, requires_grad=True)
    mass_input = mass.detach().clone().requires_grad_()
    chi1_input = chi1.detach().clone().requires_grad_()
    chi2_input = chi2.detach().clone().requires_grad_()
    mode = IMRPhenomXHM(modes=((2, 1),))(
        frequency, mass1=mass_input, mass2=mass_input,
        chi1=chi1_input, chi2=chi2_input, distance=100.0,
        return_unprojected_modes=True,
    )[2][(2, 1)]
    gradients = torch.autograd.grad(
        (mode.abs() * 1.0e22).sum(),
        (frequency, mass_input, chi1_input, chi2_input),
    )

    assert len(rhs_targets) == 1
    torch.testing.assert_close(
        rhs_targets[0][2:4], 1.0 / raw_intermediate.abs(),
        rtol=5e-12, atol=0,
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.abs(gradient).max() > 0.0


def test_even_mode_intermediate_target_sign_is_unchanged(monkeypatch):
    """Even-m consumers retain their existing raw-target behavior."""
    rhs_targets = []
    original_solve = torch.linalg.solve
    monkeypatch.setattr(
        XHMParameterFits, "amp_int1_44",
        staticmethod(lambda eta, *args: -0.25 * torch.ones_like(eta)),
    )
    monkeypatch.setattr(
        XHMParameterFits, "amp_int2_44",
        staticmethod(lambda eta, *args: -0.5 * torch.ones_like(eta)),
    )

    def capture_solve(matrix, rhs, *args, **kwargs):
        if tuple(matrix.shape[-2:]) == (6, 6):
            rhs_targets.append(rhs.detach().clone())
        return original_solve(matrix, rhs, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "solve", capture_solve)
    IMRPhenomXHM(modes=((4, 4),))(
        torch.tensor([20.0, 30.0, 40.0], dtype=DTYPE),
        mass1=35.0, mass2=35.0, chi1=0.6, chi2=-0.3,
        distance=100.0, return_unprojected_modes=True,
    )

    assert len(rhs_targets) == 1
    torch.testing.assert_close(
        rhs_targets[0][2:4], torch.full((2,), 1.0e12, dtype=DTYPE),
        rtol=0.0, atol=0.0,
    )
