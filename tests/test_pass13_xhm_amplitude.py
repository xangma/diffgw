"""Pass 13 checks for the published XHM intermediate-amplitude normalization."""

import pytest
import torch

from torchwave.approximants import imrphenom_xhm as xhm
from torchwave.approximants.imrphenom_xas import XASParameterFits
from torchwave.approximants.imrphenom_xhm import IMRPhenomXHM, _eval_f_isco
from torchwave.approximants.xhm_fits import XHMParameterFits
from torchwave.constants import C_SI, MPC_SI, MSUN_SEC, PI


DTYPE = torch.float64


def _retain_inner_nodes_unless_parameter_veto(mode, q, eta, chi1, chi2, *amplitudes):
    """Isolate collocation normalization/fallback from the ordered node vetoes."""
    alive = ~xhm._published_parameter_amplitude_veto(mode, q, eta, chi1, chi2)
    return alive, alive, torch.zeros_like(alive)


def _intermediate_21_node(index, m1_value=35.0, m2_value=20.0,
                          chi1_value=0.4, chi2_value=-0.2):
    """Return an Eq. (5.1) node and its Eq. (5.3) target q=A^Inter/A0."""
    m1 = torch.tensor(m1_value, dtype=DTYPE)
    m2 = torch.tensor(m2_value, dtype=DTYPE)
    chi1 = torch.tensor(chi1_value, dtype=DTYPE)
    chi2 = torch.tensor(chi2_value, dtype=DTYPE)
    total_mass = m1 + m2
    eta = m1 * m2 / total_mass.square()
    _, _, _, spin_combination = XHMParameterFits.compute_spin_variables(eta, chi1, chi2)
    chi_eff = (m1 * chi1 + m2 * chi2) / total_mass
    chi_f = XASParameterFits.fit_final_spin(eta, chi1, chi2)
    m_f = 1.0 - XASParameterFits.fit_erad(eta, chi1, chi2)
    f_meco = XASParameterFits.fit_f_meco(eta, chi1, chi2)
    f_isco = _eval_f_isco(chi_f)
    f_ins = 0.5 * (f_meco + torch.abs(f_isco - f_meco) *
                   (0.75 - 0.235 * chi_eff - (5.0 / 6.0) * chi_eff.square()))
    f_ring = XHMParameterFits.qnm_fring_21(chi_f) / m_f
    f_rd = 0.75 * f_ring
    f_int = f_ins + index * (f_rd - f_ins) / 3.0
    q = getattr(XHMParameterFits, f"amp_int{index}_21")(eta, chi1, chi2, spin_combination)
    return total_mass * MSUN_SEC, eta, f_int, q


@pytest.mark.parametrize("index", (1, 2))
def test_intermediate_node_matches_eq53_a0_normalized_strain(index, monkeypatch):
    """Eq. (5.3) target gets exactly one (pi*f)^(7/6) conversion to H_lm."""
    monkeypatch.setattr(xhm, "_ordered_amplitude_vetoes", _retain_inner_nodes_unless_parameter_veto)
    mass_seconds, eta, f_int, q = _intermediate_21_node(index)
    frequency = f_int / mass_seconds
    distance_seconds = 100.0 * MPC_SI / C_SI
    # This is the model's native physical leading amplitude A0^22(f).
    a0_phys = (torch.sqrt(5.0 * eta / 24.0) /
               (PI ** (2.0 / 3.0) * distance_seconds) *
               mass_seconds ** (5.0 / 6.0) * frequency ** (-7.0 / 6.0))
    expected = (16.0 * PI / 5.0) ** 0.5 * a0_phys * (PI * f_int) ** (7.0 / 6.0) * q

    model = IMRPhenomXHM(modes=((2, 1),))
    _, _, mode_strains = model(
        f=frequency.reshape(1), mass1=35.0, mass2=20.0,
        chi1=0.4, chi2=-0.2, distance=100.0,
        return_unprojected_modes=True,
    )
    normalized_strain = mode_strains[(2, 1)].abs().squeeze() / expected
    torch.testing.assert_close(normalized_strain, torch.ones_like(normalized_strain),
                               rtol=5e-12, atol=0.0)


def test_intermediate_amplitude_has_finite_strain_gradients():
    """The corrected node conversion remains differentiable in production."""
    _, _, f_int, _ = _intermediate_21_node(1)
    mass_seconds = (35.0 + 20.0) * MSUN_SEC
    frequency = torch.tensor([f_int.item(), f_int.item() + 0.001], dtype=DTYPE) / mass_seconds
    frequency.requires_grad_()
    mass1 = torch.tensor(35.0, dtype=DTYPE, requires_grad=True)
    mass2 = torch.tensor(20.0, dtype=DTYPE, requires_grad=True)
    chi1 = torch.tensor(0.4, dtype=DTYPE, requires_grad=True)
    chi2 = torch.tensor(-0.2, dtype=DTYPE, requires_grad=True)
    model = IMRPhenomXHM(modes=((2, 1),))
    _, _, mode_strains = model(
        f=frequency, mass1=mass1, mass2=mass2, chi1=chi1, chi2=chi2,
        distance=100.0, return_unprojected_modes=True,
    )
    loss = (mode_strains[(2, 1)].abs() * 1e22).sum()
    gradients = torch.autograd.grad(loss, (frequency, mass1, mass2, chi1, chi2))
    assert all(torch.isfinite(gradient).all() and (gradient != 0).any() for gradient in gradients)


def test_pole_fallback_keeps_inner_values_and_differs_from_veto(monkeypatch):
    """A derivative-induced pole uses the four-value cubic, not endpoint Hermite."""
    monkeypatch.setattr(xhm, "_ordered_amplitude_vetoes", _retain_inner_nodes_unless_parameter_veto)
    mass_seconds, eta, f_int1, q1 = _intermediate_21_node(
        1, m1_value=50.0, m2_value=20.0, chi1_value=0.6, chi2_value=-0.8,
    )
    _, _, f_int2, q2 = _intermediate_21_node(
        2, m1_value=50.0, m2_value=20.0, chi1_value=0.6, chi2_value=-0.8,
    )
    f_int = torch.stack([f_int1, f_int2])
    q = torch.stack([q1, q2])
    frequency = f_int / mass_seconds
    distance_seconds = 100.0 * MPC_SI / C_SI
    a0_phys = (torch.sqrt(5.0 * eta / 24.0) /
               (PI ** (2.0 / 3.0) * distance_seconds) *
               mass_seconds ** (5.0 / 6.0) * frequency ** (-7.0 / 6.0))
    expected = (16.0 * PI / 5.0) ** 0.5 * a0_phys * (PI * f_int) ** (7.0 / 6.0) * q

    original_insp2 = XHMParameterFits.amp_insp2_21
    monkeypatch.setattr(XHMParameterFits, "amp_insp2_21",
                        staticmethod(lambda *args: 10.0 * original_insp2(*args)))
    model = IMRPhenomXHM(modes=((2, 1),))
    options = dict(f=frequency, mass1=50.0, mass2=20.0,
                   chi1=0.6, chi2=-0.8, distance=100.0,
                   return_unprojected_modes=True)

    original_solve = torch.linalg.solve
    amplitude_solves = []

    def capture_amplitude_solve(matrix, rhs, *args, **kwargs):
        if matrix.shape[-2:] == (6, 6) and not amplitude_solves:
            amplitude_solves.append((matrix.detach().clone(), rhs.detach().clone()))
        return original_solve(matrix, rhs, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "solve", capture_amplitude_solve)
    pole_fallback = model(**options)[2][(2, 1)].abs().squeeze()
    torch.testing.assert_close(pole_fallback / expected, torch.ones_like(expected),
                               rtol=5e-12, atol=0.0)
    assert len(amplitude_solves) == 1

    # Independently inspect the production six-condition solve. The perturbed
    # endpoint derivative makes the quintic denominator genuinely negative,
    # while the four positive node values remain intact.
    matrix, rhs = amplitude_solves[0]
    coefficients = original_solve(matrix, rhs)
    p5_nodes = torch.linspace(matrix[0, 1], matrix[4, 1], 11,
                              dtype=matrix.dtype, device=matrix.device)
    p5_check = sum(coefficients[i] * p5_nodes**i for i in range(6))
    threshold = 0.05 * torch.minimum(rhs[0], rhs[4])
    assert (p5_check <= 0).any()
    value_nodes = torch.stack((matrix[0, 1], matrix[2, 1], matrix[3, 1], matrix[4, 1]))
    value_targets = torch.stack((rhs[0], rhs[2], rhs[3], rhs[4]))
    value_matrix = torch.stack(
        (torch.ones_like(value_nodes), value_nodes, value_nodes**2, value_nodes**3), dim=-1,
    )
    value_coefficients = original_solve(value_matrix, value_targets)
    value_cubic = sum(value_coefficients[i] * p5_nodes**i for i in range(4))
    assert (value_cubic > threshold).all()

    frequency.requires_grad_()
    pole_with_grad = model(**dict(options, f=frequency))[2][(2, 1)]
    gradient = torch.autograd.grad((pole_with_grad.abs() * 1e22).sum(), frequency)[0]
    assert torch.isfinite(gradient).all() and (gradient != 0).any()

    # Force only the parameter-veto branch: it removes both inner values and
    # therefore must differ from the pole fallback at the same inner node.
    monkeypatch.setattr(
        xhm, "_published_parameter_amplitude_veto",
        lambda mode, q_value, eta_value, c1, c2: torch.ones_like(eta_value, dtype=torch.bool),
    )
    endpoint_fallback = model(**options)[2][(2, 1)].abs().squeeze()
    relative_change = torch.abs(endpoint_fallback - pole_fallback) / pole_fallback
    assert (relative_change > 1e-4).all()
