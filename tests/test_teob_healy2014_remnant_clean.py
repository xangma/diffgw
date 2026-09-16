"""Focused tests for the clean-room Healy/TEOB remnant component."""

import math

import pytest
import torch

from torchwave.approximants.eob.teob_healy2014_remnant_clean import (
    PAPER_PROGRADE_ISCO,
    TEOB_COMPATIBILITY_ISCO,
    healy2014_isco_radius,
    healy2014_nonspinning_remnant,
    healy2014_nonspinning_remnant_details,
    healy2014_nonspinning_remnant_from_q,
    healy2014_nonspinning_final_mass_fraction_from_q,
    healy2014_nonspinning_final_spin_from_q,
    healy2014_specific_isco_angular_momentum,
    healy2014_specific_isco_energy,
    jimenez_forteza_printed_table_nonspinning_final_spin,
    teob_hybrid_nonspinning_remnant,
    teob_hybrid_nonspinning_remnant_from_q,
)
from torchwave.approximants.eob.teob_uib2016_remnant import (
    uib2016_nonspinning_final_spin,
)


def test_healy_compatibility_branch_matches_published_validation_values():
    # These are independent arithmetic validation points for the printed
    # Table-VI coefficients, not coefficients used to construct the fit.
    expected = {
        1.0: (0.951507, 0.686710),
        1.5: (0.9551044071058838, 0.6638802503262305),
        2.0: (0.9610590454297351, 0.6231550522011116),
        4.0: (0.9779010720182864, 0.4726323910856091),
    }
    for q, (mass_expected, spin_expected) in expected.items():
        mass, spin = healy2014_nonspinning_remnant_from_q(
            q, isco_branch=TEOB_COMPATIBILITY_ISCO
        )
        assert mass.item() == pytest.approx(mass_expected, rel=2e-14, abs=2e-15)
        assert spin.item() == pytest.approx(spin_expected, rel=2e-14, abs=2e-15)


def test_paper_prograde_branch_is_explicit_and_distinct():
    mass_paper, spin_paper = healy2014_nonspinning_remnant_from_q(
        1.5, isco_branch=PAPER_PROGRADE_ISCO
    )
    mass_compat, spin_compat = healy2014_nonspinning_remnant_from_q(
        1.5, isco_branch=TEOB_COMPATIBILITY_ISCO
    )
    assert mass_paper.item() == pytest.approx(0.9551040534973075, rel=2e-14)
    assert spin_paper.item() == pytest.approx(0.663877542047339, rel=2e-14)
    assert abs(mass_paper.item() - mass_compat.item()) > 1e-7
    assert abs(spin_paper.item() - spin_compat.item()) > 1e-7

    details = healy2014_nonspinning_remnant_details(1.0, 1.0)
    assert details.isco_branch == PAPER_PROGRADE_ISCO
    # At equal mass the solved remnant spin is nonzero, so the signed radius
    # is below Schwarzschild.
    assert details.isco_radius.item() < 6.0


def test_paper_and_compatibility_radius_formulas_are_named():
    spin = torch.tensor([0.0, 0.5, -0.5], dtype=torch.float64)
    z1 = 1.0 + (1.0 - spin.square()).pow(1.0 / 3.0) * (
        (1.0 + spin).pow(1.0 / 3.0) + (1.0 - spin).pow(1.0 / 3.0)
    )
    z2 = torch.sqrt(3.0 * spin.square() + z1.square())
    radicand = torch.clamp((3.0 - z1) * (3.0 + z1 + 2.0 * z2), min=0.0)
    expected_compat = 3.0 + z2
    expected_paper = 3.0 + z2 - torch.sign(spin) * torch.sqrt(radicand)
    torch.testing.assert_close(
        healy2014_isco_radius(spin, isco_branch="compatibility"),
        expected_compat,
        rtol=2e-14,
        atol=2e-15,
    )
    torch.testing.assert_close(
        healy2014_isco_radius(spin, isco_branch="paper"),
        expected_paper,
        rtol=2e-14,
        atol=2e-15,
    )


def test_printed_table_spin_uses_the_three_table_i_decimals():
    expected = {
        0.24: 0.6641354238886245,
        0.25: 0.6864792261527318,
        0.16: 0.47168785293115334,
    }
    for eta, value in expected.items():
        actual = jimenez_forteza_printed_table_nonspinning_final_spin(eta)
        assert actual.item() == pytest.approx(value, rel=2e-14, abs=2e-15)
    # This guards the model-version distinction from accidental replacement
    # by the full-precision standalone UIB helper.
    assert abs(
        jimenez_forteza_printed_table_nonspinning_final_spin(0.25).item()
        - uib2016_nonspinning_final_spin(0.25).item()
    ) > 1e-6


def test_hybrid_reuses_compatibility_mass_and_replaces_spin():
    mass, spin = teob_hybrid_nonspinning_remnant(1.5, 1.0)
    expected_mass, _ = healy2014_nonspinning_remnant(
        1.5, 1.0, isco_branch=TEOB_COMPATIBILITY_ISCO
    )
    expected_spin = jimenez_forteza_printed_table_nonspinning_final_spin(0.24)
    torch.testing.assert_close(mass, expected_mass, rtol=0, atol=0)
    torch.testing.assert_close(spin, expected_spin, rtol=2e-14, atol=2e-15)
    with pytest.raises(ValueError):
        teob_hybrid_nonspinning_remnant(1.5, 1.0, isco_branch=PAPER_PROGRADE_ISCO)
    mass_q, spin_q = teob_hybrid_nonspinning_remnant_from_q(1.5)
    torch.testing.assert_close(mass_q, mass, rtol=2e-14, atol=2e-15)
    torch.testing.assert_close(spin_q, spin, rtol=2e-14, atol=2e-15)


def test_implicit_spin_equation_is_satisfied_without_rounding_or_clipping():
    details = healy2014_nonspinning_remnant_details(
        2.0, 1.0, isco_branch=TEOB_COMPATIBILITY_ISCO
    )
    eta, delta = details.eta, details.delta_m
    base = (4.0 * eta).square() * (
        0.686710 + 0.801838 * delta.square() + 0.953458 * delta.pow(4)
    )
    rhs = base + eta * delta.pow(6) * healy2014_specific_isco_angular_momentum(
        details.isco_radius, details.spin
    )
    torch.testing.assert_close(details.spin, rhs, rtol=3e-14, atol=3e-15)


def test_compatibility_energy_retains_spin_dependence_at_signless_radius():
    spin = torch.tensor(0.6, dtype=torch.float64)
    radius = healy2014_isco_radius(spin, isco_branch=TEOB_COMPATIBILITY_ISCO)
    energy = healy2014_specific_isco_energy(
        radius, spin, isco_branch=TEOB_COMPATIBILITY_ISCO
    )
    sqrt_r = torch.sqrt(radius)
    expected = (radius * sqrt_r - 2.0 * sqrt_r + spin) / (
        radius.pow(0.75) * torch.sqrt(radius * sqrt_r - 3.0 * sqrt_r + 2.0 * spin)
    )
    torch.testing.assert_close(energy, expected, rtol=2e-14, atol=2e-15)
    assert abs(energy.item() - torch.sqrt(1.0 - 2.0 / (3.0 * radius)).item()) > 1e-5


def test_mass_batching_scaling_q_equivalence_and_gradients():
    mass1 = torch.tensor([[2.0], [10.0]], dtype=torch.float64, requires_grad=True)
    mass2 = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True)
    mf, spin = teob_hybrid_nonspinning_remnant(mass1, mass2)
    assert mf.shape == (2, 3)
    assert spin.shape == (2, 3)
    assert mf.dtype == torch.float64
    assert torch.isfinite(mf).all() and torch.isfinite(spin).all()
    gradients = torch.autograd.grad((mf + spin).sum(), (mass1, mass2))
    assert all(torch.isfinite(value).all() for value in gradients)
    assert all((value.abs() > 0).all() for value in gradients)

    m_q, s_q = teob_hybrid_nonspinning_remnant(2.0, 1.0)
    m_scaled, s_scaled = teob_hybrid_nonspinning_remnant(2.0e200, 1.0e200)
    torch.testing.assert_close(m_q, m_scaled, rtol=2e-14, atol=2e-15)
    torch.testing.assert_close(s_q, s_scaled, rtol=2e-14, atol=2e-15)
    m_from_q, s_from_q = teob_hybrid_nonspinning_remnant(2.0, 1.0)
    m_q_api, s_q_api = healy2014_nonspinning_remnant_from_q(
        2.0, isco_branch=TEOB_COMPATIBILITY_ISCO
    )
    torch.testing.assert_close(
        healy2014_nonspinning_final_mass_fraction_from_q(
            2.0, isco_branch=TEOB_COMPATIBILITY_ISCO
        ),
        m_q_api,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        healy2014_nonspinning_final_spin_from_q(
            2.0, isco_branch=TEOB_COMPATIBILITY_ISCO
        ),
        s_q_api,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(m_from_q, m_q_api, rtol=2e-14, atol=2e-15)
    # The hybrid's printed-table spin is intentionally different from the
    # internal Healy spin returned by the direct API.
    assert abs(s_from_q.item() - s_q_api.item()) > 1e-5


@pytest.mark.parametrize(
    "call",
    [
        lambda: healy2014_nonspinning_remnant(0.0, 1.0),
        lambda: healy2014_nonspinning_remnant(float("nan"), 1.0),
        lambda: healy2014_nonspinning_remnant_from_q(0.9),
        lambda: jimenez_forteza_printed_table_nonspinning_final_spin(0.250001),
        lambda: healy2014_isco_radius(1.1),
    ],
)
def test_invalid_domains_are_rejected(call):
    with pytest.raises(ValueError):
        call()


def test_boolean_and_mixed_device_inputs_are_rejected():
    with pytest.raises(TypeError):
        teob_hybrid_nonspinning_remnant(torch.tensor(True), 1.0)
    if torch.cuda.is_available():
        with pytest.raises(ValueError):
            teob_hybrid_nonspinning_remnant(
                torch.tensor(1.0), torch.tensor(1.0, device="cuda")
            )


def test_fixed_point_budget_and_tolerance_are_explicit_failures():
    with pytest.raises(RuntimeError):
        healy2014_nonspinning_remnant_from_q(
            1.5, isco_branch=TEOB_COMPATIBILITY_ISCO, max_iterations=1
        )
    with pytest.raises(ValueError):
        healy2014_nonspinning_remnant_from_q(1.5, tolerance=0.0)


def test_schwarzschild_radius_has_a_finite_gradient():
    spin = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    radius = healy2014_isco_radius(spin, isco_branch=PAPER_PROGRADE_ISCO)
    gradient = torch.autograd.grad(radius, spin)[0]
    assert radius.item() == pytest.approx(6.0, abs=0.0)
    assert torch.isfinite(gradient)
    assert gradient.item() == pytest.approx(-4.0 * math.sqrt(2.0 / 3.0), rel=1e-12)
