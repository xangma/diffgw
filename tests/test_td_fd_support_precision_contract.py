"""Focused numerical checks for the clean TD support arithmetic."""

import numpy as np
import pytest
import torch

from torchwave.constants import C_SI, G_SI, MSUN_SEC, MSUN_SI, PI
from torchwave.approximants.eob.td_fd_conditioning_clean import plan_td_support


def test_taylor_t3_start_uses_direct_three_eighths_relation():
    plan = plan_td_support(30.0, 20.0, 20.0, dtype=torch.float64)
    m1_si = torch.tensor(30.0, dtype=torch.float64) * MSUN_SI
    m2_si = torch.tensor(20.0, dtype=torch.float64) * MSUN_SI
    total_si = m1_si + m2_si
    inverse_time_numerator = (5.0 * total_si) * (MSUN_SEC / MSUN_SI)
    prefactor = 1.0 / (8.0 * plan.M.new_tensor(PI) * plan.M)
    expected = (
        torch.pow(
            inverse_time_numerator / (plan.eta * plan.support_time), 3.0 / 8.0
        )
        * prefactor
    )
    torch.testing.assert_close(plan.start_frequency, expected, rtol=0.0, atol=0.0)


def test_binary64_eta_follows_si_reduced_mass_definition():
    m1 = torch.tensor([30.0, 1.4, 42.0], dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor([20.0, 1.4, 17.0], dtype=torch.float64)
    plan = plan_td_support(m1, m2, torch.tensor([20.0, 30.0, 35.0], dtype=torch.float64))

    m1_si = m1 * m1.new_tensor(MSUN_SI)
    m2_si = m2 * m1.new_tensor(MSUN_SI)
    total_si = m1_si + m2_si
    expected_eta = (m1_si * m2_si / total_si) / total_si
    expected_mass_seconds = total_si * m1.new_tensor(G_SI / (C_SI**3))
    torch.testing.assert_close(plan.eta, expected_eta, rtol=0.0, atol=0.0)
    torch.testing.assert_close(plan.M, expected_mass_seconds, rtol=0.0, atol=0.0)

    plan.start_frequency.sum().backward()
    assert m1.grad is not None and torch.isfinite(m1.grad).all()


def test_float32_mass_path_is_finite_and_preserves_batch_dtype():
    plan = plan_td_support(
        torch.tensor([30.0, 1.4], dtype=torch.float32),
        torch.tensor([20.0, 1.4], dtype=torch.float32),
        torch.tensor([20.0, 30.0], dtype=torch.float32),
    )
    assert plan.eta.dtype == torch.float32
    assert plan.start_frequency.dtype == torch.float32
    assert torch.isfinite(plan.eta).all()
    assert torch.isfinite(plan.start_frequency).all()


def test_start_frequency_gradient_matches_central_difference():
    mass1 = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)
    plan = plan_td_support(mass1, 20.0, 20.0, dtype=torch.float64)
    analytic = torch.autograd.grad(plan.start_frequency, mass1)[0].item()
    epsilon = 1.0e-3
    upper = plan_td_support(30.0 + epsilon, 20.0, 20.0, dtype=torch.float64)
    lower = plan_td_support(30.0 - epsilon, 20.0, 20.0, dtype=torch.float64)
    finite_difference = (
        upper.start_frequency.item() - lower.start_frequency.item()
    ) / (2.0 * epsilon)
    assert finite_difference == pytest.approx(analytic, rel=2.0e-9, abs=2.0e-11)


@pytest.mark.parametrize(
    "m1,m2,f_low,chi1,chi2",
    (
        (30.0, 20.0, 20.0, 0.0, 0.0),
        (1.4, 1.4, 30.0, 0.0, 0.0),
        (80.0, 60.0, 80.0, 0.0, 0.0),
        (42.0, 17.0, 35.0, -0.3, 0.2),
        (5.0, 4.0, 20.0, 0.7, 0.2),
        (100.0, 5.0, 10.0, -0.95, 0.0),
    ),
)
def test_public_support_bounds_are_finite_controls(m1, m2, f_low, chi1, chi2):
    lal = pytest.importorskip("lal")
    lalsim = pytest.importorskip("lalsimulation")
    plan = plan_td_support(m1, m2, f_low, chi1, chi2, dtype=torch.float64)
    m1_si = m1 * lal.MSUN_SI
    m2_si = m2 * lal.MSUN_SI
    total_si = m1_si + m2_si
    safe = float(plan.f_safe)
    chirp = float(lalsim.SimInspiralChirpTimeBound(safe, m1_si, m2_si, chi1, chi2))
    merge = float(lalsim.SimInspiralMergeTimeBound(m1_si, m2_si))
    ringdown = float(
        lalsim.SimInspiralRingdownTimeBound(total_si, float(plan.spin_bound))
    )
    support = 1.1 * chirp + (merge + ringdown) + 3.0 / safe
    public_start = float(
        lalsim.SimInspiralChirpStartFrequencyBound(support, m1_si, m2_si)
    )
    values = (chirp, merge, ringdown, support, public_start)
    assert all(np.isfinite(value) and value > 0.0 for value in values)
    # The public bound APIs are independent scalar controls.  Their timing
    # arithmetic agrees with the clean planner to the stated scalar tolerance.
    assert abs(float(plan.chirp_time) - chirp) <= 2.0e-13 * max(1.0, abs(chirp))
    assert abs(float(plan.plunge_time) - merge) <= 2.0e-13 * max(1.0, abs(merge))
    assert abs(float(plan.ringdown_time) - ringdown) <= 2.0e-13 * max(1.0, abs(ringdown))
    assert abs(float(plan.support_time) - support) <= 2.0e-13 * max(1.0, abs(support))
    assert abs(float(plan.start_frequency) - public_start) <= 2.0e-13 * max(1.0, abs(public_start))
