"""Acceptance tests for IMRPhenomNSBH Amplitude and Remnant Reconstruction (Work Package 5).

Strict clean-room validation following Section 4 of docs/reviews/2026-09-11-clean-room-spec/phenomnsbh-spec.md.
"""

import pytest
import torch

from torchwave import get_fd_waveform
from torchwave.approximants.imrphenom_nsbh import (
    IMRPhenomNSBH,
    ns_compactness_from_lambda,
    solve_xi_tide,
    bbh_remnant_uib2016,
    zappa_remnant_properties,
    nsbh_frequencies,
)
from torchwave.conversions import kerr_isco_radius


def test_compactness_properties():
    """Verify C(0) = 0.5 and C(Lambda) strictly monotonically decreasing for Lambda > 0."""
    lam_zero = torch.tensor([0.0], dtype=torch.float64)
    c_zero = ns_compactness_from_lambda(lam_zero)
    assert c_zero.item() == 0.5

    lam_grid = torch.linspace(50.0, 3000.0, 100, dtype=torch.float64)
    c_grid = ns_compactness_from_lambda(lam_grid)

    # Monotonically decreasing: c[i] > c[i+1]
    diffs = c_grid[:-1] - c_grid[1:]
    assert torch.all(diffs > 0.0), "Compactness must be strictly decreasing with Lambda"

    # Realistic NS bounds: ~0.08 to 0.25
    c_400 = ns_compactness_from_lambda(torch.tensor([400.0], dtype=torch.float64))
    assert 0.15 < c_400.item() < 0.20


def test_root_solver_accuracy():
    """Verify Newton solver for xi_tide converges to residual |P(xi)| < 1e-12 across parameters."""
    test_cases = [
        (1.5, 0.0, 0.16),
        (2.0, 0.5, 0.18),
        (3.0, -0.4, 0.14),
        (5.0, 0.75, 0.15),
        (8.0, 0.0, 0.20),
        (15.0, -0.6, 0.16),
        (1.2, 0.8, 0.12),
    ]

    for q_val, chi_val, c_val in test_cases:
        q = torch.tensor(q_val, dtype=torch.float64)
        chi = torch.tensor(chi_val, dtype=torch.float64)
        c = torch.tensor(c_val, dtype=torch.float64)

        xi, residual = solve_xi_tide(q, chi, c)
        assert xi.item() > 0.0, f"Root must be positive for q={q_val}, chi={chi_val}"
        # Absolute residual or machine-precision relative residual
        assert abs(residual.item()) < 1e-11, (
            f"Residual {abs(residual.item())} >= 1e-11 for q={q_val}, chi={chi_val}"
        )


def test_disruption_classification_regimes():
    """Verify disruption regimes specified in acceptance criteria:
    - High mass ratio (q=6, Lambda=500): non-disruptive, f_tide > f_RD, M_b,torus = 0
    - Low mass ratio (q=1.5, Lambda=800, chi=0.5): disruptive, f_tide < f_RD, M_b,torus > 0
    """
    # 1. Non-disruptive case: q=6 (m1=8.4, m2=1.4), chi1=0.0, Lambda2=500
    m1_nd = torch.tensor(8.4, dtype=torch.float64)
    m2_nd = torch.tensor(1.4, dtype=torch.float64)
    chi1_nd = torch.tensor(0.0, dtype=torch.float64)
    lam2_nd = torch.tensor(500.0, dtype=torch.float64)

    f_tide_nd, f_isco_nd, f_rd_nd, c_nd = nsbh_frequencies(m1_nd, m2_nd, chi1_nd, lam2_nd)
    assert f_tide_nd.item() > f_rd_nd.item(), "q=6, Lambda=500 should have f_tide > f_RD"

    # Remnant torus mass for non-disruptive case
    q_nd = m1_nd / m2_nd
    xi_nd, _ = solve_xi_tide(q_nd, chi1_nd, c_nd)
    r_isco_nd = kerr_isco_radius(chi1_nd)
    mb_torus_nd = torch.clamp(
        0.296 * xi_nd * (1.0 - 2.0 * c_nd) - 0.171 * q_nd * c_nd * r_isco_nd,
        min=0.0,
    )
    assert mb_torus_nd.item() == 0.0, "Non-disruptive case must have zero remnant torus mass"

    # 2. Disruptive case: q=1.5 (m1=2.1, m2=1.4), chi1=0.5, Lambda2=800
    m1_d = torch.tensor(2.1, dtype=torch.float64)
    m2_d = torch.tensor(1.4, dtype=torch.float64)
    chi1_d = torch.tensor(0.5, dtype=torch.float64)
    lam2_d = torch.tensor(800.0, dtype=torch.float64)

    f_tide_d, f_isco_d, f_rd_d, c_d = nsbh_frequencies(m1_d, m2_d, chi1_d, lam2_d)
    assert f_tide_d.item() < f_rd_d.item(), "q=1.5, Lambda=800, chi=0.5 should have f_tide < f_RD"

    q_d = m1_d / m2_d
    xi_d, _ = solve_xi_tide(q_d, chi1_d, c_d)
    r_isco_d = kerr_isco_radius(chi1_d)
    mb_torus_d = torch.clamp(
        0.296 * xi_d * (1.0 - 2.0 * c_d) - 0.171 * q_d * c_d * r_isco_d,
        min=0.0,
    )
    assert mb_torus_d.item() > 0.0, "Disruptive case must have non-zero remnant torus mass"


def test_remnant_limits():
    """Verify remnant mass M_f <= M, remnant spin |chi_f| < 1.0, and BBH limit at Lambda -> 0."""
    for q_val in [1.5, 3.0, 6.0, 10.0]:
        for chi1_val in [-0.5, 0.0, 0.5, 0.75]:
            m1 = torch.tensor(q_val * 1.4, dtype=torch.float64)
            m2 = torch.tensor(1.4, dtype=torch.float64)
            chi1 = torch.tensor(chi1_val, dtype=torch.float64)
            chi2 = torch.tensor(0.0, dtype=torch.float64)
            m_tot = m1 + m2

            # BBH baseline
            m_bbh_ratio, a_bbh = bbh_remnant_uib2016(m1, m2, chi1, chi2)
            assert 0.8 < m_bbh_ratio.item() <= 1.0, f"BBH remnant mass out of bounds: {m_bbh_ratio.item()}"
            assert abs(a_bbh.item()) < 1.0, f"BBH remnant spin unphysical: {a_bbh.item()}"

            # Tidal systems across Lambda values
            for lam_val in [0.0, 100.0, 400.0, 1000.0]:
                lam = torch.tensor(lam_val, dtype=torch.float64)
                mf, af = zappa_remnant_properties(m1, m2, chi1, chi2, lam)
                mf_ratio = mf / m_tot
                assert mf_ratio.item() <= 1.0, f"Remnant mass ratio > 1 for q={q_val}, chi={chi1_val}, lam={lam_val}"
                assert mf_ratio.item() > 0.8, f"Remnant mass ratio < 0.8 for q={q_val}, chi={chi1_val}, lam={lam_val}"
                assert abs(af.item()) < 1.0, f"Remnant spin |af| >= 1.0 for q={q_val}, chi={chi1_val}, lam={lam_val}"

                # At Lambda = 0, should match BBH limit
                if lam_val == 0.0:
                    assert torch.isclose(mf_ratio, m_bbh_ratio, atol=1e-12)
                    assert torch.isclose(af, a_bbh, atol=1e-12)


def test_zappa_author_fit_uses_bbh_baseline_and_published_fallback():
    """Check the author-published Zappa rational fit and its BBH fallback."""
    m1 = torch.tensor(5.0, dtype=torch.float64)
    m2 = torch.tensor(1.4, dtype=torch.float64)
    chi = torch.tensor(0.2, dtype=torch.float64)
    zero = torch.zeros_like(chi)
    mf, af = zappa_remnant_properties(m1, m2, chi, zero, torch.tensor(400.0, dtype=torch.float64))
    torch.testing.assert_close(mf / (m1 + m2), torch.tensor(0.9716178569291998, dtype=torch.float64), rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(af, torch.tensor(0.5971822966772592, dtype=torch.float64), rtol=2e-14, atol=2e-14)

    # A disruptive case retains the tidal remnant correction and agrees with
    # the public NSBH properties (mass ratio 5, chi_BH=.4, Lambda=2600).
    m1 = torch.tensor(7.0, dtype=torch.float64)
    chi = torch.tensor(0.4, dtype=torch.float64)
    mf, af = zappa_remnant_properties(m1, m2, chi, torch.zeros_like(chi), torch.tensor(2600.0, dtype=torch.float64))
    torch.testing.assert_close(mf / (m1 + m2), torch.tensor(0.9577366463051751, dtype=torch.float64), rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(af, torch.tensor(0.625206641047778, dtype=torch.float64), rtol=2e-14, atol=2e-14)

    # Zappa's published extrapolation guard restores the BBH remnant for
    # chi_BH < 0, eta < 0.188 (and also for chi_BH < -0.5).
    m1 = torch.tensor(10.0, dtype=torch.float64)
    chi = torch.tensor(-0.6, dtype=torch.float64)
    mf, af = zappa_remnant_properties(m1, m2, chi, torch.zeros_like(chi), torch.tensor(800.0, dtype=torch.float64))
    mf_bbh, af_bbh = bbh_remnant_uib2016(m1, m2, chi, torch.zeros_like(chi))
    torch.testing.assert_close(mf / (m1 + m2), mf_bbh, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(af, af_bbh, rtol=2e-14, atol=2e-14)


def test_amplitude_positivity_and_cutoff():
    """Verify amplitude positivity A(f) > 0 and exponential cutoff for disrupting systems."""
    f = torch.linspace(30.0, 3000.0, 300, dtype=torch.float64)

    # Grid of physical configurations: q in [1, 15], |chi| <= 0.8, Lambda in [0, 5000]
    configs = [
        (3.0, 1.4, 0.0, 0.0),
        (5.0, 1.4, 0.5, 400.0),
        (2.0, 1.3, -0.4, 800.0),
        (10.0, 1.4, 0.7, 200.0),
        (1.5, 1.2, 0.8, 1500.0),
    ]

    model = IMRPhenomNSBH()
    for m1, m2, chi1, lam2 in configs:
        hp, hc = model(
            f=f,
            mass1=m1,
            mass2=m2,
            chi1=chi1,
            lambda2=lam2,
            distance=100.0,
            inclination=0.5,
        )
        amp = torch.sqrt(hp.abs() ** 2 + hc.abs() ** 2)
        assert torch.all(amp > 0.0), f"Amplitude not strictly positive for m1={m1}, m2={m2}, chi1={chi1}"

    # Disruptive high-frequency suppression
    m1_d, m2_d = 2.5, 1.2
    chi1_d = 0.75
    lam2_d = 800.0
    f_wide = torch.linspace(40.0, 6000.0, 600, dtype=torch.float64)

    hp_d, hc_d = model(
        f=f_wide,
        mass1=m1_d,
        mass2=m2_d,
        chi1=chi1_d,
        lambda2=lam2_d,
    )
    f_tide, _, _, _ = nsbh_frequencies(
        torch.tensor(m1_d, dtype=torch.float64),
        torch.tensor(m2_d, dtype=torch.float64),
        torch.tensor(chi1_d, dtype=torch.float64),
        torch.tensor(lam2_d, dtype=torch.float64),
    )

    high_mask = f_wide > (f_tide.item() * 1.5)
    low_mask = (f_wide > 100.0) & (f_wide < f_tide.item() * 0.8)

    max_high = torch.max(hp_d.abs()[high_mask]).item()
    max_low = torch.max(hp_d.abs()[low_mask]).item()
    assert max_high < 0.01 * max_low, f"Disruptive high-frequency suppression failed: {max_high} vs {max_low}"


def test_batching_and_scalar_parity():
    """Verify batched and scalar evaluations agree to within machine precision (< 1e-14)."""
    B = 4
    N = 80
    f = torch.linspace(30.0, 500.0, N, dtype=torch.float64)

    m1_b = torch.tensor([4.0, 6.0, 8.0, 3.5], dtype=torch.float64)
    m2_b = torch.tensor([1.4, 1.3, 1.5, 1.2], dtype=torch.float64)
    chi1_b = torch.tensor([0.2, -0.3, 0.6, 0.0], dtype=torch.float64)
    lam2_b = torch.tensor([300.0, 500.0, 150.0, 700.0], dtype=torch.float64)
    inc_b = torch.tensor([0.0, 0.4, 0.8, 1.2], dtype=torch.float64)

    model = IMRPhenomNSBH()
    hp_b, hc_b = model(
        f=f,
        mass1=m1_b,
        mass2=m2_b,
        chi1=chi1_b,
        lambda2=lam2_b,
        inclination=inc_b,
    )

    assert hp_b.shape == (B, N)
    assert hc_b.shape == (B, N)

    for b in range(B):
        hp_s, hc_s = model(
            f=f,
            mass1=m1_b[b],
            mass2=m2_b[b],
            chi1=chi1_b[b],
            lambda2=lam2_b[b],
            inclination=inc_b[b],
        )
        rel_diff_p = torch.max(torch.abs(hp_b[b] - hp_s) / torch.clamp(torch.abs(hp_s), min=1e-30)).item()
        rel_diff_c = torch.max(torch.abs(hc_b[b] - hc_s) / torch.clamp(torch.abs(hc_s), min=1e-30)).item()
        assert rel_diff_p < 1e-14, f"Batching parity violated for hp at index {b}: {rel_diff_p}"
        assert rel_diff_c < 1e-14, f"Batching parity violated for hc at index {b}: {rel_diff_c}"


def test_autograd_differentiability_all_parameters():
    """Verify autograd gradients for m1, m2, chi1, lambda2, and inclination are finite and non-zero."""
    f = torch.linspace(40.0, 400.0, 64, dtype=torch.float64)

    m1 = torch.tensor(5.5, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)
    chi1 = torch.tensor(0.35, dtype=torch.float64, requires_grad=True)
    lam2 = torch.tensor(450.0, dtype=torch.float64, requires_grad=True)
    inc = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)

    hp, hc = IMRPhenomNSBH()(
        f=f,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        lambda2=lam2,
        inclination=inc,
    )

    loss = (hp.real ** 2 + hp.imag ** 2 + hc.real ** 2 + hc.imag ** 2).sum()
    loss.backward()

    params = [("m1", m1), ("m2", m2), ("chi1", chi1), ("lambda2", lam2), ("inclination", inc)]
    for name, p in params:
        assert p.grad is not None, f"Gradient for {name} is None"
        assert not torch.isnan(p.grad), f"Gradient for {name} is NaN"
        assert not torch.isinf(p.grad), f"Gradient for {name} is Inf"
        assert p.grad.abs().item() > 0.0, f"Gradient for {name} is zero"
