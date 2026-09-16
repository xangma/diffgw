"""Tests for IMRPhenomXAS quartic intermediate amplitude collocation (IntermediateAmpVersion104).

Validates:
1. Exact satisfaction of boundary values and midpoint constraint (Eq. 6.7 and Table II of Pratten et al. 2020).
2. Exact C1 derivative continuity matching at both inspiral (f1) and ringdown (f4) boundaries.
3. Batched vs scalar evaluation consistency across parameters.
4. Numerical stability in float32 without underflow or matrix ill-conditioning.
5. Autograd gradient flow and PyTorch gradcheck through the quartic collocation solver.
6. End-to-end amplitude continuity across high-spin and asymmetric binary configurations.
"""

from typing import Tuple
import torch

from torchwave.approximants.imrphenom_xas import (
    IMRPhenomXAS,
    XASParameterFits,
    component_masses_to_eta,
    solve_intermediate_amplitude_quartic,
)
from torchwave.constants import MSUN_SEC, PI


def test_xas_quartic_boundary_and_midpoint_constraints():
    """Verify that the quartic inverse polynomial exactly matches boundary values and midpoint."""
    f1 = torch.tensor(0.025, dtype=torch.float64)
    f4 = torch.tensor(0.080, dtype=torch.float64)
    delta_f = f4 - f1

    val1 = torch.tensor(0.035, dtype=torch.float64)
    d1 = torch.tensor(0.420, dtype=torch.float64)
    val_mid = torch.tensor(0.048, dtype=torch.float64)
    val4 = torch.tensor(0.065, dtype=torch.float64)
    d4 = torch.tensor(-0.150, dtype=torch.float64)

    c0, c1, c2, c3, c4 = solve_intermediate_amplitude_quartic(
        f1, f4, val1, d1, val_mid, val4, d4
    )

    def eval_poly(u):
        return c0 + u * (c1 + u * (c2 + u * (c3 + u * c4)))

    def eval_poly_deriv(u):
        return c1 + u * (2.0 * c2 + u * (3.0 * c3 + u * 4.0 * c4))

    # Condition 1: P(0) = 1 / val1
    y1_exp = 1.0 / val1
    assert torch.allclose(eval_poly(torch.tensor(0.0, dtype=torch.float64)), y1_exp, rtol=1e-13, atol=1e-13)

    # Condition 2: P(0.5) = 1 / val_mid
    ymid_exp = 1.0 / val_mid
    assert torch.allclose(eval_poly(torch.tensor(0.5, dtype=torch.float64)), ymid_exp, rtol=1e-13, atol=1e-13)

    # Condition 3: P(1.0) = 1 / val4
    y4_exp = 1.0 / val4
    assert torch.allclose(eval_poly(torch.tensor(1.0, dtype=torch.float64)), y4_exp, rtol=1e-13, atol=1e-13)

    # Condition 4: P'(0) = -(d1 / val1^2) * delta_f
    s1_exp = -(d1 / (val1 ** 2)) * delta_f
    assert torch.allclose(eval_poly_deriv(torch.tensor(0.0, dtype=torch.float64)), s1_exp, rtol=1e-13, atol=1e-13)

    # Condition 5: P'(1.0) = -(d4 / val4^2) * delta_f
    s4_exp = -(d4 / (val4 ** 2)) * delta_f
    assert torch.allclose(eval_poly_deriv(torch.tensor(1.0, dtype=torch.float64)), s4_exp, rtol=1e-13, atol=1e-13)


def test_xas_intermediate_derivative_continuity():
    """Verify C1 derivative continuity of the reconstructed amplitude A_int = 1 / P(u)."""
    f1 = torch.tensor(0.030, dtype=torch.float64)
    f4 = torch.tensor(0.085, dtype=torch.float64)
    delta_f = f4 - f1

    val1 = torch.tensor(0.040, dtype=torch.float64)
    d1 = torch.tensor(0.500, dtype=torch.float64)
    val_mid = torch.tensor(0.055, dtype=torch.float64)
    val4 = torch.tensor(0.070, dtype=torch.float64)
    d4 = torch.tensor(-0.120, dtype=torch.float64)

    c0, c1, c2, c3, c4 = solve_intermediate_amplitude_quartic(
        f1, f4, val1, d1, val_mid, val4, d4
    )

    def amp_int(f_eval):
        u = (f_eval - f1) / delta_f
        p = c0 + u * (c1 + u * (c2 + u * (c3 + u * c4)))
        return 1.0 / p

    # 1. Exact autograd derivative check at boundaries
    f1_var = f1.clone().detach().requires_grad_(True)
    d1_auto = torch.autograd.grad(amp_int(f1_var), f1_var)[0]
    assert torch.allclose(d1_auto, d1, rtol=1e-12, atol=1e-12), f"Autograd d1 mismatch: {d1_auto} vs {d1}"

    f4_var = f4.clone().detach().requires_grad_(True)
    d4_auto = torch.autograd.grad(amp_int(f4_var), f4_var)[0]
    assert torch.allclose(d4_auto, d4, rtol=1e-12, atol=1e-12), f"Autograd d4 mismatch: {d4_auto} vs {d4}"

    # 2. Second-order one-sided finite-difference stencils (truncation error O(eps^2))
    # Right-sided stencil at f1: (-3*f(x0) + 4*f(x0+h) - f(x0+2h)) / (2h)
    eps = 1e-5
    d1_stencil = (-3.0 * amp_int(f1) + 4.0 * amp_int(f1 + eps) - amp_int(f1 + 2.0 * eps)) / (2.0 * eps)
    assert abs(d1_stencil.item() - d1.item()) < 5e-5, f"d1 stencil error: num={d1_stencil.item()}, expected={d1.item()}"

    # Left-sided stencil at f4: (3*f(x0) - 4*f(x0-h) + f(x0-2h)) / (2h)
    d4_stencil = (3.0 * amp_int(f4) - 4.0 * amp_int(f4 - eps) + amp_int(f4 - 2.0 * eps)) / (2.0 * eps)
    assert abs(d4_stencil.item() - d4.item()) < 5e-5, f"d4 stencil error: num={d4_stencil.item()}, expected={d4.item()}"


def test_xas_intermediate_amplitude_batched():
    """Verify batched evaluation matches scalar evaluation across batches."""
    f1_b = torch.tensor([0.020, 0.028, 0.035], dtype=torch.float64)
    f4_b = torch.tensor([0.075, 0.082, 0.090], dtype=torch.float64)
    val1_b = torch.tensor([0.030, 0.038, 0.045], dtype=torch.float64)
    d1_b = torch.tensor([0.400, 0.460, 0.520], dtype=torch.float64)
    val_mid_b = torch.tensor([0.042, 0.050, 0.058], dtype=torch.float64)
    val4_b = torch.tensor([0.060, 0.068, 0.075], dtype=torch.float64)
    d4_b = torch.tensor([-0.100, -0.130, -0.160], dtype=torch.float64)

    coeffs_b = solve_intermediate_amplitude_quartic(
        f1_b, f4_b, val1_b, d1_b, val_mid_b, val4_b, d4_b
    )

    for i in range(len(f1_b)):
        coeffs_s = solve_intermediate_amplitude_quartic(
            f1_b[i], f4_b[i], val1_b[i], d1_b[i], val_mid_b[i], val4_b[i], d4_b[i]
        )
        for cb, cs in zip(coeffs_b, coeffs_s):
            assert torch.allclose(cb[i], cs, rtol=1e-14, atol=1e-14)


def test_xas_intermediate_amplitude_float32():
    """Verify stability in float32 and relative agreement of evaluated amplitude against float64."""
    f1_64 = torch.tensor(0.025, dtype=torch.float64)
    f4_64 = torch.tensor(0.080, dtype=torch.float64)
    val1_64 = torch.tensor(0.035, dtype=torch.float64)
    d1_64 = torch.tensor(0.420, dtype=torch.float64)
    val_mid_64 = torch.tensor(0.048, dtype=torch.float64)
    val4_64 = torch.tensor(0.065, dtype=torch.float64)
    d4_64 = torch.tensor(-0.150, dtype=torch.float64)

    coeffs_64 = solve_intermediate_amplitude_quartic(
        f1_64, f4_64, val1_64, d1_64, val_mid_64, val4_64, d4_64
    )

    f1_32 = f1_64.float()
    f4_32 = f4_64.float()
    val1_32 = val1_64.float()
    d1_32 = d1_64.float()
    val_mid_32 = val_mid_64.float()
    val4_32 = val4_64.float()
    d4_32 = d4_64.float()

    coeffs_32 = solve_intermediate_amplitude_quartic(
        f1_32, f4_32, val1_32, d1_32, val_mid_32, val4_32, d4_32
    )

    for c in coeffs_32:
        assert torch.isfinite(c), "Quartic coefficient contains NaN/Inf in float32"

    # Evaluate reconstructed amplitude across intermediate grid in both float32 and float64
    u_grid_64 = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)
    c0_64, c1_64, c2_64, c3_64, c4_64 = coeffs_64
    p_64 = c0_64 + u_grid_64 * (c1_64 + u_grid_64 * (c2_64 + u_grid_64 * (c3_64 + u_grid_64 * c4_64)))
    amp_64 = 1.0 / p_64

    u_grid_32 = u_grid_64.float()
    c0_32, c1_32, c2_32, c3_32, c4_32 = coeffs_32
    p_32 = c0_32 + u_grid_32 * (c1_32 + u_grid_32 * (c2_32 + u_grid_32 * (c3_32 + u_grid_32 * c4_32)))
    amp_32 = 1.0 / p_32

    rel_diff = torch.abs(amp_32.double() - amp_64) / amp_64
    max_rel_diff = torch.max(rel_diff).item()
    assert max_rel_diff < 1e-4, f"Float32 amplitude deviates from float64 by {max_rel_diff} (threshold: 1e-4)"


def test_xas_intermediate_amplitude_autograd():
    """Verify clean autograd backpropagation and gradcheck for quartic intermediate collocation."""
    f1 = torch.tensor(0.025, dtype=torch.float64, requires_grad=True)
    f4 = torch.tensor(0.080, dtype=torch.float64, requires_grad=True)
    val1 = torch.tensor(0.035, dtype=torch.float64, requires_grad=True)
    d1 = torch.tensor(0.420, dtype=torch.float64, requires_grad=True)
    val_mid = torch.tensor(0.048, dtype=torch.float64, requires_grad=True)
    val4 = torch.tensor(0.065, dtype=torch.float64, requires_grad=True)
    d4 = torch.tensor(-0.150, dtype=torch.float64, requires_grad=True)

    def eval_at_point(f1_in, f4_in, v1_in, d1_in, vmid_in, v4_in, d4_in):
        c0, c1, c2, c3, c4 = solve_intermediate_amplitude_quartic(
            f1_in, f4_in, v1_in, d1_in, vmid_in, v4_in, d4_in
        )
        u = torch.tensor(0.4, dtype=torch.float64)
        p = c0 + u * (c1 + u * (c2 + u * (c3 + u * c4)))
        return 1.0 / p

    test_inputs = (f1, f4, val1, d1, val_mid, val4, d4)
    assert torch.autograd.gradcheck(eval_at_point, test_inputs, eps=1e-6, atol=1e-5)


def compute_transition_frequencies_hz(m1: float, m2: float, s1: float, s2: float) -> Tuple[float, float]:
    """Compute physical transition frequencies f_trans_amp and f_peak in Hz."""
    m1_t = torch.tensor(m1, dtype=torch.float64)
    m2_t = torch.tensor(m2, dtype=torch.float64)
    s1_t = torch.tensor(s1, dtype=torch.float64)
    s2_t = torch.tensor(s2, dtype=torch.float64)

    total_mass = m1_t + m2_t
    eta = component_masses_to_eta(m1_t, m2_t)
    m_sec = total_mass * MSUN_SEC

    f_meco = XASParameterFits.fit_f_meco(eta, s1_t, s2_t)
    final_spin = XASParameterFits.fit_final_spin(eta, s1_t, s2_t)
    erad = XASParameterFits.fit_erad(eta, s1_t, s2_t)
    m_final = torch.clamp(1.0 - erad, min=0.1)

    f_rd = XASParameterFits.fit_re_omega(final_spin) / m_final
    f_damp = XASParameterFits.fit_im_omega(final_spin) / m_final

    chi_fit = torch.clamp(final_spin, -0.999, 0.999)
    z1 = 1.0 + (1.0 - chi_fit**2) ** (1.0 / 3.0) * (
        torch.sign(1.0 + chi_fit) * torch.abs(1.0 + chi_fit) ** (1.0 / 3.0)
        + torch.sign(1.0 - chi_fit) * torch.abs(1.0 - chi_fit) ** (1.0 / 3.0)
    )
    z2 = torch.sqrt(torch.clamp(3.0 * chi_fit**2 + z1**2, min=1e-12))
    r_isco = 3.0 + z2 - torch.sign(chi_fit) * torch.sqrt(torch.clamp((3.0 - z1) * (3.0 + z1 + 2.0 * z2), min=1e-12))
    f_isco = 1.0 / (PI * (r_isco**1.5 + chi_fit))

    lam = XASParameterFits.fit_rd_amp_gamma2(eta, s1_t, s2_t)
    sigma = XASParameterFits.fit_rd_amp_gamma3(eta, s1_t, s2_t)

    diff_lam = torch.clamp(1.0 - lam**2, min=0.0)
    f_peak = torch.abs(f_rd + f_damp * sigma * (torch.sqrt(diff_lam) - 1.0) / torch.clamp(lam, min=1e-8))

    f_trans_amp = f_meco + 0.25 * (f_isco - f_meco)

    f_trans_hz = (f_trans_amp / m_sec).item()
    f_peak_hz = (f_peak / m_sec).item()
    return f_trans_hz, f_peak_hz


def test_xas_waveform_intermediate_continuity_configurations():
    """Verify full waveform positivity, boundary continuity across transitions, and finite one-sided slopes."""
    cases = [
        {"m1": 30.0, "m2": 30.0, "s1": 0.8, "s2": 0.8},
        {"m1": 70.0, "m2": 10.0, "s1": -0.5, "s2": 0.2},
        {"m1": 35.0, "m2": 30.0, "s1": 0.2, "s2": -0.3},
    ]

    model = IMRPhenomXAS()
    f_grid = torch.linspace(20.0, 800.0, 300, dtype=torch.float64)

    for c in cases:
        # 1. Broadband positivity and finiteness
        hp, hc = model(
            f=f_grid,
            mass1=c["m1"],
            mass2=c["m2"],
            chi1=c["s1"],
            chi2=c["s2"],
            distance=500.0,
            inclination=0.6,
            phi_c=0.4,
            f_ref=20.0,
        )

        assert torch.all(torch.isfinite(hp)), f"Case {c}: hp contains NaN/Inf"
        assert torch.all(torch.isfinite(hc)), f"Case {c}: hc contains NaN/Inf"

        amp = torch.abs(hp)
        assert torch.all(amp > 0.0), f"Case {c}: amplitude contains zero or negative values"

        # 2. Boundary C0 continuity across transition boundaries
        f_trans, f_peak = compute_transition_frequencies_hz(c["m1"], c["m2"], c["s1"], c["s2"])

        # Boundary jump across f_trans (inspiral -> intermediate)
        eps = 1e-4
        f_eval_trans = torch.tensor([f_trans - eps, f_trans + eps], dtype=torch.float64)
        hp_trans, _ = model(
            f=f_eval_trans,
            mass1=c["m1"],
            mass2=c["m2"],
            chi1=c["s1"],
            chi2=c["s2"],
            distance=500.0,
            inclination=0.0,
        )
        amp_left_trans = torch.abs(hp_trans[0])
        amp_right_trans = torch.abs(hp_trans[1])
        rel_jump_trans = (torch.abs(amp_right_trans - amp_left_trans) / amp_left_trans).item()
        assert rel_jump_trans < 1e-5, f"Case {c}: f_trans boundary jump={rel_jump_trans}"

        # Boundary jump across f_peak (intermediate -> ringdown)
        f_eval_peak = torch.tensor([f_peak - eps, f_peak + eps], dtype=torch.float64)
        hp_peak, _ = model(
            f=f_eval_peak,
            mass1=c["m1"],
            mass2=c["m2"],
            chi1=c["s1"],
            chi2=c["s2"],
            distance=500.0,
            inclination=0.0,
        )
        amp_left_peak = torch.abs(hp_peak[0])
        amp_right_peak = torch.abs(hp_peak[1])
        rel_jump_peak = (torch.abs(amp_right_peak - amp_left_peak) / amp_left_peak).item()
        assert rel_jump_peak < 1e-5, f"Case {c}: f_peak boundary jump={rel_jump_peak}"

        # 3. One-sided slope matching across boundaries (C1 continuity)
        h = 1e-3
        f_slopes = torch.tensor(
            [f_trans - h, f_trans, f_trans + h, f_peak - h, f_peak, f_peak + h],
            dtype=torch.float64,
        )
        hp_slopes, _ = model(
            f=f_slopes,
            mass1=c["m1"],
            mass2=c["m2"],
            chi1=c["s1"],
            chi2=c["s2"],
            distance=500.0,
            inclination=0.0,
        )
        amps = torch.abs(hp_slopes)
        slope_left_trans = (amps[1] - amps[0]) / h
        slope_right_trans = (amps[2] - amps[1]) / h
        rel_slope_trans = (torch.abs(slope_right_trans - slope_left_trans) / torch.clamp(torch.abs(slope_left_trans), min=1e-25)).item()
        assert rel_slope_trans < 5e-3, f"Case {c}: f_trans slope mismatch={rel_slope_trans}"

        slope_left_peak = (amps[4] - amps[3]) / h
        slope_right_peak = (amps[5] - amps[4]) / h
        rel_slope_peak = (torch.abs(slope_right_peak - slope_left_peak) / torch.clamp(torch.abs(slope_left_peak), min=1e-25)).item()
        assert rel_slope_peak < 5e-3, f"Case {c}: f_peak slope mismatch={rel_slope_peak}"

