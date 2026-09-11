"""Tests for IMRPhenomXAS phase reconstruction across 3-region C1 phasing engine.

Validates:
1. End-to-end C0 phase continuity and autograd C1 frequency derivative continuity at joins.
2. Phase reference (f_ref) calibration and exact time/orbital phase rotations (t_c, phi_c).
3. Batched vs scalar evaluation relative L2 error consistency across binary parameters.
4. Autograd gradcheck on normalized full complex waveform and backward gradient flow.
"""

from typing import Tuple
import torch

from torchwave.approximants.imrphenom_xas import (
    IMRPhenomXAS,
    XASParameterFits,
    component_masses_to_eta,
)
from torchwave.constants import MSUN_SEC, PI


def compute_phase_transition_frequencies_hz(
    m1: float, m2: float, s1: float, s2: float
) -> Tuple[float, float]:
    """Compute physical transition frequencies f_trans_phase and f_join_rd in Hz."""
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
    r_isco = 3.0 + z2 - torch.sign(chi_fit) * torch.sqrt(
        torch.clamp((3.0 - z1) * (3.0 + z1 + 2.0 * z2), min=1e-12)
    )
    f_isco = 1.0 / (PI * (r_isco**1.5 + chi_fit))

    f_T_phase = 0.6 * (0.5 * f_rd + f_isco)
    delta_R = 0.03 * (f_T_phase - f_meco)
    f_trans_phase = f_meco - delta_R
    f_join_rd = f_T_phase + 0.5 * delta_R

    f_trans_hz = (f_trans_phase / m_sec).item()
    f_join_hz = (f_join_rd / m_sec).item()
    return f_trans_hz, f_join_hz


def compute_normalized_phase_slope(
    model: IMRPhenomXAS,
    f_val: torch.Tensor,
    c: dict,
    f_ref_fixed: float = 20.0,
) -> float:
    """Compute phase slope dPsi/df via autograd on normalized complex phase u(f) = exp(-i*Psi).

    Slope is given by -imag(conj(u) * du/df) = (u_im * du_re - u_re * du_im),
    which avoids phase wrapping artifacts and first-order finite difference curvature error.
    """
    f_var = f_val.clone().detach().requires_grad_(True)
    _, _, p = model(
        f=f_var.unsqueeze(0),
        mass1=c["m1"],
        mass2=c["m2"],
        chi1=c["s1"],
        chi2=c["s2"],
        distance=100.0,
        f_ref=f_ref_fixed,
        return_phase=True,
    )
    u_re = torch.cos(p[0])
    u_im = -torch.sin(p[0])
    du_re = torch.autograd.grad(u_re, f_var, retain_graph=True)[0]
    du_im = torch.autograd.grad(u_im, f_var)[0]
    slope = u_im * du_re - u_re * du_im
    return slope.item()


def test_xas_phase_boundary_continuity():
    """Verify C0 phase continuity and autograd C1 frequency derivative continuity at both joins."""
    cases = [
        {"m1": 35.0, "m2": 30.0, "s1": 0.2, "s2": -0.3},
        {"m1": 30.0, "m2": 30.0, "s1": 0.0, "s2": 0.0},
        {"m1": 70.0, "m2": 10.0, "s1": -0.5, "s2": 0.2},
    ]
    model = IMRPhenomXAS()

    for c in cases:
        f_trans, f_join = compute_phase_transition_frequencies_hz(
            c["m1"], c["m2"], c["s1"], c["s2"]
        )

        for f_bound, bound_name in [(f_trans, "f_trans_phase"), (f_join, "f_join_rd")]:
            # 1. C0 phase jump across narrow 2*eps band (eps = 1e-4 Hz)
            eps = 1e-4
            f_eval = torch.tensor([f_bound - eps, f_bound + eps], dtype=torch.float64)
            _, _, phase = model(
                f=f_eval,
                mass1=c["m1"],
                mass2=c["m2"],
                chi1=c["s1"],
                chi2=c["s2"],
                distance=100.0,
                f_ref=20.0,
                return_phase=True,
            )
            phase_jump = torch.abs(phase[1] - phase[0]).item()
            assert (
                phase_jump < 1e-3
            ), f"Case {c}: {bound_name} phase jump {phase_jump} rad exceeds tolerance"

            # 2. C1 frequency derivative continuity: one-sided linear extrapolation of autograd slopes to boundary
            # Evaluating slopes at f_bound +/- h includes physical curvature Psi''(f_bound) * h.
            # Computing slopes at -h, -h/2 and +h/2, +h and linearly extrapolating to the boundary:
            #   slope_left  = 2 * slope(-h/2) - slope(-h)
            #   slope_right = 2 * slope(+h/2) - slope(+h)
            # cancels the first-order displacement error and compares the exact one-sided boundary limits.
            h = 1e-4  # Hz
            s_m1 = compute_normalized_phase_slope(
                model, torch.tensor(f_bound - h, dtype=torch.float64), c
            )
            s_m05 = compute_normalized_phase_slope(
                model, torch.tensor(f_bound - 0.5 * h, dtype=torch.float64), c
            )
            s_p05 = compute_normalized_phase_slope(
                model, torch.tensor(f_bound + 0.5 * h, dtype=torch.float64), c
            )
            s_p1 = compute_normalized_phase_slope(
                model, torch.tensor(f_bound + h, dtype=torch.float64), c
            )

            slope_left = 2.0 * s_m05 - s_m1
            slope_right = 2.0 * s_p05 - s_p1

            rel_slope_diff = abs(slope_right - slope_left) / max(abs(slope_left), 1e-12)
            assert (
                rel_slope_diff < 5e-5
            ), f"Case {c}: {bound_name} extrapolated slope mismatch {rel_slope_diff} exceeds 5e-5"


def test_xas_phase_reference_and_rotations():
    """Verify exact phase behavior under f_ref calibration, t_c shifts, and phi_c rotations."""
    model = IMRPhenomXAS()
    f_grid = torch.linspace(20.0, 300.0, 50, dtype=torch.float64)

    # 1. Reference frequency zero-phase check
    f_ref_val = 50.0
    _, _, p_ref = model(
        f=torch.tensor([f_ref_val], dtype=torch.float64),
        mass1=35.0,
        mass2=30.0,
        chi1=0.2,
        chi2=-0.3,
        f_ref=f_ref_val,
        phi_c=0.0,
        t_c=0.0,
        return_phase=True,
    )
    assert (
        torch.abs(p_ref[0] - 3.0 * PI / 4.0).item() < 1e-12 or torch.abs(p_ref[0]).item() < 1e-12
    ), f"Phase at f_ref={f_ref_val} is {p_ref[0].item()}, expected 3*pi/4 or 0.0"

    # 2. Time shift rotation: h(f; t_c) = h(f; 0) * exp(-i * 2*pi*f * t_c)
    hp_base, _ = model(
        f=f_grid,
        mass1=35.0,
        mass2=30.0,
        chi1=0.2,
        chi2=-0.3,
        t_c=0.0,
        phi_c=0.0,
    )
    delta_t = 0.005  # 5 ms
    hp_shifted, _ = model(
        f=f_grid,
        mass1=35.0,
        mass2=30.0,
        chi1=0.2,
        chi2=-0.3,
        t_c=delta_t,
        phi_c=0.0,
    )
    rot_t = torch.complex(
        torch.cos(-2.0 * PI * f_grid * delta_t),
        torch.sin(-2.0 * PI * f_grid * delta_t),
    )
    norm_base = torch.linalg.norm(hp_base)
    assert norm_base.item() > 0.0, "Base hp norm is non-positive"
    rel_diff_t = (torch.linalg.norm(hp_shifted - hp_base * rot_t) / norm_base).item()
    assert rel_diff_t < 1e-12, f"Time shift relative difference {rel_diff_t} exceeds 1e-12"

    # 3. Orbital phase rotation: total_phase contains -2*phi_c -> exp(i * 2*phi_c)
    delta_phi = 0.75
    hp_phi, _ = model(
        f=f_grid,
        mass1=35.0,
        mass2=30.0,
        chi1=0.2,
        chi2=-0.3,
        t_c=0.0,
        phi_c=delta_phi,
    )
    rot_phi = torch.complex(
        torch.cos(torch.tensor(2.0 * delta_phi, dtype=torch.float64)),
        torch.sin(torch.tensor(2.0 * delta_phi, dtype=torch.float64)),
    )
    rel_diff_phi = (torch.linalg.norm(hp_phi - hp_base * rot_phi) / norm_base).item()
    assert rel_diff_phi < 1e-12, f"Phase rotation relative diff {rel_diff_phi} exceeds 1e-12"


def test_xas_phase_scalar_vs_batched():
    """Verify batched phase and polarization evaluation matches scalar evaluation via relative L2."""
    model = IMRPhenomXAS()
    f_grid = torch.linspace(20.0, 400.0, 80, dtype=torch.float64)

    m1_b = torch.tensor([35.0, 30.0, 70.0], dtype=torch.float64)
    m2_b = torch.tensor([30.0, 30.0, 10.0], dtype=torch.float64)
    s1_b = torch.tensor([0.2, 0.0, -0.5], dtype=torch.float64)
    s2_b = torch.tensor([-0.3, 0.0, 0.2], dtype=torch.float64)

    hp_b, hc_b, p_b = model(
        f=f_grid,
        mass1=m1_b,
        mass2=m2_b,
        chi1=s1_b,
        chi2=s2_b,
        distance=500.0,
        return_phase=True,
    )

    for i in range(len(m1_b)):
        hp_s, hc_s, p_s = model(
            f=f_grid,
            mass1=m1_b[i],
            mass2=m2_b[i],
            chi1=s1_b[i],
            chi2=s2_b[i],
            distance=500.0,
            return_phase=True,
        )
        norm_hp_s = torch.linalg.norm(hp_s)
        norm_hc_s = torch.linalg.norm(hc_s)
        assert norm_hp_s.item() > 0.0, f"Scalar hp norm is not positive for batch index {i}"
        assert norm_hc_s.item() > 0.0, f"Scalar hc norm is not positive for batch index {i}"

        rel_err_hp = (torch.linalg.norm(hp_b[i] - hp_s) / norm_hp_s).item()
        rel_err_hc = (torch.linalg.norm(hc_b[i] - hc_s) / norm_hc_s).item()
        assert rel_err_hp < 1e-12, f"Batched hp relative L2 error {rel_err_hp} exceeds 1e-12"
        assert rel_err_hc < 1e-12, f"Batched hc relative L2 error {rel_err_hc} exceeds 1e-12"
        assert torch.allclose(p_b[i], p_s, rtol=1e-12, atol=1e-12)


def test_xas_phase_autograd():
    """Verify clean autograd backpropagation and gradcheck for full normalized waveform."""
    model = IMRPhenomXAS()

    # Fixed evaluation frequencies exercising all three regions:
    # 30.0 Hz (inspiral), 110.0 Hz (intermediate), 320.0 Hz (merger-ringdown)
    # for nominal parameters (35 Msun, 30 Msun, 0.2, -0.3).
    f_fixed = torch.tensor([30.0, 110.0, 320.0], dtype=torch.float64)

    m1_val = 35.0
    m2_val = 30.0
    c1_val = 0.2
    c2_val = -0.3

    # Fixed detached normalization scale from initial evaluation
    with torch.no_grad():
        hp_init, _ = model(
            f=f_fixed,
            mass1=m1_val,
            mass2=m2_val,
            chi1=c1_val,
            chi2=c2_val,
            distance=100.0,
            f_ref=30.0,
        )
        norm_scale = torch.abs(hp_init).clamp(min=1e-30)

    # 1. Full-waveform autograd gradcheck on real view of normalized complex waveform
    def eval_normalized_waveform(m1_in, m2_in, c1_in, c2_in):
        hp_eval, _ = model(
            f=f_fixed,
            mass1=m1_in,
            mass2=m2_in,
            chi1=c1_in,
            chi2=c2_in,
            distance=100.0,
            f_ref=30.0,
        )
        hp_norm = hp_eval / norm_scale
        return torch.view_as_real(hp_norm)

    test_inputs = (
        torch.tensor(m1_val, dtype=torch.float64, requires_grad=True),
        torch.tensor(m2_val, dtype=torch.float64, requires_grad=True),
        torch.tensor(c1_val, dtype=torch.float64, requires_grad=True),
        torch.tensor(c2_val, dtype=torch.float64, requires_grad=True),
    )
    assert torch.autograd.gradcheck(
        eval_normalized_waveform,
        test_inputs,
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )

    # 2. Backward smoke test checking finite non-zero gradients on phase
    m1_s = torch.tensor(m1_val, dtype=torch.float64, requires_grad=True)
    m2_s = torch.tensor(m2_val, dtype=torch.float64, requires_grad=True)
    c1_s = torch.tensor(c1_val, dtype=torch.float64, requires_grad=True)
    c2_s = torch.tensor(c2_val, dtype=torch.float64, requires_grad=True)

    _, _, phase = model(
        f=f_fixed,
        mass1=m1_s,
        mass2=m2_s,
        chi1=c1_s,
        chi2=c2_s,
        distance=100.0,
        f_ref=30.0,
        return_phase=True,
    )

    loss = torch.sum(phase**2)
    loss.backward()

    for name, param in [("m1", m1_s), ("m2", m2_s), ("chi1", c1_s), ("chi2", c2_s)]:
        assert param.grad is not None, f"Gradient for {name} is None"
        assert torch.isfinite(param.grad), f"Gradient for {name} is not finite: {param.grad}"
        assert param.grad.item() != 0.0, f"Gradient for {name} is zero"
