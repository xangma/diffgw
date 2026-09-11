"""Focused tests for IMRPhenomXAS exact analytic amplitude frequency derivative.

Validates:
1. Exact analytic derivative of pn_amp_func against independent complex-step differentiation.
2. Parity across execution contexts (normal, torch.no_grad(), torch.inference_mode()).
3. Batched vs scalar evaluation consistency with scale-aware tolerances.
4. Non-zero, finite autograd parameter gradients w.r.t. (m1, m2, chi1, chi2) agreeing with finite differences.

Literature citations:
- Pratten et al. (2020), Phys. Rev. D 102, 064001 (arXiv:2001.11412v1) Eqs. (68)-(69), (72), (77), (95)
"""

import math
import pytest
import torch

from torchwave.approximants.imrphenom_xas import IMRPhenomXAS, compute_spin_variables, PI


def _eval_pn_amp_complex_step(eta: float, s1: float, s2: float, x: float, h: float = 1e-25) -> float:
    """Independent complex-step derivative of 3PN TaylorF2 amplitude w.r.t. x = pi * M * f."""
    eta_t = torch.tensor(eta, dtype=torch.complex128)
    s1_t = torch.tensor(s1, dtype=torch.complex128)
    s2_t = torch.tensor(s2, dtype=torch.complex128)
    d_t = torch.sqrt(1.0 - 4.0 * eta_t)
    x_c = torch.tensor(complex(x, h), dtype=torch.complex128)

    amp_c = IMRPhenomXAS.pn_amp_func(eta_t, s1_t, s2_t, x_c, d_t, derivative=False)
    return float(amp_c.imag.item() / h)


@pytest.mark.parametrize(
    "eta, s1, s2, mf",
    [
        (0.25, 0.0, 0.0, 0.018),
        (0.24, 0.3, -0.2, 0.015),
        (0.20, 0.6, 0.4, 0.020),
        (0.12, -0.4, 0.1, 0.022),
        (0.08, 0.8, -0.6, 0.025),
    ],
)
def test_pn_amp_exact_derivative_vs_complex_step(eta, s1, s2, mf):
    """Verify exact analytic derivative of pn_amp_func matches complex-step differentiation to < 1e-11."""
    x = math.pi * mf
    eta_t = torch.tensor(eta, dtype=torch.float64)
    s1_t = torch.tensor(s1, dtype=torch.float64)
    s2_t = torch.tensor(s2, dtype=torch.float64)
    d_t = torch.sqrt(1.0 - 4.0 * eta_t)
    x_t = torch.tensor(x, dtype=torch.float64)

    # Analytic derivative w.r.t. dimensionless Mf
    dpn_dmf_analytic = IMRPhenomXAS.pn_amp_func(
        eta_t, s1_t, s2_t, x_t, d_t, derivative=True
    ).item()

    # Complex-step derivative w.r.t. x, scaled by dx/d(Mf) = pi
    dpn_dx_cs = _eval_pn_amp_complex_step(eta, s1, s2, x, h=1e-25)
    dpn_dmf_cs = math.pi * dpn_dx_cs

    rel_err = abs(dpn_dmf_analytic - dpn_dmf_cs) / abs(dpn_dmf_cs)
    assert rel_err < 1e-11, (
        f"Analytic derivative {dpn_dmf_analytic} disagrees with complex step {dpn_dmf_cs} "
        f"(rel err: {rel_err:.2e})"
    )


def test_pn_amp_func_return_both():
    """Verify pn_amp_func with return_both=True returns identical values to separate calls."""
    eta_t = torch.tensor(0.222, dtype=torch.float64)
    s1_t = torch.tensor(0.35, dtype=torch.float64)
    s2_t = torch.tensor(-0.15, dtype=torch.float64)
    d_t = torch.sqrt(1.0 - 4.0 * eta_t)
    x_t = torch.tensor(math.pi * 0.019, dtype=torch.float64)

    amp_single = IMRPhenomXAS.pn_amp_func(eta_t, s1_t, s2_t, x_t, d_t, derivative=False)
    deriv_single = IMRPhenomXAS.pn_amp_func(eta_t, s1_t, s2_t, x_t, d_t, derivative=True)
    amp_both, deriv_both = IMRPhenomXAS.pn_amp_func(eta_t, s1_t, s2_t, x_t, d_t, return_both=True)

    assert torch.equal(amp_single, amp_both)
    assert torch.equal(deriv_single, deriv_both)


def test_xas_execution_contexts():
    """Verify identical waveforms across normal, torch.no_grad(), and torch.inference_mode()."""
    model = IMRPhenomXAS()
    f = torch.linspace(20.0, 400.0, 200, dtype=torch.float64)
    args = {
        "sample_frequencies": f,
        "mass1": 36.0,
        "mass2": 29.0,
        "chi1": 0.3,
        "chi2": -0.2,
        "distance": 500.0,
    }

    # Normal execution
    hp_norm, hc_norm = model(**args)

    # torch.no_grad()
    with torch.no_grad():
        hp_nograd, hc_nograd = model(**args)

    # torch.inference_mode()
    with torch.inference_mode():
        hp_inf, hc_inf = model(**args)

    assert torch.equal(hp_norm, hp_nograd), "no_grad differs from normal evaluation"
    assert torch.equal(hp_norm, hp_inf), "inference_mode differs from normal evaluation"
    assert torch.equal(hc_norm, hc_nograd), "no_grad cross differs from normal evaluation"
    assert torch.equal(hc_norm, hc_inf), "inference_mode cross differs from normal evaluation"
    assert torch.all(torch.isfinite(hp_norm))
    assert torch.all(torch.isfinite(hc_norm))


def test_xas_batched_vs_scalar_scale_aware():
    """Verify batched evaluation matches scalar evaluation with scale-aware tolerances."""
    model = IMRPhenomXAS()
    f = torch.linspace(20.0, 300.0, 150, dtype=torch.float64)

    m1_b = torch.tensor([30.0, 45.0, 55.0], dtype=torch.float64)
    m2_b = torch.tensor([25.0, 15.0, 10.0], dtype=torch.float64)
    chi1_b = torch.tensor([0.2, -0.4, 0.6], dtype=torch.float64)
    chi2_b = torch.tensor([-0.1, 0.3, -0.5], dtype=torch.float64)

    hp_b, hc_b = model(
        sample_frequencies=f,
        mass1=m1_b,
        mass2=m2_b,
        chi1=chi1_b,
        chi2=chi2_b,
    )

    assert hp_b.shape == (3, 150)
    assert hc_b.shape == (3, 150)

    for i in range(3):
        hp_s, hc_s = model(
            sample_frequencies=f,
            mass1=float(m1_b[i]),
            mass2=float(m2_b[i]),
            chi1=float(chi1_b[i]),
            chi2=float(chi2_b[i]),
        )
        scale = torch.max(torch.abs(hp_s)).item()
        assert scale > 0.0, "Physical strain scale must be positive"
        assert torch.allclose(hp_b[i], hp_s, rtol=1e-12, atol=1e-12 * scale), (
            f"Batched row {i} does not match scalar evaluation"
        )
        assert torch.allclose(hc_b[i], hc_s, rtol=1e-12, atol=1e-12 * scale)


def test_xas_autograd_parameter_gradients():
    """Verify autograd parameter gradients are finite, non-zero, and agree with finite differences."""
    model = IMRPhenomXAS()
    f = torch.linspace(30.0, 250.0, 40, dtype=torch.float64)

    m1 = torch.tensor(35.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(25.0, dtype=torch.float64, requires_grad=True)
    chi1 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    chi2 = torch.tensor(-0.1, dtype=torch.float64, requires_grad=True)

    hp, _ = model(sample_frequencies=f, mass1=m1, mass2=m2, chi1=chi1, chi2=chi2)
    loss = torch.sum(hp.real)
    loss.backward()

    base_args = {"sample_frequencies": f, "mass1": 35.0, "mass2": 25.0, "chi1": 0.2, "chi2": -0.1}
    param_checks = {
        "m1": (m1, "mass1", 35.0, 1e-4),
        "m2": (m2, "mass2", 25.0, 1e-4),
        "chi1": (chi1, "chi1", 0.2, 1e-4),
        "chi2": (chi2, "chi2", -0.1, 1e-4),
    }

    for name, (param, kw_key, val, eps) in param_checks.items():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert torch.isfinite(param.grad), f"Gradient for {name} is non-finite"
        auto_val = param.grad.item()
        assert abs(auto_val) > 0.0, f"Gradient for {name} is zero"

        # Central difference at eps
        kw_p = dict(base_args)
        kw_m = dict(base_args)
        kw_p[kw_key] = val + eps
        kw_m[kw_key] = val - eps
        hp_p, _ = model(**kw_p)
        hp_m, _ = model(**kw_m)
        num_grad = ((torch.sum(hp_p.real) - torch.sum(hp_m.real)) / (2.0 * eps)).item()

        rel_diff = abs(auto_val - num_grad) / abs(num_grad)
        assert rel_diff < 0.02, f"Gradient mismatch for {name}: auto={auto_val}, num={num_grad} (rel {rel_diff:.2e})"
