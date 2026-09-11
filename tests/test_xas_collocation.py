"""Regression and consistency tests for IMRPhenomXAS dimensionless collocation solve.

Validates:
1. Exact recovery of residual values (r2, r3, r4) at the three physical nodes
   (0.5 * f_meco, 0.75 * f_meco, 1.0 * f_meco) across normal and small f_meco values.
2. Batched vs scalar evaluation consistency for 1D batch inputs.
3. Autograd differentiability: non-zero, finite gradients with respect to f_meco and residuals.
4. Numerical stability in float32 without underflow or determinant collapse.
5. End-to-end full waveform generation on high-mass configurations where f_meco is small.
"""

import torch

from torchwave.approximants.imrphenom_xas import (
    IMRPhenomXAS,
    solve_inspiral_residual_collocation,
)
from torchwave.constants import PI


def test_xas_residual_node_recovery_normal_and_small_meco():
    """Verify that reconstructed polynomial reproduces r2, r3, r4 at all 3 nodes."""
    # Test a spectrum of f_meco values: normal (0.02-0.05) and small (0.001-0.008)
    f_meco_values = [0.045, 0.025, 0.015, 0.008, 0.004, 0.001]

    # Representative residual values from parameter fits
    r2_val = -0.015
    r3_val = -0.055
    r4_val = -0.160

    r2 = torch.tensor(r2_val, dtype=torch.float64)
    r3 = torch.tensor(r3_val, dtype=torch.float64)
    r4 = torch.tensor(r4_val, dtype=torch.float64)

    for fm in f_meco_values:
        f_meco = torch.tensor(fm, dtype=torch.float64)
        rho1, rho2, rho3 = solve_inspiral_residual_collocation(f_meco, r2, r3, r4)

        # Evaluate at node 1: 0.5 * f_meco
        x1 = PI * 0.5 * f_meco
        res1 = rho1 * (x1 ** (7.0 / 3.0)) + rho2 * (x1 ** (8.0 / 3.0)) + rho3 * (x1 ** 3.0)
        rel_err1 = torch.abs((res1 - r2) / r2).item()
        assert rel_err1 < 1e-12, f"Node 1 failed at f_meco={fm}: rel_err={rel_err1}"

        # Evaluate at node 2: 0.75 * f_meco
        x2 = PI * 0.75 * f_meco
        res2 = rho1 * (x2 ** (7.0 / 3.0)) + rho2 * (x2 ** (8.0 / 3.0)) + rho3 * (x2 ** 3.0)
        rel_err2 = torch.abs((res2 - r3) / r3).item()
        assert rel_err2 < 1e-12, f"Node 2 failed at f_meco={fm}: rel_err={rel_err2}"

        # Evaluate at node 3: 1.0 * f_meco
        x3 = PI * 1.0 * f_meco
        res3 = rho1 * (x3 ** (7.0 / 3.0)) + rho2 * (x3 ** (8.0 / 3.0)) + rho3 * (x3 ** 3.0)
        rel_err3 = torch.abs((res3 - r4) / r4).item()
        assert rel_err3 < 1e-12, f"Node 3 failed at f_meco={fm}: rel_err={rel_err3}"


def test_xas_collocation_batched():
    """Verify batched evaluation matches scalar evaluation elementwise."""
    f_meco_b = torch.tensor([0.035, 0.020, 0.007, 0.002], dtype=torch.float64)
    r2_b = torch.tensor([-0.012, -0.020, -0.015, -0.008], dtype=torch.float64)
    r3_b = torch.tensor([-0.045, -0.060, -0.052, -0.030], dtype=torch.float64)
    r4_b = torch.tensor([-0.140, -0.180, -0.155, -0.090], dtype=torch.float64)

    rho1_b, rho2_b, rho3_b = solve_inspiral_residual_collocation(f_meco_b, r2_b, r3_b, r4_b)

    for i in range(len(f_meco_b)):
        rho1_s, rho2_s, rho3_s = solve_inspiral_residual_collocation(
            f_meco_b[i], r2_b[i], r3_b[i], r4_b[i]
        )
        assert torch.allclose(rho1_b[i], rho1_s, rtol=1e-14, atol=1e-14)
        assert torch.allclose(rho2_b[i], rho2_s, rtol=1e-14, atol=1e-14)
        assert torch.allclose(rho3_b[i], rho3_s, rtol=1e-14, atol=1e-14)


def test_xas_collocation_autograd():
    """Verify clean autograd backpropagation and gradcheck for collocation solver."""
    f_endpoint = torch.tensor(0.025, dtype=torch.float64, requires_grad=True)
    r2 = torch.tensor(-0.015, dtype=torch.float64, requires_grad=True)
    r3 = torch.tensor(-0.055, dtype=torch.float64, requires_grad=True)
    r4 = torch.tensor(-0.160, dtype=torch.float64, requires_grad=True)

    # Fixed independent evaluation frequency (not scaled by f_endpoint)
    f_fixed = torch.tensor(0.015, dtype=torch.float64)
    x_fixed = PI * f_fixed

    def eval_fixed_residual(f_ep, r2_in, r3_in, r4_in):
        rho1, rho2, rho3 = solve_inspiral_residual_collocation(f_ep, r2_in, r3_in, r4_in)
        return rho1 * (x_fixed ** (7.0 / 3.0)) + rho2 * (x_fixed ** (8.0 / 3.0)) + rho3 * (x_fixed ** 3.0)

    # 1. Standard backward pass check
    val = eval_fixed_residual(f_endpoint, r2, r3, r4)
    loss = val ** 2
    loss.backward()

    for param_name, param in [("f_endpoint", f_endpoint), ("r2", r2), ("r3", r3), ("r4", r4)]:
        assert param.grad is not None, f"Gradient is None for {param_name}"
        assert torch.isfinite(param.grad), f"Gradient not finite for {param_name}"
        assert param.grad.item() != 0.0, f"Gradient is zero for {param_name}"

    # 2. PyTorch gradcheck against finite differences
    test_inputs = (
        torch.tensor(0.025, dtype=torch.float64, requires_grad=True),
        torch.tensor(-0.015, dtype=torch.float64, requires_grad=True),
        torch.tensor(-0.055, dtype=torch.float64, requires_grad=True),
        torch.tensor(-0.160, dtype=torch.float64, requires_grad=True),
    )
    assert torch.autograd.gradcheck(eval_fixed_residual, test_inputs, eps=1e-6, atol=1e-5)


def test_xas_collocation_float32():
    """Verify that dimensionless collocation works reliably in float32 without underflow."""
    f_meco_f32 = torch.tensor(0.005, dtype=torch.float32)
    r2_f32 = torch.tensor(-0.015, dtype=torch.float32)
    r3_f32 = torch.tensor(-0.055, dtype=torch.float32)
    r4_f32 = torch.tensor(-0.160, dtype=torch.float32)

    rho1, rho2, rho3 = solve_inspiral_residual_collocation(
        f_meco_f32, r2_f32, r3_f32, r4_f32
    )

    assert torch.isfinite(rho1)
    assert torch.isfinite(rho2)
    assert torch.isfinite(rho3)

    # Check node recovery in float32
    for node_factor, expected in [(0.5, r2_f32), (0.75, r3_f32), (1.0, r4_f32)]:
        x = float(PI) * node_factor * f_meco_f32
        res = rho1 * (x ** (7.0 / 3.0)) + rho2 * (x ** (8.0 / 3.0)) + rho3 * (x ** 3.0)
        rel_diff = torch.abs((res - expected) / expected).item()
        assert rel_diff < 1e-4, f"Float32 node recovery diff={rel_diff} at node {node_factor}"


def test_xas_full_waveform_small_meco():
    """Verify that full waveform generation succeeds without NaNs for high-mass systems."""
    f = torch.linspace(20.0, 500.0, 100, dtype=torch.float64)
    model = IMRPhenomXAS()

    # Heavy binary configuration: total mass = 150 Msun -> low physical frequencies in Hz
    # Note: dimensionless Mf_meco depends only on eta and spins, scale-invariant under total mass
    hp, hc = model(
        f=f,
        mass1=80.0,
        mass2=70.0,
        chi1=0.2,
        chi2=-0.3,
        distance=500.0,
        inclination=0.6,
        phi_c=0.4,
        f_ref=20.0,
    )

    assert torch.all(torch.isfinite(hp)), "Plus polarization contains non-finite values"
    assert torch.all(torch.isfinite(hc)), "Cross polarization contains non-finite values"
    assert torch.linalg.norm(hp) > 0.0, "Plus polarization norm is zero"
