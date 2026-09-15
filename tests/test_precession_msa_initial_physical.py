"""Physical clean-room regressions for the MSA spin initializer.

The checks use the published spin-precession rates and the independent
Jacobi representation; they do not call a waveform implementation.
"""
import math
import torch

from torchwave.approximants.precession_msa_initial_clean import (
    initial_invariants, cubic_coefficients, ordered_roots, initial_phase,
    spin_oscillation, jacobi_sn, elliptic_K_agm,
)

MTSUN_SEC = 4.925490947641267e-6


def _physical_case(m1, m2, chi1, chi2):
    Msec = (m1 + m2) * MTSUN_SEC
    eta0 = m1 * m2 / (m1 + m2) ** 2
    v = torch.tensor(math.pi * Msec * 20., dtype=torch.float64).pow(1 / 3)
    L0 = torch.tensor(eta0, dtype=torch.float64) / v
    Lvec = torch.tensor([0., 0., L0], dtype=torch.float64)
    s1 = torch.tensor(chi1, dtype=torch.float64) * (m1 / (m1 + m2)) ** 2
    s2 = torch.tensor(chi2, dtype=torch.float64) * (m2 / (m1 + m2)) ** 2
    inv = initial_invariants(m1, m2, chi1, chi2, Lvec)
    q, eta, delta, S1, S2, L, J, xi = inv
    B, C, D = cubic_coefficients(q, eta, delta, S1 * S1, S2 * S2, L, J, xi)
    roots = ordered_roots(B, C, D)
    Ssq = (s1 + s2).dot(s1 + s2)
    triple = Lvec.dot(torch.linalg.cross(s1, s2))
    psi = initial_phase(Ssq, roots[2], roots[1], roots[0], triple)
    return v, q, eta, xi, L, J, Lvec, s1, s2, Ssq, roots, psi


def test_physical_cubic_brackets_and_ode_rate():
    cases = ((35., 25., (.2, .1, .2), (-.1, .15, -.1)),
             (45., 15., (.45, -.2, .3), (.1, .2, -.25)))
    for case in cases:
        v, q, eta, xi, L, J, Lvec, s1, s2, Ssq, roots, psi0 = _physical_case(*case)
        assert roots[0] < 0 and roots[1] >= 0 and roots[2] >= roots[1]
        assert roots[1] <= Ssq <= roots[2]
        # Reconstruct S^2 from the selected Jacobi branch.
        psi = psi0.detach().clone().requires_grad_()
        got_s2 = spin_oscillation(psi, roots[2], roots[1], roots[0])
        assert torch.allclose(got_s2, Ssq, atol=1e-13, rtol=1e-13)
        d_s2_dpsi = torch.autograd.grad(got_s2, psi)[0]
        dpsi_dt = .75 / torch.sqrt(eta) * (1 - xi * v) * v**6 * torch.sqrt(roots[2] - roots[0])
        jacobi_rate = d_s2_dpsi * dpsi_dt
        om1 = .5 * v**6 * ((4 + 3*q - 3*q/(1+q)*xi/L) * Lvec + s2)
        om2 = .5 * v**6 * ((4 + 3/q - 3/(1+q)*xi/L) * Lvec + s1)
        ode_rate = 2 * (s1 + s2).dot(torch.linalg.cross(om1, s1) + torch.linalg.cross(om2, s2))
        assert torch.allclose(jacobi_rate, ode_rate, atol=1e-17, rtol=1e-10)


def test_jacobi_period_boundary_derivatives():
    for mval in (.2, .9):
        m = torch.tensor(mval, dtype=torch.float64, requires_grad=True)
        K = elliptic_K_agm(m)
        u = (2 * K).detach().clone().requires_grad_()
        sn = jacobi_sn(u, m)
        du, dm = torch.autograd.grad(sn, (u, m))
        assert abs(float(sn.detach())) < 2e-10
        assert torch.allclose(du, torch.tensor(-1., dtype=torch.float64), atol=2e-9)
        assert torch.isfinite(dm)
