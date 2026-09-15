"""Analytic directional-derivative NQC solve for clean-room diagnostics.

Given a differentiable state vector field and the seven scalar functions
``[A,A*q1,A*q2,A*q3,p1,p2,omega]``, this evaluates Dg and D²g including the
Jacobian of the vector field, then solves the published 3x3/2x2 systems.
"""
import torch


def directional_derivatives(values_fn, state, vector_field):
    """Return values, first and second directional derivatives at one state."""
    x = state.detach().clone().requires_grad_(True)
    values = values_fn(x)
    v = vector_field(x)
    first = []
    second = []
    for i in range(values.shape[-1]):
        g = values[..., i]
        grad = (torch.autograd.grad(g, x, create_graph=True, retain_graph=True,
                                   allow_unused=True)[0] if g.requires_grad else None)
        if grad is None:
            grad = torch.zeros_like(x)
        dg = (grad * v).sum(-1)
        grad_dg = (torch.autograd.grad(dg, x, create_graph=False, retain_graph=True,
                                      allow_unused=True)[0] if dg.requires_grad else None)
        if grad_dg is None:
            grad_dg = torch.zeros_like(x)
        first.append(dg)
        second.append((grad_dg * v).sum(-1))
    return values.detach(), torch.stack(first, -1).detach(), torch.stack(second, -1).detach()


def solve_nqc(values_fn, state, vector_field, targets):
    """Solve amplitude and phase NQC coefficients from exact D/D² values."""
    value, first, second = directional_derivatives(values_fn, state, vector_field)
    amplitude_matrix = torch.stack((value[..., 1:4], first[..., 1:4], second[..., 1:4]), -2)
    amplitude_rhs = torch.stack((targets[0]-value[..., 0], targets[1]-first[..., 0],
                                 targets[2]-second[..., 0]), -1)
    phase_matrix = torch.stack((first[..., 4:6], second[..., 4:6]), -2)
    phase_rhs = torch.stack((value[..., 6]-targets[3], first[..., 6]-targets[4]), -1)
    return torch.linalg.solve(amplitude_matrix, amplitude_rhs), torch.linalg.solve(phase_matrix, phase_rhs), (value, first, second)
