"""Torch float64 conditioning diagnostic for the generic MSA spin cubic.

The raw inverse-cosine argument is returned before any endpoint handling.
``compatibility_roots`` then records the roots produced by v223's current
float64 endpoint exclusion.  The two quantities are kept separate so that
prescription cannot conceal loss of the published cubic branch.  This module
does not participate in waveform dispatch.
"""

from __future__ import annotations

import math

import torch


MTSUN_SI = 4.925490947641267e-6


def generic_cubic_boundary_diagnostic(mass1, mass2, spin1, spin2, f_ref):
    """Return raw cubic conditioning and compatibility roots in float64."""
    chi1 = torch.as_tensor(spin1, dtype=torch.float64)
    chi2 = torch.as_tensor(spin2, dtype=torch.float64, device=chi1.device)
    if chi1.shape[-1] != 3 or chi2.shape[-1] != 3:
        raise ValueError("spin vectors must end in dimension 3")
    m1, m2, fr = torch.broadcast_tensors(*(
        torch.as_tensor(value, dtype=torch.float64, device=chi1.device)
        for value in (mass1, mass2, f_ref)
    ))
    swap = m2 > m1
    cm1, cm2 = torch.where(swap, m2, m1), torch.where(swap, m1, m2)
    cs1 = torch.where(swap[..., None], chi2, chi1)
    cs2 = torch.where(swap[..., None], chi1, chi2)
    total = cm1 + cm2
    n1, n2 = cm1 / total, cm2 / total
    eta, q, delta = n1 * n2, n2 / n1, n1 - n2
    s1, s2 = n1[..., None] ** 2 * cs1, n2[..., None] ** 2 * cs2
    s1sq, s2sq = torch.sum(s1**2, -1), torch.sum(s2**2, -1)
    spin = s1 + s2
    spin_squared = torch.sum(spin**2, -1)
    xi = (1 + q) * s1[..., 2] + (1 + 1 / q) * s2[..., 2]
    velocity = (math.pi * total * MTSUN_SI * fr).pow(1 / 3)
    orbital = eta / velocity
    angular_momentum = spin + torch.stack(
        (torch.zeros_like(orbital), torch.zeros_like(orbital), orbital), -1
    )
    j = torch.linalg.vector_norm(angular_momentum, dim=-1)
    j2, l2 = j**2, orbital**2
    b = ((l2 + s1sq) * q + 2 * orbital * xi - 2 * j2 - s1sq - s2sq
         + (l2 + s2sq) / q)
    c = ((j2 - l2) ** 2 - 2 * orbital * xi * (j2 - l2)
         - 2 * (1 - q) / q * (s1sq - q * s2sq) * l2
         + 4 * eta * l2 * xi**2
         - 2 * delta * (s1sq - s2sq) * xi * orbital
         + 2 * (1 - q) / q * (q * s1sq - s2sq) * j2)
    d = ((1 - q) / q * (s2sq - q * s1sq) * (j2 - l2) ** 2
         + delta**2 / eta * (s1sq - s2sq) ** 2 * l2
         + 2 * delta * orbital * xi * (s1sq - s2sq) * (j2 - l2))
    p = c - b**2 / 3
    depressed_q = 2 * b**3 / 27 - b * c / 3 + d
    raw_argument = 3 * depressed_q / (2 * p) * torch.sqrt(-3 / p)
    argument_used = raw_argument.clamp(-1.0 + 1e-15, 1.0 - 1e-15)
    theta = torch.acos(argument_used) / 3
    k = torch.arange(3, dtype=torch.float64, device=chi1.device)
    candidates = (-b[..., None] / 3 + 2 * torch.sqrt(-p[..., None] / 3)
                  * torch.cos(theta[..., None] - 2 * torch.pi * k / 3))
    roots = torch.sort(candidates, dim=-1).values
    gap = roots[..., 2] - roots[..., 1]
    fraction = (roots[..., 2] - spin_squared) / gap
    return {
        "coefficients": torch.stack((b, c, d), -1),
        "depressed_coefficients": torch.stack((p, depressed_q), -1),
        "acos_argument": raw_argument,
        "acos_argument_used": argument_used,
        "compatibility_roots": roots,
        "compatibility_root_gap": gap,
        "total_spin_squared": spin_squared,
        "phase_fraction_unclipped": fraction,
    }


__all__ = ["generic_cubic_boundary_diagnostic"]
