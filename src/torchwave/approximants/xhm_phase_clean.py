# SPDX-License-Identifier: MIT
"""Paper-era IMRPhenomXHM phase primitives.

This module contains only algebra from García-Quirós et al.,
arXiv:2001.10914v2, Sec. VI.2.2, Eqs. (46)--(50).  It deliberately accepts
the four spheroidal derivative targets as caller data; no fit coefficients or
release-specific calibration values are bundled here.
"""
from __future__ import annotations

from typing import Tuple

import torch


__all__ = [
    "xhm_32_spheroidal_collocation_frequencies",
    "xhm_32_spheroidal_phase_coefficients",
    "xhm_32_spheroidal_phase_derivative",
    "xhm_32_spheroidal_phase",
]


def _broadcast(*values: object) -> Tuple[torch.Tensor, ...]:
    values = tuple(torch.as_tensor(v, dtype=torch.float64) for v in values)
    return torch.broadcast_tensors(*values)


def _validate_positive(*values: torch.Tensor) -> None:
    if any(bool((~torch.isfinite(v)).any()) for v in values):
        raise ValueError("frequencies and damping times must be finite")
    if any(bool((v <= 0).any()) for v in values):
        raise ValueError("frequencies and damping times must be positive")


def _validate_finite(*values: torch.Tensor) -> None:
    if any(bool((~torch.isfinite(v)).any()) for v in values):
        raise ValueError("derivative targets and coefficients must be finite")


def xhm_32_spheroidal_collocation_frequencies(
    f_ring22: object, f_ring32: object, f_damp32: object
) -> torch.Tensor:
    """Return the four Eq. (48) collocation frequencies, in order."""

    f_ring22, f_ring32, f_damp32 = _broadcast(f_ring22, f_ring32, f_damp32)
    _validate_positive(f_ring22, f_ring32, f_damp32)
    return torch.stack(
        (f_ring22, f_ring32 - 1.5 * f_damp32,
         f_ring32 - 0.5 * f_damp32, f_ring32 + 0.5 * f_damp32), dim=-1
    )


def xhm_32_spheroidal_phase_coefficients(
    f_ring22: object,
    f_ring32: object,
    f_damp32: object,
    derivative_targets: object,
) -> torch.Tensor:
    """Solve the four-condition spheroidal ringdown phase system.

    The returned last-axis coefficients are ``(alpha2, alpha4, alpha_lambda,
    dphi_rd)`` for Eq. (46).  ``derivative_targets[..., i]`` is the fitted
    value of ``d phi_32,S / df`` at collocation point ``i``.  Four conditions
    are required by the published ansatz; no reduced three-condition system
    is selected implicitly.
    """

    f_ring22, f_ring32, f_damp32 = _broadcast(f_ring22, f_ring32, f_damp32)
    targets = torch.as_tensor(derivative_targets, dtype=torch.float64)
    if targets.shape[-1:] != (4,):
        raise ValueError("derivative_targets must have four values on its last axis")
    _validate_finite(targets)
    f_ring22, f_ring32, f_damp32, _ = torch.broadcast_tensors(
        f_ring22, f_ring32, f_damp32, targets[..., 0]
    )
    targets = targets.expand(f_ring22.shape + (4,))
    _validate_positive(f_ring22, f_ring32, f_damp32)
    points = xhm_32_spheroidal_collocation_frequencies(f_ring22, f_ring32, f_damp32)
    if bool((points[..., :, None] == points[..., None, :]).sum(dim=(-1, -2)).gt(4).any()):
        raise ValueError("spheroidal collocation frequencies must be distinct")
    delta = points - f_ring32[..., None]
    lorentz = f_damp32[..., None] / (f_damp32[..., None] ** 2 + delta ** 2)
    matrix = torch.stack(
        (1.0 / points ** 2, 1.0 / points ** 4, lorentz,
         torch.ones_like(points)), dim=-1
    )
    if bool((points <= 0).any()):
        raise ValueError("all spheroidal collocation frequencies must be positive")
    return torch.linalg.solve(matrix, targets)


def xhm_32_spheroidal_phase_derivative(
    frequency: object, f_ring32: object, f_damp32: object,
    coefficients: object,
) -> torch.Tensor:
    """Evaluate Eq. (46) using ``(alpha2, alpha4, alpha_lambda, dphi_rd)``."""

    f, ring, damp = _broadcast(frequency, f_ring32, f_damp32)
    coeff = torch.as_tensor(coefficients, dtype=torch.float64)
    if coeff.shape[-1:] != (4,):
        raise ValueError("coefficients must have four values on their last axis")
    _validate_finite(coeff)
    f, ring, damp, _ = torch.broadcast_tensors(f, ring, damp, coeff[..., 0])
    coeff = coeff.expand(f.shape + (4,))
    _validate_positive(f, ring, damp)
    lorentz = damp / (damp ** 2 + (f - ring) ** 2)
    return coeff[..., 0] / f ** 2 + coeff[..., 1] / f ** 4 + coeff[..., 2] * lorentz + coeff[..., 3]


def xhm_32_spheroidal_phase(
    frequency: object, f_ring32: object, f_damp32: object,
    coefficients: object, phase_offset: object = 0.0,
) -> torch.Tensor:
    """Evaluate the integrated Eq. (47), with an optional additive offset."""

    f, ring, damp = _broadcast(frequency, f_ring32, f_damp32)
    coeff = torch.as_tensor(coefficients, dtype=torch.float64)
    offset = torch.as_tensor(phase_offset, dtype=torch.float64)
    if coeff.shape[-1:] != (4,):
        raise ValueError("coefficients must have four values on their last axis")
    _validate_finite(coeff, offset)
    f, ring, damp, offset = torch.broadcast_tensors(f, ring, damp, offset)
    f, ring, damp, _, offset = torch.broadcast_tensors(f, ring, damp, coeff[..., 0], offset)
    coeff = coeff.expand(f.shape + (4,))
    _validate_positive(f, ring, damp)
    return (-coeff[..., 0] / f - coeff[..., 1] / (3.0 * f ** 3)
            + coeff[..., 2] * torch.atan((f - ring) / damp)
            + coeff[..., 3] * f + offset)
