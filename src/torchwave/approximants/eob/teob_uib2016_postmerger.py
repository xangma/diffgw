"""Explicit UIB2016-to-TEOBResumS postmerger adapter.

The nonspinning UIB2016 remnant fit supplies the final-mass fraction and
dimensionless final spin required by the standalone TEOBResumS postmerger
22 ansatz.  This module joins those two published components only; it does
not infer a trajectory merger time, phase anchor, or NQC extraction point.

The source equations are Jiménez--Forteza et al., arXiv:1611.00332v2,
Eqs. (7) and (21), and Nagar et al., arXiv:1806.01772v2, Appendix ``NR-
informed description of merger and postmerger``.

Component masses may use any common positive unit because the UIB fit depends
only on their symmetric mass ratio.  ``time`` and ``merger_time`` are in
units of the initial total mass, as required by :func:`postmerger_22`.
"""

from __future__ import annotations

import torch

from .teob_postmerger import postmerger_22
from .teob_uib2016_remnant import uib2016_nonspinning_remnant


def uib2016_nonspinning_postmerger_22(
    time: torch.Tensor,
    mass1: torch.Tensor | float,
    mass2: torch.Tensor | float,
    phase0: torch.Tensor | float = 0.0,
    merger_time: torch.Tensor | float = 0.0,
    merger_frequency_value: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Evaluate the UIB2016-anchored nonspinning TEOBResumS 22 mode.

    ``mass1`` and ``mass2`` are positive component masses in a common unit.
    The UIB2016 nonspinning fit determines ``(M_f/M, chi_f)``; those values
    are then passed unchanged to :func:`postmerger_22`.  The caller must
    supply the merger time and phase convention when they are not zero.  An
    optional merger-frequency value overrides the published mass-ratio fit
    and is useful when a trajectory supplies its own merger frequency.
    """
    final_mass, final_spin = uib2016_nonspinning_remnant(mass1, mass2)
    # The remnant routine intentionally keeps eta internal; derive the same
    # symmetric ratio here for the postmerger merger-frequency fit.  Scaling
    # first avoids overflow for dimensional component masses.
    m1, m2 = torch.broadcast_tensors(
        torch.as_tensor(mass1, device=final_mass.device, dtype=final_mass.dtype),
        torch.as_tensor(mass2, device=final_mass.device, dtype=final_mass.dtype),
    )
    scale = torch.maximum(m1, m2)
    x1, x2 = m1 / scale, m2 / scale
    nu = (x1 * x2) / (x1 + x2).square()
    # Floating-point roundoff can place a nearly equal-mass result a few ulps
    # above the mathematical upper bound, which the postmerger validator must
    # otherwise reject.
    nu = torch.minimum(nu, nu.new_tensor(0.25))
    return postmerger_22(
        time,
        nu,
        final_mass,
        final_spin,
        phase0=phase0,
        merger_time=merger_time,
        merger_frequency_value=merger_frequency_value,
    )


__all__ = ["uib2016_nonspinning_postmerger_22"]
