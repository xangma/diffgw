"""Published orbital-angular-momentum candidates for MSA geometry.

This module is a deliberately small diagnostic companion to the clean MSA
composition.  It transcribes the aligned-spin orbital-angular-momentum
terms printed in Pratten et al., arXiv:2004.06503v2, Appendix G.  The paper
lists a 4PN orbital series, spin-orbit terms through 3.5PN, and the optional
leading-PN-at-all-orders-in-spin (``LO-S^infty``) terms.  Those equations do
not identify a separate v223 reference-frame prescription; the functions
below therefore expose named *candidates* and do not route any production
waveform.

All angular momenta are normalized by the source total mass squared.  Masses
are in solar masses and frequency is in Hz.  Inputs are broadcast to one
shape and remain differentiable tensors.  The transverse spin components do
not enter these aligned-spin ``L`` equations, as specified by the paper.
"""
from __future__ import annotations

import math

import torch

from torchwave.constants import MSUN_SEC
from .precession_beta_clean import orbital_angular_momentum


_DTYPE = torch.float64


def _device_of(values):
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.device
    return None


def _broadcast_f64(*values):
    device = _device_of(values)
    tensors = [torch.as_tensor(value, dtype=_DTYPE, device=device) for value in values]
    return torch.broadcast_tensors(*tensors)


def _x_parameter(mass1, mass2, frequency):
    """Return ``x=(pi M f)^(2/3)`` in geometric units."""
    return (math.pi * (mass1 + mass2) * MSUN_SEC * frequency).pow(2.0 / 3.0)


def _validate_physical_inputs(mass1, mass2, frequency):
    for name, value in (
        ("mass1", mass1), ("mass2", mass2), ("frequency", frequency)
    ):
        if bool(torch.any(~torch.isfinite(value))):
            raise ValueError(f"{name} must be finite")
    if bool(torch.any((mass1 <= 0.0) | (mass2 <= 0.0))):
        raise ValueError("masses must be positive")
    if bool(torch.any(frequency <= 0.0)):
        raise ValueError("frequency must be positive")


def leading_spin_infinity_coefficients(eta, delta, chi1_l, chi2_l):
    """Return the paper's ``(L_2^LO-Sinf, L_3.5^LO-Sinf)`` coefficients.

    These are the explicit aligned-spin expressions in Appendix G.  They are
    coefficients inside the common bracket

    ``L/M^2 = eta/sqrt(x) * (... + L2*x**2 + L3.5*x**3.5)``.

    ``eta`` is the symmetric mass ratio, ``delta=(m1-m2)/(m1+m2)``, and
    ``chi*_l`` are the dimensionless spin projections on the orbital axis.
    """
    eta, delta, chi1_l, chi2_l = _broadcast_f64(eta, delta, chi1_l, chi2_l)
    l2 = (
        (0.5 + 0.5 * delta - eta) * chi1_l.square()
        + 2.0 * eta * chi1_l * chi2_l
        + (0.5 - 0.5 * delta - eta) * chi2_l.square()
    )
    l35 = (
        chi1_l.pow(3) * (0.75 * delta * eta - 1.5 * eta.square() + 0.75 * eta)
        + chi1_l.square() * chi2_l * (0.75 * delta * eta + 1.5 * eta.square() + 0.75 * eta)
        + chi1_l * chi2_l.square() * (-0.75 * delta * eta + 1.5 * eta.square() + 0.75 * eta)
        + chi2_l.pow(3) * (-0.75 * delta * eta - 1.5 * eta.square() + 0.75 * eta)
    )
    return l2, l35


def orbital_angular_momentum_l4pn_losinfty(
    mass1, mass2, frequency, chi1_l, chi2_l
):
    """Return the explicit 4PN + ``LO-S^infty`` orbital-``L`` candidate.

    The 4PN and spin-orbit pieces are evaluated by the separate clean
    Eq. (26)/Appendix-G implementation.  The two optional spin terms are
    then added exactly as printed in Appendix G.  This is a diagnostic
    candidate only: the public paper does not say that v223 uses this
    expression for its source-frame metadata.
    """
    mass1, mass2, frequency, chi1_l, chi2_l = _broadcast_f64(
        mass1, mass2, frequency, chi1_l, chi2_l
    )
    _validate_physical_inputs(mass1, mass2, frequency)
    base = orbital_angular_momentum(
        mass1, mass2, frequency, chi1_l, chi2_l,
        pn_version="full", compat_102=False,
    )
    total = mass1 + mass2
    eta = mass1 * mass2 / total.square()
    delta = (mass1 - mass2) / total
    x = _x_parameter(mass1, mass2, frequency)
    l2, l35 = leading_spin_infinity_coefficients(eta, delta, chi1_l, chi2_l)
    return base + eta / torch.sqrt(x) * (l2 * x.square() + l35 * x.pow(3.5))


def _orbital_angular_momentum_nonconserved_3pn(
    mass1, mass2, frequency, chi1_l, chi2_l, *, eta_spin_coefficient
):
    """Return circular 3PN ``L/M**2`` in the older spin variables.

    Blanchet, Buonanno and Faye, arXiv:gr-qc/0605140v4, Eq. (6.9), prints
    the circular orbital angular momentum in spin variables whose individual
    norms are not conserved.  For aligned components the transverse terms
    vanish.  Its non-spinning terms through ``x**2`` and spin-orbit terms
    through ``x**(5/2)`` are combined here with the published 3PN non-spinning
    coefficient ``L_3`` from Pratten et al., arXiv:2004.06503v2, Appendix G,
    so this candidate can be compared on the same 3PN basis as the public
    version table::

        eta/sqrt(x) * (1 + (3/2+eta/6)*x
          + (-23*Sl/6 - 3*delta*Sigma/2)*x**(3/2)
          + (27/8 - 19*eta/8 + eta**2/24)*x**2 + L_3*x**3
          + ((-77/8 + C*eta)*Sl
             + delta*(-33/8 + 7*eta/4)*Sigma)*x**(5/2))

    Here ``C`` is the selected 2.5PN spin-orbit coefficient, ``Sl =
    nu1**2*chi1_l + nu2**2*chi2_l`` and
    ``Sigma = nu2*chi2_l - nu1*chi1_l`` are the dimensionless forms of the
    paper's ``S_l/(G m**2)`` and ``Sigma_l/(G m**2)``.  This is a published
    comparison candidate only, kept separate from the clean conserved-spin
    source-frame prescription and never selected by a waveform default.  Its
    two public callers below intentionally retain distinct 2.5PN coefficients
    for the MSA opening angle and source-frame reference geometry.
    """
    mass1, mass2, frequency, chi1_l, chi2_l = _broadcast_f64(
        mass1, mass2, frequency, chi1_l, chi2_l
    )
    _validate_physical_inputs(mass1, mass2, frequency)
    total = mass1 + mass2
    eta = mass1 * mass2 / total.square()
    delta = (mass1 - mass2) / total
    nu1, nu2 = mass1 / total, mass2 / total
    x = _x_parameter(mass1, mass2, frequency)
    spin_l = nu1.square() * chi1_l + nu2.square() * chi2_l
    sigma_l = nu2 * chi2_l - nu1 * chi1_l
    l3 = (
        7.0 * eta.pow(3) / 1296.0
        + 31.0 * eta.square() / 24.0
        + (41.0 * math.pi**2 / 24.0 - 6889.0 / 144.0) * eta
        + 135.0 / 16.0
    )
    terms = (
        1.0
        + (1.5 + eta / 6.0) * x
        + (-23.0 / 6.0 * spin_l - 1.5 * delta * sigma_l) * x.pow(1.5)
        + (27.0 / 8.0 - 19.0 * eta / 8.0 + eta.square() / 24.0) * x.square()
        + l3 * x.pow(3.0)
        + (
            (-77.0 / 8.0 + eta_spin_coefficient * eta) * spin_l
            + delta * (-33.0 / 8.0 + 7.0 * eta / 4.0) * sigma_l
        ) * x.pow(2.5)
    )
    return eta / torch.sqrt(x) * terms


def orbital_angular_momentum_nonconserved_3pn(
    mass1, mass2, frequency, chi1_l, chi2_l
):
    """Return the MSA opening-beta ``L/M^2`` candidate (259/72).

    This is the exact nonconserved-spin expression used for the explicit MSA
    opening-angle branch.  The source-frame helper below intentionally uses
    its own 259/75 coefficient.
    """
    return _orbital_angular_momentum_nonconserved_3pn(
        mass1, mass2, frequency, chi1_l, chi2_l,
        eta_spin_coefficient=259.0 / 72.0,
    )


def orbital_angular_momentum_msa_opening_3pn(
    mass1, mass2, frequency, chi1_l, chi2_l
):
    """Named alias for the v223/MSA opening-beta convention."""
    return orbital_angular_momentum_nonconserved_3pn(
        mass1, mass2, frequency, chi1_l, chi2_l
    )


def orbital_angular_momentum_reference_3pn(
    mass1, mass2, frequency, chi1_l, chi2_l
):
    """Return the source-frame 3PN ``L/M^2`` candidate (259/75).

    The source-frame convention differs from the MSA opening expression only
    in the displayed 2.5PN ``eta*S_l`` coefficient.  In full ``L/M^2``
    units, the difference is

    ``-(259/1800) * eta**2 * (S_l/M**2) * x**2``.

    It is exposed as a separate function so a source-frame geometry caller
    cannot accidentally reuse the opening-beta value.
    """
    return _orbital_angular_momentum_nonconserved_3pn(
        mass1, mass2, frequency, chi1_l, chi2_l,
        eta_spin_coefficient=259.0 / 75.0,
    )


orbital_angular_momentum_reference_nonconserved_3pn = orbital_angular_momentum_reference_3pn
orbital_angular_momentum_v223 = orbital_angular_momentum_msa_opening_3pn


def orbital_angular_momentum_candidates(
    mass1, mass2, frequency, chi1_l, chi2_l
):
    """Return named published/clean reference candidates for a diagnostic.

    ``three_pn_spin`` is the printed 3PN spinning prescription used for the
    clean source-frame helper; ``three_pn_spin_nonconserved`` is the older
    spin-variable series from gr-qc/0605140 Eq. (6.9); ``four_pn_spin`` is
    the full displayed series; ``four_pn_spin_losinfty`` adds the Appendix-G
    all-orders-in-spin terms; and ``native_msa_newtonian`` is the Newtonian
    ``eta/sqrt(x)`` magnitude used by the clean MSA dynamics.  The names
    intentionally describe the equations rather than claiming public
    ``PrecVersion`` support.
    """
    mass1, mass2, frequency, chi1_l, chi2_l = _broadcast_f64(
        mass1, mass2, frequency, chi1_l, chi2_l
    )
    _validate_physical_inputs(mass1, mass2, frequency)
    three_pn = orbital_angular_momentum(
        mass1, mass2, frequency, chi1_l, chi2_l,
        pn_version=102, compat_102=False,
    )
    three_pn_nonconserved = orbital_angular_momentum_nonconserved_3pn(
        mass1, mass2, frequency, chi1_l, chi2_l,
    )
    reference_source_3pn = orbital_angular_momentum_reference_3pn(
        mass1, mass2, frequency, chi1_l, chi2_l,
    )
    four_pn = orbital_angular_momentum(
        mass1, mass2, frequency, chi1_l, chi2_l,
        pn_version="full", compat_102=False,
    )
    total = mass1 + mass2
    x = _x_parameter(mass1, mass2, frequency)
    eta = mass1 * mass2 / total.square()
    return {
        "three_pn_spin": three_pn,
        "three_pn_spin_nonconserved": three_pn_nonconserved,
        "msa_opening_3pn": three_pn_nonconserved,
        "reference_source_3pn": reference_source_3pn,
        "reference_source_nonconserved_3pn": reference_source_3pn,
        "four_pn_spin": four_pn,
        "four_pn_spin_losinfty": orbital_angular_momentum_l4pn_losinfty(
            mass1, mass2, frequency, chi1_l, chi2_l
        ),
        "native_msa_newtonian": eta / torch.sqrt(x),
    }


__all__ = [
    "leading_spin_infinity_coefficients",
    "orbital_angular_momentum_l4pn_losinfty",
    "orbital_angular_momentum_nonconserved_3pn",
    "orbital_angular_momentum_msa_opening_3pn",
    "orbital_angular_momentum_v223",
    "orbital_angular_momentum_reference_3pn",
    "orbital_angular_momentum_reference_nonconserved_3pn",
    "orbital_angular_momentum_candidates",
]
