"""Scalar, source-composed complex IMRPhenomHM intrinsic modes.

London et al., arXiv:1708.00404v4, Methods Eqs. (2)--(5), (8)--(9),
(13)--(14), supply the complex assembly and frequency/phase maps.  The
restricted SPA factors come from Mishra et al., arXiv:1601.05588v1.  The
PhenomD amplitude and phase, QNMs, and fitted alpha coefficients are supplied
by the caller; no frontend or numerical fit is imported here.

``paper_intrinsic_mode`` uses the model-agnostic Eq. (2)--(9)
``exp(+i phi_lm)`` convention with a supplied carrier phase.
``public_intrinsic_mode`` adapts a positive PhenomD phase coordinate to the
project's D-anchored Eq. (14) correction and ``exp(-i Psi_lm)`` convention,
including its odd-m sign and orbital-phase/time factors.  A finite-reference
frequency gauge is an input, not derived by this reference.
"""

from __future__ import annotations

import mpmath as mp
import sympy as sp

from .imrphenomhm_amplitude_ratio import (
    beta21_1p5pn, beta32_1p5pn, beta33_leading, beta44_leading,
)
from .imrphenomhm_amplitude_ratio_43 import beta43_1p5pn
from .imrphenomhm_frequency_map import eq5_frequency_map, outer_map_cutoffs
from .imrphenomhm_phase_composition import composed_phase
from .imrphenomhm_phase_offsets import mapped_phase


COMPONENT = "hm.intrinsic.complex-assembly.eq3.v1"
SPEC_VERSION = "imrphenomhm-intrinsic-modes/v1"
COMPONENT_SOURCES = {
    COMPONENT: (
        ("https://arxiv.org/html/1708.00404v4",
         "Methods Eqs. (2)--(5), (8)--(9), (13)--(14)"),
        ("https://arxiv.org/html/1601.05588v1",
         "Eqs. (11)--(12): restricted SPA mode factors"),
    ),
}
COMPONENT_BINDINGS = {
    COMPONENT: {
        "reference": "paper_intrinsic_mode",
        "public_adapter": "public_intrinsic_mode",
        "torch": ("torchwave.approximants.imrphenom_hm", "IMRPhenomHM"),
        "jax": ("jaxwave.approximants.imrphenom_hm", "IMRPhenomHM"),
        "tests": ("test_public_complex_modes", "test_scalar_paper_assembly_and_zero"),
    },
}

SUPPORTED_MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3))
A22, BETA, KAPPA, PHI22, OFFSET = sp.symbols(
    "A_22 beta_lm kappa phi_22 Delta_lm", real=True,
)


def symbolic_eq3() -> sp.Expr:
    """London Eq. (3) before inserting maps or carrier functions."""
    return sp.Abs(BETA) * A22 * sp.exp(sp.I * (KAPPA * PHI22 + OFFSET))


def _inputs(f, ell, m, mass1, mass2, spin1, spin2):
    if (ell, m) not in SUPPORTED_MODES:
        raise ValueError("unsupported London HM intrinsic mode")
    x, m1, m2, s1, s2 = (mp.mpf(str(z)) for z in
                           (f, mass1, mass2, spin1, spin2))
    if x <= 0 or m2 <= 0 or m1 < m2 or abs(s1) > 1 or abs(s2) > 1:
        raise ValueError("require Mf>0, mass1>=mass2>0, and |spin_i|<=1")
    return x, m1, m2, s1, s2


def _beta(ell, m, x, g, m1, m2, s1, s2):
    eta = m1 * m2 / (m1 + m2) ** 2
    delta = (m1 - m2) / (m1 + m2)
    if (ell, m) == (2, 2):
        return mp.mpf(1)
    if (ell, m) == (3, 3):
        return beta33_leading(g, delta)
    if (ell, m) == (4, 4):
        return beta44_leading(g, eta)
    if (ell, m) == (2, 1):
        return beta21_1p5pn(x, g, m1, m2, s1, s2)
    if (ell, m) == (3, 2):
        return beta32_1p5pn(g, eta, (s1 + s2) / 2)
    return beta43_1p5pn(x, g, m1, m2, s1, s2)


def _amplitude(
    f, ell, m, mass1, mass2, spin1, spin2, amplitude22,
    *, fr_lm, fr22, transition,
):
    x, m1, m2, s1, s2 = _inputs(f, ell, m, mass1, mass2, spin1, spin2)
    fr_lm, fr22 = mp.mpf(str(fr_lm)), mp.mpf(str(fr22))
    if min(fr_lm, fr22) <= 0:
        raise ValueError("supplied QNM frequencies must be positive")

    if (ell, m) == (2, 2):
        mapped_amplitude = x
        beta = mp.mpf(1)
    else:
        f0_amp, _ = outer_map_cutoffs(transition, fr_lm, fr22)
        mapped_amplitude = eq5_frequency_map(x, f0_amp, fr_lm, fr22, m)
        beta = _beta(ell, m, x, mapped_amplitude, m1, m2, s1, s2)
    amplitude = mp.mpf(str(amplitude22(mapped_amplitude)))
    if amplitude < 0:
        raise ValueError("supplied carrier amplitude must be nonnegative")
    return x, m1, m2, s1, s2, fr_lm, fr22, amplitude * abs(beta)


def paper_intrinsic_mode(
    f, ell, m, mass1, mass2, spin1, spin2, amplitude22, carrier_phase,
    *, fr_lm, fr22, transition="london_2018",
) -> mp.mpc:
    """Evaluate model-agnostic London Eqs. (3)--(9), with ``exp(+i phi)``.

    Frequencies are dimensionless ``Mf``.  The supplied carrier phase must
    use the paper's phase coordinate.  A D-specific Eq. (14) correction, if
    desired, must already be present in that supplied phase; this function
    does not apply the project's D-anchored phase correction.  Eq. (4) is
    undefined for exactly symmetric odd modes and raises accordingly.
    """
    x, _, _, _, _, fr_lm, fr22, amplitude = _amplitude(
        f, ell, m, mass1, mass2, spin1, spin2, amplitude22,
        fr_lm=fr_lm, fr22=fr22, transition=transition,
    )
    if (ell, m) == (2, 2):
        phase = carrier_phase(x)
    else:
        _, f0_phi = outer_map_cutoffs(transition, fr_lm, fr22)
        phase = mapped_phase(x, f0_phi, fr_lm, fr22, ell, m, carrier_phase)
    return mp.mpc(amplitude * mp.exp(mp.j * phase))


def public_intrinsic_mode(
    f, ell, m, mass1, mass2, spin1, spin2, amplitude22, positive_d_phase,
    *, fr_lm, fr22, fd_lm, fd22, base_fr22, base_fd22, alpha4, alpha5,
    transition="phenomd_runtime", phic=0, tc=0, frequency_hz=None,
    reference_gauge=0,
) -> mp.mpc:
    """Adapt the paper composition to the project's positive-D phase API.

    ``reference_gauge`` is the *already evaluated* HM-relative D gauge in
    radians.  This reference does not derive the D reference-frequency fit.
    ``frequency_hz`` is required for nonzero ``tc``.
    """
    x, m1, m2, s1, s2 = _inputs(f, ell, m, mass1, mass2, spin1, spin2)
    if m % 2 and m1 == m2 and s1 == s2:
        # The public waveform's exactly symmetric odd modes are zero by
        # body-exchange symmetry, although London's raw Eq. (4) is 0/0.
        return mp.mpc(0)
    x, m1, m2, s1, s2, fr_lm, fr22, amplitude = _amplitude(
        x, ell, m, m1, m2, s1, s2, amplitude22,
        fr_lm=fr_lm, fr22=fr22, transition=transition,
    )
    if amplitude == 0:
        return mp.mpc(0)
    if (ell, m) == (2, 2):
        phase = positive_d_phase(x)
    else:
        fd_lm, fd22, base_fr22, base_fd22 = (
            mp.mpf(str(z)) for z in (fd_lm, fd22, base_fr22, base_fd22)
        )
        if min(fd_lm, fd22, base_fr22, base_fd22) <= 0:
            raise ValueError("supplied damping and D carrier frequencies must be positive")
        _, f0_phi = outer_map_cutoffs(transition, fr_lm, fr22)
        eta = m1 * m2 / (m1 + m2) ** 2
        phase = composed_phase(
            x, f0_phi, fr_lm, fr22, ell, m, positive_d_phase,
            eta=eta, alpha4=alpha4, alpha5=alpha5,
            fd22=fd22, fd_lm=fd_lm,
            base_fr22=base_fr22, base_fd22=base_fd22,
        )
    odd_sign = mp.sign(m1 - m2 if m1 != m2 else s1 - s2) if m % 2 else 1
    t = mp.mpf(str(tc))
    if t and frequency_hz is None:
        raise ValueError("frequency_hz is required for nonzero tc")
    f_hz = mp.mpf(0) if frequency_hz is None else mp.mpf(str(frequency_hz))
    exponent = (-phase + m * mp.mpf(str(phic))
                - 2 * mp.pi * f_hz * t + mp.mpf(str(reference_gauge)))
    return mp.mpc(odd_sign * amplitude * mp.exp(mp.j * exponent))
