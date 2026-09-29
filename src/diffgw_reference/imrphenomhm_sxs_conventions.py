"""Scalar Fourier/mode-pair convention bridge for HM versus SXS strain.

London et al., arXiv:1708.00404v4 Eq. (1), give the spin-weighted mode
expansion. García-Quirós et al., arXiv:2001.10914v2 Sec. II Eqs. (2),
(4)--(9), fix the positive-frequency Fourier carrier. The SXS catalog,
arXiv:1904.04831v2, documents the strain convention. No NR phase or
production waveform value is used to select a conversion factor here.
"""

from __future__ import annotations

import mpmath as mp
import sympy as sp


SPEC_VERSION = "imrphenomhm-sxs-conventions/v1"
COMPONENT = "hm.conventions.sxs-bridge.v1"
COMPONENT_SOURCES = {
    COMPONENT: (
        ("https://arxiv.org/html/1708.00404v4", "Eq. (1): strain mode expansion"),
        ("https://arxiv.org/html/2001.10914v2", "Sec. II Eqs. (2), (4)--(9): Fourier mode pairs"),
        ("https://arxiv.org/html/2001.10914v2", "Eq. (14): orbital rotation of strain modes"),
        ("https://arxiv.org/abs/1904.04831v2", "SXS strain convention and catalog sign revision"),
    ),
}


def symbolic_gauge_phase(m: sp.Expr, f: sp.Expr, orbital_phase: sp.Expr,
                         time_shift: sp.Expr, reference_base: sp.Expr,
                         *, fourier_sign: int = -1) -> sp.Expr:
    """Phase for orbital/time shifts and the *native* finite-f_ref gauge.

    ``fourier_sign=-1`` denotes the kernel exp(-2*pi*i*f*t). The
    ``orbital_phase`` is the public positive-carrier ``phic``; its sign is
    opposite to the physical mode rotation angle in XHM Eq. (14). The reference
    base B = (dPsi_D/d(Mf))_peak * M_seconds * f_ref is supplied: this
    component does not evaluate the fitted D derivative. The final term is
    the documented public HM adapter, not a factor inferred from SXS data.
    """
    if fourier_sign not in (-1, 1):
        raise ValueError("fourier_sign must be -1 or +1")
    return (m * orbital_phase + fourier_sign * 2 * sp.pi * f * time_shift
            + (m / 2 - 1) * reference_base)


def gauge_factor(m: int, f, *, orbital_phase=0, time_shift=0,
                 reference_base=0, global_phase=0, fourier_sign=-1) -> mp.mpc:
    """Apply a declared unit-modulus phase; no factor is inferred or fitted."""
    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer")
    if fourier_sign not in (-1, 1):
        raise ValueError("fourier_sign must be -1 or +1")
    f, alpha, tau, base, gamma = (mp.mpf(str(x)) for x in
                                  (f, orbital_phase, time_shift,
                                   reference_base, global_phase))
    if f <= 0:
        raise ValueError("positive-frequency carrier requires f>0")
    angle = (m * alpha + fourier_sign * 2 * mp.pi * f * tau
             + (mp.mpf(m) / 2 - 1) * base + gamma)
    return mp.mpc(mp.exp(mp.j * angle))


def paired_spectrum(positive_m_at_negative_f, ell: int) -> mp.mpc:
    """From h_(ell,-m)(t)=(-1)^ell conj(h_(ell,m)(t)).

    For either Fourier-kernel sign, H_(ell,-m)(+f) equals this expression
    applied to H_(ell,m)(-f). It is not a same-frequency conjugation.
    """
    if not isinstance(ell, int) or ell < 2:
        raise ValueError("ell must be an integer >= 2")
    return mp.mpc((-1) ** ell * mp.conj(positive_m_at_negative_f))


def positive_carrier_from_negative_m(negative_m_at_positive_f, ell: int) -> mp.mpc:
    """Convert N_lm=F[h_(ell,-m)] to P_lm=F[conj(h_(ell,m))].

    The time-domain pair identity gives P_lm=(-1)^ell N_lm. This is the
    carrier used by the frozen SXS holdout, before amplitude scaling and the
    single common (2,2) alignment.
    """
    if not isinstance(ell, int) or ell < 2:
        raise ValueError("ell must be an integer >= 2")
    return mp.mpc((-1) ** ell * mp.mpc(negative_m_at_positive_f))


def doubled_relative_phase(mode, carrier_at_2f_over_m, m: int) -> mp.mpc:
    """Unit invariant T_lm=u_lm(f)^2 conj[u_22(2f/m)]^m."""
    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer")
    mode, carrier = mp.mpc(mode), mp.mpc(carrier_at_2f_over_m)
    if abs(mode) == 0 or abs(carrier) == 0:
        raise ValueError("relative phase is undefined for a zero mode")
    u, u22 = mode / abs(mode), carrier / abs(carrier)
    return mp.mpc(u ** 2 * mp.conj(u22) ** m)
