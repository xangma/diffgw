# SPDX-License-Identifier: MIT
"""Clean-room, non-reexpanded SPA branch for the ``(2,1)`` mode.

This module is a diagnostic implementation of the branch described in
arXiv:2001.10914v2, Appendix E.  For mass ratios ``q < 40`` that paper keeps
the stationary-phase expression in its product form instead of re-expanding
the square root of the TaylorT4 frequency derivative.  The code here is not
connected to an XHM production waveform or to any calibration data.

The time-domain mode is written in the phase convention used by the XHM
Appendix, through the orders supported by the published PN mode formulae:

* Arun et al., arXiv:0810.5336, Eq. (4.17b) (rendered PDF p. 12), fixes the
  leading mode and its aligned-spin and 1PN terms (with the constant phase
  convention converted to the Appendix convention).
* Buonanno, Faye and Hinderer, arXiv:1209.6349, Eqs. (4.26) and (4.27a)
  (rendered PDF p. 12), supply the non-precessing 2PN spin mode and its
  aligned-spin terms at ``v**4``.
* Blanchet, arXiv:1310.1528v6, Eq. (492), H21 (rendered PDF p. 214),
  supplies the non-spinning tail and higher terms through ``v**6``.

The displayed Appendix-E ``H21`` coefficients are used only as an algebraic
re-expansion check.  They are not used to fit or select any coefficient.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch

from .taylorf2 import TaylorF2


__all__ = [
    "taylor_t4_ratio",
    "taylor_t4_ratio_printed",
    "taylor_t4_a6_3pn",
    "taylor_t4_xdot",
    "td_hhat_21",
    "spa_hhat_21",
    "spa_hhat_21_printed",
    "normalized_spa_amplitude_21",
    "normalized_spa_amplitude_21_printed",
    "physical_spa_amplitude_21",
    "physical_spa_amplitude_21_printed",
]


_PI = math.pi
_LOG2 = math.log(2.0)


def _broadcast(*values: object) -> Tuple[torch.Tensor, ...]:
    """Broadcast all inputs as differentiable float64 tensors."""

    tensors = tuple(torch.as_tensor(value, dtype=torch.float64) for value in values)
    return torch.broadcast_tensors(*tensors)


def _finite(*values: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(value).all().detach().cpu()) for value in values)


def _check_eta_delta(eta: torch.Tensor, delta: torch.Tensor) -> None:
    if not _finite(eta, delta):
        raise ValueError("eta and delta must be finite")
    if bool((eta <= 0).any()) or bool((eta > 0.25).any()):
        raise ValueError("eta must satisfy 0 < eta <= 1/4")
    if bool((delta.abs() > 1).any()):
        raise ValueError("delta must satisfy |delta| <= 1")


def _check_spins(*spins: torch.Tensor) -> None:
    if not _finite(*spins):
        raise ValueError("dimensionless spins must be finite")


def _imaginary(v: torch.Tensor) -> torch.Tensor:
    return torch.complex(torch.zeros_like(v), torch.ones_like(v))


def taylor_t4_ratio_printed(
    v: object,
    eta: object,
    delta: object,
    chi_s: object,
    chi_a: object,
) -> torch.Tensor:
    """Return the printed-truncation TaylorT4 ratio.

    Here ``x = v**2`` and the total mass is one.  The printed relative
    ``v**2``, ``v**3``, ``v**4``, ``v**5`` and ``v**7`` terms are retained
    literally; the displayed Appendix-E truncation has no relative ``v**6``
    term.  This named function preserves that algebraic convention for
    comparisons with the paper.

    ``eta`` and ``delta`` are exposed separately so coefficient checks can use
    independent symbols.  Physical binaries satisfy
    ``eta = (1-delta**2)/4``.
    """

    v, eta, delta, chi_s, chi_a = _broadcast(v, eta, delta, chi_s, chi_a)
    if bool((v < 0).any()):
        raise ValueError("v must be non-negative")
    _check_eta_delta(eta, delta)
    _check_spins(chi_s, chi_a)

    # Eq. (xdot) of arXiv:2001.10914v2, divided by (64/5) eta v**10,
    # after M=1 and m1=(1+delta)/2, m2=(1-delta)/2.  The terms are kept in
    # eta, delta, chi_s and chi_a form to make the physical exchange parity
    # and the continuation used by coefficient checks explicit.
    a2 = -(743.0 + 924.0 * eta) / 336.0
    a3 = -(
        113.0 * delta * chi_a
        - 76.0 * eta * chi_s
        + 113.0 * chi_s
        - 48.0 * _PI
    ) / 12.0
    a4 = (
        -362880.0 * eta * chi_a**2
        + 91854.0 * chi_a**2
        + 183708.0 * delta * chi_a * chi_s
        - 4536.0 * eta * chi_s**2
        + 91854.0 * chi_s**2
        + 59472.0 * eta**2
        + 122949.0 * eta
        + 34103.0
    ) / 18144.0
    a5 = (
        97356.0 * eta * delta * chi_a
        - 62638.0 * delta * chi_a
        - 53088.0 * eta**2 * chi_s
        + 183800.0 * eta * chi_s
        - 62638.0 * chi_s
        - 47628.0 * _PI * eta
        - 12477.0 * _PI
    ) / 2016.0
    a7 = -(
        4200798.0 * eta**2 * delta * chi_a
        - 14329242.0 * eta * delta * chi_a
        + 6255600.0 * delta * chi_a
        - 1817592.0 * eta**3 * chi_s
        + 14353626.0 * eta**2 * chi_s
        - 22057406.0 * eta * chi_s
        + 6255600.0 * chi_s
        - 2195880.0 * _PI * eta**2
        - 2152050.0 * _PI * eta
        + 39735.0 * _PI
    ) / 36288.0
    return 1.0 + a2 * v**2 + a3 * v**3 + a4 * v**4 + a5 * v**5 + a7 * v**7


def taylor_t4_a6_3pn(
    v: object,
    eta: object,
    delta: object,
    chi_s: object,
    chi_a: object,
) -> torch.Tensor:
    """Recover the missing relative ``v**6`` TaylorT4 coefficient.

    The coefficient is obtained by differentiating the public TaylorF2
    phase coefficients and inverting that series.  ``TaylorF2`` is called
    with ``phenom_d_calibration=False`` so the full 3PN spin-spin terms and
    the 3PN logarithm are retained.  No waveform residual or calibration
    value enters this calculation.

    The phase-coefficient callable uses physical masses, so this corrected
    branch evaluates the algebra on the physical continuation
    ``m1=(1+delta)/2``, ``m2=(1-delta)/2``.  The separate printed helper above
    remains available for independent eta/delta polynomial checks.
    """
    v, eta, delta, chi_s, chi_a = _broadcast(v, eta, delta, chi_s, chi_a)
    if bool((v < 0).any()):
        raise ValueError("v must be non-negative")
    _check_eta_delta(eta, delta)
    _check_spins(chi_s, chi_a)
    if bool((delta.abs() >= 1).any()):
        raise ValueError("the full 3PN phase branch requires |delta| < 1")

    # Unit total mass is sufficient because TaylorF2 phase coefficients are
    # dimensionless.  Reconstruct the component spins from chi_s/chi_a so
    # the public callable receives the same aligned-spin variables.
    mass1 = (1.0 + delta) / 2.0
    mass2 = (1.0 - delta) / 2.0
    spin1 = chi_s + chi_a
    spin2 = chi_s - chi_a
    phase = TaylorF2.phase_coefficients(
        mass1,
        mass2,
        spin1,
        spin2,
        spin_order=7,
        phenom_d_calibration=False,
    )
    _, p2, p3, p4, p5, l5, p6, l6, *_ = phase

    # At v=0 the product v**6 log(v) has a zero limit.  Use log(1)=0 at
    # that point so the intermediate coefficient remains finite without
    # changing any positive-frequency value.
    log4v = torch.log(torch.where(v > 0, 4.0 * v, torch.ones_like(v)))

    def b(k: int, p: torch.Tensor, logarithmic: torch.Tensor) -> torch.Tensor:
        pk = p + logarithmic * log4v
        return (
            (k - 5) * (k - 8) * pk + (2 * k - 13) * logarithmic
        ) / 40.0

    zero = torch.zeros_like(v)
    b2 = b(2, p2, zero)
    b3 = b(3, p3, zero)
    b4 = b(4, p4, zero)
    b6 = b(6, p6, l6)
    return -b6 + 2.0 * b2 * b4 + b3.square() - b2**3


def taylor_t4_ratio(
    v: object,
    eta: object,
    delta: object,
    chi_s: object,
    chi_a: object,
) -> torch.Tensor:
    """Return TaylorT4 ``R`` including the independently derived 3PN ``v**6``.

    The lower-order and relative ``v**7`` terms retain the displayed XHM
    convention.  Only the missing ``v**6`` term is added, and it is derived
    from the uncalibrated full TaylorF2 phase as documented in
    :func:`taylor_t4_a6_3pn`.
    """
    v, eta, delta, chi_s, chi_a = _broadcast(v, eta, delta, chi_s, chi_a)
    return taylor_t4_ratio_printed(v, eta, delta, chi_s, chi_a) + taylor_t4_a6_3pn(
        v, eta, delta, chi_s, chi_a
    ) * v**6


def taylor_t4_xdot(
    x: object,
    eta: object,
    delta: object,
    chi_s: object,
    chi_a: object,
) -> torch.Tensor:
    """Return the published, dimensionless TaylorT4 ``dx/dt`` for ``M=1``.

    The normalization is ``xdot = (64/5) eta x**5 R(sqrt(x))``.  For a
    physical total mass ``M`` the derivative is ``xdot/M``; the ratio ``R``
    itself is dimensionless and is the only quantity needed by the SPA
    amplitude.
    """

    x, eta, delta, chi_s, chi_a = _broadcast(x, eta, delta, chi_s, chi_a)
    if bool((x < 0).any()):
        raise ValueError("x must be non-negative")
    return (64.0 / 5.0) * eta * x**5 * taylor_t4_ratio(
        torch.sqrt(x), eta, delta, chi_s, chi_a
    )


def td_hhat_21(
    v: object,
    eta: object,
    delta: object,
    chi_s: object,
    chi_a: object,
) -> torch.Tensor:
    """Return the published aligned-spin time-domain ``hat h_21``.

    The result is normalized as in the XHM Appendix, so its leading term is
    ``+i*delta*v/3``.  Through ``v**3`` this is the aligned specialization of
    Arun et al. Eq. (4.17b), after the constant azimuthal phase conversion.
    The ``v**4`` spin terms agree with the non-precessing 2PN mode of
    Buonanno--Faye--Hinderer, while the non-spinning ``v**4``--``v**6`` terms
    agree with the published Blanchet H21 mode.  No commented TeX from the
    XHM source is used as evidence for these terms.
    """

    v, eta, delta, chi_s, chi_a = _broadcast(v, eta, delta, chi_s, chi_a)
    if bool((v < 0).any()):
        raise ValueError("v must be non-negative")
    _check_eta_delta(eta, delta)
    _check_spins(chi_s, chi_a)
    i = _imaginary(v)

    b = delta * v
    b = b - 1.5 * (chi_a + delta * chi_s) * v**2
    b = b + delta * (20.0 * eta - 17.0) / 28.0 * v**3
    b4 = (
        chi_a * (205.0 * eta - 7.0) / 14.0
        + delta * chi_s * (33.0 * eta - 7.0) / 14.0
        + _PI * delta
        - i * delta * (0.5 + 2.0 * _LOG2)
    )
    b = b + b4 * v**4
    b = b + delta * (237.0 * eta**2 - 2036.0 * eta - 172.0) / 504.0 * v**5
    b6 = delta * (
        -17.0 * _PI / 28.0
        + 3.0 * _PI * eta / 14.0
        + i
        * (
            17.0 / 56.0
            - 353.0 * eta / 28.0
            - 3.0 * eta * _LOG2 / 7.0
            + 17.0 * _LOG2 / 14.0
        )
    )
    b = b + b6 * v**6
    return i * b / 3.0


def spa_hhat_21(
    v: object,
    eta: object,
    delta: object,
    chi_s: object,
    chi_a: object,
) -> torch.Tensor:
    """Return the non-reexpanded dimensionless SPA ``Hhat_21``.

    This is Eq. (spaamp) before the TaylorT4 square-root factor is
    re-expanded:

    ``Hhat_21 = sqrt(2) * td_hhat_21 / sqrt(R)``.

    The square root is real only on the inspiral branch ``R > 0``.  A caller
    selecting the paper's branch must also enforce ``q < 40``; the mass-aware
    wrappers below perform that check.  This low-level function intentionally
    accepts independent ``eta`` and ``delta`` symbols for algebraic checks.
    """

    return _spa_hhat_21_with_ratio(
        v, eta, delta, chi_s, chi_a, taylor_t4_ratio
    )


def spa_hhat_21_printed(
    v: object,
    eta: object,
    delta: object,
    chi_s: object,
    chi_a: object,
) -> torch.Tensor:
    """Return the Appendix-E non-reexpanded SPA with printed ``R`` truncation."""
    return _spa_hhat_21_with_ratio(
        v, eta, delta, chi_s, chi_a, taylor_t4_ratio_printed
    )


def _spa_hhat_21_with_ratio(
    v: object,
    eta: object,
    delta: object,
    chi_s: object,
    chi_a: object,
    ratio_function,
) -> torch.Tensor:
    v, eta, delta, chi_s, chi_a = _broadcast(v, eta, delta, chi_s, chi_a)
    if bool((v <= 0).any()):
        raise ValueError("v must be positive for the SPA branch")
    td = td_hhat_21(v, eta, delta, chi_s, chi_a)
    ratio = ratio_function(v, eta, delta, chi_s, chi_a)
    if bool((ratio <= 0).any().detach().cpu()):
        raise ValueError("TaylorT4 xdot is non-positive on this SPA branch")
    return math.sqrt(2.0) * td / torch.sqrt(ratio)


def _mass_inputs(
    frequency: object,
    total_mass_seconds: object,
    m1: object,
    m2: object,
    chi1: object,
    chi2: object,
    distance_seconds: object | None = None,
) -> tuple[torch.Tensor, ...]:
    values = [frequency, total_mass_seconds, m1, m2, chi1, chi2]
    if distance_seconds is not None:
        values.append(distance_seconds)
    tensors = _broadcast(*values)
    f, mt, m1, m2, chi1, chi2 = tensors[:6]
    if bool((f <= 0).any()) or bool((mt <= 0).any()):
        raise ValueError("frequency and total_mass_seconds must be positive")
    if bool((m1 <= 0).any()) or bool((m2 <= 0).any()):
        raise ValueError("component masses must be positive")
    _check_spins(chi1, chi2)
    q = torch.maximum(m1, m2) / torch.minimum(m1, m2)
    if bool((q >= 40.0).any()):
        raise ValueError("the non-reexpanded (2,1) branch is defined only for q < 40")
    if distance_seconds is None:
        return f, mt, m1, m2, chi1, chi2
    distance = tensors[6]
    if bool((distance <= 0).any()):
        raise ValueError("distance_seconds must be positive")
    return f, mt, m1, m2, chi1, chi2, distance


def normalized_spa_amplitude_21(
    frequency: object,
    total_mass_seconds: object,
    m1: object,
    m2: object,
    chi1: object,
    chi2: object,
) -> torch.Tensor:
    """Return ``|Hhat_21|`` with the existing mode-normalization convention.

    The returned value is dimensionless.  ``m1`` and ``m2`` need only share a
    mass unit; ``total_mass_seconds`` is the physical total mass in seconds
    used to convert frequency to ``v=(2*pi*f*M)**(1/3)``.  The paper branch is
    explicitly limited to ``q=max(m1,m2)/min(m1,m2) < 40``.
    """

    return _normalized_spa_amplitude_21_with_hhat(
        frequency,
        total_mass_seconds,
        m1,
        m2,
        chi1,
        chi2,
        spa_hhat_21,
    )


def normalized_spa_amplitude_21_printed(
    frequency: object,
    total_mass_seconds: object,
    m1: object,
    m2: object,
    chi1: object,
    chi2: object,
) -> torch.Tensor:
    """Return ``|Hhat_21|`` using the explicitly printed TaylorT4 truncation."""
    return _normalized_spa_amplitude_21_with_hhat(
        frequency,
        total_mass_seconds,
        m1,
        m2,
        chi1,
        chi2,
        spa_hhat_21_printed,
    )


def _normalized_spa_amplitude_21_with_hhat(
    frequency: object,
    total_mass_seconds: object,
    m1: object,
    m2: object,
    chi1: object,
    chi2: object,
    hhat_function,
) -> torch.Tensor:
    f, mt, m1, m2, chi1, chi2 = _mass_inputs(
        frequency, total_mass_seconds, m1, m2, chi1, chi2
    )
    total = m1 + m2
    eta = m1 * m2 / total**2
    delta = (m1 - m2) / total
    chi_s = (chi1 + chi2) / 2.0
    chi_a = (chi1 - chi2) / 2.0
    v = (2.0 * _PI * f * mt).pow(1.0 / 3.0)
    return hhat_function(v, eta, delta, chi_s, chi_a).abs()


def physical_spa_amplitude_21(
    frequency: object,
    total_mass_seconds: object,
    m1: object,
    m2: object,
    chi1: object,
    chi2: object,
    distance_seconds: object,
) -> torch.Tensor:
    """Return the raw physical SPA amplitude magnitude in geometric units.

    Eq. (spaamp_2) has the prefactor ``M**2/D``.  Consequently the returned
    quantity is in the same Fourier-domain strain-time units as a public
    frequency-domain mode when ``total_mass_seconds`` and ``distance_seconds``
    use SI seconds.  No norm normalization is applied.
    """

    return _physical_spa_amplitude_21_with_hhat(
        frequency,
        total_mass_seconds,
        m1,
        m2,
        chi1,
        chi2,
        distance_seconds,
        spa_hhat_21,
    )


def physical_spa_amplitude_21_printed(
    frequency: object,
    total_mass_seconds: object,
    m1: object,
    m2: object,
    chi1: object,
    chi2: object,
    distance_seconds: object,
) -> torch.Tensor:
    """Return the physical SPA amplitude with the printed ``R`` truncation."""
    return _physical_spa_amplitude_21_with_hhat(
        frequency,
        total_mass_seconds,
        m1,
        m2,
        chi1,
        chi2,
        distance_seconds,
        spa_hhat_21_printed,
    )


def _physical_spa_amplitude_21_with_hhat(
    frequency: object,
    total_mass_seconds: object,
    m1: object,
    m2: object,
    chi1: object,
    chi2: object,
    distance_seconds: object,
    hhat_function,
) -> torch.Tensor:
    f, mt, m1, m2, chi1, chi2, distance = _mass_inputs(
        frequency, total_mass_seconds, m1, m2, chi1, chi2, distance_seconds
    )
    total = m1 + m2
    eta = m1 * m2 / total**2
    delta = (m1 - m2) / total
    chi_s = (chi1 + chi2) / 2.0
    chi_a = (chi1 - chi2) / 2.0
    v = (2.0 * _PI * f * mt).pow(1.0 / 3.0)
    hhat = hhat_function(v, eta, delta, chi_s, chi_a)
    return (
        mt**2
        / distance
        * _PI
        * torch.sqrt(2.0 * eta / 3.0)
        * v.pow(-3.5)
        * hhat.abs()
    )
