# SPDX-License-Identifier: MIT
"""Scalar Appendix-E reference for IMRPhenomXHM inspiral amplitudes.

This module implements only the explicitly printed, re-expanded Fourier-mode
polynomials in Garcia-Quiros et al., arXiv:2001.10914v2, Appendix E.  It has no
tensor-library dependency and contains no fitted collocation values, vetoes,
transition frequencies, or release-compatibility sign changes.
"""

from __future__ import annotations

import math


SPEC_VERSION = "imrphenomxhm-inspiral-amplitude/v1"
REFERENCE_VERSION = SPEC_VERSION
COMPONENT_IDS = {
    "printed_hhat": "xhm.amplitude.inspiral.appendix-e.v1",
    "normalized_pn_amplitude": "xhm.amplitude.inspiral.pn-normalized.v1",
}


def printed_hhat(
    mode: tuple[int, int],
    v: float,
    eta: float,
    delta: float,
    chi_s: float,
    chi_a: float,
) -> complex:
    """Return the printed complex ``Hhat_lm`` through displayed order ``v**6``."""

    v = float(v)
    eta = float(eta)
    delta = float(delta)
    cs = float(chi_s)
    ca = float(chi_a)

    if mode == (2, 1):
        bracket = (
            v * delta
            + v**2 * (-1.5 * delta * cs - 1.5 * ca)
            + v**3 * (117 * delta * eta / 56 + 335 * delta / 672)
            + v**4
            * (
                -965 * delta * eta * cs / 336
                + 3427 * delta * cs / 1344
                - math.pi * delta
                - 0.5j * delta
                - 0.5j * delta * math.log(16)
                - 2101 * eta * ca / 336
                + 3427 * ca / 1344
            )
            + v**5
            * (
                21365 * delta * eta**2 / 8064
                + 10 * delta * eta * ca**2
                + 39 * delta * eta * cs**2 / 8
                - 36529 * delta * eta / 12544
                - 307 * delta * ca**2 / 32
                - 307 * delta * cs**2 / 32
                + 3 * math.pi * delta * cs
                - 964357 * delta / 8128512
                + 213 * eta * ca * cs / 4
                - 307 * ca * cs / 16
                + 3 * math.pi * ca
            )
            + v**6
            * (
                -547 * delta * eta**2 * cs / 768
                - 15 * delta * eta * ca**2 * cs
                - 3 * delta * eta * cs**3 / 16
                - 7049629 * delta * eta * cs / 225792
                + 417 * math.pi * delta * eta / 112
                - 1489j * delta * eta / 112
                - 89j * delta * eta * math.log(2) / 28
                + 729 * delta * ca**2 * cs / 64
                + 243 * delta * cs**3 / 64
                + 143063173 * delta * cs / 5419008
                - 2455 * math.pi * delta / 1344
                - 335j * delta / 1344
                - 335j * delta * math.log(2) / 336
                + 42617 * eta**2 * ca / 1792
                - 15 * eta * ca**3
                - 489 * eta * ca * cs**2 / 16
                - 22758317 * eta * ca / 225792
                + 243 * ca**3 / 64
                + 729 * ca * cs**2 / 64
                + 143063173 * ca / 5419008
            )
        )
        return 1j * math.sqrt(2) * bracket / 3

    if mode == (3, 3):
        bracket = v * delta + v**3 * delta * (27 * eta / 8 - 1945 / 672)
        bracket += v**4 * (
            -2 * delta * eta * cs / 3
            + 65 * delta * cs / 24
            + math.pi * delta
            - 21j * delta / 5
            + 6j * delta * math.log(3 / 2)
            - 28 * eta * ca / 3
            + 65 * ca / 24
        )
        bracket += v**5 * (
            420389 * delta * eta**2 / 63360
            + 10 * delta * eta * ca**2
            + delta * eta * cs**2 / 8
            - 11758073 * delta * eta / 887040
            - 81 * delta * ca**2 / 32
            - 81 * delta * cs**2 / 32
            - 1077664867 * delta / 447068160
            + 81 * eta * ca * cs / 4
            - 81 * ca * cs / 16
        )
        bracket += v**6 * (
            -67 * delta * eta**2 * cs / 24
            - 58745 * delta * eta * cs / 4032
            + 131 * math.pi * delta * eta / 16
            - 440957j * delta * eta / 9720
            + 69j * delta * eta * math.log(3 / 2) / 4
            + 163021 * delta * cs / 16128
            - 5675 * math.pi * delta / 1344
            + 389j * delta / 32
            - 1945j * delta * math.log(3 / 2) / 112
            - 137 * eta**2 * ca / 24
            - 148501 * eta * ca / 4032
            + 163021 * ca / 16128
        )
        return -0.75j * math.sqrt(5 / 7) * bracket

    if mode == (3, 2):
        bracket = v**2 * (1 - 3 * eta) + 4 * eta * cs * v**3
        bracket += v**4 * (-589 * eta**2 / 72 + 12325 * eta / 2016 - 10471 / 10080)
        bracket += v**5 * (
            eta * (113 * delta * ca / 8 + 1081 * cs / 84 - 66j / 5)
            + (-113 * delta * ca - 113 * cs + 72j) / 24
            - 15 * eta**2 * cs
        )
        bracket += v**6 * (
            eta
            * (
                -1633 * delta * ca * cs / 48
                - 563 * ca**2 / 32
                - 2549 * cs**2 / 96
                + 8 * math.pi * cs
                - 8689883 / 149022720
            )
            + 81 * delta * ca * cs / 16
            + 837223 * eta**3 / 63360
            + eta**2 * (30 * ca**2 + 313 * cs**2 / 24 - 78584047 / 2661120)
            + 81 * ca**2 / 32
            + 81 * cs**2 / 32
            + 824173699 / 447068160
        )
        return math.sqrt(5 / 7) * bracket / 3

    if mode == (4, 4):
        bracket = v**2 * (3 * eta - 1)
        bracket += v**4 * (1063 * eta**2 / 88 - 128221 * eta / 7392 + 158383 / 36960)
        bracket += v**5 * (
            math.pi * (2 - 6 * eta)
            + (
                -eta * (1695 * delta * ca + 2075 * cs - 3579j + 2880j * math.log(2))
                + 565 * delta * ca
                + 1140 * eta**2 * cs
                + 565 * cs
                - 1008j
                + 960j * math.log(2)
            )
            / 120
        )
        bracket += v**6 * (
            eta * (243 * delta * ca * cs / 16 + 563 * ca**2 / 32 + 247 * cs**2 / 32 - 22580029007 / 880588800)
            - 81 * delta * ca * cs / 16
            - 7606537 * eta**3 / 274560
            + eta**2 * (-30 * ca**2 - 3 * cs**2 / 8 + 901461137 / 11531520)
            - 81 * ca**2 / 32
            - 81 * cs**2 / 32
            + 7888301437 / 29059430400
        )
        return 4 * math.sqrt(10 / 7) * bracket / 9

    raise ValueError(f"unsupported printed Appendix-E mode {mode}")


def normalized_pn_amplitude(
    mode: tuple[int, int],
    dimensionless_frequency: float,
    eta: float,
    delta: float,
    chi_s: float,
    chi_a: float,
) -> float:
    """Return the Sec.-IV-A ``|A_lm^PN|/A_22^0`` Appendix-E baseline."""

    m = mode[1]
    frequency = float(dimensionless_frequency)
    if frequency <= 0:
        raise ValueError("dimensionless_frequency must be positive")
    v = (2 * math.pi * frequency / m) ** (1 / 3)
    return (m / 2) ** (7 / 6) * abs(printed_hhat(mode, v, eta, delta, chi_s, chi_a))


__all__ = [
    "COMPONENT_IDS",
    "REFERENCE_VERSION",
    "SPEC_VERSION",
    "normalized_pn_amplitude",
    "printed_hhat",
]
