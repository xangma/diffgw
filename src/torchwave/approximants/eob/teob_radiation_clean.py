"""Clean-room, nonspinning TEOB factorized radiation.

This module is a small, explicit-input radiation stage reconstructed from the
rendered equations in Nagar et al., arXiv:1406.6913v1, Appendix A, and
Damour, Nagar & Bernuzzi, arXiv:1212.4357v2, Appendix D.  The 2018 update
arXiv:1806.01772v2 explains the odd-(l=5) factor already present in the
general Newtonian mass polynomial.

The caller supplies the EOB state and energies.  In particular, this module
does not evaluate a Hamiltonian, choose an EOB potential, or attach NQC,
merger, ringdown, spin, or tidal corrections.  Modes are returned in the
dimensionless convention ``R h_lm/M`` used by the cited equations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .factorized_modes import (
    PUBLISHED_PHASE_MODES,
    _newtonian_constant,
    _rho_table,
    _tail_abs_squared,
    _tail_phase,
)
from ...constants import EULER_GAMMA

Mode = tuple[int, int]
ResidualPhase = Literal["published", "zero-unpublished"]


@dataclass(frozen=True)
class _Prepared:
    """Broadcast, checked radiation inputs and the derived TEOB quantities."""

    r: Tensor
    Omega: Tensor
    phi: Tensor
    prstar: Tensor
    pphi: Tensor
    Heff: Tensor
    H: Tensor
    Hcirc: Tensor
    Aprime: Tensor
    m1: Tensor
    m2: Tensor
    nu: Tensor
    rOmega: Tensor
    vphi: Tensor
    y: Tensor
    nqc22: Tensor


def _reference_tensor(*values: object) -> Tensor:
    for value in values:
        if isinstance(value, Tensor):
            if value.is_complex():
                raise TypeError("radiation inputs must be real tensors")
            return value
    return torch.empty((), dtype=torch.get_default_dtype())


def _as_tensor(value: object, reference: Tensor, name: str) -> Tensor:
    if (isinstance(value, Tensor) and value.is_complex()) or isinstance(value, complex):
        raise TypeError(f"{name} must be a real tensor or scalar")
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.is_complex():
        raise TypeError(f"{name} must be a real tensor or scalar")
    return result


def _finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _positive(name: str, value: Tensor) -> None:
    if bool((value <= 0).any()):
        raise ValueError(f"{name} must be positive")


def non_keplerian_radius(r: Tensor, Hcirc: Tensor, Aprime: Tensor, *, Omega: Tensor | None = None) -> Tensor:
    """Return the nonspinning ``r_Omega`` from 1406.6913 Eq. (70).

    For zero spins, ``r_c=r``, ``G~=0`` and ``psi=2/(r**2*A')``.  The
    Hamiltonian in Eq. (70) is therefore supplied as ``Hcirc``, evaluated at
    ``p_r*=0`` along the current ``p_phi``:

    ``r_Omega = (r**3 * psi * Hcirc**2)**(1/3)``.

    ``Omega`` is accepted as an ignored keyword to make the helper convenient
    for callers that pass the complete kinematic tuple; it does not enter
    Eq. (70).
    """
    del Omega
    reference = _reference_tensor(r, Hcirc, Aprime)
    r_t, hc_t, ap_t = (_as_tensor(value, reference, name)
                        for value, name in ((r, "r"), (Hcirc, "Hcirc"), (Aprime, "Aprime")))
    r_t, hc_t, ap_t = torch.broadcast_tensors(r_t, hc_t, ap_t)
    _finite("r", r_t)
    _finite("Hcirc", hc_t)
    _finite("Aprime", ap_t)
    _positive("r", r_t)
    _positive("Hcirc", hc_t)
    _positive("Aprime", ap_t)
    psi = 2 / (r_t.square() * ap_t)
    return torch.pow(r_t.pow(3) * psi * hc_t.square(), 1 / 3)


def non_keplerian_velocity(r: Tensor, Omega: Tensor, Hcirc: Tensor, Aprime: Tensor) -> Tensor:
    """Return ``v_phi=r_Omega*Omega`` from 1406.6913 Eq. (69)."""
    reference = _reference_tensor(r, Omega, Hcirc, Aprime)
    r_t, omega_t, hc_t, ap_t = (
        _as_tensor(value, reference, name)
        for value, name in ((r, "r"), (Omega, "Omega"), (Hcirc, "Hcirc"), (Aprime, "Aprime"))
    )
    r_t, omega_t, hc_t, ap_t = torch.broadcast_tensors(r_t, omega_t, hc_t, ap_t)
    return non_keplerian_radius(r_t, hc_t, ap_t) * omega_t


# Names used in notes and by downstream callers.
teob_r_omega = non_keplerian_radius
teob_vphi = non_keplerian_velocity
teob_non_keplerian_radius = non_keplerian_radius


def _prepare(
    r: object,
    Omega: object,
    phi: object,
    prstar: object,
    pphi: object,
    Heff: object,
    H: object,
    Hcirc: object,
    Aprime: object,
    m1: object,
    m2: object,
    nqc22: object,
) -> _Prepared:
    reference = _reference_tensor(
        r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2
    )
    names = ("r", "Omega", "phi", "prstar", "pphi", "Heff", "H", "Hcirc", "Aprime", "m1", "m2")
    values = tuple(
        _as_tensor(value, reference, name)
        for value, name in zip(
            (r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2), names
        )
    )
    (r_t, omega_t, phi_t, pr_t, pphi_t, heff_t, h_t, hcirc_t,
     aprime_t, m1_t, m2_t) = torch.broadcast_tensors(*values)
    for name, value in zip(names, (r_t, omega_t, phi_t, pr_t, pphi_t, heff_t,
                                   h_t, hcirc_t, aprime_t, m1_t, m2_t)):
        _finite(name, value)
    nqc_is_complex = (isinstance(nqc22, Tensor) and nqc22.is_complex()) or isinstance(nqc22, complex)
    if nqc_is_complex:
        nqc_t = torch.as_tensor(
            nqc22,
            dtype=torch.complex64 if reference.dtype == torch.float32 else torch.complex128,
            device=reference.device,
        )
    else:
        nqc_t = _as_tensor(nqc22, reference, "nqc22")
    nqc_t = torch.broadcast_to(nqc_t, r_t.shape)
    _finite("nqc22", nqc_t)
    for name, value in (("r", r_t), ("Omega", omega_t), ("Heff", heff_t),
                        ("H", h_t), ("Hcirc", hcirc_t), ("Aprime", aprime_t),
                        ("m1", m1_t), ("m2", m2_t)):
        _positive(name, value)

    total_mass = m1_t + m2_t
    x1, x2 = m1_t / total_mass, m2_t / total_mass
    nu = x1 * x2
    r_omega = non_keplerian_radius(r_t, hcirc_t, aprime_t)
    vphi = r_omega * omega_t
    _finite("rOmega", r_omega)
    _finite("vphi", vphi)
    _positive("vphi", vphi)
    y = torch.pow(h_t * omega_t, 2 / 3)
    _finite("y", y)
    _positive("y", y)
    if bool((vphi == 0).any()):
        raise ValueError("vphi must be nonzero")
    return _Prepared(
        r_t, omega_t, phi_t, pr_t, pphi_t, heff_t, h_t, hcirc_t, aprime_t,
        m1_t, m2_t, nu, r_omega, vphi, y, nqc_t,
    )


def _complete_rho_table(x: Tensor, nu: Tensor) -> dict[Mode, Tensor]:
    """DIN/1212.4357 Appendix D amplitudes, including displayed tails.

    The existing clean-room DIN helper contains the common terms.  This
    completion adds the published pieces that were absent from that helper:
    the finite-mass 3PN terms in rho33/rho31 from Faye, Blanchet & Iyer,
    arXiv:1409.3546v2 Eq. (64), the test-mass 3PN terms in rho43/rho41/rho55
    from arXiv:1212.4357v2 Appendix D Eqs. (D6), (D8), and (D9), the later
    displayed test-mass terms in Appendix D1--D26, and the decimal 2PN terms
    in all eight l=8 entries D27--D33.  Each term is additive to the common
    table; the generic FactorizedModes helper is left unchanged.
    """
    rho = _rho_table(x, nu)
    eulerlog = {
        m: EULER_GAMMA + math.log(2 * m) + 0.5 * torch.log(x)
        for m in range(1, 8)
    }
    additions: dict[Mode, Tensor] = {
        (2, 1): (-63735873771463 / 16569158860800 + 5029963 / 5927040 * eulerlog[1]) * x**5,
        # The nu-dependent 3PN terms are Eq. (64a) of arXiv:1409.3546v2;
        # the x**5 term is the test-mass completion in Appendix D3.
        (3, 3): (
            ((-129509 / 25740 + 41 * math.pi**2 / 192) * nu
             - 274621 / 154440 * nu**2 + 12011 / 46332 * nu**3) * x**3
            + (-903823148417327 / 30566888352000
               + 87347 / 13860 * eulerlog[3]) * x**5
        ),
        (3, 2): (-10607269449358 / 3072140846775 + 17056 / 8505 * eulerlog[2]) * x**4,
        # Eq. (64b) of arXiv:1409.3546v2 plus the Appendix D5 x**5 term.
        (3, 1): (
            ((-9688441 / 2084940 + 41 * math.pi**2 / 192) * nu
             + 174535 / 75816 * nu**2 - 727247 / 1250964 * nu**3) * x**3
            + (430750057673539 / 297110154781440
               - 1313 / 224532 * eulerlog[1]) * x**5
        ),
        (4, 4): (845198 / 190575 * eulerlog[4]
                 - 172066910136202271 / 19426955708160000) * x**4,
        (4, 2): (300061 / 381150 * eulerlog[2]
                 - 12864377174485679 / 19426955708160000) * x**4,
        # The x**3 test-mass terms are Appendix D6 and D8; retain the
        # existing Appendix D6/D8 x**4 completions after them.
        (4, 3): (
            (-1571 / 770 * eulerlog[3] + 1664224207351 / 195343948800) * x**3
            + (-2465107182496333 / 460490801971200
               + 174381 / 67760 * eulerlog[3]) * x**4
        ),
        (4, 1): (
            (-1571 / 6930 * eulerlog[1] + 1227423222031 / 1758095539200) * x**3
            + (-29584392078751453 / 37299754959667200
               + 67553 / 261360 * eulerlog[1]) * x**4
        ),
        # Appendix D9 supplies the missing x**3 term before the existing x**4.
        (5, 5): (
            (-1546 / 429 * eulerlog[5] + 190606537999247 / 11957879934000) * x**3
            + (-1213641959949291437 / 118143853747920000
               + 376451 / 83655 * eulerlog[5]) * x**4
        ),
        (5, 4): -16213384 / 15526875 * x**2
                + (-24736 / 10725 * eulerlog[4]
                   + 6704294638171892 / 653946558890625) * x**3,
        (5, 2): -7187914 / 15526875 * x**2
                + (1539689950126502 / 653946558890625
                   - 6184 / 10725 * eulerlog[2]) * x**3,
        (5, 3): (-4638 / 3575 * eulerlog[3]
                 + 7618462680967 / 1328653326000) * x**3
                + (-77082121019870543 / 39381284582640000
                   + 2319 / 1859 * eulerlog[3]) * x**4,
        (5, 1): (-1546 / 10725 * eulerlog[1]
                 + 7685351978519 / 11957879934000) * x**3
                + (-821807362819271 / 10740350340720000
                   + 22417 / 190125 * eulerlog[1]) * x**4,
        (6, 6): -1025435 / 659736 * x**2
                + (-3604 / 1001 * eulerlog[6]
                   + 610931247213169 / 36701493028200) * x**3,
        (6, 5): -59574065 / 54286848 * x**2
                + (-22525 / 9009 * eulerlog[5]
                   + 67397117912549267 / 5798416452820992) * x**3,
        (6, 4): -476887 / 659736 * x**2
                + (-14416 / 9009 * eulerlog[4]
                   + 180067034480351 / 24467662018800) * x**3,
        (6, 3): -152153941 / 271434240 * x**2
                + (-901 / 1001 * eulerlog[3]
                   + 116042497264681103 / 28992082264104960) * x**3,
        (6, 2): -817991 / 3298680 * x**2
                + (-3604 / 9009 * eulerlog[2]
                   + 812992177581 / 453104852200) * x**3,
        (6, 1): -79192261 / 271434240 * x**2
                + (-901 / 9009 * eulerlog[1]
                   + 6277796663889319 / 28992082264104960) * x**3,
        (7, 7): -32358125 / 20986602 * x**2
                + (-11948 / 3315 * eulerlog[7]
                   + 66555794049401803 / 3856993267327200) * x**3,
        (7, 6): -195441224 / 171390583 * x**2,
        (7, 5): -17354227 / 20986602 * x**2
                + (-59740 / 32487 * eulerlog[5]
                   + 192862646381533 / 22039961527584) * x**3,
        (7, 4): -2995755988 / 4627545741 * x**2,
        (7, 3): -7804375 / 20986602 * x**2
                + (-35844 / 54145 * eulerlog[3]
                   + 1321461327981547 / 428554807480800) * x**3,
        (7, 2): -1625746984 / 4627545741 * x**2,
        (7, 1): -1055091 / 6995534 * x**2
                + (-11948 / 162435 * eulerlog[1]
                   + 142228318411021 / 550999038189600) * x**3,
        # Appendix D27--D33 print these 2PN coefficients as decimals.
        (8, 8): -1.5337092502821381 * x**2,
        (8, 7): -1.175404252991305 * x**2,
        (8, 6): -0.9061610303170207 * x**2,
        (8, 5): -0.7220789990670207 * x**2,
        (8, 4): -0.47652059150068155 * x**2,
        (8, 3): -0.4196774909106648 * x**2,
        (8, 2): -0.2261796441029474 * x**2,
        (8, 1): -0.26842133517043704 * x**2,
    }
    for mode, addition in additions.items():
        rho[mode] = rho[mode] + addition
    return rho


def _residual_phase_taylor(mode: Mode, y: Tensor, nu: Tensor) -> Tensor:
    """Updated y-based residual phases from 1212.4357 Eqs. (D35)--(D44)."""
    v = torch.sqrt(y)
    v3, v5, v6, v9 = v**3, v**5, v**6, v**9
    pi = math.pi
    if mode == (2, 2):
        return (7 / 3 * v3 - 24 * nu * v5 + 428 / 105 * pi * v6
                + (30995 / 1134 * nu + 962 / 135 * nu**2) * v**7
                + (1712 / 315 * pi**2 - 2203 / 81) * v9)
    if mode == (2, 1):
        return 2 / 3 * v3 - 493 / 4 * nu * v5 + 107 / 105 * pi * v6 + (214 / 315 * pi**2 - 272 / 81) * v9
    if mode == (3, 3):
        return 13 / 10 * v3 - 80897 / 2430 * nu * v5 + 39 / 7 * pi * v6 + (78 / 7 * pi**2 - 227827 / 3000) * v9
    if mode == (3, 2):
        return ((10 + 33 * nu) / (15 * (1 - 3 * nu)) * v3
                + 52 / 21 * pi * v6 + (208 / 63 * pi**2 - 9112 / 405) * v9)
    if mode == (3, 1):
        return 13 / 30 * v3 - 17 / 10 * nu * v5 + 13 / 21 * pi * v6 + (26 / 63 * pi**2 + 227827 / 81000) * v9
    if mode == (4, 4):
        return ((112 + 219 * nu) / (120 * (1 - 3 * nu)) * v3
                + 25136 / 3465 * pi * v6 + (201088 / 10395 * pi**2 - 55144 / 375) * v9)
    if mode == (4, 3):
        return ((486 + 4961 * nu) / (810 * (1 - 2 * nu)) * v3
                + 1571 / 385 * pi * v6)
    if mode == (4, 2):
        return (7 * (1 + 6 * nu) / (15 * (1 - 3 * nu)) * v3
                + 6284 / 3465 * pi * v6 + (25136 / 10395 * pi**2 - 6893 / 375) * v9)
    if mode == (4, 1):
        return ((2 + 507 * nu) / (10 * (1 - 2 * nu)) * v3
                + 1571 / 3465 * pi * v6)
    if mode == (5, 5):
        return (96875 + 857528 * nu) / (131250 * (1 - 2 * nu)) * v3
    raise KeyError(mode)


def _residual_phase(mode: Mode, y: Tensor, nu: Tensor) -> Tensor:
    """Updated finite-mass Padé/Taylor phase choice from 1212.4357."""
    if mode not in PUBLISHED_PHASE_MODES:
        raise KeyError(mode)
    taylor = _residual_phase_taylor(mode, y, nu)
    v = torch.sqrt(y)
    v3 = v**3
    # At nu=0 the paper retains the highest known test-mass Taylor terms;
    # positive-nu Padés omit those terms as stated immediately below Eq. D35.
    test_mass = taylor
    if mode in {(3, 2), (4, 2), (4, 4)}:
        # The bold v^9 terms printed in D38, D40, and D42 are test-mass
        # terms.  The paragraph before D35 specifies that the finite-mass
        # representations stop before those terms; retain them only on the
        # nu=0 branch.  Keep this as an explicit truncation instead of
        # silently treating the test-mass series as finite-mass information.
        highest = {
            (3, 2): 208 / 63 * math.pi**2 - 9112 / 405,
            (4, 2): 25136 / 10395 * math.pi**2 - 6893 / 375,
            (4, 4): 201088 / 10395 * math.pi**2 - 55144 / 375,
        }[mode] * v**9
        taylor = torch.where(nu == 0, taylor, taylor - highest)
        test_mass = taylor
    nu_safe = torch.where(nu == 0, torch.ones_like(nu), nu)
    pi = math.pi
    if mode == (2, 2):
        numerator = (808920 * nu_safe * pi * v + 137388 * pi**2 * v**2
                     + 35 * nu_safe**2 * (136080 + (154975 - 1359276 * nu_safe) * v**2))
        denominator = (808920 * nu_safe * pi * v + 137388 * pi**2 * v**2
                       + 35 * nu_safe**2 * (136080 + (154975 + 40404 * nu_safe) * v**2))
        pade = 7 / 3 * v3 * numerator / denominator
    elif mode == (2, 1):
        numerator = 69020 * nu_safe + 5992 * pi * v
        denominator = 5992 * pi * v + 2456 * nu_safe * (28 + 493 * nu_safe * v**2)
        pade = 2 / 3 * v3 * numerator / denominator
    elif mode == (3, 3):
        numerator = 1 + 94770 * pi * v / (566279 * nu_safe)
        denominator = numerator + 80897 * nu_safe * v**2 / 3159
        pade = 13 / 10 * v3 * numerator / denominator
    elif mode == (3, 1):
        numerator = 4641 * nu_safe + 1690 * pi * v
        denominator = numerator + 18207 * v**2 * nu_safe**2
        pade = 13 / 30 * v3 * numerator / denominator
    else:
        return taylor
    return torch.where(nu == 0, test_mass, pade)


def _newtonian_prefactor(
    prepared: _Prepared, ell: int, m: int, rho: Tensor, *,
    legacy_l5_odd: bool = False, correct_l5_odd: bool | None = None,
) -> Tensor:
    """Return the Newtonian factor and source for one positive-m mode.

    The general mass polynomial is the DIN expression
    ``X2**(ell+epsilon-1) + (-1)**m*X1**(ell+epsilon-1)``.  For odd-m,
    ell=5 this is already ``(X2-X1)*(1-2*nu)``.  ``legacy_l5_odd`` is an
    opt-in compatibility switch for the older representation that applied
    an additional ``sqrt(1-4*nu)``.  ``correct_l5_odd`` is retained only as
    a keyword compatibility alias; its default is deliberately ``None`` so
    the clean default cannot reintroduce the duplicate factor.
    """
    if correct_l5_odd is not None:
        legacy_l5_odd = bool(correct_l5_odd)
    epsilon = (ell + m) % 2
    p = ell + epsilon - 1
    c = prepared.m2 / (prepared.m1 + prepared.m2)
    c = c**p + (-1)**m * (prepared.m1 / (prepared.m1 + prepared.m2))**p
    # 1806.01772v2, Sec. II.B.1 describes this factor in an older
    # representation.  It is a legacy opt-in here because the general
    # polynomial above already contains X_AB for odd modes.
    if legacy_l5_odd and ell == 5 and m % 2:
        c = c * torch.sqrt(torch.clamp(1 - 4 * prepared.nu, min=0))
    source = prepared.Heff if epsilon == 0 else prepared.pphi / (prepared.rOmega * prepared.vphi)
    return (_newtonian_constant(ell, m) * prepared.nu
            * prepared.vphi**(ell + epsilon) * source * c * rho**ell)


def teob_tail_gauge_factor(m: int, real_energy: object, orbital_frequency: object) -> Tensor:
    """Return the TEOB tail-gauge factor from 1212.4357 Eq. (23).

    The shared factorized-mode helper evaluates the tail phase with ``r0=2M``.
    The TEOB convention uses ``r0=2GM/sqrt(e)``, which changes a positive-m
    mode by

    ``exp(-i*m*H_real*Omega)``.

    This factor is unit modulus, so it changes only the mode phase.  It is
    kept in the TEOB layer rather than changing the shared factorized helper,
    whose gauge is also used by the clean SEOB path.
    """
    if type(m) is not int or m <= 0:
        raise ValueError("m must be a positive integer")
    reference = _reference_tensor(real_energy, orbital_frequency)
    h_t = _as_tensor(real_energy, reference, "real_energy")
    omega_t = _as_tensor(orbital_frequency, reference, "orbital_frequency")
    h_t, omega_t = torch.broadcast_tensors(h_t, omega_t)
    _finite("real_energy", h_t)
    _finite("orbital_frequency", omega_t)
    _positive("real_energy", h_t)
    _positive("orbital_frequency", omega_t)
    return torch.polar(torch.ones_like(h_t), -m * h_t * omega_t)


def _positive_mode(
    prepared: _Prepared, ell: int, m: int, *, residual_phase: ResidualPhase,
    legacy_l5_odd: bool = False,
) -> Tensor:
    x = prepared.vphi.square()
    rho = _complete_rho_table(x, prepared.nu)[ell, m]
    h = _newtonian_prefactor(prepared, ell, m, rho, legacy_l5_odd=legacy_l5_odd)
    tail_amp = _tail_abs_squared(ell, m * prepared.H * prepared.Omega).sqrt()
    phase = _tail_phase(ell, m, prepared.Omega, prepared.H) - m * prepared.phi
    if (ell, m) in PUBLISHED_PHASE_MODES:
        phase = phase + _residual_phase((ell, m), prepared.y, prepared.nu)
    elif residual_phase == "published":
        raise NotImplementedError(
            "Updated finite-mass residual phase is published only for the requested subset; "
            "use residual_phase='zero-unpublished' for the remaining amplitudes"
        )
    result = h * torch.polar(tail_amp, phase)
    if ell == 2 and m == 2:
        result = result * prepared.nqc22
    # 1212.4357v2 Eq. (23): the TEOB tail uses r0=2GM/sqrt(e), while the
    # shared DIN helper above is deliberately retained at its r0=2M gauge.
    return result * teob_tail_gauge_factor(m, prepared.H, prepared.Omega)


def _mode_power(h: Tensor) -> Tensor:
    return h.real.square() + h.imag.square() if h.is_complex() else h.square()


def _angular_momentum_flux(modes: dict[Mode, Tensor], Omega: Tensor) -> Tensor:
    """Positive-m angular-momentum flux, exactly 1406.6913 Eq. (75)."""
    return sum(m**2 * Omega * _mode_power(h) for (_, m), h in modes.items()) / (8 * math.pi)


def _energy_flux(modes: dict[Mode, Tensor], Omega: Tensor) -> Tensor:
    """Positive-m energy flux corresponding to the same modes."""
    return sum((m * Omega)**2 * _mode_power(h) for (_, m), h in modes.items()) / (8 * math.pi)


def _horizon_rho(x: Tensor, nu: Tensor, mode: Mode) -> Tensor:
    """Return the 4PN horizon residual amplitude from 1212.4357 Table I.

    Only the nonspinning quadrupolar entries used by Eq. (44) are supplied.
    The coefficients are the printed Table-I values; the denominator in the
    22 1PN coefficient is kept as part of that published expression.
    """
    if mode == (2, 2):
        c1 = (4 - 21 * nu + 27 * nu**2 - 8 * nu**3) / (4 * (1 - 4 * nu + 2 * nu**2))
        c2, c3, c4 = 4.78752, 26.760136, 43.861478
    elif mode == (2, 1):
        c1, c2, c3, c4 = 0.58121, 1.01059, 7.955729, 1.650228
    else:
        raise KeyError(mode)
    return 1 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4


def _horizon_flux_ratio(prepared: _Prepared) -> Tensor:
    """Return the dimensionless horizon correction in 1212.4357 Eq. (37).

    Equation (44) gives the part before its explicit
    ``1 - 4*nu + 2*nu**2`` prefactor.  ``x`` is ``vphi**2`` and the source
    factors are the nonspinning Eq. (44) choices.  The returned ratio is
    normalized to the same Newtonian quadrupole units as the asymptotic
    angular flux assembled from the waveform modes.
    """
    x = prepared.vphi.square()
    rho22 = _horizon_rho(x, prepared.nu, (2, 2))
    rho21 = _horizon_rho(x, prepared.nu, (2, 1))
    source_even = prepared.Heff
    source_odd = prepared.pphi / (prepared.rOmega * prepared.vphi)
    prefactor = 1 - 4 * prepared.nu + 2 * prepared.nu**2
    return prefactor * (
        x**4 * source_even**2 * rho22**4
        + x**5 * source_odd**2 * rho21**4
    )


def _horizon_angular_flux(prepared: _Prepared) -> Tensor:
    """Return the positive physical angular flux absorbed by the horizons.

    The waveform sum returns a physical flux proportional to ``nu**2``.
    Multiplying the Eq. (37) ratio by the same Newtonian scale therefore
    inserts ``nu**2`` exactly once; the corresponding reduced torque is
    obtained by dividing by ``nu`` in ``radiation_reaction_torque``.
    """
    scale = (32 / 5) * prepared.nu**2 * prepared.rOmega**4 * prepared.Omega**5
    return scale * _horizon_flux_ratio(prepared)


class TEOBRadiationClean(nn.Module):
    """Explicit-input nonspinning TEOB radiation through ``lmax <= 8``.

    ``mode`` and ``modes`` take ``(r, Omega, phi, prstar, pphi, Heff, H,
    Hcirc, Aprime, m1, m2)``.  ``Hcirc`` is the caller's total EOB energy at
    ``p_r*=0`` and current ``p_phi``; it is used only in the plunge-aware
    ``r_Omega``.  ``H`` is the total EOB energy used by ``y=(H*Omega)**(2/3)``.

    ``flux`` evaluates the positive-m modes through ``lmax`` and uses 1/(8*pi),
    as in 1406.6913 Eq. (75).  Horizon absorption is available explicitly
    through ``horizon_flux`` and ``include_horizon=True`` on the flux methods;
    the latter adds the nonspinning Eq. (37) correction in the same physical
    angular-flux units.  ``modes`` defaults to the explicit
    ``zero-unpublished`` phase policy so that all 35 amplitudes can be
    requested; use ``residual_phase='published'`` to reject modes for which
    1212.4357 does not provide a residual phase.
    """

    def __init__(self, lmax: int = 8, *, residual_phase: ResidualPhase = "zero-unpublished",
                 legacy_l5_odd: bool = False, correct_l5_odd: bool | None = None):
        super().__init__()
        if type(lmax) is not int or not 2 <= lmax <= 8:
            raise ValueError("lmax must be an integer from 2 through 8")
        if residual_phase not in {"published", "zero-unpublished"}:
            raise ValueError("residual_phase must be 'published' or 'zero-unpublished'")
        self.lmax = lmax
        self.residual_phase = residual_phase
        if correct_l5_odd is not None:
            # Compatibility with the pre-contract keyword.  It now selects
            # the explicitly legacy duplicate factor rather than a default
            # correction.
            legacy_l5_odd = bool(correct_l5_odd)
        self.legacy_l5_odd = bool(legacy_l5_odd)
        self.mode_indices = tuple((ell, m) for ell in range(2, lmax + 1)
                                  for m in range(1, ell + 1))

    @property
    def correct_l5_odd(self) -> bool:
        """Deprecated compatibility view of ``legacy_l5_odd``."""
        return self.legacy_l5_odd

    def _prep(self, r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2, *, nqc22=1.0) -> _Prepared:
        return _prepare(r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2, nqc22)

    def mode(
        self, ell: int, m: int, r, Omega, phi, prstar, pphi, Heff, H, Hcirc,
        Aprime, m1, m2, *, nqc22=1.0,
    ) -> Tensor:
        """Return one signed-m complex mode, with conjugate symmetry."""
        if type(ell) is not int or type(m) is not int or not (
            2 <= ell <= self.lmax and 1 <= abs(m) <= ell
        ):
            raise ValueError("requested mode must have 2<=ell<=lmax and 1<=abs(m)<=ell")
        prepared = self._prep(r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2,
                              nqc22=nqc22)
        positive = _positive_mode(
            prepared, ell, abs(m), residual_phase=self.residual_phase,
            legacy_l5_odd=self.legacy_l5_odd,
        )
        return positive if m > 0 else (-1)**ell * positive.conj()

    def amplitudes(
        self, r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2, *, nqc22=1.0,
    ) -> dict[Mode, Tensor]:
        """Return ``|R h_lm/M|`` for every positive-m mode through lmax."""
        prepared = self._prep(r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2,
                              nqc22=nqc22)
        result = {}
        rho_table = _complete_rho_table(prepared.vphi.square(), prepared.nu)
        for ell, m in self.mode_indices:
            h = _newtonian_prefactor(
                prepared, ell, m, rho_table[ell, m], legacy_l5_odd=self.legacy_l5_odd
            )
            if ell == 2 and m == 2:
                h = h * prepared.nqc22
            result[ell, m] = h.abs() * _tail_abs_squared(
                ell, m * prepared.H * prepared.Omega
            ).sqrt()
        return result

    def modes(
        self, r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2, *, nqc22=1.0,
    ) -> dict[Mode, Tensor]:
        """Return all positive-m complex modes through lmax."""
        prepared = self._prep(r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2,
                              nqc22=nqc22)
        return {
            (ell, m): _positive_mode(
                prepared, ell, m, residual_phase=self.residual_phase,
                legacy_l5_odd=self.legacy_l5_odd,
            )
            for ell, m in self.mode_indices
        }

    def _infinity_modes(self, prepared: _Prepared) -> dict[Mode, Tensor]:
        """Build the positive-m modes used by both asymptotic fluxes."""
        rho_table = _complete_rho_table(prepared.vphi.square(), prepared.nu)
        modes = {}
        for ell, m in self.mode_indices:
            h = _newtonian_prefactor(
                prepared, ell, m, rho_table[ell, m], legacy_l5_odd=self.legacy_l5_odd
            )
            if ell == 2 and m == 2:
                h = h * prepared.nqc22
            modes[ell, m] = h * _tail_abs_squared(
                ell, m * prepared.H * prepared.Omega
            ).sqrt()
        return modes

    def _horizon_flux(self, prepared: _Prepared) -> Tensor:
        """Build the physical horizon angular flux for a prepared state."""
        return _horizon_angular_flux(prepared)

    def flux(
        self, r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2, *,
        nqc22=1.0, include_horizon: bool = False,
    ) -> Tensor:
        """Return emitted angular flux, optionally including Eq. (37) absorption.

        The default is the asymptotic waveform sum.  Setting
        ``include_horizon=True`` adds the nonspinning horizon contribution in
        the same physical units.  ``horizon_flux`` exposes that term alone.
        """
        prepared = self._prep(r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2,
                              nqc22=nqc22)
        result = _angular_momentum_flux(self._infinity_modes(prepared), prepared.Omega)
        return result + self._horizon_flux(prepared) if include_horizon else result

    def horizon_flux(
        self, r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2,
    ) -> Tensor:
        """Return the positive nonspinning horizon angular flux from Eq. (37).

        The 1212.4357 prescription contains only the 22 and 21 horizon
        multipoles and is restricted here to the explicitly nonspinning
        vacuum inputs represented by this class.
        """
        prepared = self._prep(r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2)
        return self._horizon_flux(prepared)

    def total_flux(
        self, r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2, *, nqc22=1.0,
    ) -> Tensor:
        """Return asymptotic plus horizon angular flux."""
        return self.flux(
            r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2,
            nqc22=nqc22, include_horizon=True,
        )

    def energy_flux(
        self, r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2, *,
        nqc22=1.0, include_horizon: bool = False,
    ) -> Tensor:
        """Return emitted energy flux, optionally including horizon absorption."""
        prepared = self._prep(r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2,
                              nqc22=nqc22)
        result = _energy_flux(self._infinity_modes(prepared), prepared.Omega)
        if include_horizon:
            result = result + prepared.Omega * self._horizon_flux(prepared)
        return result

    def radiation_reaction_torque(
        self, r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2, *,
        nqc22=1.0, include_horizon: bool = True,
    ) -> Tensor:
        """Return the reduced azimuthal force ``Fhat_phi`` from Eq. (36).

        The torque is the negative emitted angular flux divided by ``nu``;
        this is the minimal TEOB radiation-reaction integration and leaves
        the radial force zero.  Horizon absorption is enabled by default for
        this explicitly nonspinning vacuum component and can be disabled for
        an asymptotic-only comparison.
        """
        prepared = self._prep(r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2,
                              nqc22=nqc22)
        flux = _angular_momentum_flux(self._infinity_modes(prepared), prepared.Omega)
        if include_horizon:
            flux = flux + self._horizon_flux(prepared)
        return -flux / prepared.nu

    def radiation_reaction_forces(
        self, r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2, *,
        nqc22=1.0, include_horizon: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(Fhat_rstar, Fhat_phi)`` with the published zero radial force."""
        torque = self.radiation_reaction_torque(
            r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2,
            nqc22=nqc22, include_horizon=include_horizon,
        )
        return torch.zeros_like(torque), torque

    def forward(
        self, r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2, *, nqc22=1.0,
    ) -> dict[Mode, Tensor]:
        return self.modes(r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2,
                          nqc22=nqc22)


def teob22_mode(
    r: object, Omega: object, phi: object, prstar: object, pphi: object,
    Heff: object, H: object, Hcirc: object, Aprime: object, m1: object, m2: object,
    *, nqc22: object = 1.0,
) -> Tensor:
    """Convenience function for the published, updated nonspinning 22 mode."""
    return TEOBRadiationClean(2, residual_phase="published").mode(
        2, 2, r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2,
        nqc22=nqc22,
    )


def teob_modes(
    r: object, Omega: object, phi: object, prstar: object, pphi: object,
    Heff: object, H: object, Hcirc: object, Aprime: object, m1: object, m2: object,
    *, lmax: int = 8, residual_phase: ResidualPhase = "zero-unpublished",
    nqc22: object = 1.0, legacy_l5_odd: bool = False,
) -> dict[Mode, Tensor]:
    """Convenience function returning positive-m modes through ``lmax``."""
    return TEOBRadiationClean(
        lmax, residual_phase=residual_phase, legacy_l5_odd=legacy_l5_odd
    ).modes(
        r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2,
        nqc22=nqc22,
    )


def teob_flux(
    r: object, Omega: object, phi: object, prstar: object, pphi: object,
    Heff: object, H: object, Hcirc: object, Aprime: object, m1: object, m2: object,
    *, lmax: int = 8, nqc22: object = 1.0, include_horizon: bool = False,
    legacy_l5_odd: bool = False,
) -> Tensor:
    """Convenience function for the positive emitted angular flux.

    Set ``include_horizon=True`` to add the nonspinning Eq. (37) absorption
    correction; by default this remains the asymptotic Eq. (75) sum.
    """
    return TEOBRadiationClean(lmax, legacy_l5_odd=legacy_l5_odd).flux(
        r, Omega, phi, prstar, pphi, Heff, H, Hcirc, Aprime, m1, m2,
        nqc22=nqc22, include_horizon=include_horizon,
    )


__all__ = [
    "TEOBRadiationClean",
    "non_keplerian_radius",
    "non_keplerian_velocity",
    "teob_r_omega",
    "teob_non_keplerian_radius",
    "teob_vphi",
    "teob_tail_gauge_factor",
    "teob22_mode",
    "teob_modes",
    "teob_flux",
]
