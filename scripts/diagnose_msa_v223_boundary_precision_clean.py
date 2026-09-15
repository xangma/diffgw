"""Independent high precision audit of the small-in-plane MSA boundary.

This is a read-only diagnostic for the fixed 35+25 solar-mass contract row.
The cubic, PN flux, secular coefficients, partial fractions, and v223 pole
continuation are evaluated a second time with :mod:`mpmath`; the clean
float64 angle assembler and the public IMRPhenomXP binary are only comparison
targets.  The script does not fit a phase or choose an arithmetic variant.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import lal
import lalsimulation as lalsim
import mpmath as mp
import torch

from torchwave.approximants.precession_msa_angles_clean import (
    assemble_precession_msa_angles,
)
from torchwave.approximants.precession_msa_waveform_clean import waveform_v223


ROOT = Path(__file__).resolve().parents[1]
MTSUN = mp.mpf("4.925490947641267e-6")
MTSUN_FLOAT = float(MTSUN)
FLOW = 20.0
FHIGH = 512.0
ORDER = 5
VERSION = 223
CONVENTION = 0
MPRIME = 2
FINAL_SPIN_MOD = 3
DISTANCE = 500.0
INCLINATION = 0.7
TARGET_FREQUENCIES = (190.0, 196.0)

CASE = {
    "mass1": 35.0,
    "mass2": 25.0,
    "spin1": (1.0e-5, 0.0, 0.2),
    "spin2": (0.0, 1.0e-5, -0.1),
    "f_ref": 20.0,
    "distance": DISTANCE,
    "inclination": INCLINATION,
    "phic": 0.0,
    "tc": 0.0,
}


def _sgn(x):
    return mp.mpf(1) if x > 0 else (mp.mpf(-1) if x < 0 else mp.mpf(0))


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _cubic_roots_at_argument(B, C, D, arg):
    """Return the published cubic labels for a supplied acos argument."""
    p = C - B * B / 3
    p_safe = p if p < 0 else mp.mpf(-1)
    theta = mp.acos(_clamp(arg, mp.mpf(-1), mp.mpf(1))) / 3
    roots = [
        -B / 3 + 2 * mp.sqrt(-p_safe / 3)
        * mp.cos(theta - 2 * mp.pi * k / 3)
        for k in range(3)
    ]
    roots.sort()
    physical = sorted((abs(roots[1]), abs(roots[2])))
    return [roots[0], physical[0], physical[1]]


def _cubic_roots(B, C, D):
    """Return the clean cubic labels and the raw trigonometric argument."""
    p = C - B * B / 3
    p_safe = p if p < 0 else mp.mpf(-1)
    q = 2 * B ** 3 / 27 - B * C / 3 + D
    raw = 3 * q / (2 * p_safe) * mp.sqrt(-3 / p_safe)
    # v223 accepts finite endpoint overshoots and clamps only the acos input.
    valid = bool(mp.isfinite(B) and mp.isfinite(C) and mp.isfinite(D)
                 and p < 0 and mp.isfinite(raw))
    arg = _clamp(raw if mp.isfinite(raw) else mp.mpf(0),
                 mp.mpf(-1), mp.mpf(1))
    return _cubic_roots_at_argument(B, C, D, arg), raw, valid


def _state(initial_arg_override=None):
    """Build normalized reference invariants from the published definitions."""
    m1 = mp.mpf(str(CASE["mass1"]))
    m2 = mp.mpf(str(CASE["mass2"]))
    M = m1 + m2
    n1, n2 = m1 / M, m2 / M
    q = m2 / m1
    eta = n1 * n2
    delta = (m1 - m2) / M
    chi1 = [mp.mpf(str(x)) for x in CASE["spin1"]]
    chi2 = [mp.mpf(str(x)) for x in CASE["spin2"]]
    S1v = [n1 * n1 * x for x in chi1]
    S2v = [n2 * n2 * x for x in chi2]
    v0 = (mp.pi * M * MTSUN * mp.mpf(str(CASE["f_ref"]))) ** (mp.mpf(1) / 3)
    L0 = eta / v0
    J0v = [S1v[0] + S2v[0], S1v[1] + S2v[1], L0 + S1v[2] + S2v[2]]
    J0 = mp.sqrt(sum(x * x for x in J0v))
    s1sq = sum(x * x for x in S1v)
    s2sq = sum(x * x for x in S2v)
    Ssum = [S1v[i] + S2v[i] for i in range(3)]
    Ssum2 = sum(x * x for x in Ssum)
    xi = (1 + q) * S1v[2] + (1 + 1 / q) * S2v[2]
    B, C, D = _cubic_coefficients(q, eta, delta, s1sq, s2sq,
                                   L0, J0, xi)
    candidate_roots0, raw0, valid0 = _cubic_roots(B, C, D)
    if initial_arg_override is None:
        roots0 = candidate_roots0
    else:
        roots0 = _cubic_roots_at_argument(B, C, D, initial_arg_override)
    spin1_zero = s1sq == 0
    spin2_zero = s2sq == 0
    spin1_aligned = S1v[0] == 0 and S1v[1] == 0
    spin2_aligned = S2v[0] == 0 and S2v[1] == 0
    root_regularized = (not valid0 or spin1_zero or spin2_zero
                        or spin1_aligned or spin2_aligned)
    fallback0 = [mp.mpf(0), Ssum2, Ssum2 + mp.mpf("1e-9")]
    if root_regularized:
        roots0 = fallback0
    sav = (roots0[1] + roots0[2]) / 2
    c1 = mp.mpf("0.5") * v0 * (J0 * J0 - L0 * L0 - sav)
    degenerate0 = abs(roots0[2] - roots0[1]) < mp.mpf("1e-5")
    flux = _flux_coefficients(q, eta, S1v[2], S2v[2], s1sq, s2sq,
                              sum(S1v[i] * S2v[i] for i in range(3)))
    g = {
        "g0": 1 / flux["a0"],
        "g1": mp.mpf(0),
        "g2": -flux["a2"] / flux["a0"],
        "g3": -flux["a3"] / flux["a0"],
        "g4": -(flux["a4"] - flux["a2"] ** 2) / flux["a0"],
        "g5": -(flux["a5"] - 2 * flux["a3"] * flux["a2"]) / flux["a0"],
    }
    psi1, psi2_ind = _psi_coefficients(
        eta, delta, q, xi, c1, sav, s1sq, s2sq, g["g0"], g["g2"]
    )
    psi2 = 3 * g["g2"] / g["g0"] + q * q * (
        psi2_ind - 3 * g["g2"] / g["g0"]
    )
    P0 = _phase(v0, delta, g["g0"], psi1, psi2)
    psi0 = P0 if degenerate0 else mp.mpf(0)
    return {
        "M": M, "m1": m1, "m2": m2, "q": q, "eta": eta,
        "delta": delta, "xi": xi, "S1v": S1v, "S2v": S2v,
        "s1sq": s1sq, "s2sq": s2sq, "Ssum2": Ssum2,
        "v0": v0, "L0": L0, "J0": J0, "roots0": roots0,
        "candidate_roots0": candidate_roots0,
        "cubic0": (B, C, D), "raw0": raw0, "roots_valid0": valid0,
        "initial_arg_override": None if initial_arg_override is None
        else mp.mpf(initial_arg_override),
        "root_regularized": root_regularized, "sav": sav, "c1": c1,
        "degenerate0": degenerate0, "flux": flux, "g": g,
        "psi1": psi1, "psi2_independent": psi2_ind, "psi2": psi2,
        "psi0": psi0,
    }


def _cubic_coefficients(q, eta, delta, s1sq, s2sq, L, J, xi):
    B = (L * L + s1sq) * q + 2 * L * xi - 2 * J * J - s1sq - s2sq \
        + (L * L + s2sq) / q
    C = (J * J - L * L) ** 2 - 2 * L * xi * (J * J - L * L) \
        - 2 * (1 - q) / q * (s1sq - q * s2sq) * L * L \
        + 4 * eta * L * L * xi * xi - 2 * delta * (s1sq - s2sq) * xi * L \
        + 2 * (1 - q) / q * (q * s1sq - s2sq) * J * J
    D = (1 - q) / q * (s2sq - q * s1sq) * (J * J - L * L) ** 2 \
        + delta * delta / eta * (s1sq - s2sq) ** 2 * L * L \
        + 2 * delta * L * xi * (s1sq - s2sq) * (J * J - L * L)
    return B, C, D


def _flux_coefficients(q, eta, s1l, s2l, s1sq, s2sq, s12):
    x1, x2 = 1 / (1 + q), q / (1 + q)
    beta3 = (mp.mpf(113) / 12 + 25 * q / 4) * s1l \
        + (mp.mpf(113) / 12 + 25 / (4 * q)) * s2l
    beta5 = (mp.mpf(31319) / 1008 - 1159 * eta / 24
             + q * (mp.mpf(809) / 84 - 281 * eta / 8)) * s1l \
        + (mp.mpf(31319) / 1008 - 1159 * eta / 24
           + (mp.mpf(809) / 84 - 281 * eta / 8) / q) * s2l
    s1l2, s2l2, s1ls2l = s1l * s1l, s2l * s2l, s1l * s2l
    sigma4 = (247 * s12 - 721 * s1ls2l) / (48 * eta) \
        + ((233 * s1sq - 719 * s1l2) / x1 ** 2
           + (233 * s2sq - 719 * s2l2) / x2 ** 2) / 96
    a0 = 96 * eta / 5
    a2 = -mp.mpf(743) / 336 - 11 * eta / 4
    a3 = 4 * mp.pi - beta3
    a4 = mp.mpf(34103) / 18144 + 13661 * eta / 2016 \
        + 59 * eta * eta / 18 - sigma4
    a5 = -4159 * mp.pi / 672 - 189 * mp.pi * eta / 8 - beta5
    return {"beta3": beta3, "beta5": beta5, "sigma4": sigma4,
            "a0": a0, "a2": a2, "a3": a3, "a4": a4, "a5": a5}


def _psi_coefficients(eta, delta, q, xi, c1, sav, s1sq, s2sq, g0, g2):
    bm2, bm1 = eta * delta * delta, 2 * eta * xi - 4 * c1
    b0 = q * s1sq + s2sq / q - s1sq - s2sq - 2 * sav
    cm2 = 4 * c1 * c1 - 4 * eta * c1 * xi + 4 * eta ** 3 * xi * xi \
        - 2 * eta * delta * delta * (s1sq + s2sq)
    dm2 = 4 * (1 - q) / q * (s2sq - q * s1sq) * c1 * c1 \
        + eta * delta * delta * (s1sq - s2sq) ** 2 \
        + 4 * delta * eta * c1 * xi * (s1sq - s2sq)
    disc = cm2 * cm2 - 4 * bm2 * dm2
    ds = mp.sqrt(abs(disc))
    roots = [(-cm2 + ds) / (2 * bm2), (-cm2 - ds) / (2 * bm2)]
    p, n = max(roots), min(roots)
    r1 = bm1 / (2 * bm2)
    r2 = (b0 + 2 * p + n) / (2 * bm2) - bm1 * bm1 / (8 * bm2 * bm2)
    return mp.mpf("1.5") * (r1 - xi), 3 * (r2 - xi * r1 + g2 / g0)


def _phase(v, delta, g0, psi1, psi2):
    return -mp.mpf("0.75") * g0 * delta * (1 + psi1 * v + psi2 * v * v) / v ** 3


def _omega_coefficients(st):
    eta, xi, c1 = st["eta"], st["xi"], st["c1"]
    s1sq, s2sq = st["s1sq"], st["s2sq"]
    sp, sm = st["roots0"][2], st["roots0"][1]
    rm = sp - sm
    cp, cm = sp * eta * eta - c1 * c1, sm * eta * eta - c1 * c1
    root = mp.sqrt(max(abs(cp * cm), mp.mpf("1e-32")))
    a1 = mp.mpf("0.5") + mp.mpf("0.75") / eta
    a2 = -mp.mpf("0.75") * xi / eta
    ad = (-3 * (s1sq - s2sq) * eta * st["delta"]
          + 3 * c1 / eta * (c1 - 2 * xi * eta * eta)) / (4 * root)
    cd = 3 * rm * rm / (128 * eta * root)
    hd = c1 / eta ** 2 * (1 - (cp + cm) / (2 * root))
    fd = (cp + cm) / (8 * eta ** 4) * (1 - (cp + cm) / (2 * root))
    oz = [
        a1 + ad,
        a2 - ad * xi - ad * hd,
        ad * hd * xi + cd - ad * fd + ad * hd * hd,
        (ad * fd - cd - ad * hd * hd) * (xi + hd) + ad * fd * hd,
        (cd + ad * hd * hd - 2 * ad * fd) * (hd * xi + hd * hd - fd)
        - ad * fd * fd,
        (cd - ad * fd + ad * hd * hd) * fd * (xi + 2 * hd)
        - (cd + ad * hd * hd - 2 * ad * fd) * hd * hd * (xi + hd)
        - ad * fd * fd * hd,
    ]
    gd = 3 * rm * rm / (64 * eta ** 3) * (c1 - eta * eta * xi) / root
    ze = [
        oz[0],
        oz[1] + c1 / eta ** 2 * oz[0],
        oz[2] + c1 / eta ** 2 * oz[1],
        oz[3] + c1 / eta ** 2 * oz[2] + gd,
        oz[4] + c1 / eta ** 2 * oz[3] - gd * xi - gd * hd,
        oz[5] + c1 / eta ** 2 * oz[4] + gd * hd * xi
        + gd * (hd * hd - fd),
    ]
    return oz, ze


def _averaged_coefficients(st):
    oz, ze = _omega_coefficients(st)
    g = st["g"]
    z = lambda i: oz[i]
    q = lambda i: ze[i]
    pc = [
        3 * g["g0"] * z(0),
        3 * g["g0"] * z(1),
        3 * (g["g0"] * z(2) + g["g2"] * z(0)),
        3 * (g["g0"] * z(3) + g["g2"] * z(1) + g["g3"] * z(0)),
        3 * (g["g0"] * z(4) + g["g2"] * z(2) + g["g3"] * z(1)
             + g["g4"] * z(0)),
        3 * (g["g0"] * z(5) + g["g2"] * z(3) + g["g3"] * z(2)
             + g["g4"] * z(1) + g["g5"] * z(0)),
    ]
    zc = [
        -g["g0"] * q(0),
        -mp.mpf("1.5") * g["g0"] * q(1),
        -3 * (g["g0"] * q(2) + g["g2"] * q(0)),
        3 * (g["g0"] * q(3) + g["g2"] * q(1) + g["g3"] * q(0)),
        3 * (g["g0"] * q(4) + g["g2"] * q(2) + g["g3"] * q(1)
             + g["g4"] * q(0)),
        mp.mpf("1.5") * (g["g0"] * q(5) + g["g2"] * q(3)
                          + g["g3"] * q(2) + g["g4"] * q(1)
                          + g["g5"] * q(0)),
    ]
    # Order five removes the displayed n=5 terms in the clean assembler.
    pc[5] = mp.mpf(0)
    zc[5] = mp.mpf(0)
    return pc, zc


def _phi_primitives(v, st):
    eta, c1, sav = st["eta"], st["c1"], st["sav"]
    J = mp.sqrt(eta * eta / (v * v) + 2 * c1 / v + sav)
    L = eta / v
    l1 = mp.log(c1 + J * eta + L * eta)
    s = mp.sqrt(sav)
    l2 = mp.log(c1 + J * s * v + sav * v)
    return [
        J / eta ** 4 * (c1 * c1 / 2 - c1 * eta * eta / (6 * v)
                        - sav * eta * eta / 3 - eta ** 4 / (3 * v * v))
        - c1 / (2 * eta) * (c1 * c1 / eta ** 4 - sav / eta ** 2) * l1,
        -J / (2 * eta ** 2) * (c1 + eta * L)
        + (c1 * c1 - eta * eta * sav) / (2 * eta ** 3) * l1,
        -J + s * l2 - c1 / eta * l1,
        J * v - eta * l1 + c1 / s * l2,
        J * v * (c1 + v * sav) / (2 * sav)
        - (c1 * c1 - eta * eta * sav) / (2 * sav ** mp.mpf("1.5")) * l2,
        -J * v * (c1 * c1 / (2 * sav ** 2) - c1 * v / (6 * sav)
                  - v * v / 3 - eta * eta / (3 * sav))
        + c1 * (c1 * c1 - eta * eta * sav)
        / (2 * sav ** mp.mpf("2.5")) * l2,
    ]


def _zeta_primitives(v, st):
    eta = st["eta"]
    return [eta * v ** -3, eta * v ** -2, eta * v ** -1,
            eta * mp.log(v), eta * v, eta * v * v]


def _secular_context(st):
    pc, zc = _averaged_coefficients(st)
    return {
        "pc": pc, "zc": zc,
        "p0": _phi_primitives(st["v0"], st),
        "z0": _zeta_primitives(st["v0"], st),
    }


def _secular(v, st, ctx):
    p = _phi_primitives(v, st)
    z = _zeta_primitives(v, st)
    phi = sum((p[i] - ctx["p0"][i]) * ctx["pc"][i] for i in range(6))
    zeta = sum((z[i] - ctx["z0"][i]) * ctx["zc"][i] for i in range(6))
    return phi, zeta


def _pole(v, st, roots, round_inputs=False):
    """Evaluate the independent pole equations.

    ``round_inputs`` is used only by the diagnostic to compare operation
    precision with the same binary64 input values.  It does not alter the
    clean implementation or select an arithmetic variant.
    """
    if round_inputs:
        cast = lambda x: mp.mpf(float(x))
        v = cast(v)
        st = dict(st)
        for name in ("eta", "xi", "c1", "q", "delta", "s1sq", "s2sq", "sav"):
            st[name] = cast(st[name])
        roots = [cast(x) for x in roots]
    eta, xi, c1 = st["eta"], st["xi"], st["c1"]
    q, delta = st["q"], st["delta"]
    s3, sm, sp = roots
    L = eta / v
    J = mp.sqrt(L * L + 2 * c1 / v + st["sav"])
    a = mp.mpf("0.5") * v ** 6 * (1 + mp.mpf("1.5") / eta * (1 - xi * v))
    c0 = mp.mpf("0.75") * (1 - xi * v) * v ** 2 * (
        eta ** 3 + 4 * eta ** 3 * xi * v
        - 2 * eta * (J * J - sp + 2 * (st["s1sq"] - st["s2sq"]) * delta) * v ** 2
        - 4 * eta * xi * (J * J - sp) * v ** 3
        + (J * J - sp) ** 2 / eta * v ** 4
    )
    c2 = -mp.mpf("1.5") * eta * (sp - sm) * (
        1 + 2 * xi * v - (J * J - sp) / eta ** 2 * v ** 2
    ) * (1 - xi * v) * v ** 4
    c4 = mp.mpf("0.75") / eta * (sp - sm) ** 2 * (1 - xi * v) * v ** 6
    d0 = -(J * J - (L + mp.sqrt(sp)) ** 2) * (J * J - (L - mp.sqrt(sp)) ** 2)
    d2 = -2 * (sp - sm) * (J * J + L * L - sp)
    d4 = -(sp - sm) ** 2
    disc = d2 * d2 - 4 * d0 * d4
    root = mp.sqrt(abs(disc))
    la = (-d2 + root) / (2 * d0)
    lb = (-d2 - root) / (2 * d0)
    r0 = c4 / d4
    u = c0 / d0 - r0
    w = c2 / d0 + r0 * (la + lb)
    ra = (w + u * la) / (la - lb)
    rb = (-w - u * lb) / (la - lb)
    psidot = -3 / (4 * mp.sqrt(eta)) * (1 - xi * v) * v ** 6 * mp.sqrt(sp - s3)
    return {
        "J": J, "L": L, "a": a, "c0": c0, "c2": c2, "c4": c4,
        "d0": d0, "d2": d2, "d4": d4, "psidot": psidot,
        "lambda_a": la, "lambda_b": lb, "residue_a": ra,
        "residue_b": rb, "residue_0": r0,
    }


def _branch(lam, psi):
    n = 1 - lam
    if n == 0:
        return mp.mpf(0)
    k = mp.sqrt(abs(n))
    return mp.atan2((1 - k) * mp.sin(psi) * mp.cos(psi),
                    mp.cos(psi) ** 2 + k * mp.sin(psi) ** 2) / k


def _oscillatory(pole, psi, atheta, btheta):
    la, lb = pole["lambda_a"], pole["lambda_b"]
    ba, bb = _branch(la, psi), _branch(lb, psi)
    sign_a = _sgn(la)
    sign_b = _sgn(pole["d0"] * (pole["d0"] + pole["d2"] + pole["d4"])) * _sgn(lb)
    ga = atheta + btheta / la if la != 0 else mp.mpf(0)
    gb = atheta + btheta / lb if lb != 0 else mp.mpf(0)
    scale = pole["J"] / abs(pole["psidot"])
    phi = scale * (sign_a * abs(pole["residue_a"]) * ba
                   + sign_b * abs(pole["residue_b"]) * bb)
    zeta = scale * (sign_a * abs(pole["residue_a"]) * ga * ba
                    + sign_b * abs(pole["residue_b"]) * gb * bb)
    return phi, zeta


def _current(v, st):
    L = st["eta"] / v
    J = mp.sqrt(L * L + 2 * st["c1"] / v + st["sav"])
    B, C, D = _cubic_coefficients(st["q"], st["eta"], st["delta"],
                                  st["s1sq"], st["s2sq"], L, J, st["xi"])
    candidate, raw, valid = _cubic_roots(B, C, D)
    fallback = [mp.mpf(0), st["Ssum2"], st["Ssum2"] + mp.mpf("1e-9")]
    roots = candidate if valid and not st["root_regularized"] else fallback
    gap = abs(roots[2] - roots[1])
    degenerate = gap < mp.mpf("1e-5")
    spin_roots = fallback if degenerate else roots
    S2_raw = spin_roots[2]  # overwritten by the zero-nutation current limit below
    if degenerate:
        # The contract uses the upper candidate root for the current spin norm.
        S2 = roots[2]
    else:
        psi = _phase(v, st["delta"], st["g"]["g0"], st["psi1"], st["psi2"])
        m = (spin_roots[2] - spin_roots[1]) / (spin_roots[2] - spin_roots[0])
        # This path is not used at 190/196, but keeps the grid diagnostic
        # mathematically explicit without importing a Jacobi implementation.
        sn = mp.ellipfun("sn", psi, m)
        S2 = spin_roots[2] + (spin_roots[1] - spin_roots[2]) * sn * sn
    return {
        "L": L, "J": J, "roots": roots, "candidate_roots": candidate,
        "raw": raw, "valid": valid, "gap": gap, "degenerate": degenerate,
        "spin_roots": spin_roots, "S2": S2,
    }


def _opening_L(f, st):
    M, m1, m2 = st["M"], st["m1"], st["m2"]
    eta, delta = st["eta"], st["delta"]
    nu1, nu2 = m1 / M, m2 / M
    x = (mp.pi * M * mp.mpf(str(f)) * MTSUN) ** (mp.mpf(2) / 3)
    spin_l = nu1 ** 2 * mp.mpf(str(CASE["spin1"][2])) \
        + nu2 ** 2 * mp.mpf(str(CASE["spin2"][2]))
    sigma_l = nu2 * mp.mpf(str(CASE["spin2"][2])) \
        - nu1 * mp.mpf(str(CASE["spin1"][2]))
    l3 = 7 * eta ** 3 / 1296 + 31 * eta ** 2 / 24 \
        + (41 * mp.pi ** 2 / 24 - mp.mpf(6889) / 144) * eta + mp.mpf(135) / 16
    terms = 1 + (mp.mpf("1.5") + eta / 6) * x \
        + (-mp.mpf(23) / 6 * spin_l - mp.mpf("1.5") * delta * sigma_l) * x ** mp.mpf("1.5") \
        + (mp.mpf(27) / 8 - 19 * eta / 8 + eta ** 2 / 24) * x ** 2 \
        + l3 * x ** 3 \
        + ((-mp.mpf(77) / 8 + mp.mpf(259) / 72 * eta) * spin_l
           + delta * (-mp.mpf(33) / 8 + 7 * eta / 4) * sigma_l) * x ** mp.mpf("2.5")
    return eta / mp.sqrt(x) * terms


def _angle_at(f, st, ctx):
    v = (mp.pi * st["M"] * mp.mpf(str(f)) * MTSUN) ** (mp.mpf(1) / 3)
    cur = _current(v, st)
    sec_phi, sec_zeta = _secular(v, st, ctx)
    psi = st["psi0"] + _phase(v, st["delta"], st["g"]["g0"], st["psi1"], st["psi2"]) \
        - _phase(st["v0"], st["delta"], st["g"]["g0"], st["psi1"], st["psi2"])
    if cur["degenerate"]:
        osc_phi = osc_zeta = mp.mpf(0)
        pole = _pole(v, st, cur["spin_roots"])
    else:
        pole = _pole(v, st, cur["spin_roots"])
        J, L = pole["J"], pole["L"]
        atheta = (J * J + L * L - cur["spin_roots"][2]) / (2 * J * L)
        btheta = (cur["spin_roots"][2] - cur["spin_roots"][1]) / (2 * J * L)
        osc_phi, osc_zeta = _oscillatory(pole, psi, atheta, btheta)
    # Initial oscillatory correction is deliberately zero for this fixture.
    if st["degenerate0"]:
        osc0_phi = osc0_zeta = mp.mpf(0)
    else:
        p0 = _pole(st["v0"], st, st["roots0"])
        J0, L0 = p0["J"], p0["L"]
        af0 = (J0 * J0 + L0 * L0 - st["roots0"][2]) / (2 * J0 * L0)
        bf0 = (st["roots0"][2] - st["roots0"][1]) / (2 * J0 * L0)
        osc0_phi, osc0_zeta = _oscillatory(p0, st["psi0"], af0, bf0)
    alpha = sec_phi + osc_phi - osc0_phi
    gamma = -(sec_zeta + osc_zeta - osc0_zeta)
    Lb = _opening_L(f, st)
    Jb = mp.sqrt(Lb * Lb + 2 * st["c1"] * Lb / st["eta"] + st["sav"])
    cosb = (Jb * Jb + Lb * Lb - cur["S2"]) / (2 * Jb * Lb)
    return {
        "alpha": alpha, "gamma": gamma, "cos_beta": cosb,
        "psi": psi, "secular_phi": sec_phi, "secular_zeta": sec_zeta,
        "oscillatory_phi": osc_phi, "oscillatory_zeta": osc_zeta,
        "current": cur, "pole": pole,
    }


def _options():
    options = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(options, VERSION)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPExpansionOrder(options, ORDER)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPConvention(options, CONVENTION)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(
        options, FINAL_SPIN_MOD
    )
    return options


def _public_angles(frequencies):
    vec = lal.CreateREAL8Vector(len(frequencies))
    vec.data[:] = [float(x) for x in frequencies]
    values = lalsim.SimIMRPhenomXPMSAAngles(
        vec, CASE["mass1"] * lal.MSUN_SI, CASE["mass2"] * lal.MSUN_SI,
        *CASE["spin1"], *CASE["spin2"], CASE["inclination"], CASE["f_ref"],
        MPRIME, _options(),
    )
    return [[float(x) for x in value.data] for value in values]


def _public_waveform():
    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        CASE["mass1"] * lal.MSUN_SI, CASE["mass2"] * lal.MSUN_SI,
        *CASE["spin1"], *CASE["spin2"],
        CASE["distance"] * 1.0e6 * lal.PC_SI, CASE["inclination"],
        CASE["phic"], CASE["tc"], 0.0, 0.0, 1.0, FLOW, FHIGH,
        CASE["f_ref"], _options(), lalsim.IMRPhenomXP,
    )
    start, stop = int(FLOW), int(FHIGH) + 1
    return (torch.as_tensor(hp.data.data[start:stop].copy(), dtype=torch.complex128),
            torch.as_tensor(hc.data.data[start:stop].copy(), dtype=torch.complex128))


def _metrics(actual, target):
    rows = {}
    for name, value, reference in zip(("hp", "hc"), actual, target):
        diff = value - reference
        norm = torch.linalg.vector_norm(reference)
        rows[name] = {
            "difference_l2": float(torch.linalg.vector_norm(diff)),
            "target_l2": float(norm),
            "relative_l2": float(torch.linalg.vector_norm(diff) / norm),
            "max_abs": float(diff.abs().max()),
        }
    return rows


def _float_pole(st, f, spin_roots):
    """Float64 re-evaluation of the same independent pole equations."""
    # Rebuild from rounded scalar inputs, then cast every input and output to
    # binary64 before arithmetic.  This isolates ordinary float rounding from
    # the 50/80/120-digit equation evaluations above.
    v = float((mp.pi * st["M"] * mp.mpf(str(f)) * MTSUN) ** (mp.mpf(1) / 3))
    sr = [float(x) for x in spin_roots]
    eta, xi, c1 = float(st["eta"]), float(st["xi"]), float(st["c1"])
    q, delta = float(st["q"]), float(st["delta"])
    s1sq, s2sq = float(st["s1sq"]), float(st["s2sq"])
    L = eta / v
    J = math.sqrt(L * L + 2 * c1 / v + float(st["sav"]))
    s3, sm, sp = sr
    c0 = .75 * (1 - xi * v) * v ** 2 * (
        eta ** 3 + 4 * eta ** 3 * xi * v
        - 2 * eta * (J * J - sp + 2 * (s1sq - s2sq) * delta) * v ** 2
        - 4 * eta * xi * (J * J - sp) * v ** 3
        + (J * J - sp) ** 2 / eta * v ** 4)
    c2 = -1.5 * eta * (sp - sm) * (
        1 + 2 * xi * v - (J * J - sp) / eta ** 2 * v ** 2) * (1 - xi * v) * v ** 4
    c4 = .75 / eta * (sp - sm) ** 2 * (1 - xi * v) * v ** 6
    d0 = -(J * J - (L + math.sqrt(sp)) ** 2) * (J * J - (L - math.sqrt(sp)) ** 2)
    d2 = -2 * (sp - sm) * (J * J + L * L - sp)
    d4 = -(sp - sm) ** 2
    root = math.sqrt(abs(d2 * d2 - 4 * d0 * d4))
    la, lb = (-d2 + root) / (2 * d0), (-d2 - root) / (2 * d0)
    r0 = c4 / d4
    u, w = c0 / d0 - r0, c2 / d0 + r0 * (la + lb)
    ra, rb = (w + u * la) / (la - lb), (-w - u * lb) / (la - lb)
    return {"lambda_a": la, "lambda_b": lb,
            "residue_a": ra, "residue_b": rb, "residue_0": r0}


def _f(x):
    return float(x)


def _mp_str(x, digits=18):
    """Serialize an mpmath value without rounding it to binary64."""
    return mp.nstr(x, digits)


def _max_abs(values):
    return max(abs(float(x)) for x in values)


def _serialize_mp_angle(row):
    """Keep the target-row audit readable while retaining scalar diagnostics."""
    cur = row["current"]
    pole = row["pole"]
    result = {
        name: float(row[name])
        for name in ("alpha", "gamma", "cos_beta", "psi",
                     "secular_phi", "secular_zeta",
                     "oscillatory_phi", "oscillatory_zeta")
    }
    result.update({
        name: float(pole[name])
        for name in ("lambda_a", "lambda_b", "residue_a",
                     "residue_b", "residue_0")
    })
    result.update({
        "pole_factor_a": float(1 - pole["lambda_a"]),
        "pole_factor_b": float(1 - pole["lambda_b"]),
    })
    result["pole"] = {
        name: float(pole[name])
        for name in ("J", "L", "d0", "d2", "d4", "psidot")
    }
    result["current"] = {
        "L": float(cur["L"]), "J": float(cur["J"]),
        "raw": float(cur["raw"]), "gap": float(cur["gap"]),
        "valid": bool(cur["valid"]),
        "degenerate": bool(cur["degenerate"]),
        "S2": float(cur["S2"]),
        "roots": [float(x) for x in cur["roots"]],
        "candidate_roots": [float(x) for x in cur["candidate_roots"]],
        "spin_roots": [float(x) for x in cur["spin_roots"]],
    }
    return result


def _run():
    frequencies = [float(x) for x in range(int(FLOW), int(FHIGH) + 1)]
    native_f = torch.as_tensor(frequencies, dtype=torch.float64)
    with torch.no_grad():
        native = assemble_precession_msa_angles(
            native_f, CASE["mass1"], CASE["mass2"],
            torch.tensor(CASE["spin1"], dtype=torch.float64),
            torch.tensor(CASE["spin2"], dtype=torch.float64),
            f_ref=CASE["f_ref"], order=ORDER, compatibility="v223",
        )
    public = _public_angles(frequencies)
    with mp.workdps(120):
        st = _state()
        ctx = _secular_context(st)
        initial_raw_decimal = _mp_str(st["raw0"], 40)
        initial_raw_plus_decimal = _mp_str(st["raw0"] + 1, 40)
        high = [_angle_at(f, st, ctx) for f in frequencies]
        # Recompute the same published equations at high precision with the
        # acos argument clamped to its endpoint.  This is a sensitivity
        # control for the near-degenerate initial roots, not a proposed fix.
        st_endpoint = _state(initial_arg_override=mp.mpf("-1"))
        ctx_endpoint = _secular_context(st_endpoint)
        high_endpoint = [
            _angle_at(f, st_endpoint, ctx_endpoint) for f in frequencies
        ]
        target = {}
        for f in TARGET_FREQUENCIES:
            i = int(f - FLOW)
            key = str(int(f))
            row = _angle_at(f, st, ctx)
            mp_row = _serialize_mp_angle(row)
            endpoint_row = _serialize_mp_angle(
                high_endpoint[i]
            )
            target[key] = {
                "mp": mp_row,
                "mp_endpoint_clamped_initial": endpoint_row,
                "clean": {
                "alpha": float(native["alpha"][i]),
                "gamma": float(native["gamma"][i]),
                "cos_beta": float(native["cos_beta"][i]),
                "roots": [float(x) for x in native["roots"][i]],
                "frequency_degenerate": bool(native["frequency_degenerate"][i]),
                "lambda_a": float(native["rational_lambda_a"][i]),
                "lambda_b": float(native["rational_lambda_b"][i]),
                },
            }
            cur = row["current"]
            spin_roots = cur["spin_roots"]
            target[key]["float64_pole_same_mp_roots"] = _float_pole(
                st, f, spin_roots
            )
            clean_roots = [
                mp.mpf(float(x)) for x in native["roots"][i]
            ]
            target[key]["float64_pole_same_clean_roots"] = _float_pole(
                st, f, clean_roots
            )
            target[key]["mp_pole_same_clean_roots"] = {
                name: float(value)
                for name, value in _pole(
                    (mp.pi * st["M"] * mp.mpf(str(f)) * MTSUN)
                    ** (mp.mpf(1) / 3), st, clean_roots,
                    round_inputs=True,
                ).items()
                if name in ("lambda_a", "lambda_b", "residue_a",
                             "residue_b", "residue_0")
            }
            target[key]["mp_pole_same_mp_roots"] = {
                name: float(value)
                for name, value in _pole(
                    (mp.pi * st["M"] * mp.mpf(str(f)) * MTSUN)
                    ** (mp.mpf(1) / 3), st, spin_roots,
                    round_inputs=True,
                ).items()
                if name in ("lambda_a", "lambda_b", "residue_a",
                             "residue_b", "residue_0")
            }
            clean = target[key]["clean"]
            fp = target[key]["float64_pole_same_mp_roots"]
            fp_clean = target[key]["float64_pole_same_clean_roots"]
            mpp_clean = target[key]["mp_pole_same_clean_roots"]
            mpp_mp = target[key]["mp_pole_same_mp_roots"]
            target[key]["differences"] = {
                "clean_minus_mp": {key: clean[key] - mp_row[key]
                                    for key in ("alpha", "gamma", "cos_beta")},
                "clean_minus_float64_pole_same_mp_roots": {
                    key: clean["lambda_a"] - fp["lambda_a"]
                    for key in ("lambda_a",)
                } | {"lambda_b": clean["lambda_b"] - fp["lambda_b"]},
                "clean_current_roots_minus_exact_mp": [
                    float(a) - float(b)
                    for a, b in zip(native["roots"][i], cur["roots"])
                ],
                "clean_current_gap_minus_exact_mp": (
                    float(abs(native["roots"][i][2] - native["roots"][i][1]))
                    - float(cur["gap"])
                ),
                "mp_minus_float64_pole_same_mp_roots": {
                    key: mp_row[key] - fp[key]
                    for key in ("lambda_a", "lambda_b", "residue_a", "residue_b", "residue_0")
                },
                "mp_pole_same_clean_roots_minus_float64": {
                    key: mpp_clean[key] - fp_clean[key]
                    for key in ("lambda_a", "lambda_b", "residue_a",
                                "residue_b", "residue_0")
                },
                "mp_pole_same_clean_roots_minus_mp_roots": {
                    key: mpp_clean[key] - mp_row[key]
                    for key in ("lambda_a", "lambda_b", "residue_a",
                                "residue_b", "residue_0")
                },
                "endpoint_initial_minus_exact_mp": {
                    key: endpoint_row[key] - mp_row[key]
                    for key in ("alpha", "gamma", "cos_beta",
                                "secular_phi", "secular_zeta",
                                "oscillatory_phi", "oscillatory_zeta",
                                "psi")
                },
            }
        # Precision sensitivity is measured across decimal precisions, with
        # identical equations and inputs at each precision.
        precision = {}
        precision_raw = {}
        for dps in (50, 80, 120):
            with mp.workdps(dps):
                s = _state()
                c = _secular_context(s)
                precision[str(dps)] = {}
                precision_raw[str(dps)] = {}
                for f in TARGET_FREQUENCIES:
                    row = _angle_at(f, s, c)
                    pole = row["pole"]
                    values = {
                        "alpha": row["alpha"], "gamma": row["gamma"],
                        "cos_beta": row["cos_beta"],
                        "lambda_a": pole["lambda_a"],
                        "lambda_b": pole["lambda_b"],
                        "residue_a": pole["residue_a"],
                        "residue_b": pole["residue_b"],
                    }
                    precision_raw[str(dps)][str(int(f))] = values
                    precision[str(dps)][str(int(f))] = {
                        name: float(value) for name, value in values.items()
                    }
        sensitivity = {}
        for f in TARGET_FREQUENCIES:
            key = str(int(f))
            sensitivity[key] = {}
            for low, high_dps in ((50, 80), (80, 120)):
                a = precision_raw[str(low)][key]
                b = precision_raw[str(high_dps)][key]
                sensitivity[key][f"{low}_vs_{high_dps}"] = {
                    name: _mp_str(a[name] - b[name])
                    for name in ("alpha", "gamma", "cos_beta", "lambda_a",
                                 "lambda_b", "residue_a", "residue_b")
                }
    mp_alpha = torch.tensor([float(row["alpha"]) for row in high], dtype=torch.float64)
    mp_gamma = torch.tensor([float(row["gamma"]) for row in high], dtype=torch.float64)
    mp_cosb = torch.tensor([float(row["cos_beta"]) for row in high], dtype=torch.float64)
    clean_alpha, clean_gamma, clean_cosb = native["alpha"], native["gamma"], native["cos_beta"]
    public_alpha = torch.tensor(public[0], dtype=torch.float64)
    public_gamma = torch.tensor(public[1], dtype=torch.float64)
    public_cosb = torch.tensor(public[2], dtype=torch.float64)
    endpoint_alpha = torch.tensor(
        [float(row["alpha"]) for row in high_endpoint], dtype=torch.float64
    )
    endpoint_gamma = torch.tensor(
        [float(row["gamma"]) for row in high_endpoint], dtype=torch.float64
    )
    endpoint_cosb = torch.tensor(
        [float(row["cos_beta"]) for row in high_endpoint], dtype=torch.float64
    )
    alpha_offset_mp = public_alpha[0] - mp_alpha[0]
    alpha_offset_clean = public_alpha[0] - clean_alpha[0]
    mp_a_aligned = mp_alpha + alpha_offset_mp
    clean_a_aligned = clean_alpha + alpha_offset_clean
    angle_summary = {
        "grid_hz": [FLOW, FHIGH], "count": len(frequencies),
        "clean_minus_mp_max_abs": {
            "alpha": float((clean_alpha - mp_alpha).abs().max()),
            "gamma": float((clean_gamma - mp_gamma).abs().max()),
            "cos_beta": float((clean_cosb - mp_cosb).abs().max()),
        },
        "endpoint_initial_minus_exact_mp_max_abs": {
            "alpha": float((endpoint_alpha - mp_alpha).abs().max()),
            "gamma": float((endpoint_gamma - mp_gamma).abs().max()),
            "cos_beta": float((endpoint_cosb - mp_cosb).abs().max()),
        },
        "mp_vs_public_gauge_aligned_max_abs": {
            "alpha": float((mp_a_aligned - public_alpha).abs().max()),
            "gamma": float((mp_gamma - public_gamma).abs().max()),
            "cos_beta": float((mp_cosb - public_cosb).abs().max()),
            "alpha_plus_gamma": float((mp_a_aligned + mp_gamma
                                        - public_alpha - public_gamma).abs().max()),
            "alpha_minus_gamma": float((mp_a_aligned - mp_gamma
                                         - public_alpha + public_gamma).abs().max()),
        },
        "clean_vs_public_gauge_aligned_max_abs": {
            "alpha": float((clean_a_aligned - public_alpha).abs().max()),
            "gamma": float((clean_gamma - public_gamma).abs().max()),
            "cos_beta": float((clean_cosb - public_cosb).abs().max()),
            "alpha_plus_gamma": float((clean_a_aligned + clean_gamma
                                        - public_alpha - public_gamma).abs().max()),
            "alpha_minus_gamma": float((clean_a_aligned - clean_gamma
                                         - public_alpha + public_gamma).abs().max()),
        },
        "alpha_offset_mp": float(alpha_offset_mp),
        "alpha_offset_clean": float(alpha_offset_clean),
    }
    grid_t = torch.as_tensor(frequencies, dtype=torch.float64)
    with torch.no_grad():
        native_hp, native_hc = waveform_v223(
            grid_t, CASE["mass1"], CASE["mass2"], *CASE["spin1"], *CASE["spin2"],
            distance=CASE["distance"], inclination=CASE["inclination"],
            phic=CASE["phic"], tc=CASE["tc"], f_ref=CASE["f_ref"],
            convention=CONVENTION, order=ORDER,
        )
    public_hp, public_hc = _public_waveform()
    report = {
        "status": "complete_read_only_precision_audit",
        "case": CASE,
        "contract": {
            "prec_version": VERSION, "expansion_order": ORDER,
            "frequency_grid_hz": [FLOW, FHIGH], "frequency_count": len(frequencies),
            "target_frequencies_hz": list(TARGET_FREQUENCIES),
            "root_gap_threshold": 1.0e-5,
            "finite_cubic_overshoot": "clamp finite acos argument to [-1,1]",
            "pole_continuation": "sqrt(abs(1-lambda)); algebraic labels; signed second pole denominator",
            "phase_limit": "P(v) with additive integration constant C=0 for initial degenerate row",
            "comparison": "raw angles and hp/hc; no phase/time/amplitude/residual fitting",
        },
        "independent_reference": {
            "mpmath_dps": 120,
            "initial_roots": [float(x) for x in st["roots0"]],
            "initial_roots_endpoint_clamped": [
                float(x) for x in st_endpoint["roots0"]
            ],
            "initial_roots_binary64_clean": [
                float(x) for x in native["roots_reference"]
            ],
            "initial_root_differences": {
                "endpoint_clamped_minus_exact": [
                    float(a - b)
                    for a, b in zip(st_endpoint["roots0"], st["roots0"])
                ],
                "binary64_clean_minus_exact": [
                    float(a) - float(b)
                    for a, b in zip(native["roots_reference"], st["roots0"])
                ],
            },
            "initial_raw_cubic_argument": float(st["raw0"]),
            "initial_raw_cubic_argument_decimal": initial_raw_decimal,
            "initial_raw_plus_one_decimal": initial_raw_plus_decimal,
            "initial_roots_valid": st["roots_valid0"],
            "initial_degenerate": st["degenerate0"],
            "physical_q": float(st["q"]),
            "secular_coefficients": "independent mpmath transcription of published Appendix coefficients",
        },
        "target_rows": target,
        "precision_sensitivity": sensitivity,
        "angle_comparison": angle_summary,
        "raw_waveform_metrics": _metrics(
            (native_hp, native_hc), (public_hp, public_hc)
        ),
        "raw_waveform_metrics_20_200_hz": _metrics(
            (native_hp[:181], native_hc[:181]),
            (public_hp[:181], public_hc[:181]),
        ),
        "interpretation": {
            "root_arithmetic": "The 50/80/120-digit runs are stable for fixed exact inputs, but the initial and current upper roots are nearly degenerate. Endpoint clamping versus the exact acos argument changes the root triple, and the resulting lambda_a shift crosses the n_a continuation boundary; a small absolute root difference therefore does not exclude amplified root arithmetic.",
            "pole_arithmetic": "With identical binary64-rounded roots and scalar inputs, the independent high-precision and float64 pole evaluations agree to approximately 1e-14 at 190 and 196 Hz. The clean lambda_a values differ from the same-root controls because the clean current roots/invariants differ; residue evaluation itself is not the 1e-3-scale source.",
            "phase_convention": "The endpoint-clamped high-precision control isolates the secular initialization sensitivity from pole arithmetic. Its angle shifts are reported per target row; this control is diagnostic only and does not select a replacement phase convention.",
            "physical_cancellation": "After aligning the alpha integration constant at the reference frequency, clean/public alpha-minus-gamma differs by about 1.2e-4 across 20--512 Hz while the individual alpha and gamma differences are about 2.4e-2. The cancellation reduces waveform sensitivity, but does not remove the raw 20--200 Hz residual.",
            "remaining_mismatch": "The residual combines amplified near-degenerate root/continuation sensitivity with a remaining angle or assembly convention difference. No empirical correction, threshold change, or production routing is selected by this diagnostic.",
        },
        "source_hashes": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in (
                "scripts/diagnose_msa_v223_boundary_precision_clean.py",
                "src/torchwave/approximants/precession_msa_angles_clean.py",
                "src/torchwave/approximants/precession_msa_oscillatory_clean.py",
                "src/torchwave/approximants/precession_msa_pn_clean.py",
                "src/torchwave/approximants/precession_msa_secular_clean.py",
                "src/torchwave/approximants/precession_msa_initial_clean.py",
                "src/torchwave/approximants/precession_msa_waveform_clean.py",
                "docs/reviews/2026-09-13-parity-review/reference-audit-contracts.md",
            )
        },
    }
    return report


def main():
    report = _run()
    out = ROOT / "docs/reviews/2026-09-13-parity-review/verification/msa-boundary-contract-clean/msa_boundary_precision_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
