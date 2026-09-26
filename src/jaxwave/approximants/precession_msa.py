"""NNLO and published two-spin multiple-scale precession angles in JAX.

The MSA branch includes elliptic nutation, secular phases and leading
oscillatory corrections. The numerical distinctions between published
implementation labels 220--224 share the analytical prescription here,
with their stated fallback policy.

Mathematical references:
- Pratten et al. (2021), Phys. Rev. D 103, 104056 (arXiv:2004.06503), Sec. IV & Appendix B
- Chatziioannou et al. (2017), Phys. Rev. D 95, 104004 (arXiv:1703.03967)
- Chatziioannou et al. (2013), Phys. Rev. D 88, 063011 (arXiv:1307.4418)
"""

from typing import Optional, Tuple, Union
import math
import jax
import jax.numpy as jnp

from diffgw_core.constants import PI, EULER_GAMMA, MSUN_SEC
from jaxwave.conversions import as_array
from jaxwave.approximants.imrphenomp import IMRPhenomP


MSA_FINAL_SPIN_DOMAIN_ERROR = (
    "MSA final_spin_mod=3 requires a valid nonaligned MSA remnant root; "
    "use final_spin_mod=0 for equal-mass or coalescing-root rows."
)


def _check_msa_final_spin_domain(unsupported):
    if bool(unsupported):
        raise ValueError(MSA_FINAL_SPIN_DOMAIN_ERROR)


def _enforce_msa_final_spin_domain(unsupported: jnp.ndarray) -> None:
    """Raise eagerly, or through a compiled callback when values are traced."""
    any_unsupported = jnp.any(unsupported)
    if isinstance(any_unsupported, jax.core.Tracer):
        # A conditional callback is not vmap-safe: batching lowers lax.cond
        # to a select and may execute both branches. Keep the condition in the
        # callback so valid batched/JIT rows cannot run the error branch.
        jax.debug.callback(
            _check_msa_final_spin_domain, any_unsupported, ordered=True
        )
    elif bool(any_unsupported):
        raise ValueError(MSA_FINAL_SPIN_DOMAIN_ERROR)


def resolve_precession_version(method: str, version: Optional[int]) -> int:
    """An explicit version takes precedence; method chooses an omitted version."""
    method_l = method.lower()
    if method_l not in ("nnlo", "msa", "spintaylor", "st", "spintaylort4", "pnr", "xpnr"):
        raise ValueError("precession_method must be 'nnlo', 'msa', 'spintaylor', or 'pnr'.")
    if version is None:
        if method_l in ("pnr", "xpnr"):
            return 330
        if method_l in ("spintaylor", "st", "spintaylort4"):
            return 320
        return 223 if method_l == "msa" else 102
    if version not in (101, 102, 103, 104, 220, 221, 222, 223, 224, 300, 310, 311, 320, 321, 330):
        raise ValueError(f"Unsupported precession version: {version}.")
    if version == 300:
        return 330
    return version


def compute_effective_chip(
    m1: jnp.ndarray,
    m2: jnp.ndarray,
    s1x: jnp.ndarray,
    s1y: jnp.ndarray,
    s2x: jnp.ndarray,
    s2y: jnp.ndarray,
) -> jnp.ndarray:
    """Published effective in-plane spin, Eqs. (4.7)--(4.9), for m1 >= m2."""
    q_pratten = m1 / m2
    a1_coeff = 2.0 + 1.5 / q_pratten
    a2_coeff = 2.0 + 1.5 * q_pratten
    s1_perp = (m1 * m1) * jnp.hypot(s1x, s1y)
    s2_perp = (m2 * m2) * jnp.hypot(s2x, s2y)
    return jnp.maximum(a1_coeff * s1_perp, a2_coeff * s2_perp) / (a1_coeff * (m1 * m1))


def elliptic_f(phi: jnp.ndarray, parameter: jnp.ndarray) -> jnp.ndarray:
    """Real F(phi|m), for |phi| <= pi/2, by Carlson duplication.

    F = sin(phi) RF(cos(phi)^2, 1-m sin(phi)^2, 1). The duplication
    identity and symmetric Taylor expansion are DLMF 19.26 and 19.36.
    Fixed iteration counts preserve tensor broadcasting and autograd.
    """
    phi, parameter = jnp.broadcast_arrays(as_array(phi, dtype=jnp.float64), as_array(parameter, dtype=jnp.float64))
    s = jnp.sin(phi)
    x = jnp.cos(phi) ** 2
    y = 1.0 - parameter * s * s
    z = jnp.ones_like(s)
    x, y, z = jnp.broadcast_arrays(x, y, z)
    for _ in range(12):
        root_x = jnp.sqrt(jnp.maximum(x, 0.0))
        root_y = jnp.sqrt(jnp.maximum(y, 0.0))
        root_z = jnp.sqrt(jnp.maximum(z, 0.0))
        lam = root_x * root_y + root_x * root_z + root_y * root_z
        x = (x + lam) * 0.25
        y = (y + lam) * 0.25
        z = (z + lam) * 0.25
    mean = (x + y + z) / 3.0
    dx = (mean - x) / mean
    dy = (mean - y) / mean
    dz = (mean - z) / mean
    e2 = dx * dy - dz * dz
    e3 = dx * dy * dz
    rf = (1.0 - e2 / 10.0 + e3 / 14.0 + e2 * e2 / 24.0 - 3.0 * e2 * e3 / 44.0) / jnp.sqrt(jnp.maximum(mean, 1e-30))
    return s * rf


def jacobi_sn_cn_dn(u: jnp.ndarray, parameter: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Jacobi functions from the descending Landen transformation (DLMF 22.20)."""
    u, parameter = jnp.broadcast_arrays(as_array(u, dtype=jnp.float64), as_array(parameter, dtype=jnp.float64))
    eps = jnp.finfo(u.dtype).eps
    m = jnp.clip(parameter, 0.0, 1.0 - eps)
    a = jnp.ones_like(m)
    b = jnp.sqrt(jnp.maximum(1.0 - m, 0.0))
    ratios = []
    for _ in range(12):
        next_a = (a + b) / 2.0
        ratios.append((a - b) / (2.0 * next_a))
        a, b = next_a, jnp.sqrt(jnp.maximum(a * b, 0.0))
    period = 2.0 * PI / a
    reduced = u - jnp.round(u / period) * period
    phase = 4096.0 * a * reduced
    for ratio in reversed(ratios):
        phase = (phase + jnp.arcsin(jnp.clip(ratio * jnp.sin(phase), -1.0, 1.0))) / 2.0
    sn, cn = jnp.sin(phase), jnp.cos(phase)
    dn = jnp.sqrt(jnp.maximum(1.0 - m * sn * sn, eps))
    is_one = parameter == 1.0
    sech = 1.0 / jnp.cosh(jnp.clip(u, -80.0, 80.0))
    return (
        jnp.where(is_one, jnp.tanh(u), sn),
        jnp.where(is_one, sech, cn),
        jnp.where(is_one, sech, dn),
    )


def msa_phase_primitives(v: jnp.ndarray, eta: jnp.ndarray, c1: jnp.ndarray, sav_sq: jnp.ndarray) -> jnp.ndarray:
    """Primitives with derivative J*v**(n-3), n=0..5.

    Chatziioannou (2017), D21--D26. D22 requires c1**2 in its logarithmic
    coefficient: differentiating the printed c1 expression fails Eq. 65.
    The MSA dynamics use the Newtonian L=eta/v, as in the paper's Eq. 4.
    """
    l = eta / v
    j = jnp.sqrt(l * l + 2.0 * c1 / v + sav_sq)
    sav = jnp.sqrt(sav_sq)
    tiny = jnp.finfo(v.dtype).tiny
    log1 = jnp.log(jnp.maximum(c1 + j * eta + l * eta, tiny))
    log2 = jnp.log(jnp.maximum(c1 + j * sav * v + sav_sq * v, tiny))
    invariant = c1 * c1 - eta * eta * sav_sq
    p0 = (
        j / (eta**4) * (c1 * c1 / 2.0 - c1 * eta * eta / (6.0 * v) - sav_sq * eta * eta / 3.0 - (eta**4) / (3.0 * v * v))
        - c1 * invariant / (2.0 * (eta**5)) * log1
    )
    p1 = -j / (2.0 * eta * eta) * (c1 + eta * l) + invariant / (2.0 * (eta**3)) * log1
    p2 = -j + sav * log2 - c1 / eta * log1
    p3 = j * v - eta * log1 + c1 / sav * log2
    p4 = j * v * (c1 + v * sav_sq) / (2.0 * sav_sq) - invariant / (2.0 * (sav**3)) * log2
    p5 = (
        -j * v * (c1 * c1 / (2.0 * (sav_sq**2)) - c1 * v / (6.0 * sav_sq) - v * v / 3.0 - eta * eta / (3.0 * sav_sq))
        + c1 * invariant / (2.0 * (sav**5)) * log2
    )
    return jnp.stack((p0, p1, p2, p3, p4, p5), axis=-1)


def msa_nutation_phase(
    v: jnp.ndarray,
    v0: jnp.ndarray,
    eta: jnp.ndarray,
    delta: jnp.ndarray,
    q: jnp.ndarray,
    xi: jnp.ndarray,
    c1: jnp.ndarray,
    sav_sq: jnp.ndarray,
    s1_sq: jnp.ndarray,
    s2_sq: jnp.ndarray,
    psi_ref: jnp.ndarray,
) -> jnp.ndarray:
    """Eq. 51, with C1/C2 obtained by expanding the defining cubic.

    Vieta's relations determine the two finite leading roots and the third
    root's v^-2, v^-1, v^0 terms. Expanding Eq. 50 then integrating gives
    the published 1PN phase, avoiding cancellation-prone nested C3 factors.
    """
    b2 = eta * delta * delta
    b1 = 2.0 * eta * xi - 4.0 * c1
    b0 = s1_sq * (q - 1.0) + s2_sq * (1.0 / q - 1.0) - 2.0 * sav_sq
    c2 = 4.0 * c1 * c1 - 4.0 * eta * xi * c1 + 4.0 * (eta**3) * xi * xi - 2.0 * b2 * (s1_sq + s2_sq)
    d2 = (
        4.0 * (1.0 - q) / q * (s2_sq - q * s1_sq) * c1 * c1
        + b2 * ((s1_sq - s2_sq) ** 2)
        + 4.0 * delta * eta * xi * (s1_sq - s2_sq) * c1
    )
    root_sum = -c2 / b2
    root_product = d2 / b2
    finite_plus = (root_sum + jnp.sqrt(jnp.maximum(root_sum * root_sum - 4.0 * root_product, 0.0))) / 2.0
    sqrt1 = b1 / (2.0 * b2)
    sqrt2 = (finite_plus + b0 + root_sum) / (2.0 * b2) - (b1 * b1) / (8.0 * b2 * b2)
    psi1 = 1.5 * (sqrt1 - xi)
    psi2 = 3.0 * (sqrt2 - xi * sqrt1 + 743.0 / 336.0 + 11.0 * eta / 4.0)
    g0 = 5.0 / (96.0 * eta)

    def phase(x):
        return -3.0 * g0 * delta / 4.0 * (x**-3 + psi1 * x**-2 + psi2 / x)

    return psi_ref + phase(v) - phase(v0)


def msa_opening_angle(
    l: jnp.ndarray,
    eta: jnp.ndarray,
    c1: jnp.ndarray,
    sav_sq: jnp.ndarray,
    spin_sq: jnp.ndarray,
) -> jnp.ndarray:
    """MSA triangle, with the selected PN L in the averaged J(L) relation.

    Integrating Chatziioannou Eq. 40 with constant <S^2> gives
    J^2=L^2+2*c1*L/eta+<S^2>. Insert these J,L into Eq. 8 (Pratten 4.12).
    J remains independent of the instantaneous nutation phase. Keeping an
    instantaneous spin projection fixed while upgrading L would violate
    this precession-averaged relation.
    """
    j_sq = l * l + 2.0 * c1 * l / eta + sav_sq
    numerator = j_sq + l * l - spin_sq
    transverse_sq = jnp.maximum(4.0 * j_sq * l * l - numerator * numerator, 0.0)
    return jnp.arctan2(jnp.sqrt(transverse_sq), numerator)


def msa_oscillatory_angles(
    v: jnp.ndarray,
    j: jnp.ndarray,
    l: jnp.ndarray,
    xi: jnp.ndarray,
    eta: jnp.ndarray,
    s1_sq: jnp.ndarray,
    s2_sq: jnp.ndarray,
    delta: jnp.ndarray,
    s3_sq: jnp.ndarray,
    sm_sq: jnp.ndarray,
    sp_sq: jnp.ndarray,
    psi: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Zero-mean fast-time primitives of Eq. 30 and F1, at m=0.

    Direct real partial fractions avoid the ambiguous C_phi notation in
    Eq. F19. Their psi derivatives equal (Omega-<Omega>)/psi_dot, for both
    Euler angles, including the J factor in Eq. 30.
    """
    gap = sp_sq - sm_sq
    d0 = 4.0 * j * j * l * l - (j * j + l * l - sp_sq) ** 2
    d2 = -2.0 * gap * (j * j + l * l - sp_sq)
    d4 = -gap * gap
    c0 = (
        0.75
        * (1.0 - xi * v)
        * v
        * v
        * (
            eta**3
            + 4.0 * (eta**3) * xi * v
            - 2.0 * eta * (j * j - sp_sq + 2.0 * (s1_sq - s2_sq) * delta) * v * v
            - 4.0 * eta * xi * (j * j - sp_sq) * (v**3)
            + ((j * j - sp_sq) ** 2) / eta * (v**4)
        )
    )
    c2 = -1.5 * eta * gap * (1.0 + 2.0 * xi * v - (j * j - sp_sq) / (eta**2) * v * v) * (1.0 - xi * v) * (v**4)
    quotient = -0.75 / eta * (1.0 - xi * v) * (v**6)
    tiny = jnp.finfo(v.dtype).tiny
    disc = jnp.sqrt(jnp.maximum(d2 * d2 - 4.0 * d0 * d4, tiny))
    lambda_c = (-d2 + disc) / (2.0 * d0)
    lambda_d = (-d2 - disc) / (2.0 * d0)
    rem0 = c0 / d0 - quotient
    rem1 = c2 / d0 - quotient * d2 / d0
    residue_c = (-rem1 - rem0 * lambda_c) / (lambda_d - lambda_c)
    residue_d = rem0 - residue_c
    sinp, cosp = jnp.sin(psi), jnp.cos(psi)
    eps = jnp.finfo(v.dtype).eps

    def periodic_primitive(lam):
        root = jnp.sqrt(jnp.maximum(1.0 - lam, eps))
        return jnp.arctan2((root - 1.0) * sinp * cosp, cosp * cosp + root * sinp * sinp) / root

    primitive_c = periodic_primitive(lambda_c)
    primitive_d = periodic_primitive(lambda_d)
    psi_dot = (
        0.75
        / jnp.sqrt(eta)
        * (1.0 - xi * v)
        * (v**6)
        * jnp.sqrt(jnp.maximum(sp_sq - s3_sq, tiny))
    )
    alpha = j / psi_dot * (residue_c * primitive_c + residue_d * primitive_d)
    a_theta = (j * j + l * l - sp_sq) / (2.0 * j * l)
    b_theta = gap / (2.0 * j * l)
    epsilon = (
        j
        / psi_dot
        * (
            residue_c * (a_theta + b_theta / lambda_c) * primitive_c
            + residue_d * (a_theta + b_theta / lambda_d) * primitive_d
            - b_theta * (v**6) / 8.0 * sinp * cosp * 2.0
        )
    )
    return alpha, epsilon


def compute_l_3pn(
    v: jnp.ndarray,
    eta: jnp.ndarray,
    delta: jnp.ndarray,
    s_ell: jnp.ndarray,
    sigma_ell: jnp.ndarray,
) -> jnp.ndarray:
    """3PN orbital angular momentum: Pratten (2021), G10a-d and G10f-g.

    Formula:
        L(v) = (eta / v) * [1 + L1*v^2 + L_15_SO*v^3 + L2*v^4 + L_25_SO*v^5 + L3*v^6]
    """
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v3 * v3

    l1 = 1.5 + eta / 6.0
    l2 = 27.0 / 8.0 - (19.0 / 8.0) * eta + (eta * eta) / 24.0
    l3 = (
        135.0 / 16.0
        + (41.0 * (PI**2) / 24.0 - 6889.0 / 144.0) * eta
        + (31.0 / 24.0) * (eta * eta)
        + (7.0 / 1296.0) * (eta**3)
    )

    l_15_so = -(35.0 / 6.0) * s_ell - 2.5 * delta * sigma_ell
    l_25_so = (-77.0 / 8.0 + (427.0 / 72.0) * eta) * s_ell + delta * (-21.0 / 8.0 + (35.0 / 12.0) * eta) * sigma_ell
    poly = 1.0 + l1 * v2 + l_15_so * v3 + l2 * v4 + l_25_so * v5 + l3 * v6
    return (eta / v) * poly


def compute_l_4pn(
    v: jnp.ndarray,
    eta: jnp.ndarray,
    delta: jnp.ndarray,
    s_ell: jnp.ndarray,
    sigma_ell: jnp.ndarray,
    chi1: jnp.ndarray,
    chi2: jnp.ndarray,
    all_spin: bool = False,
) -> jnp.ndarray:
    """Pratten G10, with the 3.5PN SO term from Bohe 1212.5520 Eq. 4.7."""
    l4 = (
        -55.0 * (eta**4) / 31104.0
        - 215.0 * (eta**3) / 1728.0
        + (356035.0 / 3456.0 - 2255.0 * (PI**2) / 576.0) * (eta**2)
        + eta
        * (
            -64.0 / 3.0 * jnp.log(16.0 * v * v)
            - 6455.0 * (PI**2) / 1536.0
            - 128.0 * EULER_GAMMA / 3.0
            + 98869.0 / 5760.0
        )
        + 2835.0 / 128.0
    )
    so35 = (-405.0 / 16.0 + 1101.0 * eta / 16.0 - 29.0 * (eta**2) / 16.0) * s_ell
    so35 = so35 + delta * (-81.0 / 16.0 + 117.0 * eta / 4.0 - 15.0 * (eta**2) / 16.0) * sigma_ell
    correction = l4 * (v**8) + so35 * (v**7)
    if all_spin:
        spin2 = (0.5 + delta / 2.0 - eta) * (chi1**2) + 2.0 * eta * chi1 * chi2 + (0.5 - delta / 2.0 - eta) * (chi2**2)
        spin3 = (
            (chi1**3) * (3.0 * delta * eta / 4.0 - 1.5 * (eta**2) + 0.75 * eta)
            + (chi1**2) * chi2 * (3.0 * delta * eta / 4.0 + 1.5 * (eta**2) + 0.75 * eta)
            + chi1 * (chi2**2) * (-3.0 * delta * eta / 4.0 + 1.5 * (eta**2) + 0.75 * eta)
            + (chi2**3) * (-3.0 * delta * eta / 4.0 - 1.5 * (eta**2) + 0.75 * eta)
        )
        correction = correction + spin2 * (v**4) + spin3 * (v**7)
    return compute_l_3pn(v, eta, delta, s_ell, sigma_ell) + (eta / v) * correction


def solve_spin_roots(
    l0: jnp.ndarray,
    j0_sq: jnp.ndarray,
    s1_sq: jnp.ndarray,
    s2_sq: jnp.ndarray,
    xi: jnp.ndarray,
    q_c: jnp.ndarray,
    eta: jnp.ndarray,
    delta: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve the depressed cubic polynomial for spin magnitude roots S_3^2 <= S_-^2 <= S_+^2."""
    l0_sq = l0 * l0
    j_minus_l = j0_sq - l0_sq
    inv_qc = 1.0 / q_c
    one_minus_qc_over_qc = (1.0 - q_c) * inv_qc

    b_coeff = (l0_sq + s1_sq) * q_c + 2.0 * l0 * xi - 2.0 * j0_sq - s1_sq - s2_sq + (l0_sq + s2_sq) * inv_qc
    c_coeff = (
        j_minus_l**2
        - 2.0 * l0 * xi * j_minus_l
        - 2.0 * one_minus_qc_over_qc * (s1_sq - q_c * s2_sq) * l0_sq
        + 4.0 * eta * l0_sq * (xi**2)
        - 2.0 * delta * (s1_sq - s2_sq) * xi * l0
        + 2.0 * one_minus_qc_over_qc * (q_c * s1_sq - s2_sq) * j0_sq
    )
    d_coeff = (
        one_minus_qc_over_qc * (s2_sq - q_c * s1_sq) * (j_minus_l**2)
        + ((delta**2) / eta) * ((s1_sq - s2_sq) ** 2) * l0_sq
        + 2.0 * delta * l0 * xi * (s1_sq - s2_sq) * j_minus_l
    )

    p = c_coeff - (b_coeff**2) / 3.0
    qc_dep = (2.0 / 27.0) * (b_coeff**3) - (b_coeff * c_coeff) / 3.0 + d_coeff

    p_third = -p / 3.0
    safe_p_third = jnp.maximum(p_third, 1e-18)
    r_val = jnp.sqrt(safe_p_third)
    r3 = r_val**3

    cos_3theta = jnp.clip(-qc_dep / (2.0 * r3 + 1e-30), -1.0 + 1e-15, 1.0 - 1e-15)
    theta = jnp.arccos(cos_3theta) / 3.0

    z0 = 2.0 * r_val * jnp.cos(theta) - b_coeff / 3.0
    z1 = 2.0 * r_val * jnp.cos(theta - 2.0 * PI / 3.0) - b_coeff / 3.0
    z2 = 2.0 * r_val * jnp.cos(theta - 4.0 * PI / 3.0) - b_coeff / 3.0

    roots_stack = jnp.stack([z0, z1, z2], axis=-1)
    sorted_roots = jnp.sort(roots_stack, axis=-1)

    s3_sq = sorted_roots[..., 0]
    sm_sq = sorted_roots[..., 1]
    sp_sq = sorted_roots[..., 2]

    disc = (safe_p_third**3) - (qc_dep / 2.0) ** 2
    eps = jnp.finfo(jnp.asarray(l0).dtype).eps
    root_scale = jnp.maximum(jnp.abs(b_coeff), jnp.sqrt(jnp.abs(c_coeff)))
    disc_scale = jnp.maximum(safe_p_third**3, (qc_dep / 2.0) ** 2)
    is_valid = (
        (disc > 32.0 * eps * disc_scale)
        & (sp_sq >= sm_sq)
        & (sm_sq >= -32.0 * eps * root_scale)
        & ((sp_sq - sm_sq) > 32.0 * math.sqrt(eps) * root_scale)
        & ((1.0 / q_c - 1.0) > 32.0 * math.sqrt(eps))
    )

    return s3_sq, sm_sq, sp_sq, is_valid


def evaluate_msa_dynamics(
    v: jnp.ndarray,
    v0: jnp.ndarray,
    eta: jnp.ndarray,
    delta: jnp.ndarray,
    q_c: jnp.ndarray,
    s1: jnp.ndarray,
    s2: jnp.ndarray,
    l_pn: jnp.ndarray,
    anchor_reference: bool = True,
    return_averages: bool = False,
):
    """Published two-spin MSA solution and initialization validity."""
    s1_sq = (s1 * s1).sum(axis=-1)
    s2_sq = (s2 * s2).sum(axis=-1)
    s0_sq = ((s1 + s2) ** 2).sum(axis=-1)
    xi = (1.0 + q_c) * s1[..., 2] + (1.0 + 1.0 / q_c) * s2[..., 2]
    l0 = eta / v0
    j0_sq = l0 * l0 + 2.0 * l0 * (s1[..., 2] + s2[..., 2]) + s0_sq
    _, sm_sq, sp_sq, roots_valid = solve_spin_roots(l0, j0_sq, s1_sq, s2_sq, xi, q_c, eta, delta)
    sav0_sq = (sp_sq + sm_sq) / 2.0
    c10 = v0 * (j0_sq - l0 * l0 - sav0_sq) / 2.0
    cp0 = (eta * eta) * sp_sq - c10 * c10
    cm0 = (eta * eta) * sm_sq - c10 * c10
    valid = roots_valid & (s1_sq > 0) & (s2_sq > 0) & (cp0 > 0) & (cm0 > 0)

    def regular(x, substitute):
        return jnp.where(valid, x, jnp.full_like(x, substitute))

    q_c = regular(q_c, 0.5)
    eta = regular(eta, 2.0 / 9.0)
    delta = regular(delta, 1.0 / 3.0)
    v0 = regular(v0, 0.2)
    vec_mask = jnp.expand_dims(valid, -1)
    s1 = jnp.where(vec_mask, s1, jnp.array([0.12, 0.04, 0.08], dtype=s1.dtype))
    s2 = jnp.where(vec_mask, s2, jnp.array([-0.02, 0.03, -0.01], dtype=s2.dtype))

    s1_sq = (s1 * s1).sum(axis=-1)
    s2_sq = (s2 * s2).sum(axis=-1)
    s0_sq = ((s1 + s2) ** 2).sum(axis=-1)
    xi = (1.0 + q_c) * s1[..., 2] + (1.0 + 1.0 / q_c) * s2[..., 2]
    l0 = eta / v0
    j0_sq = l0 * l0 + 2.0 * l0 * (s1[..., 2] + s2[..., 2]) + s0_sq
    s3_sq, sm_sq, sp_sq, _ = solve_spin_roots(l0, j0_sq, s1_sq, s2_sq, xi, q_c, eta, delta)
    sav_sq = (sp_sq + sm_sq) / 2.0
    c1 = v0 * (j0_sq - l0 * l0 - sav_sq) / 2.0
    m1_norm = 1.0 / (1.0 + q_c)
    m2_norm = q_c / (1.0 + q_c)

    # Dissipative coefficients
    g0 = 5.0 / (96.0 * eta)
    a0 = 96.0 * eta / 5.0
    a2 = -743.0 / 336.0 - 2.75 * eta

    denom_spin = eta * (1.0 - q_c * q_c)
    s1_ell_avg = (c1 * (1.0 + q_c) - q_c * eta * xi) / denom_spin
    s2_ell_avg = -q_c * (c1 * (1.0 + q_c) - eta * xi) / denom_spin

    variance = ((sp_sq - sm_sq) ** 2) * v0 * v0 / (32.0 * eta * eta * ((1.0 - q_c) ** 2))
    cross_avg = s1_ell_avg * s2_ell_avg - q_c * variance
    s1_ell_sq_avg = s1_ell_avg * s1_ell_avg + variance
    s2_ell_sq_avg = s2_ell_avg * s2_ell_avg + q_c * q_c * variance

    beta3 = (113.0 / 12.0 + 25.0 * q_c / 4.0) * s1_ell_avg + (113.0 / 12.0 + 25.0 / (4.0 * q_c)) * s2_ell_avg
    a3 = 4.0 * PI - beta3
    beta5 = (
        ((31319.0 / 1008.0 - (1159.0 / 24.0) * eta) + q_c * (809.0 / 84.0 - (281.0 / 8.0) * eta)) * s1_ell_avg
        + ((31319.0 / 1008.0 - (1159.0 / 24.0) * eta) + (1.0 / q_c) * (809.0 / 84.0 - (281.0 / 8.0) * eta)) * s2_ell_avg
    )
    sigma4 = (
        (247.0 / 48.0 * (sav_sq / 2.0 - (s1_sq + s2_sq) / 2.0) - (721.0 / 48.0) * cross_avg) / eta
        + (233.0 / 96.0 * s1_sq - (719.0 / 96.0) * s1_ell_sq_avg) / (m1_norm**2)
        + (233.0 / 96.0 * s2_sq - (719.0 / 96.0) * s2_ell_sq_avg) / (m2_norm**2)
    )
    a4 = 34103.0 / 18144.0 + (13661.0 / 2016.0) * eta + (59.0 / 18.0) * (eta**2) - sigma4
    a5 = -4159.0 / 672.0 * PI - (189.0 / 8.0) * PI * eta - beta5

    g2 = -a2 / a0
    g3 = -a3 / a0
    g4 = -(a4 - a2 * a2) / a0
    g5 = -(a5 - 2.0 * a3 * a2) / a0

    # Alpha coefficients
    rm = sp_sq - sm_sq
    cp = sp_sq * (eta**2) - c1 * c1
    cm = sm_sq * (eta**2) - c1 * c1
    sqrt_cp_cm = jnp.sqrt(jnp.maximum(cp * cm, 1e-24))

    a1_om = 0.5 + 0.75 / eta
    a2_om = -3.0 / (4.0 * eta) * xi
    ad = (-3.0 * (s1_sq - s2_sq) * eta * delta + 3.0 * (c1 / eta) * (c1 - 2.0 * xi * (eta**2))) / (4.0 * sqrt_cp_cm)
    cd = (3.0 / 128.0) * (rm**2) / (eta * sqrt_cp_cm)
    hd = (c1 / (eta**2)) * (1.0 - (cp + cm) / (2.0 * sqrt_cp_cm))
    fd = ((cp + cm) / (8.0 * (eta**4))) * (1.0 - (cp + cm) / (2.0 * sqrt_cp_cm))

    om_z0 = a1_om + ad
    om_z1 = a2_om - ad * xi - ad * hd
    om_z2 = ad * hd * xi + cd - ad * fd + ad * (hd**2)
    om_z3 = (ad * fd - cd - ad * (hd**2)) * (xi + hd) + ad * fd * hd
    om_z4 = (cd + ad * (hd**2) - 2.0 * ad * fd) * (hd * xi + hd * hd - fd) - ad * (fd**2)
    om_z5 = (
        (cd - ad * fd + ad * (hd**2)) * fd * (xi + 2.0 * hd)
        - (cd + ad * (hd**2) - 2.0 * ad * fd) * (hd**2) * (xi + hd)
        - ad * (fd**2) * hd
    )

    avg_om_z0 = 3.0 * g0 * om_z0
    avg_om_z1 = 3.0 * g0 * om_z1
    avg_om_z2 = 3.0 * (g0 * om_z2 + g2 * om_z0)
    avg_om_z3 = 3.0 * (g0 * om_z3 + g2 * om_z1 + g3 * om_z0)
    avg_om_z4 = 3.0 * (g0 * om_z4 + g2 * om_z2 + g3 * om_z1 + g4 * om_z0)
    avg_om_z5 = 3.0 * (g0 * om_z5 + g2 * om_z3 + g3 * om_z2 + g4 * om_z1 + g5 * om_z0)

    # Gamma / Epsilon coefficients
    gd = (3.0 / 64.0) * (rm**2) / (eta**3) * (c1 - (eta**2) * xi) / sqrt_cp_cm
    om_zeta0 = om_z0
    om_zeta1 = om_z1 + (c1 / (eta**2)) * om_z0
    om_zeta2 = om_z2 + (c1 / (eta**2)) * om_z1
    om_zeta3 = om_z3 + (c1 / (eta**2)) * om_z2 + gd
    om_zeta4 = om_z4 + (c1 / (eta**2)) * om_z3 - gd * xi - gd * hd
    om_zeta5 = om_z5 + (c1 / (eta**2)) * om_z4 + gd * hd * xi + gd * (hd * hd - fd)

    avg_om_zeta0 = -g0 * om_zeta0
    avg_om_zeta1 = -1.5 * g0 * om_zeta1
    avg_om_zeta2 = -3.0 * (g0 * om_zeta2 + g2 * om_zeta0)
    avg_om_zeta3 = 3.0 * (g0 * om_zeta3 + g2 * om_zeta1 + g3 * om_zeta0)
    avg_om_zeta4 = 3.0 * (g0 * om_zeta4 + g2 * om_zeta2 + g3 * om_zeta1 + g4 * om_zeta0)
    avg_om_zeta5 = 1.5 * (g0 * om_zeta5 + g2 * om_zeta3 + g3 * om_zeta2 + g4 * om_zeta1 + g5 * om_zeta0)

    alpha_coeff = jnp.stack((avg_om_z0, avg_om_z1, avg_om_z2, avg_om_z3, avg_om_z4, avg_om_z5), axis=-1)
    alpha_coeff = alpha_coeff * jnp.array([1, 1, 1, 1, 1, 0], dtype=alpha_coeff.dtype)
    alpha_primitives = msa_phase_primitives(v, eta, c1, sav_sq)
    if anchor_reference:
        alpha_primitives = alpha_primitives - msa_phase_primitives(v0, eta, c1, sav_sq)
    alpha_secular = (alpha_primitives * alpha_coeff).sum(axis=-1)

    def epsilon_secular(x):
        return eta * (
            avg_om_zeta0 * x**-3
            + avg_om_zeta1 * x**-2
            + avg_om_zeta2 / x
            + avg_om_zeta3 * jnp.log(x)
            + avg_om_zeta4 * x
        )

    eps_secular = epsilon_secular(v)
    if anchor_reference:
        eps_secular = eps_secular - epsilon_secular(v0)

    eps_val = jnp.finfo(v.dtype).eps
    initial_parameter = jnp.clip((sp_sq - sm_sq) / (sp_sq - s3_sq), 0.0, 1.0 - eps_val)
    initial_fraction = jnp.clip((sp_sq - s0_sq) / (sp_sq - sm_sq), 0.0, 1.0)
    orientation = s1[..., 0] * s2[..., 1] - s1[..., 1] * s2[..., 0]
    sign = jnp.where(orientation < 0, -jnp.ones_like(orientation), jnp.ones_like(orientation))
    psi_ref = sign * elliptic_f(jnp.arcsin(jnp.sqrt(initial_fraction)), initial_parameter)
    psi = msa_nutation_phase(v, v0, eta, delta, q_c, xi, c1, sav_sq, s1_sq, s2_sq, psi_ref)
    l = eta / v
    j_sq = l * l + 2.0 * c1 / v + sav_sq
    j = jnp.sqrt(jnp.maximum(j_sq, jnp.finfo(v.dtype).tiny))
    s3, sm, sp, _ = solve_spin_roots(l, j_sq, s1_sq, s2_sq, xi, q_c, eta, delta)
    parameter = jnp.clip((sp - sm) / (sp - s3), 0.0, 1.0 - eps_val)
    sn, _, _ = jacobi_sn_cn_dn(psi, parameter)
    spin_sq = sp + (sm - sp) * sn * sn
    beta = msa_opening_angle(l_pn, eta, c1, sav_sq, spin_sq)
    alpha_osc, eps_osc = msa_oscillatory_angles(v, j, l, xi, eta, s1_sq, s2_sq, delta, s3, sm, sp, psi)
    alpha = alpha_secular + alpha_osc
    epsilon = eps_secular + eps_osc
    if anchor_reference:
        alpha_osc0, eps_osc0 = msa_oscillatory_angles(
            v0, jnp.sqrt(j0_sq), l0, xi, eta, s1_sq, s2_sq, delta, s3_sq, sm_sq, sp_sq, psi_ref
        )
        alpha = alpha - alpha_osc0
        epsilon = epsilon - eps_osc0
    if return_averages:
        return alpha, beta, -epsilon, valid, sav_sq, s1_ell_avg, s2_ell_avg
    return alpha, beta, -epsilon, valid


def compute_msa_initialization_validity(
    v0: jnp.ndarray,
    m1: jnp.ndarray,
    m2: jnp.ndarray,
    s1x: jnp.ndarray,
    s1y: jnp.ndarray,
    s1z: jnp.ndarray,
    s2x: jnp.ndarray,
    s2y: jnp.ndarray,
    s2z: jnp.ndarray,
) -> jnp.ndarray:
    """Return the published two-spin MSA initialization-validity mask."""
    (v0, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z) = jnp.broadcast_arrays(
        v0, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z
    )
    total_mass = m1 + m2
    m1_norm, m2_norm = m1 / total_mass, m2 / total_mass
    eta = m1_norm * m2_norm
    delta = m1_norm - m2_norm
    q_c = m2_norm / m1_norm
    s1 = jnp.expand_dims(m1_norm**2, -1) * jnp.stack(
        (s1x, s1y, s1z), axis=-1
    )
    s2 = jnp.expand_dims(m2_norm**2, -1) * jnp.stack(
        (s2x, s2y, s2z), axis=-1
    )
    s1_sq = (s1 * s1).sum(axis=-1)
    s2_sq = (s2 * s2).sum(axis=-1)
    s0_sq = ((s1 + s2) ** 2).sum(axis=-1)
    xi = (1.0 + q_c) * s1[..., 2] + (1.0 + 1.0 / q_c) * s2[..., 2]
    l0 = eta / v0
    j0_sq = l0 * l0 + 2.0 * l0 * (s1[..., 2] + s2[..., 2]) + s0_sq
    _, sm_sq, sp_sq, roots_valid = solve_spin_roots(
        l0, j0_sq, s1_sq, s2_sq, xi, q_c, eta, delta
    )
    sav0_sq = (sp_sq + sm_sq) / 2.0
    c10 = v0 * (j0_sq - l0 * l0 - sav0_sq) / 2.0
    cp0 = eta * eta * sp_sq - c10 * c10
    cm0 = eta * eta * sm_sq - c10 * c10
    return roots_valid & (s1_sq > 0) & (s2_sq > 0) & (cp0 > 0) & (cm0 > 0)


def compute_msa_averaged_couplings(
    v0: jnp.ndarray,
    m1: jnp.ndarray,
    m2: jnp.ndarray,
    s1x: jnp.ndarray,
    s1y: jnp.ndarray,
    s1z: jnp.ndarray,
    s2x: jnp.ndarray,
    s2y: jnp.ndarray,
    s2z: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the MSA couplings needed by the precession-averaged remnant."""
    (v0, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z) = jnp.broadcast_arrays(
        v0, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z
    )
    total_mass = m1 + m2
    m1_norm, m2_norm = m1 / total_mass, m2 / total_mass
    eta = m1_norm * m2_norm
    delta = m1_norm - m2_norm
    q_c = m2_norm / m1_norm
    s1_sq_norm = m1_norm * m1_norm
    s2_sq_norm = m2_norm * m2_norm
    s1 = jnp.expand_dims(s1_sq_norm, -1) * jnp.stack((s1x, s1y, s1z), axis=-1)
    s2 = jnp.expand_dims(s2_sq_norm, -1) * jnp.stack((s2x, s2y, s2z), axis=-1)
    s_ell = s1[..., 2] + s2[..., 2]
    sigma_ell = m2_norm * s2z - m1_norm * s1z
    l0 = compute_l_3pn(v0, eta, delta, s_ell, sigma_ell)
    result = evaluate_msa_dynamics(
        v0, v0, eta, delta, q_c, s1, s2, l0,
        anchor_reference=True, return_averages=True,
    )
    _, _, _, valid, sav_sq, s1_ell_avg, s2_ell_avg = result
    return sav_sq, s1_ell_avg, s2_ell_avg, valid


def compute_msa_final_spin_squared(
    s_av_sq: jnp.ndarray,
    s1_l_avg: jnp.ndarray,
    s2_l_avg: jnp.ndarray,
    l_fin: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the published MSA ``<S_fin^2>`` identity."""
    return s_av_sq + l_fin * l_fin + 2.0 * l_fin * (s1_l_avg + s2_l_avg)


def compute_msa_final_spin(
    v0: jnp.ndarray,
    m1: jnp.ndarray,
    m2: jnp.ndarray,
    s1x: jnp.ndarray,
    s1y: jnp.ndarray,
    s1z: jnp.ndarray,
    s2x: jnp.ndarray,
    s2y: jnp.ndarray,
    s2z: jnp.ndarray,
) -> jnp.ndarray:
    """Return the signed Kerr parameter from the published MSA remnant path.

    Nonaligned equal-mass and genuinely invalid-root rows are outside this
    prescription.  Eager calls raise ``ValueError``; traced/JIT execution
    reports the same diagnostic through JAX's compiled callback error.
    """
    from jaxwave.approximants.imrphenom_xas import XASParameterFits

    (v0, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z) = jnp.broadcast_arrays(
        as_array(v0, dtype=jnp.float64),
        as_array(m1, dtype=jnp.float64),
        as_array(m2, dtype=jnp.float64),
        as_array(s1x, dtype=jnp.float64),
        as_array(s1y, dtype=jnp.float64),
        as_array(s1z, dtype=jnp.float64),
        as_array(s2x, dtype=jnp.float64),
        as_array(s2y, dtype=jnp.float64),
        as_array(s2z, dtype=jnp.float64),
    )
    total_mass = m1 + m2
    eta = m1 * m2 / (total_mass * total_mass)
    a_aligned = XASParameterFits.fit_final_spin(eta, s1z, s2z)
    m_final_fit = 1.0 - XASParameterFits.fit_erad(eta, s1z, s2z)

    sav_sq, s1_l_avg, s2_l_avg, valid = compute_msa_averaged_couplings(
        v0, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z
    )
    inplane_sq = s1x * s1x + s1y * s1y + s2x * s2x + s2y * s2y
    eps = jnp.finfo(v0.dtype).eps
    numerical_degeneracy = inplane_sq <= 10.0 * eps * (
        1.0 + s1z * s1z + s2z * s2z
    )
    aligned = (s1x == 0.0) & (s1y == 0.0) & (s2x == 0.0) & (s2y == 0.0)
    unsupported = ((m1 == m2) & (~aligned)) | (
        (~valid) & (~numerical_degeneracy)
    )
    _enforce_msa_final_spin_domain(unsupported)

    a_parallel = a_aligned
    m_final = m_final_fit
    s_parallel = s1_l_avg + s2_l_avg
    l_fin = (m_final**2) * a_parallel - s_parallel
    s_fin_sq = compute_msa_final_spin_squared(sav_sq, s1_l_avg, s2_l_avg, l_fin)
    magnitude = jnp.sqrt(jnp.maximum(s_fin_sq, 0.0)) / (m_final**2)
    signed = jnp.where(a_parallel < 0.0, -magnitude, magnitude)

    return jnp.where(numerical_degeneracy, a_aligned, signed)


def evaluate_msa_precession(
    *args,
    **kwargs,
) -> Tuple[jnp.ndarray, ...]:
    """Evaluate Euler angles, observer inclination in J, and basis rotation.

    Supports two calling styles:
    1) High-level: evaluate_msa_precession(f, mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, ...)
       returning (alpha, cos_beta, gamma) by default.
    2) Low-level: evaluate_msa_precession(v, v0, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, iota, phi_ref, ...)
       returning (alpha, beta, gamma, theta_jn, zeta) or (alpha, beta, gamma, theta_jn, zeta, carrier_phase).
    """
    is_v_mode = "v" in kwargs or (len(args) >= 12 and not ("mass1" in kwargs or "f" in kwargs))

    if is_v_mode:
        v = kwargs.get("v", args[0] if len(args) > 0 else None)
        v0 = kwargs.get("v0", args[1] if len(args) > 1 else None)
        m1 = kwargs.get("m1", args[2] if len(args) > 2 else None)
        m2 = kwargs.get("m2", args[3] if len(args) > 3 else None)
        s1x = kwargs.get("s1x", args[4] if len(args) > 4 else 0.0)
        s1y = kwargs.get("s1y", args[5] if len(args) > 5 else 0.0)
        s1z = kwargs.get("s1z", args[6] if len(args) > 6 else 0.0)
        s2x = kwargs.get("s2x", args[7] if len(args) > 7 else 0.0)
        s2y = kwargs.get("s2y", args[8] if len(args) > 8 else 0.0)
        s2z = kwargs.get("s2z", args[9] if len(args) > 9 else 0.0)
        iota = kwargs.get("iota", args[10] if len(args) > 10 else 0.0)
        phi_ref = kwargs.get("phi_ref", args[11] if len(args) > 11 else 0.0)
        prec_version = kwargs.get("prec_version", 223)
        convention = kwargs.get("convention", 1)
        return_carrier_phase = kwargs.get("return_carrier_phase", False)
        observer_azimuth = kwargs.get("observer_azimuth", None)
        return_angles_only = kwargs.get("return_angles_only", False)
        return_cos_beta = kwargs.get("return_cos_beta", False)
    else:
        f = kwargs.get("f", args[0] if len(args) > 0 else None)
        mass1 = kwargs.get("mass1", args[1] if len(args) > 1 else 30.0)
        mass2 = kwargs.get("mass2", args[2] if len(args) > 2 else 30.0)
        s1x = kwargs.get("spin1x", kwargs.get("s1x", args[3] if len(args) > 3 else 0.0))
        s1y = kwargs.get("spin1y", kwargs.get("s1y", args[4] if len(args) > 4 else 0.0))
        s1z = kwargs.get("spin1z", kwargs.get("s1z", args[5] if len(args) > 5 else 0.0))
        s2x = kwargs.get("spin2x", kwargs.get("s2x", args[6] if len(args) > 6 else 0.0))
        s2y = kwargs.get("spin2y", kwargs.get("s2y", args[7] if len(args) > 7 else 0.0))
        s2z = kwargs.get("spin2z", kwargs.get("s2z", args[8] if len(args) > 8 else 0.0))
        iota = kwargs.get("inclination", kwargs.get("iota", args[9] if len(args) > 9 else 0.0))
        phi_ref = kwargs.get("phic", kwargs.get("phi_ref", args[10] if len(args) > 10 else 0.0))
        f_ref = kwargs.get("f_ref", args[11] if len(args) > 11 else None)
        prec_version = kwargs.get("prec_version", 223)
        convention = kwargs.get("convention", 1)
        return_carrier_phase = kwargs.get("return_carrier_phase", False)
        observer_azimuth = kwargs.get("phi0", kwargs.get("observer_azimuth", None))
        return_full = kwargs.get("return_full", False)
        return_angles_only = not return_full
        return_cos_beta = True

        m1 = as_array(mass1, dtype=jnp.float64)
        m2 = as_array(mass2, dtype=jnp.float64)
        total_mass = m1 + m2
        m_sec = total_mass * MSUN_SEC
        f_ref_val = as_array(20.0 if f_ref is None else f_ref, dtype=jnp.float64)
        f_arr = as_array(f, dtype=jnp.float64)
        f_safe = jnp.maximum(f_arr, 1e-8)
        v = (PI * m_sec * f_safe) ** (1.0 / 3.0)
        v0 = (PI * m_sec * f_ref_val) ** (1.0 / 3.0)

    prec_version = resolve_precession_version("msa", prec_version)
    if convention not in (0, 1, 5, 6, 7):
        raise ValueError(f"Unsupported precession convention: {convention}.")

    m1 = as_array(m1, dtype=jnp.float64)
    m2 = as_array(m2, dtype=jnp.float64)
    v = as_array(v, dtype=jnp.float64)
    v0 = as_array(v0, dtype=jnp.float64)
    s1x = as_array(s1x, dtype=jnp.float64)
    s1y = as_array(s1y, dtype=jnp.float64)
    s1z = as_array(s1z, dtype=jnp.float64)
    s2x = as_array(s2x, dtype=jnp.float64)
    s2y = as_array(s2y, dtype=jnp.float64)
    s2z = as_array(s2z, dtype=jnp.float64)
    iota = as_array(iota, dtype=jnp.float64)
    phi_ref = as_array(phi_ref, dtype=jnp.float64)

    total_mass = m1 + m2
    m1_norm = m1 / total_mass
    m2_norm = m2 / total_mass
    eta = m1 * m2 / (total_mass**2)
    delta = (m1 - m2) / total_mass
    q_c = m2 / m1

    s1_norm2 = m1_norm * m1_norm
    s2_norm2 = m2_norm * m2_norm
    s1x_g, s1y_g, s1z_g = s1_norm2 * s1x, s1_norm2 * s1y, s1_norm2 * s1z
    s2x_g, s2y_g, s2z_g = s2_norm2 * s2x, s2_norm2 * s2y, s2_norm2 * s2z

    in_plane_sq = s1x * s1x + s1y * s1y + s2x * s2x + s2y * s2y
    is_aligned = in_plane_sq == 0

    s_ell = s1z_g + s2z_g
    sigma_ell = m2_norm * s2z - m1_norm * s1z

    use_3pn_l = prec_version in (102, 220, 221, 222, 223, 224)
    if use_3pn_l:
        l0 = compute_l_3pn(v0, eta, delta, s_ell, sigma_ell)
        l_dyn = compute_l_3pn(v, eta, delta, s_ell, sigma_ell)
    elif prec_version == 101:
        l0 = IMRPhenomP.orbital_angular_momentum(v0, eta)
        l_dyn = IMRPhenomP.orbital_angular_momentum(v, eta)
    else:
        l0 = compute_l_4pn(v0, eta, delta, s_ell, sigma_ell, s1z, s2z, prec_version == 104)
        l_dyn = compute_l_4pn(v, eta, delta, s_ell, sigma_ell, s1z, s2z, prec_version == 104)

    j0x = s1x_g + s2x_g
    j0y = s1y_g + s2y_g
    j0z = l0 + s1z_g + s2z_g
    use_msa = prec_version in (220, 221, 222, 223, 224)
    if use_msa:
        alpha_msa_dyn, beta_msa, gamma_msa_dyn, roots_valid = evaluate_msa_dynamics(
            v,
            v0,
            eta,
            delta,
            q_c,
            jnp.stack((s1x_g, s1y_g, s1z_g), axis=-1),
            jnp.stack((s2x_g, s2y_g, s2z_g), axis=-1),
            l_dyn,
            anchor_reference=convention not in (5, 7),
        )
        fallback_to_nnlo = ~roots_valid
    else:
        fallback_to_nnlo = jnp.ones_like(eta, dtype=bool)
        alpha_msa_dyn = gamma_msa_dyn = beta_msa = jnp.zeros_like(v)

    chip = compute_effective_chip(m1, m2, s1x, s1y, s2x, s2y)
    chi_l = m1_norm * s1z + m2_norm * s2z

    (
        a_m3,
        a_m2,
        a_m1,
        a_1,
        a_log,
        e_m3,
        e_m2,
        e_m1,
        e_1,
        e_log,
    ) = IMRPhenomP.nnlo_angle_coefficients(
        eta=eta,
        delta=delta,
        m1_norm=m1_norm,
        chip=chip,
        chi_l=chi_l,
    )

    alpha_nnlo_dyn, epsilon_nnlo_dyn = IMRPhenomP.evaluate_nnlo_angles(
        v=v,
        v_ref=as_array(v0, dtype=v.dtype),
        a_minus3=a_m3,
        a_minus2=a_m2,
        a_minus1=a_m1,
        a_1=a_1,
        a_log=a_log,
        e_minus3=e_m3,
        e_minus2=e_m2,
        e_minus1=e_m1,
        e_1=e_1,
        e_log=e_log,
        anchor_reference=convention not in (5, 7),
    )
    sp_geom = chip * (m1_norm * m1_norm)
    sl_geom = (m1_norm * m1_norm) * s1z + (m2_norm * m2_norm) * s2z
    l_plus_sl = l_dyn + sl_geom
    beta_nnlo = jnp.arctan2(sp_geom, l_plus_sl)

    # Frame geometry at reference frequency f_ref
    j0_perp = jnp.hypot(j0x, j0y)
    has_j0_perp = j0_perp > 0
    theta_jl0 = jnp.arctan2(j0_perp, j0z)
    phi_jl0 = jnp.arctan2(
        jnp.where(has_j0_perp, j0y, jnp.zeros_like(j0y)),
        jnp.where(has_j0_perp, j0x, jnp.ones_like(j0x)),
    )

    sin_i = jnp.sin(iota)
    cos_i = jnp.cos(iota)
    source_phase = phi_ref if observer_azimuth is None else phi_ref - observer_azimuth
    sin_pref = jnp.sin(source_phase)
    cos_pref = jnp.cos(source_phase)

    nx_l0 = sin_i * sin_pref
    ny_l0 = sin_i * cos_pref
    nz_l0 = cos_i

    cos_tjl = jnp.cos(theta_jl0)
    sin_tjl = jnp.sin(theta_jl0)
    cos_pjl = jnp.cos(phi_jl0)
    sin_pjl = jnp.sin(phi_jl0)

    nx_rot_z = nx_l0 * cos_pjl + ny_l0 * sin_pjl
    ny_rot_z = -nx_l0 * sin_pjl + ny_l0 * cos_pjl
    nz_rot_z = nz_l0

    nx_jp = nx_rot_z * cos_tjl - nz_rot_z * sin_tjl
    ny_jp = ny_rot_z
    nz_jp = nx_rot_z * sin_tjl + nz_rot_z * cos_tjl

    has_n_perp = (nx_jp != 0) | (ny_jp != 0)
    kappa = jnp.arctan2(
        jnp.where(has_n_perp, ny_jp, jnp.zeros_like(ny_jp)),
        jnp.where(has_n_perp, nx_jp, jnp.ones_like(nx_jp)),
    )

    alpha_0_xp = PI - kappa
    epsilon_0_xp = phi_jl0 - PI
    jx_n = -cos_i * (sin_pref * j0x + cos_pref * j0y) + sin_i * j0z
    jy_n = cos_pref * j0x - sin_pref * j0y
    j_perp_n_sq = jx_n * jx_n + jy_n * jy_n
    j_norm = jnp.linalg.norm(jnp.stack((j0x, j0y, j0z), axis=-1), axis=-1)

    lx_j_scaled = cos_i * (j0x * j0x + j0y * j0y) - sin_i * j0z * (sin_pref * j0x + cos_pref * j0y)
    ly_j_scaled = sin_i * j_norm * jy_n
    has_l_perp_j = has_j0_perp & (j_perp_n_sq > 0)
    alpha_0_pv2 = jnp.arctan2(
        jnp.where(has_l_perp_j, ly_j_scaled, jnp.sin(alpha_0_xp)),
        jnp.where(has_l_perp_j, lx_j_scaled, jnp.cos(alpha_0_xp)),
    )
    alpha_0 = alpha_0_pv2 if convention in (1, 5) else alpha_0_xp
    epsilon_0 = epsilon_0_xp if convention in (1, 6) else jnp.zeros_like(phi_jl0)
    carrier_phase = (
        phi_jl0 if convention == 0 else
        jnp.zeros_like(phi_ref) if convention == 1 else phi_ref
    )

    alpha_msa_final = alpha_msa_dyn + alpha_0
    gamma_msa_final = gamma_msa_dyn - epsilon_0

    alpha_nnlo_final = alpha_nnlo_dyn + alpha_0
    epsilon_nnlo_final = epsilon_nnlo_dyn + epsilon_0
    gamma_nnlo_final = -epsilon_nnlo_final

    fb_mask = (
        jnp.expand_dims(fallback_to_nnlo, -1)
        if (v.ndim > fallback_to_nnlo.ndim)
        else fallback_to_nnlo
    )
    alpha = jnp.where(fb_mask, alpha_nnlo_final, alpha_msa_final)
    beta = jnp.where(fb_mask, beta_nnlo, beta_msa)
    gamma = jnp.where(fb_mask, gamma_nnlo_final, gamma_msa_final)

    theta_jn = jnp.arctan2(jnp.hypot(nx_jp, ny_jp), nz_jp)

    px_l0 = -cos_i * sin_pref
    py_l0 = -cos_i * cos_pref
    pz_l0 = sin_i

    px_rot_z = px_l0 * cos_pjl + py_l0 * sin_pjl
    py_rot_z = -px_l0 * sin_pjl + py_l0 * cos_pjl
    pz_rot_z = pz_l0

    px_jp = px_rot_z * cos_tjl - pz_rot_z * sin_tjl
    py_jp = py_rot_z
    pz_jp = px_rot_z * sin_tjl + pz_rot_z * cos_tjl

    cos_k = jnp.cos(kappa)
    sin_k = jnp.sin(kappa)
    xn_x = px_jp * cos_k + py_jp * sin_k
    xn_y = -px_jp * sin_k + py_jp * cos_k
    xn_z = pz_jp

    cos_tjn_b = jnp.cos(theta_jn)
    sin_tjn_b = jnp.sin(theta_jn)
    xn_dot_xp = xn_x * cos_tjn_b - xn_z * sin_tjn_b
    xn_dot_yp = xn_y

    zeta = jnp.arctan2(xn_dot_yp, xn_dot_xp)

    if return_angles_only or return_cos_beta:
        return alpha, jnp.cos(beta), gamma
    if return_carrier_phase:
        return alpha, beta, gamma, theta_jn, zeta, carrier_phase
    return alpha, beta, gamma, theta_jn, zeta


__all__ = [
    "resolve_precession_version",
    "compute_effective_chip",
    "elliptic_f",
    "jacobi_sn_cn_dn",
    "msa_phase_primitives",
    "msa_nutation_phase",
    "msa_opening_angle",
    "msa_oscillatory_angles",
    "compute_l_3pn",
    "compute_l_4pn",
    "solve_spin_roots",
    "evaluate_msa_dynamics",
    "compute_msa_averaged_couplings",
    "compute_msa_initialization_validity",
    "compute_msa_final_spin_squared",
    "compute_msa_final_spin",
    "evaluate_msa_precession",
]
