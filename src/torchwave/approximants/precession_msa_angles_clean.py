"""Diagnostic, clean-room MSA angle assembly.

This module composes the independently reviewed MSA primitives.  It is kept
deliberately separate from the waveform approximants and contains no public
waveform compatibility or fitted coefficients.
"""
from __future__ import annotations

import math
import torch

from .precession_msa_initial_clean import (
    initial_invariants, cubic_coefficients, ordered_roots, initial_phase,
    spin_oscillation, elliptic_K_agm, elliptic_E_quadrature,
)
from .precession_msa_pn_clean import (
    precession_averages, flux_coefficients, reciprocal_coefficients,
    precession_averages_v223, initial_reference_spin_couplings,
    psi_coefficients, anchored_psi,
)
from .precession_msa_secular_clean import averaged_coefficients, anchored_angles
from .precession_msa_oscillatory_clean import (
    rational_coefficients, oscillatory_correction, angle_factors,
    _partial_fractions_v223,
)
from .precession_beta_clean import orbital_angular_momentum
from .precession_msa_reference_l_clean import orbital_angular_momentum_nonconserved_3pn

MTSUN_SI = 4.925490947641267e-6
V223_AUXILIARY_ROOT_GAP = 1.0e-9


def _compatibility_value(value):
    if isinstance(value, bool):
        return "v223" if value else "equation"
    if isinstance(value, (int, float)):
        if float(value) == 223.0:
            return "v223"
        raise ValueError("version must be 223 for the native compatibility path")
    value = str(value).lower().replace("-", "_")
    if value in ("equation", "published", "default", "clean", "none"):
        return "equation"
    if value in ("v223", "223", "native", "native_msa", "msa_v223",
                 "compat", "compatibility"):
        return "v223"
    raise ValueError("compatibility must be 'equation' or 'v223'")


def _compatibility_variant(*, compatibility=None, variant=None,
                           convention=None, version=None, v223=False):
    """Resolve the explicit equation/v223 selector for angle assembly."""
    supplied = [x for x in (compatibility, variant, convention, version)
                if x is not None]
    values = [_compatibility_value(x) for x in supplied]
    if values and any(x != values[0] for x in values[1:]):
        raise ValueError("compatibility selectors disagree")
    return values[0] if values else ("v223" if v223 else "equation")


def _safe_cubic_roots(B, C, D):
    """Evaluate the trigonometric cubic roots and return validity masks.

    ``ordered_roots`` deliberately rejects repeated roots.  The native
    compatibility branch needs to identify those roots before making its
    explicit zero-nutation choice, so this helper evaluates the same formula
    with finite limiting arguments and reports ``(roots, valid, degenerate)``.
    Nondegenerate invalid cubics are still rejected by the caller.
    """
    B, C, D = torch.broadcast_tensors(*(torch.as_tensor(x, dtype=torch.float64)
                                         for x in (B, C, D)))
    finite = torch.isfinite(B) & torch.isfinite(C) & torch.isfinite(D)
    # Keep invalid rows finite while evaluating the branch that will be
    # replaced by the caller.  This avoids ``0*NaN`` in a later masked result.
    B_eval = torch.where(finite, B, torch.zeros_like(B))
    C_eval = torch.where(finite, C, torch.zeros_like(C))
    D_eval = torch.where(finite, D, torch.zeros_like(D))
    p = C - B.square() / 3.0
    p_eval = C_eval - B_eval.square() / 3.0
    p_safe = torch.where(torch.isfinite(p_eval) & (p_eval < 0.0), p_eval,
                         -torch.ones_like(p_eval))
    q_eval = 2.0 * B_eval.pow(3) / 27.0 - B_eval * C_eval / 3.0 + D_eval
    raw = 3.0 * q_eval / (2.0 * p_safe) * torch.sqrt(-3.0 / p_safe)
    raw_finite = torch.isfinite(raw)
    raw_safe = torch.where(raw_finite, raw, torch.zeros_like(raw))
    # Ordinary rows use the exact clamped cubic argument.  At an endpoint,
    # detach the argument before acos so the value is exact while its inactive
    # one-sided derivative remains finite.  No inward epsilon changes roots.
    raw_clamped = raw_safe.clamp(-1.0, 1.0)
    endpoint = raw_clamped.abs() >= 1.0
    raw_for_acos = torch.where(endpoint, raw_clamped.detach(), raw_clamped)
    theta = torch.acos(raw_for_acos) / 3.0
    k = torch.arange(3, dtype=B.dtype, device=B.device)
    roots = -B[..., None] / 3.0 + 2.0 * torch.sqrt(
        -p_safe[..., None] / 3.0
    ) * torch.cos(theta[..., None] - 2.0 * torch.pi * k / 3.0)
    roots = torch.sort(roots, dim=-1).values
    # The lower cubic root is an auxiliary root.  The two physical spin roots
    # are magnitudes, and can acquire a harmless negative sign from roundoff
    # at a singular boundary.
    physical = torch.sort(roots[..., 1:].abs(), dim=-1).values
    roots = torch.cat((roots[..., :1], physical), dim=-1)
    # A finite trigonometric argument outside [-1, 1] is still a valid native
    # endpoint continuation: ``raw_clamped`` above supplies the real cubic
    # limit.  Only undefined/nonfinite inputs remain invalid here.
    valid = finite & (p < 0.0) & raw_finite
    gap = roots[..., 2] - roots[..., 1]
    degenerate = torch.isfinite(gap) & (gap.abs() < 1.0e-5)
    return roots, valid, degenerate


def _invalid_rational_rows(values):
    """Identify rows outside the real pole-factor domain.

    The native v223 continuation keeps the algebraic quadratic labels and
    evaluates ``sqrt(abs(discriminant))`` and ``sqrt(abs(1-lambda))``.  A
    negative pole factor therefore is finite continuation data, rather than
    an invalid row.  Only nonfinite coefficients, vanishing endpoint factors,
    or a repeated pole are suppressed here.  The equation helper remains
    strict and performs its own ``lambda < 1`` validation.
    """
    c0, c2, c4, d0, d2, d4 = (values[i] for i in (2, 3, 4, 5, 6, 7))
    disc = d2.square() - 4.0 * d0 * d4
    finite = torch.ones_like(values[0], dtype=torch.bool)
    for value in values:
        finite = finite & torch.isfinite(value)
    valid = finite & (d0 != 0.0) & (d4 != 0.0) & (disc != 0.0)
    # The v223 quadratic continuation uses the real ``sqrt(abs(disc))``
    # branch.  Keep the strict equation helper's positivity check separate;
    # a negative discriminant here is finite continuation data.
    disc_safe = torch.where(torch.isfinite(disc) & (disc != 0.0),
                            disc.abs(), torch.ones_like(disc))
    root = torch.sqrt(disc_safe)
    denom = 2.0 * torch.where(d0 != 0.0, d0, torch.ones_like(d0))
    l_plus = (-d2 + root) / denom
    l_minus = (-d2 - root) / denom
    # Keep the quadratic ``+``/``-`` labels used by the v223 continuation;
    # positivity sorting would change which signed denominator belongs to the
    # second-pole contribution.
    l1, l2 = l_plus, l_minus
    valid = valid & torch.isfinite(l1) & torch.isfinite(l2)
    return ~valid

def elliptic_spin_average(splus2, sminus2, s32):
    """Cycle average of S² from elliptic turning points, stable at m=0."""
    sp, sm, s3 = torch.broadcast_tensors(*(torch.as_tensor(x, dtype=torch.float64) for x in (splus2, sminus2, s32)))
    m = (sp-sm)/(sp-s3)
    # K-E subtraction is ill-conditioned well before machine-zero m.
    safe_m = torch.where(m.abs() < 1e-8, torch.ones_like(m), m)
    K = elliptic_K_agm(m); E = elliptic_E_quadrature(m)
    ratio = (K-E)/(safe_m*K)
    return torch.where(m.abs() < 1e-8, (sp+sm)/2, sp-(sp-sm)*ratio)


def assemble_precession_msa_angles(
    f, mass1, mass2, chi1, chi2, *, f_ref=20.0, order=5,
    beta_pn=False, beta_pn_version=102, beta_compat_102=False,
    average_mode="leading", compatibility=None, variant=None,
    convention=None, version=None, v223=False,
):
    """Return raw MSA ``alpha``, ``gamma`` and ``cos_beta`` plus diagnostics.

    Masses are solar masses, frequencies are Hz, and spins are dimensionless
    Cartesian vectors.  The result is a dictionary of float64 tensors.  The
    The equation-level ``beta_pn`` branch is diagnostic only; selecting the
    explicit ``compatibility='v223'`` branch returns its native nonconserved
    3PN opening-beta value.  The spin cubic always uses the Newtonian orbital
    magnitude.
    """
    selected = _compatibility_variant(
        compatibility=compatibility, variant=variant,
        convention=convention, version=version, v223=v223,
    )
    native_v223 = selected == "v223"
    f = torch.as_tensor(f, dtype=torch.float64)
    if bool(torch.any(f <= 0)) or f.ndim == 0:
        raise ValueError("f must be a nonempty positive vector")
    f0 = torch.as_tensor(f_ref, dtype=torch.float64, device=f.device)
    if bool(torch.any(f0 <= 0)):
        raise ValueError("f_ref must be positive")
    # Canonicalize the full spin state before constructing invariants.
    m1, m2 = torch.as_tensor(mass1, dtype=torch.float64), torch.as_tensor(mass2, dtype=torch.float64)
    s1, s2 = (torch.as_tensor(x, dtype=torch.float64) for x in (chi1, chi2))
    if s1.shape[-1] != 3 or s2.shape[-1] != 3:
        raise ValueError("spin vectors must end in dimension 3")
    swap = m2 > m1
    cm1, cm2 = torch.where(swap, m2, m1), torch.where(swap, m1, m2)
    cs1 = torch.where(swap[..., None], s2, s1)
    cs2 = torch.where(swap[..., None], s1, s2)
    total = cm1 + cm2
    n1, n2 = cm1 / total, cm2 / total
    S1v, S2v = n1[..., None] ** 2 * cs1, n2[..., None] ** 2 * cs2
    v0 = (math.pi * total * MTSUN_SI * f0).pow(1 / 3)
    L0 = (n1 * n2) / v0
    orbital0 = torch.zeros((*torch.broadcast_shapes(S1v.shape[:-1], S2v.shape[:-1]), 3), dtype=torch.float64, device=f.device)
    orbital0[..., 2] = L0
    q, eta, delta, S1, S2, _, J0, xi = initial_invariants(cm1, cm2, cs1, cs2, orbital0)
    s1sq, s2sq = S1.square(), S2.square()
    J0sq = J0.square()
    # The native evaluator needs a finite root state for singular/aligned
    # initial data, while a small root gap by itself only suppresses the
    # nutation reconstruction.  Keep these decisions separate: secular
    # coefficients always see the physical q and the actual root invariants.
    Ssum2 = torch.sum((S1v + S2v) ** 2, dim=-1)
    if native_v223:
        # The one-spin/aligned regularization is the published limiting root
        # triple.  It is selected by geometry (or an invalid cubic), never by
        # equal mass or by the root-gap threshold.
        B, C, D = cubic_coefficients(q, eta, delta, s1sq, s2sq, L0, J0, xi)
        roots_candidate, roots_valid, roots_degenerate = _safe_cubic_roots(
            B, C, D
        )
        roots_candidate = torch.nan_to_num(
            roots_candidate, nan=0.0, posinf=0.0, neginf=0.0
        )
        spin1_zero = torch.sum(S1v.square(), dim=-1) == 0
        spin2_zero = torch.sum(S2v.square(), dim=-1) == 0
        spin1_aligned = torch.all(S1v[..., :2] == 0, dim=-1)
        spin2_aligned = torch.all(S2v[..., :2] == 0, dim=-1)
        root_regularized = (
            (~roots_valid) | spin1_zero | spin2_zero |
            spin1_aligned | spin2_aligned
        )
        fallback0 = torch.stack(
            (torch.zeros_like(Ssum2), Ssum2, Ssum2 + V223_AUXILIARY_ROOT_GAP), dim=-1
        )
        roots0 = torch.where(root_regularized[..., None], fallback0,
                             roots_candidate)
        safe_level = torch.clamp(Ssum2, min=1.0e-8)
        initial_gap = (roots0[..., 2] - roots0[..., 1]).abs()
        degenerate = torch.isfinite(initial_gap) & (initial_gap < 1.0e-5)
    else:
        B, C, D = cubic_coefficients(q, eta, delta, s1sq, s2sq, L0, J0, xi)
        roots0 = ordered_roots(B, C, D)
        degenerate = torch.zeros_like(q, dtype=torch.bool)
        root_regularized = torch.zeros_like(q, dtype=torch.bool)
        safe_level = Ssum2
        root_eps = torch.zeros_like(Ssum2)
        fallback0 = roots0
    s30, sm0, sp0 = roots0.unbind(-1)
    if average_mode == "leading":
        sav = (sp0 + sm0) / 2
    elif average_mode == "exact":
        sav = elliptic_spin_average(sp0, sm0, s30)
    else:
        raise ValueError("average_mode must be 'leading' or 'exact'")
    c1 = .5 * v0 * (J0sq - L0.square() - sav)
    triple = torch.sum(torch.cross(torch.nn.functional.normalize(orbital0, dim=-1), S1v, dim=-1) * S2v, dim=-1)
    if native_v223:
        delta_average = delta
        # Evaluate a finite auxiliary phase for the small-gap rows.  Its
        # oscillatory correction is masked below, while ordinary roots retain
        # their physical values and branch sign.
        phase_eps = torch.minimum(torch.full_like(safe_level, 1.0e-8),
                                  safe_level * 0.25)
        phase_roots = torch.stack((safe_level - 3.0 * phase_eps,
                                   safe_level - phase_eps,
                                   safe_level + phase_eps), dim=-1)
        phase_roots = torch.where(degenerate[..., None], phase_roots, roots0)
        psi0_raw = initial_phase(Ssum2, phase_roots[..., 2],
                                 phase_roots[..., 1], phase_roots[..., 0],
                                 triple)
        # The Eq. (51) limiting phase is installed after the flux and phase
        # coefficients below are available.  ``psi0_raw`` remains the
        # ordinary-root initial value.
        psi0 = psi0_raw
        av = precession_averages_v223(
            c1, xi, q, eta, sp0, sm0, s1sq, s2sq, v0,
        )
    else:
        psi0 = initial_phase(Ssum2, sp0, sm0, s30, triple)
        av = precession_averages(c1, xi, q, eta, sp0, sm0, s1sq, s2sq, v0)
    av["s12"] = (sav - s1sq - s2sq) / 2
    if native_v223:
        flux_couplings = initial_reference_spin_couplings(S1v, S2v)
        fc = flux_coefficients(
            eta, q, flux_couplings["s1l"], flux_couplings["s2l"],
            flux_couplings["s1sq"], flux_couplings["s2sq"],
            flux_couplings["s12"], flux_couplings["s1l2"],
            flux_couplings["s2l2"], flux_couplings["s1ls2l"],
        )
    else:
        flux_couplings = {
            "s1l": av["s1l"], "s2l": av["s2l"],
            "s1sq": s1sq, "s2sq": s2sq, "s12": av["s12"],
            "s1l2": av["s1l2"], "s2l2": av["s2l2"],
            "s1ls2l": av["s1ls2l"],
        }
        fc = flux_coefficients(
            eta, q, flux_couplings["s1l"], flux_couplings["s2l"],
            flux_couplings["s1sq"], flux_couplings["s2sq"],
            flux_couplings["s12"], flux_couplings["s1l2"],
            flux_couplings["s2l2"], flux_couplings["s1ls2l"],
        )
    g = reciprocal_coefficients(fc)
    # Keep the no-keyword call in the default path: several independent
    # diagnostics monkeypatch this published helper with a literal function.
    if native_v223:
        # Exact equal mass makes the root-asymptotic phase coefficients a
        # removable 0/0 representation.  Its phase is masked by the same
        # small-gap rule, so evaluate only that inactive coefficient branch at
        # a nearby finite q.  Secular averages and all invariants above keep
        # the physical q unchanged.
        q_phase = torch.where(q == 1.0, torch.full_like(q, 1.0 - 1.0e-6), q)
        delta_phase = (1.0 - q_phase) / (1.0 + q_phase)
        psi1, psi2 = psi_coefficients(
            eta, delta_phase, q_phase, xi, c1, sav, s1sq, s2sq,
            g["g0"], g["g2"], compatibility="v223", regularize=True,
        )
        psi1_independent, psi2_independent = psi_coefficients(
            eta, delta_phase, q_phase, xi, c1, sav, s1sq, s2sq,
            g["g0"], g["g2"], compatibility="equation", regularize=True,
        )
    else:
        psi1, psi2 = psi_coefficients(
            eta, delta, q, xi, c1, sav, s1sq, s2sq,
            g["g0"], g["g2"],
        )
        psi1_independent, psi2_independent = psi1, psi2
    if native_v223:
        # Eq. (51) is written as an unanchored phase ``P(v) + C``.  A small
        # *initial* root gap suppresses the nutation integration constant, so
        # C=0 and the phase is P(v), including at v_ref.  ``anchored_psi``
        # uses the equivalent representation ``psi0 + P(v)-P(v_ref)``;
        # therefore its initial target is P(v_ref), rather than zero total
        # phase.  Keep the physical mass-ratio delta in this limiting series.
        psi_ref = -3.0 * g["g0"] * delta / 4.0 * v0.pow(-3) * (
            1.0 + psi1 * v0 + psi2 * v0.square()
        )
        psi0 = torch.where(degenerate, psi_ref, psi0)
    f_grid = f if total.dim() == 0 else (f.unsqueeze(0) if f.dim() == 1 else f)
    total_u = total if total.dim() == 0 else total.unsqueeze(-1)
    v0_u = v0 if v0.dim() == 0 else v0.unsqueeze(-1)
    psi0_u = psi0 if psi0.dim() == 0 else psi0.unsqueeze(-1)
    delta_u = delta if delta.dim() == 0 else delta.unsqueeze(-1)
    g0_u = g["g0"] if g["g0"].dim() == 0 else g["g0"].unsqueeze(-1)
    psi1_u = psi1 if psi1.dim() == 0 else psi1.unsqueeze(-1)
    psi2_u = psi2 if psi2.dim() == 0 else psi2.unsqueeze(-1)

    psi_raw = anchored_psi(
        (math.pi * total_u * MTSUN_SI * f_grid).pow(1 / 3), v0_u, psi0_u,
        delta_u if native_v223 else delta_u, g0_u, psi1_u, psi2_u,
    )
    # Fixed reference roots and c1/Sav define the MSA expansion.
    if native_v223:
        coeff = list(averaged_coefficients(
            eta, xi, c1, s1sq, s2sq, delta, sp0, sm0,
            [g[f"g{i}"] for i in range(6)],
            regularize=True,
        ))
    else:
        coeff = list(averaged_coefficients(
            eta, xi, c1, s1sq, s2sq, delta, sp0, sm0,
            [g[f"g{i}"] for i in range(6)],
        ))
    if order == 5:
        coeff = [x.clone() for x in coeff]
        coeff[0][..., 5] = 0
        coeff[1][..., 5] = 0
    elif order != 6:
        raise ValueError("order must be 5 or 6")
    v = (math.pi * total_u * MTSUN_SI * f_grid).pow(1 / 3)
    L = eta[..., None] / v
    c1_current = c1[..., None]
    sav_current = sav[..., None]
    J = torch.sqrt(L.square() + 2 * c1_current / v + sav_current)
    if native_v223:
        q_current = q[..., None].expand_as(L)
        delta_current = delta[..., None].expand_as(L)
        s1sq_current = s1sq[..., None].expand_as(L)
        s2sq_current = s2sq[..., None].expand_as(L)
        xi_current = xi[..., None].expand_as(L)
        bb, cc, dd = cubic_coefficients(
            q_current, eta[..., None], delta_current,
            s1sq_current, s2sq_current, L, J, xi_current,
        )
        roots_candidate, current_valid, _ = _safe_cubic_roots(bb, cc, dd)
        roots_candidate = torch.nan_to_num(
            roots_candidate, nan=0.0, posinf=0.0, neginf=0.0
        )
        # Preserve the audited limiting triple exactly at the current
        # frequency: ``(0, S_initial^2, S_initial^2+1e-9)``.  A separately
        # floored phase triple above keeps inactive inverse-Jacobi arithmetic
        # finite; it must not become a physical spin floor here.
        level_current = Ssum2[..., None].expand_as(L)
        fallback_current = torch.stack(
            (torch.zeros_like(level_current), level_current,
             level_current + V223_AUXILIARY_ROOT_GAP), dim=-1
        )
        # The singular/aligned initial prescription is a zero-nutation
        # limiting state at every current frequency.  Keep its separated
        # auxiliary roots for the rational expressions; ordinary rows alone
        # follow the physical current cubic.
        # ``current_valid`` carries the frequency axis while
        # ``root_regularized`` is an intrinsic-row mask.  Keep both singleton
        # axes explicit so this broadcasts for scalar and batched rows alike.
        regularized_current = root_regularized[..., None, None]
        roots = torch.where(current_valid[..., None] & ~regularized_current,
                            roots_candidate, fallback_current)
        current_gap = (roots[..., 2] - roots[..., 1]).abs()
        freq_degenerate = torch.isfinite(current_gap) & (current_gap < 1.0e-5)
        # The auxiliary separated roots are used only to evaluate inactive
        # Jacobi/rational expressions.  The current physical upper root is
        # retained for the spin norm below.
        spin_roots = torch.where(freq_degenerate[..., None],
                                 fallback_current, roots)
    else:
        bb, cc, dd = cubic_coefficients(
            q[..., None], eta[..., None], delta[..., None],
            s1sq[..., None], s2sq[..., None], L, J, xi[..., None],
        )
        roots = ordered_roots(bb, cc, dd)
        freq_degenerate = torch.zeros_like(roots[..., 0], dtype=torch.bool)
        spin_roots = roots
    s3, sm, sp = roots.unbind(-1)
    spin_s3, spin_sm, spin_sp = spin_roots.unbind(-1)
    psi = anchored_psi(
        (math.pi * total_u * MTSUN_SI * f_grid).pow(1 / 3), v0_u, psi0_u,
        delta_u if native_v223 else delta_u, g0_u, psi1_u, psi2_u,
    )
    S2_raw = spin_oscillation(psi, spin_sp, spin_sm, spin_s3)
    S2 = torch.where(freq_degenerate, sp, S2_raw) if native_v223 else S2_raw
    sav_for_angles = sav_current
    coeff_0 = coeff[0].unsqueeze(-2) if coeff[0].dim() > 1 else coeff[0]
    coeff_1 = coeff[1].unsqueeze(-2) if coeff[1].dim() > 1 else coeff[1]
    secular_phi, secular_zeta = anchored_angles(
        v, v0_u, coeff_0, coeff_1, eta[..., None],
        c1[..., None],
        sav_for_angles,
    )
    if native_v223:
        rc = rational_coefficients(
            v, eta[..., None], xi_current, c1_current, s1sq_current,
            s2sq_current, delta_current, spin_sp, spin_sm, spin_s3, J=J, L=L,
            compatibility="v223",
        )
        # Keep the algebraic pole labels available to the clean diagnostics;
        # unlike the strict equation helper these may have negative factors.
        rational_lambda_a, rational_lambda_b, _, _, _ = _partial_fractions_v223(
            rc[2], rc[3], rc[4], rc[5], rc[6], rc[7]
        )
    else:
        rc = rational_coefficients(
            v, eta[..., None], xi[..., None], c1[..., None],
            s1sq[..., None], s2sq[..., None], delta[..., None],
            sp, sm, s3, J=J, L=L,
        )
        rational_lambda_a = torch.zeros_like(rc[0])
        rational_lambda_b = torch.zeros_like(rc[0])
    af, bf = angle_factors(
        J, L, spin_sp if native_v223 else sp,
        spin_sm if native_v223 else sm,
    )
    # rational_coefficients returns (J,a,c0,c2,c4,d0,d2,d4,psidot),
    # while the correction takes psidot immediately after J.
    rational_invalid_current = torch.zeros_like(freq_degenerate)
    rational_invalid_reference = torch.zeros_like(degenerate)

    def _osc(phase, values, atheta, btheta, suppress_mask=None):
        nonlocal rational_invalid_current, rational_invalid_reference
        if native_v223:
            rational_invalid = _invalid_rational_rows(values)
            if values[0].ndim == freq_degenerate.ndim:
                rational_invalid_current = rational_invalid_current | rational_invalid
            else:
                rational_invalid_reference = rational_invalid_reference | rational_invalid
            if suppress_mask is None:
                suppress_mask = rational_invalid
            else:
                suppress_mask = suppress_mask | rational_invalid
        if native_v223 and suppress_mask is not None:
            # Repeated roots are a deliberate no-oscillation branch.  Feed
            # the analytic integrator a harmless zero-residue rational
            # function before masking its result, so a pole which collapses
            # exactly at alignment cannot raise or create an inactive NaN.
            safe_values = (
                torch.ones_like(values[0]), torch.ones_like(values[1]),
                torch.zeros_like(values[2]), torch.zeros_like(values[3]),
                torch.zeros_like(values[4]), torch.ones_like(values[5]),
                torch.zeros_like(values[6]), -0.25 * torch.ones_like(values[7]),
                -torch.ones_like(values[8]),
            )
            values = tuple(torch.where(suppress_mask, safe, value)
                           for safe, value in zip(safe_values, values))
        rj, ra, rc0_, rc2, rc4, rd0, rd2, rd4, rp = values
        if native_v223:
            result = oscillatory_correction(
                phase, rj, rp, ra, rc0_, rc2, rc4, rd0, rd2, rd4,
                atheta, btheta, compatibility="v223",
            )
            return tuple(torch.where(suppress_mask, torch.zeros_like(value), value)
                         for value in result)
        return oscillatory_correction(
            phase, rj, rp, ra, rc0_, rc2, rc4,
            rd0, rd2, rd4, atheta, btheta,
        )
    osc_phi, osc_zeta = _osc(psi, rc, af, bf, freq_degenerate if native_v223 else None)
    if native_v223:
        # Build the reference oscillatory correction on a separated auxiliary
        # root triple whenever the initial correction is disabled.  This keeps
        # pole-factor arithmetic finite; the result is masked below.  The
        # active path uses the physical initial roots and invariants.
        osc_roots0 = torch.where(degenerate[..., None], phase_roots, roots0)
        osc_xi0 = torch.where(degenerate, torch.zeros_like(xi), xi)
        osc_c10 = torch.where(degenerate, torch.zeros_like(c1), c1)
        osc_s1sq0 = torch.where(degenerate, torch.zeros_like(s1sq), s1sq)
        osc_s2sq0 = torch.where(degenerate, torch.zeros_like(s2sq), s2sq)
        osc_delta0 = torch.where(degenerate, torch.zeros_like(delta), delta)
        osc_s30, osc_sm0, osc_sp0 = osc_roots0.unbind(-1)
        rc0 = rational_coefficients(
            v0, eta, osc_xi0, osc_c10, osc_s1sq0, osc_s2sq0,
            osc_delta0, osc_sp0, osc_sm0, osc_s30, J=J0, L=L0,
            compatibility="v223",
        )
        af0, bf0 = angle_factors(J0, L0, osc_sp0, osc_sm0)
    else:
        rc0 = rational_coefficients(
            v0, eta, xi, c1, s1sq, s2sq, delta,
            sp0, sm0, s30, J=J0, L=L0,
        )
        af0, bf0 = angle_factors(J0, L0, sp0, sm0)
    osc0_phi, osc0_zeta = _osc(psi0, rc0, af0, bf0, degenerate if native_v223 else None)
    if native_v223:
        osc_phi = torch.where(freq_degenerate, torch.zeros_like(osc_phi), osc_phi)
        osc_zeta = torch.where(freq_degenerate, torch.zeros_like(osc_zeta), osc_zeta)
        osc0_phi = torch.where(degenerate, torch.zeros_like(osc0_phi), osc0_phi)
        osc0_zeta = torch.where(degenerate, torch.zeros_like(osc0_zeta), osc0_zeta)
    alpha = secular_phi + osc_phi - osc0_phi[..., None]
    gamma = -(secular_zeta + osc_zeta - osc0_zeta[..., None])
    cosb_newtonian = (J.square() + L.square() - S2) / (2 * J * L)
    if native_v223 or beta_pn:
        if native_v223:
            # The native v223 angle tuple carries the published nonconserved
            # 3PN opening-beta L (259/72).  Equation-level callers retain
            # their Newtonian ``cos_beta`` default unless beta_pn is explicit.
            Lb = orbital_angular_momentum_nonconserved_3pn(
                cm1[..., None], cm2[..., None], f_grid,
                cs1[..., 2, None], cs2[..., 2, None],
            )
        else:
            Lb = orbital_angular_momentum(
                cm1[..., None], cm2[..., None], f_grid,
                cs1[..., 2, None], cs2[..., 2, None],
                pn_version=beta_pn_version, compat_102=beta_compat_102,
            )
        Jb = torch.sqrt(Lb.square() + 2 * c1[..., None] * Lb / eta[..., None] + sav[..., None])
        cosb_pn = (Jb.square() + Lb.square() - S2) / (2 * Jb * Lb)
    else:
        cosb_pn = cosb_newtonian
    cosb_output = cosb_pn if native_v223 else cosb_newtonian
    return {"alpha": alpha, "gamma": gamma, "cos_beta": cosb_output,
            "cos_beta_newtonian": cosb_newtonian, "cos_beta_pn": cosb_pn, "psi": psi,
            "roots_reference": roots0, "roots": roots, "c1": c1, "sav": sav,
            "spin_squared": S2,
            "psi_initial": psi0, "psi1": psi1, "psi2": psi2,
            "psi1_independent": psi1_independent,
            "psi2_independent": psi2_independent,
            "mean_s1_l": av["s1l"], "mean_s2_l": av["s2l"],
            "mean_s1_l2": av["s1l2"], "mean_s2_l2": av["s2l2"],
            "mean_s1ls2l": av["s1ls2l"],
            "flux_couplings": flux_couplings, "flux_coefficients": fc,
            "compatibility": selected, "degenerate": degenerate,
            "frequency_degenerate": freq_degenerate,
            "rational_invalid_current": rational_invalid_current,
            "rational_invalid_reference": rational_invalid_reference,
            "rational_lambda_a": rational_lambda_a,
            "rational_lambda_b": rational_lambda_b,
            "roots_reference_valid": roots_valid if native_v223 else torch.ones_like(q, dtype=torch.bool),
            "secular_phi": secular_phi, "secular_zeta": secular_zeta,
            "oscillatory_phi": osc_phi, "oscillatory_zeta": osc_zeta}


precession_msa_angles = assemble_precession_msa_angles


def assemble_precession_msa_angles_v223(*args, **kwargs):
    """Explicit native v223 angle assembly entry point."""
    kwargs.setdefault("compatibility", "v223")
    return assemble_precession_msa_angles(*args, **kwargs)


precession_msa_angles_v223 = assemble_precession_msa_angles_v223

__all__ = ["assemble_precession_msa_angles", "assemble_precession_msa_angles_v223",
           "precession_msa_angles", "precession_msa_angles_v223",
           "elliptic_spin_average"]
