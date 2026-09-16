"""Mean-zero oscillatory MSA corrections (Chatziioannou et al. App. B/F).

Clean-room diagnostic implementation.  The rational functions are evaluated
directly from the published ``a,c_i,d_i`` coefficients; partial fractions are
integrated analytically, avoiding the compact Eq. (F19) transcription.  The
Q term is retained explicitly as ``-sin(2*psi)/4`` (derivative
``sin(psi)**2 - 1/2``), and both corrections carry the published ``J/psidot``
factor.
"""
import torch


def _compatibility_value(value):
    if isinstance(value, bool):
        return "v223" if value else "equation"
    value = str(value).lower().replace("-", "_")
    if value in ("equation", "published", "default", "clean", "none"):
        return "equation"
    if value in ("v223", "223", "native", "native_msa", "msa_v223",
                 "compat", "compatibility"):
        return "v223"
    raise ValueError("compatibility must be 'equation' or 'v223'")


def _compatibility_variant(*, compatibility=None, variant=None,
                           convention=None, v223=False):
    supplied = [x for x in (compatibility, variant, convention) if x is not None]
    values = [_compatibility_value(x) for x in supplied]
    if values and any(x != values[0] for x in values[1:]):
        raise ValueError("compatibility selectors disagree")
    return values[0] if values else ("v223" if v223 else "equation")


def _t(x):
    return x if isinstance(x, torch.Tensor) and x.dtype == torch.float64 else torch.as_tensor(x, dtype=torch.float64)


def rational_coefficients(v, eta, xi, c1, s1sq, s2sq, delta, splus2, sminus2, s3sq, *, J, L,
                          compatibility=None, variant=None, convention=None,
                          v223=False):
    """Return ``(J,a,c0,c2,c4,d0,d2,d4,psidot)`` from App. B.

    Here delta=(m1-m2)/M and q=(1-delta)/(1+delta), matching the paper's
    mass-ratio notation.  ``s3sq`` is the third spin-root supplied by the
    caller and may be negative; only the two upper roots are required to be
    nonnegative.  ``J`` and ``L`` are explicit current values, consistent
    with the caller's reference ``S_av`` and ``c1``; they are not rebuilt
    internally from the current ``v``.
    """
    v, eta, xi, c1, s1sq, s2sq, delta, sp, sm, s3, J, L = [_t(x) for x in (v, eta, xi, c1, s1sq, s2sq, delta, splus2, sminus2, s3sq, J, L)]
    v, eta, xi, c1, s1sq, s2sq, delta, sp, sm, s3, J, L = torch.broadcast_tensors(v, eta, xi, c1, s1sq, s2sq, delta, sp, sm, s3, J, L)
    q = (1 - delta) / (1 + delta)
    J2 = J**2
    if bool(torch.any(~torch.isfinite(J2))) or bool(torch.any(L <= 0)) or bool(torch.any(sp < sm)) or bool(torch.any(sm < 0)) or bool(torch.any(~torch.isfinite(sp))):
        raise ValueError("invalid nondegenerate MSA domain")
    J = torch.sqrt(J2); ds = sp - sm
    a = .5*v**6 * (1 + 1.5/eta*(1-xi*v))
    c0 = .75*(1-xi*v)*v**2*(eta**3 + 4*eta**3*xi*v - 2*eta*(J2-sp + 2*(s1sq-s2sq)*delta)*v**2 - 4*eta*xi*(J2-sp)*v**3 + (J2-sp)**2/eta*v**4)
    c2 = -1.5*eta*ds*(1+2*xi*v-(J2-sp)/eta**2*v**2)*(1-xi*v)*v**4
    c4 = .75/eta*ds**2*(1-xi*v)*v**6
    d0 = -(J2-(L+torch.sqrt(sp))**2)*(J2-(L-torch.sqrt(sp))**2)
    d2 = -2*ds*(J2+L**2-sp)
    d4 = -ds**2
    psidot_positive = 3/(4*torch.sqrt(eta))*(1-xi*v)*v**6*torch.sqrt(sp-s3)
    if bool(torch.any(psidot_positive <= 0)):
        raise ValueError("psidot must be positive")
    selected = _compatibility_variant(
        compatibility=compatibility, variant=variant,
        convention=convention, v223=v223,
    )
    # The clean equation helper retains the positive magnitude.  The native
    # v223 convention carries the signed nutation rate; its angle integrals
    # below consume the positive magnitude and put the signs on the two pole
    # residues explicitly.
    psidot = -psidot_positive if selected == "v223" else psidot_positive
    return J, a, c0, c2, c4, d0, d2, d4, psidot


def _P(lam, psi):
    k = torch.sqrt(1-lam)
    arg = (k-1)*torch.sin(psi)*torch.cos(psi) / (torch.cos(psi)**2 + k*torch.sin(psi)**2)
    return torch.atan(arg) / k


def branch_correction(lam, psi):
    """Stable ``atan(tan(psi))-atan(sqrt(n)*tan(psi))`` difference.

    Writing the two arctangents as one ``atan2`` avoids evaluating ``tan`` at
    its poles and gives the continuous pi-periodic representative used by
    the partial-fraction primitives.  ``n=1-lam`` is positive throughout
    the nondegenerate rational domain.
    """
    lam, psi = [_t(x) for x in (lam, psi)]
    lam, psi = torch.broadcast_tensors(lam, psi)
    k = torch.sqrt(1.0 - lam)
    numerator = (1.0 - k) * torch.sin(psi) * torch.cos(psi)
    denominator = torch.cos(psi).square() + k * torch.sin(psi).square()
    return torch.atan2(numerator, denominator)


def _branch_correction_v223(lam, psi):
    """Return ``B(lambda,psi)`` with the v223 absolute-root continuation.

    The compatibility convention continues ``n=1-lambda`` through negative
    values with ``k=sqrt(abs(n))``.  Keeping the ``atan2`` form avoids tangent
    poles and preserves the same pi-periodic branch used by the strict
    equation helper.  Exact ``n=0`` is given an explicit zero primitive; the
    surrounding root-gap branch suppresses that inactive contribution.
    """
    lam, psi = [_t(x) for x in (lam, psi)]
    lam, psi = torch.broadcast_tensors(lam, psi)
    n = 1.0 - lam
    n_abs = torch.abs(n)
    # Avoid an infinite derivative in the explicit n=0 limiting branch.  The
    # nonzero continuation is unchanged; the masked value is only auxiliary.
    k = torch.sqrt(torch.where(n_abs > 0.0, n_abs, torch.ones_like(n_abs)))
    numerator = (1.0 - k) * torch.sin(psi) * torch.cos(psi)
    denominator = torch.cos(psi).square() + k * torch.sin(psi).square()
    branch = torch.atan2(numerator, denominator)
    k_safe = torch.where(k > 0.0, k, torch.ones_like(k))
    primitive = branch / k_safe
    return torch.where(n_abs > 0.0, primitive, torch.zeros_like(primitive))


def _P_v223(lam, psi):
    """The native primitive, extended with ``sqrt(abs(1-lambda))``."""
    return -_branch_correction_v223(lam, psi)


def _partial_fractions(c0, c2, c4, d0, d2, d4):
    """Return positive/negative poles and their independent residues."""
    disc = d2**2 - 4*d0*d4
    if bool(torch.any(d0 == 0)) or bool(torch.any(d4 == 0)) or bool(torch.any(disc <= 0)):
        raise ValueError("distinct nonzero rational-factor roots required")
    root = torch.sqrt(disc)
    l_plus = (-d2 + root)/(2*d0)
    l_minus = (-d2 - root)/(2*d0)
    r0 = c4/d4
    u = c0/d0-r0
    w = c2/d0+r0*(l_plus+l_minus)
    r_plus = (w+u*l_plus)/(l_plus-l_minus)
    r_minus = (-w-u*l_minus)/(l_plus-l_minus)
    # The algebraic labels in Appendix F are lambda_1>0 and lambda_2<0.
    # Keep this explicit instead of assuming a denominator-sign convention.
    positive = l_plus > 0
    lp = torch.where(positive, l_plus, l_minus)
    lm = torch.where(positive, l_minus, l_plus)
    rp = torch.where(positive, r_plus, r_minus)
    rm = torch.where(positive, r_minus, r_plus)
    return lp, lm, rp, rm, r0


def _partial_fractions_v223(c0, c2, c4, d0, d2, d4):
    """Return algebraically labelled pole factors and residues for v223.

    The compatibility continuation keeps the ``+``/``-`` labels generated by
    the quadratic formula.  It therefore does not reorder poles by sign.
    ``sqrt(abs(discriminant))`` supplies the real continuation used only by
    v223; the strict equation helper above retains its published domain.
    """
    disc = d2.square() - 4.0 * d0 * d4
    disc_abs = torch.abs(disc)
    # Repeated poles are rejected by the caller; keep their auxiliary root
    # finite here so masked evaluation cannot create NaNs or infinite grads.
    root = torch.sqrt(torch.where(
        disc_abs > 0.0, disc_abs, torch.ones_like(disc_abs)
    ))
    d0_safe = torch.where(d0 != 0.0, d0, torch.ones_like(d0))
    d4_safe = torch.where(d4 != 0.0, d4, torch.ones_like(d4))
    la = (-d2 + root) / (2.0 * d0_safe)
    lb = (-d2 - root) / (2.0 * d0_safe)
    r0 = c4 / d4_safe
    u = c0 / d0_safe - r0
    w = c2 / d0_safe + r0 * (la + lb)
    difference = la - lb
    difference_safe = torch.where(
        difference != 0.0, difference, torch.ones_like(difference)
    )
    ra = (w + u * la) / difference_safe
    rb = (-w - u * lb) / difference_safe
    return la, lb, ra, rb, r0


def _zeta_geometric_factor(lam, atheta, btheta):
    """Evaluate ``A_theta+B_theta/lambda`` with an explicit lambda=0 limit."""
    nonzero = lam != 0.0
    lam_safe = torch.where(nonzero, lam, torch.ones_like(lam))
    factor = atheta + btheta / lam_safe
    return torch.where(nonzero, factor, torch.zeros_like(factor))


def oscillatory_correction(psi, J, psidot, a, c0, c2, c4, d0, d2, d4, atheta, btheta,
                           *, compatibility=None, variant=None,
                           convention=None, v223=False):
    """Return (phi_z, zeta) first-order mean-zero oscillatory corrections."""
    psi, J, psidot, a, c0, c2, c4, d0, d2, d4, atheta, btheta = [_t(x) for x in (psi, J, psidot, a, c0, c2, c4, d0, d2, d4, atheta, btheta)]
    psi, J, psidot, a, c0, c2, c4, d0, d2, d4, atheta, btheta = torch.broadcast_tensors(psi, J, psidot, a, c0, c2, c4, d0, d2, d4, atheta, btheta)
    selected = _compatibility_variant(
        compatibility=compatibility, variant=variant,
        convention=convention, v223=v223,
    )
    if selected == "equation":
        l1, l2, aa, bb, r0 = _partial_fractions(c0, c2, c4, d0, d2, d4)
        if bool(torch.any(~torch.isfinite(l1))) or bool(torch.any(~torch.isfinite(l2))) or bool(torch.any(l1 >= 1)) or bool(torch.any(l2 >= 1)):
            raise ValueError("rational factors require finite lambda_i<1")
        p1, p2 = _P(l1, psi), _P(l2, psi)
        q = -torch.sin(2*psi)/4
        phi = J/psidot*(aa*p1+bb*p2)
        zeta = J/psidot*((a+r0)*btheta*q + aa*(atheta+btheta/l1)*p1 + bb*(atheta+btheta/l2)*p2)
    else:
        la, lb, ra, rb, _ = _partial_fractions_v223(
            c0, c2, c4, d0, d2, d4
        )
        finite = torch.isfinite(la) & torch.isfinite(lb) & torch.isfinite(ra) & torch.isfinite(rb)
        if bool(torch.any(~finite)):
            raise ValueError("v223 rational factors require finite poles")
        ba = _branch_correction_v223(la, psi)
        bb_ = _branch_correction_v223(lb, psi)
        # v223 uses algebraic pole labels, absolute residue magnitudes, and
        # the second pole's denominator sign.  Keep one J factor outside the
        # primitives; the rational coefficients contain no J.
        positive_psidot = psidot.abs()
        sign_a = torch.sign(la)
        sign_b = torch.sign(d0 * (d0 + d2 + d4)) * torch.sign(lb)
        phi = J / positive_psidot * (
            sign_a * ra.abs() * ba + sign_b * rb.abs() * bb_
        )
        geom_a = _zeta_geometric_factor(la, atheta, btheta)
        geom_b = _zeta_geometric_factor(lb, atheta, btheta)
        zeta = J / positive_psidot * (
            sign_a * ra.abs() * geom_a * ba
            + sign_b * rb.abs() * geom_b * bb_
        )
    return phi, zeta


def rational_coefficients_v223(v, eta, xi, c1, s1sq, s2sq, delta,
                               splus2, sminus2, s3sq, *, J, L):
    """Named v223 rational coefficients with signed ``dot(psi)``."""
    return rational_coefficients(
        v, eta, xi, c1, s1sq, s2sq, delta, splus2, sminus2, s3sq,
        J=J, L=L, compatibility="v223",
    )


def oscillatory_correction_v223(psi, J, psidot, a, c0, c2, c4,
                                d0, d2, d4, atheta, btheta):
    """Named v223 pole correction (Q term omitted by contract)."""
    return oscillatory_correction(
        psi, J, psidot, a, c0, c2, c4, d0, d2, d4, atheta, btheta,
        compatibility="v223",
    )


def angle_factors(J, L, splus2, sminus2):
    """A_theta_L and B_theta_L used by Appendix F."""
    J, L, sp, sm = [_t(x) for x in (J, L, splus2, sminus2)]
    return (J**2 + L**2-sp)/(2*J*L), (sp-sm)/(2*J*L)


def signed_nutation_rate(psidot, *, compatibility=None, variant=None,
                         convention=None, v223=False):
    """Return the positive equation or signed v223 nutation rate."""
    psidot = _t(psidot)
    selected = _compatibility_variant(
        compatibility=compatibility, variant=variant,
        convention=convention, v223=v223,
    )
    return -psidot.abs() if selected == "v223" else psidot


nutation_rate_v223 = lambda psidot: signed_nutation_rate(psidot, v223=True)


__all__ = [
    "rational_coefficients", "rational_coefficients_v223", "branch_correction",
    "oscillatory_correction", "oscillatory_correction_v223", "angle_factors",
    "signed_nutation_rate", "nutation_rate_v223",
]
