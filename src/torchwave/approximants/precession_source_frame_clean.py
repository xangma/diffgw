"""Clean-room source-frame geometry for the PhenomX precession angles.

The construction follows Pratten et al., arXiv:2004.06503v2, Appendix C:
form the total dimensionless angular momentum ``J=L+S1+S2`` at the reference
frequency, then construct the observer/J-frame Euler geometry.  This helper
is diagnostic-only and does not alter waveform routing.
"""
from __future__ import annotations

import math
import torch

from .precession_beta_clean import orbital_angular_momentum
from .precession_msa_reference_l_clean import orbital_angular_momentum_reference_3pn


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
    """Resolve the explicit source-frame equation/v223 selector."""
    supplied = [x for x in (compatibility, variant, convention, version)
                if x is not None]
    values = [_compatibility_value(x) for x in supplied]
    if values and any(x != values[0] for x in values[1:]):
        raise ValueError("compatibility selectors disagree")
    return values[0] if values else ("v223" if v223 else "equation")


def source_frame_geometry(
    m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, iota, phi_ref, f_ref,
    *, compat_102: bool = False, compatibility=None, variant=None,
    convention=None, version=None, v223=False,
):
    """Return ``(alpha0, theta_jn, zeta, phi_j)`` at ``f_ref``.

    Inputs are scalar physical parameters, masses in solar masses, and
    ``f_ref`` in Hz.  Spin vectors are dimensionless and weighted by
    ``(m_i/M)^2``.  All calculations are float64 and remain differentiable
    with respect to tensor inputs.  ``compat_102`` is an optional diagnostic
    compatibility expression from the public version-102 evaluator; the
    printed PN expression is the default used for source-frame geometry.
    ``compatibility='v223'`` selects the native source-frame reference-spin
    expression with its distinct 259/75 coefficient.
    """
    selected = _compatibility_variant(
        compatibility=compatibility, variant=variant,
        convention=convention, version=version, v223=v223,
    )
    vals = [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, iota, phi_ref, f_ref]
    ts = [torch.as_tensor(x, dtype=torch.float64) for x in vals]
    m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, iota, phi_ref, f_ref = ts
    M = m1 + m2
    n1, n2 = m1 / M, m2 / M
    # The source-frame and MSA opening-beta angular momenta have distinct
    # nonconserved-spin 2.5PN coefficients in the native compatibility path.
    if selected == "v223":
        L = orbital_angular_momentum_reference_3pn(
            m1, m2, f_ref, s1z, s2z,
        )
    else:
        # Appendix C source-frame J uses the printed Eq. (123) orbital L.  The
        # omitted version-102 term is a rejected compatibility hypothesis;
        # source-frame geometry uses the printed PN expression.
        L = orbital_angular_momentum(m1, m2, f_ref, s1z, s2z,
                                     pn_version=102, compat_102=False)
    spin = torch.stack((n1*n1*s1x + n2*n2*s2x,
                        n1*n1*s1y + n2*n2*s2y,
                        n1*n1*s1z + n2*n2*s2z))
    J = spin + torch.stack((torch.zeros_like(L), torch.zeros_like(L), L))
    jhat = J / torch.linalg.vector_norm(J)

    # Observer basis used by Appendix C/Table IV (x-y ordering is intentional).
    n = torch.stack((torch.sin(iota)*torch.sin(phi_ref),
                     torch.sin(iota)*torch.cos(phi_ref), torch.cos(iota)))
    observer_x = torch.stack((-torch.cos(iota)*torch.sin(phi_ref),
                              -torch.cos(iota)*torch.cos(phi_ref), torch.sin(iota)))
    yj = torch.linalg.cross(jhat, n)
    # At exact face-on alignment the Euler azimuth is gauge-degenerate.  Use
    # the supplied observer tangent as the continuous limiting basis; this
    # keeps the physical polarization limit finite while leaving generic
    # geometries unchanged.
    eps = torch.finfo(torch.float64).eps
    y_norm = torch.linalg.vector_norm(yj)
    tangent = torch.linalg.cross(jhat, observer_x)
    tangent_norm = torch.linalg.vector_norm(tangent)
    # Select denominators algebraically.  This avoids evaluating a singular
    # inactive branch in autograd while preserving the exact generic result.
    y_den = torch.where(y_norm > eps, y_norm, torch.ones_like(y_norm))
    t_den = torch.where(tangent_norm > eps, tangent_norm, torch.ones_like(tangent_norm))
    yj_generic = yj / y_den
    yj_fallback = tangent / t_den
    yj = torch.where(y_norm > eps, yj_generic, yj_fallback)
    xj = torch.linalg.cross(yj, jhat)
    alpha_r = torch.hypot(yj[2], xj[2])
    alpha0 = torch.atan2(torch.where(alpha_r > eps, yj[2], torch.zeros_like(yj[2])),
                        torch.where(alpha_r > eps, xj[2], torch.ones_like(xj[2])))
    j_perp = torch.hypot(J[0], J[1])
    phi_num = torch.where(j_perp > eps, J[1], torch.zeros_like(J[1]))
    phi_den = torch.where(j_perp > eps, J[0], torch.ones_like(J[0]))
    phi_j = torch.where(j_perp > torch.finfo(torch.float64).eps,
                        torch.atan2(phi_num, phi_den),
                        torch.as_tensor(torch.pi / 2, dtype=torch.float64) - phi_ref)
    # Exact endpoint values are retained.  Euler-angle derivatives at exact
    # alignment are gauge-degenerate; callers should differentiate the
    # projected waveform (the physical limit), rather than theta_jn itself.
    jn = torch.clamp(torch.dot(jhat, n), -1., 1.)
    # atan2(sin(theta), cos(theta)) has the correct exact endpoints.  At
    # alignment the angular coordinate is gauge-degenerate, so explicitly
    # stop its endpoint derivative; this prevents an undefined 0/0 derivative
    # from contaminating the differentiable projected waveform.
    cross_norm = torch.linalg.vector_norm(torch.linalg.cross(jhat, n))
    raw_theta = torch.atan2(cross_norm, jn)
    endpoint = cross_norm <= eps
    endpoint_raw = torch.where(endpoint, raw_theta.detach(), raw_theta)
    theta_jn = endpoint_raw + endpoint * (torch.where(jn < 0, torch.pi, torch.zeros_like(jn)) - endpoint_raw)
    x_sphere = torch.cos(theta_jn)*xj - torch.sin(theta_jn)*jhat
    zeta = torch.atan2(torch.dot(observer_x, yj), torch.dot(observer_x, x_sphere))
    return alpha0, theta_jn, zeta, phi_j


__all__ = ["source_frame_geometry"]
