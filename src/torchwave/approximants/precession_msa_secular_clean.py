"""Clean-room secular MSA primitives from Chatziioannou et al., arXiv:1703.03967v3.

This module is diagnostic only.  Coefficients are supplied by the caller; no
waveform implementation or calibration data is imported here.  The coefficient
definitions are the paper's ``coeff-phiz`` and ``coeff-zeta`` appendices, with
the primitives from ``phiz-final`` and ``zeta-minus-one``.
"""
import torch


def _f64(x):
    return x if isinstance(x, torch.Tensor) and x.dtype == torch.float64 else torch.as_tensor(x, dtype=torch.float64)


def _inputs(v, eta, xi, c1, s1sq, s2sq, delta, splus2, sminus2, g):
    vals = [_f64(x) for x in (v, eta, xi, c1, s1sq, s2sq, delta, splus2, sminus2)]
    vals += [_f64(x) for x in g]
    vals = torch.broadcast_tensors(*vals)
    v, eta, xi, c1, s1sq, s2sq, delta, sp, sm, *gg = vals
    if bool(torch.any(v <= 0)) or bool(torch.any(eta <= 0)):
        raise ValueError("secular MSA requires v>0 and eta>0")
    return v, eta, xi, c1, s1sq, s2sq, delta, sp, sm, gg


def angular_momentum(v, eta, c1, s_average2):
    """J=sqrt(eta^2/v^2+2*c1/v+S_av^2), Eq. (Jsol)."""
    v, eta, c1, sav = (_f64(x) for x in (v, eta, c1, s_average2))
    v, eta, c1, sav = torch.broadcast_tensors(v, eta, c1, sav)
    q = eta * eta / (v * v) + 2 * c1 / v + sav
    if bool(torch.any(q <= 0)):
        raise ValueError("J^2 must be positive")
    return torch.sqrt(q)


def omega_coefficients(eta, xi, c1, s1sq, s2sq, delta, splus2, sminus2,
                       *, regularize=False):
    """Return the Appendix ``coeff-phiz`` Omega-z coefficients and zeta ones.

    The ordinary helper keeps its strict nondegenerate-root contract.  The
    native v223 assembly also needs the published one-spin/aligned limiting
    root triple, where the signed product ``c_+ c_-`` can approach zero from
    the negative side because the auxiliary third root is split by a fixed
    ``1e-9``.  In that explicit branch the positive real continuation is the
    stable secular limit; callers still receive the ordinary strict behavior
    unless ``regularize=True`` is selected.
    """
    eta, xi, c1, s1sq, s2sq, delta, sp, sm = (_f64(x) for x in (eta, xi, c1, s1sq, s2sq, delta, splus2, sminus2))
    eta, xi, c1, s1sq, s2sq, delta, sp, sm = torch.broadcast_tensors(eta, xi, c1, s1sq, s2sq, delta, sp, sm)
    rm = sp - sm
    cp, cm = sp * eta**2 - c1**2, sm * eta**2 - c1**2
    radicand = cp * cm
    if bool(torch.any(~torch.isfinite(radicand))) or bool(torch.any(radicand <= 0)):
        if not regularize:
            raise ValueError("nondegenerate MSA roots required")
        # The one-spin/aligned root prescription is a limiting continuation,
        # not a change to the physical invariants.  Absolute value selects
        # the real branch after the auxiliary split and preserves a finite
        # differentiable expression at the exact boundary.
        root = torch.sqrt(torch.clamp(radicand.abs(), min=1.0e-32))
    else:
        root = torch.sqrt(radicand)
    # The full instantaneous Eq. (phizdot) has
    #   a = v**6/2 * [1 + 3/(2*eta)*(1 - xi*v)].
    # Therefore its v -> 0 leading coefficient is 1/2 + 3/(4*eta).
    # Appendix ``coeff-phiz`` prints 1/2 + 3*eta/4, which is inconsistent
    # with that equation (and with the mass-ratio scaling of the precession
    # frequency).  Keep the spin-dependent contribution in ``ad`` separate.
    a1 = .5 + .75 / eta
    a2 = -.75 * xi / eta
    ad = (-3 * (s1sq - s2sq) * eta * delta + 3 * c1 / eta * (c1 - 2 * xi * eta**2)) / (4 * root)
    cd = 3 * rm**2 / (128 * eta * root)
    hd = c1 / eta**2 * (1 - (cp + cm) / (2 * root))
    fd = (cp + cm) / (8 * eta**4) * (1 - (cp + cm) / (2 * root))
    oz = [a1 + ad, a2 - ad * xi - ad * hd,
          ad * hd * xi + cd - ad * fd + ad * hd**2,
          (ad * fd - cd - ad * hd**2) * (xi + hd) + ad * fd * hd,
          (cd + ad * hd**2 - 2 * ad * fd) * (hd * xi + hd**2 - fd) - ad * fd**2,
          (cd - ad * fd + ad * hd**2) * fd * (xi + 2 * hd) - (cd + ad * hd**2 - 2 * ad * fd) * hd**2 * (xi + hd) - ad * fd**2 * hd]
    gd = 3 * rm**2 / (64 * eta**3) * (c1 - eta**2 * xi) / root
    zet = [oz[0], oz[1] + c1 / eta**2 * oz[0], oz[2] + c1 / eta**2 * oz[1],
           oz[3] + c1 / eta**2 * oz[2] + gd,
           oz[4] + c1 / eta**2 * oz[3] - gd * xi - gd * hd,
           oz[5] + c1 / eta**2 * oz[4] + gd * hd * xi + gd * (hd**2 - fd)]
    return torch.stack(oz, dim=-1), torch.stack(zet, dim=-1)


def averaged_coefficients(eta, xi, c1, s1sq, s2sq, delta, splus2, sminus2, g,
                          *, regularize=False):
    """Appendix-D/F averaged coefficients, with caller-supplied g0..g5."""
    g = list(g)
    if len(g) != 6:
        raise ValueError("g must contain g0..g5")
    oz, ze = omega_coefficients(
        eta, xi, c1, s1sq, s2sq, delta, splus2, sminus2,
        regularize=regularize,
    )
    z = lambda i: oz[..., i]
    q = lambda i: ze[..., i]
    ob = torch.stack([3*g[0]*z(0), 3*g[0]*z(1), 3*(g[0]*z(2)+g[2]*z(0)),
        3*(g[0]*z(3)+g[2]*z(1)+g[3]*z(0)), 3*(g[0]*z(4)+g[2]*z(2)+g[3]*z(1)+g[4]*z(0)),
        3*(g[0]*z(5)+g[2]*z(3)+g[3]*z(2)+g[4]*z(1)+g[5]*z(0))], dim=-1)
    zb = torch.stack([-g[0]*q(0), -1.5*g[0]*q(1), -3*(g[0]*q(2)+g[2]*q(0)),
        3*(g[0]*q(3)+g[2]*q(1)+g[3]*q(0)), 3*(g[0]*q(4)+g[2]*q(2)+g[3]*q(1)+g[4]*q(0)),
        1.5*(g[0]*q(5)+g[2]*q(3)+g[3]*q(2)+g[4]*q(1)+g[5]*q(0))], dim=-1)
    return ob, zb


def phi_z_primitives(v, eta, c1, s_average2):
    """The six Appendix-F phi_z^(n), whose derivatives are J*v^(n-3)."""
    v, eta, c1, sav = torch.broadcast_tensors(*[_f64(x) for x in (v, eta, c1, s_average2)])
    J = angular_momentum(v, eta, c1, sav)
    L = eta / v
    l1 = torch.log(c1 + J * eta + L * eta)
    l2 = torch.log(c1 + J * torch.sqrt(sav) * v + sav * v)
    s = torch.sqrt(sav)
    p = [J / eta**4 * (c1**2 / 2 - c1 * eta**2 / (6*v) - sav*eta**2/3 - eta**4/(3*v**2)) - c1/(2*eta)*(c1**2/eta**4 - sav/eta**2)*l1,
         # The arXiv transcription prints c1 here; differentiating the
         # published primitive identity requires c1**2 (the known Appendix-F
         # typo), which is the clean expression used below.
         -J/(2*eta**2)*(c1 + eta*L) + (c1**2 - eta**2*sav)/(2*eta**3)*l1,
         -J + s*l2 - c1/eta*l1,
         J*v - eta*l1 + c1/s*l2,
         J*v*(c1 + v*sav)/(2*sav) - (c1**2 - eta**2*sav)/(2*sav**1.5)*l2,
         -J*v*(c1**2/(2*sav**2) - c1*v/(6*sav) - v**2/3 - eta**2/(3*sav)) + c1*(c1**2-eta**2*sav)/(2*sav**2.5)*l2]
    return torch.stack(p, dim=-1)


def zeta_primitives(v, eta):
    """Six elementary primitives in Eq. (zeta-minus-one), n=0..5."""
    v, eta = torch.broadcast_tensors(_f64(v), _f64(eta))
    if bool(torch.any(v <= 0)):
        raise ValueError("zeta primitives require v>0")
    return eta.unsqueeze(-1) * torch.stack([v**-3, v**-2, v**-1, torch.log(v), v, v**2], dim=-1)


def anchored_angles(v, v0, phi_coeff, zeta_coeff, eta, c1, s_average2):
    """Return Eq. (phiz-final) and Eq. (zeta-minus-one), anchored at ``v0``."""
    v, v0, eta, c1, sav = (_f64(x) for x in (v, v0, eta, c1, s_average2))
    pc, zc = _f64(phi_coeff), _f64(zeta_coeff)
    p = phi_z_primitives(v, eta, c1, sav)
    p0 = phi_z_primitives(v0, eta, c1, sav)
    z = zeta_primitives(v, eta); z0 = zeta_primitives(v0, eta)
    return ((p-p0)*pc).sum(-1), ((z-z0)*zc).sum(-1)
