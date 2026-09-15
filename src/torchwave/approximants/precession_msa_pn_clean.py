"""Clean-room PN coefficients for the multiple-scale precession initializer.

The formulae are from Chatziioannou et al., arXiv:1703.03967v3,
Appendices A--C, and Gerosa et al., arXiv:1307.4418v2, Appendix A.
The flux is written as ``a0 * (1 + a2*v**2 + a3*v**3 + ...)``; consequently
``g`` is the reciprocal of that complete series and includes the factor
``1/a0``.
"""
from __future__ import annotations
import torch

def _b(*xs):
    return torch.broadcast_tensors(*(torch.as_tensor(x, dtype=torch.float64) for x in xs))


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
    """Resolve the explicit equation/compatibility selector used below.

    The clean functions keep their equation-level defaults.  A separate
    selector is useful to callers which are reproducing the native v223
    convention because that convention changes only a few assembly choices;
    it must not silently change the published helpers.  ``variant`` and
    ``convention`` are accepted as readable aliases for diagnostics.
    """
    supplied = [x for x in (compatibility, variant, convention) if x is not None]
    values = [_compatibility_value(x) for x in supplied]
    if values and any(x != values[0] for x in values[1:]):
        raise ValueError("compatibility selectors disagree")
    return values[0] if values else ("v223" if v223 else "equation")

def precession_averages(c1, xi, q, eta, splus2, sminus2, s1sq, s2sq, v0,
                        *, regularize=False):
    c1, xi, q, eta, sp, sm, s1sq, s2sq, v0 = _b(c1, xi, q, eta, splus2, sminus2, s1sq, s2sq, v0)
    if regularize:
        bad = (q <= 0) | (q > 1) | (eta <= 0) | (v0 <= 0)
    else:
        bad = (q <= 0) | (q >= 1) | (eta <= 0) | (v0 <= 0)
    if bool(torch.any(bad)):
        if regularize:
            raise ValueError("require 0<q<=1, eta>0, and v0>0")
        raise ValueError("require 0<q<1, eta>0, and v0>0")
    # The native v223 boundary keeps the physical mass ratio, including the
    # equal-mass limit.  The published denominators are regularized by the
    # fixed machine-scale term from the contract; this is separate from the
    # root-gap branch used by angle assembly.
    eps = torch.as_tensor(1.0e-16, dtype=q.dtype, device=q.device)
    den = eta * (1 - q**2 + (eps if regularize else 0.0))
    u1 = (c1 * (1 + q) - q * eta * xi) / den
    u2 = -q * (c1 * (1 + q) - eta * xi) / den
    corr_den = (1 - q)**2 + (eps if regularize else 0.0)
    corr = (sp - sm)**2 * v0**2 / (32 * eta**2 * corr_den)
    sav = (sp + sm) / 2
    return {"s1l": u1, "s2l": u2, "s1l2": u1**2 + corr,
            "s2l2": u2**2 + q**2*corr, "s1ls2l": u1*u2 - q*corr,
            "s12": (sav - s1sq - s2sq) / 2}


def precession_averages_v223(c1, xi, q, eta, splus2, sminus2,
                             s1sq, s2sq, v0):
    """Named v223 entry point for the physical-q MSA averages.

    The native boundary retains the physical ``q`` and applies the published
    ``+1e-16`` denominator terms so the equal-mass limit remains finite.
    """
    return precession_averages(c1, xi, q, eta, splus2, sminus2,
                               s1sq, s2sq, v0, regularize=True)

def flux_coefficients(eta, q, s1l, s2l, s1sq, s2sq, s12, s1l2, s2l2, s1ls2l):
    eta, q, s1l, s2l, s1sq, s2sq, s12, s1l2, s2l2, s1ls2l = _b(
        eta, q, s1l, s2l, s1sq, s2sq, s12, s1l2, s2l2, s1ls2l)
    x1, x2 = 1/(1+q), q/(1+q)
    beta3 = (113/12 + 25*q/4)*s1l + (113/12 + 25/(4*q))*s2l
    beta5 = (31319/1008 - 1159*eta/24 + q*(809/84 - 281*eta/8))*s1l + (31319/1008 - 1159*eta/24 + (809/84 - 281*eta/8)/q)*s2l
    sigma4 = (247*s12 - 721*s1ls2l)/(48*eta) + ((233*s1sq - 719*s1l2)/x1**2 + (233*s2sq - 719*s2l2)/x2**2)/96
    a0 = 96*eta/5
    a2 = -743/336 - 11*eta/4
    a3 = 4*torch.pi - beta3
    a4 = 34103/18144 + 13661*eta/2016 + 59*eta**2/18 - sigma4
    a5 = -4159*torch.pi/672 - 189*torch.pi*eta/8 - beta5
    return {"beta3": beta3, "beta5": beta5, "sigma4": sigma4, "a0": a0, "a2": a2, "a3": a3, "a4": a4, "a5": a5}


def initial_reference_spin_couplings(spin1, spin2):
    """Return the flux spin couplings from already normalized ``S1``/``S2``.

    ``spin1`` and ``spin2`` here are the source-total-mass-normalized vectors
    ``S_i/M^2``.  The helper deliberately does no averaging: v223's flux
    coefficients use the initial reference projections and contractions,
    whereas the MSA averages remain the inputs to the other expansion
    coefficients.
    """
    s1, s2 = (torch.as_tensor(x, dtype=torch.float64) for x in (spin1, spin2))
    if s1.ndim == 0 or s2.ndim == 0 or s1.shape[-1] != 3 or s2.shape[-1] != 3:
        raise ValueError("reference spin vectors must end in dimension 3")
    s1, s2 = torch.broadcast_tensors(s1, s2)
    s1l, s2l = s1[..., 2], s2[..., 2]
    return {
        "S1": s1,
        "S2": s2,
        "s1l": s1l,
        "s2l": s2l,
        "s12": torch.sum(s1 * s2, dim=-1),
        "s1l2": s1l.square(),
        "s2l2": s2l.square(),
        "s1ls2l": s1l * s2l,
        "s1sq": torch.sum(s1.square(), dim=-1),
        "s2sq": torch.sum(s2.square(), dim=-1),
    }


def initial_reference_flux_coefficients(eta, q, spin1, spin2):
    """Construct ``a0..a5`` using the initial reference spin couplings.

    This is the explicit v223 flux path.  It calls :func:`flux_coefficients`
    for the published polynomial itself; only the choice of spin arguments is
    separated here.
    """
    couplings = initial_reference_spin_couplings(spin1, spin2)
    return flux_coefficients(
        eta, q, couplings["s1l"], couplings["s2l"],
        couplings["s1sq"], couplings["s2sq"], couplings["s12"],
        couplings["s1l2"], couplings["s2l2"], couplings["s1ls2l"],
    )


flux_coefficients_v223 = initial_reference_flux_coefficients

def reciprocal_coefficients(a):
    a0, a2, a3, a4, a5 = (a[k] for k in ("a0", "a2", "a3", "a4", "a5"))
    return {"g0": 1/a0, "g1": 0*a0, "g2": -a2/a0, "g3": -a3/a0,
            "g4": -(a4-a2**2)/a0, "g5": -(a5-2*a3*a2)/a0}

def psi2_compatibility(psi2_independent, q, g0, g2):
    """Apply the named v223 quadratic-phase convention.

    The independent limiting-root expansion supplies ``psi2`` including its
    reciprocal-flux term.  Appendix-C form differs by moving the flux term
    through the mass-ratio factor shown here; ``psi1`` is intentionally left
    untouched.
    """
    psi2_independent, q, g0, g2 = _b(psi2_independent, q, g0, g2)
    return 3.0 * g2 / g0 + q.square() * (
        psi2_independent - 3.0 * g2 / g0
    )


def _psi_coefficients_independent(eta, delta, q, xi, c1, sav,
                                  s1sq, s2sq, g0, g2, *, regularize=False):
    eta, delta, q, xi, c1, sav, s1sq, s2sq, g0, g2 = _b(eta, delta, q, xi, c1, sav, s1sq, s2sq, g0, g2)
    if bool(torch.any((eta <= 0) | (delta == 0) | (q <= 0))): raise ValueError("require eta>0, q>0, and unequal masses")
    bm2, bm1 = eta*delta**2, 2*eta*xi - 4*c1
    b0 = q*s1sq + s2sq/q - s1sq - s2sq - 2*sav
    cm2 = 4*c1**2 - 4*eta*c1*xi + 4*eta**3*xi**2 - 2*eta*delta**2*(s1sq+s2sq)
    dm2 = 4*(1-q)/q*(s2sq-q*s1sq)*c1**2 + eta*delta**2*(s1sq-s2sq)**2 + 4*delta*eta*c1*xi*(s1sq-s2sq)
    disc = cm2**2 - 4*bm2*dm2
    if bool(torch.any((disc < 0) | ~torch.isfinite(disc))) and not regularize:
        raise ValueError("cubic limiting-root discriminant is negative")
    # The repeated-root native branch deliberately suppresses the initial
    # nutation reconstruction.  Evaluate its inactive square root at a tiny
    # positive limiting discriminant so autograd does not form ``0*inf``;
    # equation-level callers retain the exact published ``sqrt(disc)`` path.
    if regularize:
        # The native repeated-root phase is inactive.  Keep its auxiliary
        # expansion finite even when roundoff places the limiting
        # discriminant infinitesimally below zero; ordinary rows retain the
        # strict published domain check above.
        disc = torch.where(torch.isfinite(disc) & (disc >= 0.0), disc,
                           torch.zeros_like(disc))
        disc_sqrt = torch.sqrt(torch.clamp(disc, min=1.0e-24))
    else:
        disc_sqrt = torch.sqrt(disc)
    roots = torch.stack(((-cm2 + disc_sqrt)/(2*bm2), (-cm2 - disc_sqrt)/(2*bm2)))
    p, n = torch.amax(roots, dim=0), torch.amin(roots, dim=0)
    r1, r2 = bm1/(2*bm2), (b0 + 2*p + n)/(2*bm2) - bm1**2/(8*bm2**2)
    return 1.5*(r1-xi), 3*(r2-xi*r1+g2/g0)


def psi_coefficients(eta, delta, q, xi, c1, sav, s1sq, s2sq, g0, g2,
                     *, compatibility=None, variant=None, convention=None,
                     v223=False, regularize=False):
    """Return ``(psi1, psi2)`` with an explicit phase convention.

    The default is the independent limiting-root expansion used by the clean
    equations.  Selecting ``compatibility='v223'`` applies only the named
    ``psi2`` transformation; all root and ``psi1`` terms stay unchanged.
    """
    psi1, psi2_independent = _psi_coefficients_independent(
        eta, delta, q, xi, c1, sav, s1sq, s2sq, g0, g2,
        regularize=regularize,
    )
    selected = _compatibility_variant(
        compatibility=compatibility, variant=variant,
        convention=convention, v223=v223,
    )
    if selected == "v223":
        psi2 = psi2_compatibility(psi2_independent, q, g0, g2)
    else:
        psi2 = psi2_independent
    return psi1, psi2


def psi_coefficients_v223(eta, delta, q, xi, c1, sav,
                          s1sq, s2sq, g0, g2):
    """Named v223 phase-coefficient entry point."""
    return psi_coefficients(eta, delta, q, xi, c1, sav, s1sq, s2sq,
                            g0, g2, compatibility="v223")

def anchored_psi(v, v0, psi0, delta, g0, psi1, psi2):
    v, v0, psi0, delta, g0, psi1, psi2 = _b(v, v0, psi0, delta, g0, psi1, psi2)
    if bool(torch.any((v <= 0) | (v0 <= 0))): raise ValueError("v must be positive")
    def raw(x): return -3*g0*delta/4*x**-3*(1 + psi1*x + psi2*x*x)
    return psi0 + raw(v) - raw(v0)

__all__ = [
    "precession_averages", "precession_averages_v223", "flux_coefficients",
    "initial_reference_spin_couplings", "initial_reference_flux_coefficients",
    "flux_coefficients_v223", "reciprocal_coefficients", "psi2_compatibility",
    "psi_coefficients", "psi_coefficients_v223", "anchored_psi",
]
