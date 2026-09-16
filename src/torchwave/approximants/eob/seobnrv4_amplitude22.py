"""Published SEOBNRv4 factorized (2,2) amplitude reconstruction only.

Taracchini et al. 1202.0790, Appendix A; 1311.2544, Eq. (4) and
accompanying text; Pan et al. 1006.0431v2, Eq. (29a); Buonanno et al.
1209.6349v2, Eqs. (3.4), (4.27b); Bohé et al. 1611.03703v3,
Eqs. (2.4)--(2.5). No complex phase, NQC, flux, or merger attachment.

This is a mathematical reconstruction of the published amplitude recipe,
not a claim of implementation compatibility. The paper path keeps Bohé's
stated replacement of the spinning terms at 3.5PN. An explicit compatibility
option can retain Pan's cubic Kerr-spin term alongside that replacement; the
option is off by default. The source boundary and phase ambiguities are
recorded in pass-3-seobnrv4-mode22.md.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ...constants import EULER_GAMMA
from .factorized_modes import _tail_abs_squared


def _rho22(v: Tensor, nu: Tensor, delta: Tensor, chi_s: Tensor, chi_a: Tensor,
           *, retain_pan_cubic_spin: bool = False) -> Tensor:
    """Published residual rho, with v=(M Omega)^(1/3).

    Nonspin: Taracchini 2012 (A8a), removing its calibrated 4PN term
    as prescribed in Taracchini 2014. Spin: Pan (42a), Buonanno (4.27b),
    Bohé (2.4)--(2.5), and the remaining Kerr terms of Pan (29a).
    """
    kerr = (1-2*nu)*chi_s+delta*chi_a
    chi_0 = chi_s+delta*chi_a
    el2 = EULER_GAMMA+math.log(4)+torch.log(v)
    c2 = 55*nu/84-43/42
    c3 = -2/3*((1-nu)*chi_s+delta*chi_a)
    c4 = 19583*nu**2/42336-33025*nu/21168-20555/10584+chi_0**2/2
    c5 = (-34/21+49*nu/18+209*nu**2/126)*chi_s-(34/21+19*nu/42)*delta*chi_a
    c6 = (10620745*nu**3/39118464-6292061*nu**2/3259872
          +41*math.pi**2*nu/192-48993925*nu/9779616
          -428*el2/105+1556919113/122245200+89*kerr**2/252)
    c7 = ((18733/15876+74749*nu/5292-245717*nu**2/63504+50803*nu**3/63504)*chi_s
          +(18733/15876+50140*nu/3969+97865*nu**2/63504)*delta*chi_a)
    if not isinstance(retain_pan_cubic_spin, bool):
        raise TypeError("retain_pan_cubic_spin must be an explicit bool")
    if retain_pan_cubic_spin:
        # Pan 1006.0431v2 Eq. (29a)'s Kerr term, retained by the numerical
        # compatibility path alongside Bohé 1611.03703v3 Eq. (2.5).
        c7 = c7 + kerr**3/3
    c8 = (9202*el2/2205-387216563023/160190110080
          +18353*kerr**2/21168-kerr**4/8)
    c10 = 439877*el2/55566-16094530514677/533967033600
    return (1+c2*v**2+c3*v**3+c4*v**4+c5*v**5+c6*v**6
            +c7*v**7+c8*v**8+c10*v**10)


def seobnrv4_published_amplitude22(
    omega: Tensor,
    *,
    omega_zero_radial: Tensor | float,
    heff_over_mu: Tensor | float,
    hreal_over_M: Tensor | float,
    x1: Tensor | float,
    x2: Tensor | float,
    chi1: Tensor | float = 0.,
    chi2: Tensor | float = 0.,
    retain_pan_cubic_spin: bool = False,
) -> Tensor:
    """Return the dimensionless positive factorized amplitude (R/M)|h22|.

    All arguments broadcast. ``omega`` is M*d(phi)/dt, not a frequency
    in Hz. ``omega_zero_radial`` is d(Hreal/mu)/d(p_phi) evaluated at
    p_r=0 at the same r and p_phi as the actual state; do not replace
    p_phi with a solved circular-orbit momentum. Since the tortoise
    transformation is nonsingular in the domain, p_r_star=0 is equivalent.

    ``heff_over_mu`` is Heff/mu. ``hreal_over_M`` is the total real
    energy INCLUDING REST MASS divided by M:
    sqrt(1+2*nu*(Heff/mu-1)). It is not the binding Hamiltonian.
    ``x1,x2`` are m1/M,m2/M and ``chi1,chi2`` are the signed aligned
    dimensionless component spins. Equal masses and the test-mass
    endpoints are regular. Multiply by GM/(c^2 R) for physical strain.

    Only the Newtonian prefactor uses the non-Keplerian velocity
    v_phi=omega*omega_zero_radial^(-2/3). The residual PN series uses
    v=omega^(1/3). Energies and frequencies come from the caller; this
    function neither integrates dynamics nor selects a phase convention.
    The tail's r0 convention does not affect its absolute value.
    """
    if not isinstance(omega, Tensor) or omega.dtype not in (torch.float32, torch.float64):
        raise ValueError("omega must be a float32 or float64 Torch tensor")
    named = {
        "omega": omega,
        "omega_zero_radial": omega_zero_radial,
        "heff_over_mu": heff_over_mu,
        "hreal_over_M": hreal_over_M,
        "x1": x1, "x2": x2, "chi1": chi1, "chi2": chi2,
    }
    values = {key: torch.as_tensor(value, dtype=omega.dtype, device=omega.device)
              for key, value in named.items()}
    for key, value in values.items():
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError(f"{key} must be finite")
    for key in ("omega", "omega_zero_radial", "heff_over_mu", "hreal_over_M"):
        if not bool(torch.all(values[key] > 0)):
            raise ValueError(f"{key} must be positive; energies use the stated rest-mass normalization")
    a, b = values["x1"], values["x2"]
    if (not bool(torch.all((a >= 0) & (a <= 1)))
            or not bool(torch.all((b >= 0) & (b <= 1)))
            or not bool(torch.all(torch.abs(a+b-1) <= 16*torch.finfo(omega.dtype).eps))):
        raise ValueError("mass fractions x1,x2 must lie in [0,1] and sum to one")
    if not bool(torch.all((values["chi1"].abs() <= 1) & (values["chi2"].abs() <= 1))):
        raise ValueError("aligned dimensionless spins must lie in [-1,1]")
    nu = a*b
    chi_s = (values["chi1"]+values["chi2"])/2
    chi_a = (values["chi1"]-values["chi2"])/2
    v = omega.pow(1/3)
    v_phi = omega*values["omega_zero_radial"].pow(-2/3)
    rho = _rho22(v, nu, a-b, chi_s, chi_a,
                 retain_pan_cubic_spin=retain_pan_cubic_spin)
    # The modulus is independent of the otherwise relevant tail radius
    # r0=2M/sqrt(e). k_hat=2*omega*Hreal/M contains total rest energy.
    tail_abs = _tail_abs_squared(2, 2*omega*values["hreal_over_M"]).sqrt()
    return 8*math.sqrt(math.pi/5)*nu*v_phi**2*values["heff_over_mu"]*tail_abs*rho**2
