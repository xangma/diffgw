"""Published circular factorized modes, independently reconstructed for review.

Damour, Iyer & Nagar, arXiv:0811.2069v2 (DIN), Eqs. (4)--(29),
(59), Appendix C. Optional *leading* aligned-spin residual terms are from
Pan et al., arXiv:1006.0431v2, Eqs. (41a)--(41i). This is not SEOBNRv4.
No NQC, non-Keplerian velocity replacement, or merger attachment is included.
See docs/reviews/2026-09-12-clean-room/pass-3-factorized-modes.md.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ...constants import EULER_GAMMA

Mode = tuple[int, int]
PUBLISHED_PHASE_MODES = frozenset(
    [(ell, m) for ell in range(2, 5) for m in range(1, ell + 1)] + [(5, 5)]
)


def _double_factorial(n: int) -> int:
    return math.prod(range(n, 0, -2))


def _newtonian_constant(ell: int, m: int) -> complex:
    """DIN (5)--(6), times Y_(ell-epsilon,-m)(pi/2,0)."""
    epsilon = (ell + m) % 2
    if epsilon == 0:
        n = (1j*m)**ell * 8*math.pi / _double_factorial(2*ell+1)
        n *= math.sqrt((ell+1)*(ell+2)/(ell*(ell-1)))
    else:
        n = -(1j*m)**ell * 16*math.pi*1j / _double_factorial(2*ell+1)
        n *= math.sqrt((2*ell+1)*(ell+2)*(ell*ell-m*m)
                       / ((2*ell-1)*(ell+1)*ell*(ell-1)))
    degree = ell-epsilon
    p_lm = (-1)**((degree+m)//2)*_double_factorial(degree+m-1)
    p_lm /= _double_factorial(degree-m)
    y_l_minus_m = (-1)**m*p_lm*math.sqrt(
        (2*degree+1)/(4*math.pi)*math.factorial(degree-m)/math.factorial(degree+m)
    )
    return n*y_l_minus_m


def _rho_table(x: Tensor, nu: Tensor) -> dict[Mode, Tensor]:
    """All 35 hybridized J-source residual amplitudes, DIN (C1)--(C35).

    Terms beyond the displayed order of each equation are omitted. In
    particular, higher test-mass coefficients are exactly those published
    in Appendix C; they are not promoted to finite-mass PN information.
    """
    n = nu
    el = {m: EULER_GAMMA + math.log(2*m) + .5*torch.log(x) for m in range(1, 5)}
    rho = {
        # C1--C5
        (2, 2): 1 + (55*n/84-43/42)*x
        + (19583*n**2/42336-33025*n/21168-20555/10584)*x**2
        + (10620745*n**3/39118464-6292061*n**2/3259872+41*math.pi**2*n/192
           -48993925*n/9779616-428*el[2]/105+1556919113/122245200)*x**3
        + (9202*el[2]/2205-387216563023/160190110080)*x**4
        + (439877*el[2]/55566-16094530514677/533967033600)*x**5,
        (2, 1): 1 + (23*n/84-59/56)*x
        + (617*n**2/4704-10993*n/14112-47009/56448)*x**2
        + (7613184941/2607897600-107*el[1]/105)*x**3
        + (6313*el[1]/5880-1168617463883/911303737344)*x**4,
        (3, 3): 1 + (2*n/3-7/6)*x
        + (149*n**2/330-1861*n/990-6719/3960)*x**2
        + (3203101567/227026800-26*el[3]/7)*x**3
        + (13*el[3]/3-57566572157/8562153600)*x**4,
        (3, 2): 1 + (320*n**2-1115*n+328)/(270*(3*n-1))*x
        + (3085640*n**4-20338960*n**3-4725605*n**2+8050045*n-1444528)
        / (1603800*(1-3*n)**2)*x**2
        + (5849948554/940355325-104*el[2]/63)*x**3,
        (3, 1): 1 + (-2*n/9-13/18)*x
        + (-829*n**2/1782-1685*n/1782+101/7128)*x**2
        + (11706720301/6129723600-26*el[1]/63)*x**3
        + (169*el[1]/567+2606097992581/4854741091200)*x**4,
        # C6--C10
        (4, 4): 1 + (2625*n**2-5870*n+1614)/(1320*(3*n-1))*x
        + (1252563795*n**4-6733146000*n**3-313857376*n**2+2338945704*n-511573572)
        / (317116800*(1-3*n)**2)*x**2
        + (16600939332793/1098809712000-12568*el[4]/3465)*x**3,
        (4, 3): 1 + (160*n**2-547*n+222)/(176*(2*n-1))*x-6894273/7047040*x**2,
        (4, 2): 1 + (285*n**2-3530*n+1146)/(1320*(3*n-1))*x
        + (-379526805*n**4-3047981160*n**3+1204388696*n**2+295834536*n-114859044)
        / (317116800*(1-3*n)**2)*x**2
        + (848238724511/219761942400-3142*el[2]/3465)*x**3,
        (4, 1): 1 + (288*n**2-1385*n+602)/(528*(2*n-1))*x-7775491/21141120*x**2,
        (5, 5): 1 + (512*n**2-1298*n+487)/(390*(2*n-1))*x-3353747/2129400*x**2,
        # C11--C20
        (5, 4): 1 + (33320*n**3-127610*n**2+96019*n-17448)
        / (13650*(5*n**2-5*n+1))*x,
        (5, 3): 1 + (176*n**2-850*n+375)/(390*(2*n-1))*x-410833/709800*x**2,
        (5, 2): 1 + (21980*n**3-104930*n**2+84679*n-15828)
        / (13650*(5*n**2-5*n+1))*x,
        (5, 1): 1 + (8*n**2-626*n+319)/(390*(2*n-1))*x-31877/304200*x**2,
        (6, 6): 1 + (273*n**3-861*n**2+602*n-106)/(84*(5*n**2-5*n+1))*x,
        (6, 5): 1 + (220*n**3-910*n**2+838*n-185)/(144*(3*n**2-4*n+1))*x,
        (6, 4): 1 + (133*n**3-581*n**2+462*n-86)/(84*(5*n**2-5*n+1))*x,
        (6, 3): 1 + (156*n**3-750*n**2+742*n-169)/(144*(3*n**2-4*n+1))*x,
        (6, 2): 1 + (49*n**3-413*n**2+378*n-74)/(84*(5*n**2-5*n+1))*x,
        (6, 1): 1 + (124*n**3-670*n**2+694*n-161)/(144*(3*n**2-4*n+1))*x,
        # C21--C27
        (7, 7): 1 + (1380*n**3-4963*n**2+4246*n-906)/(714*(3*n**2-4*n+1))*x,
        (7, 6): 1 + (6104*n**4-29351*n**3+37828*n**2-16185*n+2144)
        / (1666*(7*n**3-14*n**2+7*n-1))*x,
        (7, 5): 1 + (804*n**3-3523*n**2+3382*n-762)/(714*(3*n**2-4*n+1))*x,
        (7, 4): 1 + (41076*n**4-217959*n**3+298872*n**2-131805*n+17756)
        / (14994*(7*n**3-14*n**2+7*n-1))*x,
        (7, 3): 1 + (420*n**3-2563*n**2+2806*n-666)/(714*(3*n**2-4*n+1))*x,
        (7, 2): 1 + (32760*n**4-190239*n**3+273924*n**2-123489*n+16832)
        / (14994*(7*n**3-14*n**2+7*n-1))*x,
        (7, 1): 1 + (228*n**3-2083*n**2+2518*n-618)/(714*(3*n**2-4*n+1))*x,
        # C28--C35
        (8, 8): 1 + (12243*n**4-53445*n**3+64659*n**2-26778*n+3482)
        / (2736*(7*n**3-14*n**2+7*n-1))*x,
        (8, 7): 1 + (38920*n**4-207550*n**3+309498*n**2-154099*n+23478)
        / (18240*(4*n**3-10*n**2+6*n-1))*x,
        (8, 6): 1 + (2653*n**4-13055*n**3+17269*n**2-7498*n+1002)
        / (912*(7*n**3-14*n**2+7*n-1))*x,
        (8, 5): 1 + (6056*n**4-34598*n**3+54642*n**2-28055*n+4350)
        / (3648*(4*n**3-10*n**2+6*n-1))*x,
        (8, 4): 1 + (4899*n**4-28965*n**3+42627*n**2-19434*n+2666)
        / (2736*(7*n**3-14*n**2+7*n-1))*x,
        (8, 3): 1 + (24520*n**4-149950*n**3+249018*n**2-131059*n+20598)
        / (18240*(4*n**3-10*n**2+6*n-1))*x,
        (8, 2): 1 + (3063*n**4-22845*n**3+37119*n**2-17598*n+2462)
        / (2736*(7*n**3-14*n**2+7*n-1))*x,
        (8, 1): 1 + (21640*n**4-138430*n**3+236922*n**2-126451*n+20022)
        / (18240*(4*n**3-10*n**2+6*n-1))*x,
    }
    return rho


def _published_phases(v: Tensor, nu: Tensor, real_energy: Tensor) -> dict[Mode, Tensor]:
    """DIN (20)--(29); their explicitly permitted choice y_bar=x."""
    y32 = real_energy*v**3
    return {
        (2, 2): 7*y32/3+428*math.pi*y32**2/105-24*nu*v**5,
        (2, 1): 2*y32/3-493*nu*v**5/42,
        (3, 3): 13*y32/10-80897*nu*v**5/2430,
        (3, 2): (10+33*nu)/(15*(1-3*nu))*y32,
        (3, 1): 13*y32/30-17*nu*v**5/10,
        (4, 4): (112+219*nu)/(120*(1-3*nu))*y32,
        (4, 3): (486+4961*nu)/(810*(1-2*nu))*y32,
        (4, 2): 7*(1+6*nu)/(15*(1-3*nu))*y32,
        (4, 1): (2+507*nu)/(10*(1-2*nu))*y32,
        (5, 5): (96875+857528*nu)/(131250*(1-2*nu))*y32,
    }


def _tail_abs_squared(ell: int, khat: Tensor) -> Tensor:
    """DIN (59); expm1 preserves the weak-field limit."""
    result = 4*math.pi*khat/(-torch.expm1(-4*math.pi*khat))
    for s in range(1, ell+1):
        result = result*(1+4*khat**2/s**2)
    return result


def _tail_phase(ell: int, m: int, omega: Tensor, real_energy: Tensor) -> Tensor:
    """DIN (19), r0=2M. Gamma from DLMF 5.5.1 and 5.11.1.

    Shift Re(z) to 16, evaluate eight Bernoulli terms in Stirling's
    expansion, then undo the recurrence. This avoids library-specific
    complex Gamma dependencies and retains Torch derivatives.
    """
    khat = m*omega*real_energy
    z = ell+1-2j*khat
    shift = 15-ell
    w = z+shift
    log_gamma = (w-.5)*torch.log(w)-w+.5*math.log(2*math.pi)
    bernoulli = (1/6, -1/30, 1/42, -1/30, 5/66, -691/2730, 7/6, -3617/510)
    for k, number in enumerate(bernoulli, 1):
        log_gamma = log_gamma + number/(2*k*(2*k-1))*w**(-(2*k-1))
    for s in range(shift):
        log_gamma = log_gamma-torch.log(z+s)
    return log_gamma.imag+2*khat*torch.log(4*m*omega)


def _mass_times_leading_spin_f(
    ell: int, m: int, v: Tensor, x1: Tensor, x2: Tensor,
    chi1: Tensor, chi2: Tensor,
) -> Tensor:
    """c_(ell+epsilon) f_spin from Pan (41), without division by X1-X2.

    This adds the published linear-spin f term to rho_nonspin**ell;
    it does not exponentiate a spin-corrected rho or supply higher PN spin.
    """
    nu = x1*x2
    chi_s, chi_a = (chi1+chi2)/2, (chi1-chi2)/2
    delta = x1-x2
    p = ell+(ell+m)%2-1
    c = x2**p+(-1)**m*x1**p
    if m % 2:
        # Polynomial divided difference: regular even for equal masses.
        c_over_delta = -sum(x1**j*x2**(p-1-j) for j in range(p))
        if (ell, m) == (2, 1):
            return -1.5*(c*chi_s+c_over_delta*chi_a)*v
        if (ell, m) == (3, 3):
            return -((2-5*nu/2)*c*chi_s+(2-19*nu/2)*c_over_delta*chi_a)*v**3
        if (ell, m) == (3, 1):
            return -((2-13*nu/2)*c*chi_s+(2-11*nu/2)*c_over_delta*chi_a)*v**3
        if ell == 4:
            return -5*nu/(2*(2*nu-1))*(c*chi_s-c_over_delta*chi_a)*v
    if (ell, m) == (2, 2):
        return c*(-4/3)*((1-nu)*chi_s+delta*chi_a)*v**3
    if (ell, m) == (3, 2):
        return c*(-4*nu/(3*nu-1))*chi_s*v
    if (ell, m) == (4, 4):
        return c*(-4/15)*((42*nu**2-41*nu+10)*chi_s
                          +(10-39*nu)*delta*chi_a)/(1-3*nu)*v**3
    if (ell, m) == (4, 2):
        return c*(-4/15)*((78*nu**2-59*nu+10)*chi_s
                          +(10-21*nu)*delta*chi_a)/(1-3*nu)*v**3
    raise NotImplementedError("Leading finite-mass spin f is implemented only through ell=4")


def energy_flux_from_modes(modes: Mapping[Mode, Tensor], omega: Tensor) -> Tensor:
    """Positive emitted flux from positive-m modes; DIN (2), M=G=c=1.

    Each value is R*h_lm/M, or its absolute value. Negative-m partners are
    accounted for by 1/(8*pi). The circular derivative is -i*m*omega*h_lm.
    """
    if not modes or any(ell < 2 or not 1 <= m <= ell for ell, m in modes):
        raise ValueError("Supply a nonempty set of positive-m radiative modes")
    # Direct squares retain second derivatives at vanishing physical modes;
    # abs().square() does not, because abs has a cusp at zero.
    return sum((m*omega)**2*(h.real.square()+h.imag.square() if h.is_complex() else h.square())
               for (_, m), h in modes.items())/(8*math.pi)


class FactorizedModes(nn.Module):
    """Circular DIN amplitudes and flux through ell=8, with explicit choices.

    Inputs use total mass M=1: state[...,4]=(r,phi,p_r_star,p_phi),
    omega=M*Omega>0, effective_energy=H_eff/mu, real_energy=H_real/M
    **including rest mass**, and mass fractions X1+X2=1. Outputs are
    R*h_lm/M (positive m only). Spin arguments are chi_i=S_i/m_i**2.

    ``amplitudes`` and ``flux`` cover every requested ell. ``forward``
    provides complex modes; by default it rejects ell>4 because finite-mass
    delta_lm is absent in DIN except for (5,5). Explicitly setting
    ``residual_phase='zero-unpublished'`` omits these unknown residual phases
    while preserving the exact tail and all ten published delta expressions.
    ``spin_model='pan2011-leading-f'`` is limited to ell<=4. The default
    ``spin_model='none'`` rejects nonzero spin arguments.

    Radial position/momentum do not enter this circular prescription except
    through caller-supplied energy and omega. Applying it to a plunging state
    does not add NQC corrections. No complete SEOBNRv4 option is implemented.
    """

    def __init__(self, lmax: int = 8, *, spin_model: str = "none",
                 residual_phase: str = "published"):
        super().__init__()
        if type(lmax) is not int or not 2 <= lmax <= 8:
            raise ValueError("lmax must be an integer from 2 through 8")
        if spin_model not in {"none", "pan2011-leading-f"}:
            raise NotImplementedError("Supported spin models: none, pan2011-leading-f; no full v4/NQC")
        if spin_model != "none" and lmax > 4:
            raise NotImplementedError("Published leading finite-mass spin terms are implemented through ell=4")
        if residual_phase not in {"published", "zero-unpublished"}:
            raise NotImplementedError("Supported residual phases: published, zero-unpublished")
        self.lmax = lmax
        self.spin_model = spin_model
        self.residual_phase = residual_phase
        self.mode_indices = tuple((ell, m) for ell in range(2, lmax+1) for m in range(1, ell+1))
        self._newtonian_constants = {mode: _newtonian_constant(*mode) for mode in self.mode_indices}

    def _prepare(self, state, omega, effective_energy, real_energy, x1, x2, chi1, chi2):
        if not isinstance(state, Tensor) or state.dtype not in (torch.float32, torch.float64):
            raise TypeError("state must be a real float32 or float64 Torch tensor")
        if state.ndim == 0 or state.shape[-1] != 4:
            raise ValueError("state must have trailing dimension (r,phi,p_r_star,p_phi)")
        if not bool(torch.isfinite(state).all()):
            raise ValueError("state must be finite")
        values = tuple(torch.as_tensor(value, dtype=state.dtype, device=state.device)
                       for value in (omega, effective_energy, real_energy, x1, x2, chi1, chi2))
        if any(not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("Frequency, energies, mass fractions and spins must be finite")
        omega, effective_energy, real_energy, x1, x2, chi1, chi2, phi, j = torch.broadcast_tensors(
            *values, state[..., 1], state[..., 3]
        )
        if bool(((omega <= 0) | (effective_energy <= 0) | (real_energy <= 0)).any()):
            raise ValueError("omega and both rest-mass-inclusive energies must be positive")
        if bool(((x1 < 0) | (x2 < 0)).any()) or not bool(torch.allclose(
            x1+x2, torch.ones_like(x1), rtol=0, atol=8*torch.finfo(state.dtype).eps
        )):
            raise ValueError("Mass fractions must be nonnegative and sum to one")
        if bool(((chi1.abs() > 1) | (chi2.abs() > 1)).any()):
            raise ValueError("Dimensionless component spins must lie in [-1,1]")
        if self.spin_model == "none" and bool(((chi1 != 0) | (chi2 != 0)).any()):
            raise NotImplementedError("Nonzero spins require an explicitly supported spin_model")
        return omega, effective_energy, real_energy, x1, x2, chi1, chi2, phi, j

    def _prefactors(self, omega, effective_energy, real_energy, x1, x2, chi1, chi2, phi, j,
                    *, mode_indices=None):
        v, nu = omega**(1/3), x1*x2
        rho = _rho_table(v*v, nu)
        prefactors = {}
        for ell, m in self.mode_indices if mode_indices is None else mode_indices:
            epsilon = (ell+m)%2
            p = ell+epsilon-1
            mass_residual = (x2**p+(-1)**m*x1**p)*rho[ell, m]**ell
            if self.spin_model != "none":
                mass_residual = mass_residual + _mass_times_leading_spin_f(
                    ell, m, v, x1, x2, chi1, chi2
                )
            source = effective_energy if epsilon == 0 else j*v
            prefactors[ell, m] = self._newtonian_constants[ell, m]*nu*v**(ell+epsilon)*source*mass_residual
        return prefactors

    def amplitudes(self, state, omega, effective_energy, real_energy, x1, x2, *, chi1=0., chi2=0.):
        """Return |R*h_lm/M|; residual phases are unnecessary for amplitudes."""
        args = self._prepare(state, omega, effective_energy, real_energy, x1, x2, chi1, chi2)
        omega, _, real_energy, *_ = args
        prefactors = self._prefactors(*args)
        return {(ell, m): h.abs()*_tail_abs_squared(ell, m*omega*real_energy).sqrt()
                for (ell, m), h in prefactors.items()}

    def flux(self, state, omega, effective_energy, real_energy, x1, x2, *, chi1=0., chi2=0.):
        """Emitted energy flux in G=c=1 from exactly the same mode amplitudes."""
        args = self._prepare(state, omega, effective_energy, real_energy, x1, x2, chi1, chi2)
        omega, _, real_energy, *_ = args
        prefactors = self._prefactors(*args)
        signed = {(ell, m): h*_tail_abs_squared(ell, m*omega*real_energy).sqrt()
                  for (ell, m), h in prefactors.items()}
        return energy_flux_from_modes(signed, omega)

    def forward(self, state, omega, effective_energy, real_energy, x1, x2, *, chi1=0., chi2=0.):
        """Complex positive-m modes; negative partners are (-1)^ell h_lm* ."""
        missing = set(self.mode_indices)-PUBLISHED_PHASE_MODES
        if missing and self.residual_phase == "published":
            raise NotImplementedError(
                "DIN finite-mass delta_lm is unavailable for some requested modes; "
                "use amplitudes/flux, lmax<=4, or explicitly residual_phase='zero-unpublished'"
            )
        args = self._prepare(state, omega, effective_energy, real_energy, x1, x2, chi1, chi2)
        return self._complex_modes(args, self.mode_indices)

    def mode(self, ell: int, m: int, state, omega, effective_energy, real_energy, x1, x2,
             *, chi1=0., chi2=0.):
        """One complex mode, computing only its tail phase.

        This permits a 35-mode flux instance to supply a published mode
        such as (2,2) or (5,5) without inventing other residual phases.
        Negative m uses h_(ell,-m)=(-1)^ell conjugate(h_(ell,m)).
        """
        if type(ell) is not int or type(m) is not int or not (
            2 <= ell <= self.lmax and 1 <= abs(m) <= ell
        ):
            raise ValueError("Requested mode must have 2<=ell<=lmax and 1<=abs(m)<=ell")
        positive_mode = (ell, abs(m))
        if positive_mode not in PUBLISHED_PHASE_MODES and self.residual_phase == "published":
            raise NotImplementedError(
                "DIN finite-mass delta_lm is unavailable for this mode; "
                "explicitly select residual_phase='zero-unpublished' to omit it"
            )
        args = self._prepare(state, omega, effective_energy, real_energy, x1, x2, chi1, chi2)
        h = self._complex_modes(args, (positive_mode,))[positive_mode]
        return h if m > 0 else (-1)**ell*h.conj()

    def _complex_modes(self, args, mode_indices):
        omega, _, real_energy, x1, x2, _, _, phi, _ = args
        prefactors = self._prefactors(*args, mode_indices=mode_indices)
        delta = _published_phases(omega**(1/3), x1*x2, real_energy)
        result = {}
        for (ell, m), h in prefactors.items():
            phase = _tail_phase(ell, m, omega, real_energy)-m*phi
            if (ell, m) in delta:
                phase = phase+delta[ell, m]
            tail_amplitude = _tail_abs_squared(ell, m*omega*real_energy).sqrt()
            result[ell, m] = h*torch.polar(tail_amplitude, phase)
        return result
