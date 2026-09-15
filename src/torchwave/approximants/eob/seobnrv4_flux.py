"""Primary-equation spinning amplitude moduli for an explicit v4 reconstruction.

Taracchini 1202.0790 Appendix A and 1311.2544 around Eq. (4); Pan
1006.0431v2 Eqs. (28), (29), (33), (41), (42), (F26), Appendix D;
Buonanno--Faye--Hinderer 1209.6349v2 Eq. (4.27); Bohé
1611.03703v3 Eqs. (2.4), (2.5), (2.9). No residual phase or NQC.

The required truncation selection records a publication-level ambiguity:
Pan's ell=8 tabulated series end at v**2, whereas its general Eq. (33)
also supplies the even-parity leading spin term. Neither selection is a
claim of implementation parity. See pass-6-spinning-flux.md.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ...constants import EULER_GAMMA
from .factorized_modes import (
    _mass_times_leading_spin_f,
    _newtonian_constant,
    _rho_table,
    _tail_abs_squared,
    energy_flux_from_modes,
)
from .seobnrv4_amplitude22 import _rho22


def _nonspin_rho(v: Tensor, nu: Tensor) -> dict[tuple[int, int], Tensor]:
    """Taracchini (A8)--(A14), dropping calibrated nu*rho22^(4).

    DIN (C1)--(C35) already supplies the common terms. Nine additional
    test-particle terms are printed in Taracchini. Retain DIN/Pan's rho55
    v**2 sign: the extra minus in Taracchini (A11a) fails their nu=0 limit.
    """
    rho = _rho_table(v*v, nu)
    el = {m: EULER_GAMMA+math.log(2*m)+torch.log(v) for m in (1, 2, 3)}
    extra = {
        (2, 1): (5029963*el[1]/5927040-63735873771463/16569158860800)*v**10,
        (3, 2): (17056*el[2]/8505-10607269449358/3072140846775)*v**8,
        (4, 3): (1664224207351/195343948800-1571*el[3]/770)*v**6,
        (4, 1): (1227423222031/1758095539200-1571*el[1]/6930)*v**6,
        (5, 4): -16213384/15526875*v**4,
        (5, 2): -7187914/15526875*v**4,
        (6, 6): -1025435/659736*v**4,
        (6, 4): -476887/659736*v**4,
        (6, 2): -817991/3298680*v**4,
    }
    for mode, value in extra.items():
        rho[mode] = rho[mode]+value
    return rho


def _spin_rho(v, nu, delta, cs, ca, *, spin_truncation):
    """Spin portions of ordinary rho for modes without odd-m f splitting.

    Kerr spin q=chiK=(1-2nu)chiS+delta*chiA, as specified in Taracchini
    2014. Comparable-mass coefficients replace the same PN spin terms;
    they are never added on top of the test-particle counterparts.
    """
    q = (1-2*nu)*cs+delta*ca
    r32v1 = -4*nu*cs/(3*(3*nu-1))
    result = {
        # The v**2 spin square is Pan (F26), -rho32_v1**2.
        (3, 2): r32v1*v-r32v1.square()*v**2+2*q/9*v**3+q*q/3*v**4
        -2788*q/1215*v**5+488*q*q/405*v**6,
        (4, 4): -((42*nu**2-41*nu+10)*cs+(10-39*nu)*delta*ca)
        /(15*(1-3*nu))*v**3+q*q/2*v**4-69*q/55*v**5+217*q*q/3960*v**6,
        (4, 2): -((78*nu**2-59*nu+10)*cs+(10-21*nu)*delta*ca)
        /(15*(1-3*nu))*v**3+q*q/2*v**4-7*q/110*v**5+2323*q*q/3960*v**6,
        (5, 5): -2*q/3*v**3+q*q/2*v**4-241*q/195*v**5,
        (5, 4): -2*q/15*v**3+2*q*q/5*v**4,
        (5, 3): -2*q/3*v**3+q*q/2*v**4-103*q/325*v**5,
        (5, 2): -2*q/15*v**3+2*q*q/5*v**4,
        (5, 1): -2*q/3*v**3+q*q/2*v**4+139*q/975*v**5,
    }
    for m in (2, 4, 6):
        result[6, m] = -2*q/3*v**3+q*q/2*v**4
    for m in (1, 3, 5):
        result[6, m] = -2*q/9*v**3
    for m in (1, 3, 5, 7):
        result[7, m] = -2*q/3*v**3
    if spin_truncation == "generic-leading-ell8":
        for m in (2, 4, 6, 8):
            result[8, m] = -2*q/3*v**3
    return result


def _higher_kerr_f(v: Tensor, q: Tensor) -> dict[tuple[int, int], Tensor]:
    """Pan (28) spin f terms after removing finite-mass replacements.

    Excludes f21's v and v**3 and f33/f31's v**3. f43/f41 have no
    test-particle v term. Only the printed spin terms are imported;
    the entire nonspin factor is supplied separately by rho_NS**ell.
    """
    el1 = EULER_GAMMA+math.log(2)+torch.log(v)
    return {
        (2, 1): -3*q*q*v**4+3*q*(4*q*q-27)/16*v**5+4163*q*q/252*v**6
        +(-2593*q**3/168+107*q*el1/35-11847887*q/1058400)*v**7,
        (3, 3): 3*q*q/2*v**4+2*q/3*v**5-7*q*q/4*v**6
        +(6187*q/330-q**3)*v**7,
        (3, 1): -5*q*q/2*v**4+38*q/9*v**5+43*q*q/12*v**6
        +(11*q**3/3-2657*q/594)*v**7,
        (4, 3): 3*q*q/2*v**4-12113*q/1540*v**5,
        (4, 1): 3*q*q/2*v**4+(-10*q**3/3-20033*q/13860)*v**5,
    }


class SEOBNRv4PublishedFlux(nn.Module):
    """Positive-m amplitude moduli and flux for a declared reconstruction.

    ``spin_truncation`` must be explicitly chosen as ``tabulated`` (Pan
    Eq. 29/D1 truncations) or ``generic-leading-ell8`` (also Pan Eq. 33
    for even-parity ell=8). All 35 modes can be summed without phases.

    State[...,4]=(r,phi,p_r_star,p_phi), in M=G=c=1 units; p_phi is
    |L|/(mu*M)>0. Outputs are |R*h_lm/M| and emitted energy flux.
    Frequencies are M*Omega; H_eff/mu and H_real/M include rest mass.
    The explicit quantities broadcast with state's batch dimensions.
    Zero-radial omega is evaluated at fixed r,p_phi, NOT at a newly
    solved circular angular momentum. Only the Newtonian prefactor uses
    v_phi=omega*omega_zero_radial**(-2/3); residuals and odd-parity source
    p_phi*v use v=omega**(1/3). No phase, NQC or flux floor is present.
    """

    def __init__(self, lmax: int = 8, *, spin_truncation: str,
                 retain_pan_cubic_spin: bool = False):
        super().__init__()
        if type(lmax) is not int or not 2 <= lmax <= 8:
            raise ValueError("lmax must be an integer from 2 through 8")
        if spin_truncation not in {"tabulated", "generic-leading-ell8"}:
            raise NotImplementedError("Choose tabulated or generic-leading-ell8 explicitly")
        if not isinstance(retain_pan_cubic_spin, bool):
            raise TypeError("retain_pan_cubic_spin must be an explicit bool")
        self.lmax = lmax
        self.spin_truncation = spin_truncation
        self.retain_pan_cubic_spin = retain_pan_cubic_spin
        self.mode_indices = tuple((ell, m) for ell in range(2, lmax+1)
                                  for m in range(1, ell+1))
        self._constants = {mode: abs(_newtonian_constant(*mode)) for mode in self.mode_indices}

    @staticmethod
    def _prepare(state, **quantities):
        if not isinstance(state, Tensor) or state.dtype not in (torch.float32, torch.float64):
            raise TypeError("state must be a float32 or float64 Torch tensor")
        if state.ndim == 0 or state.shape[-1] != 4:
            raise ValueError("state must have trailing dimension (r,phi,p_r_star,p_phi)")
        if not bool(torch.isfinite(state).all()) or not bool((state[..., 0] > 0).all()):
            raise ValueError("state must be finite with positive radius")
        keys = tuple(quantities)
        values = [torch.as_tensor(quantities[key], dtype=state.dtype, device=state.device)
                  for key in keys]
        if any(not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("Orbital quantities, mass fractions and spins must be finite")
        *values, j = torch.broadcast_tensors(*values, state[..., 3])
        args = dict(zip(keys, values))
        for key in ("omega", "omega_zero_radial", "heff_over_mu", "hreal_over_M"):
            if not bool((args[key] > 0).all()):
                raise ValueError(f"{key} must be positive and use the stated rest-energy normalization")
        if not bool((j > 0).all()):
            raise ValueError("p_phi=|L|/(mu*M) must be positive")
        a, b = args["x1"], args["x2"]
        if (not bool(((a >= 0) & (b >= 0)).all())
                or not bool((torch.abs(a+b-1) <= 16*torch.finfo(state.dtype).eps).all())):
            raise ValueError("Mass fractions must be nonnegative and sum to one")
        if not bool(((args["chi1"].abs() <= 1) & (args["chi2"].abs() <= 1)).all()):
            raise ValueError("Aligned dimensionless spins must lie in [-1,1]")
        return args, j

    def amplitudes(self, state, *, omega, omega_zero_radial, heff_over_mu,
                   hreal_over_M, x1, x2, chi1=0., chi2=0.):
        """Return { (ell,m): |R*h_lm/M| } for all positive m through lmax."""
        signed = self._signed_amplitudes(
            state, omega=omega, omega_zero_radial=omega_zero_radial,
            heff_over_mu=heff_over_mu, hreal_over_M=hreal_over_M,
            x1=x1, x2=x2, chi1=chi1, chi2=chi2)
        return {mode: value.abs() for mode, value in signed.items()}

    def _signed_amplitudes(self, state, *, omega, omega_zero_radial, heff_over_mu,
                           hreal_over_M, x1, x2, chi1=0., chi2=0.):
        """Real residual times positive prefactors, with no phase selected."""
        args, j = self._prepare(
            state, omega=omega, omega_zero_radial=omega_zero_radial,
            heff_over_mu=heff_over_mu, hreal_over_M=hreal_over_M,
            x1=x1, x2=x2, chi1=chi1, chi2=chi2)
        w, w0, heff, hreal, a, b, s1, s2 = args.values()
        nu, delta, cs, ca = a*b, a-b, (s1+s2)/2, (s1-s2)/2
        v, vphi = w.pow(1/3), w*w0.pow(-2/3)
        rho = _nonspin_rho(v, nu)
        for mode, spin in _spin_rho(v, nu, delta, cs, ca,
                                    spin_truncation=self.spin_truncation).items():
            rho[mode] = rho[mode]+spin
        rho[2, 2] = _rho22(
            v, nu, delta, cs, ca,
            retain_pan_cubic_spin=self.retain_pan_cubic_spin,
        )
        higher_f = _higher_kerr_f(v, (1-2*nu)*cs+delta*ca)
        result = {}
        for ell, m in self.mode_indices:
            epsilon = (ell+m) % 2
            p = ell+epsilon-1
            c = b**p+(-1)**m*a**p
            mass_residual = c*rho[ell, m]**ell
            if ell <= 4 and m % 2:
                mass_residual = mass_residual+c*higher_f[ell, m]
                mass_residual = mass_residual+_mass_times_leading_spin_f(
                    ell, m, v, a, b, s1, s2)
                if (ell, m) == (2, 1):
                    # c/delta is a polynomial divided difference, including
                    # at equal masses. BFH (4.27a), divided by the J source,
                    # replaces Pan's Kerr 61*q*v**3/12 (not added to it).
                    c_over_delta = -sum(a**k*b**(p-1-k) for k in range(p))
                    mass_residual = mass_residual+(
                        c*(427+79*nu)*cs+c_over_delta*(427+131*nu)*ca)*v**3/84
            source = heff if epsilon == 0 else j*v
            tail = _tail_abs_squared(ell, m*w*hreal).sqrt()
            result[ell, m] = (self._constants[ell, m]*nu*vphi**(ell+epsilon)
                              *source*tail*mass_residual)
        return result

    def energy_flux(self, state, **quantities):
        """Bohé (2.9): positive-m sum, including negative-m partners once."""
        modes = self._signed_amplitudes(state, **quantities)
        omega = torch.as_tensor(quantities["omega"], dtype=state.dtype, device=state.device)
        # Square the real residual directly. abs().square() loses its second
        # derivative at an exactly vanishing mode, e.g. odd m at equal masses.
        return energy_flux_from_modes(modes, omega)

    def forward(self, state, **quantities):
        return self.energy_flux(state, **quantities)

    @staticmethod
    def from_hamiltonian(state, hamiltonian, *, omega=None):
        """Explicit quantities for amplitudes/energy_flux, preserving autograd.

        Uses the native Hamiltonian's inference-safe local partials. An
        optional supplied omega is retained, while omega_zero_radial is
        always evaluated from the Hamiltonian at the same r,p_phi.
        """
        if not isinstance(state, Tensor) or state.ndim == 0 or state.shape[-1] != 4:
            raise ValueError("state must have trailing dimension (r,phi,p_r_star,p_phi)")
        r, pr, j = state[..., 0], state[..., 2], state[..., 3]
        if omega is None:
            _, omega, _ = hamiltonian.equations_of_motion_conservative(r, pr, j)
        _, omega0, _ = hamiltonian.equations_of_motion_conservative(r, torch.zeros_like(pr), j)
        heff = hamiltonian.effective_hamiltonian(r, pr, j)
        def parameter(value):
            # Native Hamiltonians append trajectory dimensions to their
            # parameter batch; preserve that convention in this adapter.
            while value.ndim < heff.ndim:
                value = value.unsqueeze(-1)
            return value
        return dict(omega=omega, omega_zero_radial=omega0, heff_over_mu=heff,
                    hreal_over_M=hamiltonian.real_hamiltonian_from_eff(heff),
                    x1=parameter(hamiltonian.m1), x2=parameter(hamiltonian.m2),
                    chi1=parameter(hamiltonian.chi1), chi2=parameter(hamiltonian.chi2))

    def radiation_reaction_forces(self, omega, pr_star, pphi, *, r, hamiltonian):
        """Integrator adapter returning (F_r_star,F_phi), Bohé (2.9).

        Momenta are reduced-mass normalized: F_phi=-flux/(nu*omega),
        F_r_star=(p_r_star/p_phi)*F_phi. Exactly nu=0 is valid for
        amplitude/flux evaluation, but does not define reduced-mass forces.
        """
        if not isinstance(r, Tensor):
            raise TypeError("r must be a Torch tensor")
        r, pr_star, pphi = torch.broadcast_tensors(r, pr_star, pphi)
        if not bool((pphi > 0).all()):
            raise ValueError("Reduced-mass radiation reaction requires p_phi>0")
        state = torch.stack((r, torch.zeros_like(r), pr_star, pphi), dim=-1)
        args = self.from_hamiltonian(state, hamiltonian, omega=omega)
        nu = args["x1"]*args["x2"]
        if not bool((nu > 0).all()):
            raise ValueError("Reduced-mass radiation reaction requires nu>0")
        flux = self.energy_flux(state, **args)
        torque = -flux/(nu*args["omega"])
        return pr_star/pphi*torque, torque
