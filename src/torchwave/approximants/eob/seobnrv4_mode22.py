"""Published factorized (2,2) phase building blocks, before NQC.

These equation components have independent special-function and PN-limit
checks; they do not select the unresolved SEOBNRv4 phase inheritance rules. The mode assembler requires a supplied residual
phase; the optional composer requires each spin choice explicitly.

Sources: Taracchini 1202.0790v2, (A1)--(A6), (A16); Pan 1006.0431v2,
(26), (27a), (38a); Taracchini 1311.2544, text following (4); Bohé
1611.03703v3, (2.1), (2.6). See pass-6-eob-phase22.md for the energy
repair, source evidence and remaining choices. No implementation-parity
claim, NQC, radiation reaction, trajectory or merger attachment is made.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

from .factorized_modes import _tail_abs_squared, _tail_phase
from .nqc import nqc_22_factor
from .seobnrv4_amplitude22 import seobnrv4_published_amplitude22


SpinV6 = Literal["bohe_total_energy", "pan_test_particle", "bohe_plus_pan", "none"]


def _inputs(omega: Tensor, **kwargs: Tensor | float) -> dict[str, Tensor]:
    if not isinstance(omega, Tensor) or omega.dtype not in (torch.float32, torch.float64):
        raise ValueError("omega must be a float32 or float64 Torch tensor")
    values = {}
    for name, value in dict(omega=omega, **kwargs).items():
        if torch.is_complex(torch.as_tensor(value)):
            raise ValueError(f"{name} must be real")
        values[name] = torch.as_tensor(value, dtype=omega.dtype, device=omega.device)
        if not bool(torch.all(torch.isfinite(values[name]))):
            raise ValueError(f"{name} must be finite")
    for name in ("omega", "hreal_over_M"):
        if name in values and not bool(torch.all(values[name] > 0)):
            raise ValueError(f"{name} must be positive; hreal_over_M includes rest mass")
    if "nu" in values and not bool(torch.all((values["nu"] >= 0) & (values["nu"] <= .25))):
        raise ValueError("nu must lie in [0, 1/4]")
    for name in ("q_kerr", "chi"):
        if name in values and not bool(torch.all(values[name].abs() <= 1)):
            raise ValueError(f"{name} must lie in [-1, 1]")
    return values


def _boolean(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be an explicit bool")


def tail22_phase(omega: Tensor, *, hreal_over_M: Tensor | float) -> Tensor:
    """Continuous phase of Taracchini (A6)'s T22, zero at omega->0.

    G=c=M=1, omega=M*Omega>0, hreal_over_M=Hreal/M INCLUDING REST MASS.
    The required radius is r0/M=2/sqrt(e). With khat=2*omega*hreal_over_M,
    T22=Gamma(3-2i*khat)/2 * exp(pi*khat)
        * exp(2i*khat*log(8*omega/sqrt(e))).

    The shared Gamma phase uses recurrence and Stirling's expansion
    (DLMF 5.5.1, 5.11.1), without wrapping. Its DIN radius 2M is converted
    exactly by subtracting khat. All inputs broadcast; gradients are kept.
    """
    p = _inputs(omega, hreal_over_M=hreal_over_M)
    energy = p["hreal_over_M"]
    return _tail_phase(2, 2, omega, energy)-2*omega*energy


def tail22(omega: Tensor, *, hreal_over_M: Tensor | float) -> Tensor:
    """Complex Taracchini (A6) tail, with r0/M=2/sqrt(e)."""
    phase = tail22_phase(omega, hreal_over_M=hreal_over_M)
    energy = torch.as_tensor(hreal_over_M, dtype=omega.dtype, device=omega.device)
    return torch.polar(_tail_abs_squared(2, 2*omega*energy).sqrt(), phase)


def delta22_nonspin(
    omega: Tensor, *, hreal_over_M: Tensor | float, nu: Tensor | float,
) -> Tensor:
    """Full printed Taracchini (A16), including its (Hreal*Omega)^3 term.

    The label 'nonspin' refers to the PN coefficients. The supplied total
    real energy can depend on spin through the conservative dynamics.
    """
    p = _inputs(omega, hreal_over_M=hreal_over_M, nu=nu)
    u = omega*p["hreal_over_M"]
    return (7*u/3+428*math.pi*u**2/105
            +(1712*math.pi**2/315-2203/81)*u**3-24*p["nu"]*omega.pow(5/3))


def delta22_bohe_spin(
    omega: Tensor, *, hreal_over_M: Tensor | float, q_kerr: Tensor | float,
) -> Tensor:
    """Bohé (2.6), explicitly repaired to use TOTAL real energy.

    q_kerr=(1-2nu)*chi_S+(x1-x2)*chi_A. The printed binding-energy
    definition (2.1) would make this v^10 rather than the stated v^6 and
    remove the test-particle limit. This repair is a declared inference.
    """
    p = _inputs(omega, hreal_over_M=hreal_over_M, q_kerr=q_kerr)
    return -4*p["q_kerr"]*(omega*p["hreal_over_M"])**2/3


def delta22_pan_spin(
    omega: Tensor, *, q_kerr: Tensor | float, include_v6: bool, include_v8: bool,
) -> Tensor:
    """Explicitly selected Kerr phase terms of Pan (27a).

    At comparable masses replacing the Kerr spin by q_kerr is the declared
    hybridization, not a known finite-mass PN result. Neither term is
    selected by default. Both coefficients are in powers of omega^(1/3),
    without multiplying omega by a real-energy factor.
    """
    _boolean("include_v6", include_v6)
    _boolean("include_v8", include_v8)
    p = _inputs(omega, q_kerr=q_kerr)
    coefficient = torch.zeros_like(omega)
    if include_v6:
        coefficient = coefficient-4*omega**2/3
    if include_v8:
        coefficient = coefficient+20*omega.pow(8/3)/63
    return p["q_kerr"]*coefficient


def delta22_positive_chi_calibration(
    omega: Tensor, *, nu: Tensor | float, chi: Tensor | float,
) -> Tensor:
    """Taracchini 2014's explicit 540*nu*max(chi,0)*omega^2 term.

    chi=chi_S+(x1-x2)*chi_A/(1-2nu), with the bodies labeled together.
    Its retention in v4 is unresolved; calling this helper is a choice.
    At chi=0 the function is continuous but has a derivative kink; Torch
    uses the nonnegative branch's derivative, as the printed chi>=0 rule.
    """
    p = _inputs(omega, nu=nu, chi=chi)
    positive = torch.where(p["chi"] >= 0, p["chi"], torch.zeros_like(p["chi"]))
    return 540*p["nu"]*positive*omega**2


def seobnrv4_published_delta22(
    omega: Tensor,
    *,
    hreal_over_M: Tensor | float,
    x1: Tensor | float,
    x2: Tensor | float,
    chi1: Tensor | float,
    chi2: Tensor | float,
    spin_v6: SpinV6,
    include_pan_v8: bool,
    include_positive_chi_540: bool,
) -> Tensor:
    """Compose an EXPERIMENTAL residual phase using required caller choices.

    Always includes the full nonspin A16. ``spin_v6`` explicitly chooses
    Bohé's total-energy repair, Pan's test-particle term, their sum, or
    neither. The sum contains twice Pan's leading Kerr-limit coefficient;
    it is available as a deliberate inheritance reading, not endorsed as
    the intended v4 prescription. The two remaining choices independently
    control Pan's v^8 and Taracchini's positive-chi calibrated v^6 term.
    No combination is identified as implementation-compatible SEOBNRv4.
    """
    if spin_v6 not in ("bohe_total_energy", "pan_test_particle", "bohe_plus_pan", "none"):
        raise ValueError("spin_v6 must explicitly select bohe_total_energy, pan_test_particle, bohe_plus_pan or none")
    _boolean("include_pan_v8", include_pan_v8)
    _boolean("include_positive_chi_540", include_positive_chi_540)
    p = _inputs(omega, hreal_over_M=hreal_over_M, x1=x1, x2=x2, chi1=chi1, chi2=chi2)
    a, b = p["x1"], p["x2"]
    if (not bool(torch.all((a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)))
            or not bool(torch.all((a+b-1).abs() <= 16*torch.finfo(omega.dtype).eps))):
        raise ValueError("mass fractions x1,x2 must lie in [0,1] and sum to one")
    if not bool(torch.all((p["chi1"].abs() <= 1) & (p["chi2"].abs() <= 1))):
        raise ValueError("aligned dimensionless spins must lie in [-1,1]")
    nu = a*b
    cs, ca = (p["chi1"]+p["chi2"])/2, (p["chi1"]-p["chi2"])/2
    q_kerr = (1-2*nu)*cs+(a-b)*ca
    result = delta22_nonspin(omega, hreal_over_M=p["hreal_over_M"], nu=nu)
    if spin_v6 in ("bohe_total_energy", "bohe_plus_pan"):
        result = result+delta22_bohe_spin(omega, hreal_over_M=p["hreal_over_M"], q_kerr=q_kerr)
    result = result+delta22_pan_spin(
        omega, q_kerr=q_kerr, include_v6=spin_v6 in ("pan_test_particle", "bohe_plus_pan"),
        include_v8=include_pan_v8,
    )
    if include_positive_chi_540:
        result = result+delta22_positive_chi_calibration(omega, nu=nu, chi=q_kerr/(1-2*nu))
    return result


def seobnrv4_published_phase22(
    omega: Tensor, phi: Tensor | float, *,
    hreal_over_M: Tensor | float, residual_phase: Tensor | float,
) -> Tensor:
    """Continuous GW phase in h22=A*exp(-i*phase22), including constant pi.

    ``phi`` must be a continuous orbital phase. The required residual_phase
    is the caller's delta22, not the full GW phase. Taracchini A1/A3 and
    Pan (38a)'s negative Newtonian prefactor contributes pi, giving
    phase22=2*phi-tail_phase-delta22+pi. Thus d(phase22)/d(t/M) is the
    uncorrected dimensionless GW frequency for NQC matching. This function
    does not infer that frequency as 2*omega or unwrap sampled arguments.
    """
    p = _inputs(omega, phi=phi, hreal_over_M=hreal_over_M, residual_phase=residual_phase)
    return 2*p["phi"]-tail22_phase(omega, hreal_over_M=p["hreal_over_M"])-p["residual_phase"]+math.pi


def seobnrv4_published_mode22(
    omega: Tensor,
    phi: Tensor | float,
    *,
    residual_phase: Tensor | float,
    omega_zero_radial: Tensor | float,
    heff_over_mu: Tensor | float,
    hreal_over_M: Tensor | float,
    x1: Tensor | float,
    x2: Tensor | float,
    chi1: Tensor | float = 0.,
    chi2: Tensor | float = 0.,
    retain_pan_cubic_spin: bool = False,
) -> Tensor:
    """Return complex (R/M)*h22 before NQC for a SUPPLIED residual phase.

    The amplitude contract and its declared reconstruction choices are
    those of seobnrv4_published_amplitude22. No residual spin-phase choice
    is implicit. All quantities are dimensionless (G=c=M=1) and broadcast;
    inputs use omega's device and floating dtype. The output is complex64
    or complex128. Multiply by GM/(c^2 R) for physical strain.
    """
    phase = seobnrv4_published_phase22(
        omega, phi, hreal_over_M=hreal_over_M, residual_phase=residual_phase,
    )
    amplitude = seobnrv4_published_amplitude22(
        omega, omega_zero_radial=omega_zero_radial, heff_over_mu=heff_over_mu,
        hreal_over_M=hreal_over_M, x1=x1, x2=x2, chi1=chi1, chi2=chi2,
        retain_pan_cubic_spin=retain_pan_cubic_spin,
    )
    return torch.polar(amplitude, -phase)


def seobnrv4_published_mode22_from_hamiltonian(
    state: Tensor,
    hamiltonian,
    radiation_reaction,
    *,
    spin_v6: SpinV6,
    include_pan_v8: bool,
    include_positive_chi_540: bool,
    retain_pan_cubic_spin: bool = False,
    omega: Tensor | float | None = None,
    nqc_coefficients=None,
    nqc_sample_dims: int = 0,
) -> Tensor:
    """Assemble the published pre-NQC mode from a dynamical state.

    ``radiation_reaction`` must expose ``from_hamiltonian(state,
    hamiltonian, omega=...)`` and return the eight quantities consumed by
    :func:`seobnrv4_published_mode22`.  This adapter keeps the source-level
    separation between the factorized mode and the dissipative flux while
    making their state convention explicit.  The three unresolved residual
    phase choices remain required keyword arguments; no SEOBNRv4 inheritance
    rule is selected here.

    ``state`` may end in a trajectory sample axis, ``(..., N, 4)``. If
    coefficients have batch axes before that sample axis, pass
    ``nqc_sample_dims=1`` (the value is forwarded to :func:`nqc_22_factor`).
    The default ``0`` preserves ordinary elementwise broadcasting for a
    single state or coefficients already broadcast over every sample.

    If ``nqc_coefficients`` is supplied, it must be the caller's solved
    ``(a1,a2,a3),(b1,b2)`` coefficients.  The published papers do not provide
    an arbitrary-parameter coefficient table, so coefficients are never
    inferred by this function.  NQC is applied after the mode is assembled
    and therefore does not enter the radiation-reaction quantities.

    The continuous-frequency driver intentionally composes the lower-level
    phase/mode functions from its already computed directional-derivative
    quantities. Routing that path through this state adapter would call
    ``from_hamiltonian`` a second time and obscure the shared autograd graph;
    this adapter is the explicit state-to-mode seam for callers that do not
    already have those quantities.
    """
    if not isinstance(state, Tensor) or state.ndim == 0 or state.shape[-1] != 4:
        raise ValueError("state must be a tensor with trailing dimension (r, phi, p_r*, p_phi)")
    if state.dtype not in (torch.float32, torch.float64):
        raise TypeError("state must use float32 or float64")
    if not bool(torch.isfinite(state).all()):
        raise ValueError("state must be finite")
    if not hasattr(radiation_reaction, "from_hamiltonian"):
        raise TypeError("radiation_reaction must provide from_hamiltonian(state, hamiltonian, omega=...)")

    quantities = radiation_reaction.from_hamiltonian(state, hamiltonian, omega=omega)
    required = ("omega", "omega_zero_radial", "heff_over_mu", "hreal_over_M",
                "x1", "x2", "chi1", "chi2")
    if not isinstance(quantities, dict) or any(name not in quantities for name in required):
        raise ValueError("from_hamiltonian must return omega, energies, mass fractions and spins")
    residual = seobnrv4_published_delta22(
        **{name: quantities[name] for name in ("omega", "hreal_over_M", "x1", "x2", "chi1", "chi2")},
        spin_v6=spin_v6, include_pan_v8=include_pan_v8,
        include_positive_chi_540=include_positive_chi_540,
    )
    mode = seobnrv4_published_mode22(
        quantities["omega"], state[..., 1], residual_phase=residual,
        **{name: quantities[name] for name in required if name != "omega"},
        retain_pan_cubic_spin=retain_pan_cubic_spin,
    )
    if nqc_coefficients is not None:
        mode = mode*nqc_22_factor(
            state[..., 0], state[..., 2], quantities["omega"], nqc_coefficients,
            sample_dims=nqc_sample_dims,
        )
    return mode
