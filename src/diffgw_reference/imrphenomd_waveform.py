"""Composed, scalar IMRPhenomD waveform specification.

Composes the independently sourced amplitude and phase slices. QNM frequencies
are supplied explicitly. Optional coefficient rows are explicit test inputs;
the default remains the digits printed in Khan et al., arXiv:1508.07253v2.
No production waveform, constant, or coefficient module is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import sympy as sp

from . import imrphenomd_amplitude as amp
from . import imrphenomd_phase as phase

SPEC_VERSION = "imrphenomd-composed-polarizations/v1"
COMPONENT_VERSIONS = (amp.SPEC_VERSION, phase.SPEC_VERSION,
                      "si-iau-waveform-units/v1")
SOURCES = {
    "khan2016": ("https://arxiv.org/html/1508.07253v2", "5-7", "17-18", "36-39"),
    "cutler_flanagan1994": ("https://arxiv.org/pdf/gr-qc/9402014v1", "62-63"),
    "units": "docs/math/units.md",
}
COMPONENT_IDS = {
    "physical_scale": "phenomd.waveform.physical_scale.v1",
    "phase_adapter": "phenomd.waveform.phase_adapter.v1",
    "polarizations": "phenomd.waveform.polarizations.v1",
    "default_cutoff": "phenomd.waveform.default_cutoff.v1",
    "polarization_derivatives": "phenomd.waveform.polarization_derivatives.v1",
}
COMPONENT_SOURCES = {
    COMPONENT_IDS["physical_scale"]: ("cutler_flanagan1994", "63", "units"),
    COMPONENT_IDS["phase_adapter"]: ("khan2016", "7", "project-convention"),
    COMPONENT_IDS["polarizations"]: ("khan2016", "5-7", "cutler_flanagan1994", "62"),
    COMPONENT_IDS["default_cutoff"]: ("project-convention", "Mf=0.2"),
    COMPONENT_IDS["polarization_derivatives"]: (
        "derived-product-rule", "physical_scale", "phase_adapter", "polarizations"),
}
_TORCH = ("torchwave.approximants.imrphenomd", "IMRPhenomD.__call__")
_JAX = ("jaxwave.approximants.imrphenomd", "IMRPhenomD.evaluate")
COMPONENT_BINDINGS = {
    component: {
        "reference": name,
        "torch": _TORCH,
        "jax": _JAX,
        "tests": ("test_complete_complex_polarizations",),
    } for name, component in COMPONENT_IDS.items()
}
COMPONENT_BINDINGS[COMPONENT_IDS["polarization_derivatives"]]["tests"] = (
    "test_complex_frequency_gradients", "test_complex_extrinsic_gradients",
    "test_complex_mass_scale_gradient")

# Exact nominal/SI/IAU definitions recorded in docs/math/units.md.
C_SI = sp.Integer(299792458)
SOLAR_GM_SI = sp.Rational("1.3271244e20")
AU_SI = sp.Integer(149597870700)
MSUN_SEC = SOLAR_GM_SI / C_SI**3
MPC_SI = 10**6 * 648000 * AU_SI / sp.pi


def physical_scale(f_hz, m_sec, d_sec, eta):
    """Positive-frequency quadrupole SPA normalization, CF Eqs. (62)-(63)."""
    f = sp.sympify(f_hz)
    return sp.sqrt(sp.Rational(5, 24) * eta) * m_sec**sp.Rational(5, 6) * (
        sp.pi**sp.Rational(-2, 3) * f**sp.Rational(-7, 6) / d_sec)


def phase_adapter(phi_u, phi_ref, u, u_ref, slope, f_hz, tc, phic,
                  convention="coalescence"):
    """Frontend phase origin and time gauge, separate from intrinsic phi(u)."""
    if convention == "coalescence":
        intrinsic = -sp.pi / 4 + phi_u - slope * u
    elif convention == "reference_frequency":
        if phi_ref is None or u_ref is None:
            raise ValueError("reference_frequency requires phi_ref and u_ref")
        intrinsic = phi_u - phi_ref - slope * (u - u_ref)
    else:
        raise ValueError(f"unknown phase convention: {convention}")
    return intrinsic + 2 * sp.pi * f_hz * tc - 2 * phic


def polarizations(amplitude, psi, inclination):
    """Positive-frequency, signed-quadrature project convention."""
    c = sp.cos(inclination)
    carrier = amplitude * sp.exp(-sp.I * psi)
    return ((1 + c**2) * carrier / 2, -sp.I * c * carrier)


def default_cutoff(u):
    """Inherited frontend boundary; not a calibrated paper coefficient."""
    return sp.sympify(u) <= sp.Rational(1, 5)


@dataclass(frozen=True)
class WaveformSpec:
    m_sec: sp.Expr
    d_sec: sp.Expr
    eta: sp.Expr
    amplitude: Mapping[str, sp.Expr]
    phase: Mapping[str, sp.Expr]
    slope: sp.Expr

    def expressions(self, frequency_hz, *, inclination=0, tc=0, phic=0,
                    convention="coalescence", f_ref_hz=None):
        """Return (h_plus, h_cross, unwrapped phase) for one positive f."""
        f = amp.exact(frequency_hz)
        if f <= 0:
            raise ValueError("the scalar reference requires positive frequency")
        u = self.m_sec * f
        if not default_cutoff(u):
            return sp.S.Zero, sp.S.Zero, sp.S.Zero
        a = self.amplitude
        p = self.phase
        amp_branch = ("inspiral" if u <= a["f1"] else
                      "intermediate" if u <= a["f3"] else "merger_ringdown")
        phase_branch = ("inspiral" if u <= p["f1"] else
                        "intermediate" if u <= p["f2"] else "merger_ringdown")
        scaled = a[amp_branch].subs(amp.U, u)
        phi_u = p[phase_branch].subs(phase.U, u)
        phi_ref = u_ref = None
        if convention == "reference_frequency":
            if f_ref_hz is None or amp.exact(f_ref_hz) <= 0:
                raise ValueError("positive f_ref_hz required")
            u_ref = self.m_sec * amp.exact(f_ref_hz)
            ref_branch = ("inspiral" if u_ref <= p["f1"] else
                          "intermediate" if u_ref <= p["f2"] else "merger_ringdown")
            phi_ref = p[ref_branch].subs(phase.U, u_ref)
        psi = phase_adapter(phi_u, phi_ref, u, u_ref, self.slope, f,
                            amp.exact(tc), amp.exact(phic), convention)
        h_plus, h_cross = polarizations(
            physical_scale(f, self.m_sec, self.d_sec, self.eta) * scaled,
            psi, amp.exact(inclination))
        return h_plus, h_cross, psi

    def evaluate(self, frequency_hz, *, dps=60, **kwargs):
        """Numerically evaluate the scalar expression at decimal precision."""
        return tuple(complex(sp.N(x, dps)) if i < 2 else float(sp.N(x, dps))
                     for i, x in enumerate(self.expressions(frequency_hz, **kwargs)))

    def evaluate_derivative(self, frequency_hz, parameter, *, dps=60, **kwargs):
        """Evaluate one smooth-branch complex polarization derivative."""
        return tuple(complex(sp.N(x, dps)) for x in
                     polarization_derivatives(self, frequency_hz, parameter, **kwargs))


def polarization_derivatives(spec: WaveformSpec, frequency_hz, parameter, *,
                             inclination=0, tc=0, phic=0,
                             convention="coalescence", f_ref_hz=None):
    """Exact partials at fixed QNM/fit inputs and away from the hard cutoff."""
    allowed = {"frequency_hz", "distance_mpc", "inclination", "tc", "phic",
               "mass_scale"}
    if parameter not in allowed:
        raise ValueError(f"unknown derivative parameter: {parameter}")
    f = amp.exact(frequency_hz)
    if f <= 0:
        raise ValueError("the scalar reference requires positive frequency")
    u = spec.m_sec * f
    if u > sp.Rational(1, 5):
        return sp.S.Zero, sp.S.Zero
    if parameter in ("frequency_hz", "mass_scale") and u == sp.Rational(1, 5):
        raise ValueError("frequency or mass-scale derivative is undefined at the hard cutoff")
    iota = amp.exact(inclination)
    hp, hc, psi = spec.expressions(
        f, inclination=iota, tc=tc, phic=phic,
        convention=convention, f_ref_hz=f_ref_hz)
    if parameter == "distance_mpc":
        distance_mpc = spec.d_sec * C_SI / MPC_SI
        return -hp / distance_mpc, -hc / distance_mpc
    if parameter == "tc":
        return -2 * sp.pi * sp.I * f * hp, -2 * sp.pi * sp.I * f * hc
    if parameter == "phic":
        return 2 * sp.I * hp, 2 * sp.I * hc
    a = spec.amplitude
    amp_branch = ("inspiral" if u <= a["f1"] else
                  "intermediate" if u <= a["f3"] else "merger_ringdown")
    scale = physical_scale(f, spec.m_sec, spec.d_sec, spec.eta)
    scaled = a[amp_branch].subs(amp.U, u)
    carrier = scale * scaled * sp.exp(-sp.I * psi)
    if parameter == "inclination":
        return (-sp.cos(iota) * sp.sin(iota) * carrier,
                sp.I * sp.sin(iota) * carrier)
    p = spec.phase
    phase_branch = ("inspiral" if u <= p["f1"] else
                    "intermediate" if u <= p["f2"] else "merger_ringdown")
    scaled_prime = sp.diff(a[amp_branch], amp.U).subs(amp.U, u)
    intrinsic_prime = sp.diff(p[phase_branch], phase.U).subs(phase.U, u) - spec.slope
    if parameter == "mass_scale":
        phase_ln = u * intrinsic_prime
        if convention == "reference_frequency":
            u_ref = spec.m_sec * amp.exact(f_ref_hz)
            ref_branch = ("inspiral" if u_ref <= p["f1"] else
                          "intermediate" if u_ref <= p["f2"] else "merger_ringdown")
            phase_ln -= u_ref * (
                sp.diff(p[ref_branch], phase.U).subs(phase.U, u_ref) - spec.slope)
        carrier_ln = scale * sp.exp(-sp.I * psi) * (
            sp.Rational(5, 6) * scaled + u * scaled_prime
            - sp.I * scaled * phase_ln)
        cosine = sp.cos(iota)
        return ((1 + cosine**2) * carrier_ln / 2,
                -sp.I * cosine * carrier_ln)
    phase_prime = spec.m_sec * intrinsic_prime + 2 * sp.pi * amp.exact(tc)
    amplitude_prime = scale * (
        -sp.Rational(7, 6) * scaled / f + spec.m_sec * scaled_prime)
    carrier_prime = sp.exp(-sp.I * psi) * (
        amplitude_prime - sp.I * scale * scaled * phase_prime)
    cosine = sp.cos(iota)
    return ((1 + cosine**2) * carrier_prime / 2,
            -sp.I * cosine * carrier_prime)


def compose(m1, m2, chi1, chi2, distance_mpc, f_rd, f_damp,
            *, coefficient_rows: Mapping[str, tuple[str, ...]] | None = None):
    """Bind masses/spins, fit rows and supplied dimensionless QNM pair."""
    m1, m2, chi1, chi2, distance = map(
        amp.exact, (m1, m2, chi1, chi2, distance_mpc))
    f_rd, f_damp = amp.exact(f_rd), amp.exact(f_damp)
    if min(m1, m2, distance, f_rd, f_damp) <= 0:
        raise ValueError("masses, distance and QNM frequencies must be positive")
    eta, delta, chi_s, chi_a, chi_pn = amp.spin_coordinates(m1,m2,chi1,chi2)
    rows = coefficient_rows or {}
    a_fit = {name: amp.fit_expr(name, eta, chi_pn,
              coefficients=rows.get(name)) for name in amp.FIT_NAMES}
    p_fit = {name: phase.fit_expr(name, eta, chi_pn,
              coefficients=rows.get(name)) for name in phase.MODEL_FIT_NAMES}
    a = amp.amplitude_pieces(eta=eta,delta=delta,chi_s=chi_s,chi_a=chi_a,
                             f_rd=f_rd,f_damp=f_damp,**a_fit)
    p = phase.phase_pieces(eta=eta,delta=delta,chi_s=chi_s,chi_a=chi_a,
                           f_rd=f_rd,f_damp=f_damp,
                           **{name:p_fit[name] for name in
                              ("sigma1","sigma2","sigma3","sigma4","beta2","beta3",
                               "alpha2","alpha3","alpha4","alpha5")})
    slope = phase.peak_gauge_slope(
        eta=eta,alpha1_raw=p_fit["alpha1"],
        alpha2=p_fit["alpha2"],alpha3=p_fit["alpha3"],
        alpha4=p_fit["alpha4"],alpha5=p_fit["alpha5"],
        f_peak=a["f3"],f_rd=f_rd,f_damp=f_damp)
    return WaveformSpec((m1+m2)*MSUN_SEC, distance*MPC_SI/C_SI,
                        eta,a,p,slope)
