"""Scalar composition of the IMRPhenomXAS carrier's complex polarizations.

The intrinsic pieces are independently recorded in the adjacent XAS modules.
Calibrated fit outputs, QNM frequencies, and join locations are supplied inputs.
The carrier's intermediate normalization and phase sign are explicitly selected;
they are not asserted to be the paper-literal alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping

import sympy as sp

from . import imrphenomxas_amplitude as amp
from . import imrphenomxas_intermediate as middle
from . import imrphenomxas_inspiral_phase as inspiral
from . import imrphenomxas_phase_branches as phase
from . import imrphenomd_waveform as units

SPEC_VERSION = "imrphenomxas-composed-polarizations-carrier/v1"
MASS_SCALING_VERSION = "imrphenomxas-total-mass-scaling/v1"
MASS_SCALING_COMPONENT_VERSIONS = (SPEC_VERSION,)
COMPONENT_VERSIONS = (amp.SPEC_VERSION, middle.SPEC_VERSION,
                      inspiral.SPEC_VERSION, phase.SPEC_VERSION,
                      "si-iau-waveform-units/v1")
SOURCES = {
    "pratten2020v1": ("https://arxiv.org/html/2001.11412v1", "1", "6", "14-16", "52", "66-67", "71", "84", "85", "90", "96"),
    "units": "docs/math/units.md",
}
COMPONENT_IDS = {
    "branch_selection": "phenomxas.waveform.branch_selection.v1",
    "physical_scale": "phenomxas.waveform.physical_scale.v1",
    "phase_adapter": "phenomxas.waveform.phase_adapter.v1",
    "polarizations": "phenomxas.waveform.polarizations.v1",
    "nonpositive_mask": "phenomxas.waveform.nonpositive_mask.v1",
    "extrinsic_derivatives": "phenomxas.waveform.extrinsic_derivatives.v1",
    "frequency_derivatives": "phenomxas.waveform.frequency_derivatives.v1",
    "mass_scaling": "phenomxas.waveform.mass_scaling.v1",
    "mass_derivatives": "phenomxas.waveform.mass_derivatives.v1",
}
COMPONENT_SOURCES = {
    COMPONENT_IDS["branch_selection"]: ("pratten2020v1", "52", "71", "85", "90", "96", "carrier-convention"),
    COMPONENT_IDS["physical_scale"]: ("pratten2020v1", "14-16", "66-67", "units", "dimensional-restoration"),
    COMPONENT_IDS["phase_adapter"]: ("pratten2020v1", "14-16", "carrier-convention"),
    COMPONENT_IDS["polarizations"]: ("pratten2020v1", "14-16", "carrier-convention"),
    COMPONENT_IDS["nonpositive_mask"]: ("carrier-convention", "f<=0"),
    COMPONENT_IDS["extrinsic_derivatives"]: ("derived-product-rule", "polarizations", "phase_adapter"),
    COMPONENT_IDS["frequency_derivatives"]: ("derived-product-rule", "branch_selection", "physical_scale", "phase_adapter"),
    COMPONENT_IDS["mass_scaling"]: ("pratten2020v1", "1", "14-16", "66-67", "dimensional-similarity", "carrier-convention"),
    COMPONENT_IDS["mass_derivatives"]: ("derived-chain-rule", "mass_scaling", "branch_selection", "phase_adapter"),
}

U = sp.symbols("u", positive=True)


def exact(value):
    """Keep supplied decimal calibration outputs distinct from copied fit rows."""
    return sp.Rational(str(value)) if not isinstance(value, sp.Basic) else value


def branch(u, lower, upper):
    """Both joins belong to the branch on their low-frequency side."""
    return "inspiral" if u <= lower else "intermediate" if u <= upper else "merger_ringdown"


def physical_scale(f_hz, m_sec, distance_mpc, eta):
    d_sec = exact(distance_mpc) * units.MPC_SI / units.C_SI
    return units.physical_scale(f_hz, m_sec, d_sec, eta)


def phase_adapter(raw, reference, f_hz, f_ref_hz, tshift_sec, tc, phic):
    """Carrier phase: optional reference origin, time shift, and 3pi/4 offset."""
    phase_ref = 0 if reference is None else reference
    shift_ref = 0 if reference is None else f_ref_hz
    return (raw - phase_ref + tshift_sec * (f_hz - shift_ref)
            + 2 * sp.pi * f_hz * tc - 2 * phic + 3 * sp.pi / 4)


def polarizations(amplitude, psi, inclination):
    cosine = sp.cos(inclination)
    carrier = amplitude * sp.exp(-sp.I * psi)
    return (sp.Rational(1, 2) * (1 + cosine**2) * carrier,
            -sp.I * cosine * carrier)


def nonpositive_mask(f_hz):
    return exact(f_hz) <= 0


def extrinsic_derivatives(hp, hc, f_hz, distance_mpc, parameter,
                          *, inclination=0):
    """Closed partials at fixed intrinsic fit inputs and positive frequency."""
    if parameter == "distance_mpc":
        return -hp / distance_mpc, -hc / distance_mpc
    if parameter == "tc":
        return -2 * sp.pi * sp.I * f_hz * hp, -2 * sp.pi * sp.I * f_hz * hc
    if parameter == "phic":
        return 2 * sp.I * hp, 2 * sp.I * hc
    if parameter == "inclination":
        cosine, sine = sp.cos(inclination), sp.sin(inclination)
        return (-2 * cosine * sine / (1 + cosine**2) * hp,
                2 * sp.I * sine / (1 + cosine**2) * hp)
    raise ValueError(f"unknown extrinsic derivative: {parameter}")


@dataclass(frozen=True)
class WaveformSpec:
    m_sec: sp.Expr
    eta: sp.Expr
    distance_mpc: sp.Expr
    amplitude: Mapping[str, sp.Expr]
    amp_joins: tuple[sp.Expr, sp.Expr]
    phase: Mapping[str, sp.Expr]
    phase_joins: tuple[sp.Expr, sp.Expr]
    tshift_sec: sp.Expr

    def scaled_total_mass(self, factor):
        """Scale both component masses at fixed q, spins and dimensionless fits."""
        factor = exact(factor)
        if factor <= 0:
            raise ValueError("total-mass scale must be positive")
        return replace(self, m_sec=factor * self.m_sec,
                       tshift_sec=factor * self.tshift_sec)

    def expressions(self, frequency_hz, *, inclination=0, tc=0, phic=0,
                    f_ref_hz=None):
        """Return h+, hx and unwrapped phase at one positive frequency."""
        f = exact(frequency_hz)
        if nonpositive_mask(f):
            return sp.S.Zero, sp.S.Zero, sp.S.Zero
        u = self.m_sec * f
        a = self.amplitude[branch(u, *self.amp_joins)].subs(U, u)
        raw = self.phase[branch(u, *self.phase_joins)].subs(U, u)
        reference = None
        if f_ref_hz is not None:
            f_ref_hz = exact(f_ref_hz)
            if f_ref_hz <= 0:
                raise ValueError("f_ref_hz must be positive")
            u_ref = self.m_sec * f_ref_hz
            reference = self.phase[branch(u_ref, *self.phase_joins)].subs(U, u_ref)
        psi = phase_adapter(raw, reference, f, f_ref_hz, self.tshift_sec,
                            exact(tc), exact(phic))
        hp, hc = polarizations(
            physical_scale(f, self.m_sec, self.distance_mpc, self.eta) * a,
            psi, exact(inclination))
        return hp, hc, psi

    def evaluate(self, frequency_hz, *, dps=45, **kwargs):
        return tuple(complex(sp.N(x, dps)) if i < 2 else float(sp.N(x, dps))
                     for i, x in enumerate(self.expressions(frequency_hz, **kwargs)))

    def frequency_derivatives(self, frequency_hz, *, inclination=0, tc=0,
                              phic=0, f_ref_hz=None):
        """Smooth-branch d(h+,hx)/df_Hz at fixed fits and fixed f_ref."""
        f = exact(frequency_hz)
        if f <= 0:
            raise ValueError("frequency derivative requires positive frequency")
        u = self.m_sec * f
        if u in self.amp_joins or u in self.phase_joins:
            raise ValueError("frequency derivative is undefined at a branch join")
        a_expr = self.amplitude[branch(u, *self.amp_joins)]
        p_expr = self.phase[branch(u, *self.phase_joins)]
        a = a_expr.subs(U, u)
        a_prime = sp.diff(a_expr, U).subs(U, u)
        p_prime = sp.diff(p_expr, U).subs(U, u)
        scale = physical_scale(f, self.m_sec, self.distance_mpc, self.eta)
        amplitude = scale * a
        amplitude_prime = scale * (self.m_sec * a_prime - sp.Rational(7, 6)*a/f)
        _, _, psi = self.expressions(f, inclination=inclination, tc=tc,
                                     phic=phic, f_ref_hz=f_ref_hz)
        psi_prime = self.m_sec*p_prime + self.tshift_sec + 2*sp.pi*exact(tc)
        carrier_prime = sp.exp(-sp.I*psi) * (
            amplitude_prime - sp.I*amplitude*psi_prime)
        cosine = sp.cos(exact(inclination))
        return (sp.Rational(1, 2)*(1+cosine**2)*carrier_prime,
                -sp.I*cosine*carrier_prime)

    def log_total_mass_derivatives(self, frequency_hz, *, inclination=0,
                                   tc=0, phic=0, f_ref_hz=None):
        """Return d(h+,hx,Psi)/d ln M at fixed q, spins, Hz inputs and fits.

        The physical coalescence time and reference frequency remain fixed.
        The supplied dimensionless fits and joins are mass invariant; the
        carrier's dimensional time shift scales linearly with total mass.
        Only smooth positive-frequency branch interiors are supported.
        """
        f = exact(frequency_hz)
        if f <= 0:
            raise ValueError("mass derivative requires positive frequency")
        u = self.m_sec * f
        if u in self.amp_joins or u in self.phase_joins:
            raise ValueError("mass derivative is undefined at a branch join")
        a_expr = self.amplitude[branch(u, *self.amp_joins)]
        p_expr = self.phase[branch(u, *self.phase_joins)]
        a = a_expr.subs(U, u)
        a_prime = sp.diff(a_expr, U).subs(U, u)
        p_prime = sp.diff(p_expr, U).subs(U, u)
        reference_tangent = sp.S.Zero
        shift_ref = sp.S.Zero
        if f_ref_hz is not None:
            f_ref = exact(f_ref_hz)
            if f_ref <= 0:
                raise ValueError("f_ref_hz must be positive")
            u_ref = self.m_sec * f_ref
            if u_ref in self.phase_joins:
                raise ValueError("mass derivative is undefined at a reference join")
            p_ref_expr = self.phase[branch(u_ref, *self.phase_joins)]
            reference_tangent = u_ref * sp.diff(p_ref_expr, U).subs(U, u_ref)
            shift_ref = f_ref
        phase_tangent = (u * p_prime - reference_tangent
                         + self.tshift_sec * (f - shift_ref))
        scale = physical_scale(f, self.m_sec, self.distance_mpc, self.eta)
        amplitude_tangent = scale * (sp.Rational(5, 6) * a + u * a_prime)
        _, _, psi = self.expressions(f, inclination=inclination, tc=tc,
                                     phic=phic, f_ref_hz=f_ref_hz)
        carrier_tangent = sp.exp(-sp.I * psi) * (
            amplitude_tangent - sp.I * scale * a * phase_tangent)
        cosine = sp.cos(exact(inclination))
        return (sp.Rational(1, 2) * (1 + cosine**2) * carrier_tangent,
                -sp.I * cosine * carrier_tangent, phase_tangent)


def compose(m1, m2, chi1, chi2, distance_mpc, parameters: Mapping[str, object]):
    """Compose from independently supplied calibrated scalars and QNM inputs.

    ``parameters`` contains the named output of the carrier's fit/collocation
    stage. It is deliberately supplied by the caller, not imported here.
    """
    m1, m2, chi1, chi2, distance = map(exact, (m1, m2, chi1, chi2, distance_mpc))
    if min(m1, m2, distance) <= 0:
        raise ValueError("masses and distance must be positive")
    if m1 < m2:
        m1, m2, chi1, chi2 = m2, m1, chi2, chi1
    p = {name: exact(value) for name, value in parameters.items()}
    eta = m1 * m2 / (m1 + m2)**2
    delta = (m1 - m2) / (m1 + m2)
    m_sec = (m1 + m2) * units.MSUN_SEC
    a_ins = amp.inspiral_shape(U, eta, chi1, chi2, delta,
                               tuple(p[f"rho{i}"] for i in range(1, 4)))
    polynomial = middle.inverse_polynomial(
        U, p["f1_col"], p["f4_col"],
        tuple(p[f"c{i}_amp"] for i in range(5)))
    a_mid = U**sp.Rational(7, 6) / polynomial
    a_rd = amp.ringdown_amplitude(
        U, p["f_rd"], p["f_damp"], p["width_rd"] / p["f_damp"],
        p["lam"], p["a_r"])
    chi_s, chi_a = (chi1 + chi2)/2, (chi1 - chi2)/2
    ph_ins = inspiral.inspiral_phase(
        U, eta, delta, chi_s, chi_a,
        tuple(p[f"sig{i}"] for i in range(5)))
    ph_mid = phase.intermediate_primitive(
        U, eta, p["f_rd"], p["f_damp"],
        tuple(p[f"b{i}_val"] for i in range(5)), p["c0_val"],
        a_phi=1, carrier_sign=True)
    ph_rd = phase.ringdown_primitive(
        U, eta, p["f_rd"], p["f_damp"],
        (p["c_rd_val"], p["c1_val"], p["c2_val"], p["c4_val"], p["c0_val"]),
        a_phi=1)
    ph_rd += p["dt_rd"] * U + p["dphi_rd"]
    return WaveformSpec(
        m_sec, eta, distance,
        {"inspiral": a_ins, "intermediate": a_mid, "merger_ringdown": a_rd},
        (p["f_trans_amp"], p["f_peak"]),
        {"inspiral": ph_ins, "intermediate": ph_mid, "merger_ringdown": ph_rd},
        (p["f_trans_phase"], p["f_join_rd"]), p["tshift_sec"])
