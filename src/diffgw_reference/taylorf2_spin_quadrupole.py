"""Published spin-induced quadrupole slice of aligned TaylorF2, through 3.5PN.

This independent SymPy specification composes the previously versioned
nonspinning, aligned-BH-spin, and adiabatic-tidal references without importing
Torch/JAX implementations.  C_Qi=1 is the Kerr quadrupole; the additions below
are only the *difference* from that baseline.  The signed mass coordinate is
delta=(m1-m2)/M, Xi=mi/M, eta=X1*X2, and ai=Xi*chi_i.

Nagar et al. (2018), Eqs. (2), (38), (44), (46), (47), (50)-(52), supply the
quadratic-in-spin corrections at 2PN, 3PN, and the 3.5PN tail. Dietrich et al.
(2019), Eq. (27), supplies the C_Q-dependent cubic-in-spin term at 3.5PN. These
multiply v**4, v**6, v**7 inside 3/(128*eta*v**5). Spin-induced octupoles
are independent physical inputs and remain at their inherited Kerr value;
using this slice at spin_order=8 does not make a complete neutron-star model.
No EOS relation between C_Qi and tidal Lambda_i is imposed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import sympy as sp

from . import taylorf2_tidal as tidal

aligned, base = tidal.aligned, tidal.base
SPEC_VERSION = "taylorf2-spin-induced-quadrupole-3.5pn/v1"
REFERENCE_VERSION = SPEC_VERSION
INHERITED_SPEC_VERSIONS = {
    "nonspinning": base.SPEC_VERSION,
    "aligned_spin": aligned.SPEC_VERSION,
    "tidal": tidal.SPEC_VERSION,
}
INHERITED_COMPONENT_IDS = {
    "nonspinning": dict(base.COMPONENT_IDS),
    "aligned_spin": dict(aligned.COMPONENT_IDS),
    "tidal": dict(tidal.COMPONENT_IDS),
}

V, F, M_SEC, D_SEC, F_REF = tidal.V, tidal.F, tidal.M_SEC, tidal.D_SEC, tidal.F_REF
IOTA, TC, PHIC = tidal.IOTA, tidal.TC, tidal.PHIC
DELTA, CHI1, CHI2 = tidal.DELTA, tidal.CHI1, tidal.CHI2
LAMBDA1, LAMBDA2 = tidal.LAMBDA1, tidal.LAMBDA2
CQ1, CQ2 = sp.symbols("C_Q1 C_Q2", real=True)
SYMBOLS = {**tidal.SYMBOLS, CQ1.name: CQ1, CQ2.name: CQ2,
           "quad_monopole1": CQ1, "quad_monopole2": CQ2}
QUAD_TERMS = ("Q4", "Q6", "Q7_tail", "Q7_cubic")
SOURCES = {
    **tidal.SOURCES,
    "nagar2018": {
        "arxiv": "1812.07923v1",
        "url": "https://arxiv.org/pdf/1812.07923v1",
        "doi": "10.1103/PhysRevD.99.044007",
        "equations": ("2", "38", "44", "46", "47", "50", "51", "52"),
        "use": "Spin-induced quadrupole phase at 2PN, 3PN and 3.5PN tail; PDF pp. 2, 9-10.",
    },
    "dietrich2019": {
        "arxiv": "1905.06011v2",
        "url": "https://arxiv.org/pdf/1905.06011v2",
        "doi": "10.1103/PhysRevD.100.044003",
        "equations": ("26", "27"),
        "use": "Black-hole-subtracted C_Q-dependent cubic-spin phase at 3.5PN; PDF p. 9.",
    },
}
COMPONENT_IDS = {name: f"tf2.spinq.{name}.v1" for name in
                 (*QUAD_TERMS, "intrinsic_phase", "coalescence_phase",
                  "reference_phase", "polarizations")}
COMPONENT_SOURCES = {
    COMPONENT_IDS["Q4"]: ("nagar2018", "2", "38", "44", "50"),
    COMPONENT_IDS["Q6"]: ("nagar2018", "2", "38", "46", "51"),
    COMPONENT_IDS["Q7_tail"]: ("nagar2018", "2", "38", "47", "52"),
    COMPONENT_IDS["Q7_cubic"]: ("dietrich2019", "26", "27"),
    **{COMPONENT_IDS[name]: ("project-convention", SPEC_VERSION)
       for name in ("intrinsic_phase", "coalescence_phase", "reference_phase", "polarizations")},
}
DERIVATIONS = {
    "coordinates": "ai=Xi*chi_i, Xi=(1+/-delta)/2, eta=(1-delta**2)/4; qi=(C_Qi-1)*ai**2.",
    "normalization": "Qk multiplies v**k inside 3/(128*eta*v**5), not inside a second 1/eta.",
    "baseline": "Subtract C_Q1=C_Q2=1 from Nagar's full quadrupolar equations; all other terms are inherited unchanged.",
    "tail": "Nagar Eq. (52) gives -75*pi/(8*eta)*sum(C_Qi*ai**2)*v**2; divided by 3/(128*eta)*v**2 gives -400*pi*sum(C_Qi*ai**2).",
    "cubic": "Dietrich Eq. (27) subtracts the C_Q=1 baseline and retains independent C_Oc=1; its C_Q-dependent term is 10*sum(qi*((Xi**2+308*Xi/3)*chi_i+(Xj**2-89*Xj/3)*chi_j)).",
}


def quadrupole_corrections(delta: sp.Expr = DELTA, chi1: sp.Expr = CHI1,
                           chi2: sp.Expr = CHI2, cq1: sp.Expr = CQ1,
                           cq2: sp.Expr = CQ2) -> dict[str, sp.Expr]:
    """Exact C_Q-1 coefficients in the standard TaylorF2 phase bracket."""
    delta, chi1, chi2, cq1, cq2 = map(sp.sympify, (delta, chi1, chi2, cq1, cq2))
    x1, x2 = (1 + delta) / 2, (1 - delta) / 2
    eta = (1 - delta**2) / 4
    q1, q2 = (cq1 - 1) * (x1 * chi1)**2, (cq2 - 1) * (x2 * chi2)**2
    cubic1 = ((x1**2 + sp.Rational(308, 3) * x1) * chi1
              + (x2**2 - sp.Rational(89, 3) * x2) * chi2)
    cubic2 = ((x2**2 + sp.Rational(308, 3) * x2) * chi2
              + (x1**2 - sp.Rational(89, 3) * x1) * chi1)
    return {
        "Q4": -50 * (q1 + q2),
        "Q6": (sp.Rational(15635, 21) + 120 * eta) * (q1 + q2)
              + sp.Rational(2215, 12) * delta * (q1 - q2),
        "Q7_tail": -400 * sp.pi * (q1 + q2),
        "Q7_cubic": 10 * (q1 * cubic1 + q2 * cubic2),
    }


def _selected_terms(terms: Iterable[str]) -> tuple[str, ...]:
    if isinstance(terms, str):
        raise TypeError("quad_terms must be an iterable, not a string")
    selected = tuple(terms)
    if any(name not in QUAD_TERMS for name in selected) or len(set(selected)) != len(selected):
        raise ValueError(f"quad_terms must contain unique names from {QUAD_TERMS}")
    return tuple(name for name in QUAD_TERMS if name in selected)


def frontend_terms(spin_order: int, phenom_d_calibration: bool = False) -> tuple[str, ...]:
    """Map the legacy frontend's historical spin selectors to this slice."""
    if isinstance(spin_order, bool) or not isinstance(spin_order, int):
        raise TypeError("spin_order must be an integer")
    if not isinstance(phenom_d_calibration, bool):
        raise TypeError("phenom_d_calibration must be boolean")
    selected = []
    if spin_order >= 4 or spin_order < 0:
        selected.append("Q4")
    if not phenom_d_calibration and (spin_order >= 6 or spin_order < 0):
        selected.append("Q6")
    if not phenom_d_calibration and spin_order >= 8:
        selected.extend(("Q7_tail", "Q7_cubic"))
    return tuple(selected)


def coefficients(delta: sp.Expr = DELTA, chi1: sp.Expr = CHI1,
                 chi2: sp.Expr = CHI2, lambda1: sp.Expr = LAMBDA1,
                 lambda2: sp.Expr = LAMBDA2, cq1: sp.Expr = CQ1,
                 cq2: sp.Expr = CQ2, *,
                 spin_terms: Iterable[str] = aligned.SPIN_TERMS,
                 tidal_terms: Iterable[str] = tidal.TIDAL_TERMS,
                 quad_terms: Iterable[str] = QUAD_TERMS) -> dict[str, sp.Expr]:
    """Composed phase coefficients, adding only selected C_Q-1 corrections."""
    selected = _selected_terms(quad_terms)
    result = tidal.coefficients(delta, chi1, chi2, lambda1, lambda2,
                                spin_terms=spin_terms, tidal_terms=tidal_terms)
    corrections = quadrupole_corrections(delta, chi1, chi2, cq1, cq2)
    for term in selected:
        result[f"phi{term[1]}"] += corrections[term]
    return result


def quadrupole_phase_expr(v: sp.Expr = V, delta: sp.Expr = DELTA,
                          chi1: sp.Expr = CHI1, chi2: sp.Expr = CHI2,
                          cq1: sp.Expr = CQ1, cq2: sp.Expr = CQ2, *,
                          quad_terms: Iterable[str] = QUAD_TERMS) -> sp.Expr:
    selected = _selected_terms(quad_terms)
    v, delta = sp.sympify(v), sp.sympify(delta)
    corrections = quadrupole_corrections(delta, chi1, chi2, cq1, cq2)
    bracket = sum((corrections[name] * v**int(name[1]) for name in selected), sp.S.Zero)
    return sp.Rational(3, 128) * bracket / (((1 - delta**2) / 4) * v**5)


def phase_expr(v: sp.Expr = V, delta: sp.Expr = DELTA,
               chi1: sp.Expr = CHI1, chi2: sp.Expr = CHI2,
               lambda1: sp.Expr = LAMBDA1, lambda2: sp.Expr = LAMBDA2,
               cq1: sp.Expr = CQ1, cq2: sp.Expr = CQ2, *,
               spin_terms: Iterable[str] = aligned.SPIN_TERMS,
               tidal_terms: Iterable[str] = tidal.TIDAL_TERMS,
               quad_terms: Iterable[str] = QUAD_TERMS) -> sp.Expr:
    return (tidal.phase_expr(v, delta, chi1, chi2, lambda1, lambda2,
                             spin_terms=spin_terms, tidal_terms=tidal_terms)
            + quadrupole_phase_expr(v, delta, chi1, chi2, cq1, cq2,
                                    quad_terms=quad_terms))


def amplitude_expr() -> sp.Expr:
    return tidal.amplitude_expr()


def waveform_expr(*, reference_frequency: bool = False,
                  spin_terms: Iterable[str] = aligned.SPIN_TERMS,
                  tidal_terms: Iterable[str] = tidal.TIDAL_TERMS,
                  quad_terms: Iterable[str] = QUAD_TERMS) -> dict[str, sp.Expr]:
    selected = _selected_terms(quad_terms)
    inherited = tidal.waveform_expr(reference_frequency=reference_frequency,
                                    spin_terms=spin_terms, tidal_terms=tidal_terms)
    correction = quadrupole_phase_expr((sp.pi * M_SEC * F)**sp.Rational(1, 3),
                                       quad_terms=selected)
    if reference_frequency:
        correction -= quadrupole_phase_expr((sp.pi * M_SEC * F_REF)**sp.Rational(1, 3),
                                            quad_terms=selected)
    phase = inherited["phase"] + correction
    amplitude = inherited["amplitude"]
    carrier = amplitude * sp.exp(-sp.I * phase)
    cosine = sp.cos(IOTA)
    return {"amplitude": amplitude, "phase": phase,
            "h_plus": (1 + cosine**2) * carrier / 2,
            "h_cross": -sp.I * cosine * carrier}


def derivative_expr(expression: sp.Expr, variable: sp.Symbol | str,
                    order: int = 1) -> sp.Expr:
    if isinstance(variable, str):
        variable = SYMBOLS[variable]
    return tidal.derivative_expr(expression, variable, order)


def evaluate_phase(*, frequency, mass_seconds, delta, chi1, chi2,
                   lambda1, lambda2, cq1, cq2, tc="0", phic="0",
                   reference_frequency=None, spin_terms: Iterable[str] = aligned.SPIN_TERMS,
                   tidal_terms: Iterable[str] = tidal.TIDAL_TERMS,
                   quad_terms: Iterable[str] = QUAD_TERMS, dps: int = 80):
    """Evaluate the public phase gauge in high precision with exact-decimal inputs."""
    values = {F: frequency, M_SEC: mass_seconds, DELTA: delta, CHI1: chi1,
              CHI2: chi2, LAMBDA1: lambda1, LAMBDA2: lambda2,
              CQ1: cq1, CQ2: cq2, TC: tc, PHIC: phic}
    if reference_frequency is not None:
        values[F_REF] = reference_frequency
    expression = waveform_expr(reference_frequency=reference_frequency is not None,
                               spin_terms=spin_terms, tidal_terms=tidal_terms,
                               quad_terms=quad_terms)["phase"]
    return evaluate(expression, values, dps=dps)


def evaluate(expression: sp.Expr, values: Mapping[sp.Symbol | str, object], *,
             dps: int = 80):
    bound = {SYMBOLS[key] if isinstance(key, str) else key: value
             for key, value in values.items()}
    return tidal.evaluate(expression, bound, dps=dps)


def evaluate_coefficients(delta, chi1, chi2, lambda1, lambda2, cq1, cq2, *,
                          spin_terms: Iterable[str] = aligned.SPIN_TERMS,
                          tidal_terms: Iterable[str] = tidal.TIDAL_TERMS,
                          quad_terms: Iterable[str] = QUAD_TERMS, dps: int = 80) -> dict:
    values = {DELTA: delta, CHI1: chi1, CHI2: chi2, LAMBDA1: lambda1,
              LAMBDA2: lambda2, CQ1: cq1, CQ2: cq2}
    return {name: evaluate(expr, values, dps=dps) for name, expr in
            coefficients(spin_terms=spin_terms, tidal_terms=tidal_terms,
                         quad_terms=quad_terms).items()}


def evaluate_waveform(*, frequency, mass_seconds, delta, chi1, chi2,
                      lambda1, lambda2, cq1, cq2, distance_seconds,
                      inclination="0", tc="0", phic="0", reference_frequency=None,
                      spin_terms: Iterable[str] = aligned.SPIN_TERMS,
                      tidal_terms: Iterable[str] = tidal.TIDAL_TERMS,
                      quad_terms: Iterable[str] = QUAD_TERMS, dps: int = 80) -> dict:
    values = {F: frequency, M_SEC: mass_seconds, DELTA: delta, CHI1: chi1,
              CHI2: chi2, LAMBDA1: lambda1, LAMBDA2: lambda2, CQ1: cq1,
              CQ2: cq2, D_SEC: distance_seconds, IOTA: inclination,
              TC: tc, PHIC: phic}
    if reference_frequency is not None:
        values[F_REF] = reference_frequency
    expressions = waveform_expr(reference_frequency=reference_frequency is not None,
                                spin_terms=spin_terms, tidal_terms=tidal_terms,
                                quad_terms=quad_terms)
    return {name: evaluate(expr, values, dps=dps) for name, expr in expressions.items()}
