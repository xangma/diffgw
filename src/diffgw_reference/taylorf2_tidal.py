"""Adiabatic quadrupolar gravitoelectric TaylorF2 tides through 6.5PN.

This independent, separately versioned phase specification composes the
unchanged nonspinning and aligned-spin references. The dimensionless inputs
lambda1/lambda2 are the usual Lambda_i = G*lambda_i*(c**2/(G*m_i))**5,
not the dimensionful polarizabilities denoted lambda_i in Wade et al. The
signed mass coordinate delta=(m1-m2)/(m1+m2) gives Xi=(1+/-delta)/2 and
eta=(1-delta**2)/4. No equation-of-state relation between inputs is imposed.

Wade et al. Eq. (A26) supplies 5PN/6PN coefficients, while Eqs. (4)--(6)
supply an independently expressed effective-parameter identity. Their
sqrt(1-4*eta), under the stated m1>m2 convention, becomes signed delta;
the extension to either labelling follows exactly from the component-mass
polynomials. delta_lambda_tilde is odd under label exchange; lambda_tilde
and the physical phase are even. These coordinates are smooth at equal mass.

Damour, Nagar and Villain Eqs. (B8)--(B12) give the relative 1.5PN tidal
tail factor -pi*x**(3/2). Their Eq. (6), with ell=2, gives
kappa_i=3*(1-Xi)*Xi**4*Lambda_i. Substitution reproduces Wade's leading
phase and yields T13=-pi*T10. Here x=v**2 and every T_k multiplies v**k
inside the standard second-harmonic factor 3/(128*eta*v**5).

Only phase coefficients are taken from these papers. Wade Eq. (A23) prints
exp(+i*psi), whereas Damour et al. Eq. (15) uses exp(-i*Psi). The waveform
here retains the base specification's exp(-i*Psi), polarizations, restricted
Newtonian amplitude and both explicitly documented phase gauges. Finite
reference-frequency subtraction includes the tidal phase as well.

The tidal correction is linear in the adiabatic electric quadrupolar
polarizabilities: 5PN, 6PN, and their 6.5PN tail. Spin and tide phase sectors
are added, without spin-tidal couplings. Inherited spin-induced multipoles
retain black-hole values; this is not a complete spinning neutron-star
model. Magnetic, octupolar, dynamical and nonlinear tides, internal tidal dissipation,
tidal amplitude corrections, merger, termination and tapering are excluded.
The analytic domain does not guarantee PN accuracy at large frequency or
polarizability. Partial derivatives hold all other canonical inputs fixed.
No production or external waveform implementation supplies coefficients.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import sympy as sp

from . import taylorf2_aligned_spin as aligned

base = aligned.base
SPEC_VERSION = "taylorf2-adiabatic-quadrupolar-tides-6.5pn/v1"
REFERENCE_VERSION = SPEC_VERSION
INHERITED_SPEC_VERSIONS = {
    "nonspinning": base.SPEC_VERSION,
    "aligned_spin": aligned.SPEC_VERSION,
}
INHERITED_COMPONENT_IDS = {
    "nonspinning": dict(base.COMPONENT_IDS),
    "aligned_spin": dict(aligned.COMPONENT_IDS),
}

V, F, M_SEC, D_SEC, F_REF = aligned.V, aligned.F, aligned.M_SEC, aligned.D_SEC, aligned.F_REF
IOTA, TC, PHIC = aligned.IOTA, aligned.TC, aligned.PHIC
DELTA, CHI1, CHI2 = aligned.DELTA, aligned.CHI1, aligned.CHI2
LAMBDA1, LAMBDA2 = sp.symbols("lambda1 lambda2", real=True)
ETA_EXPR = aligned.ETA_EXPR
X1_EXPR, X2_EXPR = (1 + DELTA) / 2, (1 - DELTA) / 2
SYMBOLS = {**aligned.SYMBOLS, LAMBDA1.name: LAMBDA1, LAMBDA2.name: LAMBDA2}
TIDAL_TERMS = ("T10", "T12", "T13")

SOURCES = {
    **aligned.SOURCES,
    "wade2014": {
        "arxiv": "1402.5156v1",
        "url": "https://arxiv.org/pdf/1402.5156v1",
        "doi": "10.1103/PhysRevD.89.103012",
        "equations": ("1", "4", "5", "6", "A23", "A26", "A27"),
        "use": "Direct 5PN/6PN phase (PDF p. 13); effective parameters (pp. 3-4).",
    },
    "damour2012": {
        "arxiv": "1203.4352v1",
        "url": "https://arxiv.org/pdf/1203.4352v1",
        "doi": "10.1103/PhysRevD.85.123007",
        "equations": ("6", "15", "16", "B8", "B9", "B10", "B11", "B12", "B14"),
        "use": "General-mass tidal phase and 6.5PN tail, PDF p. 26; polarizability conversion Eq. (6).",
    },
}
COMPONENT_IDS = {
    name: f"tf2.tidal65.{name}.v1"
    for name in (*TIDAL_TERMS, "lambda_tilde", "delta_lambda_tilde", "intrinsic_phase",
                 "coalescence_phase", "reference_phase", "polarizations")
}
COMPONENT_SOURCES = {
    COMPONENT_IDS["T10"]: ("wade2014", "A26"),
    COMPONENT_IDS["T12"]: ("wade2014", "A26"),
    COMPONENT_IDS["T13"]: ("damour2012", "6", "B8", "B9", "B10", "B12", "B14"),
    COMPONENT_IDS["lambda_tilde"]: ("wade2014", "5"),
    COMPONENT_IDS["delta_lambda_tilde"]: ("wade2014", "6"),
    **{COMPONENT_IDS[name]: ("project-convention", SPEC_VERSION)
       for name in ("intrinsic_phase", "coalescence_phase", "reference_phase", "polarizations")},
}
DERIVATIONS = {
    "component_to_effective_parameters": {
        "sources": (("wade2014", "4", "5", "6", "A26"),),
        "expressions": ("tidal_corrections", "effective_tidal_corrections"),
        "coordinates": "Xi=(1+/-delta)/2, eta=(1-delta**2)/4; replace the ordered-mass square root by signed delta.",
        "units": "Wade dimensionful lambda_i/M**5 = Xi**5*Lambda_i in G=c=1 units.",
    },
    COMPONENT_IDS["T13"]: {
        "sources": (("damour2012", "6", "B8", "B9", "B10", "B11", "B12", "B14"),),
        "normalization": "kappa_i=3*(1-Xi)*Xi**4*Lambda_i, x=v**2; divide phase by 3/(128*eta*v**5).",
        "tail": "p2_i=-pi relative to the leading per-body tide, so T13=-pi*T10.",
    },
}


def _parameters(delta, lambda1, lambda2):
    delta, lambda1, lambda2 = map(sp.sympify, (delta, lambda1, lambda2))
    return delta, lambda1, lambda2, (1 - delta**2) / 4, (1 + delta) / 2, (1 - delta) / 2


def tidal_parameters(delta: sp.Expr = DELTA, lambda1: sp.Expr = LAMBDA1,
                     lambda2: sp.Expr = LAMBDA2) -> dict[str, sp.Expr]:
    """Exact effective tidal parameters, smoothly extended to signed delta.

    The inputs are dimensionless individual deformabilities. These effective
    parameters are a change of coordinates, without an EOS assumption.
    """
    delta, lambda1, lambda2, eta, _, _ = _parameters(delta, lambda1, lambda2)
    r = sp.Rational
    symmetric, antisymmetric = lambda1 + lambda2, lambda1 - lambda2
    return {
        "lambda_tilde": r(8, 13) * (
            (1 + 7 * eta - 31 * eta**2) * symmetric
            + delta * (1 + 9 * eta - 11 * eta**2) * antisymmetric),
        "delta_lambda_tilde": r(1, 2) * (
            delta * (1 - r(13272, 1319) * eta + r(8944, 1319) * eta**2) * symmetric
            + (1 - r(15910, 1319) * eta + r(32850, 1319) * eta**2
               + r(3380, 1319) * eta**3) * antisymmetric),
    }


def tidal_corrections(delta: sp.Expr = DELTA, lambda1: sp.Expr = LAMBDA1,
                      lambda2: sp.Expr = LAMBDA2) -> dict[str, sp.Expr]:
    """Direct component-weighted coefficients inside 3/(128*eta*v**5).

    T10/T12/T13 multiply v**10/v**12/v**13 respectively. The tail is
    constructed from the general-mass Damour et al. relative coefficient.
    """
    _, lambda1, lambda2, _, x1, x2 = _parameters(delta, lambda1, lambda2)
    bodies = ((x1, lambda1), (x2, lambda2))
    t10 = -24 * sum(lam * x**4 * (12 - 11 * x) for x, lam in bodies)
    t12 = -sp.Rational(5, 28) * sum(
        lam * x**4 * (3179 - 919 * x - 2286 * x**2 + 260 * x**3)
        for x, lam in bodies)
    return {"T10": t10, "T12": t12, "T13": -sp.pi * t10}


def effective_tidal_corrections(delta: sp.Expr = DELTA, lambda1: sp.Expr = LAMBDA1,
                                lambda2: sp.Expr = LAMBDA2) -> dict[str, sp.Expr]:
    """Alternative effective-parameter form for exact identity checks."""
    delta = sp.sympify(delta)
    parameters = tidal_parameters(delta, lambda1, lambda2)
    lam, dlam = parameters["lambda_tilde"], parameters["delta_lambda_tilde"]
    return {
        "T10": -sp.Rational(39, 2) * lam,
        "T12": -sp.Rational(3115, 64) * lam + sp.Rational(6595, 364) * delta * dlam,
        "T13": sp.Rational(39, 2) * sp.pi * lam,
    }


def _selected_terms(terms: Iterable[str]) -> tuple[str, ...]:
    if isinstance(terms, str):
        raise TypeError("tidal_terms must be an iterable of physical term names, not a string")
    selected = tuple(terms)
    if any(name not in TIDAL_TERMS for name in selected):
        raise ValueError(f"tidal_terms must contain only {TIDAL_TERMS}")
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate physical tidal terms are not allowed")
    return tuple(name for name in TIDAL_TERMS if name in selected)


def coefficients(delta: sp.Expr = DELTA, chi1: sp.Expr = CHI1,
                 chi2: sp.Expr = CHI2, lambda1: sp.Expr = LAMBDA1,
                 lambda2: sp.Expr = LAMBDA2, *,
                 spin_terms: Iterable[str] = aligned.SPIN_TERMS,
                 tidal_terms: Iterable[str] = TIDAL_TERMS) -> dict[str, sp.Expr]:
    """Inherited coefficients plus phi10/phi12/phi13; unselected tides are zero."""
    selected = _selected_terms(tidal_terms)
    result = aligned.coefficients(delta, chi1, chi2, terms=spin_terms)
    corrections = tidal_corrections(delta, lambda1, lambda2)
    for name in TIDAL_TERMS:
        result[f"phi{name[1:]}"] = corrections[name] if name in selected else sp.S.Zero
    return result


def tidal_phase_expr(v: sp.Expr = V, delta: sp.Expr = DELTA,
                     lambda1: sp.Expr = LAMBDA1, lambda2: sp.Expr = LAMBDA2, *,
                     tidal_terms: Iterable[str] = TIDAL_TERMS) -> sp.Expr:
    """Additive intrinsic tidal phase for the selected physical terms."""
    selected = _selected_terms(tidal_terms)
    v, delta = sp.sympify(v), sp.sympify(delta)
    eta = (1 - delta**2) / 4
    corrections = tidal_corrections(delta, lambda1, lambda2)
    polynomial = sum((corrections[name] * v**int(name[1:]) for name in selected), sp.S.Zero)
    return sp.Rational(3, 128) * polynomial / (eta * v**5)


def phase_expr(v: sp.Expr = V, delta: sp.Expr = DELTA,
               chi1: sp.Expr = CHI1, chi2: sp.Expr = CHI2,
               lambda1: sp.Expr = LAMBDA1, lambda2: sp.Expr = LAMBDA2, *,
               spin_terms: Iterable[str] = aligned.SPIN_TERMS,
               tidal_terms: Iterable[str] = TIDAL_TERMS) -> sp.Expr:
    """Compose the unchanged aligned-spin phase with additive tidal terms."""
    return (aligned.phase_expr(v, delta, chi1, chi2, terms=spin_terms)
            + tidal_phase_expr(v, delta, lambda1, lambda2, tidal_terms=tidal_terms))


def amplitude_expr() -> sp.Expr:
    """Unchanged restricted Newtonian amplitude; no finite-size amplitude terms."""
    return aligned.amplitude_expr()


def waveform_expr(*, reference_frequency: bool = False,
                  spin_terms: Iterable[str] = aligned.SPIN_TERMS,
                  tidal_terms: Iterable[str] = TIDAL_TERMS) -> dict[str, sp.Expr]:
    """Restricted waveform with both unchanged base phase-gauge conventions."""
    selected = _selected_terms(tidal_terms)
    inherited = aligned.waveform_expr(reference_frequency=reference_frequency, terms=spin_terms)
    correction = tidal_phase_expr((sp.pi * M_SEC * F)**sp.Rational(1, 3), tidal_terms=selected)
    if reference_frequency:
        correction -= tidal_phase_expr((sp.pi * M_SEC * F_REF)**sp.Rational(1, 3), tidal_terms=selected)
    phase = inherited["phase"] + correction
    amplitude = inherited["amplitude"]
    carrier = amplitude * sp.exp(-sp.I * phase)
    cosine = sp.cos(IOTA)
    return {"amplitude": amplitude, "phase": phase,
            "h_plus": (1 + cosine**2) * carrier / 2,
            "h_cross": -sp.I * cosine * carrier}


def derivative_expr(expression: sp.Expr, variable: sp.Symbol | str,
                    order: int = 1) -> sp.Expr:
    """Exact canonical partial derivative, without differentiating an EOS map."""
    if isinstance(variable, str):
        variable = SYMBOLS[variable]
    return aligned.derivative_expr(expression, variable, order)


def evaluate(expression: sp.Expr, values: Mapping[sp.Symbol | str, object], *,
             dps: int = 80):
    """Scalar evaluation using the unchanged isolated mpmath expression walker.

    All supplied inputs are validated, including unused values. Lambda_i must
    be finite and nonnegative; signed delta and spin domains are inherited.
    Decimal strings or exact SymPy values preserve intended input precision;
    Python floats keep their binary approximation. A precision convergence
    check verifies numerical stability, not the physical approximation.
    """
    bound = {}
    for key, value in values.items():
        symbol = SYMBOLS[key] if isinstance(key, str) else key
        if symbol in bound:
            raise ValueError(f"duplicate input for {symbol}")
        if symbol in (LAMBDA1, LAMBDA2):
            number = base.evaluate(symbol, {symbol: value}, dps=dps)
            if number < 0:
                raise ValueError(f"{symbol} must be nonnegative")
        bound[symbol] = value
    return aligned.evaluate(expression, bound, dps=dps)


def evaluate_phase(v, delta, chi1, chi2, lambda1, lambda2, *,
                   spin_terms: Iterable[str] = aligned.SPIN_TERMS,
                   tidal_terms: Iterable[str] = TIDAL_TERMS, dps: int = 80):
    """Evaluate the composed intrinsic phase for one scalar binary and velocity."""
    values = {V: v, DELTA: delta, CHI1: chi1, CHI2: chi2, LAMBDA1: lambda1, LAMBDA2: lambda2}
    return evaluate(phase_expr(spin_terms=spin_terms, tidal_terms=tidal_terms), values, dps=dps)


def evaluate_coefficients(delta, chi1, chi2, lambda1, lambda2, *,
                          spin_terms: Iterable[str] = aligned.SPIN_TERMS,
                          tidal_terms: Iterable[str] = TIDAL_TERMS, dps: int = 80) -> dict:
    """Evaluate combined coefficients without production constants or functions."""
    values = {DELTA: delta, CHI1: chi1, CHI2: chi2, LAMBDA1: lambda1, LAMBDA2: lambda2}
    expressions = coefficients(spin_terms=spin_terms, tidal_terms=tidal_terms)
    return {name: evaluate(expr, values, dps=dps) for name, expr in expressions.items()}


def evaluate_waveform(*, frequency, mass_seconds, delta, chi1, chi2, lambda1, lambda2,
                      distance_seconds, inclination="0", tc="0", phic="0",
                      reference_frequency=None, spin_terms: Iterable[str] = aligned.SPIN_TERMS,
                      tidal_terms: Iterable[str] = TIDAL_TERMS, dps: int = 80) -> dict:
    """Scalar waveform in geometric seconds, with explicit physical term sets.

    ``reference_frequency=None`` selects the inherited coalescence gauge; a
    positive value selects finite-reference subtraction. Each lambda is an
    explicit real number; legacy frontend missing-input behavior is separate.
    """
    values = {F: frequency, M_SEC: mass_seconds, DELTA: delta, CHI1: chi1, CHI2: chi2,
              LAMBDA1: lambda1, LAMBDA2: lambda2, D_SEC: distance_seconds,
              IOTA: inclination, TC: tc, PHIC: phic}
    if reference_frequency is not None:
        values[F_REF] = reference_frequency
    expressions = waveform_expr(reference_frequency=reference_frequency is not None,
                                spin_terms=spin_terms, tidal_terms=tidal_terms)
    return {name: evaluate(expr, values, dps=dps) for name, expr in expressions.items()}
