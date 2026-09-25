"""Full intrinsic IMRPhenomD composition with explicit fits and QNM data.

The remnant equations come from Husa et al., arXiv:1508.07250v2. The
amplitude and phase come from Khan et al., arXiv:1508.07253v2. Numerical QNM
knots and any inherited extra fit digits are caller-supplied inputs. This
module imports no production waveform, coefficient, or QNM-data helpers.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence

import sympy as sp

from . import imrphenomd_remnant as remnant
from . import imrphenomd_waveform as waveform

SPEC_VERSION = "imrphenomd-intrinsic-composition/v1"
COMPONENT_VERSIONS = (waveform.SPEC_VERSION, remnant.SPEC_VERSION)
COMPONENT_IDS = {
    "remnant_qnm": "phenomd.intrinsic.remnant_qnm.v1",
    "polarizations": "phenomd.intrinsic.polarizations.v1",
    "derivatives": "phenomd.intrinsic.derivatives.v1",
}
COMPONENT_SOURCES = {
    COMPONENT_IDS["remnant_qnm"]: ("husa2016", "17", "22-24", "qnm-data-input"),
    COMPONENT_IDS["polarizations"]: ("khan2016", "5-7", "17-18", "36-39"),
    COMPONENT_IDS["derivatives"]: ("derived-central-difference", "remnant_qnm",
                                   "polarizations"),
}
_TORCH = ("torchwave.approximants.imrphenomd", "IMRPhenomD.__call__")
_JAX = ("jaxwave.approximants.imrphenomd", "IMRPhenomD.evaluate")
COMPONENT_BINDINGS = {
    COMPONENT_IDS["remnant_qnm"]: {
        "reference": "remnant_qnm", "torch": ("torchwave.approximants.imrphenomd",
                                             "IMRPhenomD.remnant_properties"),
        "jax": ("jaxwave.approximants.imrphenomd",
                "IMRPhenomD.remnant_properties"),
        "tests": ("test_full_remnant_qnm_and_tangents",),
    },
    COMPONENT_IDS["polarizations"]: {
        "reference": "compose", "torch": _TORCH, "jax": _JAX,
        "tests": ("test_full_intrinsic_polarizations",
                  "test_high_spin_complete_complex_waveform"),
    },
    COMPONENT_IDS["derivatives"]: {
        "reference": "polarization_derivative", "torch": _TORCH, "jax": _JAX,
        "tests": ("test_full_intrinsic_derivatives",
                  "test_high_spin_complex_intrinsic_derivatives"),
    },
}


def _rows(rows: Mapping[str, Sequence[str]] | None):
    rows = rows or {}
    return {
        "eta_coeffs": rows.get("spin_eta", remnant.SPIN_ETA_PRINTED),
        "s_eta_coeffs": rows.get("spin_s_eta", remnant.SPIN_S_ETA_PRINTED),
        "s_eta2_coeffs": rows.get("spin_s_eta2", remnant.SPIN_S_ETA2_PRINTED),
        "ns_coeffs": rows.get("energy_ns", remnant.ENERGY_NS_PRINTED),
        "num_coeffs": rows.get("energy_num", remnant.ENERGY_NUM_PRINTED),
        "den_coeffs": rows.get("energy_den", remnant.ENERGY_DEN_PRINTED),
    }


def remnant_expressions(m1, m2, chi1, chi2, *, remnant_rows=None):
    """Return (M, M_f, a_f) using only the sourced symbolic remnant fits."""
    total, eta, spin, s_hat = remnant.coordinates(m1, m2, chi1, chi2)
    rows = _rows(remnant_rows)
    final_spin = remnant.final_spin_expr(eta, spin, **{
        name: rows[name] for name in ("eta_coeffs", "s_eta_coeffs", "s_eta2_coeffs")})
    radiated = remnant.radiated_fraction_expr(eta, s_hat, **{
        name: rows[name] for name in ("ns_coeffs", "num_coeffs", "den_coeffs")})
    return total, remnant.final_mass_expr(total, radiated), final_spin


def remnant_qnm(m1, m2, chi1, chi2, *, knot_spins, cyclic_ordinates,
                remnant_rows=None, generated_splines=None):
    """Return M_f, capped a_f and total-mass-scaled cyclic QNM inputs.

    The optional generated Kerr splines are explicit numerical data, not
    paper coefficients. Above |a_f|=0.994 they replace the rounded cubic.
    """
    total, final_mass, final_spin = remnant_expressions(
        *map(remnant.exact, (m1, m2, chi1, chi2)), remnant_rows=remnant_rows)
    if min(remnant.exact(m1), remnant.exact(m2)) <= 0:
        raise ValueError("component masses must be positive")
    final_spin = sp.Min(sp.Rational(999, 1000),
                        sp.Max(-sp.Rational(999, 1000), final_spin))
    spin = float(sp.N(final_spin, 50))
    if generated_splines is not None and abs(spin) > 0.994:
        from scipy.interpolate import BSpline

        knots, coefficients = generated_splines[
            "positive" if spin >= 0 else "negative"]
        transformed = -math.log1p(-abs(spin))
        angular_real, angular_imag = BSpline(
            knots, coefficients, generated_splines["degree"],
            extrapolate=False)(transformed)
        qnm_final = (float(angular_real) / (2 * math.pi),
                     -float(angular_imag) / (2 * math.pi))
    else:
        qnm_final = remnant.natural_cubic_qnm(
            knot_spins, cyclic_ordinates, spin)
    qnm_total = tuple(remnant.qnm_total_mass_expr(
        total, final_mass, remnant.exact(value)) for value in qnm_final)
    return final_mass, final_spin, *qnm_total


def qnm_tangent(m1, m2, chi1, chi2, parameter, *, knot_spins,
                cyclic_ordinates, remnant_rows=None):
    """Analytic remnant chain rule plus the natural-spline first derivative.

    ``mass_ratio`` means q=m1/m2 at fixed total mass; spins are independent.
    The spline derivative applies inside its knot domain, away from endpoints.
    """
    if parameter not in ("mass_ratio", "chi1", "chi2"):
        raise ValueError(f"unknown intrinsic parameter: {parameter}")
    q, s1, s2 = sp.symbols("q chi1 chi2", real=True)
    total_value = remnant.exact(m1) + remnant.exact(m2)
    m1_expr = total_value * q / (1 + q)
    m2_expr = total_value / (1 + q)
    total, final_mass, final_spin = remnant_expressions(
        m1_expr, m2_expr, s1, s2, remnant_rows=remnant_rows)
    variable = {"mass_ratio": q, "chi1": s1, "chi2": s2}[parameter]
    subs = {q: remnant.exact(m1) / remnant.exact(m2),
            s1: remnant.exact(chi1), s2: remnant.exact(chi2)}
    spin = float(sp.N(final_spin.subs(subs), 50))
    from scipy.interpolate import CubicSpline

    if not knot_spins[0] < spin < knot_spins[-1]:
        raise ValueError("QNM derivative requires an interior final spin")
    spline = CubicSpline(knot_spins, cyclic_ordinates, axis=0, bc_type="natural")
    ordinate, slope = spline(spin), spline(spin, 1)
    mass = float(sp.N(final_mass.subs(subs), 50))
    mass_prime = float(sp.N(sp.diff(final_mass, variable).subs(subs), 50))
    spin_prime = float(sp.N(sp.diff(final_spin, variable).subs(subs), 50))
    total_float = float(sp.N(total.subs(subs), 50))
    qnm_prime = total_float / mass * (
        slope * spin_prime - ordinate * mass_prime / mass)
    return mass_prime, spin_prime, *map(float, qnm_prime)


def compose(m1, m2, chi1, chi2, distance_mpc, *, knot_spins,
            cyclic_ordinates, coefficient_rows=None, remnant_rows=None,
            generated_splines=None):
    """Compose the unchanged scalar waveform with sourced remnant/QNM inputs."""
    _, _, f_rd, f_damp = remnant_qnm(
        m1, m2, chi1, chi2, knot_spins=knot_spins,
        cyclic_ordinates=cyclic_ordinates, remnant_rows=remnant_rows,
        generated_splines=generated_splines)
    return waveform.compose(m1, m2, chi1, chi2, distance_mpc, f_rd, f_damp,
                            coefficient_rows=coefficient_rows)


def polarization_derivative(m1, m2, chi1, chi2, distance_mpc, frequency_hz,
                            parameter, *, knot_spins, cyclic_ordinates,
                            coefficient_rows=None, remnant_rows=None,
                            generated_splines=None, step=None, dps=60,
                            **waveform_options):
    """Convergent fourth-order scalar derivative of the full intrinsic chain.

    This differentiates independently evaluated reference expressions, not a
    frontend autodiff graph. Explicit steps crossing a QNM knot, interpolation
    switch, remnant cap, moving join, physical endpoint, or cutoff are rejected.
    The default step is halved until all four samples lie in the center's
    smooth region, or fails after ten halvings.
    """
    values = tuple(map(remnant.exact, (m1, m2, chi1, chi2)))
    if parameter not in ("mass_ratio", "chi1", "chi2"):
        raise ValueError(f"unknown intrinsic parameter: {parameter}")
    center = values[0] / values[1] if parameter == "mass_ratio" else values[2 if parameter == "chi1" else 3]
    if parameter != "mass_ratio" and not -1 < center < 1:
        raise ValueError("central spin derivative requires an interior spin")
    total = values[0] + values[1]
    qnm_region = _qnm_region(float(sp.N(remnant_expressions(
        *values, remnant_rows=remnant_rows)[2], 50)), knot_spins,
        generated_splines)
    scale = sp.Rational(1, 10000000) if qnm_region[1] != "rounded" else sp.Rational(1, 10000)
    h = remnant.exact(step) if step is not None else scale * max(1, abs(center))
    if h <= 0:
        raise ValueError("step must be positive")
    base = compose(*values, distance_mpc, knot_spins=knot_spins,
                   cyclic_ordinates=cyclic_ordinates,
                   coefficient_rows=coefficient_rows, remnant_rows=remnant_rows,
                   generated_splines=generated_splines)
    signature = _branch_signature(base, frequency_hz, waveform_options.get("f_ref_hz"))
    for attempt in range(11):
        samples = {}
        try:
            for offset in (-2, -1, 1, 2):
                trial = center + offset * h
                if parameter == "mass_ratio":
                    if trial <= 0:
                        raise _StencilBoundary("mass-ratio stencil requires positive q")
                    candidate = (total * trial / (1 + trial), total / (1 + trial),
                                 values[2], values[3])
                else:
                    if not -1 <= trial <= 1:
                        raise _StencilBoundary("spin stencil exceeds the physical domain")
                    candidate = (values[0], values[1], trial, values[3]) if parameter == "chi1" else (
                        values[0], values[1], values[2], trial)
                final_spin = float(sp.N(remnant_expressions(
                    *candidate, remnant_rows=remnant_rows)[2], 50))
                try:
                    candidate_region = _qnm_region(
                        final_spin, knot_spins, generated_splines)
                except ValueError as exc:
                    raise _StencilBoundary(str(exc)) from exc
                if candidate_region != qnm_region:
                    if candidate_region[:2] != qnm_region[:2]:
                        raise _StencilBoundary(
                            "intrinsic derivative stencil crosses a QNM switch or remnant cap")
                    raise _StencilBoundary(
                        "intrinsic derivative stencil crosses a QNM spline knot")
                spec = compose(*candidate, distance_mpc, knot_spins=knot_spins,
                               cyclic_ordinates=cyclic_ordinates,
                               coefficient_rows=coefficient_rows, remnant_rows=remnant_rows,
                               generated_splines=generated_splines)
                try:
                    candidate_signature = _branch_signature(
                        spec, frequency_hz, waveform_options.get("f_ref_hz"))
                except ValueError as exc:
                    raise _StencilBoundary("intrinsic derivative stencil reaches a waveform boundary") from exc
                if candidate_signature != signature:
                    raise _StencilBoundary("intrinsic derivative stencil crosses a waveform branch")
                samples[offset] = spec.evaluate(frequency_hz, dps=dps, **waveform_options)[:2]
        except _StencilBoundary as exc:
            if step is not None:
                raise ValueError(str(exc)) from exc
            if attempt == 10:
                raise ValueError("no smooth intrinsic derivative stencil after ten halvings") from exc
            h /= 2
            continue
        width = float(h)
        return tuple((samples[-2][i] - 8*samples[-1][i] + 8*samples[1][i]
                      - samples[2][i]) / (12*width) for i in (0, 1))


class _StencilBoundary(ValueError):
    """A finite-difference sample left the center's smooth region."""


def _qnm_region(raw_spin, knot_spins, generated_splines):
    """Interpolation and cap branch, plus its open spline-knot interval."""
    if not math.isfinite(raw_spin):
        raise ValueError("QNM derivative requires finite final spin")
    if abs(abs(raw_spin) - .999) <= 64 * math.ulp(.999):
        raise ValueError("QNM derivative is undefined at the remnant cap")
    capped = max(-.999, min(.999, raw_spin))
    cap_branch = "capped" if abs(raw_spin) > .999 else "unclamped"
    if generated_splines is not None:
        if abs(abs(capped) - .994) <= 64 * math.ulp(.994):
            raise ValueError("QNM derivative is undefined at the interpolation switch")
        if abs(capped) > .994:
            label = "positive" if capped > 0 else "negative"
            knots = generated_splines[label][0]
            coordinate = -math.log1p(-abs(capped))
            lower, upper = _qnm_cell(coordinate, knots)
            return cap_branch, label, lower, upper
    lower, upper = _qnm_cell(capped, knot_spins)
    return cap_branch, "rounded", lower, upper


def _qnm_cell(final_spin, knot_spins):
    """Open spline interval; a center on a knot has no fourth-order stencil."""
    if len(knot_spins) < 3 or not math.isfinite(final_spin):
        raise ValueError("QNM derivative requires finite spin and at least three knots")
    index = bisect_left(knot_spins, final_spin)
    if index == 0 or index == len(knot_spins):
        raise ValueError("QNM derivative requires a final spin inside one open knot interval")
    if min(abs(final_spin - knot_spins[index - 1]),
           abs(final_spin - knot_spins[index])) <= 64 * math.ulp(final_spin):
        raise ValueError("QNM derivative center is too close to a spline knot")
    return knot_spins[index - 1], knot_spins[index]


def _branch_signature(spec, frequency_hz, f_ref_hz):
    def region(u, first, second):
        if u == first or u == second or u == sp.Rational(1, 5):
            raise ValueError("derivative is undefined at a branch boundary")
        return 0 if u < first else 1 if u < second else 2
    u = spec.m_sec * waveform.amp.exact(frequency_hz)
    if u > sp.Rational(1, 5):
        raise ValueError("intrinsic derivative is outside the default cutoff")
    result = (region(u, spec.amplitude["f1"], spec.amplitude["f3"]),
              region(u, spec.phase["f1"], spec.phase["f2"]))
    if f_ref_hz is not None:
        u_ref = spec.m_sec * waveform.amp.exact(f_ref_hz)
        result += (region(u_ref, spec.phase["f1"], spec.phase["f2"]),)
    return result
