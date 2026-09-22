"""Source-level checks independent of either tensor frontend."""

import ast
import json
from pathlib import Path

import mpmath as mp
import pytest
import sympy as sp

from dev.native_rom.reference import SPEC_VERSION
from dev.native_rom.reference.published import (
    D, calibration, chi, mass_spin, nu, peak_targets, ringdown_shape,
)
from dev.native_rom.reference.ringdown import h22
from dev.native_rom.reference.remnant import final_mass_fraction, final_spin_projection


def test_manifest_is_explicit_and_versioned():
    root = Path(__file__).resolve().parents[1]
    repository = root.parents[2]
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["specification"] == SPEC_VERSION
    components = manifest["components"]
    assert len({item["id"] for item in components}) == len(components)
    assert {item["status"] for item in components} == {"executable", "conditional", "pending"}
    assert any(item["status"] == "pending" for item in components)
    for item in components:
        assert item["source"].startswith("bohe-2017-v3")
        if item["status"] == "pending":
            assert item["blocker"]
        else:
            if item["status"] == "conditional":
                assert item["condition"]
            assert all(item[key] for key in ("reference", "torch", "jax", "test"))
            for mapping in (item["torch"], item["jax"]):
                for entry in mapping.split("; "):
                    assert (repository / entry.split(":")[0]).is_file()
            for entry in item["test"].split("; "):
                filename, function = entry.split(":")
                test_file = repository / filename
                assert test_file.is_file()
                assert any(isinstance(node, ast.FunctionDef) and node.name == function
                           for node in ast.parse(test_file.read_text()).body)


def test_parameter_mapping():
    n, c = mass_spin(sp.Rational(3), sp.Rational(2), sp.Rational(1, 2), sp.Rational(-1, 4))
    assert n == sp.Rational(6, 25)
    assert c == sp.Rational(7, 26)
    assert mass_spin(6, 4, sp.Rational(1, 2), sp.Rational(-1, 4)) == (n, c)
    assert mass_spin(2, 3, sp.Rational(-1, 4), sp.Rational(1, 2)) == (n, c)
    with pytest.raises(ValueError):
        mass_spin(1, -1, 0, 0)


def test_calibration_fits():
    fits = calibration()
    assert sp.simplify(fits["K"].subs(chi, 0) - fits["K_nonspinning"]) == 0
    assert sp.simplify(fits["peak_lag"].subs(chi, 0) - fits["peak_lag_nonspinning"]) == 0
    assert fits["K_nonspinning"].subs(nu, 0) == D("1.733598")
    assert fits["peak_lag_nonspinning"].subs(nu, 0) == D("-2.504992")
    assert fits["d_SO"].subs({nu: 0, chi: 0}) == D("-44.532373")
    assert fits["d_SS"].subs({nu: 0, chi: 0}) == D("6.068071")


def test_peak_targets():
    targets = peak_targets()
    assert targets["amplitude_derivative"] == 0
    assert targets["amplitude"].subs({nu: D("0.001"), chi: 0}) / D("0.001") == D("1.452857")
    assert targets["amplitude"].subs({nu: D("0.25"), chi: 0}) / D("0.25") == D("1.577458")
    assert targets["amplitude_curvature"].subs({nu: D("0.25"), chi: 0}) / D("0.25") == D("-0.00412651")
    assert sp.simplify(targets["frequency"].subs(nu, 0) -
                       (D("0.562679")+(D("-0.087062")+D("0.001743")*chi)
                        *sp.log(D("25.850378")-D("25.819795")*chi))) == 0
    assert sp.simplify(targets["frequency_derivative"].subs(nu, D("0.25")) -
                       (D("0.011282")+D("0.000287")*chi)) == 0


def test_ringdown_continuity():
    with mp.workdps(60):
        n, c = mp.mpf("0.24"), mp.mpf("0.15")
        a0, w0, phi0 = mp.mpf("0.35"), mp.mpf("0.32"), mp.mpf("0.4")
        sigma = mp.mpc("0.55", "-0.08")
        def f(t):
            return h22(t, nu=n, chi=c, sigma=sigma, amplitude=a0,
                       frequency=w0, phase=phi0)
        anchor = a0*mp.exp(-1j*phi0)
        slope = -1j*w0*anchor  # at the amplitude peak, d|h|/dt=0
        assert abs(f(0)-anchor) < mp.mpf("1e-55")
        # The ansatz is defined for t >= 0; use a one-sided derivative.
        step = mp.mpf("1e-28")
        assert abs((f(step)-f(0))/step-slope) < mp.mpf("1e-27")
        # Finite values at increasing precision are a convergence check, not
        # additional accuracy for the paper's decimal fit coefficients.
        lo = h22("1.5", nu=n, chi=c, sigma=sigma, amplitude=a0,
                 frequency=w0, phase=phi0, digits=40)
        hi = h22("1.5", nu=n, chi=c, sigma=sigma, amplitude=a0,
                 frequency=w0, phase=phi0, digits=60)
        assert abs(lo-hi) < mp.mpf("1e-38")


def test_ringdown_attachment_equations_symbolically():
    t = sp.symbols("t", real=True)
    n, a0, gamma, c1, c2, d1, d2, w, w0 = sp.symbols(
        "n a0 gamma c1 c2 d1 d2 w w0", positive=True,
    )
    c1c = gamma*a0*sp.cosh(c2)**2/(n*c1)
    c2c = a0/n-c1c*sp.tanh(c2)
    envelope = n*(c1c*sp.tanh(c1*t+c2)+c2c)*sp.exp(-gamma*t)
    residual = -(w-w0)*(1+d2)/(d1*d2)*sp.log((1+d2*sp.exp(-d1*t))/(1+d2))
    assert sp.simplify(envelope.subs(t, 0)-a0) == 0
    assert sp.simplify(sp.diff(envelope, t).subs(t, 0)) == 0
    assert sp.simplify(sp.diff(residual-w*t, t).subs(t, 0)+w0) == 0
    assert sp.simplify(sp.limit(residual, d2, 0) -
                       (w-w0)*(1-sp.exp(-d1*t))/d1) == 0


def test_ringdown_shape_printed_constant_terms():
    values = {key: expr.subs({nu: 0, chi: 0}) for key, expr in ringdown_shape().items()}
    assert values == {"c1f": D("0.0830664"), "c2f": D("-0.623953"),
                      "d1f": D("0.147584"), "d2f": D("2.46654")}


def test_remnant_mass_paper_limit_and_identity():
    n, e, f = sp.symbols("n e f")
    barausse_form = n*(1-e)+4*n**2*(4*f+e-1)
    combined_form = n*(1-4*n)*(1-e)+16*n**2*f
    assert sp.expand(barausse_form-combined_form) == 0

    q, c1, c2 = sp.symbols("q c1 c2")
    a = (q**2*c1+c2)/(q+1)**2
    paper_spin_argument = a*(1+1/q)**2/(1+1/q**2)
    weighted_spin = (q**2*c1+c2)/(q**2+1)
    assert sp.cancel(paper_spin_argument-weighted_spin) == 0

    with mp.workdps(60):
        # At equal mass, the ISCO contribution cancels and the Hemberger fit
        # is recovered exactly, including its printed central coefficients.
        b0, b1, b2 = mp.mpf("0.00258"), mp.mpf("-0.07730"), mp.mpf("-1.6939")
        for spin in ("-0.9", "0", "0.8"):
            expected = 1-b0-b1/(b2+mp.mpf(spin))
            assert abs(final_mass_fraction(1, 1, spin, spin)-expected) < mp.mpf("1e-55")
        assert final_mass_fraction(30, 20, ".7", "-.2") == final_mass_fraction(
            20, 30, "-.2", ".7")
        assert final_mass_fraction(30, 20, ".7", "-.2") == final_mass_fraction(
            3, 2, ".7", "-.2")
    with pytest.raises(ValueError):
        final_mass_fraction(-1, 2, 0, 0)


@pytest.mark.parametrize("fit_order", [(1, 2), (3, 3), (3, 4)])
def test_hofmann_published_limits(fit_order):
    with mp.workdps(60):
        assert abs(final_spin_projection(1, 1, 0, 0, fit_order=fit_order)
                   - mp.mpf("0.68646")) < mp.mpf("1e-55")
        a = final_spin_projection(30, 20, ".7", "-.2", fit_order=fit_order)
        b = final_spin_projection(20, 30, "-.2", ".7", fit_order=fit_order)
        c = final_spin_projection(3, 2, ".7", "-.2", fit_order=fit_order)
        assert a == b == c
    with pytest.raises(ValueError):
        final_spin_projection(1, 1, 0, 0, fit_order=(2, 2))
