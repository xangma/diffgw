"""Source-level checks independent of either tensor frontend."""

import ast
import json
from pathlib import Path

import mpmath as mp
import pytest
import sympy as sp

from dev.native_rom.reference import SPEC_VERSION
from dev.native_rom.reference.flux import (
    emitted_flux, factorized_22_amplitude, factorized_31_amplitude,
    factorized_42_amplitude,
    factorized_43_amplitude, factorized_55_amplitude, factorized_62_amplitude,
    factorized_54_52_amplitudes, factorized_66_64_amplitudes,
    factorized_6_odd_amplitudes, factorized_7_even_amplitudes,
    factorized_8_even_amplitudes, factorized_8_odd_amplitudes,
    factorized_41_amplitude, factorized_53_51_amplitudes,
    factorized_7_odd_amplitudes,
    newtonian_22_mode,
)
from dev.native_rom.reference.hamiltonian import (
    aligned_effective_hamiltonian, body_parameters, effective_spin_mapping, energy_map,
    nonspinning_effective_hamiltonian, radial_potentials,
)
from dev.native_rom.reference.published import (
    D, calibration, chi, mass_spin, nu, peak_targets, ringdown_shape,
)
from dev.native_rom.reference.qnm import berti_220_fit, sigma_initial_mass_units
from dev.native_rom.reference.ringdown import h22
from dev.native_rom.reference.remnant import final_mass_fraction, final_spin_projection


def test_manifest_is_explicit_and_versioned():
    root = Path(__file__).resolve().parents[1]
    repository = root.parents[2]
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["specification"] == SPEC_VERSION
    components = manifest["components"]
    assert len({item["id"] for item in components}) == len(components)
    assert {item["status"] for item in components} == {
        "executable", "conditional", "diagnostic", "pending",
    }
    assert any(item["status"] == "pending" for item in components)
    for item in components:
        assert item["source"].startswith(("bohe-2017-v3", "Buonanno--Damour gr-qc/0001013v2"))
        if item["status"] == "pending":
            assert item["blocker"]
        else:
            if item["status"] in {"conditional", "diagnostic"}:
                assert item["condition"]
            assert all(item[key] for key in ("reference", "test"))
            assert item.get("torch") or item.get("jax")
            for mapping in (item.get("torch"), item.get("jax")):
                if mapping is None:
                    continue
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


def test_radial_potential_paper_limits_and_symmetry():
    with mp.workdps(70):
        zero = radial_potentials(12, 0, mp.mpf("0.4"), mp.mpf("1.7"))
        assert abs(zero["delta_t"]/144 - (1-mp.mpf(2)/12+mp.mpf("0.16")/144)) < mp.mpf("1e-55")
        for m1, m2, s1, s2 in ((30, 20, ".6", "-.2"), (20, 20, "-.8", ".8")):
            args = body_parameters(m1, m2, s1, s2)
            exchanged = body_parameters(m2, m1, s2, s1)
            assert all(abs(x-y) < mp.mpf("1e-65") for x, y in zip(args, exchanged))
            p = radial_potentials(9, *args)
            h = mp.mpf("1e-5")
            finite_slope = (radial_potentials(9+h, *args)["delta_t"]
                            - radial_potentials(9-h, *args)["delta_t"])/(2*h)
            assert abs(p["delta_t_prime"]-finite_slope) < mp.mpf("1e-9")
            assert p["delta_t"] > 0 and p["delta_r"] > 0 and p["xi"] > 0


def test_energy_map_paper_normalization():
    with mp.workdps(70):
        n, h_eff = mp.mpf("0.24"), mp.mpf("0.94")
        total, reduced = energy_map(n, h_eff)
        assert abs((total-1)/n-reduced) < mp.mpf("1e-59")
        assert abs(total**2-(1+2*n*(h_eff-1))) < mp.mpf("1e-59")


def test_nonspinning_effective_hamiltonian_schwarzschild_limit():
    with mp.workdps(70):
        r, pr_star, p_phi = mp.mpf(12), mp.mpf("0.03"), mp.mpf("3.2")
        expected = mp.sqrt(pr_star**2 + (1-2/r)*(1+p_phi**2/r**2))
        actual = nonspinning_effective_hamiltonian(r, pr_star, p_phi, 0, 1.7)
        assert abs(actual-expected) < mp.mpf("1e-55")


def test_effective_spin_mapping_limits_and_exchange():
    with mp.workdps(70):
        args = (30, 20, ".6", "-.2", 9, ".15", ".002")
        actual = effective_spin_mapping(*args)
        exchanged = effective_spin_mapping(20, 30, "-.2", ".6", 9, ".15", ".002")
        assert abs(actual-exchanged) < mp.mpf("1e-59")
        assert effective_spin_mapping(30, 20, 0, 0, 9, ".15", ".002") == 0
        # In the test-particle limit both S* and every correction to S_Kerr
        # vanish with nu; the map itself must tend to zero.
        assert abs(effective_spin_mapping(1, "1e-8", ".6", 0, 9, ".15", ".002")) < mp.mpf("1e-6")
    with pytest.raises(ValueError):
        effective_spin_mapping(30, 20, ".6", "-.2", 9, ".001", ".002")


def test_aligned_effective_hamiltonian_symmetries_and_nonspin_limit():
    with mp.workdps(70):
        args = (30, 20, ".6", "-.2", 9, ".01", 3)
        actual = aligned_effective_hamiltonian(*args)
        exchanged = aligned_effective_hamiltonian(20, 30, "-.2", ".6", 9, ".01", 3)
        rescaled = aligned_effective_hamiltonian(3, 2, ".6", "-.2", 9, ".01", 3)
        assert abs(actual-exchanged) < mp.mpf("1e-58")
        assert abs(actual-rescaled) < mp.mpf("1e-58")
        n, _, k = body_parameters(30, 20, 0, 0)
        nonspin = nonspinning_effective_hamiltonian(9, ".01", 3, n, k)
        assert abs(aligned_effective_hamiltonian(30, 20, 0, 0, 9, ".01", 3)
                   - nonspin) < mp.mpf("1e-58")


def test_flux_paper_normalization_and_mode_partner():
    with mp.workdps(70):
        n, w, phase = mp.mpf("0.24"), mp.mpf("0.04"), mp.mpf("0.37")
        mode = newtonian_22_mode(n, w, phase)
        actual = emitted_flux(w, {(2, 2): mode})
        expected = mp.mpf(32)/5*n**2*w**(mp.mpf(10)/3)
        assert abs(actual/expected-1) < mp.mpf("1e-59")
        # The 1/(8*pi) positive-m norm counts each negative-m partner once.
        two_modes = emitted_flux(w, {(2, 2): mode, (3, 2): 2*mode})
        assert abs(two_modes/actual-5) < mp.mpf("1e-59")
        with pytest.raises(ValueError):
            emitted_flux(w, {(2, -2): mode})


def test_factorized_22_newtonian_limit():
    with mp.workdps(70):
        w = mp.mpf("1e-15")
        actual = factorized_22_amplitude(30, 20, 0, 0, w, w, 1, 1)
        expected = abs(newtonian_22_mode(mp.mpf("0.24"), w))
        assert abs(actual/expected-1) < mp.mpf("1e-9")


def test_factorized_55_newtonian_limit_and_mass_exchange():
    with mp.workdps(70):
        w = mp.mpf("1e-15")
        n = mp.mpf("0.24")
        x1, x2 = mp.mpf("0.6"), mp.mpf("0.4")
        expected = (mp.mpf(125)/792*mp.sqrt(330*mp.pi)*n*w**(mp.mpf(5)/3)
                    *abs(x2**4-x1**4))
        actual = factorized_55_amplitude(30, 20, 0, 0, w, w, 1, 1)
        assert abs(actual/expected-1) < mp.mpf("1e-9")
        forward = factorized_55_amplitude(30, 20, ".7", "-.4", ".08", ".077", ".91", ".975")
        exchanged = factorized_55_amplitude(20, 30, "-.4", ".7", ".08", ".077", ".91", ".975")
        assert abs(forward-exchanged) < mp.mpf("1e-60")
        assert factorized_55_amplitude(20, 20, ".9", "-.9", ".08", ".077", ".91", ".975") == 0


def test_factorized_54_52_newtonian_limits_and_mass_exchange():
    with mp.workdps(70):
        w, n = mp.mpf("1e-15"), mp.mpf("0.24")
        c6 = mp.mpf("0.6")**5+mp.mpf("0.4")**5
        constants = (mp.mpf(256)/1485*mp.sqrt(33*mp.pi),
                     mp.mpf(16)/1485*mp.sqrt(11*mp.pi))
        actual = factorized_54_52_amplitudes(30, 20, 0, 0, w, w, 3, 1)
        assert all(abs(value/(constant*n*c6*w**2*3*w**(mp.mpf(1)/3))-1)
                   < mp.mpf("1e-9") for value, constant in zip(actual, constants))
        forward = factorized_54_52_amplitudes(30, 20, ".7", "-.4", ".08", ".077", 3, ".975")
        exchanged = factorized_54_52_amplitudes(20, 30, "-.4", ".7", ".08", ".077", 3, ".975")
        assert all(abs(a-b) < mp.mpf("1e-60") for a, b in zip(forward, exchanged))


def test_factorized_7_even_newtonian_limits_and_spin_truncation():
    with mp.workdps(70):
        w, n = mp.mpf("1e-15"), mp.mpf("0.24")
        c8 = mp.mpf("0.6")**7+mp.mpf("0.4")**7
        constants = (mp.mpf(648)/25025*mp.sqrt(2145*mp.pi),
                     mp.mpf(1024)/225225*mp.sqrt(330*mp.pi),
                     mp.mpf(8)/45045*mp.sqrt(15*mp.pi))
        actual = factorized_7_even_amplitudes(30, 20, 0, 0, w, w, 3, 1)
        assert all(abs(value/(constant*n*c8*w**(mp.mpf(8)/3)*3*w**(mp.mpf(1)/3))-1)
                   < mp.mpf("1e-9") for value, constant in zip(actual, constants))
        forward = factorized_7_even_amplitudes(30, 20, ".7", "-.4", ".08", ".077", 3, ".975")
        exchanged = factorized_7_even_amplitudes(20, 30, "-.4", ".7", ".08", ".077", 3, ".975")
        assert all(abs(a-b) < mp.mpf("1e-60") for a, b in zip(forward, exchanged))
        # The selected printed 1PN truncation has no explicit spin terms.
        nonspin = factorized_7_even_amplitudes(30, 20, 0, 0, ".08", ".077", 3, ".975")
        assert all(abs(a-b) < mp.mpf("1e-60") for a, b in zip(forward, nonspin))


def test_factorized_6_odd_newtonian_limits_and_mass_exchange():
    with mp.workdps(70):
        w, n = mp.mpf("1e-15"), mp.mpf("0.24")
        c7 = abs(mp.mpf("0.4")**6-mp.mpf("0.6")**6)
        constants = (mp.mpf(625)/27027*mp.sqrt(2145*mp.pi),
                     mp.mpf(81)/5005*mp.sqrt(13*mp.pi),
                     mp.sqrt(130*mp.pi)/135135)
        actual = factorized_6_odd_amplitudes(30, 20, 0, 0, w, w, 3, 1)
        assert all(abs(value/(constant*n*c7*w**(mp.mpf(7)/3)*3*w**(mp.mpf(1)/3))-1)
                   < mp.mpf("1e-9") for value, constant in zip(actual, constants))
        forward = factorized_6_odd_amplitudes(30, 20, ".7", "-.4", ".08", ".077", 3, ".975")
        exchanged = factorized_6_odd_amplitudes(20, 30, "-.4", ".7", ".08", ".077", 3, ".975")
        assert all(abs(a-b) < mp.mpf("1e-60") for a, b in zip(forward, exchanged))
        assert factorized_6_odd_amplitudes(20, 20, ".7", "-.4", ".08", ".077", 3, ".975") == (0, 0, 0)


def test_factorized_8_odd_newtonian_limits_exchange_and_truncation():
    with mp.workdps(70):
        w, n = mp.mpf("1e-15"), mp.mpf("0.24")
        c9 = abs(mp.mpf("0.4")**8-mp.mpf("0.6")**8)
        constants = (mp.mpf(117649)/15752880*mp.sqrt(34034*mp.pi),
                     mp.mpf(15625)/22054032*mp.sqrt(24310*mp.pi),
                     mp.mpf(81)/1361360*mp.sqrt(1122*mp.pi),
                     mp.sqrt(1190*mp.pi)/110270160)
        actual = factorized_8_odd_amplitudes(30, 20, 0, 0, w, w, 3, 1)
        assert all(abs(value/(constant*n*c9*w**3*3*w**(mp.mpf(1)/3))-1)
                   < mp.mpf("1e-9") for value, constant in zip(actual, constants))
        forward = factorized_8_odd_amplitudes(30, 20, ".7", "-.4", ".08", ".077", 3, ".975")
        exchanged = factorized_8_odd_amplitudes(20, 30, "-.4", ".7", ".08", ".077", 3, ".975")
        assert all(abs(a-b) < mp.mpf("1e-60") for a, b in zip(forward, exchanged))
        nonspin = factorized_8_odd_amplitudes(30, 20, 0, 0, ".08", ".077", 3, ".975")
        assert all(abs(a-b) < mp.mpf("1e-60") for a, b in zip(forward, nonspin))
        assert factorized_8_odd_amplitudes(20, 20, ".7", "-.4", ".08", ".077", 3, ".975") == (0, 0, 0, 0)


def test_factorized_41_newtonian_exchange_and_equal_mass_spin():
    with mp.workdps(70):
        w, n = mp.mpf("1e-15"), mp.mpf("0.24")
        c5 = abs(mp.mpf("0.4")**4-mp.mpf("0.6")**4)
        actual = factorized_41_amplitude(30, 20, 0, 0, w, w, 3, 1)
        expected = mp.sqrt(2*mp.pi)/105*n*c5*w**(mp.mpf(5)/3)*3*w**(mp.mpf(1)/3)
        assert abs(actual/expected-1) < mp.mpf("1e-9")
        forward = factorized_41_amplitude(30, 20, ".7", "-.4", ".08", ".077", 3, ".975")
        exchanged = factorized_41_amplitude(20, 30, "-.4", ".7", ".08", ".077", 3, ".975")
        assert abs(forward-exchanged) < mp.mpf("1e-60")
        assert factorized_41_amplitude(20, 20, ".7", "-.7", ".08", ".077", 3, ".975") > 0
        assert factorized_41_amplitude(20, 20, 0, 0, ".08", ".077", 3, ".975") == 0


def test_factorized_53_51_newtonian_exchange_and_equal_mass_limit():
    with mp.workdps(70):
        w, n = mp.mpf("1e-15"), mp.mpf("0.24")
        c5 = abs(mp.mpf("0.4")**4-mp.mpf("0.6")**4)
        constants = (mp.mpf(9)/440*mp.sqrt(66*mp.pi), mp.sqrt(77*mp.pi)/13860)
        actual = factorized_53_51_amplitudes(30, 20, 0, 0, w, w, 1, 1)
        assert all(abs(value/(constant*n*c5*w**(mp.mpf(5)/3))-1)
                   < mp.mpf("1e-9") for value, constant in zip(actual, constants))
        forward = factorized_53_51_amplitudes(30, 20, ".7", "-.4", ".08", ".077", ".91", ".975")
        exchanged = factorized_53_51_amplitudes(20, 30, "-.4", ".7", ".08", ".077", ".91", ".975")
        assert all(abs(a-b) < mp.mpf("1e-60") for a, b in zip(forward, exchanged))
        assert factorized_53_51_amplitudes(20, 20, ".7", "-.4", ".08", ".077", ".91", ".975") == (0, 0)


def test_factorized_7_odd_newtonian_exchange_and_equal_mass_limit():
    with mp.workdps(70):
        w, n = mp.mpf("1e-15"), mp.mpf("0.24")
        c7 = abs(mp.mpf("0.4")**6-mp.mpf("0.6")**6)
        constants = (mp.mpf(16807)/772200*mp.sqrt(30030*mp.pi),
                     mp.mpf(3125)/216216*mp.sqrt(330*mp.pi),
                     mp.mpf(243)/200200*mp.sqrt(30*mp.pi),
                     mp.sqrt(10*mp.pi)/1081080)
        actual = factorized_7_odd_amplitudes(30, 20, 0, 0, w, w, 1, 1)
        assert all(abs(value/(constant*n*c7*w**(mp.mpf(7)/3))-1)
                   < mp.mpf("1e-9") for value, constant in zip(actual, constants))
        forward = factorized_7_odd_amplitudes(30, 20, ".7", "-.4", ".08", ".077", ".91", ".975")
        exchanged = factorized_7_odd_amplitudes(20, 30, "-.4", ".7", ".08", ".077", ".91", ".975")
        assert all(abs(a-b) < mp.mpf("1e-60") for a, b in zip(forward, exchanged))
        assert factorized_7_odd_amplitudes(20, 20, ".7", "-.4", ".08", ".077", ".91", ".975") == (0, 0, 0, 0)


def test_factorized_31_newtonian_limit_exchange_and_equal_mass_spin():
    with mp.workdps(70):
        w = mp.mpf("1e-15")
        n, delta = mp.mpf("0.24"), mp.mpf("0.2")
        expected = mp.sqrt(2*mp.pi/35)/3*n*w*delta
        actual = factorized_31_amplitude(30, 20, 0, 0, w, w, 1, 1)
        assert abs(actual/expected-1) < mp.mpf("1e-9")
        forward = factorized_31_amplitude(30, 20, ".7", "-.4", ".08", ".077", ".91", ".975")
        exchanged = factorized_31_amplitude(20, 30, "-.4", ".7", ".08", ".077", ".91", ".975")
        assert abs(forward-exchanged) < mp.mpf("1e-60")
        assert factorized_31_amplitude(20, 20, ".9", ".9", ".08", ".077", ".91", ".975") == 0
        assert factorized_31_amplitude(20, 20, ".9", "-.9", ".08", ".077", ".91", ".975") > 0


def test_factorized_66_64_newtonian_limits_and_mass_exchange():
    with mp.workdps(70):
        w, n = mp.mpf("1e-15"), mp.mpf("0.24")
        c6 = mp.mpf("0.6")**5+mp.mpf("0.4")**5
        expected = (mp.mpf(432)/3575*mp.sqrt(715*mp.pi)*n*c6*w**2,
                    mp.mpf(1024)/96525*mp.sqrt(390*mp.pi)*n*c6*w**2)
        actual = factorized_66_64_amplitudes(30, 20, 0, 0, w, w, 1, 1)
        assert all(abs(value/leading-1) < mp.mpf("1e-9")
                   for value, leading in zip(actual, expected))
        forward = factorized_66_64_amplitudes(30, 20, ".7", "-.4", ".08", ".077", ".91", ".975")
        exchanged = factorized_66_64_amplitudes(20, 30, "-.4", ".7", ".08", ".077", ".91", ".975")
        assert all(abs(a-b) < mp.mpf("1e-60") for a, b in zip(forward, exchanged))


def test_factorized_62_newtonian_limit_and_mass_exchange():
    with mp.workdps(70):
        w, n = mp.mpf("1e-15"), mp.mpf("0.24")
        c6 = mp.mpf("0.6")**5+mp.mpf("0.4")**5
        expected = mp.mpf(16)/19305*mp.sqrt(13*mp.pi)*n*c6*w**2
        actual = factorized_62_amplitude(30, 20, 0, 0, w, w, 1, 1)
        assert abs(actual/expected-1) < mp.mpf("1e-9")
        forward = factorized_62_amplitude(30, 20, ".7", "-.4", ".08", ".077", ".91", ".975")
        exchanged = factorized_62_amplitude(20, 30, "-.4", ".7", ".08", ".077", ".91", ".975")
        assert abs(forward-exchanged) < mp.mpf("1e-60")


def test_factorized_8_even_newtonian_limits_exchange_and_truncation():
    with mp.workdps(70):
        w, n = mp.mpf("1e-15"), mp.mpf("0.24")
        c8 = mp.mpf("0.6")**7+mp.mpf("0.4")**7
        constants = (mp.mpf(131072)/5360355*mp.sqrt(34034*mp.pi),
                     mp.mpf(1944)/2977975*mp.sqrt(255255*mp.pi),
                     mp.mpf(1024)/3828825*mp.sqrt(1870*mp.pi),
                     mp.mpf(8)/765765*mp.sqrt(17*mp.pi))
        actual = factorized_8_even_amplitudes(
            30, 20, 0, 0, w, w, 1, 1, spin_truncation="tabulated")
        assert all(abs(value/(constant*n*c8*w**(mp.mpf(8)/3))-1) < mp.mpf("1e-9")
                   for value, constant in zip(actual, constants))
        forward = factorized_8_even_amplitudes(
            30, 20, ".7", "-.4", ".08", ".077", ".91", ".975",
            spin_truncation="generic-leading-ell8")
        exchanged = factorized_8_even_amplitudes(
            20, 30, "-.4", ".7", ".08", ".077", ".91", ".975",
            spin_truncation="generic-leading-ell8")
        assert all(abs(a-b) < mp.mpf("1e-60") for a, b in zip(forward, exchanged))
        tabulated = factorized_8_even_amplitudes(
            30, 20, ".7", "-.4", ".08", ".077", ".91", ".975",
            spin_truncation="tabulated")
        assert all(abs(a-b) > mp.mpf("1e-12")*abs(a)
                   for a, b in zip(forward, tabulated))
        with pytest.raises(ValueError):
            factorized_8_even_amplitudes(30, 20, 0, 0, w, w, 1, 1,
                                         spin_truncation="unspecified")


def test_factorized_42_newtonian_limit_and_mass_exchange():
    with mp.workdps(70):
        w = mp.mpf("1e-15")
        n = mp.mpf("0.24")
        expected = mp.mpf(8)/63*mp.sqrt(mp.pi)*n*w**(mp.mpf(4)/3)*(1-3*n)
        actual = factorized_42_amplitude(30, 20, 0, 0, w, w, 1, 1)
        assert abs(actual/expected-1) < mp.mpf("1e-9")
        forward = factorized_42_amplitude(30, 20, ".7", "-.4", ".08", ".077", ".91", ".975")
        exchanged = factorized_42_amplitude(20, 30, "-.4", ".7", ".08", ".077", ".91", ".975")
        assert abs(forward-exchanged) < mp.mpf("1e-60")


def test_factorized_43_newtonian_limit_and_equal_mass_spin():
    with mp.workdps(70):
        w = mp.mpf("1e-15")
        n, delta = mp.mpf("0.24"), mp.mpf("0.2")
        expected = (mp.mpf(9)/5*mp.sqrt(2*mp.pi/7)*n*w**(mp.mpf(5)/3)
                    *3*w**(mp.mpf(1)/3)*delta*(1-2*n))
        actual = factorized_43_amplitude(30, 20, 0, 0, w, w, 3, 1)
        assert abs(actual/expected-1) < mp.mpf("1e-9")
        forward = factorized_43_amplitude(30, 20, ".7", "-.4", ".08", ".077", 3, ".975")
        exchanged = factorized_43_amplitude(20, 30, "-.4", ".7", ".08", ".077", 3, ".975")
        assert abs(forward-exchanged) < mp.mpf("1e-60")
        assert factorized_43_amplitude(20, 20, ".9", ".9", ".08", ".077", 3, ".975") == 0
        assert factorized_43_amplitude(20, 20, ".9", "-.9", ".08", ".077", 3, ".975") > 0


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


def test_berti_fit_is_a_diagnostic_and_qnm_convention():
    with mp.workdps(60):
        frequency, damping = berti_220_fit(0)
        assert frequency == mp.mpf("0.3683")
        assert abs(frequency/(2*damping)-mp.mpf("2.1187")) < mp.mpf("1e-59")
        # The fit differs from the numerical value printed in Berti Table II.
        assert abs(frequency-mp.mpf("0.3737")) > mp.mpf("0.005")
        sigma = sigma_initial_mass_units("0.6", "0.08", "0.95")
        assert abs(sigma-mp.mpc(mp.mpf("0.6")/mp.mpf("0.95"),
                                -mp.mpf("0.08")/mp.mpf("0.95"))) < mp.mpf("1e-59")
        assert abs(mp.exp(-1j*sigma*mp.mpf("0.95"))
                   -mp.exp(mp.mpc("-0.08", "-0.6"))) < mp.mpf("1e-59")
    with pytest.raises(ValueError):
        berti_220_fit("0.995")
    with pytest.raises(ValueError):
        sigma_initial_mass_units("0.6", "0.08", "0")


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
