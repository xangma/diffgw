"""Independent SymPy checks of the non-Kerr spin quadrupole TaylorF2 slice."""

from functools import lru_cache

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import taylorf2_spin_quadrupole as ref
from diffgw_reference.aligned_spin_traceability import frontend_terms as spin_terms
from diffgw_reference.tidal_traceability import frontend_terms as tidal_terms
from test_taylorf2_aligned_spin_math_reference import host
from test_taylorf2_math_reference import backend as backend, exact
from test_taylorf2_tidal_math_reference import reference_values as tidal_values


CASES = [
    dict(mass1=1.4, mass2=1.4, chi1=.2, chi2=-.1, lambda1=400., lambda2=400.,
         quad_monopole1=4., quad_monopole2=6., distance=100., inclination=.7, tc=.01, phic=.2),
    dict(mass1=1.6, mass2=1.2, chi1=.3, chi2=.07, lambda1=200., lambda2=800.,
         quad_monopole1=3., quad_monopole2=8., distance=200., inclination=1.2, tc=-.03, phic=-.4),
    dict(mass1=7., mass2=1.4, chi1=.5, chi2=.02, lambda1=0., lambda2=500.,
         quad_monopole1=1., quad_monopole2=5., distance=400., inclination=1., tc=0., phic=.7),
]


def values(case, frequency, f_ref=40.):
    result = tidal_values(case, frequency, f_ref)
    result.update({ref.CQ1: exact(case["quad_monopole1"]),
                   ref.CQ2: exact(case["quad_monopole2"])})
    return result


@lru_cache(maxsize=None)
def expressions(spin_order, use_ref, calibration=False):
    return ref.waveform_expr(reference_frequency=use_ref,
                             spin_terms=spin_terms(spin_order, phenom_d_calibration=calibration),
                             tidal_terms=tidal_terms(13),
                             quad_terms=ref.frontend_terms(spin_order, calibration))


def test_quadrupole_paper_identities():
    d, c1, c2 = ref.DELTA, ref.CHI1, ref.CHI2
    x1, x2 = (1+d)/2, (1-d)/2
    q1, q2 = (ref.CQ1-1)*(x1*c1)**2, (ref.CQ2-1)*(x2*c2)**2
    terms = ref.quadrupole_corrections()
    # Dietrich et al. Eq. (27): their equivalent body-by-body 3PN expression.
    diet = sp.Rational(5, 84) * (
        q1*(9407+8218*x1-2016*x1**2) + q2*(9407+8218*x2-2016*x2**2))
    assert sp.simplify(terms["Q6"]-diet) == 0
    for name in ref.QUAD_TERMS:
        assert sp.simplify(terms[name].subs({ref.CQ1: 1, ref.CQ2: 1})) == 0
        swapped = terms[name].subs({d: -d, c1: c2, c2: c1,
                                    ref.CQ1: ref.CQ2, ref.CQ2: ref.CQ1}, simultaneous=True)
        assert sp.simplify(terms[name]-swapped) == 0
    assert sp.simplify(sp.diff(terms["Q7_cubic"], c1, 3)) != 0


@pytest.mark.parametrize("order,calibration,selected", [
    (0, False, ()), (4, False, ("Q4",)), (5, False, ("Q4",)),
    (6, False, ("Q4", "Q6")), (7, False, ("Q4", "Q6")),
    (8, False, ref.QUAD_TERMS), (-1, False, ("Q4", "Q6")),
    (8, True, ("Q4",)),
])
def test_quadrupole_selector_and_coefficients(backend, order, calibration, selected):
    _, model, array = backend
    assert ref.frontend_terms(order, calibration) == selected
    case = CASES[1]
    actual = model.phase_coefficients(*(array(case[k]) for k in
        ("mass1", "mass2", "chi1", "chi2", "lambda1", "lambda2",
         "quad_monopole1", "quad_monopole2")), spin_order=order,
        tidal_order=13, phenom_d_calibration=calibration)
    expected = ref.coefficients(spin_terms=spin_terms(order, phenom_d_calibration=calibration),
                                tidal_terms=tidal_terms(13), quad_terms=selected)
    vals = values(case, 80.)
    for index, key in ((3, "phi4"), (6, "phi6"), (8, "phi7")):
        np.testing.assert_allclose(float(host(actual[index])),
                                   float(ref.evaluate(expected[key], vals)), rtol=3e-14, atol=3e-12)


@pytest.mark.parametrize("use_ref", [False, True])
def test_quadrupole_waveform_and_phase(backend, use_ref):
    _, model, array = backend
    for case in CASES:
        args = {key: array(value) for key, value in case.items()}
        f = array([20., 80., 160.])
        opts = dict(spin_order=8, tidal_order=13, f_ref=array(40.) if use_ref else None,
                    phasing_convention="reference_frequency" if use_ref else "coalescence")
        phase = model.evaluate_phase(f, *(args[k] for k in ("mass1", "mass2", "chi1", "chi2")),
                                     lambda1=args["lambda1"], lambda2=args["lambda2"],
                                     quad_monopole1=args["quad_monopole1"],
                                     quad_monopole2=args["quad_monopole2"],
                                     tc=args["tc"], phic=args["phic"], **opts)
        hp, hc = model.evaluate(f, **args, f_isco_cutoff=False, **opts)
        expr = expressions(8, use_ref)
        for i, frequency in enumerate((20., 80., 160.)):
            bound = values(case, frequency)
            expected = {name: ref.evaluate(item, bound) for name, item in expr.items()}
            np.testing.assert_allclose(float(host(phase)[i]), float(expected["phase"]), rtol=5e-14, atol=2e-10)
            amplitude = float(expected["amplitude"])
            for actual, name in ((hp, "h_plus"), (hc, "h_cross")):
                assert abs(complex(host(actual)[i])-complex(expected[name]))/amplitude < 5e-10


def test_quadrupole_kerr_zero_and_omitted_defaults(backend):
    _, model, array = backend
    case = CASES[1]
    masses_spins = [array(case[k]) for k in ("mass1", "mass2", "chi1", "chi2")]
    base = model.phase_coefficients(*masses_spins, spin_order=8)
    kerr = model.phase_coefficients(*masses_spins, quad_monopole1=array(1.),
                                    quad_monopole2=array(1.), spin_order=8)
    assert all(np.array_equal(host(a), host(b)) for a, b in zip(base, kerr))
    one = model.phase_coefficients(*masses_spins, quad_monopole1=array(3.), spin_order=8)
    explicit = model.phase_coefficients(*masses_spins, quad_monopole1=array(3.),
                                        quad_monopole2=array(1.), spin_order=8)
    assert all(np.array_equal(host(a), host(b)) for a, b in zip(one, explicit))
    zero_spin = model.phase_coefficients(masses_spins[0], masses_spins[1], array(0.), array(0.),
                                         quad_monopole1=array(5.), quad_monopole2=array(9.), spin_order=8)
    baseline = model.phase_coefficients(masses_spins[0], masses_spins[1], array(0.), array(0.), spin_order=8)
    assert all(np.array_equal(host(a), host(b)) for a, b in zip(zero_spin, baseline))


@pytest.mark.parametrize("use_ref", [False, True])
def test_quadrupole_ad_derivatives(backend, use_ref):
    name, model, array = backend
    case = CASES[1]
    f = array(80.)
    fixed = {k: array(case[k]) for k in ("lambda1", "lambda2", "tc", "phic")}
    opts = dict(spin_order=8, tidal_order=13, f_ref=array(40.) if use_ref else None,
                phasing_convention="reference_frequency" if use_ref else "coalescence")

    def phase(x):
        return model.evaluate_phase(f, x[0], x[1], x[2], x[3],
                                    lambda1=fixed["lambda1"], lambda2=fixed["lambda2"],
                                    quad_monopole1=x[4], quad_monopole2=x[5],
                                    tc=fixed["tc"], phic=fixed["phic"], **opts).reshape(())

    x = array([case[k] for k in ("mass1", "mass2", "chi1", "chi2",
                                 "quad_monopole1", "quad_monopole2")])
    if name == "torch":
        import torch
        actual = host(torch.autograd.functional.jacobian(phase, x))
    else:
        import jax
        actual = host(jax.jacfwd(phase)(x))
    expr = expressions(8, use_ref)["phase"]
    bound = values(case, 80.)
    m1, m2 = exact(case["mass1"]), exact(case["mass2"])
    from diffgw_core.constants import MSUN_SEC
    d_m = ref.derivative_expr(expr, ref.M_SEC)
    d_delta = ref.derivative_expr(expr, ref.DELTA)
    expected_exprs = [exact(MSUN_SEC)*d_m + 2*m2/(m1+m2)**2*d_delta,
                      exact(MSUN_SEC)*d_m - 2*m1/(m1+m2)**2*d_delta,
                      *(ref.derivative_expr(expr, symbol) for symbol in
                        (ref.CHI1, ref.CHI2, ref.CQ1, ref.CQ2))]
    expected = np.array([float(ref.evaluate(item, bound)) for item in expected_exprs])
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=3e-9)


def test_quadrupole_batch_and_compilation(backend):
    name, model, array = backend
    m1, m2 = array([1.4, 1.6]), array([1.4, 1.2])
    chi1, chi2 = array([.2, .3]), array([-.1, .07])
    cq1, cq2 = array([4., 3.]), array([6., 8.])

    def calc(c1, c2):
        return model.phase_coefficients(m1, m2, chi1, chi2,
                                        quad_monopole1=c1, quad_monopole2=c2,
                                        spin_order=8)[3:9:3]

    expected = calc(cq1, cq2)
    if name == "torch":
        import torch
        compiled = torch.compile(calc, backend="aot_eager", fullgraph=True)
    else:
        import jax
        compiled = jax.jit(calc)
    actual = compiled(cq1, cq2)
    assert all(np.allclose(host(a), host(b), rtol=1e-14) for a, b in zip(actual, expected))
    updated = compiled(cq1+1, cq2)
    assert not np.array_equal(host(updated[0]), host(actual[0]))


def test_quadrupole_gradient_at_kerr_baseline(backend):
    name, model, array = backend
    case = CASES[1]
    vals = values(dict(case, quad_monopole1=1., quad_monopole2=1.), 80.)
    expression = expressions(8, False)["phase"]
    expected = float(ref.evaluate(ref.derivative_expr(expression, ref.CQ1), vals))

    def function(cq):
        return model.evaluate_phase(array(80.), array(case["mass1"]), array(case["mass2"]),
                                    array(case["chi1"]), array(case["chi2"]),
                                    lambda1=array(case["lambda1"]), lambda2=array(case["lambda2"]),
                                    quad_monopole1=cq, quad_monopole2=array(1.),
                                    spin_order=8, tidal_order=13).reshape(())

    if name == "torch":
        import torch
        cq = array(1.).requires_grad_()
        actual = torch.autograd.grad(function(cq), cq)[0]
    else:
        import jax
        actual = jax.grad(function)(array(1.))
    np.testing.assert_allclose(float(host(actual)), expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("use_ref", [False, True])
def test_quadrupole_frequency_derivative(backend, use_ref):
    name, model, array = backend
    case = CASES[1]
    frequency = 80.
    args = [array(case[k]) for k in ("mass1", "mass2", "chi1", "chi2")]
    opts = dict(lambda1=array(case["lambda1"]), lambda2=array(case["lambda2"]),
                quad_monopole1=array(case["quad_monopole1"]),
                quad_monopole2=array(case["quad_monopole2"]),
                tc=array(case["tc"]), phic=array(case["phic"]),
                spin_order=8, tidal_order=13,
                f_ref=array(40.) if use_ref else None,
                phasing_convention="reference_frequency" if use_ref else "coalescence")
    if name == "torch":
        actual = model.evaluate_phase_derivative(array(frequency), *args,
                                                  **{key: value for key, value in opts.items() if key != "phic"})
    else:
        import jax
        actual = jax.grad(lambda f: model.evaluate_phase(f, *args, **opts).reshape(()))(
            array(frequency))
    expected_expr = ref.derivative_expr(expressions(8, use_ref)["phase"], ref.F)
    expected = float(ref.evaluate(expected_expr, values(case, frequency)))
    np.testing.assert_allclose(float(host(actual)), expected, rtol=3e-13, atol=2e-11)


def test_quadrupole_jax_generator_dynamic_arguments(backend):
    name, _, array = backend
    if name != "jax":
        return
    from jaxwave.waveforms import WaveformGenerator
    frequencies = array([30., 60., 90.])
    generator = WaveformGenerator("TaylorF2", sample_frequencies=frequencies, use_jit=True,
                                  spin_order=8, tidal_order=13, f_isco_cutoff=False)
    case = CASES[1]
    params = {key: array(value) for key, value in case.items()}
    first = generator.generate_fd_waveform(**params)
    second = generator.generate_fd_waveform(**dict(params, quad_monopole1=array(5.)))
    direct = WaveformGenerator("TaylorF2", sample_frequencies=frequencies, use_jit=False,
                               spin_order=8, tidal_order=13, f_isco_cutoff=False).generate_fd_waveform(
                                   **dict(params, quad_monopole1=array(5.)))
    assert any(np.any(host(a) != host(b)) for a, b in zip(first, second))
    assert all(np.allclose(host(a), host(b), rtol=1e-14) for a, b in zip(second, direct))


def test_quadrupole_compiled_waveform(backend):
    name, model, array = backend
    case = CASES[1]
    f = array([30., 60., 90.])
    fixed = {key: array(case[key]) for key in case if key not in
             ("quad_monopole1", "quad_monopole2")}

    def waveform(cq1, cq2):
        return model.evaluate(f, **fixed, quad_monopole1=cq1, quad_monopole2=cq2,
                              spin_order=8, tidal_order=13, f_isco_cutoff=False)

    if name == "torch":
        import torch
        compiled = torch.compile(waveform, backend="aot_eager", fullgraph=True)
    else:
        import jax
        compiled = jax.jit(waveform)
    c1, c2 = array(case["quad_monopole1"]), array(case["quad_monopole2"])
    actual = compiled(c1, c2)
    expected = waveform(c1, c2)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(host(a), host(b), rtol=3e-12, atol=0)
    changed = compiled(c1+1, c2)
    assert any(np.any(host(a) != host(b)) for a, b in zip(actual, changed))
