"""FP64 verification of adiabatic quadrupolar TaylorF2 tides.

This composes the independently sourced spin reference with electric tides;
it does not qualify spin-tidal couplings, non-Kerr spin quadrupoles, magnetic
tides, dynamical tides, merger/contact physics, or frequency cutoffs.
"""

from functools import lru_cache
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_core.constants import MSUN_SEC
from diffgw_reference import taylorf2_aligned_spin as aligned
from diffgw_reference import taylorf2_tidal as ref
from diffgw_reference.aligned_spin_traceability import frontend_terms as spin_terms
from diffgw_reference.tidal_traceability import frontend_terms as tidal_terms
from diffgw_reference.traceability import PHASE_COEFFICIENT_ORDER
from test_taylorf2_aligned_spin_math_reference import host, reference_values as spin_values
from test_taylorf2_math_reference import backend as backend, exact


CASES = [
    dict(mass1=1.4, mass2=1.4, chi1=.03, chi2=-.02, lambda1=400., lambda2=400.,
         distance=100., inclination=.7, tc=.01, phic=.2),
    dict(mass1=1.6, mass2=1.2, chi1=.1, chi2=-.05, lambda1=200., lambda2=800.,
         distance=200., inclination=1.2, tc=-.03, phic=-.4),
    dict(mass1=7., mass2=1.4, chi1=.5, chi2=.02, lambda1=0., lambda2=500.,
         distance=400., inclination=np.pi/2, tc=0., phic=.7),
]
SELECTORS = [(0, ()), (9, ()), (10, ("T10",)), (11, ("T10",)),
             (12, ("T10", "T12")), (13, ("T10", "T12", "T13")),
             (-1, ("T10", "T12", "T13"))]
COEFFICIENT_KEYS = {key: key for key in PHASE_COEFFICIENT_ORDER[:9]} | {
    "phi10_tidal": "phi10", "phi12_tidal": "phi12", "phi13_tidal": "phi13",
}


def reference_values(case, frequency, f_ref=40.):
    values = spin_values(case, frequency, f_ref)
    values.update({ref.LAMBDA1: exact(case["lambda1"]), ref.LAMBDA2: exact(case["lambda2"])})
    return values


@lru_cache(maxsize=None)
def expressions(order, use_ref, spin_order=8):
    return ref.waveform_expr(reference_frequency=use_ref, spin_terms=spin_terms(spin_order),
                             tidal_terms=tidal_terms(order))


def check_sample(actual, case, frequency, *, order=13, use_ref=False, f_ref=40., spin_order=8):
    values = reference_values(case, frequency, f_ref)
    expected = {key: ref.evaluate(expr, values, dps=80)
                for key, expr in expressions(order, use_ref, spin_order).items()}
    amplitude = float(expected["amplitude"])
    np.testing.assert_allclose(float(actual[0])/amplitude, 1, rtol=5e-14, atol=0)
    np.testing.assert_allclose(float(actual[1]), float(expected["phase"]), rtol=5e-14, atol=2e-10)
    for value, key in zip(actual[2:], ("h_plus", "h_cross")):
        assert abs(complex(value)-complex(expected[key]))/amplitude < 3e-10


def actual_waveform(backend, case, frequencies, *, order=13, use_ref=False, f_ref=40., compiled=False):
    name, model, array = backend
    args = {key: array(value) if value is not None else None for key, value in case.items()}
    opts = dict(spin_order=8, f_ref=array(f_ref) if use_ref else None,
                phasing_convention="reference_frequency" if use_ref else "coalescence")
    if order is not None:
        opts["tidal_order"] = order

    def output(f, parameters):
        amplitude = model.evaluate_amplitude(f, parameters["mass1"], parameters["mass2"], parameters["distance"])
        phase = model.evaluate_phase(f, *(parameters[key] for key in ("mass1", "mass2", "chi1", "chi2")),
                                     lambda1=parameters["lambda1"], lambda2=parameters["lambda2"],
                                     tc=parameters["tc"], phic=parameters["phic"], **opts)
        hp, hc = model.evaluate(f, **parameters, f_isco_cutoff=False, **opts)
        return amplitude, phase, hp, hc

    if compiled:
        if name == "torch":
            import torch
            output = torch.compile(output, backend="aot_eager", fullgraph=True)
        else:
            import jax
            output = jax.jit(output)
    f = array(frequencies)
    result = output(f, args)
    if name == "torch":
        import torch
        assert result[0].dtype == result[1].dtype == torch.float64
        assert result[2].dtype == result[3].dtype == torch.complex128
        assert all(value.device == f.device for value in result)
    else:
        assert result[0].dtype == result[1].dtype == np.float64
        assert result[2].dtype == result[3].dtype == np.complex128
        assert all(value.devices() == f.devices() for value in result)
    return tuple(host(value) for value in result)


@pytest.mark.parametrize("order,terms", SELECTORS)
def test_tidal_selector_contract(order, terms):
    assert tidal_terms(order) == terms
    for present in ((False, True), (True, False), (False, False)):
        assert tidal_terms(order, lambda1_present=present[0], lambda2_present=present[1]) == ()


def test_tidal_combinations_against_reference(backend):
    name, _, array = backend
    conversion = importlib.import_module(f"{name}wave.conversions").tidal_combinations
    for case in CASES + [dict(CASES[1], lambda1=0., lambda2=0.)]:
        actual = conversion(*(array(case[key]) for key in ("mass1", "mass2", "lambda1", "lambda2")))
        values = reference_values(case, 40.)
        for value, key in zip(actual, ("lambda_tilde", "delta_lambda_tilde")):
            expected = ref.evaluate(ref.tidal_parameters()[key], values)
            # The antisymmetric combination can cancel; its absolute budget
            # accounts for roundoff in lambda values of order 10^3.
            np.testing.assert_allclose(float(value), float(expected), rtol=2e-14, atol=2e-12)


@pytest.mark.parametrize("order,terms", SELECTORS)
def test_tidal_coefficients_against_reference(backend, order, terms):
    _, model, array = backend
    for case in CASES:
        for spin_order in (0, 8):
            actual = model.phase_coefficients(*(array(case[key]) for key in
                                                ("mass1", "mass2", "chi1", "chi2", "lambda1", "lambda2")),
                                              spin_order=spin_order, tidal_order=order)
            assert len(actual) == len(PHASE_COEFFICIENT_ORDER)
            expected = ref.coefficients(spin_terms=spin_terms(spin_order), tidal_terms=terms)
            values = reference_values(case, 40.)
            for index, key in enumerate(PHASE_COEFFICIENT_ORDER):
                np.testing.assert_allclose(float(actual[index]), float(ref.evaluate(expected[COEFFICIENT_KEYS[key]], values)),
                                           rtol=2e-14, atol=2e-13, err_msg=f"{key}, {case}, order={order}")


@pytest.mark.parametrize("order", [10, 12, 13])
@pytest.mark.parametrize("use_ref", [False, True])
def test_tidal_values_against_reference(backend, order, use_ref):
    frequencies = [20., 80., 300.]
    for case in CASES:
        actual = actual_waveform(backend, case, frequencies, order=order, use_ref=use_ref)
        for index, frequency in enumerate(frequencies):
            check_sample(tuple(value[index] for value in actual), case, frequency, order=order, use_ref=use_ref)
    # A true scalar frequency exercises a different frontend broadcasting path.
    actual = actual_waveform(backend, CASES[0], 80., order=order, use_ref=use_ref)
    check_sample(tuple(value.item() for value in actual), CASES[0], 80., order=order, use_ref=use_ref)
    neighbors = [np.nextafter(40., 0.), 40., np.nextafter(40., np.inf)]
    actual = actual_waveform(backend, CASES[1], neighbors, order=order, use_ref=use_ref)
    for index, frequency in enumerate(neighbors):
        check_sample(tuple(value[index] for value in actual), CASES[1], frequency, order=order, use_ref=use_ref)


@pytest.mark.parametrize("use_ref", [False, True])
def test_tidal_default_and_missing_arguments(backend, use_ref):
    _, model, array = backend
    case = CASES[1]
    actual = actual_waveform(backend, case, [80.], order=None, use_ref=use_ref)
    check_sample(tuple(value[0] for value in actual), case, 80., order=12, use_ref=use_ref)
    negative = actual_waveform(backend, case, [80.], order=-1, use_ref=use_ref)
    check_sample(tuple(value[0] for value in negative), case, 80., order=13, use_ref=use_ref)
    coefficients = model.phase_coefficients(*(array(case[key]) for key in
                                              ("mass1", "mass2", "chi1", "chi2", "lambda1", "lambda2")))
    expected = ref.coefficients(spin_terms=spin_terms(7), tidal_terms=("T10", "T12"))
    for index, key in enumerate(PHASE_COEFFICIENT_ORDER):
        np.testing.assert_allclose(float(coefficients[index]),
                                   float(ref.evaluate(expected[COEFFICIENT_KEYS[key]], reference_values(case, 80.))),
                                   rtol=2e-14, atol=2e-13)
    for lambda1, lambda2 in ((None, 800.), (200., None), (None, None), (0., 0.)):
        omitted = dict(case, lambda1=lambda1, lambda2=lambda2)
        actual = actual_waveform(backend, omitted, [80.], order=13, use_ref=use_ref)
        check_sample(tuple(value[0] for value in actual), dict(case, lambda1=0., lambda2=0.),
                     80., order=13, use_ref=use_ref)
        coefficients = model.phase_coefficients(array(case["mass1"]), array(case["mass2"]),
                                                 array(case["chi1"]), array(case["chi2"]),
                                                 lambda1=array(lambda1) if lambda1 is not None else None,
                                                 lambda2=array(lambda2) if lambda2 is not None else None,
                                                 tidal_order=13)
        assert all(float(value) == 0 for value in coefficients[-3:])


@lru_cache(maxsize=None)
def derivatives(use_ref):
    return {key: {symbol: ref.derivative_expr(expr, symbol)
                  for symbol in (ref.M_SEC, ref.DELTA, ref.LAMBDA1, ref.LAMBDA2)}
            for key, expr in expressions(13, use_ref).items()}


def jacobian_function(backend, case, frequency, use_ref):
    name, model, array = backend
    f = array(frequency)
    fixed = {key: array(case[key]) for key in ("chi1", "chi2", "distance", "inclination", "tc", "phic")}
    opts = dict(spin_order=8, tidal_order=13, f_ref=array(40.) if use_ref else None,
                phasing_convention="reference_frequency" if use_ref else "coalescence")

    def output(x):
        m1, m2, lambda1, lambda2 = x
        amplitude = model.evaluate_amplitude(f, m1, m2, fixed["distance"])
        phase = model.evaluate_phase(f, m1, m2, fixed["chi1"], fixed["chi2"], lambda1=lambda1, lambda2=lambda2,
                                     tc=fixed["tc"], phic=fixed["phic"], **opts)
        hp, hc = model.evaluate(f, m1, m2, lambda1=lambda1, lambda2=lambda2, **fixed,
                                **opts, f_isco_cutoff=False)
        rows = tuple(value.reshape(()) for value in (amplitude, phase, hp.real, hp.imag, hc.real, hc.imag))
        if name == "torch":
            import torch
            return torch.stack(rows)
        import jax.numpy as jnp
        return jnp.stack(rows)

    return output


def check_jacobian(actual, case, frequency, use_ref):
    values = reference_values(case, frequency)
    m1, m2 = exact(case["mass1"]), exact(case["mass2"])
    expected = []
    for key, partials in derivatives(use_ref).items():
        # Signed delta keeps both body labels and remains smooth at equal mass.
        expressions_dm = [
            exact(MSUN_SEC)*partials[ref.M_SEC] + 2*m2/(m1+m2)**2*partials[ref.DELTA],
            exact(MSUN_SEC)*partials[ref.M_SEC] - 2*m1/(m1+m2)**2*partials[ref.DELTA],
            partials[ref.LAMBDA1], partials[ref.LAMBDA2],
        ]
        row = np.array([complex(ref.evaluate(expr, values)) for expr in expressions_dm])
        expected.extend([row.real, row.imag] if key in ("h_plus", "h_cross") else [row.real])
    amplitude = float(ref.evaluate(expressions(13, use_ref)["amplitude"], values))
    scales = np.array([amplitude, 1., amplitude, amplitude, amplitude, amplitude])[:, None]
    # Lambda derivatives are small in single-unit coordinates. Compare changes
    # per 1000 units (a fixed scale, including at lambda=0), making the absolute
    # budget 2e-12 per lambda after amplitude normalization, not 2e-9.
    parameter_scales = np.array([1., 1., 1000., 1000.])[None, :]
    np.testing.assert_allclose(actual/scales*parameter_scales, np.array(expected)/scales*parameter_scales,
                               rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize("case,frequency", [
    (dict(CASES[0], lambda1=0., lambda2=0.), 80.),
    (CASES[1], np.nextafter(40., np.inf)),
    (CASES[2], 80.),
    (dict(CASES[2], mass1=1.4, mass2=7., chi1=.02, chi2=.5, lambda1=500., lambda2=0.), 80.),
])
@pytest.mark.parametrize("use_ref", [False, True])
def test_tidal_parameter_jacobian(backend, case, frequency, use_ref):
    name, _, array = backend
    function = jacobian_function(backend, case, frequency, use_ref)
    x = array([case[key] for key in ("mass1", "mass2", "lambda1", "lambda2")])
    if name == "torch":
        import torch
        actual = torch.autograd.functional.jacobian(function, x)
    else:
        import jax
        actual = jax.jacfwd(function)(x)
    check_jacobian(host(actual), case, frequency, use_ref)


@pytest.mark.parametrize("use_ref", [False, True])
def test_tidal_frequency_derivative(backend, use_ref):
    name, model, array = backend
    case, frequencies = CASES[1], [20., 80., 300.]
    args = [array(case[key]) for key in ("mass1", "mass2", "chi1", "chi2", "lambda1", "lambda2")]
    opts = dict(tc=array(case["tc"]), spin_order=8, tidal_order=13, f_ref=array(40.) if use_ref else None,
                phasing_convention="reference_frequency" if use_ref else "coalescence")
    if name == "torch":
        actual = host(model.evaluate_phase_derivative(array(frequencies), *args, **opts))
    else:
        import jax
        jacobian = host(jax.jacfwd(lambda f: model.evaluate_phase(f, *args, **opts))(array(frequencies)))
        actual = np.diag(jacobian)
        np.testing.assert_array_equal(jacobian-np.diag(actual), 0)
    derivative = ref.derivative_expr(expressions(13, use_ref)["phase"], ref.F)
    expected = [float(ref.evaluate(derivative, reference_values(case, f))) for f in frequencies]
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-12)


def test_tidal_zero_reduction_and_exchange():
    zero = {ref.LAMBDA1: 0, ref.LAMBDA2: 0}
    exchange = {ref.DELTA: -ref.DELTA, ref.LAMBDA1: ref.LAMBDA2, ref.LAMBDA2: ref.LAMBDA1}
    direct, effective = ref.tidal_corrections(), ref.effective_tidal_corrections()
    assert direct.keys() == effective.keys()
    assert sp.expand(direct["T13"]+sp.pi*direct["T10"]) == 0
    for key, expression in direct.items():
        assert not expression.atoms(sp.Float)
        assert sp.expand(expression-effective[key]) == 0
        assert expression.subs(zero) == 0
        assert sp.Poly(expression, ref.LAMBDA1, ref.LAMBDA2).total_degree() == 1
        assert sp.expand(expression.subs(exchange, simultaneous=True)-expression) == 0
    parameters = ref.tidal_parameters()
    assert sp.expand(parameters["lambda_tilde"].subs(ref.DELTA, 0)-(ref.LAMBDA1+ref.LAMBDA2)/2) == 0
    assert sp.expand(parameters["lambda_tilde"].subs(exchange, simultaneous=True)-parameters["lambda_tilde"]) == 0
    assert sp.expand(parameters["delta_lambda_tilde"].subs(exchange, simultaneous=True)
                     + parameters["delta_lambda_tilde"]) == 0
    for use_ref in (False, True):
        baseline = aligned.waveform_expr(reference_frequency=use_ref)
        for key, expression in expressions(13, use_ref).items():
            assert sp.simplify(expression.subs(zero)-baseline[key]) == 0
            no_tides = ref.waveform_expr(reference_frequency=use_ref, tidal_terms=())[key]
            assert sp.simplify(no_tides-baseline[key]) == 0


def test_tidal_label_exchange(backend):
    for case in CASES:
        swapped = dict(case, mass1=case["mass2"], mass2=case["mass1"], chi1=case["chi2"], chi2=case["chi1"],
                       lambda1=case["lambda2"], lambda2=case["lambda1"])
        for use_ref in (False, True):
            actual = actual_waveform(backend, case, [80.], use_ref=use_ref)
            exchange = actual_waveform(backend, swapped, [80.], use_ref=use_ref)
            for original, permuted in zip(actual, exchange):
                np.testing.assert_allclose(permuted, original, rtol=2e-14, atol=0)
            check_sample(tuple(value[0] for value in exchange), swapped, 80., use_ref=use_ref)


@pytest.mark.parametrize("use_ref", [False, True])
def test_tidal_seeded_parameter_sweep(backend, use_ref):
    rng = np.random.default_rng(20260924)
    for index in range(10):
        # Alternate BNS and BH-NS inspired parameters, without asserting EOS or
        # non-Kerr spin-multipole consistency. Both f and f_ref have v<.30.
        bhns = bool(index % 2)
        case = dict(mass1=rng.uniform(3., 12.) if bhns else rng.uniform(1., 2.2),
                    mass2=rng.uniform(1., 2.2), chi1=rng.uniform(-.5, .5), chi2=rng.uniform(-.1, .1),
                    lambda1=0. if bhns else rng.uniform(0., 2000.), lambda2=rng.uniform(0., 2000.),
                    distance=rng.uniform(10., 500.), inclination=rng.uniform(0., np.pi),
                    tc=rng.uniform(-.05, .05), phic=rng.uniform(-np.pi, np.pi))
        m_sec = (case["mass1"]+case["mass2"])*MSUN_SEC
        frequency, f_ref = rng.uniform(.12, .30, size=2)**3/(np.pi*m_sec)
        order = (10, 12, 13)[index % 3]
        actual = actual_waveform(backend, case, [frequency], order=order, use_ref=use_ref, f_ref=f_ref)
        check_sample(tuple(value[0] for value in actual), case, frequency, order=order, use_ref=use_ref, f_ref=f_ref)


@pytest.mark.parametrize("grid_kind", ["shared", "per_binary", "per_binary_many"])
@pytest.mark.parametrize("use_ref", [False, True])
def test_tidal_batch_and_compilation(backend, grid_kind, use_ref):
    cases = [CASES[0], CASES[2]]
    batch = {key: [case[key] for case in cases] for key in cases[0]}
    grid = {"shared": [30., 80.], "per_binary": [[30.], [80.]],
            "per_binary_many": [[30., 80.], [50., 150.]]}[grid_kind]
    references = [35., 45.]
    actual = actual_waveform(backend, batch, grid, use_ref=use_ref, f_ref=references, compiled=True)
    count = 1 if grid_kind == "per_binary" else 2
    assert all(value.shape == (2, count) for value in actual)
    for row, case in enumerate(cases):
        for column, frequency in enumerate(grid if grid_kind == "shared" else grid[row]):
            check_sample(tuple(value[row, column] for value in actual), case, frequency,
                         use_ref=use_ref, f_ref=references[row])


def test_tidal_transformed_jacobian(backend):
    name, _, array = backend
    cases = [CASES[0], dict(CASES[0], mass1=1.6, mass2=1.2, lambda1=0., lambda2=800.)]
    function = jacobian_function(backend, cases[0], 80., True)
    x = array([[case[key] for key in ("mass1", "mass2", "lambda1", "lambda2")] for case in cases])
    if name == "torch":
        import torch
        transformed = torch.compile(torch.func.vmap(torch.func.jacrev(function)), backend="aot_eager", fullgraph=True)
    else:
        import jax
        transformed = jax.jit(jax.vmap(jax.jacfwd(function)))
    actual = host(transformed(x))
    assert actual.shape == (2, 6, 4)
    for row, case in enumerate(cases):
        check_jacobian(actual[row], case, 80., True)


def test_tidal_reference_domains_and_convergence():
    values = reference_values(CASES[1], "40.0000000000000000000001")
    for expression in expressions(13, True).values():
        low, high = ref.evaluate(expression, values, dps=100), ref.evaluate(expression, values, dps=160)
        assert abs(low-high) <= abs(high)*sp.Rational(1, 10**65)
    for symbol, value in ((ref.LAMBDA1, -1), (ref.LAMBDA2, "nan"), (ref.LAMBDA1, "inf")):
        invalid = dict(values)
        invalid[symbol] = value
        with pytest.raises(ValueError):
            ref.evaluate(expressions(13, True)["h_plus"], invalid)


def test_tidal_reference_artifacts_are_current():
    from diffgw_reference.tidal_artifacts import coefficient_table, fixtures

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/math/taylorf2-tidal-coefficients.md").read_text() == coefficient_table()
    stored = json.loads((root / "tests/data/taylorf2_tidal_reference_v1.json").read_text())
    # One regeneration checks 100/160-digit convergence and all frozen fields;
    # snapshots detect drift but are not a second independent mathematical oracle.
    assert stored == json.loads(json.dumps(fixtures()))
