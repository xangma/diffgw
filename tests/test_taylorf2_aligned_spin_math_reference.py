"""Independent aligned-spin TaylorF2 qualification in FP64.

The test adapter translates legacy frontend selectors into literature term sets;
those selectors are not interpreted as literal PN truncation orders. The kernel
and its numerical evaluator do not import either frontend. Unit provenance and
nonspinning FP32 qualification remain in their separate maintained suites.
"""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_core.constants import MSUN_SEC
from diffgw_reference import taylorf2 as nonspinning
from diffgw_reference import taylorf2_aligned_spin as ref
from diffgw_reference.aligned_spin_traceability import frontend_terms
from diffgw_reference.traceability import PHASE_COEFFICIENT_ORDER
from test_taylorf2_math_reference import (
    backend as backend,
    exact,
    reference_values as nonspinning_values,
)


CASES = [
    dict(mass1=30.0, mass2=30.0, chi1=0.6, chi2=0.6,
         distance=400.0, inclination=0.0, tc=0.01, phic=0.2),
    dict(mass1=1.4, mass2=1.3, chi1=0.3, chi2=-0.2,
         distance=100.0, inclination=1.2, tc=-0.03, phic=-0.4),
    dict(mass1=50.0, mass2=5.0, chi1=-0.8, chi2=0.7,
         distance=750.0, inclination=np.pi/2, tc=0.0, phic=0.7),
]
FREQUENCIES = [20.0, 31.25, 80.0, 160.0]
SELECTORS = [
    (0, False, ()),
    (2, False, ()),
    (3, False, ("SO3",)),
    (4, False, ("SO3", "SS4")),
    (5, False, ("SO3", "SS4", "SO5", "SO6", "SO7")),
    (6, False, ("SO3", "SS4", "SO5", "SO6", "SS6", "SO7")),
    (7, False, ("SO3", "SS4", "SO5", "SO6", "SS6", "SO7")),
    (8, False, ("SO3", "SS4", "SO5", "SO6", "SS6", "SO7", "SS7", "SSS7")),
    (-1, False, ("SO3", "SS4", "SO5", "SO6", "SS6", "SO7")),
    (8, True, ("SO3", "SS4", "SO5", "SO6", "SO7")),
]


def host(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def reference_values(case, frequency, f_ref=40.0):
    values = nonspinning_values(case, frequency, f_ref)
    del values[nonspinning.ETA]
    m1, m2 = exact(case["mass1"]), exact(case["mass2"])
    values.update({ref.DELTA: (m1-m2)/(m1+m2), ref.CHI1: exact(case["chi1"]),
                   ref.CHI2: exact(case["chi2"])})
    return values


@lru_cache(maxsize=None)
def expressions(spin_order, use_ref):
    return ref.waveform_expr(reference_frequency=use_ref, terms=frontend_terms(spin_order))


def expected_waveform(case, frequency, *, spin_order, use_ref, f_ref=40.0):
    values = reference_values(case, frequency, f_ref)
    return {key: ref.evaluate(expr, values, dps=80)
            for key, expr in expressions(spin_order, use_ref).items()}


def check_sample(actual, case, frequency, *, spin_order, use_ref, f_ref=40.0):
    expected = expected_waveform(case, frequency, spin_order=spin_order, use_ref=use_ref, f_ref=f_ref)
    amplitude = float(expected["amplitude"])
    np.testing.assert_allclose(float(actual[0])/amplitude, 1, rtol=5e-14, atol=0)
    np.testing.assert_allclose(float(actual[1]), float(expected["phase"]), rtol=5e-14, atol=2e-10)
    for value, key in zip(actual[2:], ("h_plus", "h_cross")):
        assert abs(complex(value)-complex(expected[key]))/amplitude < 3e-10


def actual_waveform(backend, case, frequencies, *, spin_order, use_ref, f_ref=40.0, compiled=False):
    name, model, array = backend
    args = {key: array(value) for key, value in case.items()}
    opts = dict(f_ref=array(f_ref) if use_ref else None, spin_order=spin_order,
                phasing_convention="reference_frequency" if use_ref else "coalescence")

    def output(f, parameters):
        amplitude = model.evaluate_amplitude(f, parameters["mass1"], parameters["mass2"], parameters["distance"])
        phase = model.evaluate_phase(f, parameters["mass1"], parameters["mass2"], parameters["chi1"],
                                     parameters["chi2"], tc=parameters["tc"], phic=parameters["phic"], **opts)
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


@pytest.mark.parametrize("spin_order,calibration,terms", SELECTORS)
def test_aligned_spin_selector_contract(spin_order, calibration, terms):
    # Literal expectations are independent of the adapter implementation.
    assert frontend_terms(spin_order, phenom_d_calibration=calibration) == terms


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("spin_order,calibration,terms", SELECTORS)
def test_aligned_spin_coefficients_against_reference(backend, case, spin_order, calibration, terms):
    _, model, array = backend
    actual = model.phase_coefficients(*(array(case[key]) for key in ("mass1", "mass2", "chi1", "chi2")),
                                      spin_order=spin_order, phenom_d_calibration=calibration)
    assert len(actual) == len(PHASE_COEFFICIENT_ORDER)
    expected = ref.coefficients(terms=terms)
    values = reference_values(case, 40.0)
    for index, key in enumerate(PHASE_COEFFICIENT_ORDER):
        if key in expected:
            np.testing.assert_allclose(float(actual[index]), float(ref.evaluate(expected[key], values)),
                                       rtol=2e-14, atol=2e-13, err_msg=f"{key}, {terms}, {case}")
        else:
            assert float(actual[index]) == 0, key


@pytest.mark.parametrize("case", CASES)
def test_aligned_spin_corrections_by_spin_parity(backend, case):
    _, model, array = backend

    def coefficients(scale):
        result = model.phase_coefficients(array(case["mass1"]), array(case["mass2"]),
                                          array(scale*case["chi1"]), array(scale*case["chi2"]), spin_order=8)
        return np.array([float(value) for value in result[:9]])

    positive, negative, half, minus_half, zero = [coefficients(scale) for scale in (1, -1, 0.5, -0.5, 0)]
    odd = (positive-negative)/2
    cubic = (4/3)*(odd-(half-minus_half))
    quadratic = (positive+negative)/2-zero
    linear = odd-cubic
    # Scaling and sign reversal isolate spin degree independently of production
    # internal names; even the 3.5PN quadratic and cubic pieces are checked apart.
    correction_groups = {
        "linear": {"phi3": "SO3", "phi5": "SO5", "phi5_log": "SO5_log", "phi6": "SO6", "phi7": "SO7"},
        "quadratic": {"phi4": "SS4", "phi6": "SS6", "phi7": "SS7"},
        "cubic": {"phi7": "SSS7"},
    }
    values = reference_values(case, 40.0)
    corrections = ref.spin_corrections()
    for group, actual in (("linear", linear), ("quadratic", quadratic), ("cubic", cubic)):
        expected = np.zeros(9)
        for key, term in correction_groups[group].items():
            expected[PHASE_COEFFICIENT_ORDER.index(key)] = float(ref.evaluate(corrections[term], values))
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-12, err_msg=group)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("spin_order", [5, 7, 8, -1])
@pytest.mark.parametrize("use_ref", [False, True])
def test_aligned_spin_values_against_reference(backend, case, spin_order, use_ref):
    actual = actual_waveform(backend, case, FREQUENCIES, spin_order=spin_order, use_ref=use_ref)
    for idx, frequency in enumerate(FREQUENCIES):
        check_sample(tuple(value[idx] for value in actual), case, frequency,
                     spin_order=spin_order, use_ref=use_ref)


@pytest.mark.parametrize("use_ref", [False, True])
def test_aligned_spin_default_profiles(backend, use_ref):
    _, model, array = backend
    case = CASES[0]
    args = {key: array(value) for key, value in case.items()}
    f = array([31.25])
    opts = dict(f_ref=array(40.0) if use_ref else None,
                phasing_convention="reference_frequency" if use_ref else "coalescence")
    # Historical low-level phase default5 and public waveform default7 differ.
    # Qualify each against its documented terms rather than hiding that distinction.
    phase = model.evaluate_phase(f, *(args[key] for key in ("mass1", "mass2", "chi1", "chi2")),
                                 tc=args["tc"], phic=args["phic"], **opts)
    phase_reference = expected_waveform(case, 31.25, spin_order=5, use_ref=use_ref)
    np.testing.assert_allclose(float(phase[0]), float(phase_reference["phase"]), rtol=5e-14, atol=2e-10)
    hp, hc = model.evaluate(f, **args, f_isco_cutoff=False, **opts)
    public_reference = expected_waveform(case, 31.25, spin_order=7, use_ref=use_ref)
    amplitude = float(public_reference["amplitude"])
    for actual, key in ((hp[0], "h_plus"), (hc[0], "h_cross")):
        assert abs(complex(actual)-complex(public_reference[key]))/amplitude < 3e-10
    assert abs(float(public_reference["phase"]-phase_reference["phase"])) > 1e-3


@pytest.mark.parametrize("use_ref", [False, True])
def test_aligned_spin_calibration_phase(backend, use_ref):
    _, model, array = backend
    case = CASES[2]
    values = reference_values(case, 31.25)
    terms = frontend_terms(8, phenom_d_calibration=True)
    expected = ref.evaluate(ref.waveform_expr(reference_frequency=use_ref, terms=terms)["phase"], values)
    phase = model.evaluate_phase(array([31.25]), *(array(case[key]) for key in ("mass1", "mass2", "chi1", "chi2")),
                                 tc=array(case["tc"]), phic=array(case["phic"]), spin_order=8,
                                 phenom_d_calibration=True, f_ref=array(40.) if use_ref else None,
                                 phasing_convention="reference_frequency" if use_ref else "coalescence")
    np.testing.assert_allclose(float(phase[0]), float(expected), rtol=5e-14, atol=2e-10)


@lru_cache(maxsize=None)
def waveform_derivatives(spin_order, use_ref):
    return {key: {symbol: ref.derivative_expr(expr, symbol)
                  for symbol in (ref.M_SEC, ref.DELTA, ref.CHI1, ref.CHI2)}
            for key, expr in expressions(spin_order, use_ref).items()}


@pytest.mark.parametrize("case,frequency", [
    (dict(CASES[0], chi1=0.0, chi2=0.0), 31.25),
    (dict(CASES[2], chi1=0.0, chi2=0.0), 31.25),
    (CASES[1], np.nextafter(40.0, np.inf)),
    (CASES[2], 31.25),
])
@pytest.mark.parametrize("spin_order", [7, 8])
@pytest.mark.parametrize("use_ref", [False, True])
def test_aligned_spin_parameter_jacobian(backend, case, frequency, spin_order, use_ref):
    name, model, array = backend
    f = array(frequency)
    opts = dict(f_ref=array(40.0) if use_ref else None, spin_order=spin_order,
                phasing_convention="reference_frequency" if use_ref else "coalescence")
    args = {key: array(case[key]) for key in ("distance", "inclination", "tc", "phic")}

    def output(x):
        mass1, mass2, chi1, chi2 = x
        amplitude = model.evaluate_amplitude(f, mass1, mass2, args["distance"])
        phase = model.evaluate_phase(f, mass1, mass2, chi1, chi2, tc=args["tc"], phic=args["phic"], **opts)
        hp, hc = model.evaluate(f, mass1, mass2, chi1, chi2, **args, **opts, f_isco_cutoff=False)
        rows = tuple(value.reshape(()) for value in (amplitude, phase, hp.real, hp.imag, hc.real, hc.imag))
        if name == "torch":
            import torch
            return torch.stack(rows)
        import jax.numpy as jnp
        return jnp.stack(rows)

    x = array([case[key] for key in ("mass1", "mass2", "chi1", "chi2")])
    if name == "torch":
        import torch
        actual = host(torch.autograd.functional.jacobian(output, x))
    else:
        import jax
        actual = host(jax.jacfwd(output)(x))
    values = reference_values(case, frequency)
    m1, m2 = exact(case["mass1"]), exact(case["mass2"])
    expected = []
    for key in ("amplitude", "phase", "h_plus", "h_cross"):
        partials = waveform_derivatives(spin_order, use_ref)[key]
        # eta is already derived from signed delta in the kernel, so the mass
        # chain rule includes only total mass and delta, not an extra eta term.
        derivatives = [
            exact(MSUN_SEC)*partials[ref.M_SEC] + 2*m2/(m1+m2)**2*partials[ref.DELTA],
            exact(MSUN_SEC)*partials[ref.M_SEC] - 2*m1/(m1+m2)**2*partials[ref.DELTA],
            partials[ref.CHI1], partials[ref.CHI2],
        ]
        row = np.array([complex(ref.evaluate(expr, values)) for expr in derivatives])
        expected.extend([row.real, row.imag] if key in ("h_plus", "h_cross") else [row.real])
    amplitude = float(ref.evaluate(expressions(spin_order, use_ref)["amplitude"], values))
    scales = np.array([amplitude, 1, amplitude, amplitude, amplitude, amplitude])[:, None]
    np.testing.assert_allclose(actual/scales, np.array(expected)/scales, rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize("spin_order", [5, 7, 8, -1])
def test_aligned_spin_label_exchange(backend, spin_order):
    for case in CASES:
        swapped = dict(case, mass1=case["mass2"], mass2=case["mass1"], chi1=case["chi2"], chi2=case["chi1"])
        actual = actual_waveform(backend, case, FREQUENCIES, spin_order=spin_order, use_ref=True)
        exchange = actual_waveform(backend, swapped, FREQUENCIES, spin_order=spin_order, use_ref=True)
        for original, permuted in zip(actual, exchange):
            np.testing.assert_allclose(permuted, original, rtol=2e-14, atol=0)
        for idx, frequency in enumerate(FREQUENCIES):
            check_sample(tuple(value[idx] for value in exchange), swapped, frequency, spin_order=spin_order, use_ref=True)


def test_aligned_spin_zero_spin_reduction():
    baseline = nonspinning.coefficients(eta=ref.ETA_EXPR)
    for spin_order in (0, 5, 7, 8, -1):
        terms = frontend_terms(spin_order)
        for key, expression in ref.coefficients(terms=terms).items():
            assert not expression.atoms(sp.Float)
            assert sp.simplify(expression.subs({ref.CHI1: 0, ref.CHI2: 0})-baseline[key]) == 0
        for use_ref in (False, True):
            for key, expression in expressions(spin_order, use_ref).items():
                zero = expression.subs({ref.CHI1: 0, ref.CHI2: 0})
                expected = nonspinning.waveform_expr(reference_frequency=use_ref)[key].subs(nonspinning.ETA, ref.ETA_EXPR)
                assert sp.simplify(zero-expected) == 0, (spin_order, key)
    # The signed coordinate retains body labels; simultaneous exchange is exact.
    exchange = {ref.DELTA: -ref.DELTA, ref.CHI1: ref.CHI2, ref.CHI2: ref.CHI1}
    for expression in ref.spin_corrections().values():
        assert sp.simplify(expression.subs(exchange, simultaneous=True)-expression) == 0
    for case in CASES:
        zero_case = dict(case, chi1=0.0, chi2=0.0)
        for use_ref in (False, True):
            spun = expected_waveform(zero_case, 31.25, spin_order=8, use_ref=use_ref)
            values = nonspinning_values(case, 31.25)
            for key, expression in nonspinning.waveform_expr(reference_frequency=use_ref).items():
                baseline_value = nonspinning.evaluate(expression, values)
                assert abs(spun[key]-baseline_value) <= abs(baseline_value)*sp.Rational(1, 10**65)


@pytest.mark.parametrize("spin_order", [5, 7, 8])
def test_aligned_spin_zero_spin_backend_values(backend, spin_order):
    for original in (CASES[0], CASES[2]):
        case = dict(original, chi1=0.0, chi2=0.0)
        for use_ref in (False, True):
            actual = actual_waveform(backend, case, [31.25], spin_order=spin_order, use_ref=use_ref)
            expected = {key: nonspinning.evaluate(expr, nonspinning_values(case, 31.25))
                        for key, expr in nonspinning.waveform_expr(reference_frequency=use_ref).items()}
            amplitude = float(expected["amplitude"])
            np.testing.assert_allclose(float(actual[0][0])/amplitude, 1, rtol=5e-14, atol=0)
            np.testing.assert_allclose(float(actual[1][0]), float(expected["phase"]), rtol=5e-14, atol=2e-10)
            for value, key in zip(actual[2:], ("h_plus", "h_cross")):
                assert abs(complex(value[0])-complex(expected[key]))/amplitude < 3e-10


@pytest.mark.parametrize("use_ref", [False, True])
def test_aligned_spin_seeded_parameter_sweep(backend, use_ref):
    # Fixed sequence, M in [2.7,100] Msun, q in [1,20], signed spins [-.95,.95].
    # Frequencies stay positive and below v=.30; no cutoff/clamp/tides are tested.
    rng = np.random.default_rng(20260923)
    for idx in range(12):
        total_mass = np.exp(rng.uniform(np.log(2.7), np.log(100.0)))
        ratio = np.exp(rng.uniform(0, np.log(20.0)))
        case = dict(mass1=total_mass*ratio/(1+ratio), mass2=total_mass/(1+ratio),
                    chi1=rng.uniform(-.95, .95), chi2=rng.uniform(-.95, .95),
                    distance=np.exp(rng.uniform(np.log(10.), np.log(2000.))),
                    inclination=rng.uniform(0, np.pi), tc=rng.uniform(-.1, .1), phic=rng.uniform(-np.pi, np.pi))
        frequency = rng.uniform(.12, .30)**3/(np.pi*total_mass*MSUN_SEC)
        f_ref = rng.uniform(.12, .30)**3/(np.pi*total_mass*MSUN_SEC)
        spin_order = (5, 7, 8)[idx % 3]
        actual = actual_waveform(backend, case, [frequency], spin_order=spin_order, use_ref=use_ref, f_ref=f_ref)
        check_sample(tuple(value[0] for value in actual), case, frequency,
                     spin_order=spin_order, use_ref=use_ref, f_ref=f_ref)


@pytest.mark.parametrize("use_ref", [False, True])
@pytest.mark.parametrize("spin_order", [7, 8])
@pytest.mark.parametrize("per_binary", [False, True])
def test_aligned_spin_batch_and_compilation(backend, use_ref, spin_order, per_binary):
    # Equal batch/frequency axis lengths are intentional, guarding the previously
    # shared Torch/JAX ambiguity; f is a shared 1D grid for every binary.
    cases = [CASES[0], CASES[2]]
    frequencies = [25., 70.]
    batch = {key: [case[key] for case in cases] for key in cases[0]}
    grid = [[f] for f in frequencies] if per_binary else frequencies
    references = [35., 45.] if per_binary else [40., 40.]
    actual = actual_waveform(backend, batch, grid, spin_order=spin_order, use_ref=use_ref, compiled=True,
                             f_ref=references if per_binary else 40.)
    assert all(value.shape == (2, 1 if per_binary else 2) for value in actual)
    for row, case in enumerate(cases):
        row_frequencies = [frequencies[row]] if per_binary else frequencies
        for column, frequency in enumerate(row_frequencies):
            check_sample(tuple(value[row, column] for value in actual), case, frequency,
                         spin_order=spin_order, use_ref=use_ref, f_ref=references[row])


@pytest.mark.parametrize("spin_order", [5, 7, 8])
@pytest.mark.parametrize("use_ref", [False, True])
def test_aligned_spin_frequency_derivative(backend, spin_order, use_ref):
    name, model, array = backend
    case = CASES[1]
    frequencies = [20., 31.25, 80.]
    parameters = [array(case[key]) for key in ("mass1", "mass2", "chi1", "chi2")]
    opts = dict(tc=array(case["tc"]), spin_order=spin_order, f_ref=array(40.) if use_ref else None,
                phasing_convention="reference_frequency" if use_ref else "coalescence")
    if name == "torch":
        # Torch exposes an independently maintained analytic dPsi/df function.
        actual = host(model.evaluate_phase_derivative(array(frequencies), *parameters, **opts))
    else:
        import jax

        def phase(f):
            return model.evaluate_phase(f, *parameters, phic=array(case["phic"]), **opts)

        jacobian = host(jax.jacfwd(phase)(array(frequencies)))
        actual = np.diag(jacobian)
        np.testing.assert_array_equal(jacobian-np.diag(actual), 0)
    derivative = ref.derivative_expr(expressions(spin_order, use_ref)["phase"], ref.F)
    expected = [float(ref.evaluate(derivative, reference_values(case, f))) for f in frequencies]
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-12)


def test_aligned_spin_extremal_spin_bounds(backend):
    case = dict(CASES[2], chi1=1.0, chi2=-1.0)
    for use_ref in (False, True):
        actual = actual_waveform(backend, case, [31.25], spin_order=8, use_ref=use_ref)
        check_sample(tuple(value[0] for value in actual), case, 31.25, spin_order=8, use_ref=use_ref)


def test_aligned_spin_reference_precision_convergence():
    values = reference_values(CASES[2], "40.0000000000000000000001")
    for expression in expressions(8, True).values():
        # Extra working digits cover the 22-digit reference subtraction and the
        # nearly edge-on cross polarization while retaining 65-digit agreement.
        low = ref.evaluate(expression, values, dps=100)
        high = ref.evaluate(expression, values, dps=160)
        assert abs(low-high) <= abs(high)*sp.Rational(1, 10**65)


@pytest.mark.parametrize("symbol,value", [(ref.DELTA, -1), (ref.DELTA, 1), (ref.DELTA, "nan"),
                                          (ref.CHI1, "1.001"), (ref.CHI2, "-1.001"), (ref.CHI2, "inf")])
def test_aligned_spin_reference_domains(symbol, value):
    values = reference_values(CASES[0], 40.0)
    values[symbol] = value
    with pytest.raises(ValueError):
        ref.evaluate(expressions(8, True)["h_plus"], values)


def test_aligned_spin_reference_artifacts_are_current():
    from diffgw_reference.aligned_spin_artifacts import coefficient_table, fixtures

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/math/taylorf2-aligned-spin-coefficients.md").read_text() == coefficient_table()
    stored = json.loads((root / "tests/data/taylorf2_aligned_spin_reference_v1.json").read_text())
    # Fixture generation checks convergence at 100/160 digits before storing 65
    # significant digits; these snapshots are not an independent second oracle.
    assert stored == json.loads(json.dumps(fixtures()))


def test_aligned_spin_spa_spin7_balance_factor():
    # Independently integrate a generic 3.5PN term in dt/dv. Integration constants
    # are the separately specified coalescence time/orbital phase and are set to 0.
    transfer = sp.Symbol("T7", real=True)
    v, mass, eta = ref.V, ref.M_SEC, nonspinning.ETA
    dt_dv = 5*mass*transfer/(32*eta*v**2)
    time_term = sp.integrate(dt_dv, v)
    orbital_phase_term = sp.integrate(v**3/mass*dt_dv, v)
    fourier_phase_term = 2*v**3/mass*time_term-2*orbital_phase_term
    coefficient = sp.cancel(fourier_phase_term/(3*v**2/(128*eta)))
    assert coefficient == -20*transfer

    balance = ref.ss7_balance_terms()
    ss7 = ref.spin_corrections()["SS7"]
    assert sp.expand(ss7+20*balance["T7SS"]) == 0
    assert sp.expand(balance["SS7_tail"]-20*balance["F7SS"]) == 0
    assert sp.expand(ss7-balance["SS7_without_tail"]-20*balance["F7SS"]) == 0
    assert balance["F7SS"] != 0
    assert sp.Poly(ss7, ref.CHI1, ref.CHI2).total_degree() == 2
    exchanged = ss7.subs({ref.DELTA: -ref.DELTA, ref.CHI1: ref.CHI2, ref.CHI2: ref.CHI1},
                         simultaneous=True)
    assert sp.expand(ss7-exchanged) == 0

    # Independent transcription of Pratten et al. arXiv:2001.11412v1:
    # A19's quadratic terms require the separately printed A24 tail increment.
    # Keeping both expressions detects a return to the incomplete A19 subset.
    delta, eta = ref.DELTA, ref.ETA_EXPR
    chi_s, chi_a = (ref.CHI1+ref.CHI2)/2, (ref.CHI1-ref.CHI2)/2
    a19 = sp.pi*(-1140*delta*chi_a*chi_s+(40*eta-570)*chi_s**2+(2240*eta-570)*chi_a**2)
    a24 = sp.pi*(325*delta*chi_a*chi_s+(sp.Rational(325, 2)-640*eta)*chi_a**2
                 +(sp.Rational(325, 2)-10*eta)*chi_s**2)
    assert sp.expand(a19-balance["SS7_without_tail"]) == 0
    assert sp.expand(a24-balance["SS7_tail"]) == 0
    assert sp.expand(a19+a24-ss7) == 0
