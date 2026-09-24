"""Literature reference checks; neither frontend supplies expected mathematics.

Unit conversion is conditional on the constants supplied by diffgw_core. Their
physical provenance is outside this test, and they never enter the math kernel.
"""

from functools import lru_cache
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
mpmath = pytest.importorskip("mpmath")

from diffgw_core.constants import C_SI, MPC_SI, MSUN_SEC
from diffgw_reference import taylorf2 as ref
from diffgw_reference.traceability import PHASE_COEFFICIENT_ORDER


CASES = [
    # Equal mass, long BNS phase, and asymmetric BBH; scalar inputs in API units.
    dict(mass1=30.0, mass2=30.0, distance=400.0, inclination=0.0, tc=0.01, phic=0.2),
    dict(mass1=1.4, mass2=1.3, distance=100.0, inclination=1.2, tc=-0.03, phic=-0.4),
    dict(mass1=50.0, mass2=5.0, distance=750.0, inclination=np.pi / 2, tc=0.0, phic=0.7),
]
FREQUENCIES = [20.0, 31.25, 80.0, 160.0]
NAMES = ("f", "mass1", "mass2", "distance", "inclination", "tc", "phic", "f_ref")


def exact(value):
    return sp.Rational(str(value))


def reference_values(case, frequency, f_ref=40.0):
    m1, m2 = exact(case["mass1"]), exact(case["mass2"])
    return {
        ref.F: exact(frequency),
        ref.M_SEC: (m1 + m2) * exact(MSUN_SEC),
        ref.ETA: m1 * m2 / (m1 + m2) ** 2,
        ref.D_SEC: exact(case["distance"]) * exact(MPC_SI) / exact(C_SI),
        ref.IOTA: exact(case["inclination"]),
        ref.TC: exact(case["tc"]),
        ref.PHIC: exact(case["phic"]),
        ref.F_REF: exact(f_ref),
    }


@pytest.fixture(params=["torch", "jax"])
def backend(request, reference_device):
    # Explicit placement prevents accelerator qualification from silently using CPU.
    if reference_device == "mps":
        pytest.fail("The math-reference gate requires FP64, which MPS does not support.")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.taylorf2 import TaylorF2

        if reference_device == "cuda" and not torch.cuda.is_available():
            pytest.fail("Requested CUDA math-reference qualification, but Torch CUDA is unavailable.")
        yield "torch", TaylorF2, lambda x: torch.as_tensor(x, dtype=torch.float64, device=reference_device)
    else:
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp
        from jaxwave.approximants.taylorf2 import TaylorF2

        platform = "gpu" if reference_device == "cuda" else "cpu"
        try:
            devices = jax.devices(platform)
        except RuntimeError as exc:
            pytest.fail(f"Requested {reference_device} math-reference qualification is unavailable: {exc}")
        if reference_device == "cuda" and "cuda" not in devices[0].client.platform_version.lower():
            pytest.fail("Requested CUDA math-reference qualification, but JAX GPU is not CUDA.")
        previous = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)
        try:
            with jax.default_device(devices[0]):
                yield "jax", TaylorF2, lambda x: jnp.asarray(x, dtype=jnp.float64)
        finally:
            jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("case", CASES)
def test_pn_coefficients_against_exact_reference(backend, case):
    _, model, array = backend
    values = reference_values(case, 40)
    expected = ref.evaluate_coefficients(values[ref.ETA], dps=80)
    actual = model.phase_coefficients(array(case["mass1"]), array(case["mass2"]), array(0), array(0))
    assert len(actual) == len(PHASE_COEFFICIENT_ORDER)
    for index, name in enumerate(PHASE_COEFFICIENT_ORDER):
        if name in expected:
            np.testing.assert_allclose(float(actual[index]), float(expected[name]), rtol=2e-14, atol=2e-13,
                                       err_msg=name)
        else:
            assert float(actual[index]) == 0, name


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("use_ref", [False, True])
def test_values_against_reference(backend, case, use_ref):
    _, model, array = backend
    convention = "reference_frequency" if use_ref else "coalescence"
    opts = dict(f_ref=40.0 if use_ref else None, phasing_convention=convention)
    args = {key: array(value) for key, value in case.items()}
    frequencies = array(FREQUENCIES)
    phase = model.evaluate_phase(frequencies, args["mass1"], args["mass2"], array(0), array(0),
                                 tc=args["tc"], phic=args["phic"], **opts)
    amplitude = model.evaluate_amplitude(frequencies, args["mass1"], args["mass2"], args["distance"])
    hp, hc = model.evaluate(f=frequencies, **args, f_isco_cutoff=False, **opts)
    expressions = ref.waveform_expr(reference_frequency=use_ref)
    for idx, frequency in enumerate(FREQUENCIES):
        values = reference_values(case, frequency)
        expected = {key: ref.evaluate(expr, values, dps=80) for key, expr in expressions.items()}
        scale = float(expected["amplitude"])
        np.testing.assert_allclose(float(phase[idx]), float(expected["phase"]), rtol=5e-14, atol=2e-10)
        np.testing.assert_allclose(float(amplitude[idx]) / scale, 1, rtol=5e-14, atol=0)
        # Normalize by A rather than h_cross, which vanishes at edge-on inclination.
        for result, key in [(hp[idx], "h_plus"), (hc[idx], "h_cross")]:
            assert abs(complex(result) - complex(expected[key])) / scale < 3e-10


@pytest.mark.parametrize("use_ref", [False, True])
def test_public_dispatch_against_nonspinning_reference(backend, use_ref):
    """The catalog ID reaches the scoped native carrier through the public API."""
    import diffgw

    name, _, array = backend
    case = dict(CASES[0], inclination=0.7)
    frequencies = (20.0, 31.25, 80.0)
    hp, hc = diffgw.get_fd_waveform(
        "TaylorF2", backend=name, sample_frequencies=array(frequencies),
        **case, f_ref=40.0 if use_ref else None, f_isco_cutoff=False,
    )
    expressions = ref.waveform_expr(reference_frequency=use_ref)
    for index, frequency in enumerate(frequencies):
        values = reference_values(case, frequency)
        amplitude = float(ref.evaluate(expressions["amplitude"], values, dps=80))
        for actual, key in ((hp[index], "h_plus"), (hc[index], "h_cross")):
            expected = complex(ref.evaluate(expressions[key], values, dps=80))
            assert abs(complex(actual) - expected) / amplitude < 3e-10


@lru_cache(maxsize=None)
def reference_derivatives(use_ref):
    expressions = ref.waveform_expr(reference_frequency=use_ref)
    return {key: {symbol: ref.derivative_expr(expr, symbol)
                  for symbol in (ref.F, ref.M_SEC, ref.ETA, ref.D_SEC, ref.IOTA, ref.TC, ref.PHIC, ref.F_REF)}
            for key, expr in expressions.items()}


def expected_jacobian(case, frequency, use_ref, f_ref=40.0):
    values = reference_values(case, frequency, f_ref)
    m1, m2 = exact(case["mass1"]), exact(case["mass2"])
    jacobian = []
    for key in ("amplitude", "phase", "h_plus", "h_cross"):
        derivatives = reference_derivatives(use_ref)[key]
        # Chain rule from geometric parameters to component solar masses and Mpc.
        expressions = [
            derivatives[ref.F],
            derivatives[ref.M_SEC] * exact(MSUN_SEC)
            + derivatives[ref.ETA] * m2 * (m2 - m1) / (m1 + m2) ** 3,
            derivatives[ref.M_SEC] * exact(MSUN_SEC)
            + derivatives[ref.ETA] * m1 * (m1 - m2) / (m1 + m2) ** 3,
            derivatives[ref.D_SEC] * exact(MPC_SI) / exact(C_SI),
            derivatives[ref.IOTA], derivatives[ref.TC], derivatives[ref.PHIC], derivatives[ref.F_REF],
        ]
        row = np.array([complex(ref.evaluate(expr, values, dps=80)) for expr in expressions])
        if key in ("h_plus", "h_cross"):
            jacobian.extend([row.real, row.imag])
        else:
            jacobian.append(row.real)
    return np.array(jacobian)


@pytest.mark.parametrize("case,frequency", [
    (CASES[0], 31.25),
    (CASES[1], 31.25),
    (CASES[2], 8.0),
    (dict(CASES[1], inclination=np.pi), np.nextafter(40.0, np.inf)),
])
@pytest.mark.parametrize("use_ref", [False, True])
def test_parameter_jacobian_against_symbolic_derivatives(backend, case, frequency, use_ref):
    name, model, array = backend
    parameters = [frequency, *(case[key] for key in NAMES[1:-1]), 40.0]

    def output(x):
        args = dict(zip(NAMES, x))
        if not use_ref:
            args["f_ref"] = None
        args["phasing_convention"] = "reference_frequency" if use_ref else "coalescence"
        hp, hc = model.evaluate(**args, f_isco_cutoff=False)
        amplitude = model.evaluate_amplitude(x[0], x[1], x[2], x[3])
        phase = model.evaluate_phase(x[0], x[1], x[2], array(0), array(0), tc=x[5], phic=x[6],
                                     f_ref=args["f_ref"], phasing_convention=args["phasing_convention"])
        # The reference-frequency path can return a length-one phase for scalar f.
        result = tuple(value.reshape(()) for value in (amplitude, phase, hp.real, hp.imag, hc.real, hc.imag))
        if name == "torch":
            import torch
            return torch.stack(result)
        import jax.numpy as jnp
        return jnp.stack(result)

    if name == "torch":
        import torch
        actual = torch.autograd.functional.jacobian(output, array(parameters)).detach().cpu().numpy()
    else:
        import jax
        actual = np.asarray(jax.jacfwd(output)(array(parameters)))
    expected = expected_jacobian(case, frequency, use_ref)
    amplitude = float(ref.evaluate(ref.amplitude_expr(), reference_values(case, frequency)))
    scales = np.array([amplitude, 1, amplitude, amplitude, amplitude, amplitude])[:, None]
    # Error in units of the local amplitude; relative tolerance handles large phase sensitivities.
    np.testing.assert_allclose(actual / scales, expected / scales, rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize("frequencies", [[25.0, 70.0, 150.0], [25.0, 70.0]])
@pytest.mark.parametrize("use_ref", [False, True])
def test_batch_and_jax_jit_against_scalar_reference(backend, frequencies, use_ref):
    name, model, array = backend
    masses1, masses2 = [30.0, 1.4], [20.0, 1.3]
    opts = dict(f_ref=array(40.) if use_ref else None,
                phasing_convention="reference_frequency" if use_ref else "coalescence")

    def evaluate(m1, m2):
        phase = model.evaluate_phase(array(frequencies), m1, m2, array(0), array(0), **opts)
        hp, hc = model.evaluate(array(frequencies), m1, m2, distance=100., inclination=0.7,
                                f_isco_cutoff=False, **opts)
        return phase, hp, hc

    if name == "jax":
        import jax
        evaluate = jax.jit(evaluate)
    phase, hp, hc = evaluate(array(masses1), array(masses2))
    assert phase.shape == hp.shape == hc.shape == (2, len(frequencies))
    for batch, (m1, m2) in enumerate(zip(masses1, masses2)):
        case = dict(mass1=m1, mass2=m2, distance=100., inclination=0.7, tc=0., phic=0.)
        for idx, frequency in enumerate(frequencies):
            values = reference_values(case, frequency)
            expressions = ref.waveform_expr(reference_frequency=use_ref)
            amplitude = float(ref.evaluate(expressions["amplitude"], values))
            np.testing.assert_allclose(float(phase[batch, idx]), float(ref.evaluate(expressions["phase"], values)),
                                       rtol=5e-14, atol=2e-10)
            for result, key in [(hp[batch, idx], "h_plus"), (hc[batch, idx], "h_cross")]:
                assert abs(complex(result) - complex(ref.evaluate(expressions[key], values))) / amplitude < 3e-10


@pytest.mark.parametrize("use_ref", [False, True])
def test_per_binary_frequency_grid_and_reference_vector(backend, use_ref):
    name, model, array = backend
    # Explicit (B,1) evaluates one different frequency for each binary. A length-B
    # f_ref vector has this same per-binary meaning; a 1D f is a shared grid.
    cases = [dict(CASES[0], inclination=0.7), dict(CASES[1], inclination=np.pi)]
    frequencies, references = [25.0, 70.0], [35.0, 45.0]
    args = {key: array([case[key] for case in cases]) for key in cases[0]}
    f = array([[frequency] for frequency in frequencies])
    opts = dict(f_ref=array(references) if use_ref else None,
                phasing_convention="reference_frequency" if use_ref else "coalescence")

    def evaluate(f):
        phase = model.evaluate_phase(f, args["mass1"], args["mass2"], array(0), array(0),
                                     tc=args["tc"], phic=args["phic"], **opts)
        hp, hc = model.evaluate(f, **args, f_isco_cutoff=False, **opts)
        return phase, hp, hc

    if name == "jax":
        import jax
        evaluate = jax.jit(evaluate)
    phase, hp, hc = evaluate(f)
    assert phase.shape == hp.shape == hc.shape == (2, 1)
    expressions = ref.waveform_expr(reference_frequency=use_ref)
    for idx, (case, frequency, f_ref) in enumerate(zip(cases, frequencies, references)):
        values = reference_values(case, frequency, f_ref)
        expected = {key: ref.evaluate(expr, values) for key, expr in expressions.items()}
        np.testing.assert_allclose(float(phase[idx, 0]), float(expected["phase"]), rtol=5e-14, atol=2e-10)
        for result, key in [(hp[idx, 0], "h_plus"), (hc[idx, 0], "h_cross")]:
            assert abs(complex(result) - complex(expected[key])) / float(expected["amplitude"]) < 3e-10


def test_exact_equations_and_phase_gauge():
    intrinsic = ref.phase_expr()
    assert not intrinsic.atoms(sp.Float)
    c = ref.coefficients()
    assert sp.limit(intrinsic * 128 * ref.ETA * ref.V**5 / 3, ref.V, 0, dir="+") == 1
    v_lso = 1 / sp.sqrt(6)
    assert sp.simplify(ref.paper_phase_expr() - intrinsic + 9*c["phi5"]*sp.log(v_lso)/(128*ref.ETA)) == 0
    phase = ref.waveform_expr(reference_frequency=True)["phase"]
    assert sp.simplify(phase.subs(ref.F, ref.F_REF) - (3*sp.pi/4 + 2*sp.pi*ref.F_REF*ref.TC - 2*ref.PHIC)) == 0


def test_newtonian_amplitude_reduction_and_polarization_limits():
    # Arun Eq. (6.13), n=0,k=2 and Eq. (D1), with the cutoff omitted.
    v = (sp.pi * ref.M_SEC * ref.F) ** sp.Rational(1, 3)
    source_amplitude = 2 * ref.M_SEC**2 / ref.D_SEC * sp.sqrt(5*sp.pi*ref.ETA/96) * v**sp.Rational(-7, 2)
    assert sp.simplify(source_amplitude - ref.amplitude_expr()) == 0
    expressions = ref.waveform_expr()
    hp, hc = expressions["h_plus"], expressions["h_cross"]
    assert sp.simplify((hc + sp.I*hp).subs(ref.IOTA, 0)) == 0
    assert hc.subs(ref.IOTA, sp.pi/2) == 0
    assert sp.simplify(sp.diff(hp, ref.D_SEC) + hp/ref.D_SEC) == 0


def test_reference_evaluator_against_sympy_evalf_without_global_precision_change():
    values = reference_values(CASES[1], 31.25)
    original_precision = mpmath.mp.dps
    for expr in ref.waveform_expr(reference_frequency=True).values():
        actual = ref.evaluate(expr, values, dps=100)
        symbolic_numeric = expr.evalf(100, subs=values)
        ctx = actual.context
        expected = ctx.mpc(str(sp.re(symbolic_numeric)), str(sp.im(symbolic_numeric)))
        assert abs(actual - expected) < abs(expected) * ctx.mpf("1e-85")
    assert mpmath.mp.dps == original_precision


def test_maintained_reference_artifacts_are_current():
    from diffgw_reference.artifacts import coefficient_table, fixtures

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/math/taylorf2-coefficients.md").read_text() == coefficient_table()
    stored = json.loads((root / "tests/data/taylorf2_reference_v1.json").read_text())
    assert stored == json.loads(json.dumps(fixtures()))


@pytest.mark.parametrize("symbol,value", [(ref.F, "0"), (ref.M_SEC, "-1"), (ref.D_SEC, "0"),
                                          (ref.ETA, "0"), (ref.ETA, "0.251"), (ref.F_REF, "-1"),
                                          (ref.F, "nan"), (ref.F, "inf")])
def test_reference_rejects_out_of_domain(symbol, value):
    values = reference_values(CASES[0], 40)
    values[symbol] = value
    with pytest.raises(ValueError):
        ref.evaluate(ref.waveform_expr(reference_frequency=True)["h_plus"], values)


@pytest.mark.parametrize("use_ref", [False, True])
def test_reference_precision_convergence(use_ref):
    # Close frequencies stress cancellation in reference-frequency subtraction.
    values = reference_values(CASES[1], "40.0000000000000000000001")
    for expr in ref.waveform_expr(reference_frequency=use_ref).values():
        low, high = ref.evaluate(expr, values, dps=80), ref.evaluate(expr, values, dps=120)
        assert abs(low - high) < abs(high) * sp.Float("1e-65")


def sweep_samples(*, single_precision=False):
    """Fixed seed and bounded domain, independent of either frontend's outputs.

    FP64: M in [2.7,100] solar masses, q in [1,20], v in [0.07,0.30].
    FP32: v and v_ref in [0.20,0.30]; this deliberately does not qualify the
    many-cycle low-frequency regime for single-precision phase accuracy.
    Spins, tides, frequency clamps, and termination windows remain excluded.
    """
    rng = np.random.default_rng(20260922)
    for _ in range(12):
        total_mass = np.exp(rng.uniform(np.log(2.7), np.log(100.0)))
        ratio = np.exp(rng.uniform(0, np.log(20.0)))
        case = dict(
            mass1=total_mass * ratio / (1 + ratio), mass2=total_mass / (1 + ratio),
            distance=np.exp(rng.uniform(np.log(10.0), np.log(2000.0))),
            inclination=rng.uniform(0, np.pi), tc=rng.uniform(-0.1, 0.1),
            phic=rng.uniform(-np.pi, np.pi),
        )
        v = rng.uniform(0.20 if single_precision else 0.07, 0.30)
        v_ref = rng.uniform(0.20 if single_precision else 0.08, 0.30)
        frequency = v**3 / (np.pi * total_mass * MSUN_SEC)
        f_ref = v_ref**3 / (np.pi * total_mass * MSUN_SEC)
        yield case, frequency, f_ref


def check_conditioned_values(backend, case, frequency, f_ref, use_ref, *, bits=64, evaluate=None):
    """Compare against the oracle with an input-conditioned rounding budget.

    Cancellation in I(f)-I(f_ref) is governed by the two unreduced phases,
    not by their small difference. 64 epsilon times their absolute scale is
    an engineering budget for this short polynomial/logarithm evaluation,
    not an interval proof. The original fixed FP64 tolerances remain tested.
    The complex-error budget follows |exp(ix)-exp(iy)| <= |x-y| and adds
    32 epsilon for amplitude and projection rounding; no phase fit is used.
    """
    name, model, array = backend
    scalar = np.float32 if bits == 32 else np.float64
    case = {key: float(scalar(value)) for key, value in case.items()}
    frequency, f_ref = float(scalar(frequency)), float(scalar(f_ref))
    values = reference_values(case, frequency, f_ref)
    expressions = ref.waveform_expr(reference_frequency=use_ref)
    expected = {key: ref.evaluate(expr, values, dps=80) for key, expr in expressions.items()}
    # Build the conditioning scale from exact expressions, never production output.
    intrinsic = ref.phase_expr().subs(ref.V, (sp.pi * ref.M_SEC * ref.F)**sp.Rational(1, 3))
    condition = abs(float(ref.evaluate(intrinsic, values)))
    if use_ref:
        condition += abs(float(ref.evaluate(intrinsic, {**values, ref.F: values[ref.F_REF]})))
    condition += 2*np.pi*abs(frequency*case["tc"]) + 2*abs(case["phic"]) + np.pi
    eps = np.finfo(scalar).eps
    phase_budget, amplitude_budget = 64*eps*condition, 32*eps
    if bits == 32:
        # An explicit scientific acceptance cap prevents large phase scales from
        # making a precision-aware tolerance vacuous in this qualified subdomain.
        assert phase_budget < 0.02, f"FP32 sample exceeds the 0.02 rad qualification budget: {phase_budget}"

    args = {key: array(value) for key, value in case.items()}
    f = array([frequency])
    opts = dict(f_ref=array(f_ref) if use_ref else None,
                phasing_convention="reference_frequency" if use_ref else "coalescence")
    phase = model.evaluate_phase(f, args["mass1"], args["mass2"], array(0), array(0),
                                 tc=args["tc"], phic=args["phic"], **opts)
    amplitude = model.evaluate_amplitude(f, args["mass1"], args["mass2"], args["distance"])
    evaluate = model.evaluate if evaluate is None else evaluate
    hp, hc = evaluate(f=f, **args, f_isco_cutoff=False, **opts)
    if name == "torch":
        import torch
        assert phase.dtype == amplitude.dtype == (torch.float32 if bits == 32 else torch.float64)
        assert hp.dtype == hc.dtype == (torch.complex64 if bits == 32 else torch.complex128)
        assert hp.device == f.device
    else:
        assert phase.dtype == amplitude.dtype == scalar
        assert hp.dtype == hc.dtype == (np.complex64 if bits == 32 else np.complex128)
        assert hp.devices() == f.devices()
    scale = float(expected["amplitude"])
    detail = f"{name} FP{bits}, case={case}, f={frequency}, f_ref={f_ref}, use_ref={use_ref}"
    assert abs(float(phase[0]) - float(expected["phase"])) <= phase_budget, detail
    assert abs(float(amplitude[0])/scale - 1) <= amplitude_budget, detail
    for result, key in [(hp[0], "h_plus"), (hc[0], "h_cross")]:
        assert abs(complex(result) - complex(expected[key])) / scale <= phase_budget + amplitude_budget, detail


@pytest.mark.parametrize("use_ref", [False, True])
def test_seeded_parameter_sweep(backend, use_ref):
    for case, frequency, f_ref in sweep_samples():
        check_conditioned_values(backend, case, frequency, f_ref, use_ref)


@pytest.mark.parametrize("use_ref", [False, True])
def test_low_velocity_and_inclination_limits(backend, use_ref):
    for inclination in (0.0, np.pi/2, np.pi):
        for case, frequency in [(CASES[1], 1.0), (CASES[2], 0.5)]:
            check_conditioned_values(backend, dict(case, inclination=inclination), frequency, 40.0, use_ref)


def test_reference_frequency_neighbors(backend):
    # Three adjacent representable inputs exercise the cancellation boundary
    # actually visible to FP64 frontends, including the exact reference anchor.
    for case in (CASES[0], CASES[1], CASES[2]):
        for frequency in (np.nextafter(40.0, -np.inf), 40.0, np.nextafter(40.0, np.inf)):
            check_conditioned_values(backend, case, frequency, 40.0, True)


@pytest.mark.parametrize("use_ref", [False, True])
def test_single_precision_parameter_sweep(backend, use_ref):
    name, model, array64 = backend
    if name == "torch":
        import torch

        def array(value):
            return array64(value).to(torch.float32)

        for case, frequency, f_ref in sweep_samples(single_precision=True):
            check_conditioned_values((name, model, array), case, frequency, f_ref, use_ref, bits=32)
    else:
        import jax
        import jax.numpy as jnp
        previous = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", False)
        try:
            # Public JAX evaluate requests float64 internally. With x64 disabled,
            # JAX intentionally truncates that request; output dtype is asserted
            # above so this cannot accidentally qualify promoted FP64 arithmetic.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Explicitly requested dtype.*float64.*not available")

                def array(value):
                    return jnp.asarray(value, dtype=jnp.float32)

                for case, frequency, f_ref in sweep_samples(single_precision=True):
                    check_conditioned_values((name, model, array), case, frequency, f_ref, use_ref, bits=32)
        finally:
            jax.config.update("jax_enable_x64", previous)


def test_nonspinning_symmetries(backend):
    name, model, array = backend
    f = array([20.0, 50.0, 100.0])
    args = dict(mass1=array(50.0), mass2=array(5.0), distance=array(100.0),
                inclination=array(0.7), f_ref=array(40.0), f_isco_cutoff=False)
    hp, hc = model.evaluate(f, **args)
    swapped = model.evaluate(f, **{**args, "mass1": args["mass2"], "mass2": args["mass1"]})
    distant = model.evaluate(f, **{**args, "distance": 2*args["distance"]})
    reflected = model.evaluate(f, **{**args, "inclination": array(np.pi)-args["inclination"]})

    def host(value):
        return value.detach().cpu().numpy() if name == "torch" else np.asarray(value)

    for original, other in zip((hp, hc), swapped):
        np.testing.assert_array_equal(host(original), host(other))
    for original, other in zip((hp, hc), distant):
        np.testing.assert_allclose(host(2*other), host(original), rtol=2e-14, atol=0)
    np.testing.assert_allclose(host(reflected[0]), host(hp), rtol=2e-14, atol=0)
    np.testing.assert_allclose(host(reflected[1]), -host(hc), rtol=2e-14, atol=0)


def test_compiled_waveform_against_reference(backend):
    name, model, _ = backend
    if name == "torch":
        import torch
        # Exercise complete Dynamo/AOT graph capture without a platform-specific
        # compiler dependency. This is not qualification of Inductor codegen.
        evaluate = torch.compile(model.evaluate, backend="aot_eager", fullgraph=True)
    else:
        import jax
        evaluate = jax.jit(model.evaluate, static_argnames=("f_isco_cutoff", "phasing_convention"))
    for case in (CASES[0], CASES[1]):
        check_conditioned_values(backend, case, 31.25, 40.0, True, evaluate=evaluate)
