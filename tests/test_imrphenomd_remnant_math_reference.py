"""Published remnant equations and conditional QNM checks for Torch and JAX."""

from decimal import Decimal
import json
from pathlib import Path

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("mpmath")
pytest.importorskip("scipy")

from diffgw_reference import imrphenomd_remnant as ref


CASES = (
    (30.0, 30.0, 0.0, 0.0),
    (45.0, 15.0, 0.45, -0.25),
    (75.0, 25.0, 0.8, 0.6),
    (15.0, 45.0, -0.25, 0.45),
)


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device == "mps":
        pytest.fail("The math-reference gate requires FP64, unavailable on MPS")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.imrphenomd import IMRPhenomD
        from torchwave.approximants.kerr_qnm_d_sampling_clean import evaluate_d_qnm_220

        if reference_device == "cuda" and not torch.cuda.is_available():
            pytest.fail("Torch CUDA unavailable")
        def array(value):
            return torch.as_tensor(value, dtype=torch.float64, device=reference_device)

        yield "torch", IMRPhenomD, evaluate_d_qnm_220, array
        return

    jax = pytest.importorskip("jax")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd import IMRPhenomD, evaluate_d_qnm_220

        platform = "gpu" if reference_device == "cuda" else "cpu"
        try:
            devices = jax.devices(platform)
        except RuntimeError as exc:
            pytest.fail(f"JAX {platform} unavailable: {exc}")
        if reference_device == "cuda" and "cuda" not in devices[0].client.platform_version.lower():
            pytest.fail("JAX GPU is not CUDA")
        with jax.default_device(devices[0]):
            def array(value):
                return jax.device_put(jnp.asarray(value, dtype=jnp.float64), devices[0])

            yield "jax", IMRPhenomD, evaluate_d_qnm_220, array
    finally:
        jax.config.update("jax_enable_x64", previous)


def inherited_expressions():
    """Bind production's extra inherited digits explicitly to paper algebra."""
    from diffgw_core.fits import (
        FINAL_SPIN_ETA_POLY, FINAL_SPIN_S_COEFFS, ERAD_ETA_NUMERATOR,
        ERAD_S_NUMERATOR, ERAD_S_DENOMINATOR,
    )

    spin = ref.final_spin_expr(
        eta_coeffs=tuple(map(str, FINAL_SPIN_ETA_POLY)),
        s_eta_coeffs=tuple(str(row[0]) for row in FINAL_SPIN_S_COEFFS),
        s_eta2_coeffs=tuple(str(row[1]) for row in FINAL_SPIN_S_COEFFS),
    )
    energy = ref.radiated_fraction_expr(
        ns_coeffs=tuple(map(str, ERAD_ETA_NUMERATOR)),
        num_coeffs=tuple(map(str, ERAD_S_NUMERATOR)),
        den_coeffs=tuple(map(str, ERAD_S_DENOMINATOR)),
    )
    return spin, energy


def bound_expressions(*, inherited=True):
    total, eta, spin, s_hat = ref.coordinates()
    a, energy = inherited_expressions() if inherited else (
        ref.final_spin_expr(), ref.radiated_fraction_expr())
    a = a.subs({ref.ETA: eta, ref.S: spin})
    energy = energy.subs({ref.ETA: eta, ref.S_HAT: s_hat})
    return total, a, energy, ref.final_mass_expr(total, energy)


def substitutions(case):
    return dict(zip((ref.M1, ref.M2, ref.CHI1, ref.CHI2), map(ref.exact, case)))


def generated_cyclic_qnm(name, array, spin):
    """Read the frontend's generated Kerr path for endpoint selection checks."""
    if name == "torch":
        from torchwave.approximants.kerr_qnm_d_sampling_clean import (
            evaluate_d_qnm_220_generated,
        )

        return [float(value) for value in evaluate_d_qnm_220_generated(array(spin))]
    from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220

    return [float(value) / (2 * np.pi)
            for value in evaluate_kerr_qnm_220(array(spin))]


def test_component_bindings_are_complete_and_resolve():
    assert set(ref.COMPONENT_BINDINGS) == set(ref.COMPONENT_SOURCES)
    for binding in ref.COMPONENT_BINDINGS.values():
        assert callable(getattr(ref, binding["reference"]))
        for frontend in ("torch", "jax"):
            module_name, name = binding[frontend]
            target = pytest.importorskip(module_name)
            for part in name.split("."):
                target = getattr(target, part)
            assert callable(target)
        assert binding["tests"]
        assert all(callable(globals()[name]) for name in binding["tests"])


def test_paper_decimal_transcription_and_limiting_cases():
    assert ref.SOURCES["husa2016"]["arxiv"] == "1508.07250v2"
    assert set(ref.COMPONENT_IDS.values()) == set(ref.COMPONENT_SOURCES)
    assert all(isinstance(ref.exact(c), sp.Rational) for row in (
        ref.SPIN_S_ETA_PRINTED, ref.SPIN_S_ETA2_PRINTED,
        ref.ENERGY_NS_PRINTED, ref.ENERGY_NUM_PRINTED, ref.ENERGY_DEN_PRINTED,
    ) for c in row)
    assert sp.simplify(ref.final_spin_expr(eta=0) - ref.S) == 0
    assert sp.simplify(ref.radiated_fraction_expr(s_hat=0)
                       - sum(ref.exact(c) * ref.ETA**i
                             for i, c in enumerate(ref.ENERGY_NS_PRINTED, 1))) == 0
    total, eta, spin, s_hat = ref.coordinates(1, 1, ref.CHI1, ref.CHI2)
    assert total == 2 and eta == sp.Rational(1, 4)
    assert sp.simplify(spin - (ref.CHI1 + ref.CHI2) / 4) == 0
    assert sp.simplify(s_hat - (ref.CHI1 + ref.CHI2) / 2) == 0


def test_inherited_digits_are_separate_from_printed_fits():
    """Audit every printed fit digit; only the spin eta*S² term disagrees."""
    from diffgw_core.fits import (
        FINAL_SPIN_ETA_POLY, FINAL_SPIN_S_COEFFS, ERAD_ETA_NUMERATOR,
        ERAD_S_NUMERATOR, ERAD_S_DENOMINATOR,
    )

    rows = (
        ("eta polynomial", FINAL_SPIN_ETA_POLY[1:], ref.SPIN_ETA_PRINTED[1:]),
        ("eta*S^k", tuple(row[0] for row in FINAL_SPIN_S_COEFFS),
         ref.SPIN_S_ETA_PRINTED),
        ("eta^2*S^k", tuple(row[1] for row in FINAL_SPIN_S_COEFFS),
         ref.SPIN_S_ETA2_PRINTED),
        ("energy nonspinning", ERAD_ETA_NUMERATOR, ref.ENERGY_NS_PRINTED),
        ("energy numerator", ERAD_S_NUMERATOR, ref.ENERGY_NUM_PRINTED),
        ("energy denominator", ERAD_S_DENOMINATOR, ref.ENERGY_DEN_PRINTED),
    )
    mismatches = []
    for name, inherited_values, printed_values in rows:
        assert len(inherited_values) == len(printed_values)
        for power, (inherited_value, printed_value) in enumerate(
                zip(inherited_values, printed_values), 1):
            rounded = Decimal(str(inherited_value)).quantize(Decimal(printed_value))
            if rounded != Decimal(printed_value):
                mismatches.append((name, power, str(rounded), printed_value))
    assert mismatches == [("eta*S^k", 2, "0.101", "0.102")]
    report = json.loads((Path(__file__).parents[1] / "docs" / "math" / "data"
                         / "imrphenomd_final_spin_refit_2017_v1.json").read_text())
    assert (Decimal(str(FINAL_SPIN_S_COEFFS[1][0]))
            == Decimal(str(report["comparison_b2"]["value"])))


def test_historical_b2_difference_has_a_bounded_symbolic_effect():
    """Isolate the compatibility coefficient without altering the paper spec."""
    from diffgw_core.fits import FINAL_SPIN_S_COEFFS

    inherited_b2 = str(FINAL_SPIN_S_COEFFS[1][0])
    b2_only = list(ref.SPIN_S_ETA_PRINTED)
    b2_only[1] = inherited_b2
    difference = sp.simplify(
        ref.final_spin_expr(s_eta_coeffs=tuple(b2_only)) - ref.final_spin_expr())
    delta_b2 = ref.exact(inherited_b2) - ref.exact(ref.SPIN_S_ETA_PRINTED[1])
    assert sp.simplify(difference - delta_b2 * ref.ETA * ref.S**2) == 0
    # |S| <= 1-2*eta for |chi_i| <= 1; eta*(1-2*eta)^2 peaks
    # at eta=1/6 on the physical interval (0, 1/4].
    envelope = ref.ETA * (1 - 2 * ref.ETA)**2
    assert sp.simplify(sp.diff(envelope, ref.ETA)
                       - (1 - 2 * ref.ETA) * (1 - 6 * ref.ETA)) == 0
    assert envelope.subs(ref.ETA, sp.Rational(1, 6)) == sp.Rational(2, 27)
    assert envelope.subs(ref.ETA, sp.Rational(1, 4)) < sp.Rational(2, 27)
    assert abs(delta_b2) * sp.Rational(2, 27) < sp.Rational(4, 100000)


@pytest.mark.parametrize("case", CASES)
def test_remnant_against_published_and_inherited_equations(backend, case):
    _, model, _, array = backend
    _, paper_spin, paper_energy, _ = bound_expressions(inherited=False)
    _, inherited_spin, inherited_energy, inherited_mass = bound_expressions()
    subs = substitutions(case)
    actual = model.remnant_properties(*(array(value) for value in case))
    expected_spin = float(ref.evaluate(inherited_spin, subs))
    expected_energy = float(ref.evaluate(inherited_energy, subs))
    expected_mass = float(ref.evaluate(inherited_mass, subs))
    np.testing.assert_allclose(float(actual[0]), expected_mass, rtol=2e-14)
    np.testing.assert_allclose(float(actual[1]), expected_spin, rtol=2e-14)
    np.testing.assert_allclose(1 - float(actual[0]) / sum(case[:2]), expected_energy,
                               rtol=2e-13, atol=2e-15)
    # Paper precision and inherited extra digits are related but not identical.
    assert abs(float(actual[1]) - float(ref.evaluate(paper_spin, subs))) < 5e-4
    assert abs(expected_energy - float(ref.evaluate(paper_energy, subs))) < 5e-5


@pytest.mark.parametrize("spin", (-0.7312, -0.1234, 0.1234, 0.7312))
def test_qnm_interpolation_and_mass_units(backend, spin):
    from diffgw_core.qnm_data import KNOT_SPINS, ROUNDED_CYCLIC_ORDINATES

    _, _, qnm, array = backend
    expected = ref.natural_cubic_qnm(KNOT_SPINS, ROUNDED_CYCLIC_ORDINATES, spin)
    actual = qnm(array(spin))
    np.testing.assert_allclose([float(v) for v in actual], expected,
                               rtol=3e-12, atol=2e-12)
    m, mf = sp.symbols("M M_f", positive=True)
    for ordinate in (ref.QNM_REAL, ref.QNM_DAMP):
        assert sp.simplify(ref.qnm_total_mass_expr(m, mf, ordinate)
                           - m * ordinate / mf) == 0


def test_qnm_reference_rejects_unvalidated_extrapolation():
    with pytest.raises(ValueError, match="outside"):
        ref.natural_cubic_qnm((-.9, 0, .9), ((.04, .01), (.06, .01), (.1, .008)), 1.)


@pytest.mark.parametrize("spin", (-0.999, -0.998, -0.9941, -0.994,
                                   0.994, 0.9941, 0.998, 0.999))
def test_qnm_endpoint_policy(backend, spin):
    """Qualify the declared numerical continuation on each side of ±.994."""
    from diffgw_core.qnm_data import KNOT_SPINS, ROUNDED_CYCLIC_ORDINATES

    name, _, qnm, array = backend
    if abs(spin) > 0.994:
        expected = generated_cyclic_qnm(name, array, spin)
    else:
        expected = ref.natural_cubic_qnm(KNOT_SPINS, ROUNDED_CYCLIC_ORDINATES, spin)
    np.testing.assert_allclose([float(value) for value in qnm(array(spin))], expected,
                               rtol=3e-12, atol=2e-12)


@pytest.mark.parametrize("case,clamped", (
    ((150.0, 1.0, 0.999, 0.999), False),
    ((1000.0, 1.0, 1.0, 1.0), True),
))
def test_remnant_high_spin_and_clamp_policy(backend, case, clamped):
    """Check the full remnant call across fallback and at the hard spin cap."""
    name, model, _, array = backend
    total, a_expr, _, mf_expr = bound_expressions()
    subs = substitutions(case)
    raw_spin = float(ref.evaluate(a_expr, subs))
    final_spin = float(np.clip(raw_spin, -0.999, 0.999))
    assert (raw_spin > 0.999) == clamped
    assert final_spin > 0.994
    final_mass = float(ref.evaluate(mf_expr, subs))
    ordinates = generated_cyclic_qnm(name, array, final_spin)
    actual = model.remnant_properties(*(array(value) for value in case))
    expected = [final_mass, final_spin, *(float(total.subs(subs)) * np.asarray(
        ordinates) / final_mass)]
    np.testing.assert_allclose([float(value) for value in actual], expected,
                               rtol=3e-11, atol=2e-12)
    if clamped:
        if name == "torch":
            chi1 = array(case[2]).detach().requires_grad_()
            a_final = model.remnant_properties(array(case[0]), array(case[1]),
                                               chi1, array(case[3]))[1]
            slope = float(__import__("torch").autograd.grad(a_final, chi1)[0])
        else:
            import jax

            def spin_fn(chi1):
                return model.remnant_properties(array(case[0]), array(case[1]),
                                                chi1, array(case[3]))[1]

            slope = float(jax.grad(spin_fn)(array(case[2])))
        assert slope == 0.0


def test_qnm_schwarzschild_against_author_published_datum(backend):
    """Berti's s2l2.dat row 0 is in 2M angular units, not cyclic units."""
    _, _, qnm, array = backend
    assert ref.SOURCES["berti_schwarzschild"]["row"] == 0
    expected = np.array([float(value) for value in
                         ref.AUTHOR_SCHWARZSCHILD_220_2M_OMEGA]) / (4 * np.pi)
    actual = np.array([float(value) for value in qnm(array(0.0))])
    assert np.all(np.abs(actual - expected) <= 1e-8)


@pytest.mark.parametrize("case", CASES)
def test_remnant_qnm_composition(backend, case):
    from diffgw_core.qnm_data import KNOT_SPINS, ROUNDED_CYCLIC_ORDINATES

    _, model, _, array = backend
    total, a_expr, _, mf_expr = bound_expressions()
    subs = substitutions(case)
    spin = float(ref.evaluate(a_expr, subs))
    mf = float(ref.evaluate(mf_expr, subs))
    ordinates = ref.natural_cubic_qnm(KNOT_SPINS, ROUNDED_CYCLIC_ORDINATES, spin)
    actual = model.remnant_properties(*(array(v) for v in case))
    for actual_value, ordinate in zip(actual[2:], ordinates):
        expected = float(ref.evaluate(ref.qnm_total_mass_expr(total, mf, ordinate), subs))
        np.testing.assert_allclose(float(actual_value), expected, rtol=3e-11, atol=2e-12)


def test_qnm_data_against_strict_eigenproblem_holdouts(backend):
    """A numerical-data check; the holdouts share the solver lineage."""
    _, _, qnm, array = backend
    path = Path(__file__).with_name("data") / "kerr_qnm_220_leaver.json"
    rows = json.loads(path.read_text())["holdouts"]
    selected = [row for row in rows if abs(row["a"]) < .9][::max(1, len(rows) // 8)]
    assert len(selected) >= 3
    for row in selected:
        actual = qnm(array(row["a"]))
        expected = np.array([row["wr"], row["damping"]]) / (2 * np.pi)
        error = np.abs(np.array([float(v) for v in actual]) - expected)
        assert np.all(error <= 1e-6) and np.all(error / expected <= 1e-5)


def test_qnm_high_spin_against_eigenproblem_holdouts(backend):
    """Check the endpoint approximation against all stored high-spin solves."""
    name, _, qnm, array = backend
    evidence = json.loads((Path(__file__).with_name("data") /
                           "kerr_qnm_220_leaver.json").read_text())
    rows = [row for source in ("strict", "holdouts") for row in evidence[source]
            if 0.994 < abs(row["a"]) <= 0.99999]
    assert len(rows) == 93 and any(row["a"] == 0.9976 for row in rows)
    expected = np.array([[row["wr"], row["damping"]] for row in rows]) / (2 * np.pi)
    channels = qnm(array([row["a"] for row in rows]))
    if name == "torch":
        actual = np.stack([channel.detach().cpu().numpy() for channel in channels], axis=-1)
    else:
        actual = np.stack([np.asarray(channel) for channel in channels], axis=-1)
    error = np.abs(actual - expected)
    assert np.all(error <= 1e-8) and np.all(error / expected <= 1e-6)


@pytest.mark.parametrize("spin,angular_real,angular_damping", (
    (-0.9999, 0.2915589799182418, 0.0880261027108493),
    (0.9999, 0.9856735483829924, 0.0034686730676612),
))
def test_qnm_near_extremal_against_published_gsn_rows(
    backend, spin, angular_real, angular_damping,
):
    """Check two Lo et al. GSN rows, from a different numerical method.

    Source: https://the-center-of-gravity.com/documents/169/QNMQNEF_s-2.tar.gz
    Members s-2l2m-2n0.dat and s-2l2m2n0.dat at a/M=.9999.  The full author
    table is not redistributed; a known .9976 damping outlier is not used.
    """
    _, _, qnm, array = backend
    expected = np.array([angular_real, angular_damping]) / (2 * np.pi)
    actual = np.array([float(value) for value in qnm(array(spin))])
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("spin", (-0.9971, 0.9971))
def test_jax_high_spin_qnm_jit_and_gradient(reference_device, spin):
    """The generated continuation must survive JIT and differentiation."""
    if reference_device == "mps":
        pytest.fail("The math-reference gate requires FP64, unavailable on MPS")
    jax = pytest.importorskip("jax")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd import evaluate_d_qnm_220

        platform = "gpu" if reference_device == "cuda" else "cpu"
        devices = jax.devices(platform)
        if reference_device == "cuda" and "cuda" not in devices[0].client.platform_version.lower():
            pytest.fail("JAX GPU is not CUDA")
        with jax.default_device(devices[0]):
            def channels(a):
                return jnp.stack(evaluate_d_qnm_220(a))

            compiled = jax.jit(channels)
            input_spin = jnp.asarray(spin, dtype=jnp.float64)
            np.testing.assert_allclose(np.asarray(compiled(input_spin)),
                                       np.asarray(channels(input_spin)), rtol=2e-12)
            slope = np.asarray(jax.jit(jax.jacfwd(channels))(input_spin))
            step = 1e-6
            stencil = (-np.asarray(compiled(input_spin + 2 * step))
                       + 8 * np.asarray(compiled(input_spin + step))
                       - 8 * np.asarray(compiled(input_spin - step))
                       + np.asarray(compiled(input_spin - 2 * step))) / (12 * step)
            assert np.all(np.isfinite(slope)) and np.all(np.abs(slope) > 1e-5)
            np.testing.assert_allclose(slope, stencil, rtol=1e-5, atol=1e-7)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_remnant_spin_and_qnm_spin_derivative(backend):
    from diffgw_core.qnm_data import KNOT_SPINS, ROUNDED_CYCLIC_ORDINATES
    from scipy.interpolate import CubicSpline

    name, model, _, array = backend
    case = CASES[1]
    total, a_expr, _, mf_expr = bound_expressions()
    subs = substitutions(case)
    spin = float(ref.evaluate(a_expr, subs))
    mf = float(ref.evaluate(mf_expr, subs))
    da = float(ref.evaluate(sp.diff(a_expr, ref.CHI1), subs))
    dmf = float(ref.evaluate(sp.diff(mf_expr, ref.CHI1), subs))
    spline = CubicSpline(KNOT_SPINS, ROUNDED_CYCLIC_ORDINATES,
                         axis=0, bc_type="natural")
    q, dq = spline(spin), spline(spin, 1)
    expected = np.array([dmf, da, *(
        float(ref.evaluate(total, subs)) * (dq * da / mf - q * dmf / mf**2))])
    if name == "torch":
        chi1 = array(case[2]).detach().requires_grad_()
        output = model.remnant_properties(array(case[0]), array(case[1]),
                                          chi1, array(case[3]))
        actual = np.array([float(__import__("torch").autograd.grad(
            value, chi1, retain_graph=True)[0]) for value in output])
    else:
        import jax

        def fn(chi1):
            return model.remnant_properties(array(case[0]), array(case[1]),
                                            chi1, array(case[3]))
        actual = np.array(jax.jacfwd(fn)(array(case[2])), dtype=float)
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-10)
