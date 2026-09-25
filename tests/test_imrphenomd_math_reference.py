"""Paper-precision fits and independent amplitude/join equations for both frontends."""

from decimal import Decimal

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import imrphenomd_amplitude as ref


CASES = (
    (30.0, 30.0, 0.0, 0.0),
    (45.0, 15.0, 0.45, -0.25),
    (60.0, 15.0, 0.99, 0.99),  # Exercises the gamma2 > 1 continuation.
)


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device == "mps":
        pytest.fail("The math-reference gate requires FP64, unavailable on MPS")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.imrphenomd import IMRPhenomD

        if reference_device == "cuda" and not torch.cuda.is_available():
            pytest.fail("Torch CUDA unavailable")
        yield "torch", IMRPhenomD, lambda value: torch.as_tensor(
            value, device=reference_device, dtype=torch.float64)
        return

    jax = pytest.importorskip("jax")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd import IMRPhenomD

        platform = "gpu" if reference_device == "cuda" else "cpu"
        try:
            devices = jax.devices(platform)
        except RuntimeError as exc:
            pytest.fail(f"JAX {platform} unavailable: {exc}")
        if reference_device == "cuda" and "cuda" not in devices[0].client.platform_version.lower():
            pytest.fail("JAX GPU is not CUDA")
        with jax.default_device(devices[0]):
            yield "jax", IMRPhenomD, lambda value: jax.device_put(
                jnp.asarray(value, dtype=jnp.float64), devices[0])
    finally:
        jax.config.update("jax_enable_x64", previous)


def exact(value):
    return sp.Rational(str(value))


def coordinates(case):
    return ref.spin_coordinates(*(exact(value) for value in case))


def test_published_rows_are_pinned_to_printed_table():
    """This only checks transcription; extra production digits have separate provenance."""
    from torchwave.phenomd_published_fits import PARAMETER_NAMES, TABLE_V_ROWS

    assert ref.FIT_NAMES == ("rho1", "rho2", "rho3", "v2", "gamma1", "gamma2", "gamma3")
    for index, name in enumerate(ref.FIT_NAMES):
        assert PARAMETER_NAMES[index].replace("_", "") == name
        assert ref.PUBLISHED_ROWS[name] == TABLE_V_ROWS[index]
        assert len(ref.PUBLISHED_ROWS[name]) == 11
        assert all(isinstance(ref.exact(value), sp.Rational) for value in ref.PUBLISHED_ROWS[name])


def test_inherited_digits_are_consistent_with_printed_rounding_only():
    """Rounding compatibility is necessary, but does not establish provenance."""
    from diffgw_core.fits import TABLE_V_UNROUNDED_COEFFS

    found_extra_digits = False
    for name, row in ref.PUBLISHED_ROWS.items():
        for printed, inherited in zip(row, TABLE_V_UNROUNDED_COEFFS[name]):
            paper = Decimal(printed)
            actual = Decimal(str(inherited))
            half_printed_unit = Decimal(1).scaleb(paper.as_tuple().exponent) / 2
            assert abs(actual - paper) <= half_printed_unit, (name, printed, inherited)
            found_extra_digits |= actual != paper
    assert found_extra_digits


@pytest.mark.parametrize("case", CASES)
def test_pn_amplitude_coefficients_from_paper(backend, case):
    _, model, array = backend
    eta, delta, chi_s, chi_a, _ = coordinates(case)
    expected = [float(ref.evaluate(expr, {ref.ETA: eta, ref.DELTA: delta,
                                         ref.CHI_S: chi_s, ref.CHI_A: chi_a}))
                for expr in ref.pn_coefficients()]
    actual = model.pn_amplitude_coefficients(*(array(float(v)) for v in
                                               (eta, delta, chi_s, chi_a)))
    np.testing.assert_allclose([float(value) for value in actual], expected,
                               rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("case", CASES)
def test_hierarchical_fit_formula_and_peak(backend, case):
    """Test fit algebra with the exact inherited row as an explicit input."""
    from diffgw_core.fits import TABLE_V_UNROUNDED_COEFFS

    _, model, array = backend
    eta, _, _, _, chi_pn = coordinates(case)
    for name in ref.FIT_NAMES:
        expected = float(ref.evaluate(ref.fit_expr(
            name, coefficients=tuple(str(x) for x in TABLE_V_UNROUNDED_COEFFS[name])),
            {ref.ETA: eta, ref.CHI_PN: chi_pn}))
        actual = model.eval_table_v(name, array(float(eta)), array(float(chi_pn)))
        np.testing.assert_allclose(float(actual), expected, rtol=2e-11, atol=2e-11,
                                   err_msg=name)

    gamma2 = float(model.eval_table_v("gamma2", array(float(eta)), array(float(chi_pn))))
    gamma3 = float(model.eval_table_v("gamma3", array(float(eta)), array(float(chi_pn))))
    expected_peak = float(ref.evaluate(ref.peak_compatibility(.09, .014, gamma2, gamma3)))
    actual_peak = model.amplitude_peak_frequency(*map(array, (.09, .014, gamma2, gamma3)))
    np.testing.assert_allclose(float(actual_peak), expected_peak, rtol=2e-14)


def test_quartic_has_five_exact_constraints():
    middle = ref.intermediate_expr()
    substitutions = {ref.F1: sp.Rational(14, 1000), ref.F3: sp.Rational(8, 100),
                     ref.V1: sp.Rational(4, 5), ref.V2: sp.Rational(9, 10),
                     ref.V3: sp.Rational(6, 5), ref.D1: sp.Rational(2, 1),
                     ref.D3: sp.Rational(-3, 1)}
    f1, f3 = substitutions[ref.F1], substitutions[ref.F3]
    bound = middle.subs(substitutions)
    assert sp.simplify(bound.subs(ref.U, f1) - substitutions[ref.V1]) == 0
    assert sp.simplify(bound.subs(ref.U, (f1 + f3) / 2) - substitutions[ref.V2]) == 0
    assert sp.simplify(bound.subs(ref.U, f3) - substitutions[ref.V3]) == 0
    assert sp.simplify(sp.diff(bound, ref.U).subs(ref.U, f1) - substitutions[ref.D1]) == 0
    assert sp.simplify(sp.diff(bound, ref.U).subs(ref.U, f3) - substitutions[ref.D3]) == 0
    assert sp.degree(bound, ref.U) == 4


@pytest.mark.parametrize("case", CASES)
def test_frontend_quartic_against_independent_hermite_basis(backend, case):
    """Compare value and slope over the interval, including its two joins."""
    from diffgw_core.fits import TABLE_V_UNROUNDED_COEFFS

    _, model, array = backend
    eta, delta, chi_s, chi_a, chi_pn = coordinates(case)
    fitted = {name: ref.evaluate(ref.fit_expr(
        name, coefficients=tuple(str(x) for x in TABLE_V_UNROUNDED_COEFFS[name])),
        {ref.ETA: eta, ref.CHI_PN: chi_pn}) for name in ref.FIT_NAMES}
    pieces = ref.amplitude_pieces(
        eta=eta, delta=delta, chi_s=chi_s, chi_a=chi_a,
        f_rd=exact(.09), f_damp=exact(.014), **fitted)
    f1, f3 = float(pieces["f1"]), float(pieces["f3"])
    values = [pieces[key] for key in ("v1", "v2", "v3", "d1", "d3")]
    coefficients = model.intermediate_amplitude_coeffs(
        f1, array(f3), *(array(float(value)) for value in values))
    for fraction in (0, .2, .5, .8, 1):
        u = f1 + fraction * (f3 - f1)
        actual = sum(float(c) * u**i for i, c in enumerate(coefficients))
        expected = float(ref.evaluate(pieces["intermediate"], {ref.U: u}))
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-8)
    for u, expected in ((f1, pieces["d1"]), (f3, pieces["d3"])):
        actual = sum(i * float(c) * u**(i - 1) for i, c in enumerate(coefficients) if i)
        np.testing.assert_allclose(actual, float(expected), rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize("case", CASES[:2])
def test_waveform_amplitude_ratios_across_three_regions(backend, case):
    """Common unit and orientation factors cancel; phase is outside this slice."""
    from diffgw_core.constants import MSUN_SEC
    from diffgw_core.fits import TABLE_V_UNROUNDED_COEFFS

    name, model, array = backend
    eta, delta, chi_s, chi_a, chi_pn = coordinates(case)
    fitted = {key: ref.evaluate(ref.fit_expr(
        key, coefficients=tuple(str(x) for x in TABLE_V_UNROUNDED_COEFFS[key])),
        {ref.ETA: eta, ref.CHI_PN: chi_pn}) for key in ref.FIT_NAMES}
    pieces = ref.amplitude_pieces(
        eta=eta, delta=delta, chi_s=chi_s, chi_a=chi_a,
        f_rd=exact(.09), f_damp=exact(.014), **fitted)
    f3 = float(pieces["f3"])
    us = [.010, .014 - 1e-7, .014, .014 + 1e-7,
          .014 + .2 * (f3 - .014), .014 + .7 * (f3 - .014),
          f3 - 1e-7, f3, f3 + 1e-7, f3 + .008, .12]
    frequencies = array(np.array(us) / ((case[0] + case[1]) * MSUN_SEC))

    def remnant_provider(m1, m2, s1, s2):
        return array(case[0] + case[1]), array(0), array(.09), array(.014)

    kwargs = dict(f=frequencies, mass1=array(case[0]), mass2=array(case[1]),
                  chi1=array(case[2]), chi2=array(case[3]), distance=array(400),
                  inclination=array(.7), remnant_provider=remnant_provider)
    hp, _ = model()(**kwargs) if name == "torch" else model.evaluate(**kwargs)
    actual = np.abs(np.asarray(hp.detach().cpu() if name == "torch" else hp))
    expected = np.array([
        float(ref.evaluate(pieces["inspiral" if u <= .014 else
                                  "intermediate" if u <= f3 else "merger_ringdown"],
                           {ref.U: u})) * u**(-7 / 6) for u in us])
    np.testing.assert_allclose(actual / actual[0], expected / expected[0],
                               rtol=3e-8, atol=3e-8)


def test_waveform_amplitude_frequency_derivative(backend):
    """Autodiff of the complete frontend agrees with the analytic amplitude slope."""
    from diffgw_core.constants import MSUN_SEC
    from diffgw_core.fits import TABLE_V_UNROUNDED_COEFFS

    name, model, array = backend
    case = CASES[1]
    eta, delta, chi_s, chi_a, chi_pn = coordinates(case)
    fitted = {key: ref.evaluate(ref.fit_expr(
        key, coefficients=tuple(str(x) for x in TABLE_V_UNROUNDED_COEFFS[key])),
        {ref.ETA: eta, ref.CHI_PN: chi_pn}) for key in ref.FIT_NAMES}
    pieces = ref.amplitude_pieces(
        eta=eta, delta=delta, chi_s=chi_s, chi_a=chi_a,
        f_rd=exact(.09), f_damp=exact(.014), **fitted)
    f3 = float(pieces["f3"])
    mass_seconds = (case[0] + case[1]) * MSUN_SEC

    def remnant_provider(m1, m2, s1, s2):
        return array(case[0] + case[1]), array(0), array(.09), array(.014)

    def amplitude(u):
        kwargs = dict(f=(u / mass_seconds).reshape((1,)), mass1=array(case[0]),
                      mass2=array(case[1]), chi1=array(case[2]), chi2=array(case[3]),
                      distance=array(400), inclination=array(.7),
                      remnant_provider=remnant_provider)
        hp, _ = model()(**kwargs) if name == "torch" else model.evaluate(**kwargs)
        return hp[0].abs() if name == "torch" else abs(hp[0])

    baseline = amplitude(array(.01))
    expected_base = float(ref.evaluate(pieces["inspiral"], {ref.U: .01})) * .01**(-7 / 6)
    for u, key in ((.01, "inspiral"), (.014 + .4 * (f3 - .014), "intermediate"),
                   (f3 + .01, "merger_ringdown")):
        if name == "torch":
            import torch
            derivative = torch.autograd.functional.jacobian(
                lambda value: amplitude(value) / baseline, array(u))
        else:
            import jax
            derivative = jax.grad(lambda value: amplitude(value) / baseline)(array(u))
        scaled = pieces[key] * ref.U**sp.Rational(-7, 6)
        expected = float(ref.evaluate(sp.diff(scaled, ref.U), {ref.U: u})) / expected_base
        np.testing.assert_allclose(float(derivative), expected, rtol=3e-7, atol=3e-7)
