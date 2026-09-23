"""Complex-polarization IMRPhenomD check against the composed scalar reference."""

import importlib
from functools import lru_cache

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import imrphenomd_waveform as ref
from diffgw_reference import imrphenomd_intrinsic as full_ref

CASES = ((30., 30., 0., 0.), (45., 15., .45, -.25))


def test_composed_component_bindings_are_complete():
    assert set(ref.COMPONENT_IDS.values()) == set(ref.COMPONENT_SOURCES)
    assert set(ref.COMPONENT_BINDINGS) == set(ref.COMPONENT_SOURCES)
    assert ref.COMPONENT_VERSIONS[:2] == (ref.amp.SPEC_VERSION, ref.phase.SPEC_VERSION)
    for binding in ref.COMPONENT_BINDINGS.values():
        assert callable(getattr(ref, binding["reference"]))
        for frontend in ("torch", "jax"):
            module_name, name = binding[frontend]
            target = importlib.import_module(module_name)
            for part in name.split("."):
                target = getattr(target, part)
            assert callable(target)
        assert all(callable(globals()[name]) for name in binding["tests"])


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device == "mps":
        pytest.fail("the composed reference requires FP64")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.imrphenomd import IMRPhenomD
        if reference_device == "cuda" and not torch.cuda.is_available():
            pytest.fail("Torch CUDA unavailable")
        yield "torch", IMRPhenomD, lambda x: torch.as_tensor(
            x, dtype=torch.float64, device=reference_device)
        return
    jax = pytest.importorskip("jax")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd import IMRPhenomD
        devices = jax.devices("gpu" if reference_device == "cuda" else "cpu")
        with jax.default_device(devices[0]):
            yield "jax", IMRPhenomD, lambda x: jax.device_put(
                jnp.asarray(x, dtype=jnp.float64), devices[0])
    finally:
        jax.config.update("jax_enable_x64", previous)


def _numbers(value, backend_name):
    return np.asarray(value.detach().cpu() if backend_name == "torch" else value)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("convention", ("coalescence", "reference_frequency"))
def test_complete_complex_polarizations(backend, case, convention):
    """Absolute SI normalization, both complex signs, all branches and cutoff."""
    from diffgw_core.constants import MSUN_SEC
    from diffgw_core.fits import TABLE_V_UNROUNDED_COEFFS

    name, model, array = backend
    rows = {key: tuple(str(x) for x in TABLE_V_UNROUNDED_COEFFS[key])
            for key in set(ref.amp.FIT_NAMES) | set(ref.phase.MODEL_FIT_NAMES)}
    spec = ref.compose(*case, 400., .09, .014, coefficient_rows=rows)
    m_sec_frontend = sum(case[:2]) * MSUN_SEC
    # Both sides of amplitude/phase joins, merger-ringdown, and the default cutoff.
    us = np.array([.010, .014-1e-6, .014+1e-6, .018-1e-6,
                   .018+1e-6, .024, .045-1e-6, .045+1e-6,
                   .070, .095, .150, .199, .201])
    frequencies = us / m_sec_frontend
    ref_frequency = .024 / m_sec_frontend

    def remnant_provider(*_):
        return (array(sum(case[:2])), array(0), array(.09), array(.014))

    kwargs = dict(f=array(frequencies), mass1=array(case[0]), mass2=array(case[1]),
                  chi1=array(case[2]), chi2=array(case[3]), distance=array(400.),
                  inclination=array(.7), tc=array(.003), phic=array(.23),
                  remnant_provider=remnant_provider, return_phase=True,
                  phasing_convention=convention)
    if convention == "reference_frequency":
        kwargs["f_ref"] = array(ref_frequency)
    actual = model()(**kwargs) if name == "torch" else model.evaluate(**kwargs)
    actual_plus, actual_cross, actual_phase = (_numbers(x, name) for x in actual)
    expected = np.array([
        spec.evaluate(f, inclination=.7, tc=.003, phic=.23,
                      convention=convention,
                      f_ref_hz=ref_frequency if convention == "reference_frequency" else None)
        for f in frequencies], dtype=complex)
    # Comparison is direct, without phase/time alignment or amplitude rescaling.
    np.testing.assert_allclose(actual_plus, expected[:, 0], rtol=3e-8, atol=1e-31)
    np.testing.assert_allclose(actual_cross, expected[:, 1], rtol=3e-8, atol=1e-31)
    np.testing.assert_allclose(actual_phase[:-1], expected[:-1, 2].real,
                               rtol=3e-8, atol=3e-7)
    assert actual_plus[-1] == actual_cross[-1] == 0


def test_polarization_geometry_and_distance_scaling():
    """Analytic limiting identities independent of the frontend comparisons."""
    a, psi = sp.symbols("a psi", real=True)
    hp, hc = ref.polarizations(a, psi, sp.pi / 2)
    assert sp.simplify(hc) == 0
    assert sp.simplify(hp - a*sp.exp(-sp.I*psi)/2) == 0
    hp, hc = ref.polarizations(a, psi, 0)
    assert sp.simplify(hc + sp.I*hp) == 0
    d = sp.symbols("d", positive=True)
    f, m, eta = sp.symbols("f m eta", positive=True)
    assert sp.simplify(ref.physical_scale(f,m,2*d,eta)
                       / ref.physical_scale(f,m,d,eta) - sp.Rational(1,2)) == 0


@pytest.fixture(scope="module")
def derivative_spec():
    """Use the declared unequal-spin binary and explicit inherited fit rows."""
    from diffgw_core.constants import MSUN_SEC
    from diffgw_core.fits import TABLE_V_UNROUNDED_COEFFS

    case = CASES[1]
    rows = {key: tuple(str(x) for x in TABLE_V_UNROUNDED_COEFFS[key])
            for key in set(ref.amp.FIT_NAMES) | set(ref.phase.MODEL_FIT_NAMES)}
    return case, ref.compose(*case, 400., .09, .014, coefficient_rows=rows), sum(case[:2])*MSUN_SEC


def _complex_derivative(backend, case, m_sec, u, parameter, convention):
    name, model, array = backend
    frequency = u / m_sec
    f_ref = .024 / m_sec
    values = {"frequency_hz": frequency, "distance_mpc": 400.,
              "inclination": .7, "tc": .003, "phic": .23, "mass_scale": 1.}

    def remnant_provider(m1, m2, *_):
        return m1 + m2, array(0.), array(.09), array(.014)

    def output(value):
        kwargs = dict(f=array([frequency]), mass1=array(case[0]),
                      mass2=array(case[1]), chi1=array(case[2]),
                      chi2=array(case[3]), distance=array(400.),
                      inclination=array(.7), tc=array(.003), phic=array(.23),
                      remnant_provider=remnant_provider,
                      phasing_convention=convention)
        if convention == "reference_frequency":
            kwargs["f_ref"] = array(f_ref)
        if parameter == "mass_scale":
            kwargs["mass1"] = value * array(case[0])
            kwargs["mass2"] = value * array(case[1])
        elif parameter == "frequency_hz":
            kwargs["f"] = value.reshape((1,))
        else:
            kwargs[parameter if parameter != "distance_mpc" else "distance"] = value
        hp, hc = model()(**kwargs) if name == "torch" else model.evaluate(**kwargs)
        if name == "torch":
            import torch
            return torch.stack((hp[0].real, hp[0].imag,
                                hc[0].real, hc[0].imag))
        import jax.numpy as jnp
        return jnp.stack((hp[0].real, hp[0].imag,
                          hc[0].real, hc[0].imag))

    if name == "torch":
        import torch
        actual = torch.autograd.functional.jacobian(output, array(values[parameter]))
    else:
        import jax
        actual = jax.jacfwd(output)(array(values[parameter]))
    return np.asarray(_numbers(actual, name)), frequency, f_ref


def _assert_complex_derivative(actual, expected):
    expected = np.array([expected[0].real, expected[0].imag,
                         expected[1].real, expected[1].imag])
    scale = np.max(np.abs(expected))
    assert scale > 0
    np.testing.assert_allclose(actual / scale, expected / scale,
                               rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("u,convention", ((.012, "coalescence"),
                                          (.028, "coalescence"),
                                          (.080, "coalescence"),
                                          (.028, "reference_frequency")))
def test_complex_frequency_gradients(backend, derivative_spec, u, convention):
    """Compare the full complex frequency slope in all smooth branches."""
    case, spec, m_sec = derivative_spec
    actual, frequency, f_ref = _complex_derivative(
        backend, case, m_sec, u, "frequency_hz", convention)
    expected = spec.evaluate_derivative(
        frequency, "frequency_hz", inclination=.7, tc=.003, phic=.23,
        convention=convention,
        f_ref_hz=f_ref if convention == "reference_frequency" else None)
    _assert_complex_derivative(actual, expected)


@pytest.mark.parametrize("parameter", ("distance_mpc", "inclination", "tc", "phic"))
def test_complex_extrinsic_gradients(backend, derivative_spec, parameter):
    """Check four independently derived polarizations/phase identities."""
    case, spec, m_sec = derivative_spec
    actual, frequency, _ = _complex_derivative(
        backend, case, m_sec, .028, parameter, "coalescence")
    expected = spec.evaluate_derivative(
        frequency, parameter, inclination=.7, tc=.003, phic=.23)
    _assert_complex_derivative(actual, expected)


@pytest.mark.parametrize("u,convention", ((.012, "coalescence"),
                                          (.028, "coalescence"),
                                          (.080, "coalescence"),
                                          (.028, "reference_frequency")))
def test_complex_mass_scale_gradient(backend, derivative_spec, u, convention):
    """Scale both masses together while holding eta, spins and QNMs fixed."""
    case, spec, m_sec = derivative_spec
    actual, frequency, f_ref = _complex_derivative(
        backend, case, m_sec, u, "mass_scale", convention)
    expected = spec.evaluate_derivative(
        frequency, "mass_scale", inclination=.7, tc=.003, phic=.23,
        convention=convention,
        f_ref_hz=f_ref if convention == "reference_frequency" else None)
    _assert_complex_derivative(actual, expected)


def test_full_intrinsic_component_bindings_are_complete():
    assert set(full_ref.COMPONENT_IDS.values()) == set(full_ref.COMPONENT_SOURCES)
    assert set(full_ref.COMPONENT_BINDINGS) == set(full_ref.COMPONENT_SOURCES)
    assert full_ref.COMPONENT_VERSIONS == (
        ref.SPEC_VERSION, full_ref.remnant.SPEC_VERSION)
    for binding in full_ref.COMPONENT_BINDINGS.values():
        assert callable(getattr(full_ref, binding["reference"]))
        for frontend in ("torch", "jax"):
            module_name, name = binding[frontend]
            target = importlib.import_module(module_name)
            for part in name.split("."):
                target = getattr(target, part)
            assert callable(target)
        assert all(callable(globals()[name]) for name in binding["tests"])


@pytest.fixture(scope="module")
def full_intrinsic_inputs():
    """Explicit inherited fit digits and numerical QNM knots at the test boundary."""
    from diffgw_core.constants import MSUN_SEC
    from diffgw_core.fits import (
        TABLE_V_UNROUNDED_COEFFS, FINAL_SPIN_ETA_POLY,
        FINAL_SPIN_S_COEFFS, ERAD_ETA_NUMERATOR, ERAD_S_NUMERATOR,
        ERAD_S_DENOMINATOR)
    from diffgw_core.qnm_data import KNOT_SPINS, ROUNDED_CYCLIC_ORDINATES

    case = CASES[1]
    rows = {key: tuple(str(x) for x in TABLE_V_UNROUNDED_COEFFS[key])
            for key in set(ref.amp.FIT_NAMES) | set(ref.phase.MODEL_FIT_NAMES)}
    remnant_rows = {
        "spin_eta": tuple(map(str, FINAL_SPIN_ETA_POLY)),
        "spin_s_eta": tuple(str(row[0]) for row in FINAL_SPIN_S_COEFFS),
        "spin_s_eta2": tuple(str(row[1]) for row in FINAL_SPIN_S_COEFFS),
        "energy_ns": tuple(map(str, ERAD_ETA_NUMERATOR)),
        "energy_num": tuple(map(str, ERAD_S_NUMERATOR)),
        "energy_den": tuple(map(str, ERAD_S_DENOMINATOR)),
    }
    options = dict(knot_spins=KNOT_SPINS, cyclic_ordinates=ROUNDED_CYCLIC_ORDINATES,
                   coefficient_rows=rows, remnant_rows=remnant_rows)
    return case, sum(case[:2]) * MSUN_SEC, options


@pytest.mark.parametrize("parameter", ("mass_ratio", "chi1", "chi2"))
def test_full_remnant_qnm_and_tangents(backend, full_intrinsic_inputs, parameter):
    """Check analytic remnant plus spline chain slopes before waveform composition."""
    case, _, options = full_intrinsic_inputs
    name, model, array = backend
    expected = full_ref.remnant_qnm(*case, **{
        key: options[key] for key in ("knot_spins", "cyclic_ordinates", "remnant_rows")})
    actual = model.remnant_properties(*(array(x) for x in case))
    np.testing.assert_allclose([float(x) for x in actual],
                               [float(x) for x in expected], rtol=3e-11, atol=2e-12)
    tangent = full_ref.qnm_tangent(*case, parameter, **{
        key: options[key] for key in ("knot_spins", "cyclic_ordinates", "remnant_rows")})
    total = sum(case[:2])
    q0 = case[0] / case[1]

    def output(value):
        m1, m2 = ((total * value / (1 + value), total / (1 + value))
                  if parameter == "mass_ratio" else (array(case[0]), array(case[1])))
        chi1 = value if parameter == "chi1" else array(case[2])
        chi2 = value if parameter == "chi2" else array(case[3])
        result = model.remnant_properties(m1, m2, chi1, chi2)
        if name == "torch":
            import torch
            return torch.stack(result)
        import jax.numpy as jnp
        return jnp.stack(result)

    value = array(q0 if parameter == "mass_ratio" else case[2 if parameter == "chi1" else 3])
    if name == "torch":
        import torch
        actual_tangent = torch.autograd.functional.jacobian(output, value)
    else:
        import jax
        actual_tangent = jax.jacfwd(output)(value)
    np.testing.assert_allclose(_numbers(actual_tangent, name), tangent,
                               rtol=2e-8, atol=2e-10)


@pytest.mark.parametrize("u,convention", ((.012, "coalescence"),
                                          (.028, "coalescence"),
                                          (.080, "coalescence"),
                                          (.028, "reference_frequency")))
def test_full_intrinsic_polarizations(backend, full_intrinsic_inputs, u, convention):
    """Check the unmodified production remnant/QNM path in all waveform regions."""
    case, m_sec, options = full_intrinsic_inputs
    name, model, array = backend
    frequency = u / m_sec
    f_ref = .024 / m_sec
    spec = full_ref.compose(*case, 400., **options)
    kwargs = dict(f=array([frequency]), mass1=array(case[0]), mass2=array(case[1]),
                  chi1=array(case[2]), chi2=array(case[3]), distance=array(400.),
                  inclination=array(.7), tc=array(.003), phic=array(.23),
                  phasing_convention=convention)
    if convention == "reference_frequency":
        kwargs["f_ref"] = array(f_ref)
    actual = model()(**kwargs) if name == "torch" else model.evaluate(**kwargs)
    expected = spec.evaluate(frequency, inclination=.7, tc=.003, phic=.23,
                             convention=convention,
                             f_ref_hz=f_ref if convention == "reference_frequency" else None)
    np.testing.assert_allclose([complex(_numbers(x[0], name)) for x in actual],
                               expected[:2], rtol=3e-8, atol=1e-31)


@pytest.fixture(scope="module")
def high_spin_options(full_intrinsic_inputs):
    """Pass the numerical Kerr data explicitly to the independent reference."""
    from diffgw_core.kerr_qnm_220_data import (
        DEGREE, KNOTS_220, COEFFS_220, KNOTS_2M20, COEFFS_2M20)

    _, _, options = full_intrinsic_inputs
    return {**options, "generated_splines": {
        "degree": DEGREE,
        "positive": (KNOTS_220, COEFFS_220),
        "negative": (KNOTS_2M20, COEFFS_2M20),
    }}


@pytest.mark.parametrize("case,clamped", (
    ((150., 1., .999, .999), False),
    ((1000., 1., 1., 1.), True),
    ((2000., 1., -1., -1.), False),
))
@pytest.mark.parametrize("convention", ("coalescence", "reference_frequency"))
def test_high_spin_complete_complex_waveform(
        backend, high_spin_options, case, clamped, convention):
    """Exercise signed Kerr continuation, remnant cap, joins and cutoff."""
    name, model, array = backend
    options = high_spin_options
    raw_spin = float(full_ref.remnant_expressions(
        *case, remnant_rows=options["remnant_rows"])[2])
    expected_spin = float(np.clip(raw_spin, -.999, .999))
    assert abs(expected_spin) > .994
    assert (abs(raw_spin) > .999) == clamped
    expected_remnant = full_ref.remnant_qnm(*case, **{
        key: options[key] for key in (
            "knot_spins", "cyclic_ordinates", "remnant_rows", "generated_splines")})
    actual_remnant = model.remnant_properties(*(array(value) for value in case))
    np.testing.assert_allclose([float(value) for value in actual_remnant],
                               [float(value) for value in expected_remnant],
                               rtol=3e-11, atol=2e-12)

    spec = full_ref.compose(*case, 400., **options)
    joins = [float(spec.amplitude[key]) for key in ("f1", "f3")]
    joins += [float(spec.phase[key]) for key in ("f1", "f2")]
    us = np.array(sorted({.010, .025, .199, .201,
                          *(join + side * 1e-5 for join in joins
                            for side in (-1, 1))}))
    m_sec = float(spec.m_sec)
    frequencies = us / m_sec
    f_ref = .024 / m_sec
    kwargs = dict(f=array(frequencies), mass1=array(case[0]),
                  mass2=array(case[1]), chi1=array(case[2]),
                  chi2=array(case[3]), distance=array(400.),
                  inclination=array(.7), tc=array(.003), phic=array(.23),
                  phasing_convention=convention, return_phase=True)
    if convention == "reference_frequency":
        kwargs["f_ref"] = array(f_ref)
    actual = model()(**kwargs) if name == "torch" else model.evaluate(**kwargs)
    plus, cross, phase = (_numbers(value, name) for value in actual)
    expected = np.array([
        spec.evaluate(f, inclination=.7, tc=.003, phic=.23,
                      convention=convention,
                      f_ref_hz=f_ref if convention == "reference_frequency" else None)
        for f in frequencies], dtype=complex)
    np.testing.assert_allclose(plus[:-1], expected[:-1, 0], rtol=3e-7, atol=1e-40)
    np.testing.assert_allclose(cross[:-1], expected[:-1, 1], rtol=3e-7, atol=1e-40)
    np.testing.assert_allclose(phase[:-1], expected[:-1, 2].real,
                               rtol=0, atol=3e-7)
    assert plus[-1] == cross[-1] == 0


@pytest.mark.parametrize("case,clamped", (
    ((150., 1., .999, .999), False),
    ((1000., 1., .9998, .9998), True),
    ((2000., 1., -.999, -.999), False),
))
@pytest.mark.parametrize("parameter", ("mass_ratio", "chi1", "chi2"))
@pytest.mark.parametrize("u,convention", ((.012, "coalescence"),
                                          (.028, "coalescence"),
                                          (.080, "reference_frequency")))
def test_high_spin_complex_intrinsic_derivatives(
        backend, high_spin_options, case, clamped, parameter, u, convention):
    """Check the full complex Jacobian through both Kerr branches and the cap."""
    from diffgw_core.constants import MSUN_SEC

    raw_spin = float(full_ref.remnant_expressions(
        *case, remnant_rows=high_spin_options["remnant_rows"])[2])
    assert abs(raw_spin) > .994
    assert (abs(raw_spin) > .999) == clamped
    m_sec = sum(case[:2]) * MSUN_SEC
    expected = full_ref.polarization_derivative(
        *case, 400., u / m_sec, parameter, **high_spin_options,
        inclination=.7, tc=.003, phic=.23, convention=convention,
        f_ref_hz=.024 / m_sec if convention == "reference_frequency" else None)
    actual = _full_intrinsic_autodiff(
        backend, case, m_sec, parameter, u, convention)
    _assert_complex_derivative(actual, expected)


def test_high_spin_clamp_crossing_stencil_is_rejected(high_spin_options):
    """A finite difference must remain on the active remnant-cap branch."""
    from diffgw_core.constants import MSUN_SEC

    case = (1000., 1., .9998, .9998)
    with pytest.raises(ValueError, match="remnant cap"):
        full_ref.polarization_derivative(
            *case, 400., .028 / (sum(case[:2]) * MSUN_SEC), "chi1",
            **high_spin_options, step="0.0001")


@pytest.mark.parametrize("case", ((150., 1., .999, .999),
                                  (1000., 1., .9998, .9998),
                                  (2000., 1., -.999, -.999)))
def test_high_spin_derivative_stencil_converges(high_spin_options, case):
    """The independent high-spin mass-ratio slope is stable to step halving."""
    from diffgw_core.constants import MSUN_SEC

    args = (*case, 400., .028 / (sum(case[:2]) * MSUN_SEC), "mass_ratio")
    kwargs = dict(high_spin_options, inclination=.7, tc=.003, phic=.23)
    coarse = full_ref.polarization_derivative(
        *args, step="0.0002", **kwargs)
    fine = full_ref.polarization_derivative(
        *args, step="0.0001", **kwargs)
    scale = max(abs(z) for z in fine)
    np.testing.assert_allclose(np.asarray(coarse) / scale,
                               np.asarray(fine) / scale,
                               rtol=2e-7, atol=2e-7)


@pytest.fixture(scope="module")
def full_intrinsic_expected(full_intrinsic_inputs):
    case, m_sec, options = full_intrinsic_inputs

    @lru_cache(maxsize=None)
    def expected(parameter, u, convention):
        f_ref = .024 / m_sec
        return full_ref.polarization_derivative(
            *case, 400., u / m_sec, parameter, **options,
            inclination=.7, tc=.003, phic=.23, convention=convention,
            f_ref_hz=f_ref if convention == "reference_frequency" else None)

    return expected


@pytest.mark.parametrize("parameter", ("mass_ratio", "chi1", "chi2"))
@pytest.mark.parametrize("u,convention", ((.012, "coalescence"),
                                          (.028, "coalescence"),
                                          (.080, "coalescence"),
                                          (.028, "reference_frequency")))
def test_full_intrinsic_derivatives(backend, full_intrinsic_inputs,
                                    full_intrinsic_expected, parameter, u, convention):
    """Compare full-chain fourth-order scalar slopes with Torch/JAX autodiff."""
    case, m_sec, _ = full_intrinsic_inputs
    actual = _full_intrinsic_autodiff(backend, case, m_sec, parameter, u, convention)
    _assert_complex_derivative(actual, full_intrinsic_expected(parameter, u, convention))


def _full_intrinsic_autodiff(backend, case, m_sec, parameter, u,
                            convention="coalescence"):
    """Return the frontend complex Jacobian for a fixed-total-mass parameter."""
    name, model, array = backend
    total = sum(case[:2])
    q0 = case[0] / case[1]
    frequency = u / m_sec

    def output(value):
        m1, m2 = ((total * value / (1 + value), total / (1 + value))
                  if parameter == "mass_ratio" else (array(case[0]), array(case[1])))
        kwargs = dict(f=array([frequency]), mass1=m1, mass2=m2,
                      chi1=value if parameter == "chi1" else array(case[2]),
                      chi2=value if parameter == "chi2" else array(case[3]),
                      distance=array(400.), inclination=array(.7), tc=array(.003),
                      phic=array(.23), phasing_convention=convention)
        if convention == "reference_frequency":
            kwargs["f_ref"] = array(.024 / m_sec)
        hp, hc = model()(**kwargs) if name == "torch" else model.evaluate(**kwargs)
        if name == "torch":
            import torch
            return torch.stack((hp[0].real, hp[0].imag, hc[0].real, hc[0].imag))
        import jax.numpy as jnp
        return jnp.stack((hp[0].real, hp[0].imag, hc[0].real, hc[0].imag))

    value = array(q0 if parameter == "mass_ratio" else case[2 if parameter == "chi1" else 3])
    if name == "torch":
        import torch
        actual = torch.autograd.functional.jacobian(output, value)
    else:
        import jax
        actual = jax.jacfwd(output)(value)
    return _numbers(actual, name)


def test_full_intrinsic_derivative_stencil_converges(full_intrinsic_inputs):
    """The scalar derivative is stable to halving its independent stencil."""
    case, m_sec, options = full_intrinsic_inputs
    kwargs = dict(inclination=.7, tc=.003, phic=.23)
    coarse = full_ref.polarization_derivative(
        *case, 400., .08 / m_sec, "chi1", **options, step="0.0001", **kwargs)
    fine = full_ref.polarization_derivative(
        *case, 400., .08 / m_sec, "chi1", **options, step="0.00005", **kwargs)
    _assert_complex_derivative(np.array([fine[0].real, fine[0].imag,
                                         fine[1].real, fine[1].imag]), coarse)


def test_full_intrinsic_explicit_knot_crossing_is_rejected(full_intrinsic_inputs):
    case, m_sec, options = full_intrinsic_inputs
    with pytest.raises(ValueError, match="QNM spline knot"):
        full_ref.polarization_derivative(
            *case, 400., .08 / m_sec, "chi1", **options, step="0.001")


def test_full_intrinsic_default_step_avoids_knot(full_intrinsic_inputs):
    case, m_sec, options = full_intrinsic_inputs
    near_knot = (case[0], case[1], .4485, case[3])
    final_spin = float(full_ref.remnant_expressions(
        *near_knot, remnant_rows=options["remnant_rows"])[2])
    assert .720 < final_spin < .7201
    with pytest.raises(ValueError, match="QNM spline knot"):
        full_ref.polarization_derivative(
            *near_knot, 400., .08 / m_sec, "chi1", **options,
            step="0.0001")
    kwargs = dict(inclination=.7, tc=.003, phic=.23)
    adaptive = full_ref.polarization_derivative(
        *near_knot, 400., .08 / m_sec, "chi1", **options, **kwargs)
    fine = full_ref.polarization_derivative(
        *near_knot, 400., .08 / m_sec, "chi1", **options,
        step="0.0000125", **kwargs)
    np.testing.assert_allclose(adaptive, fine, rtol=1e-10, atol=1e-31)


def test_full_intrinsic_moving_join_step_policy(full_intrinsic_inputs):
    case, _, options = full_intrinsic_inputs
    spec = full_ref.compose(*case, 400., **options)
    frequency = (float(spec.amplitude["f3"]) + 1e-7) / float(spec.m_sec)
    kwargs = dict(inclination=.7, tc=.003, phic=.23)
    with pytest.raises(ValueError, match="waveform branch"):
        full_ref.polarization_derivative(
            *case, 400., frequency, "chi2", **options,
            step="0.0001", **kwargs)
    adaptive = full_ref.polarization_derivative(
        *case, 400., frequency, "chi2", **options, **kwargs)
    fine = full_ref.polarization_derivative(
        *case, 400., frequency, "chi2", **options,
        step="0.00000625", **kwargs)
    np.testing.assert_allclose(adaptive, fine, rtol=1e-9, atol=1e-31)


@pytest.mark.parametrize("case", ((30., 30., .35, -.2),
                                  (40., 20., -.6, -.3)))
@pytest.mark.parametrize("parameter", ("mass_ratio", "chi1", "chi2"))
def test_full_intrinsic_derivative_grid(backend, full_intrinsic_inputs,
                                        case, parameter):
    """Cover equal masses and negative spins away from spline boundaries."""
    from diffgw_core.constants import MSUN_SEC

    _, _, options = full_intrinsic_inputs
    m_sec = sum(case[:2]) * MSUN_SEC
    expected = full_ref.polarization_derivative(
        *case, 400., .028 / m_sec, parameter, **options,
        inclination=.7, tc=.003, phic=.23)
    actual = _full_intrinsic_autodiff(backend, case, m_sec, parameter, .028)
    _assert_complex_derivative(actual, expected)


@pytest.mark.parametrize("join", ("amplitude", "phase"))
@pytest.mark.parametrize("side", (-1, 1))
def test_full_intrinsic_derivative_near_moving_join(
        backend, full_intrinsic_inputs, join, side):
    """Check both sides of the amplitude peak and ringdown phase join."""
    case, m_sec, options = full_intrinsic_inputs
    spec = full_ref.compose(*case, 400., **options)
    u = float(spec.amplitude["f3"] if join == "amplitude"
              else spec.phase["f2"]) + side * .0002
    expected = full_ref.polarization_derivative(
        *case, 400., u / m_sec, "chi1", **options,
        inclination=.7, tc=.003, phic=.23)
    actual = _full_intrinsic_autodiff(backend, case, m_sec, "chi1", u)
    _assert_complex_derivative(actual, expected)


def test_full_intrinsic_spin_endpoint_policy(full_intrinsic_inputs):
    case, m_sec, options = full_intrinsic_inputs
    with pytest.raises(ValueError, match="central spin derivative requires an interior"):
        full_ref.polarization_derivative(
            case[0], case[1], 1., case[3], 400., .028 / m_sec,
            "chi1", **options)
    with pytest.raises(ValueError, match="outside the supplied knot domain"):
        full_ref.remnant_qnm(
            *case, knot_spins=options["knot_spins"][:3],
            cyclic_ordinates=options["cyclic_ordinates"][:3],
            remnant_rows=options["remnant_rows"])
