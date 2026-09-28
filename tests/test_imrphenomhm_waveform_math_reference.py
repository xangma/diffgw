"""Full HM qualification with no frontend-derived reference inputs."""

import ast
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")
pytest.importorskip("sympy")

from diffgw_reference import imrphenomhm_waveform as ref
from diffgw_reference.imrphenomhm_frequency_map import eq8_phase_frequency_map, outer_map_cutoffs

MODES = ref.SUPPORTED_MODES
CASES = ((35., 20., .2, -.1), (30., 30., .35, -.2))
EXTRINSIC = dict(inclination=.93, phi0=.27, phic=.37, tc=.008)


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device != "cpu":
        pytest.fail("The maintained independent HM qualification is FP64 CPU")
    name = request.param
    module = pytest.importorskip(f"{name}wave.approximants.imrphenom_hm")
    if name == "torch":
        torch = pytest.importorskip("torch")
        yield name, module.IMRPhenomHM(), lambda x: torch.as_tensor(x, dtype=torch.float64)
        return
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        with jax.default_device(jax.devices("cpu")[0]):
            yield name, module.IMRPhenomHM.evaluate, lambda x: jnp.asarray(x, dtype=jnp.float64)
    finally:
        jax.config.update("jax_enable_x64", previous)


def _arguments(case, transition, reference_frequency):
    return dict(zip(("mass1", "mass2", "chi1", "chi2"), case)) | dict(
        distance=100., calibration="published", f_ref=reference_frequency,
        hm_mapping_transitions=transition, return_unprojected_modes=True,
        **EXTRINSIC, **ref.SOURCE_PN_VARIANTS)


def _assert_close(actual, expected, rtol=3e-8):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.all(np.isfinite(actual))
    scale = max(float(np.max(np.abs(expected))), 1e-100)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=scale * 2e-10)


def _branch_samples(spec, mode, transition):
    """Outer joins and inverse images of D amplitude/phase joins, in Mf."""
    ell, m = mode
    fr, _ = spec.hm_qnm[mode]
    fr22 = spec.hm_qnm[(2, 2)][0]
    f0a, f0p = outer_map_cutoffs(transition, fr, fr22)
    joins = {f0a, f0p, fr}
    if mode == (2, 2):
        joins.update(spec.amplitude_joins + spec.phase_joins)
    else:
        for is_phase, f0, carrier_joins in (
                (False, f0a, spec.amplitude_joins),
                (True, f0p, spec.phase_joins)):
            low_endpoint = 2 * f0 / m
            for g in carrier_joins:
                if g <= low_endpoint:
                    joins.add(m * g / 2)
                elif g <= fr22:
                    joins.add(f0 + (g - low_endpoint) * (fr - f0) / (fr22 - low_endpoint))
                else:
                    joins.add(g * fr / fr22 if is_phase else g + fr - fr22)
    points = {mp.mpf(".005"), mp.mpf(".035"), mp.mpf(".12")}
    for join in joins:
        for factor in (mp.mpf(".999999"), mp.mpf(1), mp.mpf("1.000001")):
            points.add(join * factor)
    return np.array(sorted(float(point) for point in points))


def test_independent_reference_import_boundary():
    """The complete oracle may share data, never production evaluators/fits."""
    tree = ast.parse(Path(ref.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith(("torch", "jax")) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith(("torch", "jax"))
            if module.startswith("diffgw_core"):
                assert module in {"diffgw_core.qnm_data", "diffgw_core.kerr_qnm_220_data",
                                  "diffgw_core.hm_qnm_n61_data"}
    assert set(ref.COMPONENT_BINDINGS) == set(ref.COMPONENT_SOURCES)
    for binding in ref.COMPONENT_BINDINGS.values():
        assert callable(getattr(ref, binding["reference"]))
        assert all(callable(globals()[name]) for name in binding["tests"])


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("transition", ("london_2018", "phenomd_runtime"))
@pytest.mark.parametrize("reference_frequency", (None, 40.))
def test_published_complete_modes_and_projection(backend, case, transition, reference_frequency):
    _complete_modes_and_projection(backend, case, transition, reference_frequency, "legacy")


def _complete_modes_and_projection(backend, case, transition, reference_frequency, phase,
                                   convention="legacy", inclination=None):
    _, native, array = backend
    spec = ref.compose(*case)
    arguments = _arguments(case, transition, reference_frequency) | dict(
        hm_reference_phase=phase, hm_phase_convention=convention)
    extrinsic = EXTRINSIC | ({} if inclination is None else dict(inclination=inclination))
    arguments.update(extrinsic)
    options = dict(transition=transition, f_ref_hz=reference_frequency,
                   hm_reference_phase=phase, hm_phase_convention=convention, **extrinsic)
    for mode in MODES:
        with mp.workdps(50):
            frequencies = _branch_samples(spec, mode, transition) / float(spec.m_sec)
            expected = [spec.evaluate(str(f), modes=(mode,), **options) for f in frequencies]
        hp, hc, modes = native(f=array(frequencies), modes=(mode,), **arguments)
        _assert_close(modes[mode], [complex(row[2][mode]) for row in expected])
        _assert_close(hp, [complex(row[0]) for row in expected])
        _assert_close(hc, [complex(row[1]) for row in expected])
    # A combined six-mode call separately checks the sum and mode dictionary.
    frequencies = np.array((.005, .035, .12)) / float(spec.m_sec)
    hp, hc, modes = native(f=array(frequencies), modes=MODES, **arguments)
    expected = [spec.evaluate(str(f), **options) for f in frequencies]
    _assert_close(hp, [complex(row[0]) for row in expected])
    _assert_close(hc, [complex(row[1]) for row in expected])
    assert set(modes) == set(MODES)


def test_published_symmetry_zeros(backend):
    _, native, array = backend
    case = (30., 30., .2, .2)
    spec = ref.compose(*case)
    frequencies = np.array((20., 100., 400.))
    hp, hc, modes = native(f=array(frequencies), modes=MODES,
                          **_arguments(case, "london_2018", 40.))
    expected = [spec.evaluate(str(f), transition="london_2018", f_ref_hz=40.,
                              **EXTRINSIC) for f in frequencies]
    for mode in ((2, 1), (3, 3), (4, 3)):
        assert np.array_equal(np.asarray(modes[mode]), np.zeros(3))
        assert all(row[2][mode] == 0 for row in expected)
    _assert_close(hp, [complex(row[0]) for row in expected])
    _assert_close(hc, [complex(row[1]) for row in expected])


def _reference_vector(spec, values, transition, convention="legacy"):
    frequency, inclination, phi0, phic, tc, distance = values
    hp, hc, modes = spec.evaluate(
        frequency, transition=transition, f_ref_hz=40., inclination=inclination,
        phi0=phi0, phic=phic, tc=tc, hm_phase_convention=convention)
    # Exact distance scaling avoids rebuilding unchanged symbolic fits.
    scale = 100 / distance
    values = (hp, hc, *(modes[lm] for lm in MODES))
    return tuple(scale * part for value in values for part in (value.real, value.imag))


def _reference_jacobian(spec, values, transition, convention="legacy"):
    """Converged fourth-order differences, retaining 50-digit arithmetic."""
    with mp.workdps(50):
        values = tuple(mp.mpf(str(value)) for value in values)
        columns = []
        for index, center in enumerate(values):
            step = mp.mpf(".00001") * max(abs(center), mp.mpf(".01"))
            def estimate(h):
                rows = []
                for offset in (-2, -1, 1, 2):
                    point = list(values)
                    point[index] += offset * h
                    rows.append(_reference_vector(spec, point, transition, convention))
                return tuple((rows[0][i] - 8 * rows[1][i] + 8 * rows[2][i] - rows[3][i])
                             / (12 * h) for i in range(len(rows[0])))
            coarse, fine = estimate(step), estimate(step / 2)
            scale = max(abs(value) for value in fine)
            assert max(abs(a - b) for a, b in zip(coarse, fine)) < scale * mp.mpf("1e-7")
            columns.append([float(value) for value in fine])
        return np.asarray(columns).T


@pytest.mark.parametrize("geometric_frequency", (.005, .035, .12))
@pytest.mark.parametrize("transition", ("london_2018", "phenomd_runtime"))
def test_published_smooth_derivatives(backend, geometric_frequency, transition):
    _smooth_derivatives(backend, geometric_frequency, transition)


def _smooth_derivatives(backend, geometric_frequency, transition, convention="legacy"):
    name, native, array = backend
    case = CASES[0]
    spec = ref.compose(*case)
    point = [geometric_frequency / float(spec.m_sec), EXTRINSIC["inclination"],
             EXTRINSIC["phi0"], EXTRINSIC["phic"], EXTRINSIC["tc"], 100.]
    arguments = _arguments(case, transition, 40.) | dict(hm_phase_convention=convention)
    for key in ("inclination", "phi0", "phic", "tc", "distance"):
        arguments.pop(key)
    if name == "torch":
        import torch
        def function(x):
            hp, hc, modes = native(f=x[0:1], inclination=x[1], phi0=x[2],
                                  phic=x[3], tc=x[4], distance=x[5], modes=MODES,
                                  **arguments)
            complex_values = torch.cat((hp.reshape(-1), hc.reshape(-1),
                                         *(modes[lm].reshape(-1) for lm in MODES)))
            return torch.view_as_real(complex_values).reshape(-1)
        actual = torch.autograd.functional.jacobian(function, array(point)).detach().numpy()
    else:
        import jax
        import jax.numpy as jnp
        def function(x):
            hp, hc, modes = native(f=x[0:1], inclination=x[1], phi0=x[2],
                                  phic=x[3], tc=x[4], distance=x[5], modes=MODES,
                                  **arguments)
            values = jnp.concatenate((hp.reshape(-1), hc.reshape(-1),
                                      *(modes[lm].reshape(-1) for lm in MODES)))
            return jnp.stack((values.real, values.imag), axis=-1).reshape(-1)
        actual = np.asarray(jax.jacfwd(function)(array(point)))
    expected = _reference_jacobian(spec, point, transition, convention)
    # Scale each physical derivative separately; tiny components near zero
    # must not inherit the much larger time-derivative absolute tolerance.
    for index in range(len(point)):
        _assert_close(actual[:, index], expected[:, index], rtol=3e-6)


@pytest.mark.parametrize("transition", ("london_2018", "phenomd_runtime"))
def test_orbital_complete_modes_and_projection(backend, transition):
    """The new composition reuses every independently sampled branch/join."""
    _complete_modes_and_projection(backend, CASES[0], transition, 40., "orbital")


@pytest.mark.parametrize("transition", ("london_2018", "phenomd_runtime"))
def test_orbital_reference_covariance(backend, transition):
    """Predict phase shifts from scalar D math, never by fitting native modes."""
    _, native, array = backend
    for case in CASES:
        spec = ref.compose(*case)
        # All mapped carrier evaluations are below Mf=.2, away from zeros.
        geometric_frequencies = (.005, .025, .06)
        for ell, m in MODES:
            fr, fr22 = spec.hm_qnm[(ell, m)][0], spec.hm_qnm[(2, 2)][0]
            _, f0 = outer_map_cutoffs(transition, fr, fr22)
            assert max(2 * f0 / m, fr22) < mp.mpf(".2")
            assert all(eq8_phase_frequency_map(x, f0, fr, fr22, m) < mp.mpf(".2")
                       for x in geometric_frequencies)
        frequencies = np.array(geometric_frequencies) / float(spec.m_sec)
        arguments = _arguments(case, transition, None) | dict(hm_reference_phase="orbital")
        outputs = {}
        for frequency in (None, 30., 40.):
            arguments["f_ref"] = frequency
            outputs[frequency] = native(f=array(frequencies), modes=MODES, **arguments)
        for before, after in ((None, 30.), (None, 40.), (30., 40.)):
            with mp.workdps(50):
                delta = spec.reference_phase_offset(after) - spec.reference_phase_offset(before)
                factors = {lm: complex(mp.exp(-mp.j * lm[1] * delta / 2)) for lm in MODES}
            for mode in MODES:
                _assert_close(outputs[after][2][mode],
                              np.asarray(outputs[before][2][mode]) * factors[mode], rtol=2e-10)
            # phic enters as +m*phic: the predicted +C/2 shift compensates
            # the carrier constant for every mode and both polarizations.
            arguments.update(f_ref=after, phic=EXTRINSIC["phic"] + float(delta / 2))
            compensated = native(f=array(frequencies), modes=MODES, **arguments)
            for index in (0, 1):
                _assert_close(compensated[index], outputs[before][index], rtol=2e-10)
            for mode in MODES:
                _assert_close(compensated[2][mode], outputs[before][2][mode], rtol=2e-10)


@pytest.mark.parametrize("calibration", ("published", "inherited"))
def test_reference_phase_reductions(backend, calibration):
    """Default output stays bitwise legacy; orbital preserves m=2 and None."""
    _, native, array = backend
    spec = ref.compose(*CASES[0])
    frequencies = array((20., 90., 210.))
    for frequency in (None, 40.):
        arguments = _arguments(CASES[0], "london_2018", frequency) | dict(calibration=calibration)
        default = native(f=frequencies, modes=MODES, **arguments)
        legacy = native(f=frequencies, modes=MODES, hm_reference_phase="legacy", **arguments)
        orbital = native(f=frequencies, modes=MODES, hm_reference_phase="orbital", **arguments)
        for index in (0, 1):
            np.testing.assert_array_equal(default[index], legacy[index])
        for mode in MODES:
            np.testing.assert_array_equal(default[2][mode], legacy[2][mode])
            if frequency is None or mode[1] == 2:
                np.testing.assert_array_equal(orbital[2][mode], legacy[2][mode])
        if frequency is None:
            for index in (0, 1):
                np.testing.assert_array_equal(orbital[index], legacy[index])
        scalar_default = spec.evaluate(90., f_ref_hz=frequency)
        assert scalar_default == spec.evaluate(90., f_ref_hz=frequency, hm_reference_phase="legacy")
        scalar_orbital = spec.evaluate(90., f_ref_hz=frequency, hm_reference_phase="orbital")
        for mode in ((2, 2), (3, 2)):
            assert scalar_default[2][mode] == scalar_orbital[2][mode]
    with pytest.raises(ValueError, match="reference phase"):
        native(f=frequencies, hm_reference_phase="unknown",
               **_arguments(CASES[0], "london_2018", 40.))
    with pytest.raises(ValueError, match="hm_reference_phase"):
        spec.evaluate(90., hm_reference_phase="unknown")


def _real_reference_values(spec, frequency, fref, convention="legacy"):
    hp, hc, modes = spec.evaluate(frequency, transition="london_2018", f_ref_hz=fref,
                                 hm_reference_phase="orbital",
                                 hm_phase_convention=convention, **EXTRINSIC)
    return tuple(part for value in (hp, hc, *(modes[lm] for lm in MODES))
                 for part in (value.real, value.imag))


@lru_cache(maxsize=2)
def _orbital_derivative_reference(convention="legacy"):
    """Scalar fourth-order derivatives in f_ref and primary mass, converged twice."""
    case = CASES[0]
    frequency = 130.
    columns = []
    with mp.workdps(50):
        for key, center in (("fref", mp.mpf(40)), ("mass1", mp.mpf(str(case[0])))):
            def estimate(step):
                rows = []
                for offset in (-2, -1, 1, 2):
                    value = center + offset * step
                    spec = (ref.compose(value, *case[1:]) if key == "mass1"
                            else ref.compose(*case))
                    rows.append(_real_reference_values(spec, frequency,
                                                        value if key == "fref" else 40,
                                                        convention))
                return tuple((rows[0][i] - 8 * rows[1][i] + 8 * rows[2][i] - rows[3][i])
                             / (12 * step) for i in range(len(rows[0])))
            step = center * mp.mpf("1e-5")
            coarse, fine = estimate(step), estimate(step / 2)
            scale = max(abs(value) for value in fine)
            assert max(abs(a - b) for a, b in zip(coarse, fine)) < scale * mp.mpf("1e-7")
            columns.append(tuple(float(value) for value in fine))
    return np.asarray(columns).T


def test_orbital_reference_derivatives(backend):
    """Independent intrinsic and reference-frequency derivatives of all outputs."""
    _orbital_reference_derivatives(backend)


def _orbital_reference_derivatives(backend, convention="legacy"):
    name, native, array = backend
    arguments = _arguments(CASES[0], "london_2018", 40.) | dict(
        hm_reference_phase="orbital", hm_phase_convention=convention)
    arguments.pop("mass1")
    arguments.pop("f_ref")
    if name == "torch":
        import torch
        def function(x):
            hp, hc, modes = native(f=array((130.,)), f_ref=x[0], mass1=x[1],
                                  modes=MODES, **arguments)
            values = torch.cat((hp.reshape(-1), hc.reshape(-1),
                                *(modes[lm].reshape(-1) for lm in MODES)))
            return torch.view_as_real(values).reshape(-1)
        actual = torch.autograd.functional.jacobian(function, array((40., CASES[0][0]))).detach().numpy()
    else:
        import jax
        import jax.numpy as jnp
        def function(x):
            hp, hc, modes = native(f=array((130.,)), f_ref=x[0], mass1=x[1],
                                  modes=MODES, **arguments)
            values = jnp.concatenate((hp.reshape(-1), hc.reshape(-1),
                                      *(modes[lm].reshape(-1) for lm in MODES)))
            return jnp.stack((values.real, values.imag), axis=-1).reshape(-1)
        actual = np.asarray(jax.jacfwd(function)(array((40., CASES[0][0]))))
    expected = _orbital_derivative_reference(convention)
    for index in range(2):
        _assert_close(actual[:, index], expected[:, index], rtol=3e-6)


@pytest.mark.parametrize("convention", ("legacy", "spa_sxs_v1"))
def test_orbital_batched_reference_and_dispatch(backend, convention):
    """Batched f_ref/intrinsics and JAX jit agree with independent scalar output."""
    import diffgw
    name, native, array = backend
    frequencies = array((20., 90., 210.))
    arguments = _arguments(CASES[0], "london_2018", None) | dict(
        hm_reference_phase="orbital", hm_phase_convention=convention)
    arguments.pop("f_ref")
    for index, key in enumerate(("mass1", "mass2", "chi1", "chi2")):
        arguments[key] = array([case[index] for case in CASES])
    def function(fref):
        return native(f=frequencies, f_ref=fref, modes=MODES, **arguments)
    if name == "jax":
        import jax
        function = jax.jit(function)
    actual = function(array((30., 40.)))
    dispatched = diffgw.get_fd_waveform(
        approximant="IMRPhenomHM", backend=name, sample_frequencies=frequencies,
        f_ref=array((30., 40.)), modes=MODES, **arguments)
    for index in (0, 1):
        _assert_close(dispatched[index], actual[index], rtol=2e-10)
    for mode in MODES:
        _assert_close(dispatched[2][mode], actual[2][mode], rtol=2e-10)
    for batch, (case, fref) in enumerate(zip(CASES, (30., 40.))):
        expected = [ref.compose(*case).evaluate(
            f, transition="london_2018", f_ref_hz=fref,
            hm_reference_phase="orbital", hm_phase_convention=convention, **EXTRINSIC) for f in (20., 90., 210.)]
        for index in (0, 1):
            _assert_close(actual[index][batch], [complex(row[index]) for row in expected])
        for mode in MODES:
            _assert_close(actual[2][mode][batch], [complex(row[2][mode]) for row in expected])


def test_orbital_carrier_phase_alias_policy(backend):
    """Explicit D phase aliases must preserve a single HM orbital origin."""
    _, native, array = backend
    arguments = _arguments(CASES[0], "london_2018", None) | dict(hm_reference_phase="orbital")
    frequencies = array((20., 90., 210.))
    for key in ("phasing_convention", "reference_frame"):
        for alias in ("reference_frequency", "reffreq", "ref"):
            with pytest.raises(ValueError, match="explicit f_ref"):
                native(f=frequencies, **arguments, **{key: alias})
    arguments["f_ref"] = 40.
    implicit = native(f=frequencies, **arguments)
    explicit = native(f=frequencies, **arguments, reference_frame="ref")
    # Amplitude extraction takes abs(complex carrier); explicit aliases can
    # change its otherwise irrelevant phase and hence its last FP64 bit.
    # This is an alias-equivalence check, not the bitwise legacy reduction.
    for index in (0, 1):
        np.testing.assert_allclose(implicit[index], explicit[index],
                                   rtol=8 * np.finfo(float).eps, atol=0)
    for mode in MODES:
        np.testing.assert_allclose(implicit[2][mode], explicit[2][mode],
                                   rtol=8 * np.finfo(float).eps, atol=0)
    # reference_frame takes precedence. Coalescence phase ignores even a
    # provided f_ref, and can override an implicit reference-frequency alias.
    arguments["f_ref"] = None
    baseline = native(f=frequencies, **arguments)
    for fref in (None, 40.):
        arguments["f_ref"] = fref
        coalescence = native(f=frequencies, **arguments,
                             reference_frame="coalescence", phasing_convention="ref")
        for index in (0, 1):
            np.testing.assert_array_equal(baseline[index], coalescence[index])
        for mode in MODES:
            np.testing.assert_array_equal(baseline[2][mode], coalescence[2][mode])


def test_orbital_rejects_unsupported_carriers(backend):
    """The orbital contract is restricted to exact native D, even for subclasses."""
    from importlib import import_module
    name, native, array = backend
    d_class = import_module(f"{name}wave.approximants.imrphenomd").IMRPhenomD
    xas_class = import_module(f"{name}wave.approximants.imrphenom_xas").IMRPhenomXAS

    class DerivedD(d_class):
        pass

    # inherited calibration ensures rejection is caused by the orbital
    # carrier contract, not the separate published-calibration restriction.
    arguments = _arguments(CASES[0], "london_2018", 40.) | dict(
        hm_reference_phase="orbital", calibration="inherited")
    frequencies = array((90.,))
    if name == "torch":
        import torch

        class CustomCarrier(torch.nn.Module):
            def forward(self, *args, **kwargs):
                pytest.fail("unsupported carrier must not be evaluated")

        for carrier in (DerivedD(), xas_class(), CustomCarrier()):
            with pytest.raises(ValueError, match="orbital HM reference phase requires the native D carrier"):
                type(native)(base_22=carrier)(f=frequencies, **arguments)
    else:
        class CustomCarrier:
            @staticmethod
            def evaluate(*args, **kwargs):
                pytest.fail("unsupported carrier must not be evaluated")

        for carrier in (DerivedD, xas_class, CustomCarrier):
            with pytest.raises(ValueError, match="orbital HM reference phase requires the native D carrier"):
                native(f=frequencies, base_22=carrier, **arguments)


@pytest.mark.parametrize("transition", ("london_2018", "phenomd_runtime"))
@pytest.mark.parametrize("phase,reference_frequency,inclination", (
    ("legacy", None, .93), ("legacy", 40., 2.2),
    ("orbital", None, 2.2), ("orbital", 40., .93),
))
def test_spa_complete_modes_and_projection(backend, transition, phase,
                                           reference_frequency, inclination):
    """Independent full complex comparisons across joins and both hemispheres."""
    _complete_modes_and_projection(backend, CASES[0], transition, reference_frequency,
                                   phase, "spa_sxs_v1", inclination)


@pytest.mark.parametrize("geometric_frequency", (.005, .035, .12))
@pytest.mark.parametrize("transition", ("london_2018", "phenomd_runtime"))
def test_spa_smooth_derivatives(backend, geometric_frequency, transition):
    _smooth_derivatives(backend, geometric_frequency, transition, "spa_sxs_v1")


def test_spa_reference_derivatives(backend):
    _orbital_reference_derivatives(backend, "spa_sxs_v1")


@pytest.mark.parametrize("phase", ("legacy", "orbital"))
@pytest.mark.parametrize("calibration", ("inherited", "published"))
def test_spa_reductions_projection_and_masks(backend, phase, calibration):
    """An exact opt-in default, a once-only factor, and the public mode returns."""
    _, native, array = backend
    frequencies = array((-10., 0., 20., 90., 210.))
    arguments = _arguments(CASES[0], "london_2018", 40.) | dict(
        calibration=calibration, hm_reference_phase=phase)
    default = native(f=frequencies, **arguments)
    legacy = native(f=frequencies, hm_phase_convention="legacy", **arguments)
    actual = native(f=frequencies, hm_phase_convention="spa_sxs_v1", **arguments)
    for index in (0, 1):
        np.testing.assert_array_equal(default[index], legacy[index])
        np.testing.assert_array_equal(np.asarray(actual[index])[:2], np.zeros(2))
    # Exact radicals are obtained from the unchanged independently derived
    # signed-pair expression. No native output supplies the phase factor.
    import sympy as sp
    from diffgw_reference.imrphenomhm_signed_projection import symbolic_phase_factor
    for mode in MODES:
        np.testing.assert_array_equal(default[2][mode], legacy[2][mode])
        if mode[1] == 2:
            np.testing.assert_array_equal(actual[2][mode], legacy[2][mode])
        factor = complex(sp.N(symbolic_phase_factor(mode[1], 3 * sp.pi / 4), 50))
        _assert_close(actual[2][mode], factor * np.asarray(legacy[2][mode]), rtol=2e-14)
        np.testing.assert_array_equal(np.asarray(actual[2][mode])[:2], np.zeros(2))
    arguments.pop("return_unprojected_modes")
    projected = native(f=frequencies, return_modes=True,
                       hm_phase_convention="spa_sxs_v1", **arguments)
    for index in (0, 1):
        np.testing.assert_array_equal(projected[index], actual[index])
        _assert_close(sum(np.asarray(value[index]) for value in projected[2].values()),
                      actual[index], rtol=2e-14)
    # Check each returned projection against independent Wigner coefficients.
    spec = ref.compose(*CASES[0])
    if calibration == "published":
        for mode in MODES:
            expected = [spec.evaluate(f, transition="london_2018", f_ref_hz=40.,
                                     hm_reference_phase=phase, hm_phase_convention="spa_sxs_v1",
                                     modes=(mode,), **EXTRINSIC) for f in (20., 90., 210.)]
            for index in (0, 1):
                _assert_close(np.asarray(projected[2][mode][index])[2:],
                              [complex(row[index]) for row in expected])
    with pytest.raises(ValueError, match="HM phase convention"):
        native(f=frequencies, hm_phase_convention="unknown", **arguments)
    with pytest.raises(ValueError, match="hm_phase_convention"):
        spec.evaluate(90., hm_phase_convention="unknown")
    assert spec.evaluate(90.) == spec.evaluate(90., hm_phase_convention="legacy")


def test_spa_symmetry_and_carrier_rejection(backend):
    """All odd-mode signs reverse on body exchange; symmetric zeros survive."""
    from importlib import import_module
    name, native, array = backend
    frequencies = array((20., 90., 210.))
    arguments = _arguments(CASES[0], "london_2018", 40.) | dict(
        hm_reference_phase="orbital", hm_phase_convention="spa_sxs_v1")
    original = native(f=frequencies, **arguments)
    swapped = arguments | dict(mass1=CASES[0][1], mass2=CASES[0][0],
                               chi1=CASES[0][3], chi2=CASES[0][2])
    exchanged = native(f=frequencies, **swapped)
    for mode in MODES:
        _assert_close(exchanged[2][mode], (-1)**mode[1] * np.asarray(original[2][mode]),
                      rtol=2e-10)
    symmetric = native(f=frequencies, **(arguments | dict(mass1=30., mass2=30.,
                                                         chi1=.2, chi2=.2)))
    for mode in ((2, 1), (3, 3), (4, 3)):
        np.testing.assert_array_equal(symmetric[2][mode], np.zeros(3))
    d_class = import_module(f"{name}wave.approximants.imrphenomd").IMRPhenomD
    class DerivedD(d_class):
        pass
    arguments.update(calibration="inherited", hm_reference_phase="legacy")
    with pytest.raises(ValueError, match="SPA/SXS HM phase convention requires the native D carrier"):
        if name == "torch":
            type(native)(base_22=DerivedD())(f=frequencies, **arguments)
        else:
            native(f=frequencies, base_22=DerivedD, **arguments)
