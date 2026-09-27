"""Full HM qualification with no frontend-derived reference inputs."""

import ast
from pathlib import Path

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")
pytest.importorskip("sympy")

from diffgw_reference import imrphenomhm_waveform as ref
from diffgw_reference.imrphenomhm_frequency_map import outer_map_cutoffs

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
    _, native, array = backend
    spec = ref.compose(*case)
    arguments = _arguments(case, transition, reference_frequency)
    options = dict(transition=transition, f_ref_hz=reference_frequency, **EXTRINSIC)
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


def _reference_vector(spec, values, transition):
    frequency, inclination, phi0, phic, tc, distance = values
    hp, hc, modes = spec.evaluate(
        frequency, transition=transition, f_ref_hz=40., inclination=inclination,
        phi0=phi0, phic=phic, tc=tc)
    # Exact distance scaling avoids rebuilding unchanged symbolic fits.
    scale = 100 / distance
    values = (hp, hc, *(modes[lm] for lm in MODES))
    return tuple(scale * part for value in values for part in (value.real, value.imag))


def _reference_jacobian(spec, values, transition):
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
                    rows.append(_reference_vector(spec, point, transition))
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
    name, native, array = backend
    case = CASES[0]
    spec = ref.compose(*case)
    point = [geometric_frequency / float(spec.m_sec), EXTRINSIC["inclination"],
             EXTRINSIC["phi0"], EXTRINSIC["phic"], EXTRINSIC["tc"], 100.]
    arguments = _arguments(case, transition, 40.)
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
    expected = _reference_jacobian(spec, point, transition)
    # Scale each physical derivative separately; tiny components near zero
    # must not inherit the much larger time-derivative absolute tolerance.
    for index in range(len(point)):
        _assert_close(actual[:, index], expected[:, index], rtol=3e-6)
