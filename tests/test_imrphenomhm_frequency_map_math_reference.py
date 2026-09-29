"""London Eq. (5)/(8) scalar, Torch and JAX frequency-map checks."""

import importlib

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
mp = pytest.importorskip("mpmath")

from diffgw_reference import imrphenomhm_frequency_map as ref


# Synthetic QNM frequencies exercise every supported m without importing fits.
# The 22 case checks that both mappings reduce to the identity.
CASES = ((2, 2, .050), (2, 1, .035), (3, 3, .072),
         (3, 2, .060), (4, 4, .090), (4, 3, .080))
F22 = .050
MODES = tuple((l, m) for l, m, _ in CASES)
REMNANT_INPUTS = ((35.0, 20.0, 0.2, -0.1),
                  (50.0, 10.0, 0.7, -0.3))


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device == "mps":
        pytest.fail("FP64 math-reference qualification is unavailable on MPS")
    name = request.param
    module = pytest.importorskip(f"{name}wave.approximants.imrphenom_hm")
    if name == "torch":
        torch = pytest.importorskip("torch")
        if reference_device == "cuda" and not torch.cuda.is_available():
            pytest.fail("Torch CUDA unavailable")

        def array(value):
            return torch.as_tensor(value, dtype=torch.float64, device=reference_device)

        yield name, module, array
        return

    jax = pytest.importorskip("jax")
    old_x64 = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        import jax.numpy as jnp

        platform = "gpu" if reference_device == "cuda" else "cpu"
        try:
            device = jax.devices(platform)[0]
        except (RuntimeError, IndexError) as exc:
            pytest.fail(f"JAX {platform} unavailable: {exc}")
        if reference_device == "cuda" and "cuda" not in device.client.platform_version.lower():
            pytest.fail("JAX GPU is not CUDA")
        with jax.default_device(device):
            def array(value):
                return jax.device_put(jnp.asarray(value, dtype=jnp.float64), device)

            yield name, module, array
    finally:
        jax.config.update("jax_enable_x64", old_x64)


def test_component_bindings_are_complete_and_resolve():
    assert set(ref.COMPONENT_BINDINGS) == set(ref.COMPONENT_SOURCES)
    assert all(source == ref.SOURCE for source, _ in ref.COMPONENT_SOURCES.values())
    for binding in ref.COMPONENT_BINDINGS.values():
        assert callable(getattr(ref, binding["reference"]))
        for name in ("torch", "jax"):
            module_name, function_name = binding[name]
            assert callable(getattr(importlib.import_module(module_name), function_name))
        assert all(callable(globals()[name]) for name in binding["tests"])


@pytest.mark.parametrize("phase_d_carrier", (False, True))
def test_symbolic_branches_joins_and_one_sided_slopes(phase_d_carrier):
    low, middle, high = ref.branch_expressions(phase_d_carrier=phase_d_carrier)
    assert sp.simplify(low.subs(ref.F, ref.F0) - middle.subs(ref.F, ref.F0)) == 0
    assert sp.simplify(middle.subs(ref.F, ref.FR) - high.subs(ref.F, ref.FR)) == 0
    assert sp.simplify(middle.subs(ref.F, ref.FR) - ref.F22) == 0
    assert tuple(sp.diff(branch, ref.F) for branch in (low, middle, high)) == (
        2 / ref.M, (ref.F22 - 2 * ref.F0 / ref.M) / (ref.FR - ref.F0),
        ref.F22 / ref.FR if phase_d_carrier else sp.Integer(1))
    for branch in (low, middle, high):
        assert sp.simplify(branch.subs({ref.M: 2, ref.FR: ref.F22}) - ref.F) == 0


def test_scalar_example_and_domain():
    # Exact hand reduction: m=3, f0=1/50, f_RD_lm=3/50, f_RD_22=1/20.
    with mp.workdps(60):
        args = ("0.02", "0.06", "0.05", 3)
        assert abs(ref.eq5_frequency_map("0.04", *args) - mp.mpf(19) / 600) < mp.mpf("1e-55")
        assert abs(ref.eq5_frequency_map("0.08", *args) - mp.mpf(7) / 100) < mp.mpf("1e-55")
        assert abs(ref.eq8_phase_frequency_map("0.08", *args) - mp.mpf(1) / 15) < mp.mpf("1e-55")
        for actual, expected in zip(ref.branch_slopes(*args),
                                    (mp.mpf(2) / 3, mp.mpf(11) / 12, mp.mpf(1))):
            assert abs(actual - expected) < mp.mpf("1e-55")
    with pytest.raises(ValueError, match="f0"):
        ref.eq5_frequency_map(.03, .06, .05, .05, 3)
    with pytest.raises(ValueError, match="positive integer"):
        ref.eq5_frequency_map(.03, .02, .05, .05, 0)


@pytest.mark.parametrize("l,m,fr", CASES)
@pytest.mark.parametrize("phase_d_carrier", (False, True))
def test_frontend_values_and_joins(backend, l, m, fr, phase_d_carrier):
    name, module, array = backend
    f0 = (.014 if phase_d_carrier else .018) * fr / F22
    fn = module.frequency_mapping_phase if phase_d_carrier else module.frequency_mapping_amplitude
    scalar = ref.eq8_phase_frequency_map if phase_d_carrier else ref.eq5_frequency_map
    eps = 1e-8
    frequencies = np.array((f0 / 2, f0 - eps, f0, f0 + eps,
                            (f0 + fr) / 2, fr - eps, fr, fr + eps, fr * 1.3))
    actual = np.asarray(fn(array(frequencies), array(f0), array(fr), array(F22), m))
    with mp.workdps(60):
        expected = np.array([float(scalar(str(f), str(f0), str(fr), str(F22), m))
                             for f in frequencies])
        slopes = tuple(map(float, ref.branch_slopes(str(f0), str(fr), str(F22), m,
                                                   phase_d_carrier=phase_d_carrier)))
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(actual[[2, 6]], (2 * f0 / m, F22), rtol=2e-14)
    np.testing.assert_allclose(actual[[1, 3, 5, 7]],
                               (actual[2] - eps * slopes[0], actual[2] + eps * slopes[1],
                                actual[6] - eps * slopes[1], actual[6] + eps * slopes[2]),
                               rtol=2e-14, atol=2e-15)
    if (l, m) == (2, 2):
        np.testing.assert_allclose(actual, frequencies, rtol=2e-14, atol=2e-15)


@pytest.mark.parametrize("phase_d_carrier", (False, True))
def test_frontend_frequency_derivatives(backend, phase_d_carrier):
    name, module, array = backend
    fn = module.frequency_mapping_phase if phase_d_carrier else module.frequency_mapping_amplitude
    f0, fr, f22, m = .02, .08, .05, 3
    slopes = tuple(map(float, ref.branch_slopes(f0, fr, f22, m,
                                                phase_d_carrier=phase_d_carrier)))
    # The paper defines the derivative at a join by its lower-frequency limit.
    for f, expected in ((.01, slopes[0]), (f0, slopes[0]),
                        (.04, slopes[1]), (fr, slopes[1]), (.10, slopes[2])):
        if name == "torch":
            import torch
            x = array(f).requires_grad_()
            derivative = torch.autograd.grad(fn(x, f0, fr, f22, m), x)[0]
        else:
            import jax
            derivative = jax.grad(lambda x: fn(x, f0, fr, f22, m))(array(f))
        np.testing.assert_allclose(float(derivative), expected, rtol=2e-13, atol=2e-14)


@pytest.mark.parametrize("phase_d_carrier", (False, True))
def test_frontend_parameter_derivatives(backend, phase_d_carrier):
    name, module, array = backend
    fn = module.frequency_mapping_phase if phase_d_carrier else module.frequency_mapping_amplitude
    branches = ref.branch_expressions(phase_d_carrier=phase_d_carrier)
    for f, branch in ((.01, branches[0]), (.04, branches[1]), (.10, branches[2])):
        values = (f, .02, .08, .05)
        substitutions = dict(zip((ref.F, ref.F0, ref.FR, ref.F22), values)) | {ref.M: 3}
        expected = [float(sp.N(sp.diff(branch, variable).subs(substitutions), 40))
                    for variable in (ref.F, ref.F0, ref.FR, ref.F22)]
        if name == "torch":
            import torch
            args = tuple(array(value).requires_grad_() for value in values)
            actual = torch.autograd.grad(fn(*args, 3), args, allow_unused=True)
            actual = [0.0 if grad is None else float(grad) for grad in actual]
        else:
            import jax
            args = tuple(array(value) for value in values)
            actual = jax.grad(lambda *xs: fn(*xs, 3), argnums=(0, 1, 2, 3))(*args)
            actual = [float(grad) for grad in actual]
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)


@pytest.mark.parametrize("variant", ("phenomd_runtime", "london_2018"))
@pytest.mark.parametrize("masses_spins", REMNANT_INPUTS)
@pytest.mark.parametrize("l,m", MODES)
def test_public_transition_selection(backend, monkeypatch, variant,
                                     masses_spins, l, m):
    """Six public D modes select Eq. (5)/(8) maps from supplied QNM inputs."""
    name, module, array = backend
    model = module.IMRPhenomHM(modes=((l, m),))
    mass1, mass2, chi1, chi2 = masses_spins
    rd_lm = float(module.IMRPhenomHM.mode_qnm_frequencies(
        l, m, *(array(x) for x in masses_spins))[0])
    rd_22 = float(module.IMRPhenomHM.mode_qnm_frequencies(
        2, 2, *(array(x) for x in masses_spins))[0])
    f0_amp, f0_phi = (float(x) for x in ref.outer_map_cutoffs(variant, rd_lm, rd_22))
    h = 1e-6
    grid = np.unique(np.array([
        f0_amp / 2, f0_phi / 2, (max(f0_amp, f0_phi) + rd_lm) / 2,
        *(join + offset for join in (f0_amp, f0_phi, rd_lm)
          for offset in (-h, 0, h)),
        rd_lm * 1.1,
    ]))
    frequencies = array(grid / ((mass1 + mass2) * float(module.MSUN_SEC)))
    kwargs = dict(mass1=mass1, mass2=mass2, chi1=chi1, chi2=chi2,
                  return_unprojected_modes=True)
    compare_default = (variant == "phenomd_runtime" or
                       ((l, m) == (3, 3) and masses_spins == REMNANT_INPUTS[0]))
    baseline = (np.asarray(model(frequencies, **kwargs)[2][(l, m)])
                if compare_default else None)
    seen = {}
    amp_fn = module.frequency_mapping_amplitude
    phase_fn = module._production_mapped_phase_c1

    def capture_amp(f, f0, r, q, m):
        seen["amp_inputs"] = (float(f0), float(r), float(q), m)
        result = amp_fn(f, f0, r, q, m)
        seen["amp_map"] = np.asarray(result)
        return result

    def capture_phase(f, f0, r, q, m, ell, carrier_phase):
        seen["phase_inputs"] = (float(f0), float(r), float(q), m)

        def capture_carrier(y):
            if np.asarray(y).shape == grid.shape:
                seen["phase_map"] = np.asarray(y)
            return carrier_phase(y)

        return phase_fn(f, f0, r, q, m, ell, capture_carrier)

    with monkeypatch.context() as patcher:
        patcher.setattr(module, "frequency_mapping_amplitude", capture_amp)
        patcher.setattr(module, "_production_mapped_phase_c1", capture_phase)
        actual = np.asarray(model(frequencies, hm_mapping_transitions=variant,
                                  **kwargs)[2][(l, m)])

    if (l, m) == (3, 3) and masses_spins == REMNANT_INPUTS[0]:
        dispatch = importlib.import_module(f"{name}wave.waveforms").get_fd_waveform
        dispatched = dispatch("IMRPhenomHM", sample_frequencies=frequencies,
                              modes=((l, m),), hm_mapping_transitions=variant,
                              **kwargs)[2][(l, m)]
        np.testing.assert_allclose(np.asarray(dispatched), actual, rtol=1e-13, atol=0)

    for key, f0, scalar, phase in (
        ("amp", f0_amp, ref.eq5_frequency_map, False),
        ("phase", f0_phi, ref.eq8_phase_frequency_map, True),
    ):
        native_f0, r, q, mapped_m = seen[f"{key}_inputs"]
        np.testing.assert_allclose((native_f0, r, q), (f0, rd_lm, rd_22),
                                   rtol=2e-11, atol=0)
        assert mapped_m == m
        mapped = seen[f"{key}_map"]
        expected = np.array([float(scalar(str(f), str(f0), str(rd_lm),
                                          str(rd_22), m)) for f in grid])
        np.testing.assert_allclose(mapped, expected, rtol=2e-11, atol=2e-13)
        slopes = tuple(float(s) for s in ref.branch_slopes(
            f0, rd_lm, rd_22, m, phase_d_carrier=phase))
        for join, left, right in ((f0, slopes[0], slopes[1]),
                                  (rd_lm, slopes[1], slopes[2])):
            idx = int(np.argmin(abs(grid - join)))
            np.testing.assert_allclose(
                ((mapped[idx] - mapped[idx - 1]) / h,
                 (mapped[idx + 1] - mapped[idx]) / h),
                (left, right), rtol=2e-6, atol=2e-6)
        if (l, m) == (2, 2):
            np.testing.assert_allclose(mapped, grid, rtol=2e-11, atol=2e-13)

    if variant == "phenomd_runtime":
        np.testing.assert_allclose(actual, baseline, rtol=1e-13, atol=0)
    elif baseline is not None:
        assert np.max(abs(actual - baseline)) > 1e-8 * np.max(abs(baseline))

    if (l, m) == (3, 3) and masses_spins == REMNANT_INPUTS[0]:
        with pytest.raises(ValueError, match="Unknown HM mapping transitions"):
            model(frequencies, hm_mapping_transitions="unknown", **kwargs)
        non_d = module.IMRPhenomHM(modes=((l, m),), base_22=object())
        with pytest.raises(ValueError, match="require the D carrier"):
            non_d(frequencies, hm_mapping_transitions="london_2018", **kwargs)
