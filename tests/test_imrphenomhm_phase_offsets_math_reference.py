"""London HM phase offsets against an independent scalar and branch calculus."""

import importlib

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")
sp = pytest.importorskip("sympy")

from diffgw_reference import imrphenomhm_phase_offsets as ref


MODES = ((2, 1, .035), (3, 3, .072), (3, 2, .060), (4, 4, .090), (4, 3, .080))
F22 = .050


def carrier(x):
    return 1 + 2 * x + x * x


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

        def array(x):
            return torch.as_tensor(x, dtype=torch.float64, device=reference_device)

        yield name, module._production_mapped_phase_c1, array
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
        with jax.default_device(device):
            def array(x):
                return jax.device_put(jnp.asarray(x, dtype=jnp.float64), device)

            yield name, module._production_mapped_phase_c1, array
    finally:
        jax.config.update("jax_enable_x64", old_x64)


def test_component_bindings_resolve():
    assert set(ref.COMPONENT_BINDINGS) == set(ref.COMPONENT_SOURCES)
    for binding in ref.COMPONENT_BINDINGS.values():
        assert callable(getattr(ref, binding["reference"]))
        for backend_name in ("torch", "jax"):
            module_name, function_name = binding[backend_name]
            assert callable(getattr(importlib.import_module(module_name), function_name))
        assert all(callable(globals()[name]) for name in binding["tests"])


def test_hand_reduced_low_branch_and_domain():
    with mp.workdps(70):
        expected = mp.mpf(3) / 2 * (mp.mpf(151) / 150) ** 2 + 7 * mp.pi / 2
        actual = ref.mapped_phase(".01", ".02", ".06", ".05", 3, 3, carrier)
        assert abs(actual - expected) < mp.mpf("1e-65")
        assert ref.low_frequency_offset(3, 3) == 7 * mp.pi / 2
    with pytest.raises(ValueError, match="f_rd_22"):
        ref.mapped_phase(.01, .02, .06, .001, 3, 3, carrier)


@pytest.mark.parametrize("ell,m,fr", MODES)
def test_frontend_phase_values(backend, ell, m, fr):
    _, fn, array = backend
    f0 = .014 * fr / F22
    eps = 1e-8
    frequencies = np.array((f0 / 2, f0 - eps, f0, f0 + eps,
                            (f0 + fr) / 2, fr - eps, fr, fr + eps, fr * 1.3))
    actual = np.asarray(fn(array(frequencies), array(f0), array(fr), array(F22),
                           m, ell, carrier))
    with mp.workdps(70):
        expected = [float(ref.mapped_phase(str(f), str(f0), str(fr), str(F22),
                                           ell, m, carrier)) for f in frequencies]
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)


def _symbolic_branch_derivatives(ell, m, branch):
    f, f0, fr, f22 = sp.symbols("f f0 fr f22", positive=True)
    g0 = 2 * f0 / m
    slope = (f22 - g0) / (fr - f0)
    gs = (2 * f / m, g0 + slope * (f - f0), f22 * f / fr)
    ks = (sp.Rational(m, 2), 1 / slope, fr / f22)
    def p(g):
        return 1 + 2 * g + g * g
    low = sp.pi * sp.Rational(3 * ell + (ell + m) % 2 - 2, 2)
    mid = low + (ks[0] - ks[1]) * p(g0)
    high = mid + (ks[1] - ks[2]) * p(f22)
    phase = ks[branch] * p(gs[branch]) + (low, mid, high)[branch]
    return (f, f0, fr, f22), [sp.diff(phase, x) for x in (f, f0, fr, f22)]


@pytest.mark.parametrize("ell,m,fr", MODES)
def test_frontend_phase_derivatives(backend, ell, m, fr):
    name, fn, array = backend
    f0 = .014 * fr / F22
    for branch, f in enumerate((f0 / 2, (f0 + fr) / 2, fr * 1.3)):
        symbols, derivatives = _symbolic_branch_derivatives(ell, m, branch)
        values = (f, f0, fr, F22)
        substitutions = dict(zip(symbols, values))
        expected = [float(sp.N(expr.subs(substitutions), 40)) for expr in derivatives]
        if name == "torch":
            import torch

            args = tuple(array(v).requires_grad_() for v in values)
            actual = torch.autograd.grad(fn(*args, m, ell, carrier), args,
                                         allow_unused=True)
            actual = [0.0 if grad is None else float(grad) for grad in actual]
        else:
            import jax

            args = tuple(array(v) for v in values)
            actual = jax.grad(lambda *xs: fn(*xs, m, ell, carrier),
                              argnums=(0, 1, 2, 3))(*args)
            actual = [float(grad) for grad in actual]
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)
