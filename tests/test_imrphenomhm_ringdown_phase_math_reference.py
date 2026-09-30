"""London HM ringdown phase correction against an independent scalar reference."""

import importlib

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")

from diffgw_reference import imrphenomhm_ringdown_phase as ref


# eta, alpha4, alpha5, HM f_RD22, HM f_RDlm, HM f_damp22,
# HM f_damplm, D-carrier f_RD22, D-carrier f_damp22.
CASES = (
    (.22, -10.0, .9, .080, .100, .010, .015, .080, .010),
    (.16, 6.5, 1.1, .077, .113, .009, .017, .073, .008),
    (.25, -3.2, .8, .082, .094, .012, .011, .079, .010),
)


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

        yield name, module._production_eq14_delta_c1, array
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

            yield name, module._production_eq14_delta_c1, array
    finally:
        jax.config.update("jax_enable_x64", old_x64)


def _production(fn, array, f, case):
    eta, a4, a5, h22, hlm, d22, dlm, b22, bd22 = case
    return fn(array(f), array(eta), array(0.0), array(h22), array(hlm),
              array(d22), array(dlm), array(a4), array(a5),
              base_fr22=array(b22), base_fd22=array(bd22))


def test_component_bindings_resolve():
    assert set(ref.COMPONENT_BINDINGS) == set(ref.COMPONENT_SOURCES)
    for binding in ref.COMPONENT_BINDINGS.values():
        assert callable(getattr(ref, binding["reference"]))
        for backend_name in ("torch", "jax"):
            module_name, function_name = binding[backend_name]
            assert callable(getattr(importlib.import_module(module_name), function_name))
        assert all(callable(globals()[name]) for name in binding["tests"])


def test_paper_limit():
    # When the two 22 QNM conventions coincide, the carrier correction is
    # exactly London's modified term minus Khan's unmodified Eq. (14) term.
    with mp.workdps(70):
        case = CASES[0]
        eta, a4, a5, h22, hlm, _, dlm, b22, bd22 = case
        for f in ("0.035", "0.055", "0.10"):
            actual = ref.carrier_delta(f, *case)
            expected = (ref.paper_hm_term(f, eta, a4, a5, h22, hlm, dlm)
                        - ref.phenomd_term(f, eta, a4, a5, b22, bd22))
            assert abs(actual - expected) < mp.mpf("1e-65")
    with pytest.raises(ValueError, match="f must be positive"):
        ref.carrier_delta_c1(0, *CASES[0])


@pytest.mark.parametrize("case", CASES)
def test_frontend_values_and_gradients(backend, case):
    name, fn, array = backend
    join = case[7] / 2
    frequencies = np.array((join * .8, join, join + 1e-4,
                            case[2] * case[7], join * 1.7, join * 3.0))
    with mp.workdps(70):
        expected = np.array([float(ref.carrier_delta_c1(str(f), *case))
                             for f in frequencies])
    actual = np.asarray(_production(fn, array, frequencies, case))
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)

    f = join * 1.7
    with mp.workdps(70):
        expected_slope = float(mp.diff(lambda x: ref.carrier_delta_c1(x, *case),
                                       mp.mpf(str(f))))
        a4 = mp.mpf(str(case[1]))
        expected_a4 = float(mp.diff(
            lambda x: ref.carrier_delta_c1(str(f), case[0], x, *case[2:]), a4
        ))
    if name == "torch":
        import torch

        tf = array(f).requires_grad_()
        ta4 = array(case[1]).requires_grad_()
        eta, _, a5, h22, hlm, d22, dlm, b22, bd22 = case
        out = fn(tf, array(eta), array(0), array(h22), array(hlm),
                 array(d22), array(dlm), ta4, array(a5),
                 base_fr22=array(b22), base_fd22=array(bd22))
        slopes = torch.autograd.grad(out, (tf, ta4))
    else:
        import jax

        def output(freq, alpha4):
            eta, _, a5, h22, hlm, d22, dlm, b22, bd22 = case
            return fn(freq, array(eta), array(0), array(h22), array(hlm),
                      array(d22), array(dlm), alpha4, array(a5),
                      base_fr22=array(b22), base_fd22=array(bd22))

        slopes = jax.grad(output, argnums=(0, 1))(array(f), array(case[1]))
    np.testing.assert_allclose([float(v) for v in slopes],
                               [expected_slope, expected_a4],
                               rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("case", CASES)
def test_c1_join_and_22_control(backend, case):
    name, fn, array = backend
    join = case[7] / 2
    assert float(_production(fn, array, join, case)) == 0.0
    with mp.workdps(70):
        assert ref.carrier_delta_c1(str(join), *case) == 0
        # Differentiate the right-hand formula, independent of the piecewise
        # branch selection at the join.
        slope = mp.diff(lambda x: ref.carrier_delta(x, *case),
                        mp.mpf(str(join)))
        right_slope = mp.diff(
            lambda x: ref.carrier_delta(x, *case)
            - ref.carrier_delta(str(join), *case)
            - slope * (x - mp.mpf(str(join))), mp.mpf(str(join))
        )
        assert abs(right_slope) < mp.mpf("1e-65")
        near_right = join + 1e-6
        expected_near_right = float(mp.diff(
            lambda x: ref.carrier_delta_c1(x, *case), mp.mpf(str(near_right))
        ))
    if name == "torch":
        import torch

        tf = array(join).requires_grad_()
        production_slope = torch.autograd.grad(
            _production(fn, array, tf, case), tf
        )[0]
        tf_right = array(near_right).requires_grad_()
        production_near_right = torch.autograd.grad(
            _production(fn, array, tf_right, case), tf_right
        )[0]
    else:
        import jax

        frequency_gradient = jax.grad(lambda f: _production(fn, array, f, case))
        production_slope = frequency_gradient(array(join))
        production_near_right = frequency_gradient(array(near_right))
    assert abs(float(production_slope)) < 1e-12
    np.testing.assert_allclose(float(production_near_right),
                               expected_near_right, rtol=3e-9, atol=3e-10)

    # The (2,2) control has both HM ratios equal to one, even if its D QNM
    # values use a distinct convention.
    eta, a4, a5, h22, _, d22, _, b22, bd22 = case
    control = (eta, a4, a5, h22, h22, d22, d22, b22, bd22)
    for f in (join * .8, join, join * 1.5, join * 3):
        assert abs(float(_production(fn, array, f, control))) < 1e-12
        with mp.workdps(70):
            assert abs(ref.carrier_delta_c1(str(f), *control)) < mp.mpf("1e-65")
