"""London Eq. (4) leading (3,3)/(4,4) public Torch/JAX amplitude gates."""

import importlib
import math

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
mp = pytest.importorskip("mpmath")

from diffgw_reference import imrphenomhm_amplitude_ratio as ref
from diffgw_reference import imrphenomhm_frequency_map as fmap


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device != "cpu":
        pytest.fail("This component's frozen FP64 qualification is CPU-only")
    name = request.param
    module = pytest.importorskip(f"{name}wave.approximants.imrphenom_hm")
    if name == "torch":
        torch = pytest.importorskip("torch")

        def array(value):
            return torch.as_tensor(value, dtype=torch.float64)

        yield name, module, array
        return

    jax = pytest.importorskip("jax")
    old_x64 = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        import jax.numpy as jnp

        device = jax.devices("cpu")[0]
        with jax.default_device(device):
            def array(value):
                return jax.device_put(jnp.asarray(value, dtype=jnp.float64), device)

            yield name, module, array
    finally:
        jax.config.update("jax_enable_x64", old_x64)


def test_source_bindings_and_symbolic_reduction():
    assert set(ref.COMPONENT_BINDINGS) == set(ref.COMPONENT_SOURCES)
    binding = ref.COMPONENT_BINDINGS["hm.amp.beta33.leading.eq4.v1"]
    assert callable(getattr(ref, binding["reference"]))
    for name in ("torch", "jax"):
        module, cls = binding[name]
        assert callable(getattr(importlib.import_module(module), cls))
    assert all(callable(globals()[test]) for test in binding["tests"])
    expected = sp.Rational(3, 4) * sp.sqrt(sp.Rational(5, 7))
    expected *= ref.DELTA * (sp.pi * ref.G) ** sp.Rational(1, 3)
    assert sp.simplify(ref.symbolic_eq4_beta33() - expected) == 0
    assert ref.Y not in ref.symbolic_eq4_beta33().free_symbols


def test_scalar_example_and_domain():
    with mp.workdps(60):
        # pi*g=1 and delta=3/11 give 9/44*sqrt(5/7) by hand.
        observed = ref.beta33_leading(1 / mp.pi, mp.mpf(3) / 11)
        expected = mp.mpf(9) / 44 * mp.sqrt(mp.mpf(5) / 7)
        assert abs(observed - expected) < mp.mpf("1e-55")
        f0, fr, f22, delta = ".02", ".06", ".05", "0.3"
        for join in (f0, fr):
            left, right = ref.beta33_eq5_one_sided_slopes(
                join, f0, fr, f22, delta)
            assert left > 0 and right > 0 and left != right
    for g, delta in ((0, .3), (.02, 0), (.02, 1)):
        with pytest.raises(ValueError):
            ref.beta33_leading(g, delta)


def test_beta44_source_reduction_and_scalar_domain():
    binding = ref.COMPONENT_BINDINGS["hm.amp.beta44.leading.eq4.v1"]
    assert callable(getattr(ref, binding["reference"]))
    assert all(callable(globals()[test]) for test in binding["tests"])
    expected = sp.Rational(4, 9) * sp.sqrt(sp.Rational(10, 7))
    expected *= (1 - 3 * ref.ETA) * (sp.pi * ref.G) ** sp.Rational(2, 3)
    expression = ref.symbolic_eq4_beta44()
    assert sp.simplify(expression - expected) == 0
    assert ref.Y not in expression.free_symbols
    with mp.workdps(60):
        # pi*g=1 and eta=1/4 leave sqrt(10/7)/9.
        observed = ref.beta44_leading(1 / mp.pi, mp.mpf(1) / 4)
        assert abs(observed - mp.sqrt(mp.mpf(10) / 7) / 9) < mp.mpf("1e-55")
        f0, fr, f22, eta = ".02", ".06", ".045", "0.24"
        for join in (f0, fr):
            left, right = ref.beta44_eq5_one_sided_slopes(
                join, f0, fr, f22, eta)
            assert left > 0 and right > 0 and left != right
    for g, eta in ((0, .24), (.02, 0), (.02, .251)):
        with pytest.raises(ValueError):
            ref.beta44_leading(g, eta)


@pytest.mark.parametrize("variant", ("phenomd_runtime", "london_2018"))
def test_public_beta33_values_and_one_sided_slopes(backend, variant):
    """Remove only the public D carrier magnitude from the public (3,3) mode."""
    name, module, array = backend
    m1, m2, chi1, chi2, distance = 35.0, 20.0, 0.2, -0.1, 100.0
    delta = (m1 - m2) / (m1 + m2)
    model = module.IMRPhenomHM(modes=((3, 3),))
    qnm_args = (array(m1), array(m2), array(chi1), array(chi2))
    fr = float(module.IMRPhenomHM.mode_qnm_frequencies(3, 3, *qnm_args)[0])
    f22 = float(module.IMRPhenomHM.mode_qnm_frequencies(2, 2, *qnm_args)[0])
    f0 = float(fmap.outer_map_cutoffs(variant, fr, f22)[0])
    h = 1e-7
    x = np.array((f0 / 2, f0 - h, f0, f0 + h,
                  (f0 + fr) / 2, fr - h, fr, fr + h, fr * 1.1))
    mapped = np.array([float(fmap.eq5_frequency_map(
        str(value), str(f0), str(fr), str(f22), 3)) for value in x])
    m_seconds = (m1 + m2) * float(module.MSUN_SEC)
    public_args = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2,
                       distance=distance, inclination=0.0, phic=0.0,
                       tc=0.0, f_ref=None)
    mode = np.asarray(model(
        f=array(x / m_seconds), hm_mapping_transitions=variant,
        return_unprojected_modes=True, **public_args)[2][(3, 3)])
    if name == "torch":
        hp, _ = module.IMRPhenomD()(f=array(mapped / m_seconds), **public_args)
    else:
        hp, _ = module.IMRPhenomD.evaluate(
            f=array(mapped / m_seconds), **public_args)
    carrier = np.abs(np.asarray(hp)) / math.sqrt(5.0 / (16.0 * math.pi))
    assert np.all(np.isfinite(carrier)) and np.all(carrier > 0)
    observed = np.abs(mode) / carrier
    expected = np.array([float(ref.beta33_leading(str(g), str(delta)))
                         for g in mapped])
    np.testing.assert_allclose(observed, expected, rtol=5e-11, atol=5e-13)
    for join, lower, center, upper in ((f0, 1, 2, 3), (fr, 5, 6, 7)):
        left, right = ref.beta33_eq5_one_sided_slopes(
            str(join), str(f0), str(fr), str(f22), str(delta))
        secants = ((observed[center] - observed[lower]) / h,
                   (observed[upper] - observed[center]) / h)
        np.testing.assert_allclose(secants, (float(left), float(right)),
                                   rtol=2e-5, atol=2e-8)


@pytest.mark.parametrize("variant", ("phenomd_runtime", "london_2018"))
@pytest.mark.parametrize("masses", ((35.0, 20.0), (30.0, 30.0)))
def test_public_beta44_values_and_one_sided_slopes(backend, variant, masses):
    """Measure the public (4,4) factor after removing the public D carrier."""
    name, module, array = backend
    m1, m2 = masses
    chi1, chi2, distance = 0.2, -0.1, 100.0
    eta = m1 * m2 / (m1 + m2) ** 2
    model = module.IMRPhenomHM(modes=((4, 4),))
    qnm_args = (array(m1), array(m2), array(chi1), array(chi2))
    fr = float(module.IMRPhenomHM.mode_qnm_frequencies(4, 4, *qnm_args)[0])
    f22 = float(module.IMRPhenomHM.mode_qnm_frequencies(2, 2, *qnm_args)[0])
    f0 = float(fmap.outer_map_cutoffs(variant, fr, f22)[0])
    h = 1e-7
    x = np.array((f0 / 2, f0 - h, f0, f0 + h,
                  (f0 + fr) / 2, fr - h, fr, fr + h, fr * 1.1))
    mapped = np.array([float(fmap.eq5_frequency_map(
        str(value), str(f0), str(fr), str(f22), 4)) for value in x])
    m_seconds = (m1 + m2) * float(module.MSUN_SEC)
    public_args = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2,
                       distance=distance, inclination=0.0, phic=0.0,
                       tc=0.0, f_ref=None)
    mode = np.asarray(model(
        f=array(x / m_seconds), hm_mapping_transitions=variant,
        return_unprojected_modes=True, **public_args)[2][(4, 4)])
    if name == "torch":
        hp, _ = module.IMRPhenomD()(f=array(mapped / m_seconds), **public_args)
    else:
        hp, _ = module.IMRPhenomD.evaluate(
            f=array(mapped / m_seconds), **public_args)
    carrier = np.abs(np.asarray(hp)) / math.sqrt(5.0 / (16.0 * math.pi))
    assert np.all(np.isfinite(carrier)) and np.all(carrier > 0)
    observed = np.abs(mode) / carrier
    expected = np.array([float(ref.beta44_leading(str(g), str(eta)))
                         for g in mapped])
    np.testing.assert_allclose(observed, expected, rtol=5e-11, atol=5e-13)
    for join, lower, center, upper in ((f0, 1, 2, 3), (fr, 5, 6, 7)):
        left, right = ref.beta44_eq5_one_sided_slopes(
            str(join), str(f0), str(fr), str(f22), str(eta))
        secants = ((observed[center] - observed[lower]) / h,
                   (observed[upper] - observed[center]) / h)
        np.testing.assert_allclose(secants, (float(left), float(right)),
                                   rtol=3e-5, atol=3e-8)
