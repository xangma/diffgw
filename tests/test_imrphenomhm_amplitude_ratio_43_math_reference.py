"""Source-first London/Mishra (4,3) amplitude gate through public Torch/JAX."""

import importlib
import math

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
mp = pytest.importorskip("mpmath")

from diffgw_reference import imrphenomhm_amplitude_ratio_43 as ref
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
    binding = ref.COMPONENT_BINDINGS[ref.COMPONENT]
    assert callable(getattr(ref, binding["reference"]))
    for name in ("torch", "jax"):
        module, cls = binding[name]
        assert callable(getattr(importlib.import_module(module), cls))
    assert all(callable(globals()[test]) for test in binding["tests"])
    v = sp.symbols("v")
    assert sp.expand(sp.Poly(ref.symbolic_p43(v), v).coeff_monomial(v**4) -
                     sp.Rational(5, 2) * ref.ETA * (ref.CHI_A - ref.DELTA * ref.CHI_S)) == 0
    vg = (2 * sp.pi * ref.G / 3) ** sp.Rational(1, 3)
    vx = (2 * sp.pi * ref.X / 3) ** sp.Rational(1, 3)
    vd = (4 * sp.pi * ref.X / 9) ** sp.Rational(1, 3)
    expected = (sp.Rational(3, 4) * sp.sqrt(sp.Rational(3, 35))
                * ref.symbolic_p43(vg) * ref.symbolic_p43(vx) / ref.symbolic_p43(vd))
    assert sp.simplify(ref.symbolic_eq4_beta43() / expected - 1) == 0


def test_scalar_example_domain_and_symmetry():
    with mp.workdps(60):
        x = 1 / mp.pi
        expected = mp.mpf(15) / 64 * mp.sqrt(mp.mpf(3) / 35)
        assert abs(ref.beta43_1p5pn(x, x, 3, 1, 0, 0) - expected) < mp.mpf("1e-55")
        assert ref.beta43_1p5pn(".02", ".03", 30, 30, ".2", "-.1") > 0
        for join in (".02", ".06"):
            slopes = tuple(ref.beta43_eq5_one_sided_slopes(
                join, ".02", ".06", ".05", 35, 20, ".2", "-.1"))
            assert all(mp.isfinite(s) for s in slopes)
    for args in ((0, .02, 35, 20, .2, -.1),
                 (.02, 0, 35, 20, .2, -.1),
                 (.02, .03, 20, 35, .2, -.1),
                 (.02, .03, 35, 20, 1.1, -.1),
                 (.02, .03, 30, 30, .2, .2)):
        with pytest.raises(ValueError):
            ref.beta43_1p5pn(*args)


@pytest.mark.parametrize("variant", ("phenomd_runtime", "london_2018"))
@pytest.mark.parametrize("case", ((35.0, 20.0, 0.2, -0.1),
                                  (35.0, 20.0, 0.0, 0.0),
                                  (30.0, 30.0, 0.2, -0.1)))
def test_public_beta43_values_and_one_sided_slopes(backend, variant, case):
    name, module, array = backend
    m1, m2, chi1, chi2 = case
    model = module.IMRPhenomHM(modes=((4, 3),))
    qnm_args = (array(m1), array(m2), array(chi1), array(chi2))
    fr = float(module.IMRPhenomHM.mode_qnm_frequencies(4, 3, *qnm_args)[0])
    f22 = float(module.IMRPhenomHM.mode_qnm_frequencies(2, 2, *qnm_args)[0])
    f0 = float(fmap.outer_map_cutoffs(variant, fr, f22)[0])
    h = 1e-7
    x = np.array((f0 / 2, f0 - h, f0, f0 + h,
                  (f0 + fr) / 2, fr - h, fr, fr + h, fr * 1.1))
    mapped = np.array([float(fmap.eq5_frequency_map(
        str(value), str(f0), str(fr), str(f22), 3)) for value in x])
    m_seconds = (m1 + m2) * float(module.MSUN_SEC)
    public_args = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2,
                       distance=100.0, inclination=0.0, phic=0.0,
                       tc=0.0, f_ref=None)
    mode = np.asarray(model(
        f=array(x / m_seconds), hm_mapping_transitions=variant,
        hm_43_pn_variant="mishra_eq12g", return_unprojected_modes=True,
        **public_args)[2][(4, 3)])
    if name == "torch":
        hp, _ = module.IMRPhenomD()(f=array(mapped / m_seconds), **public_args)
    else:
        hp, _ = module.IMRPhenomD.evaluate(f=array(mapped / m_seconds), **public_args)
    carrier = np.abs(np.asarray(hp)) / math.sqrt(5.0 / (16.0 * math.pi))
    assert np.all(np.isfinite(carrier)) and np.all(carrier > 0)
    observed = np.abs(mode) / carrier
    expected = np.array([float(ref.beta43_1p5pn(
        str(y), str(g), str(m1), str(m2), str(chi1), str(chi2)))
        for y, g in zip(x, mapped)])
    np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-12)
    for join, lower, center, upper in ((f0, 1, 2, 3), (fr, 5, 6, 7)):
        left, right = ref.beta43_eq5_one_sided_slopes(
            str(join), str(f0), str(fr), str(f22),
            str(m1), str(m2), str(chi1), str(chi2))
        secants = ((observed[center] - observed[lower]) / h,
                   (observed[upper] - observed[center]) / h)
        np.testing.assert_allclose(secants, (float(left), float(right)),
                                   rtol=5e-5, atol=5e-8)


@pytest.mark.parametrize("variant", ("phenomd_runtime", "london_2018"))
def test_public_beta43_symmetric_zero(backend, variant):
    _, module, array = backend
    model = module.IMRPhenomHM(modes=((4, 3),))
    m1 = m2 = 30.0
    chi1 = chi2 = 0.2
    qnm_args = (array(m1), array(m2), array(chi1), array(chi2))
    fr = float(module.IMRPhenomHM.mode_qnm_frequencies(4, 3, *qnm_args)[0])
    f22 = float(module.IMRPhenomHM.mode_qnm_frequencies(2, 2, *qnm_args)[0])
    f0 = float(fmap.outer_map_cutoffs(variant, fr, f22)[0])
    x = np.array((f0 / 2, f0, (f0 + fr) / 2, fr, fr * 1.1))
    mode = np.asarray(model(
        f=array(x / ((m1 + m2) * float(module.MSUN_SEC))),
        mass1=m1, mass2=m2, chi1=chi1, chi2=chi2,
        distance=100.0, inclination=0.0, phic=0.0, tc=0.0, f_ref=None,
        hm_mapping_transitions=variant, hm_43_pn_variant="mishra_eq12g",
        return_unprojected_modes=True)[2][(4, 3)])
    assert np.all(np.isfinite(mode))
    np.testing.assert_array_equal(mode, np.zeros_like(mode))


def test_public_beta43_default_and_paper_variant_are_separate(backend):
    _, module, array = backend
    model = module.IMRPhenomHM(modes=((4, 3),))
    args = dict(f=array([70.0, 100.0]), mass1=35.0, mass2=20.0,
                chi1=0.2, chi2=-0.1, distance=100.0, inclination=0.0,
                phic=0.0, tc=0.0, f_ref=None, return_unprojected_modes=True)

    def mode(**options):
        return np.asarray(model(**args, **options)[2][(4, 3)])

    default = mode()
    explicit_default = mode(hm_43_pn_variant="binary_compatible")
    paper = mode(hm_43_pn_variant="mishra_eq12g")
    np.testing.assert_array_equal(default, explicit_default)
    assert np.max(np.abs(default - paper)) / np.max(np.abs(default)) > 1e-6
    with pytest.raises(ValueError, match=r"Unknown HM \(4,3\) PN variant"):
        mode(hm_43_pn_variant="unknown")
