"""Eight-mode TaylorF2_HM checks against the independent scalar carrier.

Constants enter only at the API-unit boundary; expected expressions import no
Torch, JAX, or native mode coefficients. This is a sampled math gate, not a
physical accuracy claim or a same-model external waveform comparison.
"""

from functools import lru_cache

import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_core.constants import C_SI, MPC_SI, MSUN_SEC
from diffgw_reference import taylorf2_hm as ref


def exact(value):
    return sp.Rational(str(value))


CASES = (
    dict(mass1=35.0, mass2=12.0, chi1=0.0, chi2=0.0, distance=410.0,
         inclination=0.77, phic=0.3, phi0=-0.27, tc=0.012),
    dict(mass1=22.0, mass2=14.0, chi1=0.31, chi2=-0.22, distance=630.0,
         inclination=1.34, phic=-0.19, phi0=0.41, tc=-0.008),
)


def reference_values(case, frequency, f_ref=40.0):
    m1, m2 = exact(case["mass1"]), exact(case["mass2"])
    return {
        ref.F: exact(frequency),
        ref.F_REF: exact(f_ref),
        ref.M_SEC: (m1 + m2) * exact(MSUN_SEC),
        ref.D_SEC: exact(case["distance"]) * exact(MPC_SI) / exact(C_SI),
        ref.DELTA: (m1 - m2) / (m1 + m2),
        ref.CHI1: exact(case["chi1"]), ref.CHI2: exact(case["chi2"]),
        ref.IOTA: exact(case["inclination"]), ref.PHIC: exact(case["phic"]),
        ref.PHI0: exact(case["phi0"]), ref.TC: exact(case["tc"]),
    }


@pytest.fixture(params=["torch", "jax"])
def backend(request, reference_device):
    if reference_device == "mps":
        pytest.fail("The math-reference gate requires FP64, which MPS does not support.")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.taylorf2_hm import TaylorF2_HM
        if reference_device == "cuda" and not torch.cuda.is_available():
            pytest.fail("Requested CUDA math-reference qualification, but Torch CUDA is unavailable.")
        yield "torch", TaylorF2_HM, lambda x: torch.as_tensor(x, dtype=torch.float64, device=reference_device)
    else:
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp
        from jaxwave.approximants.taylorf2_hm import TaylorF2_HM
        platform = "gpu" if reference_device == "cuda" else "cpu"
        try:
            devices = jax.devices(platform)
        except RuntimeError as exc:
            pytest.fail(f"Requested {reference_device} math-reference qualification is unavailable: {exc}")
        previous = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)
        try:
            with jax.default_device(devices[0]):
                yield "jax", TaylorF2_HM, lambda x: jnp.asarray(x, dtype=jnp.float64)
        finally:
            jax.config.update("jax_enable_x64", previous)


def native(backend, case, frequencies, **options):
    name, model, array = backend
    args = {key: array(value) for key, value in case.items()}
    kwargs = dict(f=array(frequencies), f_isco_cutoff=False, return_modes=True, **args, **options)
    return model()(**kwargs) if name == "torch" else model.evaluate(**kwargs)


@lru_cache(maxsize=None)
def expected_expr(mode, use_ref):
    return ref.mode_expr(*mode, reference_frequency=use_ref)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("use_ref", [False, True])
def test_each_mode_complex_polarizations_and_sum(backend, case, use_ref):
    frequencies = (24.0, 37.5)
    hp, hc, modes = native(backend, case, frequencies, f_ref=40.0 if use_ref else None)
    assert tuple(modes) == ref.MODES
    for index, frequency in enumerate(frequencies):
        values = reference_values(case, frequency)
        scale = float(ref.evaluate(ref.mode_expr(2, 2)["amplitude"], values))
        expected_plus = expected_cross = 0j
        for mode in ref.MODES:
            expressions = expected_expr(mode, use_ref)
            for actual, key in ((modes[mode][0][index], "h_plus"),
                                (modes[mode][1][index], "h_cross")):
                expected = complex(ref.evaluate(expressions[key], values))
                assert abs(complex(actual) - expected) / scale < 4e-10, (mode, key)
                if key == "h_plus":
                    expected_plus += expected
                else:
                    expected_cross += expected
        assert abs(complex(hp[index]) - expected_plus) / scale < 4e-10
        assert abs(complex(hc[index]) - expected_cross) / scale < 4e-10


def test_angular_derivative_for_every_mode(backend):
    name, model, array = backend
    case = CASES[0]
    frequency = 31.25
    values = reference_values(case, frequency)
    scale = float(ref.evaluate(ref.mode_expr(2, 2)["amplitude"], values))
    if name == "torch":
        import torch
        phi0 = array(case["phi0"]).detach().requires_grad_()
        hp, hc, modes = native(backend, dict(case, phi0=phi0), (frequency,))
        for mode in ref.MODES:
            for polarization, key in ((modes[mode][0][0], "h_plus"),
                                      (modes[mode][1][0], "h_cross")):
                actual = complex(torch.autograd.grad(polarization.real, phi0,
                                                      retain_graph=True)[0],
                                 torch.autograd.grad(polarization.imag, phi0,
                                                     retain_graph=True)[0])
                expected = complex(ref.evaluate(sp.diff(expected_expr(mode, False)[key], ref.PHI0), values))
                assert abs(actual - expected) / scale < 6e-10, (mode, key)
    else:
        import jax
        for mode in ref.MODES:
            for part, key in ((0, "h_plus"), (1, "h_cross")):
                def component(phi):
                    args = {item: array(value) for item, value in case.items()}
                    args["phi0"] = phi
                    return model.evaluate(f=array((frequency,)), f_isco_cutoff=False,
                                          modes=(mode,), return_modes=True, **args)[2][mode][part][0]
                actual = complex(jax.grad(lambda p: component(p).real)(array(case["phi0"])),
                                 jax.grad(lambda p: component(p).imag)(array(case["phi0"])))
                expected = complex(ref.evaluate(sp.diff(expected_expr(mode, False)[key], ref.PHI0), values))
                assert abs(actual - expected) / scale < 6e-10, (mode, key)


def test_equal_mass_odd_modes_and_cutoff(backend):
    case = dict(CASES[0], mass1=24.0, mass2=24.0, chi1=0.0, chi2=0.0)
    values = reference_values(case, 20.0)
    for mode in ((2, 1), (3, 1), (3, 3), (4, 3)):
        assert sp.simplify(ref.amplitude_ratio(*mode, delta=0)) == 0
    _, _, modes = native(backend, case, (20.0,))
    for mode in ((2, 1), (3, 1), (3, 3), (4, 3)):
        assert complex(modes[mode][0][0]) == 0j
        assert complex(modes[mode][1][0]) == 0j
    # Sample on either side of each mode's Schwarzschild threshold.
    name, model, array = backend
    for mode in ref.MODES:
        cut = float(ref.evaluate(ref.cutoff_frequency_expr(*mode), values))
        args = {key: array(value) for key, value in case.items()}
        below, above = 0.99 * cut, 1.01 * cut
        output = model()(f=array((below, above)), modes=(mode,), return_modes=True,
                         f_isco_cutoff=True, **args) if name == "torch" else model.evaluate(
                             f=array((below, above)), modes=(mode,), return_modes=True,
                             f_isco_cutoff=True, **args)
        assert complex(output[2][mode][0][1]) == 0j
        assert complex(output[2][mode][1][1]) == 0j
        if mode not in ((2, 1), (3, 1), (3, 3), (4, 3)):
            assert complex(output[2][mode][0][0]) != 0j


def test_component_mass_derivative_against_scalar_difference(backend):
    name, model, array = backend
    case, mode, frequency = CASES[1], (3, 3), 30.0
    step = 1e-4
    expression = expected_expr(mode, False)["h_plus"]
    upper = complex(ref.evaluate(expression, reference_values(dict(case, mass1=case["mass1"] + step), frequency)))
    lower = complex(ref.evaluate(expression, reference_values(dict(case, mass1=case["mass1"] - step), frequency)))
    expected = ((upper - lower) / (2 * step)).real
    scale = float(ref.evaluate(ref.mode_expr(2, 2)["amplitude"], reference_values(case, frequency)))
    if name == "torch":
        import torch
        mass1 = array(case["mass1"]).detach().requires_grad_()
        args = {key: array(value) for key, value in case.items()}
        args["mass1"] = mass1
        result = model()(f=array((frequency,)), modes=(mode,), return_modes=True,
                         f_isco_cutoff=False, **args)[2][mode][0][0].real
        actual = float(torch.autograd.grad(result, mass1)[0])
    else:
        import jax
        def component(mass1):
            args = {key: array(value) for key, value in case.items()}
            args["mass1"] = mass1
            return model.evaluate(f=array((frequency,)), modes=(mode,), return_modes=True,
                                  f_isco_cutoff=False, **args)[2][mode][0][0].real
        actual = float(jax.grad(component)(array(case["mass1"])))
    assert abs(actual - expected) / scale < 2e-7
