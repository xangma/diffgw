"""XHM v2 Eqs. (2), (4)-(9): complete five-mode positive-frequency projection.

The expected angular coefficients come from Wigner matrices, not either native
frontend's polynomial spin-weighted-harmonic helper. Intrinsic carriers are
inputs here; their amplitude, phase, and calibration have separate gates.
"""

import cmath
import math

import pytest

pytest.importorskip("sympy")

from diffgw_reference import imrphenomxhm_projection as ref


FREQUENCIES = (35.0, 145.0, 290.0)
ANGLES = ((0.37, 0.27), (1.21, -0.41), (2.47, 0.63))
PARAMETERS = dict(mass1=35.0, mass2=20.0, chi1=0.4, chi2=-0.3,
                  distance=400.0)


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device == "mps":
        pytest.fail("The XHM projection reference requires FP64, unavailable on MPS.")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.imrphenom_xhm import IMRPhenomXHM
        if reference_device == "cuda" and not torch.cuda.is_available():
            pytest.fail("Requested Torch CUDA math-reference device is unavailable.")
        def array(values):
            return torch.as_tensor(values, dtype=torch.float64,
                                   device=reference_device)
        yield lambda **kwargs: IMRPhenomXHM()(f=array(FREQUENCIES), **kwargs)
    else:
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenom_xhm import IMRPhenomXHM
        platform = "gpu" if reference_device == "cuda" else "cpu"
        try:
            device = jax.devices(platform)[0]
        except (RuntimeError, IndexError) as exc:
            pytest.fail(f"Requested JAX {reference_device} device is unavailable: {exc}")
        previous = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)
        try:
            with jax.default_device(device):
                yield lambda **kwargs: IMRPhenomXHM.evaluate(
                    f=jnp.asarray(FREQUENCIES, dtype=jnp.float64), **kwargs)
        finally:
            jax.config.update("jax_enable_x64", previous)


def _close(actual, expected, scale):
    assert abs(complex(actual) - expected) <= 4e-11 * scale + 1e-45


def test_scalar_harmonic_normalization_and_odd_ell_sign():
    inclination, phi0 = ANGLES[0]
    c = math.cos(inclination)
    y22 = math.sqrt(5 / (64 * math.pi)) * cmath.exp(-2j * phi0)
    plus, cross = ref.evaluate(2, 2, inclination, phi0)
    _close(plus, y22 * (1 + c * c), 1.0)
    _close(cross, -2j * y22 * c, 1.0)
    # Analytic spin -2 harmonics at simple angles fix the odd-l orientation.
    plus, cross = ref.evaluate(3, 2, 0.0, 0.0)
    norm32 = math.sqrt(7 / (4 * math.pi)) / 2
    _close(plus, -norm32, 1.0)
    _close(cross, 1j * norm32, 1.0)
    plus, cross = ref.evaluate(3, 3, math.pi / 2, 0.0)
    _close(plus, math.sqrt(21 / (128 * math.pi)), 1.0)
    _close(cross, 0j, 1.0)


@pytest.mark.parametrize("inclination,phi0", ANGLES)
def test_each_mode_and_sum_from_independent_projection(backend, inclination, phi0):
    kwargs = dict(PARAMETERS, inclination=inclination, phi0=phi0,
                  phic=0.23, tc=0.013)
    hp, hc, projected = backend(**kwargs, return_modes=True)
    _, _, carrier = backend(**kwargs, return_unprojected_modes=True)
    assert tuple(projected) == ref.MODES
    expected_plus = [0j] * len(FREQUENCIES)
    expected_cross = [0j] * len(FREQUENCIES)
    scale_sum = [0.0] * len(FREQUENCIES)
    for l, m in ref.MODES:
        d_plus, d_cross = ref.evaluate(l, m, inclination, phi0)
        for index in range(len(FREQUENCIES)):
            h_mode = complex(carrier[(l, m)][index])
            scale = abs(h_mode)
            assert scale > 0, (l, m, index)
            e_plus, e_cross = d_plus * h_mode, d_cross * h_mode
            _close(projected[(l, m)][0][index], e_plus, scale)
            _close(projected[(l, m)][1][index], e_cross, scale)
            expected_plus[index] += e_plus
            expected_cross[index] += e_cross
            scale_sum[index] += scale
    for index in range(len(FREQUENCIES)):
        _close(hp[index], expected_plus[index], scale_sum[index])
        _close(hc[index], expected_cross[index], scale_sum[index])


def test_common_time_and_mode_phase_factor(backend):
    base = backend(**PARAMETERS, return_unprojected_modes=True)[2]
    shifted = backend(**PARAMETERS, phic=0.23, tc=0.013,
                      return_unprojected_modes=True)[2]
    for l, m in ref.MODES:
        for index, frequency in enumerate(FREQUENCIES):
            factor = cmath.exp(-1j * (2 * math.pi * frequency * 0.013 - m * 0.23))
            original = complex(base[(l, m)][index])
            _close(shifted[(l, m)][index], original * factor, abs(original))
