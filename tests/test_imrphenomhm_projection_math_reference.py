"""London Eq. (1) and positive-frequency Fourier mode-pair projection gate.

The oracle builds spin -2 harmonics from Wigner matrices and takes only public
unprojected modes as carrier inputs. It does not assess intrinsic mode physics.
"""

import cmath
import math

import pytest

pytest.importorskip("sympy")

from diffgw_reference import imrphenomhm_projection as ref


FREQUENCIES = (35.0, 145.0, 290.0)
ANGLES = ((0.0, 0.27), (math.pi / 2, -0.41), (0.83, 0.37), (2.47, -0.63))
PARAMETERS = dict(mass1=35.0, mass2=20.0, chi1=0.4, chi2=-0.3,
                  distance=400.0)


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device == "mps":
        pytest.fail("The HM projection reference requires FP64, unavailable on MPS.")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.imrphenom_hm import IMRPhenomHM
        if reference_device == "cuda" and not torch.cuda.is_available():
            pytest.fail("Requested Torch CUDA math-reference device is unavailable.")

        def evaluate(**kwargs):
            f = torch.as_tensor(FREQUENCIES, dtype=torch.float64,
                                device=reference_device)
            return IMRPhenomHM()(f=f, **kwargs)

        yield evaluate
    else:
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenom_hm import IMRPhenomHM
        platform = "gpu" if reference_device == "cuda" else "cpu"
        try:
            device = jax.devices(platform)[0]
        except (RuntimeError, IndexError) as exc:
            pytest.fail(f"Requested JAX {reference_device} device is unavailable: {exc}")
        previous = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)
        try:
            with jax.default_device(device):
                def evaluate(**kwargs):
                    f = jnp.asarray(FREQUENCIES, dtype=jnp.float64)
                    return IMRPhenomHM.evaluate(f=f, **kwargs)

                yield evaluate
        finally:
            jax.config.update("jax_enable_x64", previous)


def _close(actual, expected, scale):
    assert abs(complex(actual) - expected) <= 4e-11 * scale + 1e-45


def test_scalar_wigner_normalization_selection_and_odd_ell_sign():
    inclination, phi0 = ANGLES[2]
    c = math.cos(inclination)
    y22 = math.sqrt(5 / (64 * math.pi)) * cmath.exp(2j * phi0)
    plus, cross = ref.evaluate(2, 2, inclination, phi0)
    _close(plus, y22 * (1 + c * c), 1.0)
    _close(cross, -2j * y22 * c, 1.0)

    pole_survivors = {(2, 2), (3, 2)}
    for mode in ref.MODES:
        plus, cross = ref.evaluate(*mode, 0.0, 0.0)
        if mode in pole_survivors:
            assert abs(plus) > 0.1
            _close(cross, -1j * plus, 1.0)
        else:
            _close(plus, 0j, 1.0)
            _close(cross, 0j, 1.0)
    plus32, cross32 = ref.evaluate(3, 2, 0.0, 0.0)
    norm32 = math.sqrt(7 / (4 * math.pi)) / 2
    _close(plus32, norm32, 1.0)
    _close(cross32, -1j * norm32, 1.0)


@pytest.mark.parametrize("inclination,phi0", ANGLES)
def test_each_public_mode_and_six_mode_sum(backend, inclination, phi0):
    kwargs = dict(PARAMETERS, inclination=inclination, phi0=phi0,
                  phic=0.23, tc=0.013)
    hp, hc, projected = backend(**kwargs, return_modes=True)
    _, _, carrier = backend(**kwargs, return_unprojected_modes=True)
    assert tuple(projected) == ref.MODES
    assert tuple(carrier) == ref.MODES

    expected_plus = [0j] * len(FREQUENCIES)
    expected_cross = [0j] * len(FREQUENCIES)
    total_scale = [0.0] * len(FREQUENCIES)
    for l, m in ref.MODES:
        d_plus, d_cross = ref.evaluate(l, m, inclination, phi0)
        for index in range(len(FREQUENCIES)):
            mode_value = complex(carrier[(l, m)][index])
            scale = abs(mode_value)
            assert scale > 0, (l, m, index)
            expected_mode_plus = d_plus * mode_value
            expected_mode_cross = d_cross * mode_value
            _close(projected[(l, m)][0][index], expected_mode_plus, scale)
            _close(projected[(l, m)][1][index], expected_mode_cross, scale)
            expected_plus[index] += expected_mode_plus
            expected_cross[index] += expected_mode_cross
            total_scale[index] += scale
    for index in range(len(FREQUENCIES)):
        _close(hp[index], expected_plus[index], total_scale[index])
        _close(hc[index], expected_cross[index], total_scale[index])


def test_azimuth_orbital_phase_and_time_covariance(backend):
    base = backend(**PARAMETERS, inclination=0.83, phi0=0.0,
                   return_unprojected_modes=True)[2]
    shifted_kwargs = dict(PARAMETERS, inclination=0.83, phi0=0.37,
                          phic=0.23, tc=0.013)
    _, _, shifted_carrier = backend(**shifted_kwargs, return_unprojected_modes=True)
    _, _, shifted_projection = backend(**shifted_kwargs, return_modes=True)
    for l, m in ref.MODES:
        d_plus, d_cross = ref.evaluate(l, m, 0.83, 0.0)
        for index, frequency in enumerate(FREQUENCIES):
            original = complex(base[(l, m)][index])
            factor = cmath.exp(1j * (m * 0.23 - 2 * math.pi * frequency * 0.013))
            _close(shifted_carrier[(l, m)][index], factor * original, abs(original))
            azimuth = cmath.exp(1j * m * 0.37)
            _close(shifted_projection[(l, m)][0][index],
                   azimuth * d_plus * factor * original, abs(original))
            _close(shifted_projection[(l, m)][1][index],
                   azimuth * d_cross * factor * original, abs(original))


def test_odd_m_exchange_and_symmetric_zero(backend):
    parameters = dict(PARAMETERS, inclination=0.83, phi0=0.17)
    _, _, first = backend(**parameters, return_unprojected_modes=True)
    swapped = dict(parameters, mass1=PARAMETERS["mass2"],
                   mass2=PARAMETERS["mass1"], chi1=PARAMETERS["chi2"],
                   chi2=PARAMETERS["chi1"])
    _, _, second = backend(**swapped, return_unprojected_modes=True)
    for l, m in ref.MODES:
        for index in range(len(FREQUENCIES)):
            value = complex(first[(l, m)][index])
            _close(second[(l, m)][index], (-1) ** m * value, abs(value))

    equal = dict(parameters, mass1=30.0, mass2=30.0, chi1=0.2, chi2=0.2)
    _, _, symmetric = backend(**equal, return_unprojected_modes=True)
    for mode in ((2, 1), (3, 3), (4, 3)):
        for value in symmetric[mode]:
            _close(value, 0j, 1e-23)
