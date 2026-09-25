"""Paper-level convention checks for the native TaylorF2_HM carrier.

These checks use Garcia-Quiros et al. (2020), Eqs. (7)--(9), (29), and
(34).  They reconstruct the positive-frequency negative-m carrier from the
public per-mode polarizations without importing either frontend's projection
or phase helpers.  This is mathematical evidence, not an external waveform
comparison.
"""

from functools import lru_cache

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import taylorf2_hm as ref


CASE = dict(
    mass1=41.0,
    mass2=17.0,
    chi1=0.23,
    chi2=-0.14,
    distance=510.0,
    inclination=0.91,
    phic=0.21,
    tc=0.0043,
    phi0=0.37,
)
FREQUENCIES = np.linspace(28.0, 92.0, 9)


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device == "mps":
        pytest.fail("The math-reference gate requires FP64, which MPS does not support.")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.taylorf2_hm import TaylorF2_HM

        if reference_device == "cuda" and not torch.cuda.is_available():
            pytest.fail(
                "Requested CUDA math-reference qualification, but Torch CUDA is unavailable."
            )
        yield (
            "torch",
            TaylorF2_HM,
            lambda value: torch.as_tensor(
                value, dtype=torch.float64, device=reference_device
            ),
        )
        return

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jaxwave.approximants.taylorf2_hm import TaylorF2_HM

    platform = "gpu" if reference_device == "cuda" else "cpu"
    try:
        devices = jax.devices(platform)
    except RuntimeError as exc:
        pytest.fail(
            f"Requested {reference_device} math-reference qualification is unavailable: {exc}"
        )
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        with jax.default_device(devices[0]):
            yield "jax", TaylorF2_HM, lambda value: jnp.asarray(
                value, dtype=jnp.float64
            )
    finally:
        jax.config.update("jax_enable_x64", previous)


def native(backend, frequencies, *, case=CASE, modes=ref.MODES, f_ref=None):
    name, model, array = backend
    args = {key: array(value) for key, value in case.items()}
    options = dict(
        f=array(frequencies),
        modes=modes,
        f_ref=None if f_ref is None else array(f_ref),
        f_isco_cutoff=False,
        return_modes=True,
        **args,
    )
    return model()(**options) if name == "torch" else model.evaluate(**options)


@lru_cache(maxsize=None)
def paper_projection(mode):
    """Garcia-Quiros et al. (2020), Eqs. (8)--(9), via Wigner d."""
    values = {
        ref.IOTA: sp.Rational(str(CASE["inclination"])),
        ref.PHI0: sp.Rational(str(CASE["phi0"])),
    }
    return tuple(
        complex(ref.evaluate(expression, values))
        for expression in ref.projection(*mode)
    )


def reconstructed_carrier(mode_data, mode):
    """Recover h_{l,-m} independently from each published polarization."""
    d_plus, d_cross = paper_projection(mode)
    h_plus, h_cross = (np.asarray(value) for value in mode_data[mode])
    from_plus = h_plus / d_plus
    from_cross = h_cross / d_cross
    np.testing.assert_allclose(from_plus, from_cross, rtol=7e-13, atol=0.0)
    return from_plus


def unit_carrier(carrier):
    return carrier / np.abs(carrier)


def test_published_projection_recovers_native_negative_m_carrier(backend):
    """Eqs. (7)--(9) identify one common h_{l,-m} for every mode pair."""
    _, _, mode_data = native(backend, FREQUENCIES)
    assert tuple(mode_data) == ref.MODES
    for mode in ref.MODES:
        carrier = reconstructed_carrier(mode_data, mode)
        assert np.all(np.isfinite(carrier))
        assert np.all(np.abs(carrier) > 0.0)


@pytest.mark.parametrize("f_ref", (None, 43.0))
def test_published_spa_phase_map_and_low_frequency_offset(backend, f_ref):
    """Eqs. (29),(34): Phi_lm=m/2 Phi_22(2f/m)+3pi/4(1-m/2)."""
    _, _, mode_data = native(backend, FREQUENCIES, f_ref=f_ref)
    for mode in ref.MODES:
        _, m = mode
        carrier = unit_carrier(reconstructed_carrier(mode_data, mode))
        scaled_frequencies = 2.0 * FREQUENCIES / m
        scaled_f_ref = None if f_ref is None else 2.0 * f_ref / m
        _, _, quadrupole_data = native(
            backend,
            scaled_frequencies,
            modes=((2, 2),),
            f_ref=scaled_f_ref,
        )
        quadrupole = unit_carrier(
            reconstructed_carrier(quadrupole_data, (2, 2))
        )

        # Squaring removes the half-integer power ambiguity for odd m.
        actual = carrier**2 / quadrupole**m
        delta = 3.0 * np.pi / 4.0 * (1.0 - m / 2.0)
        expected = np.full_like(actual, np.exp(-2j * delta))
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("f_ref", (None, 43.0))
def test_native_exp_minus_i_phase_signs(backend, f_ref):
    """Eq. (7) implies +im*dphic and -i2pi*f*dt carrier shifts."""
    _, _, baseline_data = native(backend, FREQUENCIES, f_ref=f_ref)
    delta_phic = 0.137
    delta_tc = 0.0017
    _, _, phase_data = native(
        backend,
        FREQUENCIES,
        case=dict(CASE, phic=CASE["phic"] + delta_phic),
        f_ref=f_ref,
    )
    _, _, time_data = native(
        backend,
        FREQUENCIES,
        case=dict(CASE, tc=CASE["tc"] + delta_tc),
        f_ref=f_ref,
    )

    for mode in ref.MODES:
        _, m = mode
        baseline = reconstructed_carrier(baseline_data, mode)
        phase_shifted = reconstructed_carrier(phase_data, mode)
        time_shifted = reconstructed_carrier(time_data, mode)
        np.testing.assert_allclose(
            phase_shifted / baseline,
            np.exp(1j * m * delta_phic),
            rtol=2e-12,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            time_shifted / baseline,
            np.exp(-2j * np.pi * FREQUENCIES * delta_tc),
            rtol=2e-12,
            atol=2e-12,
        )
