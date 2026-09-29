"""Paper-form scalar checks of the bounded, physical-energy Kerr 220 carrier.

QNM and angular factors are declared inputs to the reference; their provenance
and numerical accuracy require separate evidence.
"""

import math

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")

from diffgw_core.constants import C_SI, MPC_SI, MSUN_SEC
from diffgw_reference import kerr_ringdown as ref


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device == "mps":
        pytest.fail("Kerr scalar reference requires FP64, which MPS does not support")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.kerr_qnm_generated import signed_qnm_220
        from torchwave.approximants.kerr_angular_table_convention import (
            spin_weighted_spheroidal_harmonic_table_compat as angular,
        )
        from torchwave.approximants.kerr_ringdown_public_clean import kerr220_td, kerr220_fd
        if reference_device == "cuda" and not torch.cuda.is_available():
            pytest.fail("Requested CUDA math-reference qualification is unavailable")
        array = lambda x: torch.as_tensor(x, dtype=torch.float64, device=reference_device)
        yield signed_qnm_220, angular, kerr220_td, kerr220_fd, array
    else:
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp
        from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220
        from jaxwave.approximants.kerr_ringdown_public_clean import (
            spin_weighted_spheroidal_harmonic_table_compat as angular,
            kerr220_td, kerr220_fd,
        )
        platform = "gpu" if reference_device == "cuda" else "cpu"
        try:
            device = jax.devices(platform)[0]
        except (RuntimeError, IndexError) as exc:
            pytest.fail(f"Requested {reference_device} math-reference qualification is unavailable: {exc}")
        previous = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)
        try:
            with jax.default_device(device):
                array = lambda x: jnp.asarray(x, dtype=jnp.float64)
                yield evaluate_kerr_qnm_220, angular, kerr220_td, kerr220_fd, array
        finally:
            jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("mass,spin,distance,energy,inc,phase,onset", [
    (53., .35, 230., .017, 1.1, .41, .002),
    (92., .82, 430., .005, .55, -.27, -.001),
])
def test_public_220_carrier_against_scalar_reference(
    backend, mass, spin, distance, energy, inc, phase, onset,
):
    qnm, angular, td, fd, array = backend
    dimensionless_omega, dimensionless_decay = qnm(array(spin))
    omega = float(dimensionless_omega) / (mass * MSUN_SEC)
    decay = float(dimensionless_decay) / (mass * MSUN_SEC)
    plus = complex(angular(array(spin), array(inc), array(phase)))
    mirror = complex(angular(array(spin), array(math.pi - inc), array(phase)))
    args = dict(final_mass=mass, final_spin=spin, distance=distance,
                energy_fraction=energy, inclination=inc, phic=phase, tc=onset)
    with mp.workdps(60):
        amplitude = ref.energy_amplitude(
            mass * MSUN_SEC, distance * MPC_SI / C_SI, energy, omega, decay
        )
        times = np.array([onset - .001, onset, onset + .002, onset + .008])
        hp, hc = td(array(times), **args)
        for index, time in enumerate(times):
            expected = ref.time_polarizations(time, onset, amplitude, omega, decay, plus, mirror)
            for actual, target in zip((hp[index], hc[index]), expected):
                assert abs(float(actual) - float(target)) / float(amplitude) < 5e-11

        frequencies = np.array([-130., 0., 35., 110., 270.])
        hp, hc = fd(array(frequencies), **args)
        for index, frequency in enumerate(frequencies):
            expected = ref.frequency_polarizations(
                frequency, onset, amplitude, omega, decay, plus, mirror
            )
            for actual, target in zip((hp[index], hc[index]), expected):
                assert abs(complex(actual) - complex(target)) / float(amplitude / decay) < 5e-11


def test_scalar_laplace_transform_agrees_with_time_quadrature():
    with mp.workdps(50):
        args = (35., .002, mp.mpf("0.2"), 850., 190.,
                complex(.31, -.12), complex(-.08, .07))
        analytic = ref.frequency_polarizations(*args)
        integrated = ref.frequency_by_quadrature(*args)
        for direct, numerical in zip(analytic, integrated):
            assert abs(direct - numerical) / abs(direct) < mp.mpf("1e-12")
