"""Bounded direct-class 220 physical-energy whole-waveform qualification.

The scalar QNM root, spectral angular projection, energy amplitude and TD/FD
composition never use native QNM or angular outputs. CPU FP64 is required.
"""

from functools import lru_cache

import mpmath as mp
import numpy as np
import pytest

from diffgw_reference.kerr_ringdown import (
    direct_class_220_frequency, direct_class_220_inputs, direct_class_220_time,
)


CASES = (
    (0.35, 53.0, 230.0, 0.017, 1.1, 0.41, 0.002),
    (0.7, 92.0, 430.0, 0.005, 0.55, -0.27, -0.001),
)
FREQUENCIES = np.array([-130.0, 0.0, 35.0, 110.0, 270.0])
VALUE_TOL = 2e-10
GRAD_TOL = 3e-8


@lru_cache(maxsize=2)
def _scalar_case(index):
    spin, mass, distance, energy, angle, phi, onset = CASES[index]
    with mp.workdps(40):
        inputs = direct_class_220_inputs(mass, spin, distance, energy, angle, phi)
        times = np.array([onset - 0.001, onset, onset + 0.002, onset + 0.008])
        td = tuple(np.array([
            float(direct_class_220_time(t, onset, *inputs)[i]) for t in times
        ]) for i in (0, 1))
        fd = tuple(np.array([
            complex(direct_class_220_frequency(f, onset, *inputs)[i])
            for f in FREQUENCIES
        ]) for i in (0, 1))
    return times, td, fd


def _kwargs(index):
    spin, mass, distance, energy, angle, phi, onset = CASES[index]
    return dict(final_mass=mass, final_spin=spin, distance=distance,
                energy_fraction=energy, inclination=angle, phi=phi, t0=onset,
                energy_mode=(2, 2, 0), angular_basis="spheroidal",
                qnm_backend="independent")


def _backend(backend):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.kerr_ringdown import KerrRingdown
        return torch, KerrRingdown, lambda x: torch.as_tensor(x, dtype=torch.float64)
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxwave.approximants.kerr_ringdown import KerrRingdown
    return jax, KerrRingdown, lambda x: jnp.asarray(x, dtype=jnp.float64)


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case_index", (0, 1))
def test_direct_class_220_entire_td_fd_against_scalar(backend, case_index):
    _, klass, array = _backend(backend)
    times, expected_td, expected_fd = _scalar_case(case_index)
    kwargs = _kwargs(case_index)
    for method, grid, expected in (
        (klass.time_domain, times, expected_td),
        (klass.frequency_domain, FREQUENCIES, expected_fd),
    ):
        actual = np.stack([np.asarray(x) for x in method(array(grid), **kwargs)])
        reference = np.stack(expected)
        relative_l2 = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
        assert relative_l2 < VALUE_TOL


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("domain", ("time", "frequency"))
@pytest.mark.parametrize("parameter", ("distance", "energy_fraction"))
def test_direct_class_220_extrinsic_gradient(backend, domain, parameter):
    library, klass, array = _backend(backend)
    times, _, _ = _scalar_case(0)
    grid = times if domain == "time" else FREQUENCIES
    method = klass.time_domain if domain == "time" else klass.frequency_domain
    kwargs = _kwargs(0)
    value = kwargs[parameter]

    def signal(x):
        args = dict(kwargs, **{parameter: x})
        plus, cross = method(array(grid), **args)
        if domain == "frequency":
            plus, cross = plus.real, cross.real
        return 1e20 * (plus[2] + 0.3 * cross[3])

    if backend == "torch":
        x = library.tensor(value, dtype=library.float64, requires_grad=True)
        output = signal(x)
        derivative = library.autograd.grad(output, x)[0]
        output, derivative = float(output.detach()), float(derivative.detach())
    else:
        x = array(value)
        output, derivative = float(signal(x)), float(library.grad(signal)(x))
    expected = -output / value if parameter == "distance" else output / (2 * value)
    assert np.isclose(derivative, expected, rtol=GRAD_TOL, atol=1e-12)
