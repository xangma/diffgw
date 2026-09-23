"""Independent Eq. (7) NQC matching checks for both native frontends."""

import mpmath as mp
import pytest

from dev.native_rom.reference.nqc import match_22
from dev.native_rom.reference.published import evaluate, peak_targets


def _case(nu, chi, match_time):
    """Analytic trajectory with degree-six amplitude products and cubic phase bases."""
    def u(t):
        return mp.mpf("0.6") + mp.mpf("0.12")*t

    def momentum(t):
        return mp.mpf("-0.3") + mp.mpf("0.04")*t

    def ratio(t):
        return mp.mpf("-0.15") + mp.mpf("0.025")*t

    def radius(t):
        return u(t)**-2

    def omega(t):
        return momentum(t)/(radius(t)*ratio(t))

    def amplitude(t):
        return mp.mpf("0.3") + mp.mpf("0.01")*t

    def frequency(t):
        return mp.mpf("0.4") + mp.mpf("0.02")*t + mp.mpf("0.0001")*t*t

    values = evaluate(peak_targets(), nu, chi, digits=70)
    keys = ("amplitude", "amplitude_derivative", "amplitude_curvature",
            "frequency", "frequency_derivative")
    targets = tuple(mp.mpf(str(values[key])) for key in keys)
    functions = (radius, momentum, omega, amplitude, frequency)
    samples = [mp.mpf(str(match_time)) + mp.mpf(k-4)/4 for k in range(9)]
    return functions, targets, samples


@pytest.mark.parametrize("nu,chi,match_time", [
    ("0.24", "0.1", "0.13"),
    ("0.18", "-0.3", "-0.2"),
])
def test_scalar_matching_satisfies_five_paper_conditions(nu, chi, match_time):
    with mp.workdps(70):
        functions, targets, _ = _case(nu, chi, match_time)
        r, p, omega, amplitude, frequency = functions
        a, b = match_22(*functions, match_time, targets, digits=70)
        t = mp.mpf(match_time)

        def q(x):
            z = p(x)/(r(x)*omega(x))
            return (z*z, z*z/r(x), z*z/r(x)**mp.mpf("1.5"))

        def phase(x):
            z = p(x)/(r(x)*omega(x))
            return (z, z*p(x)**2)

        def corrected_amplitude(x):
            return amplitude(x)*(1+sum(ai*qi for ai, qi in zip(a, q(x))))

        def corrected_frequency(x):
            return frequency(x)-sum(bi*mp.diff(lambda y, i=i: phase(y)[i], x)
                                    for i, bi in enumerate(b))

        actual = (corrected_amplitude(t), mp.diff(corrected_amplitude, t),
                  mp.diff(corrected_amplitude, t, 2), corrected_frequency(t),
                  mp.diff(corrected_frequency, t))
        for found, expected in zip(actual, targets):
            assert abs(found-expected) < mp.mpf("1e-55")


@pytest.mark.parametrize("nu,chi,match_time", [
    ("0.24", "0.1", "0.13"),
    ("0.18", "-0.3", "-0.2"),
])
@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_frontend_matching_against_scalar_reference(nu, chi, match_time, backend):
    with mp.workdps(70):
        functions, targets, times = _case(nu, chi, match_time)
        expected_a, expected_b = match_22(*functions, match_time, targets, digits=70)
        arrays = [[float(t) for t in times]]
        arrays += [[float(function(t)) for t in times] for function in functions]
        if backend == "torch":
            import torch
            from torchwave.approximants.eob.nqc import NQCMatchingTargets, calibrate_nqc_22

            values = [torch.tensor(array, dtype=torch.float64) for array in arrays]
            target = NQCMatchingTargets(*(torch.tensor(float(v), dtype=torch.float64)
                                          for v in targets))
            found = calibrate_nqc_22(*values, torch.tensor(float(match_time), dtype=torch.float64),
                                     target, degree=6).coefficients
        else:
            import jax
            import jax.numpy as jnp
            from jaxwave.approximants.eob.nqc import NQCMatchingTargets, calibrate_nqc_22

            jax.config.update("jax_enable_x64", True)
            values = [jnp.asarray(array, dtype=jnp.float64) for array in arrays]
            target = NQCMatchingTargets(*(jnp.asarray(float(v), dtype=jnp.float64)
                                          for v in targets))
            found = calibrate_nqc_22(*values, jnp.asarray(float(match_time)),
                                     target, degree=6).coefficients
        for actual, expected in zip(found.amplitude, expected_a):
            assert float(actual) == pytest.approx(float(expected), rel=2e-8, abs=2e-8)
        for actual, expected in zip(found.phase, expected_b):
            assert float(actual) == pytest.approx(float(expected), rel=2e-8, abs=2e-8)
