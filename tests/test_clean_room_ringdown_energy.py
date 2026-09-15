"""Independent sphere/time energy integrals, units and derivatives; no QNM data."""

import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad

from torchwave.approximants.kerr_ringdown import _ringdown_projection
from torchwave.approximants.kerr_ringdown_energy import (
    ringdown_amplitude_from_energy_fraction as amplitude,
)
from torchwave.constants import C_SI, G_SI, MPC_SI, MSUN_SI


def _sphere():
    cosine, weights = np.polynomial.legendre.leggauss(32)
    azimuth = np.arange(64)[None, :] * (2 * math.pi / 64)
    return cosine[:, None], azimuth, weights[:, None] * (2 * math.pi / 64)


@pytest.mark.parametrize("quality", [.5, 2., 10.])
@pytest.mark.parametrize("convention", ["single_mode", "pair_member", "native_pair"])
def test_energy_from_independent_sphere_and_time_quadrature(quality, convention):
    mass, distance, efficiency, tau, phase = 63., 410., .017, .004, .73
    omega = 2 * quality / tau
    strain = amplitude(mass, distance, efficiency, omega, tau, convention=convention).item()
    cosine, azimuth, weights = _sphere()
    # Explicit unit-normalized spherical 22 functions; no conversion formula
    # or stored waveform/QNM values are used by this energy integral.
    yp = math.sqrt(5 / (64 * math.pi)) * (1 + cosine)**2 * np.exp(2j * azimuth)
    ym = math.sqrt(5 / (64 * math.pi)) * (1 - cosine)**2 * np.exp(-2j * azimuth)
    if convention == "native_pair":
        dp, dc = _ringdown_projection(
            2, 2, torch.tensor(0j), torch.from_numpy(np.arccos(cosine)),
            torch.from_numpy(azimuth), "spherical",
        )
        dp, dc = dp.numpy(), dc.numpy()

    def integrand(x):
        # Derivatives with respect to x=u/tau, leaving SI factors outside
        # quadrature so the error tolerance acts on order-one numbers.
        ds = (-1 - 2j * quality) * np.exp((-1 - 2j * quality) * x + 1j * phase)
        if convention == "single_mode":
            dh = ds * yp
            power = dh.real**2 + dh.imag**2
        elif convention == "pair_member":
            dh = ds * yp + ds.conjugate() * ym
            power = dh.real**2 + dh.imag**2
        else:
            # Native s uses positive frequency, then projects two real strains.
            ds = ds.conjugate()
            power = (ds * dp).real**2 + (ds * dc).real**2
        return float(np.sum(weights * power))

    integral = quad(integrand, 0, np.inf, epsabs=1e-10, epsrel=3e-12)[0]
    energy = C_SI**3 * (distance * MPC_SI)**2 / (16 * math.pi * G_SI) * strain**2 / tau * integral
    np.testing.assert_allclose(energy / (mass * MSUN_SI * C_SI**2), efficiency, rtol=5e-13)


@pytest.mark.parametrize("ell,m,basis", [(2, 1, "spherical"), (3, 3, "spherical"), (3, 2, "spheroidal")])
def test_native_pair_energy_other_modes_and_phases(ell, m, basis):
    mass, distance, efficiency, omega, tau = 51., 230., .009, 1700., .003
    strain = amplitude(mass, distance, efficiency, omega, tau).item()
    cosine, azimuth, weights = _sphere()
    # Nonzero complex spheroidicity checks numerical angular normalization;
    # its frequency is caller-supplied and requires no QNM backend.
    dp, dc = _ringdown_projection(
        ell, m, torch.tensor(.4 - .08j, dtype=torch.complex128),
        torch.from_numpy(np.arccos(cosine)), torch.from_numpy(azimuth), basis,
    )
    dp, dc = dp.numpy(), dc.numpy()
    for phase in (0., .41, 2.3):
        def integrand(x):
            ds = (-1 + 1j * omega * tau) * np.exp((-1 + 1j * omega * tau) * x + 1j * phase)
            return float(np.sum(weights * ((ds * dp).real**2 + (ds * dc).real**2)))

        integral = quad(integrand, 0, np.inf, epsabs=1e-10, epsrel=3e-12)[0]
        energy = C_SI**3 * (distance * MPC_SI)**2 / (16 * math.pi * G_SI) * strain**2 / tau * integral
        np.testing.assert_allclose(energy / (mass * MSUN_SI * C_SI**2), efficiency, rtol=5e-13)


def test_si_units_and_mass_distance_efficiency_scaling():
    args = (60., 300., .02, 1800., .004)
    base = amplitude(*args, convention="single_mode")
    mass, distance, efficiency, omega, tau = args
    expected = math.sqrt(32 * math.pi * G_SI * mass * MSUN_SI * efficiency * tau /
                         (C_SI * (distance * MPC_SI)**2 * (1 + (omega * tau)**2)))
    torch.testing.assert_close(base, base.new_tensor(expected), atol=0, rtol=2e-14)
    cases = [((240., distance, efficiency, omega, tau), 2.),
             ((mass, 600., efficiency, omega, tau), .5),
             ((mass, distance, .08, omega, tau), 2.),
             # Fixed dimensionless QNM frequencies: omega~1/M and tau~M.
             ((120., distance, efficiency, omega / 2, tau * 2), 2.)]
    for values, factor in cases:
        torch.testing.assert_close(amplitude(*values, convention="single_mode"),
                                   factor * base, atol=0, rtol=2e-14)


def test_exact_finite_quality_correction():
    quality = torch.tensor([.5, 1., 2., 10.], dtype=torch.float64)
    omega, tau = 1000., 2 * quality / 1000.
    exact = amplitude(65., 400., .01, omega, tau, convention="single_mode")
    high_quality = torch.sqrt(32 * math.pi * G_SI * 65 * MSUN_SI * .01 /
                              (C_SI * (400 * MPC_SI)**2 * omega**2 * tau))
    torch.testing.assert_close(exact / high_quality, (1 + 1 / (4 * quality**2)).rsqrt(),
                               atol=0, rtol=2e-14)


def test_all_parameter_gradients_and_broadcasting():
    values = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True)
                   for v in ([[60.], [80.]], 300., [[.01, .03]], 1700., .003))
    # Physical strains are ~1e-21; scaling prevents a vacuous absolute-error
    # gradcheck. All five physical inputs, including omega and tau, vary.
    assert torch.autograd.gradcheck(lambda *x: 1e21 * amplitude(*x), values,
                                    eps=1e-7, atol=2e-6, rtol=3e-5)
    output = amplitude(*values)
    assert output.shape == (2, 2)
    for i in range(2):
        for j in range(2):
            scalar = amplitude(values[0][i, 0], values[1], values[2][0, j], values[3], values[4])
            torch.testing.assert_close(output[i, j], scalar, rtol=0, atol=0)


def test_logarithmic_derivatives():
    values = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True)
                   for v in (63., 410., .017, 1600., .004))
    result = amplitude(*values)
    derivatives = torch.autograd.grad(result.log(), values)
    mass, distance, efficiency, omega, tau = values
    expected = (.5 / mass, -1 / distance, .5 / efficiency,
                -omega * tau**2 / (1 + (omega * tau)**2),
                .5 / tau - omega**2 * tau / (1 + (omega * tau)**2))
    for actual, target in zip(derivatives, expected):
        torch.testing.assert_close(actual, target, rtol=2e-14, atol=0)


def test_zero_efficiency_and_its_boundary_derivative():
    values = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True)
                   for v in (63., 410., 0., 1600., .004))
    result = amplitude(*values)
    assert result.item() == 0
    derivatives = torch.autograd.grad(result, values)
    for index, derivative in enumerate(derivatives):
        assert derivative.isposinf() if index == 2 else derivative.item() == 0


def test_dtype_promotion_and_float32_scale():
    inputs = (63., 410., .017, 1600., .004)
    reference = amplitude(*inputs)
    assert reference.dtype == torch.float64
    single = amplitude(torch.tensor(63., dtype=torch.float32), *inputs[1:])
    assert single.dtype == torch.float32 and single > 0
    torch.testing.assert_close(single.double(), reference, rtol=5e-6, atol=0)
    mass = torch.tensor(63., dtype=torch.float32, requires_grad=True)
    efficiency = torch.tensor(.017, dtype=torch.float64, requires_grad=True)
    mixed = amplitude(mass, 410., efficiency, 1600., .004)
    assert mixed.dtype == torch.float64
    assert all(torch.isfinite(g) for g in torch.autograd.grad(mixed, (mass, efficiency)))
    with torch.inference_mode():
        torch.testing.assert_close(amplitude(*inputs), reference, rtol=0, atol=0)
    assert amplitude(torch.tensor(63), *inputs[1:]).dtype == torch.float64


def test_finite_extreme_positive_inputs_do_not_overflow_intermediates():
    # omega*tau and squared SI distance overflow if formed directly.
    result = amplitude(60., 1., .01, 1e200, 1e200)
    assert torch.isfinite(result) and result > 0
    result = amplitude(1e200, 1e200, .01, 1e200, 1e-200)
    assert torch.isfinite(result) and result > 0
    result = amplitude(60., 1., .01, 1e200, 1e-200)
    assert torch.isfinite(result) and result > 0


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("bad", [-1., float("nan"), float("inf")])
def test_rejects_invalid_physical_values(index, bad):
    values = [60., 300., .02, 1800., .004]
    values[index] = bad
    with pytest.raises(ValueError, match="finite"):
        amplitude(*values)


@pytest.mark.parametrize("index", [0, 1, 3, 4])
def test_rejects_zero_physical_dimensions(index):
    values = [60., 300., .02, 1800., .004]
    values[index] = 0
    with pytest.raises(ValueError, match="positive"):
        amplitude(*values)


@pytest.mark.parametrize("bad", [True, 60 + 0j, torch.tensor(True), torch.tensor(60., dtype=torch.float16)])
def test_rejects_unsupported_types(bad):
    with pytest.raises(TypeError):
        amplitude(bad, 300., .02, 1800., .004)


def test_rejects_unknown_convention():
    with pytest.raises(ValueError, match="convention"):
        amplitude(60., 300., .02, 1800., .004, convention="unknown")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_device_and_gradient():
    mass = torch.tensor([60., 80.], device="cuda", dtype=torch.float64, requires_grad=True)
    result = amplitude(mass, 300., .02, 1800., .004)
    assert result.device.type == "cuda"
    assert torch.isfinite(torch.autograd.grad(result.sum(), mass)[0]).all()
    torch.testing.assert_close(result.cpu(), amplitude(mass.cpu(), 300., .02, 1800., .004))
    with pytest.raises(ValueError, match="share a device"):
        amplitude(mass, torch.tensor(300.), .02, 1800., .004)
