"""Analytic quadrupole, Fourier scaling and normalized energy-balance checks."""
import math

import pytest
import torch

from torchwave.approximants.eob.waveform import _interp1d, _quadrupole_fd


def test_scalar_interpolation_preserves_table_batches_and_query_derivative():
    grid = torch.tensor([0., 1., 3.], dtype=torch.float64)
    values = torch.tensor([[0., 2., 4.], [1., 4., 12.]], dtype=torch.float64)
    query = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    actual = _interp1d(query, grid, values)
    torch.testing.assert_close(actual, torch.tensor([3., 8.], dtype=torch.float64))
    torch.testing.assert_close(torch.autograd.functional.jacobian(
        lambda q: _interp1d(q, grid, values), query),
        torch.tensor([1., 4.], dtype=torch.float64))
    assert _interp1d(query, grid, values[0]).shape == ()
    assert torch.count_nonzero(_interp1d(query + 2, grid, values)) == 0


@pytest.mark.parametrize("inclination", [0.0, 0.7, math.pi / 2])
def test_periodic_quadrupole_has_correct_positive_frequency_normalization(inclination):
    # Eight complete cycles: exact discrete Fourier orthogonality, no SPA.
    n, dt, k, amplitude = 1024, 1 / 1024, 8, 2.3
    time = torch.arange(n, dtype=torch.float64) * dt
    h22 = amplitude * torch.exp(-2j * math.pi * k * time)
    f = torch.tensor([k], dtype=torch.float64)
    inc = torch.tensor(inclination, dtype=torch.float64)
    hp, hc = _quadrupole_fd(f, h22, torch.tensor(dt, dtype=torch.float64), inc)
    c = math.cos(inclination)
    expected_p = math.sqrt(5 / (16 * math.pi)) * (1 + c*c) * amplitude / 2
    expected_c = -1j * math.sqrt(5 / (4 * math.pi)) * c * amplitude / 2
    torch.testing.assert_close(hp, torch.full_like(hp, expected_p), rtol=2e-13, atol=1e-14)
    torch.testing.assert_close(hc, torch.full_like(hc, expected_c), rtol=2e-13, atol=1e-14)


@pytest.mark.parametrize(
    "inclination,phase_offset",
    [(0.13, 0.0), (0.7, 0.37), (1.41, -1.2)],
)
def test_quadrupole_fd_matches_finite_fourier_sum_for_general_mode_phase(
    inclination, phase_offset
):
    """The published 22 projection and the negative-frequency FFT agree analytically."""
    n, dt, k, amplitude = 1024, 1 / 1024, 8, 2.3
    time = torch.arange(n, dtype=torch.float64) * dt
    h22 = amplitude * torch.exp(
        torch.tensor(1j * phase_offset, dtype=torch.complex128)
        - 2j * math.pi * k * time
    )
    f = torch.tensor([k / (n * dt)], dtype=torch.float64)
    inc = torch.tensor(inclination, dtype=torch.float64)
    actual_p, actual_c = _quadrupole_fd(f, h22, torch.tensor(dt), inc)

    c = math.cos(inclination)
    projected_p = math.sqrt(5 / (16 * math.pi)) * (1 + c * c) * h22.real
    projected_c = -math.sqrt(5 / (4 * math.pi)) * c * h22.imag
    expected_p = (projected_p * torch.exp(-2j * math.pi * f[0] * time)).sum() * dt
    expected_c = (projected_c * torch.exp(-2j * math.pi * f[0] * time)).sum() * dt
    torch.testing.assert_close(actual_p[0], expected_p, rtol=2e-13, atol=1e-14)
    torch.testing.assert_close(actual_c[0], expected_c, rtol=2e-13, atol=1e-14)


def test_fourier_mass_scale_preserves_gradient_and_batched_sampling():
    time = torch.arange(64, dtype=torch.float64)
    signal = torch.exp(-time / 20) * torch.exp(-0.5j * time)
    f = torch.tensor([.071, .139], dtype=torch.float64)
    scale = torch.tensor(.9, dtype=torch.float64, requires_grad=True)
    def evaluate(s):
        return _quadrupole_fd(f, signal * s, s * .5, s * .2)[0]
    assert torch.autograd.gradcheck(evaluate, (scale,), atol=1e-5, rtol=1e-4)
    scales = torch.tensor([.7, 1.3], dtype=torch.float64)
    batched = _quadrupole_fd(f, signal[:, None] * scales, scales * .5, scales * .2)
    for i, s in enumerate(scales):
        single = _quadrupole_fd(f, signal * s, s * .5, s * .2)
        for b, one in zip(batched, single):
            torch.testing.assert_close(b[i], one, atol=1e-14, rtol=1e-13)
