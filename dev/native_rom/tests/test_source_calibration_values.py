"""Preserve rounded historical calibration values and parameter derivatives."""

import numpy  # noqa: F401  # Load OpenMP libraries before torch on macOS.
import torch

from dev.native_rom.eob.torch.source_calibration_hamiltonian import (
    SOURCE_V4_COEFFICIENTS,
    SourceCalibratedHistoricalHamiltonian,
)


def test_rounded_calibration_matches_source_table_and_gradients():
    nu = torch.tensor([0.2499, 8 / 81, 4.5 / 30.25], dtype=torch.float64,
                      requires_grad=True)
    chi1 = torch.tensor([0.8, 0.8, -0.8], dtype=torch.float64,
                        requires_grad=True)
    chi2 = torch.tensor([0.8, 0.4, -0.8], dtype=torch.float64,
                        requires_grad=True)
    model = SourceCalibratedHistoricalHamiltonian(nu, chi1, chi2)
    radius = torch.tensor([6.0, 10.0, 20.0], dtype=torch.float64)
    spin, _, _ = model.spin_combinations(radius)
    effective_spin = spin / (1 - 2 * nu)
    reference = tuple(sum(value * nu**i * effective_spin**j
                          for i, row in enumerate(SOURCE_V4_COEFFICIENTS[name])
                          for j, value in enumerate(row))
                      for name in ("K", "dSO", "dSS"))
    actual = model.calibration_parameters(radius)
    for expected, got in zip(reference, actual):
        torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)
        expected_grad = torch.autograd.grad(expected.sum(), (nu, chi1, chi2),
                                            retain_graph=True)
        actual_grad = torch.autograd.grad(got.sum(), (nu, chi1, chi2),
                                          retain_graph=True)
        for expected_part, got_part in zip(expected_grad, actual_grad):
            torch.testing.assert_close(got_part, expected_part,
                                       rtol=1e-11, atol=1e-10)
